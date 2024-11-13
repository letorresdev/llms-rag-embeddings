from typing import List, Dict
import requests
import xml.etree.ElementTree as ET
import json
from fastapi import HTTPException
from loguru import logger

from models.schemas import Article,AnalysisResponse
# from models.schemas import Article, AnalysisResponse
from services.llm import LLMService
from services.paper_processor import PaperProcessor
from core.config import settings


class ArxivService:
    def __init__(self):
        self.llm_service = LLMService()
        self.processor = PaperProcessor()
        self.base_url = settings.ARXIV_BASE_URL

    async def fetch_articles(
            self,
            query: str = settings.DEFAULT_SEARCH_QUERY,
            max_results: int = settings.MAX_RESULTS
    ) -> List[Dict]:
        """Fetch recent articles from ArXiv."""
        params = {
            "search_query": f"all:{query}",
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "start": 0,
            "max_results": max_results
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return self._parse_feed(response.text)
        except requests.RequestException as e:
            logger.error(f"Failed to fetch articles: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to fetch articles from ArXiv"
            )

    def _parse_feed(self, xml_data: str) -> List[Dict]:
        """Parse ArXiv XML feed."""
        root = ET.fromstring(xml_data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        return [{
            "title": entry.find("atom:title", ns).text,
            "authors": [
                author.find("atom:name", ns).text
                for author in entry.findall("atom:author", ns)
            ],
            "published": entry.find("atom:published", ns).text,
            "summary": entry.find("atom:summary", ns).text,
            "link": entry.find("atom:id", ns).text
        } for entry in root.findall("atom:entry", ns)]

    async def analyze_article(self, article: Dict) -> Article:
        """Analyze a paper using full text extraction and LLM analysis."""
        try:
            # Extract and chunk content
            content = await self.processor.extract_content(article["link"])
            chunks = self.processor.chunk_text(content)
            logger.info(f"Split article into {len(chunks)} chunks")

            # Analyze each chunk
            analyses = []
            for i, chunk in enumerate(chunks, 1):
                analysis = await self.llm_service.generate_response(
                    system_prompt=self._get_chunk_analysis_prompt(),
                    user_content=chunk
                )
                logger.info(f"Analyzed chunk {i}/{len(chunks)}")
                analyses.append(analysis['message']['content'])

            # Generate final analysis
            final_analysis = await self.llm_service.generate_response(
                system_prompt=self._get_final_analysis_prompt(),
                user_content="\n".join(analyses),
                format="json"
            )

            analysis_data = json.loads(final_analysis['message']['content'])

            return Article(
                title=article["title"],
                authors=article["authors"],
                published=article["published"],
                summary=analysis_data["Overall_Summary"],
                key_findings=analysis_data["Key_Findings"],
                methodology=analysis_data["Methodology"],
                conclusions=analysis_data["Conclusions"],
                relevance=analysis_data["Field_Relevance"],
                technical_details=analysis_data["Technical_Details"],
                link=article["link"]
            )
        except Exception as e:
            logger.error(f"Error analyzing article: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error analyzing article: {str(e)}"
            )

    def _get_chunk_analysis_prompt(self) -> str:
        """Get the prompt for analyzing individual chunks."""
        return """You are an expert scientific paper analyzer. 
        Analyze this section of the paper and identify:
        1. Key findings or statements
        2. Methodology details
        3. Conclusions or implications
        Be concise but thorough."""

    def _get_final_analysis_prompt(self) -> str:
        """Get the prompt for final analysis synthesis."""
        return """You are a technical research analyst tasked with synthesizing 
        complex analyses into a structured insight. For the input, provide an 
        analytical response in the following JSON format. Each value should be 
        a single comprehensive string:

        {
            "Overall_Summary": "Provide a concise overview in a single paragraph.",
            "Key_Findings": "Summarize the main points in a detailed paragraph.",
            "Methodology": "Describe the approach taken in a clear, single string.",
            "Conclusions": "State the conclusions in one coherent paragraph.",
            "Field_Relevance": "Explain the significance to the relevant field in a string.",
            "Technical_Details": "Provide any technical specifics in a clear paragraph format."
        }
        """
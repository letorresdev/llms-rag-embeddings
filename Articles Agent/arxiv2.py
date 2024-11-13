from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from datetime import datetime, timedelta
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Union
import ollama
from openai import OpenAI
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers.html2text import Html2TextTransformer
import markdown2
from pydantic import BaseModel
from urllib.parse import urlparse
import asyncio
from loguru import logger
import json
import os


class ModelHandler:
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")


        if self.openai_key:
            self.client = OpenAI(api_key=self.openai_key)
            logger.info("Using OpenAI model")
        else:
            self.client = ollama
            logger.info("Using Ollama model as fallback")

    async def generate_response(self, system_prompt: str, user_content: str, format: str = None) -> Dict:
        """Generate response using either OpenAI or Ollama model."""
        try:
            if self.openai_key:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]
                response = self.client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=messages,
                    response_format={"type": "json_object"} if format == "json" else None
                )
                return {
                    "message": {
                        "content": response.choices[0].message.content
                    }
                }
            else:
                return self.client.chat(
                    model='llama3.2',
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    format=format,
                    stream=False
                )
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")

class Article(BaseModel):
    title: str
    authors: List[str]
    published: str
    summary: str
    key_findings: str
    methodology: str
    conclusions: str
    relevance: str
    link: str
    technical_details: str

class PaperProcessor:
    def __init__(self, chunk_size: int = 20000):
        self.chunk_size = chunk_size
        self.html2text = Html2TextTransformer()

    async def extract_content(self, link: str) -> str:
        """Extract content from ArXiv paper link."""
        try:
            # Convert abstract URL to HTML URL
            html_link = link.replace('/abs/', '/html/')
            logger.info(f"Extracting content from: {html_link}")

            # Load HTML content
            loader = AsyncHtmlLoader([html_link])
            docs = await asyncio.get_event_loop().run_in_executor(None, loader.load)

            # Transform HTML to text
            docs_transformed = self.html2text.transform_documents(docs)
            content = docs_transformed[0].page_content

            # Clean and structure the content
            content = self._clean_content(content)

            return content
        except Exception as e:
            logger.error(f"Error extracting content: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to extract paper content: {str(e)}")

    def _clean_content(self, content: str) -> str:
        """Clean and structure the extracted content."""
        # Remove multiple newlines
        content = '\n'.join(line.strip() for line in content.split('\n') if line.strip())

        # Remove common HTML artifacts
        content = content.replace('\\n', '\n').replace('\\t', ' ')

        # Remove references section if present
        if 'References' in content:
            content = content.split('References')[0]

        return content

    def chunk_text(self, text: str) -> List[str]:
        """Split text into manageable chunks for LLM processing."""
        chunks = []
        current_chunk = []
        current_size = 0

        for paragraph in text.split('\n'):
            if len(paragraph) + current_size > self.chunk_size:
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_size = len(paragraph)
            else:
                current_chunk.append(paragraph)
                current_size += len(paragraph)

        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks

class ArxivAnalyzer:
    def __init__(self):
        self.model_handler = ModelHandler()
        self.base_url = "http://export.arxiv.org/api/query"
        self.processor = PaperProcessor()

    async def fetch_articles(self, query: str = "RAG LLM", days: int = 1) -> List[Dict]:
        """Fetch recent articles from ArXiv."""
        params = {
            "search_query": f"all:{query}",
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "start": 0,
            "max_results": 1
        }

        response = requests.get(self.base_url, params=params)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch articles")

        return self._parse_feed(response.text)

    def _parse_feed(self, xml_data: str) -> List[Dict]:
        """Parse ArXiv XML feed."""
        root = ET.fromstring(xml_data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        return [{
            "title": entry.find("atom:title", ns).text,
            "authors": [author.find("atom:name", ns).text for author in entry.findall("atom:author", ns)],
            "published": entry.find("atom:published", ns).text,
            "summary": entry.find("atom:summary", ns).text,
            "link": entry.find("atom:id", ns).text
        } for entry in root.findall("atom:entry", ns)]

    async def analyze_article(self, article: Dict) -> Article:
        """Analyze a paper using full text extraction and LLM analysis."""
        try:
            content = await self.processor.extract_content(article["link"])
            chunks = self.processor.chunk_text(content)
            logger.info(f"Split article into {len(chunks)} chunks")

            analysis_results = []
            for count, chunk in enumerate(chunks, 1):
                chunk_analysis = await self.model_handler.generate_response(
                    system_prompt="""You are an expert scientific paper analyzer. 
                    Analyze this section of the paper and identify:
                    1. Key findings or statements
                    2. Methodology details
                    3. Conclusions or implications
                    Be concise but thorough.""",
                    user_content=chunk
                )
                logger.info(f"Chunk No-{count} analyzed")
                analysis_results.append(chunk_analysis['message']['content'])

            final_analysis = await self.model_handler.generate_response(
                system_prompt="""You are a technical research analyst tasked with synthesizing complex analyses into a structured insight. For the input, provide an analytical response in the following JSON format. Each value should be a single comprehensive string. Ensure that no values are lists, dictionaries, or other data structures:

                {
                    "Overall_Summary": "Provide a concise overview in a single paragraph.",
                    "Key_Findings": "Summarize the main points in a detailed paragraph.",
                    "Methodology": "Describe the approach taken in a clear, single string.",
                    "Conclusions": "State the conclusions in one coherent paragraph.",
                    "Field_Relevance": "Explain the significance to the relevant field in a string.",
                    "Technical_Details": "Provide any technical specifics in a clear paragraph format."
                }
                """,
                user_content="\n".join(analysis_results),
                format="json"
            )

            analysis_text = json.loads(final_analysis['message']['content'])

            def handle_text_field(field):
                if isinstance(field, list):
                    return "\n".join(field)
                elif isinstance(field, str):
                    return field
                else:
                    return "Not available"

            return Article(
                title=article["title"],
                authors=article["authors"],
                published=article["published"],
                summary=handle_text_field(analysis_text.get("Overall_Summary", "Not available")),
                key_findings=handle_text_field(analysis_text.get("Key_Findings", "Not available")),
                methodology=handle_text_field(analysis_text.get("Methodology", "Not available")),
                conclusions=handle_text_field(analysis_text.get("Conclusions", "Not available")),
                relevance=handle_text_field(analysis_text.get("Field_Relevance", "Not available")),
                technical_details=handle_text_field(analysis_text.get("Technical_Details", "Not available")),
                link=article["link"]
            )
        except Exception as e:
            logger.error(f"Error analyzing article: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error analyzing article: {str(e)}")

app = FastAPI(title="ArXiv Paper Analyzer")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Return simple HTML homepage."""
    model_type = "OpenAI" if os.getenv("OPENAI_API_KEY") else "Ollama"
    return f"""
    <html>
        <head>
            <title>ArXiv Paper Analyzer</title>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown.min.css">
            <style>
                .markdown-body {{ max-width: 800px; margin: 0 auto; padding: 20px; }}
            </style>
        </head>
        <body class="markdown-body">
            <h1>ArXiv Paper Analyzer</h1>
            <p>Using {model_type} for analysis</p>
            <p>Visit <code>/papers</code> to see recent LLM-related papers with detailed analysis</p>
        </body>
    </html>
    """

@app.get("/papers", response_class=HTMLResponse)
async def get_papers():
    """Fetch, analyze, and display papers as markdown."""
    try:
        analyzer = ArxivAnalyzer()
        articles = await analyzer.fetch_articles()

        analyzed_papers = []
        for article in articles:
            analysis = await analyzer.analyze_article(article)
            analyzed_papers.append(analysis)

        markdown_content = "# Recent ArXiv LLM Papers Analysis\n\n"
        for paper in analyzed_papers:
            markdown_content += f"## {paper.title}\n\n"
            markdown_content += f"**Authors:** {', '.join(paper.authors)}\n\n"
            markdown_content += f"**Published:** {paper.published}\n\n"
            markdown_content += "### Summary\n" + paper.summary + "\n\n"
            markdown_content += "### Key Findings\n" + paper.key_findings + "\n\n"
            markdown_content += "### Methodology\n" + paper.methodology + "\n\n"
            markdown_content += "### Conclusions\n" + paper.conclusions + "\n\n"
            markdown_content += "### Relevance\n" + paper.relevance + "\n\n"
            markdown_content += f"[Read Full Paper]({paper.link})\n\n---\n\n"

        html_content = markdown2.markdown(markdown_content)

        return f"""
        <html>
            <head>
                <title>ArXiv Paper Analysis</title>
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown.min.css">
                <style>
                    .markdown-body {{ max-width: 800px; margin: 0 auto; padding: 20px; }}
                </style>
            </head>
            <body class="markdown-body">
                {html_content}
            </body>
        </html>
        """
    except Exception as e:
        logger.error(f"Error in get_papers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
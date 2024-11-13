from typing import List
from fastapi import HTTPException
from loguru import logger
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers.html2text import Html2TextTransformer
import asyncio
from core.config import settings

class PaperProcessor:
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.html2text = Html2TextTransformer()

    async def extract_content(self, link: str) -> str:
        """Extract content from ArXiv paper link."""
        try:
            # Convert abstract URL to HTML URL
            html_link = link.replace('/abs/', '/html/')
            logger.info(f"Extracting content from: {html_link}")

            # Load HTML content
            loader = AsyncHtmlLoader([html_link])
            docs = await asyncio.get_event_loop().run_in_executor(
                None,
                loader.load
            )

            # Transform HTML to text
            docs_transformed = self.html2text.transform_documents(docs)
            content = docs_transformed[0].page_content

            return self._clean_content(content)
        except Exception as e:
            logger.error(f"Error extracting content: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to extract paper content: {str(e)}"
            )

    def _clean_content(self, content: str) -> str:
        """Clean and structure the extracted content."""
        # Remove multiple newlines
        content = '\n'.join(
            line.strip() for line in content.split('\n') if line.strip()
        )

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
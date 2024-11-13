from typing import AsyncGenerator
from services.arxiv import ArxivService
from services.llm import LLMService

async def get_arxiv_service() -> AsyncGenerator[ArxivService, None]:
    """Dependency for ArxivService."""
    service = ArxivService()
    try:
        yield service
    finally:
        # Add cleanup if needed
        pass

async def get_llm_service() -> AsyncGenerator[LLMService, None]:
    """Dependency for LLMService."""
    service = LLMService()
    try:
        yield service
    finally:
        # Add cleanup if needed
        pass
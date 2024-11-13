from pydantic import BaseModel, HttpUrl
from typing import List
from datetime import datetime

class Article(BaseModel):
    """Schema for processed article data."""
    title: str
    authors: List[str]
    published: datetime
    summary: str
    key_findings: str
    methodology: str
    conclusions: str
    relevance: str
    link: HttpUrl
    technical_details: str

class AnalysisResponse(BaseModel):
    """Schema for LLM analysis response."""
    overall_summary: str
    key_findings: str
    methodology: str
    conclusions: str
    field_relevance: str
    technical_details: str
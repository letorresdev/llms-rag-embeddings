from typing import Dict, Optional
from openai import OpenAI
import ollama
from fastapi import HTTPException
from loguru import logger
from core.config import settings


class LLMService:
    def __init__(self):
        self.client = self._initialize_client()

    def _initialize_client(self):
        """Initialize the LLM client based on available API keys."""
        if settings.OPENAI_API_KEY:
            logger.info("Using OpenAI model")
            return OpenAI(api_key=settings.OPENAI_API_KEY)
        logger.info("Using Ollama model as fallback")
        return ollama

    async def generate_response(
            self,
            system_prompt: str,
            user_content: str,
            format: Optional[str] = None
    ) -> Dict:
        """Generate response using either OpenAI or Ollama model."""
        try:
            if isinstance(self.client, OpenAI):
                return await self._generate_openai_response(
                    system_prompt,
                    user_content,
                    format
                )
            return await self._generate_ollama_response(
                system_prompt,
                user_content,
                format
            )
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Model error: {str(e)}"
            )

    async def _generate_openai_response(
            self,
            system_prompt: str,
            user_content: str,
            format: Optional[str]
    ) -> Dict:
        """Generate response using OpenAI."""
        response = self.client.chat.completions.create(
            model=settings.DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"} if format == "json" else None
        )
        return {
            "message": {
                "content": response.choices[0].message.content
            }
        }

    async def _generate_ollama_response(
            self,
            system_prompt: str,
            user_content: str,
            format: Optional[str]
    ) -> Dict:
        """Generate response using Ollama."""
        return self.client.chat(
            model=settings.FALLBACK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            format=format,
            stream=False
        )
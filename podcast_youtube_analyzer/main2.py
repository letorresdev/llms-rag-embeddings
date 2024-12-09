import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import json
import re
from datetime import datetime
import numpy as np
from itertools import islice
import asyncio
import ollama
import gradio as gr
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """Represents a segment of the transcript with timing information."""
    text: str
    start: float
    duration: float
    embedding: Optional[np.ndarray] = None

    @property
    def end(self) -> float:
        """Calculate the end time of the segment."""
        return self.start + self.duration


@dataclass
class TranscriptChunk:
    """Represents a meaningful chunk of the transcript."""
    text: str
    start_time: float
    end_time: float
    embedding: Optional[np.ndarray] = None

    def get_timestamp_range(self) -> str:
        """Return a formatted timestamp range for this chunk."""
        start = f"{int(self.start_time // 60):02d}:{int(self.start_time % 60):02d}"
        end = f"{int(self.end_time // 60):02d}:{int(self.end_time % 60):02d}"
        return f"[{start} - {end}]"


class TranscriptProcessor:
    """Handles transcript chunking and processing."""

    def __init__(self, target_chunk_size: int = 500):
        self.target_chunk_size = target_chunk_size

    def merge_segments(self, segments: List[dict]) -> List[dict]:
        """Merge transcript segments into larger chunks."""
        if not segments:
            return []

        merged_segments = []
        current_chunk = {"text": "", "start": segments[0]["start"], "duration": 0}

        for segment in segments:
            # Check if adding this segment would exceed target size
            if (
                    len(current_chunk["text"]) + len(segment["text"]) > self.target_chunk_size
                    and current_chunk["text"]
            ):
                merged_segments.append(current_chunk)
                current_chunk = {"text": "", "start": segment["start"], "duration": 0}

            # Add segment to current chunk
            if current_chunk["text"]:
                current_chunk["text"] += " "
            current_chunk["text"] += segment["text"]
            current_chunk["duration"] += segment["duration"]

        # Add the last chunk if it contains text
        if current_chunk["text"]:
            merged_segments.append(current_chunk)

        return merged_segments

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean transcript text by removing annotations and extra whitespace."""
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"\(.*?\)", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


class TranscriptMemory:
    """Manages transcript chunks and their embeddings for retrieval."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", chunk_window_size: int = 3):
        self.encoder = SentenceTransformer(model_name)
        self.chunks: List[TranscriptChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.chunk_window_size = chunk_window_size
        self.processor = TranscriptProcessor()

    def create_chunks(self, segments: List[dict]) -> List[TranscriptChunk]:
        """Create transcript chunks from segments."""
        merged_segments = self.processor.merge_segments(segments)
        return [
            TranscriptChunk(
                text=self.processor.clean_text(segment["text"]),
                start_time=segment["start"],
                end_time=segment["start"] + segment["duration"],
            )
            for segment in merged_segments
        ]

    def add_transcript(self, segments: List[dict]) -> None:
        """Process and store a new transcript."""
        if not segments:
            raise ValueError("No segments provided")

        self.chunks = self.create_chunks(segments)
        texts = [chunk.text for chunk in self.chunks]

        # Convert embeddings to numpy arrays and store them
        embeddings = self.encoder.encode(texts, convert_to_tensor=True)
        self.embeddings = embeddings.cpu().numpy()

        for chunk, embedding in zip(self.chunks, self.embeddings):
            chunk.embedding = embedding

    def get_relevant_context(self, query: str, top_k: int = 5) -> str:
        """Retrieve relevant context based on query similarity."""
        if not self.chunks or self.embeddings is None:
            return ""

        query_embedding = self.encoder.encode(query, convert_to_tensor=True).cpu().numpy()

        # Calculate similarities and get top matches
        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Expand context window around top matches
        expanded_indices = set()
        for idx in top_indices:
            start_idx = max(0, idx - self.chunk_window_size)
            end_idx = min(len(self.chunks), idx + self.chunk_window_size + 1)
            expanded_indices.update(range(start_idx, end_idx))

        # Construct context with timestamps
        context_parts = [
            f"{self.chunks[idx].get_timestamp_range()}: {self.chunks[idx].text}"
            for idx in sorted(expanded_indices)
        ]

        return "\n\n".join(context_parts)


class PodcastTranscriptProcessor:
    """Main class for processing podcast transcripts and handling chat interactions."""

    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name
        self.output_dir = Path("transcripts")
        self.output_dir.mkdir(exist_ok=True)
        self.memory = TranscriptMemory()
        self.current_video_id: Optional[str] = None
        self.chat_history: List[Tuple[str, str]] = []

    def extract_video_id(self, url: str) -> str:
        """Extract YouTube video ID from URL."""
        youtube_regex = (
            r'(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|'
            r'youtu\.be\/)([a-zA-Z0-9_-]{11})'
        )
        match = re.search(youtube_regex, url)
        if not match:
            raise ValueError("Invalid YouTube URL format")
        return match.group(1)

    def get_transcript(self, video_id: str) -> List[dict]:
        """Fetch transcript for a YouTube video."""
        try:
            return YouTubeTranscriptApi.get_transcript(video_id)
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            logger.error(f"Failed to fetch transcript for video {video_id}: {str(e)}")
            raise

    async def chat_with_transcript(self, user_message: str) -> str:
        """Generate a response based on the transcript context and user message."""
        try:
            context = self.memory.get_relevant_context(user_message)

            print("context", context)
            prompt = f"""You are an AI assistant analyzing a transcript. Use the information give to give an answer.

            Transcript Context:
            {context}

            Question: {user_message}"""

            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            return f"Error generating response: {str(e)}"

    async def process_video_url(self, url: str) -> str:
        """Process a YouTube video URL and load its transcript."""
        try:
            video_id = self.extract_video_id(url)
            if video_id == self.current_video_id:
                return "This video is already loaded."

            segments = self.get_transcript(video_id)
            self.memory.add_transcript(segments)
            self.current_video_id = video_id
            return f"Successfully loaded transcript for video {video_id}!"
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return f"Error processing video: {str(e)}"

    def create_gradio_interface(self) -> gr.Blocks:
        """Create and configure the Gradio interface."""
        with gr.Blocks(title="Podcast Transcript Chat") as interface:
            gr.Markdown("# Podcast Transcript Chat")
            gr.Markdown("Enter a YouTube URL to chat about its transcript!")

            with gr.Row():
                url_input = gr.Textbox(
                    label="YouTube URL",
                    placeholder="Enter YouTube URL here..."
                )
                load_btn = gr.Button("Load Transcript", variant="primary")

            status_output = gr.Markdown()

            with gr.Column():
                chatbot = gr.Chatbot(label="Chat History", height=400)
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Your Message",
                        placeholder="Ask something about the transcript...",
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

            async def respond(user_message: str, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str]:
                if not self.current_video_id:
                    return history + [(user_message, "Please load a video first!")], ""
                if not user_message.strip():
                    return history, ""

                bot_response = await self.chat_with_transcript(user_message)
                history.append((user_message, bot_response))
                return history, ""

            load_btn.click(self.process_video_url, inputs=url_input, outputs=status_output)
            msg_input.submit(respond, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input])
            send_btn.click(respond, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input])

        return interface


def main():
    """Main entry point for the application."""
    try:
        processor = PodcastTranscriptProcessor()
        interface = processor.create_gradio_interface()
        interface.launch(
            server_name="0.0.0.0",
            share=True,
            show_error=True
        )
    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}")
        raise


if __name__ == "__main__":
    main()
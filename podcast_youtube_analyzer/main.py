import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime
import ollama
import gradio as gr
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from sentence_transformers import SentenceTransformer
import numpy as np
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """Represents a segment of the transcript with timing information."""
    text: str
    start: float
    duration: float
    embedding: Optional[np.ndarray] = None


class TranscriptMemory:
    """Manages transcript segments and their embeddings for retrieval."""

    def __init__(self):
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.segments: List[TranscriptSegment] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_segments(self, segments: List[TranscriptSegment]):
        """Add segments and compute their embeddings."""
        self.segments = segments
        texts = [segment.text for segment in segments]
        self.embeddings = self.encoder.encode(texts, convert_to_tensor=True)

    def get_relevant_context(self, query: str, top_k: int = 3) -> str:
        """Retrieve most relevant segments for a query."""
        if not self.segments or self.embeddings is None:
            return ""

        # Convert embeddings to CPU numpy arrays
        embeddings_cpu = self.embeddings.cpu().numpy()

        # Encode the query and move it to CPU
        query_embedding = self.encoder.encode(query, convert_to_tensor=True).cpu().numpy()

        # Compute similarities
        similarities = np.dot(embeddings_cpu, query_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Construct context from top matches
        context_segments = [self.segments[i].text for i in top_indices]
        return " ".join(context_segments)


class PodcastTranscriptProcessor:
    def __init__(self, model_name: str = "llama3.2"):
        """
        Initialize the processor with specified Ollama model.

        Args:
            model_name: Name of the Ollama model to use
        """
        self.model_name = model_name
        self.output_dir = Path("transcripts")
        self.output_dir.mkdir(exist_ok=True)
        self.memory = TranscriptMemory()
        self.current_video_id: Optional[str] = None
        self.chat_history: List[Tuple[str, str]] = []

    def extract_video_id(self, url: str) -> str:
        """Extract YouTube video ID from URL."""
        if "youtu.be" in url:
            return url.split("/")[-1]
        elif "youtube.com" in url:
            if "v=" in url:
                return url.split("v=")[1].split("&")[0]
        raise ValueError("Invalid YouTube URL format")

    def get_transcript(self, video_id: str) -> List[TranscriptSegment]:
        """Fetch transcript for a YouTube video."""
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            return [
                TranscriptSegment(
                    text=entry['text'],
                    start=entry['start'],
                    duration=entry['duration']
                )
                for entry in transcript_list
            ]
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            logger.error(f"Failed to fetch transcript for video {video_id}: {str(e)}")
            raise

    async def chat_with_transcript(self, user_message: str) -> str:
        """
        Chat with the transcript using RAG.

        Args:
            user_message: User's question or message

        Returns:
            str: AI response
        """
        # Get relevant context
        context = self.memory.get_relevant_context(user_message)

        print("context",context)

        # Construct prompt
        prompt = f"""Based on the following context from a podcast transcript, please answer the question.
        If the answer cannot be found in the context, say so.

        Context:
        {context}

        Question: {user_message}"""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}]
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            return f"Error generating response: {str(e)}"

    async def process_video_url(self, url: str) -> str:
        """Process a new video URL and prepare it for chatting."""
        try:
            video_id = self.extract_video_id(url)
            if video_id == self.current_video_id:
                return "This video is already loaded!"

            segments = self.get_transcript(video_id)
            self.memory.add_segments(segments)
            self.current_video_id = video_id
            self.chat_history = []

            return f"Successfully loaded transcript for video {video_id}. You can now ask questions about the content!"

        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return f"Error processing video: {str(e)}"

    def create_gradio_interface(self):
        """Create and launch the Gradio interface."""
        with gr.Blocks(title="Podcast Transcript Chat") as interface:
            gr.Markdown("# Podcast Transcript Chat")
            gr.Markdown("Enter a YouTube URL to load its transcript, then chat about the content!")

            with gr.Row():
                url_input = gr.Textbox(label="YouTube URL")
                load_btn = gr.Button("Load Transcript")

            status_output = gr.Markdown()

            chatbot = gr.Chatbot(label="Chat History")
            msg_input = gr.Textbox(label="Your Message", placeholder="Ask about the podcast content...")
            send_btn = gr.Button("Send")

            async def respond(user_message, history):
                if not self.current_video_id:
                    return history + [(user_message, "Please load a video transcript first!")]

                bot_response = await self.chat_with_transcript(user_message)
                history.append((user_message, bot_response))
                return history

            async def load_video(url):
                return await self.process_video_url(url)

            load_btn.click(load_video, inputs=[url_input], outputs=[status_output])
            msg_input.submit(respond, inputs=[msg_input, chatbot], outputs=[chatbot])
            send_btn.click(respond, inputs=[msg_input, chatbot], outputs=[chatbot])

        return interface


def main():
    processor = PodcastTranscriptProcessor()
    interface = processor.create_gradio_interface()
    interface.launch(share=True)


if __name__ == "__main__":
    main()
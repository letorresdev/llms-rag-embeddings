import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime
import ollama
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

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

    def extract_video_id(self, url: str) -> str:
        """
        Extract YouTube video ID from URL.

        Args:
            url: YouTube video URL

        Returns:
            str: Video ID
        """
        if "youtu.be" in url:
            return url.split("/")[-1]
        elif "youtube.com" in url:
            if "v=" in url:
                return url.split("v=")[1].split("&")[0]
        raise ValueError("Invalid YouTube URL format")

    def get_transcript(self, video_id: str) -> List[TranscriptSegment]:
        """
        Fetch transcript for a YouTube video.

        Args:
            video_id: YouTube video ID

        Returns:
            List[TranscriptSegment]: List of transcript segments
        """
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

    def create_markdown(self, segments: List[TranscriptSegment], video_id: str) -> str:
        """
        Convert transcript segments to markdown format.

        Args:
            segments: List of transcript segments
            video_id: YouTube video ID

        Returns:
            str: Markdown formatted text
        """
        markdown_content = f"# Transcript for YouTube Video: {video_id}\n\n"
        markdown_content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        for segment in segments:
            # timestamp = f"{int(segment.start // 60)}:{int(segment.start % 60):02d}"
            markdown_content = f"{segment.text}\n\n"

        return markdown_content

    async def summarize_transcript(self, text: str) -> str:
        """
        Summarize transcript using Ollama.

        Args:
            text: Full transcript text

        Returns:
            str: Summarized text
        """
        prompt = f"""Please summarize the following podcast transcript, focusing on:
        1. Main topics discussed
        2. Key insights and takeaways
        3. Important quotes or statements

        Transcript:
        {text}
        """

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}]
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Failed to generate summary: {str(e)}")
            raise

    def save_outputs(self, video_id: str, markdown: str, summary: str):
        """
        Save markdown and summary to files.

        Args:
            video_id: YouTube video ID
            markdown: Markdown formatted transcript
            summary: Generated summary
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save markdown
        markdown_path = self.output_dir / f"{video_id}_{timestamp}_transcript.md"
        markdown_path.write_text(markdown)

        # Save summary
        summary_path = self.output_dir / f"{video_id}_{timestamp}_summary.md"
        summary_path.write_text(summary)

        logger.info(f"Saved transcript and summary for video {video_id}")

    async def process_video(self, video_url: str):
        """
        Process a YouTube video: extract transcript, create markdown, and generate summary.

        Args:
            video_url: YouTube video URL
        """
        try:
            video_id = self.extract_video_id(video_url)
            logger.info(f"Processing video: {video_id}")

            # Get transcript
            segments = self.get_transcript(video_id)

            # Create markdown
            markdown = self.create_markdown(segments, video_id)

            # Generate summary
            summary = await self.summarize_transcript(
                " ".join(segment.text for segment in segments)
            )

            # Save outputs
            self.save_outputs(video_id, markdown, summary)

            logger.info(f"Successfully processed video: {video_id}")

        except Exception as e:
            logger.error(f"Failed to process video {video_url}: {str(e)}")
            raise


async def main():
    # Example usage
    processor = PodcastTranscriptProcessor()
    video_url = "https://www.youtube.com/watch?v=pAcF3GV4ygM&t=1263s"
    await processor.process_video(video_url)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

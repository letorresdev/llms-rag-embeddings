from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from datetime import datetime, timedelta
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict
import ollama
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers.html2text import Html2TextTransformer
import markdown2
from pydantic import BaseModel

app = FastAPI(title="ArXiv Article Analyzer")


class Article(BaseModel):
    title: str
    authors: List[str]
    published: str
    summary: str
    relevance: str
    link: str


class ArxivAnalyzer:
    def __init__(self, llm_pipeline):
        self.llm_pipeline = llm_pipeline
        self.base_url = "http://export.arxiv.org/api/query"

    async def fetch_articles(self, query: str = "RAG LLM", days: int = 1) -> List[Dict]:
        """Fetch recent articles from ArXiv."""
        params = {
            "search_query": f"all:{query}",
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "start": 0,
            "max_results": 5  # Increased for more results
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
        """Analyze a single article using the LLM."""
        try:
            # Generate analysis using LLM
            analysis = self.llm_pipeline.chat(
                model='llama3.2',
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing scientific papers"},
                    {"role": "user",
                     "content": f"Analyze this paper and provide a detailed summary:\n\n{article['summary']}"}
                ],
                stream=False
            )

            return Article(
                title=article["title"],
                authors=article["authors"],
                published=article["published"],
                summary=analysis['message']['content'],
                relevance="High" if "LLM" in article["title"] else "Medium",  # Simplified relevance
                link=article["link"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error analyzing article: {str(e)}")


# Initialize analyzer with Ollama
analyzer = ArxivAnalyzer(llm_pipeline=ollama)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Return simple HTML homepage."""
    return """
    <html>
        <head>
            <title>ArXiv Article Analyzer</title>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown.min.css">
            <style>
                .markdown-body { max-width: 800px; margin: 0 auto; padding: 20px; }
            </style>
        </head>
        <body class="markdown-body">
            <h1>ArXiv Article Analyzer</h1>
            <p>Visit <code>/articles</code> to see recent LLM-related papers</p>
        </body>
    </html>
    """


@app.get("/articles", response_class=HTMLResponse)
async def get_articles():
    """Fetch, analyze, and display articles as markdown."""
    try:
        # Fetch articles
        articles = await analyzer.fetch_articles()
        print("articles", articles)

        # Analyze each article
        analyzed_articles = []
        for article in articles:
            analysis = await analyzer.analyze_article(article)
            analyzed_articles.append(analysis)

        # Convert to markdown
        markdown_content = "# Recent ArXiv LLM Papers\n\n"
        for article in analyzed_articles:
            markdown_content += f"## {article.title}\n\n"
            markdown_content += f"**Authors:** {', '.join(article.authors)}\n\n"
            markdown_content += f"**Published:** {article.published}\n\n"
            markdown_content += f"**Summary:**\n{article.summary}\n\n"
            markdown_content += f"**Relevance:** {article.relevance}\n\n"
            markdown_content += f"[Read Paper]({article.link})\n\n---\n\n"

        # Convert markdown to HTML
        html_content = markdown2.markdown(markdown_content)

        return f"""
        <html>
            <head>
                <title>ArXiv Article Analysis</title>
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
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
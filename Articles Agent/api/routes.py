from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from loguru import logger
import markdown2
from services.arxiv import ArxivService
from core.config import settings

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def root():
    """Return simple HTML homepage."""
    return await generate_html_response(
        content=get_homepage_content(),
        title="ArXiv Paper Analyzer"
    )


@router.get("/papers", response_class=HTMLResponse)
async def get_papers():
    """Fetch, analyze, and display papers as markdown."""
    try:
        analyzer = ArxivService()
        articles = await analyzer.fetch_articles()

        analyzed_papers = []
        for article in articles:
            analysis = await analyzer.analyze_article(article)
            analyzed_papers.append(analysis)

        markdown_content = generate_markdown_content(analyzed_papers)
        html_content = markdown2.markdown(markdown_content)

        return await generate_html_response(
            content=html_content,
            title="ArXiv Paper Analysis"
        )
    except Exception as e:
        logger.error(f"Error in get_papers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def get_homepage_content() -> str:
    """Generate homepage content."""
    model_type = "OpenAI" if settings.OPENAI_API_KEY else "Ollama"
    return f"""
        <h1>ArXiv Paper Analyzer</h1>
        <p>Using {model_type} for analysis</p>
        <p>Visit <code>/papers</code> to see recent LLM-related papers with detailed analysis</p>
    """


def generate_markdown_content(papers) -> str:
    """Generate markdown content from analyzed papers."""
    content = "# Recent ArXiv LLM Papers Analysis\n\n"

    for paper in papers:
        content += f"## {paper.title}\n\n"
        content += f"**Authors:** {', '.join(paper.authors)}\n\n"
        content += f"**Published:** {paper.published}\n\n"
        content += "### Summary\n" + paper.summary + "\n\n"
        content += "### Key Findings\n" + paper.key_findings + "\n\n"
        content += "### Methodology\n" + paper.methodology + "\n\n"
        content += "### Conclusions\n" + paper.conclusions + "\n\n"
        content += "### Relevance\n" + paper.relevance + "\n\n"
        content += f"[Read Full Paper]({paper.link})\n\n---\n\n"

    return content


async def generate_html_response(content: str, title: str) -> str:
    """Generate HTML response with standard template."""
    return f"""
    <html>
        <head>
            <title>{title}</title>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown.min.css">
            <style>
                .markdown-body {{
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }}
            </style>
        </head>
        <body class="markdown-body">
            {content}
        </body>
    </html>
    """
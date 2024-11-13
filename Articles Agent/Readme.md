# ArXiv Paper Analyzer

The ArXiv Paper Analyzer is a web application that fetches and analyzes recent research papers from the ArXiv repository. 
It uses large language models (LLMs) to extract key information and insights from the paper content.

## Features

- Fetches recent research papers from ArXiv based on a search query
- Extracts and processes the full text of the papers
- Analyzes the paper content using LLMs to identify:
  - Overall summary
  - Key findings
  - Methodology
  - Conclusions
  - Relevance to the field
  - Technical details
- Presents the analysis results in a user-friendly Markdown-formatted web page

## Requirements

- Python 3.9 or higher
- FastAPI
- Uvicorn
- Pydantic
- Pydantic-settings
- Requests
- OpenAI or Ollama (LLM providers)
- Langchain-community
- Markdown2
- Loguru
- Python-dotenv
- Html2text

## Installation

1. Clone the repository:

   ```
   git clone ... 
   ```

2. Change to the project directory:

   ```
   cd Articles Agent
   ```

3. Create a virtual environment and activate it:

   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

4. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

5. Create a `.env` file in the root directory and configure the necessary settings:

   ```
   cp .env-example .env
   ```

   Edit the `.env` file and set your configuration values, including the OpenAI API key if you want to use the OpenAI model.

6. Start the application:

   ```
   python main.py
   ```

   The application will start running on `http://0.0.0.0:8000`.

## Usage

1. Visit `http://0.0.0.0:8000` in your web browser to see the home page.
2. Click on the `/papers` link to view the analysis of recent LLM-related research papers from ArXiv.
3. The page will display the analyzed papers in Markdown format, including the title, authors, publication date, summary, key findings, methodology, conclusions, and relevance.
4. Click on the "Read Full Paper" link to navigate to the original ArXiv paper.

## Configuration

The application's configuration is managed using Pydantic-settings. The available settings are:

- `PROJECT_NAME`: The name of the project.
- `VERSION`: The version of the application.
- `DEBUG`: Whether the application is running in debug mode.
- `HOST`: The host to run the application on.
- `PORT`: The port to run the application on.
- `OPENAI_API_KEY`: The API key for the OpenAI model (if using OpenAI).
- `ARXIV_BASE_URL`: The base URL for the ArXiv API.
- `DEFAULT_SEARCH_QUERY`: The default search query for fetching recent papers.
- `MAX_RESULTS`: The maximum number of results to fetch per search.
- `DEFAULT_MODEL`: The default LLM model to use.
- `FALLBACK_MODEL`: The fallback LLM model to use if the default is unavailable.
- `CHUNK_SIZE`: The maximum size of text chunks to send to the LLM for analysis.

You can customize these settings by editing the `.env` file.


## License

This project is licensed under the [MIT License](LICENSE).
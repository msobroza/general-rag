# General RAG Application

A general-purpose RAG (Retrieval-Augmented Generation) application with multilingual support, built using Streamlit and LangChain.

## Features

- Multilingual document support
- Advanced document parsing
- Hybrid search (semantic + keyword)
- Reranking capabilities
- Hypothetical document embeddings
- Multi-query retrieval
- PDF preview and source tracking

## Prerequisites

- Python 3.9 or higher
- Poetry for dependency management
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd general-rag
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Create a `.env` file in the project root with your OpenAI API key:
```bash
OPENAI_API_KEY=your-api-key-here
LLM_MODEL_NAME=gpt-4-turbo-preview
```

## Running the Application

You can run the application in two ways:

1. Using Poetry script:
```bash
poetry run start
```

2. Using Streamlit directly:
```bash
poetry run streamlit run rag/app.py
```

The application will be available at `http://localhost:8501` by default.

## Project Structure

```
general-rag/
├── rag/
│   ├── __init__.py
│   ├── app.py
│   └── languages.py
├── models/
│   ├── bge-m3/
│   ├── bge-reranker-v2-m3/
│   └── multilingual-e5-large-instruct/
├── pyproject.toml
├── README.md
└── .env
```

## Development

1. Activate the Poetry shell:
```bash
poetry shell
```

2. Run tests:
```bash
pytest
```

3. Format code:
```bash
black .
isort .
```

4. Lint code:
```bash
flake8
```

## License

MIT License

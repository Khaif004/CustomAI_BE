# Joule Replacement - Backend

AI Agent System for SAP BTP Development using FastAPI and LangChain

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- OpenAI API key (or alternative LLM provider)

### Installation

1. **Clone the repository and navigate to backend**
   ```bash
   cd codebase/backend
   ```

2. **Create virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Copy example env file
   copy .env.example .env  # Windows
   cp .env.example .env    # macOS/Linux

   # Edit .env and add your OpenAI API key
   ```

5. **Run the application**
   ```bash
   python -m app.main
   ```

   Or use uvicorn directly:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Verify Installation

Open your browser and visit:
- API Root: http://localhost:8000
- Health Check: http://localhost:8000/health
- API Docs: http://localhost:8000/docs (Swagger UI)
- ReDoc: http://localhost:8000/redoc

## 📁 Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py         # Application configuration
│   ├── api/                    # API routes (to be implemented)
│   │   └── __init__.py
│   ├── agents/                 # AI agents (to be implemented)
│   │   └── __init__.py
│   ├── tools/                  # Custom tools (to be implemented)
│   │   └── __init__.py
│   ├── knowledge/              # Vector store & RAG (to be implemented)
│   │   └── __init__.py
│   ├── models/                 # Pydantic models (to be implemented)
│   │   └── __init__.py
│   └── utils/                  # Utilities (to be implemented)
│       └── __init__.py
├── requirements.txt            # Python dependencies
├── .env.example               # Example environment variables
└── README.md                  # This file
```

## 🔧 Configuration

### Environment Variables

Key environment variables in `.env`:

```bash
# LLM Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Application Settings
APP_NAME=JouleReplacement
DEBUG=true
LOG_LEVEL=INFO

# Server
HOST=0.0.0.0
PORT=8000

# CORS
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
```

See `.env.example` for all available options.

## 🧪 Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/
```

### Code Formatting

```bash
# Format code with black
black app/

# Sort imports
isort app/

# Lint with flake8
flake8 app/
```

## 📚 Next Steps

1. **Implement Agents**: Add AI agents in `app/agents/`
2. **Add API Routes**: Create REST endpoints in `app/api/`
3. **Set up Knowledge Base**: Implement RAG in `app/knowledge/`
4. **Create Tools**: Add custom tools in `app/tools/`
5. **Add Tests**: Write tests for all modules

## 🔗 Related Documentation

- [Main Documentation](../../Documents/README.md)
- [Architecture Deep Dive](../../Documents/02-Architecture-Deep-Dive.md)
- [Quick Start Guide](../../Documents/03-Quick-Start-Guide.md)
- [Deployment Guide](../../Documents/04-Deployment-Guide.md)

## 🐛 Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're in the backend directory and virtual environment is activated
2. **OpenAI API errors**: Verify your API key in `.env` file
3. **Port already in use**: Change PORT in `.env` or kill the process using port 8000

### Getting Help

- Check the [Documentation](../../Documents/)
- Review [Architecture Guide](../../Documents/02-Architecture-Deep-Dive.md)
- See [Troubleshooting FAQ](../../Documents/10-Troubleshooting-FAQ.md)

## 📝 License

This project is part of the Joule Replacement initiative.

---

**Version**: 0.1.0  
**Last Updated**: March 14, 2026
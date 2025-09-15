# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# WSL/Linux recommended for development
mkdir -p ~/.venvs && python3 -m venv ~/.venvs/ccseeker
. ~/.venvs/ccseeker/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install --only-binary=:all: -r requirements.txt

# Create environment file (required for API keys)
printf "YOUTUBE_API_KEY=...\nGEMINI_API_KEY=...\n" > .env
```

### Running the Application
```bash
# Activate environment each session
. ~/.venvs/ccseeker/bin/activate

# Run main Streamlit UI (production-like)
streamlit run app/app_seed_gemini_hardened.py

# Run on specific address/port (for remote access)
streamlit run app/app_seed_gemini_hardened.py --server.address 0.0.0.0 --server.port 8501

# Alternative: use the shell script
./run_streamlit.sh

# Run MCP Server
python -m mcp_server.server

# Debug mode with verbose logs
streamlit run app/app_seed_gemini_hardened.py --logger.level=debug
```

### Testing
```bash
# Basic import test (manual testing approach)
python tests/test_basic.py
```

## Architecture Overview

CCSeeker is a YouTube creator discovery tool with dual interfaces:

### Core Components
- **Streamlit UI** (`app/`): Web interface for interactive creator search and analysis
- **MCP Server** (`mcp_server/`): Model Context Protocol server for AI agent integration
- **YouTube Tools** (`mcp_server/tools/youtube.py`): YouTube Data API v3 integration with advanced search
- **AI Generation** (`mcp_server/tools/ai_generation.py`): Gemini AI for summaries and outreach emails
- **Cache Manager** (`mcp_server/resources/cache.py`): Smart caching to reduce API calls

### Data Flow
1. **Search Input**: Boolean queries (AND/OR operators) or seed channel analysis
2. **YouTube API**: Fetch channel data with filtering (subscribers, country, upload recency)
3. **AI Processing**: Generate summaries and personalized outreach emails via Gemini
4. **Caching**: TTL-based caching (1-hour default) to minimize API usage

### Key Features
- **Boolean Search**: Advanced query syntax for precise channel discovery
- **Seed Channel Analysis**: Find similar creators based on existing channels
- **Multi-language Support**: AI-generated outreach in multiple languages
- **Smart Filtering**: Subscriber count, country, and upload activity filters
- **MCP Integration**: Exposes tools for AI agents via Model Context Protocol

## Configuration

### Required Environment Variables
- `YOUTUBE_API_KEY`: YouTube Data API v3 key
- `GEMINI_API_KEY`: Google Gemini API key for AI generation

### File Structure
```
CCSeeker/
├── app/                           # Streamlit UI
│   ├── app_seed_gemini_hardened.py   # Main production app
│   └── seed_topics_hardened.py       # Topic seeding utilities
├── mcp_server/                    # MCP server implementation
│   ├── server.py                      # Main MCP server
│   ├── tools/                         # Tool implementations
│   │   ├── youtube.py                     # YouTube API integration
│   │   └── ai_generation.py               # AI generation tools
│   └── resources/                     # Shared resources
│       └── cache.py                       # Cache management
├── tests/                         # Test files
└── .streamlit/config.toml         # Streamlit configuration
```

## Development Guidelines

### Code Style
- Python 3.x with PEP 8 (4 spaces)
- snake_case for modules/functions/variables
- PascalCase for classes
- Keep helper functions small and testable

### Security
- Never commit API keys or secrets
- Use .env file for sensitive configuration
- Handle API errors and quota limits gracefully

### Commit Standards
- Use imperative mood: "feat: add Spanish outreach option"
- Scope commits appropriately
- Include purpose and summary in PRs

### Platform Notes
- WSL/Linux recommended for development to avoid NTFS venv issues
- Use Linux-side virtual environment: `~/.venvs/ccseeker`
- Scripts have LF line endings enforced via .gitattributes
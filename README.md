# CCSeeker - YouTube Creator Discovery MCP Server

AI-powered YouTube creator discovery with MCP (Model Context Protocol) support.

## Quick Start
```bash
# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# Add your YOUTUBE_API_KEY and GEMINI_API_KEY

# Run MCP Server
python -m mcp_server.server

# Run Streamlit UI
streamlit run app/app_seed_gemini_hardened.py
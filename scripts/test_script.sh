#!/usr/bin/env bash
# CCSeeker Project Testing Script
# Verifies all components are working correctly

set -euo pipefail

echo "🧪 CCSeeker Project Tests"
echo "========================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PASSED=0
FAILED=0

# Function to run a test
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo -ne "${BLUE}Testing:${NC} $test_name ... "
    
    if eval "$test_command" &> /dev/null; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((FAILED++))
        return 1
    fi
}

# Test 1: Virtual environment
echo -e "${YELLOW}=== Environment Tests ===${NC}"
run_test "Virtual environment activated" "python -c 'import sys; sys.exit(0 if hasattr(sys, \"real_prefix\") or (hasattr(sys, \"base_prefix\") and sys.base_prefix != sys.prefix) else 1)'"

# Test 2: Python version
run_test "Python 3.x available" "python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)'"

echo ""
echo -e "${YELLOW}=== Dependency Tests ===${NC}"

# Test 3: Core dependencies
run_test "Streamlit installed" "python -c 'import streamlit'"
run_test "Google API client installed" "python -c 'import googleapiclient'"
run_test "Pandas installed" "python -c 'import pandas'"
run_test "Gemini AI installed" "python -c 'import google.generativeai'"
run_test "pycountry installed" "python -c 'import pycountry'"

echo ""
echo -e "${YELLOW}=== Configuration Tests ===${NC}"

# Test 4: Environment file
if [ -f .env ]; then
    echo -ne "${BLUE}Testing:${NC} .env file exists ... "
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASSED++))
    
    # Check for API keys (without revealing them)
    if grep -q "YOUTUBE_API_KEY=" .env && [ "$(grep "YOUTUBE_API_KEY=" .env | cut -d'=' -f2)" != "" ]; then
        echo -ne "${BLUE}Testing:${NC} YOUTUBE_API_KEY set ... "
        echo -e "${GREEN}✓ PASS${NC}"
        ((PASSED++))
    else
        echo -ne "${BLUE}Testing:${NC} YOUTUBE_API_KEY set ... "
        echo -e "${RED}✗ FAIL${NC}"
        ((FAILED++))
    fi
    
    if grep -q "GEMINI_API_KEY=" .env && [ "$(grep "GEMINI_API_KEY=" .env | cut -d'=' -f2)" != "" ]; then
        echo -ne "${BLUE}Testing:${NC} GEMINI_API_KEY set ... "
        echo -e "${GREEN}✓ PASS${NC}"
        ((PASSED++))
    else
        echo -ne "${BLUE}Testing:${NC} GEMINI_API_KEY set ... "
        echo -e "${RED}✗ FAIL${NC}"
        ((FAILED++))
    fi
else
    echo -ne "${BLUE}Testing:${NC} .env file exists ... "
    echo -e "${RED}✗ FAIL${NC}"
    echo -e "  ${YELLOW}→ Create .env with: YOUTUBE_API_KEY=... and GEMINI_API_KEY=...${NC}"
    ((FAILED+=3))
fi

echo ""
echo -e "${YELLOW}=== File Structure Tests ===${NC}"

# Test 5: Required files
run_test "Main app exists" "[ -f app/app_seed_gemini_hardened.py ]"
run_test "Seed module exists" "[ -f app/seed_topics_hardened.py ]"
run_test "CSS file exists" "[ -f app/theme_ccseeker_dark.css ]"
run_test "App icons folder exists" "[ -d appicons ]"
run_test "MCP server exists" "[ -f mcp_server/server.py ]"
run_test "Requirements file exists" "[ -f requirements.txt ]"
run_test "Launch script exists" "[ -f run_streamlit.sh ]"

echo ""
echo -e "${YELLOW}=== Import Tests ===${NC}"

# Test 6: Module imports
if run_test "Main app imports" "cd app && python -c 'import seed_topics_hardened' && cd .."; then
    :
fi

if run_test "MCP server imports" "python -c 'from mcp_server.tools import youtube, ai_generation'"; then
    :
fi

echo ""
echo -e "${YELLOW}=== Functional Tests ===${NC}"

# Test 7: YouTube API connection (if keys are set)
if [ -f .env ] && grep -q "YOUTUBE_API_KEY=." .env; then
    echo -ne "${BLUE}Testing:${NC} YouTube API connection ... "
    if python -c "
import os
from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()
api_key = os.getenv('YOUTUBE_API_KEY')
if not api_key:
    exit(1)

try:
    youtube = build('youtube', 'v3', developerKey=api_key)
    # Simple test: search for one channel
    response = youtube.search().list(q='test', type='channel', part='id', maxResults=1).execute()
    exit(0 if response.get('items') else 1)
except Exception:
    exit(1)
" 2>/dev/null; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}"
        echo -e "  ${YELLOW}→ Check your YOUTUBE_API_KEY and quota${NC}"
        ((FAILED++))
    fi
else
    echo -ne "${BLUE}Testing:${NC} YouTube API connection ... "
    echo -e "${YELLOW}⊘ SKIP (no API key)${NC}"
fi

# Test 8: Gemini API connection (if keys are set)
if [ -f .env ] && grep -q "GEMINI_API_KEY=." .env; then
    echo -ne "${BLUE}Testing:${NC} Gemini API connection ... "
    if python -c "
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    exit(1)

try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content('Hello')
    exit(0 if response.text else 1)
except Exception:
    exit(1)
" 2>/dev/null; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}"
        echo -e "  ${YELLOW}→ Check your GEMINI_API_KEY and quota${NC}"
        ((FAILED++))
    fi
else
    echo -ne "${BLUE}Testing:${NC} Gemini API connection ... "
    echo -e "${YELLOW}⊘ SKIP (no API key)${NC}"
fi

# Summary
echo ""
echo "================================"
TOTAL=$((PASSED + FAILED))
echo -e "Results: ${GREEN}$PASSED passed${NC}, ${RED}$FAILED failed${NC} (out of $TOTAL tests)"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed! Your project is ready.${NC}"
    echo ""
    echo "Run the app with:"
    echo "  streamlit run app/app_seed_gemini_hardened.py"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Some tests failed. Please fix the issues above.${NC}"
    echo ""
    exit 1
fi

"""
CCSeeker Test Suite
===================

Unit tests for the extracted core modules.
Run with: pytest tests/ -v

Test Modules:
    test_query_utils.py  - Query validation, URL parsing, string utilities
    test_relevance.py    - Keyword relevance scoring for channels
    test_youtube_api.py  - YouTube Data API wrappers (mocked)
    test_gemini_api.py   - Gemini AI API functions (mocked)
    test_pipeline.py     - Search pipeline integration tests

Coverage:
    - 120 tests total
    - All core modules have test coverage
    - API calls are mocked for isolation
"""

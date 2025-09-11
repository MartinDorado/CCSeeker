def test_imports():
    """Test that our modules import without errors."""
    try:
        from mcp_server.server import CCSeekerMCPServer
        from mcp_server.tools.youtube import YouTubeTools
        from mcp_server.tools.ai_generation import AIGenerationTools
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

if __name__ == "__main__":
    test_imports()
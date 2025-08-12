import google.generativeai as genai
import os

# --- Paste your Gemini API Key here ---
GEMINI_API_KEY = "AIzaSyAgaInyRL0MMYwy8_-OCf7A2UkXVz2_RGU"

print("Checking available Gemini models...")

if not GEMINI_API_KEY or GEMINI_API_KEY == "PASTE_YOUR_GEMINI_API_KEY_HERE":
    print("Error: Please paste your Gemini API key into the script.")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)

        print("\nAvailable models and their supported methods:")
        for m in genai.list_models():
            # We are looking for a model that supports the 'generateContent' method
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")

    except Exception as e:
        print(f"\nAn error occurred while trying to list the models: {e}")
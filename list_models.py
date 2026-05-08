import google.generativeai as genai
import re

# Extract API key from app.py
try:
    with open('app.py', 'r') as f:
        content = f.read()
        match = re.search(r"app\.config\['GEMINI_API_KEY'\]\s*=\s*'([^']+)'", content)
        if match:
            api_key = match.group(1)
            print(f"Found API Key: {api_key[:5]}...")
            genai.configure(api_key=api_key)
            
            print("Listing available models...")
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    print(m.name)
        else:
            print("API Key not found in app.py")
except Exception as e:
    print(f"Error: {e}")

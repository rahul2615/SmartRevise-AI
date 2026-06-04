import google.generativeai as genai

# Cache the last known working model name to avoid retrying failed models in subsequent calls.
_working_models = {
    "fast": None,
    "think": None
}

def generate_gemini_content(api_key, prompt, model_type="fast"):
    """
    Generates content using Gemini API by trying candidate models.
    Supports fallback if a model is deprecated, not found (404), or rate-limited/quota-exceeded (429).
    """
    genai.configure(api_key=api_key)
    
    if model_type == "think":
        candidates = ['gemini-2.5-pro', 'gemini-3.5-flash', 'gemini-flash-latest']
    else:
        candidates = ['gemini-3.5-flash', 'gemini-flash-latest', 'gemini-2.5-flash']

    # Prioritize the last known working model if we found one
    cached_model = _working_models.get(model_type)
    if cached_model and cached_model in candidates:
        # Move it to the front of the candidate list
        candidates = [cached_model] + [c for c in candidates if c != cached_model]

    last_err = None
    for model_name in candidates:
        try:
            print(f"[Gemini Helper] Trying model '{model_name}'...")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if response and response.text:
                _working_models[model_type] = model_name
                return response
        except Exception as e:
            last_err = e
            print(f"[Gemini Helper] Model '{model_name}' failed: {e}")
            continue

    if last_err:
        raise last_err
    raise Exception("All Gemini candidate models failed to generate content.")

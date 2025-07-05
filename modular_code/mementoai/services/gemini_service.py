import google.generativeai as genai
from config.settings import settings

genai_configured = False

def configure_gemini():
    """
    Configures the Google Gemini API.
    """
    global genai_configured
    try:
        if settings.GOOGLE_API_KEY:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            genai_configured = True
            print("INFO: Google API Key configured.")
        else:
            print("WARNING: Google API Key not found/configured.")
    except Exception as e:
        print(f"ERROR during Gemini API Key setup: {e}")
        genai_configured = False

def generate_summary(prompt: str) -> str:
    """
    Generates a summary using the configured Gemini model.
    """
    if not genai_configured:
        return "AI Summarization skipped (API Key not configured)."

    try:
        model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)
        response = model.generate_content(prompt, request_options={"timeout": 60})
        return response.text if response.parts else "Gemini empty response."
    except Exception as e:
        print(f"ERROR calling Gemini: {e}")
        return f"Error during AI summarization: {e}"

# Configure Gemini when the module is imported
configure_gemini()


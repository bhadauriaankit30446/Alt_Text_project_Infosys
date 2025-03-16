import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Gemini Configuration
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

def get_gemini_client():
    """Initialize and return Gemini client"""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    genai.configure(api_key=GEMINI_API_KEY)
    return genai

# Model configurations
GEMINI_CONFIG = {
    "text_model": "gemini-1.5-flash",  # For text generation
    "vision_model": "gemini-1.5-flash-002",  # For image analysis
    "temperature": 0.7,
    "max_output_tokens": 2048,
}

def format_success_response(data):
    """Format successful response"""
    return {
        'success': True,
        'data': data
    }

def format_error_response(error_message, error_code=None):
    """Format error response"""
    return {
        'success': False,
        'error': {
            'message': error_message,
            'code': error_code
        }
    } 
"""
Centralized configuration for AI services
"""
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_gemini_client():
    """Initialize and return Gemini client"""
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    genai.configure(api_key=api_key)
    return genai

# Standard model configurations
GEMINI_CONFIG = {
    "text_model": "gemini-1.5-flash",
    "vision_model": "gemini-1.5-flash-002",
    "temperature": 0.7,
    "max_output_tokens": 2048
}

# Response formatting helpers
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
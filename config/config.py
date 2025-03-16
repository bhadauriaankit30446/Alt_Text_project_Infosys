import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Upload Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# File Size Limits
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size for regular uploads

# Gemini Config
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

# Flask Config
FLASK_ENV = os.environ.get('FLASK_ENV', 'development')
FLASK_DEBUG = os.environ.get('FLASK_DEBUG', '1') == '1'
SECRET_KEY = os.environ.get('SECRET_KEY', 'default-secret-key')

# Model Config
BLIP_MODEL = "Salesforce/blip-image-captioning-base"

# Medical Image Config
MEDICAL_IMAGE_EXTENSIONS = {'dcm', 'tiff', 'png', 'jpg', 'jpeg'}
MAX_MEDICAL_IMAGE_SIZE = 32 * 1024 * 1024  # 32MB max for medical images 

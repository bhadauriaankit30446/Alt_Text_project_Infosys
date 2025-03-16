from PIL import Image
import logging
from config.ai_config import get_gemini_client, GEMINI_CONFIG, format_success_response, format_error_response

logger = logging.getLogger(__name__)

def analyze_medical_image(image_path, context=None):
    """
    Analyze medical image using Gemini Vision
    Args:
        image_path (str): Path to the medical image
        context (str, optional): Additional context about the image
    Returns:
        dict: Analysis results
    """
    try:
        # Load and validate image
        try:
            image = Image.open(image_path)
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            return format_error_response("Invalid image file", "INVALID_IMAGE")

        # Initialize Gemini
        genai = get_gemini_client()
        model = genai.GenerativeModel(GEMINI_CONFIG['vision_model'])

        # Prepare prompt
        prompt = """Analyze this medical image and provide a detailed report.
        Focus on:
        1. Anatomical structures visible
        2. Any notable patterns or abnormalities
        3. Image quality and technical aspects
        4. Potential clinical relevance
        
        Format your response in these sections:
        1. Technical Assessment
        2. Anatomical Observations
        3. Notable Findings
        4. Recommendations
        
        Remember: This is for educational purposes only, not for diagnosis."""

        if context:
            prompt += f"\n\nAdditional Context: {context}"

        # Generate analysis
        response = model.generate_content([prompt, image])
        analysis = response.text.strip()

        # Parse sections
        sections = parse_medical_report(analysis)

        return format_success_response({
            'analysis': sections,
            'raw_response': analysis
        })

    except Exception as e:
        logger.error(f"Error analyzing medical image: {str(e)}")
        return format_error_response(str(e), "ANALYSIS_ERROR")

def parse_medical_report(text):
    """Parse medical report into sections"""
    sections = {
        'technical_assessment': '',
        'anatomical_observations': '',
        'notable_findings': '',
        'recommendations': ''
    }
    
    current_section = None
    current_content = []
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if 'Technical Assessment' in line:
            current_section = 'technical_assessment'
        elif 'Anatomical Observations' in line:
            if current_section == 'technical_assessment':
                sections['technical_assessment'] = '\n'.join(current_content)
            current_section = 'anatomical_observations'
            current_content = []
        elif 'Notable Findings' in line:
            if current_section == 'anatomical_observations':
                sections['anatomical_observations'] = '\n'.join(current_content)
            current_section = 'notable_findings'
            current_content = []
        elif 'Recommendations' in line:
            if current_section == 'notable_findings':
                sections['notable_findings'] = '\n'.join(current_content)
            current_section = 'recommendations'
            current_content = []
        elif current_section and not line.startswith(('1.', '2.', '3.', '4.')):
            current_content.append(line)
            
    # Handle last section
    if current_section == 'recommendations':
        sections['recommendations'] = '\n'.join(current_content)
        
    return sections
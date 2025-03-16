from config.ai_config import get_gemini_client, format_success_response, format_error_response, GEMINI_CONFIG
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_seo_description(context, alt_text=None):
    """
    Generate SEO-optimized content from image context
    Args:
        context (str): Image context to generate SEO content from
        alt_text (str, optional): Additional alt text for context
    Returns:
        dict: Response containing generated SEO content
    """
    try:
        genai = get_gemini_client()
        model = genai.GenerativeModel(GEMINI_CONFIG['text_model'])
        
        base_prompt = f"""Generate SEO-optimized content for this image:

Context: {context}
"""
        if alt_text:
            base_prompt += f"\nAdditional Description: {alt_text}"
            
        base_prompt += """

Please provide:
1. A compelling meta title (50-60 characters)
2. A meta description (150-160 characters)
3. Three alternative titles for A/B testing
4. Five relevant keywords
5. A detailed product description (200-300 words)

Format the response in clear sections."""

        response = model.generate_content(base_prompt)
        content = response.text.strip()
        
        # Parse the sections
        sections = parse_seo_content(content)
        
        # Generate social media variations
        social_prompt = f"""Generate social media variations for this content:

Original Content: {content}

Provide:
1. Three Instagram captions (each under 200 characters)
2. Three Twitter/X posts (each under 280 characters)
3. A longer Facebook post (400-600 characters)
4. Five relevant hashtags

Format each variation clearly."""

        social_response = model.generate_content(social_prompt)
        social_content = social_response.text.strip()
        
        # Combine results
        sections.update(parse_social_content(social_content))
        
        return format_success_response(sections)
        
    except Exception as e:
        logger.error(f"Error generating SEO content: {str(e)}")
        return format_error_response(str(e), 'SEO_GENERATION_ERROR')

def parse_seo_content(content):
    """Parse SEO content into sections"""
    sections = {
        'meta_title': '',
        'meta_description': '',
        'alternative_titles': [],
        'keywords': [],
        'product_description': ''
    }
    
    current_section = None
    current_content = []
    
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if 'meta title' in line.lower() or 'title:' in line.lower():
            current_section = 'meta_title'
        elif 'meta description' in line.lower() or 'description:' in line.lower():
            if current_section == 'meta_title':
                sections['meta_title'] = ' '.join(current_content).strip()
            current_section = 'meta_description'
            current_content = []
        elif 'alternative titles' in line.lower() or 'a/b testing' in line.lower():
            if current_section == 'meta_description':
                sections['meta_description'] = ' '.join(current_content).strip()
            current_section = 'alternative_titles'
            current_content = []
        elif 'keywords' in line.lower():
            if current_section == 'alternative_titles':
                sections['alternative_titles'] = [t.strip() for t in current_content if t.strip()]
            current_section = 'keywords'
            current_content = []
        elif 'product description' in line.lower():
            if current_section == 'keywords':
                sections['keywords'] = [k.strip() for k in ' '.join(current_content).split(',')]
            current_section = 'product_description'
            current_content = []
        elif current_section:
            current_content.append(line)
            
    # Handle last section
    if current_section == 'product_description':
        sections['product_description'] = ' '.join(current_content).strip()
        
    return sections

def parse_social_content(content):
    """Parse social media content into sections"""
    sections = {
        'instagram_captions': [],
        'twitter_posts': [],
        'facebook_post': '',
        'hashtags': []
    }
    
    current_section = None
    current_content = []
    
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if 'instagram' in line.lower():
            current_section = 'instagram_captions'
            current_content = []
        elif 'twitter' in line.lower() or 'x post' in line.lower():
            if current_section == 'instagram_captions':
                sections['instagram_captions'] = [c.strip() for c in current_content if c.strip()]
            current_section = 'twitter_posts'
            current_content = []
        elif 'facebook' in line.lower():
            if current_section == 'twitter_posts':
                sections['twitter_posts'] = [c.strip() for c in current_content if c.strip()]
            current_section = 'facebook_post'
            current_content = []
        elif 'hashtag' in line.lower():
            if current_section == 'facebook_post':
                sections['facebook_post'] = ' '.join(current_content).strip()
            current_section = 'hashtags'
            current_content = []
        elif current_section and not line.lower().startswith(('1.', '2.', '3.', '4.', '5.')):
            current_content.append(line)
            
    # Handle last section
    if current_section == 'hashtags':
        sections['hashtags'] = [h.strip() for h in ' '.join(current_content).split('#') if h.strip()]
        
    return sections 

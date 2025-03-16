from nltk.sentiment.vader import SentimentIntensityAnalyzer
from config.gemini_config import get_gemini_client, format_success_response, format_error_response, GEMINI_CONFIG
import logging
import re
from itertools import groupby
from app.services.image_service import image_processor
import PIL.Image

logger = logging.getLogger(__name__)

def generate_context(alt_text):
    """
    Generates context from alt text using Gemini.
    Args:
        alt_text (str): Alt text to generate context from
    Returns:
        dict: Response containing generated context
    """
    try:
        # Clean the alt text first
        cleaned_alt_text = clean_text(alt_text)
        
        genai = get_gemini_client()
        model = genai.GenerativeModel(GEMINI_CONFIG['text_model'])
        
        prompt = f"""
        Generate a clear and concise description for this image.
        Avoid any repetition or redundant phrases.
        
        Original description: {cleaned_alt_text}
        """

        response = model.generate_content(prompt)
        context = response.text.strip()
        return format_success_response({'context': context})

    except Exception as e:
        logger.error(f"Error generating context: {str(e)}")
        return format_error_response(str(e), 'CONTEXT_GENERATION_ERROR')

def enhance_context(context):
    """
    Enhances the context with additional details using Gemini.
    Args:
        context (str): Original context to enhance
    Returns:
        dict: Response containing enhanced context
    """
    try:
        genai = get_gemini_client()
        model = genai.GenerativeModel(GEMINI_CONFIG['text_model'])
        
        prompt = f"""Enhance this context with more descriptive details while maintaining accuracy:

Original: {context}

Requirements:
1. Add sensory details
2. Include specific measurements or technical details if applicable
3. Maintain factual accuracy
4. Keep the enhanced version under 100 words"""

        response = model.generate_content(prompt)
        enhanced = response.text.strip()
        return format_success_response({'enhanced_context': enhanced})
    except Exception as e:
        return format_error_response(
            error_message=f"Error enhancing context: {str(e)}",
            error_code="CONTEXT_ENHANCEMENT_ERROR"
        )

def clean_text(text, remove_duplicates=True, remove_repetitive_chars=True):
    """Unified text cleaning function"""
    if not text:
        return text
        
    # Split into words
    words = text.split()
    
    if remove_duplicates:
        # Remove consecutive duplicate words
        words = [next(group) for key, group in groupby(words)]
    
    # Join words back together
    cleaned_text = ' '.join(words)
    
    if remove_repetitive_chars:
        # Remove repetitive characters within words
        cleaned_text = re.sub(r'(.)\1+', r'\1', cleaned_text)
    
    return cleaned_text

def social_media_caption(context):
    """Generate engaging social media caption using Gemini"""
    try:
        genai = get_gemini_client()
        model = genai.GenerativeModel(GEMINI_CONFIG['text_model'])
        
        prompt = f"""
        Create an engaging social media caption for this image.
        Make it conversational, include relevant emojis, and keep it under 200 characters.
        Avoid any word repetition or stuttering patterns.
        
        Image context: {context}
        """

        response = model.generate_content(prompt)
        caption = response.text.strip()
        # Clean any potential repetitions
        caption = clean_text(caption)
        return format_success_response({'caption': caption})

    except Exception as e:
        logger.error(f"Error generating caption: {str(e)}")
        return format_error_response(str(e), 'CAPTION_GENERATION_ERROR')

def analyze_sentiment(text):
    """
    Analyzes sentiment of text using VADER.
    Args:
        text (str): Text to analyze
    Returns:
        dict: Response containing sentiment analysis
    """
    try:
        if not text:
            return format_error_response(
                error_message="No text provided for sentiment analysis",
                error_code="EMPTY_TEXT_ERROR"
            )

        try:
            analyzer = SentimentIntensityAnalyzer()
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzer: {str(e)}")
            return format_error_response(
                error_message="Error initializing sentiment analyzer. Please ensure NLTK data is properly installed.",
                error_code="SENTIMENT_INIT_ERROR"
            )

        try:
            scores = analyzer.polarity_scores(text)
        except Exception as e:
            logger.error(f"Error calculating sentiment scores: {str(e)}")
            return format_error_response(
                error_message="Error calculating sentiment scores",
                error_code="SENTIMENT_CALCULATION_ERROR"
            )
        
        # Determine sentiment category
        compound = scores['compound']
        if compound >= 0.05:
            category = 'Positive'
        elif compound <= -0.05:
            category = 'Negative'
        else:
            category = 'Neutral'
            
        return format_success_response({
            'sentiment': {
                'score': compound,
                'category': category,
                'details': scores
            }
        })
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        return format_error_response(
            error_message=f"Error analyzing sentiment: {str(e)}",
            error_code="SENTIMENT_ANALYSIS_ERROR"
        )

def analyze_medical_image(image_path, alt_text):
    """
    Analyzes medical image and generates detailed report using Gemini.
    """
    try:
        print("Starting medical image analysis") # Debug log
        
        if not image_path or not alt_text:
            return format_error_response(
                error_message="Image and alt text are required for analysis",
                error_code="MISSING_INPUT"
            )

        # First get base image description using BLIP
        base_description = image_processor.generate_alt_text(image_path)
        print(f"Base description: {base_description}") # Debug log
        
        genai = get_gemini_client()
        model = genai.GenerativeModel(GEMINI_CONFIG['vision_model'])
        
        # Load the image
        image = PIL.Image.open(image_path)
        
        prompt = f"""Analyze this medical image and provide a detailed medical report:

Base Image Description: {base_description}
Additional Context: {alt_text}

Provide a comprehensive analysis in this exact format:

1. Key Findings:
[Provide detailed observations about visible anatomical structures, tissue characteristics, and any notable patterns]

2. Potential Observations:
[List possible medical interpretations, noting any concerning patterns or areas needing attention]

3. Recommendations:
[Suggest specific follow-up actions, additional tests, or monitoring protocols]

Use precise medical terminology and maintain professional objectivity."""

        response = model.generate_content([prompt, image])
        analysis = response.text.strip()
        print(f"Gemini response: {analysis}") # Debug log
        
        # Parse sections
        sections = {
            'key_findings': '',
            'potential_observations': '',
            'recommendations': ''
        }
        
        current_section = None
        for line in analysis.split('\n'):
            if '1. Key Findings:' in line:
                current_section = 'key_findings'
            elif '2. Potential Observations:' in line:
                current_section = 'potential_observations'
            elif '3. Recommendations:' in line:
                current_section = 'recommendations'
            elif current_section and line.strip():
                sections[current_section] += line.strip() + '\n'
        
        return format_success_response({
            'analysis': sections,
            'raw_response': analysis
        })
        
    except Exception as e:
        logger.error(f"Error analyzing medical image: {str(e)}")
        return format_error_response(str(e), 'MEDICAL_ANALYSIS_ERROR')

def generate_hashtags(text):
    """Generate relevant hashtags from the text"""
    try:
        genai = get_gemini_client()
        model = genai.GenerativeModel(GEMINI_CONFIG['text_model'])
        
        prompt = f"""
        Generate 8-10 relevant, trending hashtags for this social media post.
        Make them specific, engaging, and properly formatted with # symbol.
        Each hashtag should be a single word or compound words without spaces.
        
        Text: {text}
        
        Example format:
        #Photography #Nature #Wildlife #Beautiful
        """

        response = model.generate_content(prompt)
        # Extract hashtags from response
        hashtags_text = response.text.strip()
        # Split by spaces and filter valid hashtags
        hashtags = [tag.strip() for tag in hashtags_text.split() if tag.strip().startswith('#')]
        
        # Ensure we have at least some hashtags
        if not hashtags:
            hashtags = ["#Photography", "#Social", "#Content"]  # Default fallback hashtags
            
        logger.info(f"Generated hashtags: {hashtags}")  # Add logging
        return format_success_response({'hashtags': hashtags})

    except Exception as e:
        logger.error(f"Error generating hashtags: {str(e)}")
        return format_error_response(str(e), 'HASHTAG_GENERATION_ERROR')

def enhance_alt_text(alt_text, min_words=6):
    """
    Ensures alt text is at least the minimum number of words.
    If it's shorter, enhances it using Gemini.
    
    Args:
        alt_text (str): Original alt text
        min_words (int): Minimum number of words required
        
    Returns:
        dict: Response containing enhanced alt text
    """
    try:
        # Count words in alt text
        word_count = len(alt_text.split())
        
        # If already meets minimum length, return as is
        if word_count >= min_words:
            return format_success_response({'enhanced_alt_text': alt_text})
        
        # Otherwise, enhance it using Gemini
        genai = get_gemini_client()
        model = genai.GenerativeModel(GEMINI_CONFIG['text_model'])
        
        prompt = f"""
        Enhance this image description to be more detailed and descriptive.
        The current description is too short: "{alt_text}"
        
        Please provide a more detailed description that is AT LEAST {min_words} words long.
        Focus on what is visible in the image, spatial relationships, and key details.
        Keep it factual and objective.
        """
        
        response = model.generate_content(prompt)
        enhanced_text = response.text.strip()
        
        # Ensure the enhanced text is actually longer
        if len(enhanced_text.split()) < min_words:
            # If still too short, add generic descriptive terms
            enhanced_text = f"{enhanced_text} with various details and elements visible in the scene"
        
        logger.info(f"Enhanced alt text from '{alt_text}' to '{enhanced_text}'")
        return format_success_response({'enhanced_alt_text': enhanced_text})
        
    except Exception as e:
        logger.error(f"Error enhancing alt text: {str(e)}")
        # Return original text as fallback
        return format_success_response({'enhanced_alt_text': alt_text})

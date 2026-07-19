#!/usr/bin/env python3
import sys
import re
from textblob import TextBlob

def clean_text(text):
    """Clean text for sentiment analysis while preserving emotional context"""
    # Remove code blocks but preserve surrounding context
    text = re.sub(r'```[\s\S]*?```', ' ', text)
    text = re.sub(r'`[^`]*`', ' ', text)
    # Remove URLs but keep surrounding words
    text = re.sub(r'https?://\S+', ' ', text)
    # Remove HTML tags but keep content
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove markdown formatting but keep text
    text = re.sub(r'[#*_\[\](){}]', ' ', text)
    # Remove excessive punctuation
    text = re.sub(r'[.]{2,}', '.', text)
    text = re.sub(r'[-]{2,}', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def analyze_sentiment_with_textblob(text):
    """Analyze sentiment using TextBlob with confidence filtering"""
    cleaned_text = clean_text(text)
    print(f"DEBUG: Cleaned text preview: {cleaned_text[:200]}...", file=sys.stderr)
    
    # Filter out very short or low-quality text
    if len(cleaned_text) < 30:
        print("DEBUG: Text too short, returning neutral", file=sys.stderr)
        return 0
    
    blob = TextBlob(cleaned_text)
    sentiment = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # If text is very objective (low subjectivity), sentiment is less reliable
    if subjectivity < 0.1:
        print(f"DEBUG: Low subjectivity ({subjectivity:.3f}), dampening sentiment", file=sys.stderr)
        sentiment *= 0.5
    
    print(f"DEBUG: TextBlob sentiment: {sentiment} (subjectivity: {subjectivity:.3f})", file=sys.stderr)
    return sentiment

def main():
    text = sys.stdin.read().strip()

    if not text or len(text) < 10:
        print(0)
        return

    sentiment = analyze_sentiment_with_textblob(text)

    # Clamp sentiment to reasonable bounds (-1 to 1)
    sentiment = max(-1, min(1, sentiment))

    print(sentiment)

if __name__ == "__main__":
    main()

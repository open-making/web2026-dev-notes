#!/usr/bin/env python3
import sys
import re

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# Model context is 512 tokens; leave headroom for special tokens.
MAX_CHUNK_TOKENS = 450

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

def chunk_text(text, tokenizer):
    """Split text into chunks that fit the model's token limit"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and len(tokenizer.encode(candidate)) > MAX_CHUNK_TOKENS:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks

def analyze_sentiment(text):
    """Score text in [-1, 1] with a transformer model, chunking long input"""
    cleaned_text = clean_text(text)
    print(f"DEBUG: Cleaned text preview: {cleaned_text[:200]}...", file=sys.stderr)

    # Filter out very short or low-quality text
    if len(cleaned_text) < 30:
        print("DEBUG: Text too short, returning neutral", file=sys.stderr)
        return 0

    from transformers import pipeline
    classifier = pipeline("sentiment-analysis", model=MODEL, top_k=None)

    chunks = chunk_text(cleaned_text, classifier.tokenizer)
    print(f"DEBUG: Scoring {len(chunks)} chunk(s)", file=sys.stderr)

    # Average pos - neg across chunks, weighted by chunk length
    weighted_sum = 0.0
    total_weight = 0
    for chunk in chunks:
        scores = {r["label"].lower(): r["score"]
                  for r in classifier(chunk, truncation=True, max_length=512)[0]}
        chunk_sentiment = scores.get("positive", 0) - scores.get("negative", 0)
        weighted_sum += chunk_sentiment * len(chunk)
        total_weight += len(chunk)
        print(f"DEBUG: Chunk sentiment: {chunk_sentiment:.3f} ({len(chunk)} chars)", file=sys.stderr)

    sentiment = weighted_sum / total_weight
    print(f"DEBUG: Overall sentiment: {sentiment:.3f}", file=sys.stderr)
    return sentiment

def main():
    text = sys.stdin.read().strip()

    if not text or len(text) < 10:
        print(0)
        return

    try:
        sentiment = analyze_sentiment(text)
    except Exception as e:
        # Never break CI on a model download/inference hiccup
        print(f"DEBUG: Sentiment analysis failed: {e}", file=sys.stderr)
        print(0)
        return

    # Clamp sentiment to reasonable bounds (-1 to 1)
    sentiment = max(-1, min(1, sentiment))

    print(sentiment)

if __name__ == "__main__":
    main()

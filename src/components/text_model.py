"""
Text Model: Sentiment Analysis
Uses DistilBERT for lightweight sentiment classification
Sentiments: Positive, Negative, Neutral
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class TextSentimentAnalyzer:
    """
    Sentiment analysis using DistilBERT (lightweight transformer)
    CPU-friendly and efficient
    """
    
    SENTIMENTS = ['Negative', 'Neutral', 'Positive']
    
    def __init__(self, model_name='distilbert-base-uncased-finetuned-sst-2-english'):
        """
        Initialize sentiment analyzer
        
        Args:
            model_name: HuggingFace model name
        """
        self.device = torch.device('cpu')
        print("Loading text sentiment model...")
        
        # Load pre-trained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print("✓ Text model loaded")
    
    def preprocess_text(self, text: str) -> Dict:
        """
        Tokenize and preprocess text
        
        Args:
            text: Input text string
        
        Returns:
            Tokenized inputs
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def predict(self, text: str) -> Dict[str, float]:
        """
        Predict sentiment probabilities
        
        Args:
            text: Input text
        
        Returns:
            Dict of {sentiment: probability}
        """
        # Preprocess
        inputs = self.preprocess_text(text)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
        
        # Convert to dict
        probs = probabilities.cpu().numpy()[0]
        
        # Map model outputs to our sentiment labels
        # DistilBERT SST-2 gives: [negative, positive]
        # We add neutral as middle ground
        neg_score = float(probs[0])
        pos_score = float(probs[1])
        
        # Calculate neutral score (if scores are close)
        diff = abs(pos_score - neg_score)
        if diff < 0.3:  # Ambiguous -> likely neutral
            neutral_score = 1 - diff
            neg_score *= (1 - neutral_score)
            pos_score *= (1 - neutral_score)
        else:
            neutral_score = 0.0
        
        # Normalize
        total = neg_score + neutral_score + pos_score
        
        return {
            'Negative': neg_score / total,
            'Neutral': neutral_score / total,
            'Positive': pos_score / total
        }
    
    def get_top_sentiment(self, text: str) -> Tuple[str, float]:
        """
        Get most likely sentiment
        
        Args:
            text: Input text
        
        Returns:
            (sentiment_name, confidence)
        """
        sentiments = self.predict(text)
        return max(sentiments.items(), key=lambda x: x[1])
    
    def analyze_batch(self, texts: list) -> list:
        """
        Analyze multiple texts
        
        Args:
            texts: List of text strings
        
        Returns:
            List of sentiment dicts
        """
        return [self.predict(text) for text in texts]


# Test function
if __name__ == "__main__":
    print("Testing TextSentimentAnalyzer...")
    
    # Initialize
    analyzer = TextSentimentAnalyzer()
    
    # Test cases
    test_texts = [
        "I love this! It's absolutely amazing!",
        "This is terrible and disappointing.",
        "It's okay, nothing special.",
        "No, I think the project is going really well."
    ]
    
    print("\n📊 Sample Predictions:\n")
    for text in test_texts:
        sentiments = analyzer.predict(text)
        top_sentiment, confidence = analyzer.get_top_sentiment(text)
        
        print(f"Text: '{text}'")
        print(f"Top: {top_sentiment} ({confidence*100:.1f}%)")
        for sent, prob in sorted(sentiments.items(), key=lambda x: x[1], reverse=True):
            print(f"  {sent}: {prob*100:.1f}%")
        print()

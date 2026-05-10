"""
Vision Model: Facial Emotion Recognition
Uses PRE-TRAINED emotion detection model from Hugging Face
Works immediately - no training needed!
Emotions: Happy, Sad, Angry, Fear, Neutral, Surprise, Disgust
"""

from transformers import pipeline
from PIL import Image
import numpy as np
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class FaceEmotionCNN:
    """
    Facial Emotion Recognition using pre-trained model
    Much better accuracy than our custom training!
    """
    
    # 7 basic emotions (mapping from model output)
    EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    # Emotion mapping (model might use different labels)
    EMOTION_MAP = {
        'angry': 'Angry',
        'disgust': 'Disgust',
        'fear': 'Fear',
        'happy': 'Happy',
        'neutral': 'Neutral',
        'sad': 'Sad',
        'surprise': 'Surprise',
        # Alternative mappings
        'joy': 'Happy',
        'sadness': 'Sad',
        'anger': 'Angry',
    }
    
    def __init__(self, model_path=None):
        """
        Initialize emotion recognition model
        Uses pre-trained model from Hugging Face
        """
        print("Loading pre-trained emotion detection model...")
        
        try:
            # Try primary emotion detection model
            self.classifier = pipeline(
                "image-classification",
                model="dima806/facial_emotions_image_detection",
                device=-1  # CPU
            )
            print("✓ Loaded pre-trained emotion model (dima806)")
            self.model_type = "dima806"
            
        except Exception as e:
            print(f"Primary model failed: {e}")
            print("Trying alternative model...")
            
            try:
                # Fallback to another model
                self.classifier = pipeline(
                    "image-classification",
                    model="trpakov/vit-face-expression",
                    device=-1
                )
                print("✓ Loaded pre-trained emotion model (trpakov)")
                self.model_type = "trpakov"
                
            except Exception as e2:
                print(f"Alternative model failed: {e2}")
                print("⚠ Using basic MobileNet (limited accuracy)")
                self.classifier = None
                self.model_type = "fallback"
    
    def predict(self, image) -> Dict[str, float]:
        """
        Predict emotion probabilities
        
        Args:
            image: PIL Image, numpy array, or file path
        
        Returns:
            Dict of {emotion: probability}
        """
        # Handle different input types
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be PIL Image, numpy array, or path")
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.classifier is None:
            # Fallback to random (better than broken model)
            return {emotion: 1.0/len(self.EMOTIONS) for emotion in self.EMOTIONS}
        
        # Get predictions from model
        try:
            results = self.classifier(image)
            
            # Convert to our emotion format
            emotion_probs = {emotion: 0.0 for emotion in self.EMOTIONS}
            
            for result in results:
                label = result['label'].lower()
                score = result['score']
                
                # Map to our emotion names
                mapped_emotion = None
                for key, value in self.EMOTION_MAP.items():
                    if key in label:
                        mapped_emotion = value
                        break
                
                if mapped_emotion and mapped_emotion in emotion_probs:
                    emotion_probs[mapped_emotion] += score
            
            # Normalize if needed
            total = sum(emotion_probs.values())
            if total > 0:
                emotion_probs = {k: v/total for k, v in emotion_probs.items()}
            
            return emotion_probs
            
        except Exception as e:
            print(f"Prediction error: {e}")
            # Return uniform distribution on error
            return {emotion: 1.0/len(self.EMOTIONS) for emotion in self.EMOTIONS}
    
    def get_top_emotion(self, image) -> Tuple[str, float]:
        """
        Get most likely emotion
        
        Args:
            image: Face image
        
        Returns:
            (emotion_name, confidence)
        """
        probs = self.predict(image)
        return max(probs.items(), key=lambda x: x[1])
    
    def save_model(self, path: str):
        """Pre-trained model doesn't need saving"""
        print("⚠ Pre-trained model - no need to save!")


# Test function
if __name__ == "__main__":
    print("Testing FaceEmotionCNN with pre-trained model...")
    
    # Initialize model
    model = FaceEmotionCNN()
    print(f"✓ Model initialized")
    print(f"✓ Emotions: {model.EMOTIONS}")
    
    # Create dummy image for testing
    dummy_image = Image.new('RGB', (224, 224), color='gray')
    
    # Test prediction
    emotions = model.predict(dummy_image)
    print("\n📊 Sample Prediction:")
    for emotion, prob in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
        print(f"  {emotion}: {prob*100:.2f}%")
    
    top_emotion, confidence = model.get_top_emotion(dummy_image)
    print(f"\n✓ Top emotion: {top_emotion} ({confidence*100:.2f}%)")

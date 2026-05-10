"""
Vision Model: Facial Emotion Recognition
Uses pre-trained CNN for emotion detection from face images
NOW LOADS TRAINED MODEL AUTOMATICALLY!
Emotions: Happy, Sad, Angry, Fear, Neutral, Surprise, Disgust
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from typing import Dict, Tuple
import os
import warnings
warnings.filterwarnings('ignore')


class FaceEmotionCNN:
    """
    Facial Emotion Recognition using MobileNetV2 (lightweight, CPU-friendly)
    """
    
    # 7 basic emotions
    EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    def __init__(self, model_path=None):
        """
        Initialize emotion recognition model
        
        Args:
            model_path: Path to saved weights (optional, auto-detects trained model)
        """
        self.device = torch.device('cpu')
        self.model = self._build_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Auto-load trained model if exists
        if model_path is None:
            # Check for trained model
            trained_path = 'models/vision_trained.pth'
            if os.path.exists(trained_path):
                model_path = trained_path
                print(f"🎯 Found trained model: {trained_path}")
        
        if model_path:
            self._load_weights(model_path)
    
    def _build_model(self) -> nn.Module:
        """Build CNN using pre-trained MobileNetV2"""
        # Lightweight backbone
        model = models.mobilenet_v2(pretrained=True)
        
        # Freeze early layers
        for param in model.features[:-3].parameters():
            param.requires_grad = False
        
        # Custom classifier for emotions
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, len(self.EMOTIONS))
        )
        
        return model
    
    def _load_weights(self, model_path: str):
        """Load trained weights"""
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✓ Loaded trained weights from {model_path}")
        except Exception as e:
            print(f"⚠ Could not load weights: {e}")
            print("Using base pre-trained model")
    
    def preprocess_image(self, image) -> torch.Tensor:
        """
        Preprocess image for model
        
        Args:
            image: PIL Image, numpy array, or file path
        
        Returns:
            Preprocessed tensor
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
        
        # Transform
        tensor = self.transform(image)
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def predict(self, image) -> Dict[str, float]:
        """
        Predict emotion probabilities
        
        Args:
            image: Face image
        
        Returns:
            Dict of {emotion: probability}
        """
        # Preprocess
        input_tensor = self.preprocess_image(image).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        # Convert to dict
        probs = probabilities.cpu().numpy()[0]
        return {
            emotion: float(prob) 
            for emotion, prob in zip(self.EMOTIONS, probs)
        }
    
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
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
        print(f"✓ Model saved to {path}")


# Test function
if __name__ == "__main__":
    print("Testing FaceEmotionCNN...")
    
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

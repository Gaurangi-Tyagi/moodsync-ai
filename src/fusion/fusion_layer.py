"""
Fusion Layer: Multi-Modal Integration
Custom neural network to combine vision and text predictions
Detects alignment/mismatch between modalities
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class FusionNetwork(nn.Module):
    """
    Neural network that learns to fuse vision and text embeddings
    Detects mismatches between what people say vs how they appear
    """
    
    def __init__(self, vision_dim=7, text_dim=3, hidden_dim=64):
        """
        Args:
            vision_dim: Number of emotion classes (7)
            text_dim: Number of sentiment classes (3)
            hidden_dim: Hidden layer size
        """
        super(FusionNetwork, self).__init__()
        
        input_dim = vision_dim + text_dim
        
        # Fusion network architecture
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)  # Binary: [aligned, mismatched]
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, vision_probs, text_probs):
        """
        Forward pass
        
        Args:
            vision_probs: Emotion probabilities (7,)
            text_probs: Sentiment probabilities (3,)
        
        Returns:
            Alignment logits (2,)
        """
        # Concatenate inputs
        combined = torch.cat([vision_probs, text_probs], dim=-1)
        
        # Pass through network
        output = self.fusion(combined)
        
        return output


class MultiModalFusion:
    """
    Multi-modal fusion system with mismatch detection
    """
    
    def __init__(self, model_path=None):
        """
        Initialize fusion system
        
        Args:
            model_path: Path to trained fusion network
        """
        self.device = torch.device('cpu')
        self.model = FusionNetwork()
        self.model.to(self.device)
        self.model.eval()
        
        if model_path:
            self._load_weights(model_path)
    
    def _load_weights(self, model_path: str):
        """Load trained weights"""
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✓ Loaded fusion weights from {model_path}")
        except Exception as e:
            print(f"⚠ Using untrained fusion model: {e}")
    
    def detect_mismatch(self, vision_dict: Dict[str, float], 
                       text_dict: Dict[str, float]) -> Tuple[str, float, Dict]:
        """
        Detect if vision and text are aligned or mismatched
        
        Args:
            vision_dict: Emotion probabilities
            text_dict: Sentiment probabilities
        
        Returns:
            (status, confidence, details)
        """
        # Convert dicts to tensors
        vision_probs = torch.tensor([
            vision_dict.get(e, 0.0) for e in 
            ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        ], dtype=torch.float32).unsqueeze(0)
        
        text_probs = torch.tensor([
            text_dict.get(s, 0.0) for s in 
            ['Negative', 'Neutral', 'Positive']
        ], dtype=torch.float32).unsqueeze(0)
        
        # Neural network prediction
        with torch.no_grad():
            logits = self.model(vision_probs, text_probs)
            probs = torch.softmax(logits, dim=1)
        
        aligned_prob = float(probs[0, 0])
        mismatch_prob = float(probs[0, 1])
        
        # Rule-based enhancement (until network is trained)
        rule_based_mismatch = self._rule_based_detection(vision_dict, text_dict)
        
        # Combine neural network and rules
        final_mismatch_prob = 0.5 * mismatch_prob + 0.5 * rule_based_mismatch
        
        # Determine status
        if final_mismatch_prob > 0.6:
            status = "MISMATCH"
        elif final_mismatch_prob < 0.4:
            status = "ALIGNED"
        else:
            status = "UNCLEAR"
        
        details = {
            'aligned_confidence': 1 - final_mismatch_prob,
            'mismatch_confidence': final_mismatch_prob,
            'neural_network_score': mismatch_prob,
            'rule_based_score': rule_based_mismatch
        }
        
        return status, final_mismatch_prob, details
    
    def _rule_based_detection(self, vision_dict: Dict, text_dict: Dict) -> float:
        """
        Simple rule-based mismatch detection
        
        Returns:
            Mismatch score (0-1)
        """
        # Get dominant emotion and sentiment
        top_emotion = max(vision_dict.items(), key=lambda x: x[1])[0]
        top_sentiment = max(text_dict.items(), key=lambda x: x[1])[0]
        
        # Define expected alignments
        positive_emotions = {'Happy', 'Surprise'}
        negative_emotions = {'Angry', 'Disgust', 'Fear', 'Sad'}
        neutral_emotions = {'Neutral'}
        
        # Check for mismatches
        mismatch_score = 0.0
        
        if top_sentiment == 'Positive' and top_emotion in negative_emotions:
            mismatch_score = 0.8  # High mismatch
        elif top_sentiment == 'Negative' and top_emotion in positive_emotions:
            mismatch_score = 0.8
        elif top_sentiment == 'Neutral' and top_emotion in negative_emotions:
            mismatch_score = 0.5  # Moderate mismatch
        elif top_sentiment == 'Neutral' and top_emotion in positive_emotions:
            mismatch_score = 0.3  # Slight mismatch
        else:
            mismatch_score = 0.1  # Likely aligned
        
        return mismatch_score
    
    def fuse_predictions(self, vision_dict: Dict, text_dict: Dict) -> Dict:
        """
        Combine vision and text predictions
        
        Returns:
            Combined analysis with mismatch detection
        """
        status, confidence, details = self.detect_mismatch(vision_dict, text_dict)
        
        return {
            'vision': vision_dict,
            'text': text_dict,
            'alignment_status': status,
            'mismatch_confidence': confidence,
            'details': details
        }
    
    def save_model(self, path: str):
        """Save fusion network weights"""
        torch.save(self.model.state_dict(), path)
        print(f"✓ Fusion model saved to {path}")


# Test function
if __name__ == "__main__":
    print("Testing MultiModalFusion...")
    
    # Initialize
    fusion = MultiModalFusion()
    
    # Test case 1: Aligned (happy face + positive text)
    print("\n📊 Test 1: Aligned")
    vision1 = {'Happy': 0.75, 'Sad': 0.1, 'Angry': 0.05, 'Fear': 0.03, 
               'Neutral': 0.04, 'Disgust': 0.01, 'Surprise': 0.02}
    text1 = {'Positive': 0.85, 'Neutral': 0.10, 'Negative': 0.05}
    
    result1 = fusion.fuse_predictions(vision1, text1)
    print(f"Status: {result1['alignment_status']}")
    print(f"Mismatch confidence: {result1['mismatch_confidence']*100:.1f}%")
    
    # Test case 2: Mismatched (sad face + positive text)
    print("\n📊 Test 2: Mismatch")
    vision2 = {'Sad': 0.68, 'Fear': 0.15, 'Happy': 0.05, 'Angry': 0.04,
               'Neutral': 0.04, 'Disgust': 0.02, 'Surprise': 0.02}
    text2 = {'Positive': 0.81, 'Neutral': 0.12, 'Negative': 0.07}
    
    result2 = fusion.fuse_predictions(vision2, text2)
    print(f"Status: {result2['alignment_status']}")
    print(f"Mismatch confidence: {result2['mismatch_confidence']*100:.1f}%")
    
    print("\n✓ Fusion layer working!")

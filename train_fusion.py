"""
Training Script for Fusion Neural Network
Trains the fusion layer to better detect emotion-sentiment mismatches
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple
import sys
import os

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src', 'fusion'))

from fusion_layer import FusionNetwork


class FusionTrainer:
    """
    Trainer for the fusion network
    """
    
    def __init__(self, learning_rate=0.001):
        """Initialize trainer"""
        self.device = torch.device('cpu')
        self.model = FusionNetwork()
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        print("✓ Fusion trainer initialized")
    
    def generate_training_data(self, num_samples=1000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate synthetic training data
        
        Returns:
            vision_data: (N, 7) emotion probabilities
            text_data: (N, 3) sentiment probabilities
            labels: (N,) 0=aligned, 1=mismatched
        """
        vision_data = []
        text_data = []
        labels = []
        
        # Emotion categories
        positive_emotions = [3]  # Happy
        negative_emotions = [0, 2, 5]  # Angry, Fear, Sad
        neutral_emotions = [4]  # Neutral
        
        for _ in range(num_samples):
            # 50% aligned, 50% mismatched
            is_aligned = np.random.random() > 0.5
            
            if is_aligned:
                # Aligned examples
                sentiment_type = np.random.choice(['positive', 'negative', 'neutral'])
                
                if sentiment_type == 'positive':
                    # Positive text + happy emotion
                    vision_probs = self._create_emotion_vector([3])  # Happy
                    text_probs = np.array([0.05, 0.10, 0.85])  # Positive
                elif sentiment_type == 'negative':
                    # Negative text + negative emotion
                    emotion_idx = np.random.choice(negative_emotions)
                    vision_probs = self._create_emotion_vector([emotion_idx])
                    text_probs = np.array([0.85, 0.10, 0.05])  # Negative
                else:
                    # Neutral text + neutral emotion
                    vision_probs = self._create_emotion_vector([4])  # Neutral
                    text_probs = np.array([0.15, 0.70, 0.15])  # Neutral
                
                label = 0  # Aligned
            else:
                # Mismatched examples
                mismatch_type = np.random.choice(['pos_text_neg_face', 'neg_text_pos_face'])
                
                if mismatch_type == 'pos_text_neg_face':
                    # Positive text but sad/worried face
                    emotion_idx = np.random.choice(negative_emotions)
                    vision_probs = self._create_emotion_vector([emotion_idx])
                    text_probs = np.array([0.05, 0.10, 0.85])  # Positive
                else:
                    # Negative text but happy face
                    vision_probs = self._create_emotion_vector([3])  # Happy
                    text_probs = np.array([0.85, 0.10, 0.05])  # Negative
                
                label = 1  # Mismatched
            
            vision_data.append(vision_probs)
            text_data.append(text_probs)
            labels.append(label)
        
        # Convert to tensors
        vision_tensor = torch.tensor(vision_data, dtype=torch.float32)
        text_tensor = torch.tensor(text_data, dtype=torch.float32)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        
        return vision_tensor, text_tensor, label_tensor
    
    def _create_emotion_vector(self, dominant_emotions: List[int]) -> np.ndarray:
        """
        Create emotion probability vector with dominant emotions
        
        Args:
            dominant_emotions: List of emotion indices to emphasize
        
        Returns:
            7-dimensional probability vector
        """
        probs = np.random.dirichlet(np.ones(7) * 0.5)  # Random distribution
        
        # Boost dominant emotions
        for idx in dominant_emotions:
            probs[idx] += 0.5
        
        # Normalize
        probs = probs / probs.sum()
        
        return probs
    
    def train(self, epochs=50, batch_size=32, num_samples=1000):
        """
        Train the fusion network
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            num_samples: Number of training samples to generate
        """
        print(f"\n{'='*60}")
        print("TRAINING FUSION NETWORK")
        print(f"{'='*60}\n")
        
        # Generate training data
        print(f"Generating {num_samples} training samples...")
        vision_data, text_data, labels = self.generate_training_data(num_samples)
        
        # Split into train/val
        split_idx = int(0.8 * len(vision_data))
        train_vision, val_vision = vision_data[:split_idx], vision_data[split_idx:]
        train_text, val_text = text_data[:split_idx], text_data[split_idx:]
        train_labels, val_labels = labels[:split_idx], labels[split_idx:]
        
        print(f"Training samples: {len(train_vision)}")
        print(f"Validation samples: {len(val_vision)}")
        print()
        
        # Training loop
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            
            # Shuffle training data
            indices = torch.randperm(len(train_vision))
            
            for i in range(0, len(train_vision), batch_size):
                batch_indices = indices[i:i+batch_size]
                
                batch_vision = train_vision[batch_indices]
                batch_text = train_text[batch_indices]
                batch_labels = train_labels[batch_indices]
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_vision, batch_text)
                loss = self.criterion(outputs, batch_labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == batch_labels).sum().item()
            
            # Validation phase
            self.model.eval()
            val_correct = 0
            
            with torch.no_grad():
                val_outputs = self.model(val_vision, val_text)
                _, val_predicted = torch.max(val_outputs, 1)
                val_correct = (val_predicted == val_labels).sum().item()
            
            # Calculate metrics
            train_acc = 100.0 * train_correct / len(train_vision)
            val_acc = 100.0 * val_correct / len(val_vision)
            avg_loss = train_loss / (len(train_vision) / batch_size)
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Train Acc: {train_acc:.2f}% | "
                      f"Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"{'='*60}\n")
    
    def save_model(self, path='models/fusion_trained.pth'):
        """Save trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"✓ Model saved to {path}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("FUSION NETWORK TRAINING")
    print("="*60 + "\n")
    
    # Initialize trainer
    trainer = FusionTrainer(learning_rate=0.001)
    
    # Train
    trainer.train(
        epochs=50,
        batch_size=32,
        num_samples=2000
    )
    
    # Save trained model
    trainer.save_model('models/fusion_trained.pth')
    
    print("\n✓ Training complete!")
    print("\nTo use the trained model:")
    print("1. The app will automatically load it")
    print("2. Restart the app: python app.py")
    print()

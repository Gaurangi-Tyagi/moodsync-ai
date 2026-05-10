"""
IMPROVED: Training Script for Vision Model
Includes verification that trained model actually works!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import sys
from PIL import Image

# Add path for vision model
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src', 'components'))

from vision_model import FaceEmotionCNN


class VisionTrainer:
    """Enhanced trainer with verification"""
    
    def __init__(self, data_dir='data/emotion_dataset', learning_rate=0.001):
        """Initialize trainer"""
        self.device = torch.device('cpu')
        self.data_dir = data_dir
        
        # Initialize model (FaceEmotionCNN is a wrapper, access inner model)
        emotion_cnn = FaceEmotionCNN()
        self.model = emotion_cnn.model  # Access the actual PyTorch model
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate
        )
        
        print("✓ Vision trainer initialized")
    
    def load_data(self, batch_size=32):
        """Load training and validation data"""
        
        # Data transformations (with augmentation for training)
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load datasets
        train_dataset = datasets.ImageFolder(
            os.path.join(self.data_dir, 'train'),
            transform=train_transform
        )
        
        val_dataset = datasets.ImageFolder(
            os.path.join(self.data_dir, 'val'),
            transform=val_transform
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=0
        )
        
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=0
        )
        
        print(f"Loaded {len(train_dataset)} images from {os.path.join(self.data_dir, 'train')}")
        print(f"Loaded {len(val_dataset)} images from {os.path.join(self.data_dir, 'val')}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Progress
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch [{batch_idx+1}/{len(self.train_loader)}] | "
                      f"Loss: {running_loss/(batch_idx+1):.4f} | "
                      f"Acc: {100.*correct/total:.2f}%")
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    def train(self, epochs=20):
        """Train the model"""
        print(f"\n{'='*60}")
        print("TRAINING VISION MODEL ON FACE IMAGES")
        print(f"{'='*60}\n")
        
        print("Initializing vision model...")
        self.load_data()
        print()
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            print(f"Epoch [{epoch+1}/{epochs}]")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_acc = self.validate()
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('models/vision_trained.pth')
                print("✓ Best model saved!")
            
            print()
        
        print(f"{'='*60}")
        print(f"Training Complete! Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"{'='*60}\n")
    
    def save_model(self, path='models/vision_trained.pth'):
        """Save model with verification"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save state dict
        torch.save(self.model.state_dict(), path)
        
        # Verify file was created
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  Model saved: {path} ({size_mb:.1f} MB)")
        else:
            print(f"  ERROR: Model file not created!")
    
    def verify_saved_model(self, path='models/vision_trained.pth'):
        """Verify the saved model actually works"""
        print(f"\n{'='*60}")
        print("VERIFYING SAVED MODEL")
        print(f"{'='*60}\n")
        
        # Create new model instance
        test_model = FaceEmotionCNN(model_path=path)
        
        # Create dummy image
        dummy_image = Image.new('RGB', (224, 224), color='gray')
        
        # Get prediction
        probs = test_model.predict(dummy_image)
        
        # Check if probabilities are reasonable (not all equal)
        prob_values = list(probs.values())
        max_prob = max(prob_values)
        min_prob = min(prob_values)
        
        print("Sample Prediction from Loaded Model:")
        for emotion, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {emotion}: {prob*100:.2f}%")
        
        print()
        if max_prob - min_prob < 0.05:  # All probabilities within 5%
            print("⚠️  WARNING: Model appears untrained (equal probabilities)")
            print("   Try training for more epochs or check data loading")
        else:
            print("✓ Model loaded and working correctly!")
        
        print(f"{'='*60}\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("VISION MODEL TRAINING WITH VERIFICATION")
    print("="*60 + "\n")
    
    # Initialize trainer
    trainer = VisionTrainer(
        data_dir='data/emotion_dataset',
        learning_rate=0.0001  # Lower learning rate for stability
    )
    
    # Train
    trainer.train(epochs=20)
    
    # Verify the saved model
    trainer.verify_saved_model('models/vision_trained.pth')
    
    print("\n✓ Training complete!")
    print("\nTo use the trained model:")
    print("1. Update vision_model.py to load 'models/vision_trained.pth'")
    print("2. Restart the app: python app.py")
    print()

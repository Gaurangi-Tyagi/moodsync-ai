"""
Vision Model Training Script
Train CNN on facial expression images for better emotion recognition
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import sys
import numpy as np
from tqdm import tqdm

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src', 'components'))

from vision_model import FaceEmotionCNN


class EmotionDataset(Dataset):
    """
    Dataset for facial expression images
    Expects folder structure:
    data/emotion_dataset/
    ├── train/
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── neutral/
    │   ├── sad/
    │   └── surprise/
    └── val/
        ├── angry/
        ├── ...
    """
    
    EMOTION_LABELS = {
        'angry': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3,
        'neutral': 4,
        'sad': 5,
        'surprise': 6
    }
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Path to train/ or val/ folder
            transform: Image transformations
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load all images
        for emotion_name, label in self.EMOTION_LABELS.items():
            emotion_dir = os.path.join(root_dir, emotion_name)
            
            if not os.path.exists(emotion_dir):
                continue
            
            for img_name in os.listdir(emotion_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(emotion_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(label)
        
        print(f"Loaded {len(self.images)} images from {root_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class VisionTrainer:
    """
    Trainer for vision model on real face images
    """
    
    def __init__(self, data_dir='data/emotion_dataset', learning_rate=0.001):
        """
        Args:
            data_dir: Path to dataset folder
            learning_rate: Learning rate
        """
        self.device = torch.device('cpu')
        self.data_dir = data_dir
        
        # Initialize model
        print("Initializing vision model...")
        self.model = FaceEmotionCNN()
        self.model.model.train()
        self.model.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.model.parameters()),
            lr=learning_rate
        )
        
        # Data transforms (same as model uses)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print("✓ Vision trainer initialized")
    
    def prepare_data(self, batch_size=16):
        """Load datasets"""
        train_dir = os.path.join(self.data_dir, 'train')
        val_dir = os.path.join(self.data_dir, 'val')
        
        # Check if directories exist
        if not os.path.exists(train_dir):
            raise FileNotFoundError(
                f"Training data not found at {train_dir}\n"
                "Please organize images in the folder structure (see instructions)"
            )
        
        # Create datasets
        train_dataset = EmotionDataset(train_dir, transform=self.transform)
        val_dataset = EmotionDataset(val_dir, transform=self.transform)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        return len(train_dataset), len(val_dataset)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(self.train_loader, desc="Training"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate model"""
        self.model.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model.model(images)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100.0 * correct / total
        return val_acc
    
    def train(self, epochs=20, batch_size=16):
        """
        Full training loop
        
        Args:
            epochs: Number of epochs
            batch_size: Batch size
        """
        print(f"\n{'='*60}")
        print("TRAINING VISION MODEL ON FACE IMAGES")
        print(f"{'='*60}\n")
        
        # Prepare data
        train_size, val_size = self.prepare_data(batch_size)
        print(f"Training samples: {train_size}")
        print(f"Validation samples: {val_size}")
        print()
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            print(f"\nEpoch [{epoch+1}/{epochs}]")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_acc = self.validate()
            
            # Print results
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('models/vision_trained.pth')
                print("✓ Best model saved!")
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"{'='*60}\n")
    
    def save_model(self, path='models/vision_trained.pth'):
        """Save trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.model.state_dict(), path)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("VISION MODEL TRAINING ON FACE IMAGES")
    print("="*60 + "\n")
    
    print("Expected folder structure:")
    print("data/emotion_dataset/")
    print("  ├── train/")
    print("  │   ├── angry/")
    print("  │   ├── happy/")
    print("  │   ├── sad/")
    print("  │   ├── ... (other emotions)")
    print("  └── val/")
    print("      ├── angry/")
    print("      └── ...")
    print()
    
    # Check if dataset exists
    if not os.path.exists('data/emotion_dataset/train'):
        print("❌ Dataset not found!")
        print("\nOptions:")
        print("1. Download FER-2013 dataset")
        print("2. Use kaggle dataset: fer2013 or CK+")
        print("3. Or create small custom dataset (20-30 images per emotion)")
        print()
        sys.exit(1)
    
    # Initialize trainer
    trainer = VisionTrainer(
        data_dir='data/emotion_dataset',
        learning_rate=0.0001  # Lower LR for fine-tuning
    )
    
    # Train
    trainer.train(
        epochs=20,
        batch_size=8  # Small batch for limited resources
    )
    
    print("\n✓ Training complete!")
    print("\nTo use the trained model:")
    print("1. Update vision_model.py to load 'models/vision_trained.pth'")
    print("2. Restart the app: python app.py")
    print()

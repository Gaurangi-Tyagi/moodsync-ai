"""
Webcam Dataset Creator
Capture facial expressions to create custom emotion dataset
"""

import cv2
import os
from datetime import datetime
import time

class DatasetCapture:
    """Capture facial expressions for emotion dataset"""
    
    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    def __init__(self, base_dir='data/emotion_dataset'):
        """Initialize capture system"""
        self.base_dir = base_dir
        self.camera = None
        
        # Load face detector
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        print("✓ Dataset capture initialized")
    
    def start_camera(self):
        """Start webcam"""
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            print("❌ Could not open webcam")
            return False
        print("✓ Webcam started")
        return True
    
    def stop_camera(self):
        """Release webcam"""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
    
    def detect_face(self, frame):
        """Detect and crop face from frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
        
        if len(faces) == 0:
            return None, None
        
        # Get largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # Crop with padding
        padding = 30
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(frame.shape[1], x + w + padding)
        y_end = min(frame.shape[0], y + h + padding)
        
        face_crop = frame[y_start:y_end, x_start:x_end]
        
        return face_crop, (x, y, w, h)
    
    def capture_emotion_set(self, emotion, target_count=15, split='train'):
        """
        Capture images for one emotion
        
        Args:
            emotion: Emotion name (angry, happy, etc.)
            target_count: Number of images to capture
            split: 'train' or 'val'
        """
        # Create directory
        save_dir = os.path.join(self.base_dir, split, emotion)
        os.makedirs(save_dir, exist_ok=True)
        
        # Count existing images
        existing = len([f for f in os.listdir(save_dir) if f.endswith('.jpg')])
        
        print(f"\n{'='*60}")
        print(f"📸 CAPTURING: {emotion.upper()} ({split})")
        print(f"{'='*60}")
        print(f"Target: {target_count} images")
        print(f"Already captured: {existing}")
        print(f"\n💡 INSTRUCTIONS:")
        print(f"   1. Make a {emotion} face")
        print(f"   2. Press SPACE to capture")
        print(f"   3. Press Q to finish this emotion")
        print(f"   4. Vary your expression slightly each time")
        print(f"\n⏰ Starting in 3 seconds...")
        time.sleep(3)
        
        captured = 0
        
        while captured < target_count:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            # Detect face
            face_crop, bbox = self.detect_face(frame)
            
            # Draw on display frame
            display_frame = frame.copy()
            
            if bbox is not None:
                x, y, w, h = bbox
                # Green box for detected face
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(display_frame, f"{emotion.upper()}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # Warning if no face detected
                cv2.putText(display_frame, "NO FACE DETECTED", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Progress info
            progress_text = f"Captured: {captured}/{target_count} | Press SPACE to capture | Q to quit"
            cv2.putText(display_frame, progress_text, (10, display_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow(f'Capture {emotion}', display_frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space to capture
                if face_crop is not None:
                    # Save image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"{emotion}_{timestamp}.jpg"
                    filepath = os.path.join(save_dir, filename)
                    
                    cv2.imwrite(filepath, face_crop)
                    captured += 1
                    print(f"✓ Captured {captured}/{target_count}: {filename}")
                    
                    # Brief pause
                    time.sleep(0.5)
                else:
                    print("⚠ No face detected - position your face in frame")
            
            elif key == ord('q'):  # Q to quit this emotion
                print(f"Finished {emotion} early ({captured} images)")
                break
        
        cv2.destroyAllWindows()
        print(f"\n✓ {emotion} complete! Captured {captured} images")
        print(f"Saved to: {save_dir}")
    
    def capture_full_dataset(self, images_per_emotion=15):
        """
        Capture complete dataset for all emotions
        
        Args:
            images_per_emotion: Number of images per emotion
        """
        print("\n" + "="*60)
        print("CUSTOM DATASET CREATION")
        print("="*60)
        print(f"\nYou will capture {images_per_emotion} images for each emotion")
        print(f"Total images: {images_per_emotion * len(self.EMOTIONS) * 2} (train + val)")
        print(f"\nEmotions: {', '.join(self.EMOTIONS)}")
        print(f"\n💡 Tips:")
        print(f"   - Good lighting is important")
        print(f"   - Keep face centered")
        print(f"   - Vary expression slightly each capture")
        print(f"   - Exaggerate emotions for better training")
        
        input("\nPress ENTER to start...")
        
        if not self.start_camera():
            return
        
        try:
            # Capture training set
            print(f"\n{'='*60}")
            print("PHASE 1: TRAINING SET")
            print(f"{'='*60}")
            
            for emotion in self.EMOTIONS:
                self.capture_emotion_set(emotion, images_per_emotion, 'train')
                print(f"\n⏸ Short break... Next: {emotion} (val set)")
                time.sleep(2)
            
            # Capture validation set (smaller)
            print(f"\n{'='*60}")
            print("PHASE 2: VALIDATION SET (Fewer images)")
            print(f"{'='*60}")
            
            val_count = max(3, images_per_emotion // 3)
            
            for emotion in self.EMOTIONS:
                self.capture_emotion_set(emotion, val_count, 'val')
                print(f"\n⏸ Short break...")
                time.sleep(2)
            
            print(f"\n{'='*60}")
            print("✅ DATASET CREATION COMPLETE!")
            print(f"{'='*60}")
            print(f"\nDataset location: {self.base_dir}")
            print(f"Next step: python train_vision.py")
            
        finally:
            self.stop_camera()


def main():
    """Main function"""
    print("\n" + "="*60)
    print("📸 WEBCAM DATASET CREATOR")
    print("="*60 + "\n")
    
    print("This script will help you create a custom emotion dataset")
    print("using your webcam.\n")
    
    # Get user preference
    print("How many images per emotion? (Recommended: 10-15)")
    print("  - Quick test: 5")
    print("  - Good training: 10-15")
    print("  - Best results: 20+")
    
    try:
        count = int(input("\nImages per emotion [default: 12]: ") or "12")
    except:
        count = 12
    
    # Initialize and run
    capturer = DatasetCapture()
    capturer.capture_full_dataset(images_per_emotion=count)


if __name__ == "__main__":
    main()

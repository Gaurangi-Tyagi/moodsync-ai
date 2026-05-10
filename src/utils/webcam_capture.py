"""
Webcam Capture Utility
Real-time face capture from webcam for mood detection
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class WebcamCapture:
    """
    Handle webcam input for real-time emotion detection
    """
    
    def __init__(self, camera_id=0):
        """
        Initialize webcam
        
        Args:
            camera_id: Camera device ID (0 for default)
        """
        self.camera_id = camera_id
        self.cap = None
        self.face_cascade = None
        self._load_face_detector()
    
    def _load_face_detector(self):
        """Load Haar Cascade face detector"""
        try:
            # Load pre-trained face detector
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            print("✓ Face detector loaded")
        except Exception as e:
            print(f"⚠ Could not load face detector: {e}")
    
    def start_camera(self) -> bool:
        """
        Start webcam capture
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                print("❌ Could not open webcam")
                return False
            print("✓ Webcam started")
            return True
        except Exception as e:
            print(f"❌ Error starting webcam: {e}")
            return False
    
    def stop_camera(self):
        """Release webcam"""
        if self.cap is not None:
            self.cap.release()
            cv2.destroyAllWindows()
            print("✓ Webcam stopped")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture single frame from webcam
        
        Returns:
            Frame as numpy array or None if failed
        """
        if self.cap is None or not self.cap.isOpened():
            print("❌ Webcam not started")
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            return None
        
        return frame
    
    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, Tuple]]:
        """
        Detect face in frame and return cropped face
        
        Args:
            frame: Input frame
        
        Returns:
            (cropped_face, (x, y, w, h)) or None if no face detected
        """
        if self.face_cascade is None:
            return None
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )
        
        if len(faces) == 0:
            return None
        
        # Get largest face (closest to camera)
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        # Crop face with some padding
        padding = 20
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(frame.shape[1], x + w + padding)
        y_end = min(frame.shape[0], y + h + padding)
        
        face_crop = frame[y_start:y_end, x_start:x_end]
        
        return face_crop, (x, y, w, h)
    
    def capture_face(self) -> Optional[Image.Image]:
        """
        Capture frame and extract face
        
        Returns:
            PIL Image of face or None
        """
        frame = self.capture_frame()
        if frame is None:
            return None
        
        result = self.detect_face(frame)
        if result is None:
            print("⚠ No face detected in frame")
            return None
        
        face_crop, _ = result
        
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(face_rgb)
        
        return pil_image
    
    def draw_face_box(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw rectangle around detected face
        
        Args:
            frame: Input frame
        
        Returns:
            Frame with face box drawn
        """
        result = self.detect_face(frame)
        if result is None:
            return frame
        
        _, (x, y, w, h) = result
        
        # Draw rectangle
        frame_copy = frame.copy()
        cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame_copy, "Face Detected", (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame_copy
    
    def show_live_preview(self, duration_seconds=10):
        """
        Show live webcam preview with face detection
        
        Args:
            duration_seconds: How long to show preview
        """
        if not self.start_camera():
            return
        
        print(f"Showing live preview for {duration_seconds} seconds...")
        print("Press 'q' to quit early")
        
        import time
        start_time = time.time()
        
        while (time.time() - start_time) < duration_seconds:
            frame = self.capture_frame()
            if frame is None:
                break
            
            # Draw face box
            display_frame = self.draw_face_box(frame)
            
            # Show frame
            cv2.imshow('Webcam Preview - Press Q to quit', display_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.stop_camera()


# Test function
if __name__ == "__main__":
    print("Testing WebcamCapture...")
    print("-" * 60)
    
    # Initialize
    webcam = WebcamCapture()
    
    # Start camera
    if webcam.start_camera():
        print("\n📸 Capturing face from webcam...")
        
        # Try to capture face
        for attempt in range(5):
            print(f"Attempt {attempt + 1}/5...")
            face_image = webcam.capture_face()
            
            if face_image:
                print(f"✓ Face captured! Size: {face_image.size}")
                
                # Save for testing
                face_image.save("test_webcam_capture.jpg")
                print("✓ Saved as test_webcam_capture.jpg")
                break
            
            import time
            time.sleep(1)
        
        # Show live preview
        print("\n🎥 Starting live preview...")
        print("Position your face in front of camera")
        webcam.show_live_preview(duration_seconds=5)
        
    else:
        print("❌ Could not start webcam")
        print("\nTroubleshooting:")
        print("1. Check if webcam is connected")
        print("2. Check if another app is using webcam")
        print("3. Try different camera_id (0, 1, 2...)")
    
    print("\n✓ Test complete!")

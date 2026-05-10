"""
Main Pipeline: MoodSyncAI
Integrates all components: Vision, Text, Fusion, and Generative
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class MoodSyncPipeline:
    """
    Complete multi-modal emotion analysis pipeline
    """
    
    def __init__(self, vision_model_path=None, fusion_model_path=None):
        """
        Initialize all components
        
        Args:
            vision_model_path: Path to vision model weights
            fusion_model_path: Path to fusion model weights
        """
        print("🚀 Initializing MoodSyncAI Pipeline...")
        print("-" * 60)
        
        # Import components (lazy loading)
        print("Loading vision model...")
        from vision_model import FaceEmotionCNN
        self.vision_model = FaceEmotionCNN(vision_model_path)
        
        print("Loading text model...")
        from text_model import TextSentimentAnalyzer
        self.text_model = TextSentimentAnalyzer()
        
        print("Loading fusion layer...")
        from fusion_layer import MultiModalFusion
        self.fusion_model = MultiModalFusion(fusion_model_path)
        
        print("Loading generative explainer...")
        from generative_model import EmotionExplainer
        self.explainer = EmotionExplainer()
        
        print("-" * 60)
        print("✓ Pipeline ready!")
        print()
    
    def analyze(self, image, text: str, verbose=True) -> Dict:
        """
        Complete multi-modal analysis
        
        Args:
            image: Face image (path, PIL Image, or numpy array)
            text: Text input (what the person said)
            verbose: Print detailed output
        
        Returns:
            Complete analysis dictionary
        """
        if verbose:
            print("=" * 60)
            print("ANALYZING INPUT")
            print("=" * 60)
            print(f"Text: '{text}'")
            print()
        
        # Step 1: Vision analysis
        if verbose:
            print("🖼️  Analyzing facial emotion...")
        vision_results = self.vision_model.predict(image)
        top_emotion, emotion_conf = self.vision_model.get_top_emotion(image)
        
        if verbose:
            print(f"   Top emotion: {top_emotion} ({emotion_conf*100:.1f}%)")
            print()
        
        # Step 2: Text analysis
        if verbose:
            print("📝 Analyzing text sentiment...")
        text_results = self.text_model.predict(text)
        top_sentiment, sentiment_conf = self.text_model.get_top_sentiment(text)
        
        if verbose:
            print(f"   Top sentiment: {top_sentiment} ({sentiment_conf*100:.1f}%)")
            print()
        
        # Step 3: Fusion
        if verbose:
            print("🔄 Fusing modalities...")
        fusion_results = self.fusion_model.fuse_predictions(vision_results, text_results)
        
        if verbose:
            status = fusion_results['alignment_status']
            mismatch = fusion_results['mismatch_confidence']
            print(f"   Status: {status}")
            print(f"   Mismatch confidence: {mismatch*100:.1f}%")
            print()
        
        # Step 4: Generate explanation
        if verbose:
            print("🤖 Generating explanation...")
            print()
        
        explanation = self.explainer.generate_explanation(
            vision_results, text_results, fusion_results
        )
        
        # Compile results
        results = {
            'vision': {
                'probabilities': vision_results,
                'top_emotion': top_emotion,
                'confidence': emotion_conf
            },
            'text': {
                'probabilities': text_results,
                'top_sentiment': top_sentiment,
                'confidence': sentiment_conf
            },
            'fusion': fusion_results,
            'explanation': explanation
        }
        
        if verbose:
            print("=" * 60)
            print("EXPLANATION")
            print("=" * 60)
            print(explanation)
            print("=" * 60)
        
        return results
    
    def generate_report(self, image, text: str) -> str:
        """
        Generate detailed analysis report
        
        Args:
            image: Face image
            text: Text input
        
        Returns:
            Formatted report string
        """
        results = self.analyze(image, text, verbose=False)
        
        report = self.explainer.generate_detailed_report(
            results['vision']['probabilities'],
            results['text']['probabilities'],
            results['fusion']
        )
        
        return report


# Example usage
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MOODSYNCAI PIPELINE TEST")
    print("=" * 60)
    print()
    
    # Initialize pipeline
    pipeline = MoodSyncPipeline()
    
    # Test with dummy data
    print("Creating test data...")
    from PIL import Image
    dummy_image = Image.new('RGB', (224, 224), color='gray')
    test_text = "No, I think the project is going really well."
    
    print("\n" + "=" * 60)
    print("Running analysis...")
    print("=" * 60)
    
    # Run analysis
    results = pipeline.analyze(dummy_image, test_text)
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nPipeline is working! Ready for real data.")

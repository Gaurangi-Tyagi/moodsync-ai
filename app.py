"""
Gradio UI: MoodSyncAI Web Interface
Supports both image upload and webcam capture
SIMPLIFIED: No charts, clean interface
"""

import gradio as gr
import sys
import os
from PIL import Image
import numpy as np

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src', 'components'))
sys.path.insert(0, os.path.join(current_dir, 'src', 'fusion'))
sys.path.insert(0, os.path.join(current_dir, 'src', 'utils'))

# Import components
from vision_model import FaceEmotionCNN
from text_model import TextSentimentAnalyzer
from fusion_layer import MultiModalFusion
from generative_model import EmotionExplainer


class MoodSyncApp:
    """
    Web interface for MoodSyncAI
    """
    
    def __init__(self, fusion_model_path='models/fusion_trained.pth'):
        """Initialize all models"""
        print("🚀 Loading MoodSyncAI models...")
        
        self.vision_model = FaceEmotionCNN()
        self.text_model = TextSentimentAnalyzer()
        
        # Try to load trained fusion model
        if os.path.exists(fusion_model_path):
            print(f"Loading trained fusion model from {fusion_model_path}...")
            self.fusion_model = MultiModalFusion(fusion_model_path)
        else:
            print("Using untrained fusion model")
            self.fusion_model = MultiModalFusion()
        
        self.explainer = EmotionExplainer()
        
        print("✓ All models loaded!")
    
    def analyze_mood(self, image, text):
        """
        Main analysis function for Gradio
        
        Args:
            image: PIL Image or numpy array from Gradio
            text: Text input
        
        Returns:
            Multiple outputs for Gradio interface
        """
        # Validate inputs
        if image is None:
            return (
                "❌ Please provide an image (upload or webcam)",
                "", "", ""
            )
        
        if not text or text.strip() == "":
            return (
                "❌ Please enter what the person said",
                "", "", ""
            )
        
        try:
            # Convert numpy array to PIL if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Step 1: Vision analysis
            vision_results = self.vision_model.predict(image)
            top_emotion, emotion_conf = self.vision_model.get_top_emotion(image)
            
            # Step 2: Text analysis
            text_results = self.text_model.predict(text)
            top_sentiment, sentiment_conf = self.text_model.get_top_sentiment(text)
            
            # Step 3: Fusion
            fusion_results = self.fusion_model.fuse_predictions(
                vision_results, text_results
            )
            
            # Step 4: Generate explanation
            explanation = self.explainer.generate_explanation(
                vision_results, text_results, fusion_results
            )
            
            # Format outputs
            status = fusion_results['alignment_status']
            mismatch_conf = fusion_results['mismatch_confidence']
            
            # Status badge with emoji
            if status == "MISMATCH":
                status_badge = f"### ⚠️ MISMATCH DETECTED\n**Confidence:** {mismatch_conf*100:.1f}%"
            elif status == "ALIGNED":
                status_badge = f"### ✅ ALIGNED\n**Confidence:** {(1-mismatch_conf)*100:.1f}%"
            else:
                status_badge = f"### ❓ UNCLEAR\n**Mismatch Score:** {mismatch_conf*100:.1f}%"
            
            # Emotion details with all scores
            emotion_text = f"**Top Emotion:** {top_emotion} ({emotion_conf*100:.1f}%)\n\n"
            emotion_text += "**All Emotions:**\n"
            for emotion, prob in sorted(vision_results.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(prob * 20)
                emotion_text += f"- {emotion}: {bar} {prob*100:.1f}%\n"
            
            # Sentiment details with all scores
            sentiment_text = f"**Top Sentiment:** {top_sentiment} ({sentiment_conf*100:.1f}%)\n\n"
            sentiment_text += "**All Sentiments:**\n"
            for sent, prob in sorted(text_results.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(prob * 20)
                sentiment_text += f"- {sent}: {bar} {prob*100:.1f}%\n"
            
            return (
                status_badge,
                emotion_text,
                sentiment_text,
                explanation
            )
        
        except Exception as e:
            error_msg = f"❌ Error during analysis: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return (error_msg, "", "", "")
    
    def create_interface(self):
        """Create Gradio interface"""
        
        # Custom CSS for styling
        custom_css = """
        .status-box {
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .result-box {
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            margin: 10px 0;
        }
        """
        
        with gr.Blocks(css=custom_css, title="MoodSyncAI", theme=gr.themes.Soft()) as interface:
            
            # Header
            gr.Markdown("""
            # 🎭 MoodSyncAI: Multi-Modal Sentiment & Emotion Analyzer
            
            Upload a photo or use your webcam, then enter what the person said. 
            The system will analyze emotional alignment between facial expressions and words.
            """)
            
            with gr.Row():
                # Left column: Inputs
                with gr.Column(scale=1):
                    gr.Markdown("### 📸 Step 1: Provide Image")
                    
                    # Image input with webcam support
                    image_input = gr.Image(
                        label="Face Image",
                        type="pil",
                        sources=["upload", "webcam"],
                        height=300
                    )
                    
                    gr.Markdown("### 💬 Step 2: Enter Text")
                    text_input = gr.Textbox(
                        label="What did the person say?",
                        placeholder="Example: No, I think the project is going really well.",
                        lines=3
                    )
                    
                    analyze_btn = gr.Button("🔍 Analyze Mood", variant="primary", size="lg")
                    
                    gr.Markdown("""
                    **Example Texts:**
                    - "I'm fine, everything is great!"
                    - "This is terrible and frustrating."
                    - "No, I think the project is going really well."
                    - "I'm so happy about this!"
                    """)
                
                # Right column: Results
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Analysis Results")
                    
                    # Status
                    status_output = gr.Markdown(
                        value="*Awaiting analysis...*",
                        elem_classes="status-box"
                    )
                    
                    # Emotion results
                    gr.Markdown("### 😊 Visual Emotion Analysis")
                    emotion_output = gr.Markdown(
                        value="*No results yet*",
                        elem_classes="result-box"
                    )
                    
                    # Sentiment results
                    gr.Markdown("### 📝 Textual Sentiment Analysis")
                    sentiment_output = gr.Markdown(
                        value="*No results yet*",
                        elem_classes="result-box"
                    )
                    
                    # Explanation
                    gr.Markdown("### 🤖 AI Explanation")
                    explanation_output = gr.Textbox(
                        label="Detailed Analysis",
                        lines=6,
                        interactive=False,
                        show_label=False
                    )
            
            # Examples
            gr.Markdown("### 📝 Quick Examples")
            gr.Examples(
                examples=[
                    [None, "I love this! Everything is going perfectly!"],
                    [None, "This is absolutely terrible and disappointing."],
                    [None, "No, I think the project is going really well."],
                    [None, "I'm fine, just a bit tired today."],
                ],
                inputs=[image_input, text_input],
                label="Click to try (add your image)"
            )
            
            # Connect button to analysis
            analyze_btn.click(
                fn=self.analyze_mood,
                inputs=[image_input, text_input],
                outputs=[
                    status_output,
                    emotion_output,
                    sentiment_output,
                    explanation_output
                ]
            )
            
            # Footer
            gr.Markdown("""
            ---
            **MoodSyncAI** | Multi-Modal Emotion Analysis | Data Analytics-3 Final Project  
            🎓 **Models:** Vision CNN (FER-2013 trained) + Custom Fusion Network (95.75% acc) + DistilBERT
            """)
        
        return interface


def launch_app():
    """Launch the Gradio app"""
    print("=" * 60)
    print("MOODSYNCAI WEB INTERFACE")
    print("=" * 60)
    print()
    
    # Initialize app
    app = MoodSyncApp()
    
    # Create interface
    interface = app.create_interface()
    
    # Launch
    print("\n🚀 Launching web interface...")
    print("=" * 60)
    
    interface.launch(
        share=False,  # Set to True to create public link
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )


if __name__ == "__main__":
    launch_app()

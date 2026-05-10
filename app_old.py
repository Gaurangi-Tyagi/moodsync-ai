"""
Gradio UI: MoodSyncAI Web Interface
Supports both image upload and webcam capture
"""

import gradio as gr
import sys
import os
from PIL import Image
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import components
from vision_model import FaceEmotionCNN
from text_model import TextSentimentAnalyzer
from fusion_layer import MultiModalFusion
from generative_model import EmotionExplainer


class MoodSyncApp:
    """
    Web interface for MoodSyncAI
    """
    
    def __init__(self):
        """Initialize all models"""
        print("🚀 Loading MoodSyncAI models...")
        
        self.vision_model = FaceEmotionCNN()
        self.text_model = TextSentimentAnalyzer()
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
                None, None, None, ""
            )
        
        if not text or text.strip() == "":
            return (
                "❌ Please enter what the person said",
                None, None, None, ""
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
            
            # Format outputs for Gradio
            status = fusion_results['alignment_status']
            mismatch_conf = fusion_results['mismatch_confidence']
            
            # Status badge
            if status == "MISMATCH":
                status_badge = f"⚠️ **MISMATCH DETECTED** ({mismatch_conf*100:.1f}% confidence)"
            elif status == "ALIGNED":
                status_badge = f"✓ **ALIGNED** ({(1-mismatch_conf)*100:.1f}% confidence)"
            else:
                status_badge = f"❓ **UNCLEAR** ({mismatch_conf*100:.1f}% mismatch)"
            
            # Vision chart data
            vision_chart = self._create_bar_chart(vision_results, "Emotion")
            
            # Text chart data
            text_chart = self._create_bar_chart(text_results, "Sentiment")
            
            return (
                status_badge,
                vision_chart,
                text_chart,
                f"**{top_emotion}** ({emotion_conf*100:.1f}%)",
                f"**{top_sentiment}** ({sentiment_conf*100:.1f}%)",
                explanation
            )
        
        except Exception as e:
            error_msg = f"❌ Error during analysis: {str(e)}"
            print(error_msg)
            return (error_msg, None, None, None, None, "")
    
    def _create_bar_chart(self, results_dict, label_type):
        """
        Create bar chart data for Gradio
        
        Returns:
            Dictionary for Gradio BarPlot
        """
        # Sort by value
        sorted_items = sorted(results_dict.items(), 
                            key=lambda x: x[1], reverse=True)
        
        labels = [item[0] for item in sorted_items]
        values = [item[1] * 100 for item in sorted_items]  # Convert to percentage
        
        return {
            "labels": labels,
            "values": values
        }
    
    def create_interface(self):
        """Create Gradio interface"""
        
        # Custom CSS for styling
        custom_css = """
        .status-box {
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            font-size: 18px;
            text-align: center;
        }
        .explanation-box {
            padding: 15px;
            background-color: #f5f5f5;
            border-radius: 8px;
            margin-top: 10px;
        }
        """
        
        with gr.Blocks(css=custom_css, title="MoodSyncAI") as interface:
            
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
                        sources=["upload", "webcam"],  # Both upload and webcam
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
                    **Examples:**
                    - "I'm fine, everything is great!"
                    - "This is terrible and frustrating."
                    - "No, I think the project is going really well."
                    """)
                
                # Right column: Results
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Results")
                    
                    # Status badge
                    status_output = gr.Markdown(
                        label="Alignment Status",
                        value="*Awaiting analysis...*"
                    )
                    
                    # Top predictions
                    with gr.Row():
                        top_emotion = gr.Markdown("**Emotion:** -")
                        top_sentiment = gr.Markdown("**Sentiment:** -")
                    
                    # Charts
                    with gr.Row():
                        vision_chart = gr.BarPlot(
                            x="labels",
                            y="values",
                            title="Visual Emotion Confidence",
                            height=250,
                            y_title="Confidence (%)"
                        )
                        
                        text_chart = gr.BarPlot(
                            x="labels",
                            y="values",
                            title="Textual Sentiment Confidence",
                            height=250,
                            y_title="Confidence (%)"
                        )
                    
                    # Explanation
                    gr.Markdown("### 🤖 AI Explanation")
                    explanation_output = gr.Textbox(
                        label="Analysis Summary",
                        lines=8,
                        interactive=False
                    )
            
            # Examples
            gr.Markdown("### 📝 Try These Examples")
            gr.Examples(
                examples=[
                    [None, "I love this! Everything is going perfectly!"],
                    [None, "This is absolutely terrible and disappointing."],
                    [None, "No, I think the project is going really well."],
                    [None, "I'm fine, just a bit tired today."],
                ],
                inputs=[image_input, text_input],
                label="Sample Texts (Add your own image)"
            )
            
            # Connect button to analysis function
            analyze_btn.click(
                fn=self.analyze_mood,
                inputs=[image_input, text_input],
                outputs=[
                    status_output,
                    vision_chart,
                    text_chart,
                    top_emotion,
                    top_sentiment,
                    explanation_output
                ]
            )
            
            # Footer
            gr.Markdown("""
            ---
            **MoodSyncAI** | Multi-Modal Emotion Analysis | Data Analytics-3 Project
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

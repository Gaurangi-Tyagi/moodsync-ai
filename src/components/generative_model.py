"""
Generative Component: Natural Language Explanations
Generates human-readable summaries of emotional analysis
"""

import random
from typing import Dict


class EmotionExplainer:
    """
    Generates natural language explanations of multi-modal analysis
    Combines vision + text + fusion results into coherent summaries
    """
    
    def __init__(self):
        """Initialize explanation templates"""
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict:
        """Load explanation templates for different scenarios"""
        return {
            'aligned_positive': [
                "The speaker appears genuinely {emotion} and their words convey {sentiment}. "
                "There is strong alignment between their facial expression and verbal communication.",
                
                "Both visual and textual cues indicate a {emotion} and {sentiment} state. "
                "The person's emotional expression matches their words authentically.",
                
                "The individual shows {emotion} facial cues that align well with their {sentiment} message. "
                "This suggests genuine emotional expression."
            ],
            
            'aligned_negative': [
                "The speaker displays {emotion} expressions that match their {sentiment} tone. "
                "Their emotional state appears consistent across both modalities.",
                
                "Visual cues of {emotion} align with {sentiment} verbal content. "
                "This indicates a coherent emotional state.",
                
                "The person's {emotion} facial expression corresponds to their {sentiment} language, "
                "suggesting they are genuinely experiencing this emotional state."
            ],
            
            'mismatch_positive_text': [
                "⚠️ Despite expressing {sentiment} sentiment verbally, the speaker's facial cues "
                "indicate {emotion}. This incongruence is worth noting in the context of the conversation.",
                
                "⚠️ There is a noticeable disconnect: while the words convey {sentiment}, "
                "the facial expression shows {emotion}. The person may be masking their true feelings.",
                
                "⚠️ Interesting mismatch detected. The verbal message is {sentiment}, but "
                "facial indicators suggest {emotion}. Consider the broader context of this interaction."
            ],
            
            'mismatch_negative_text': [
                "⚠️ The speaker's words are {sentiment}, yet their face displays {emotion}. "
                "This could indicate discomfort, sarcasm, or emotional masking.",
                
                "⚠️ Facial expressions ({emotion}) contradict the {sentiment} verbal content. "
                "The person may not be expressing their true emotional state through words.",
                
                "⚠️ Notable mismatch: {sentiment} language paired with {emotion} facial cues. "
                "This warrants careful interpretation in context."
            ],
            
            'neutral_aligned': [
                "The speaker maintains a {emotion} demeanor with {sentiment} communication. "
                "Their emotional state appears balanced and measured.",
                
                "Both verbal and visual signals indicate a {sentiment}, {emotion} state. "
                "The person seems emotionally composed.",
                
                "The individual presents {emotion} expressions matching their {sentiment} tone, "
                "suggesting a calm and controlled emotional state."
            ],
            
            'unclear': [
                "The emotional signals are somewhat ambiguous. The facial expression suggests {emotion} "
                "with {confidence_vision:.0f}% confidence, while the text indicates {sentiment} "
                "with {confidence_text:.0f}% confidence.",
                
                "There are mixed signals in this analysis. Visual cues lean toward {emotion}, "
                "and verbal content suggests {sentiment}, but neither is strongly conclusive.",
                
                "The emotional state is not entirely clear. Consider gathering more context "
                "as both visual ({emotion}) and textual ({sentiment}) indicators show moderate confidence."
            ]
        }
    
    def generate_explanation(self, vision_dict: Dict, text_dict: Dict, 
                           fusion_result: Dict) -> str:
        """
        Generate natural language explanation
        
        Args:
            vision_dict: Emotion probabilities
            text_dict: Sentiment probabilities
            fusion_result: Fusion analysis results
        
        Returns:
            Human-readable explanation string
        """
        # Extract key information
        top_emotion, emotion_conf = max(vision_dict.items(), key=lambda x: x[1])
        top_sentiment, sentiment_conf = max(text_dict.items(), key=lambda x: x[1])
        alignment_status = fusion_result['alignment_status']
        mismatch_conf = fusion_result['mismatch_confidence']
        
        # Select appropriate template category
        template_key = self._select_template_key(
            top_emotion, top_sentiment, alignment_status
        )
        
        # Choose random template from category
        templates = self.templates[template_key]
        template = random.choice(templates)
        
        # Format template
        explanation = template.format(
            emotion=top_emotion.lower(),
            sentiment=top_sentiment.lower(),
            confidence_vision=emotion_conf * 100,
            confidence_text=sentiment_conf * 100
        )
        
        # Add confidence scores
        scores_summary = self._generate_confidence_summary(
            top_emotion, emotion_conf, top_sentiment, sentiment_conf, mismatch_conf
        )
        
        full_explanation = f"{explanation}\n\n{scores_summary}"
        
        return full_explanation
    
    def _select_template_key(self, emotion: str, sentiment: str, status: str) -> str:
        """Select appropriate template category"""
        
        # Define emotion categories
        positive_emotions = {'Happy', 'Surprise'}
        negative_emotions = {'Sad', 'Angry', 'Fear', 'Disgust'}
        neutral_emotions = {'Neutral'}
        
        # Unclear status
        if status == 'UNCLEAR':
            return 'unclear'
        
        # Aligned cases
        if status == 'ALIGNED':
            if sentiment == 'Neutral' or emotion in neutral_emotions:
                return 'neutral_aligned'
            elif sentiment == 'Positive' and emotion in positive_emotions:
                return 'aligned_positive'
            elif sentiment == 'Negative' and emotion in negative_emotions:
                return 'aligned_negative'
            else:
                return 'aligned_positive'
        
        # Mismatch cases
        if status == 'MISMATCH':
            if sentiment == 'Positive':
                return 'mismatch_positive_text'
            else:
                return 'mismatch_negative_text'
        
        return 'unclear'
    
    def _generate_confidence_summary(self, emotion: str, emotion_conf: float,
                                    sentiment: str, sentiment_conf: float,
                                    mismatch_conf: float) -> str:
        """Generate confidence score summary"""
        
        summary_parts = []
        
        # Vision confidence
        summary_parts.append(
            f"📷 Visual Emotion: {emotion} ({emotion_conf*100:.1f}% confidence)"
        )
        
        # Text confidence
        summary_parts.append(
            f"📝 Textual Sentiment: {sentiment} ({sentiment_conf*100:.1f}% confidence)"
        )
        
        # Alignment status
        if mismatch_conf > 0.6:
            summary_parts.append(
                f"⚠️ Mismatch Detected ({mismatch_conf*100:.1f}% confidence)"
            )
        elif mismatch_conf < 0.4:
            summary_parts.append(
                f"✓ Aligned ({(1-mismatch_conf)*100:.1f}% confidence)"
            )
        else:
            summary_parts.append(
                f"❓ Unclear alignment ({mismatch_conf*100:.1f}% mismatch)"
            )
        
        return "\n".join(summary_parts)
    
    def generate_detailed_report(self, vision_dict: Dict, text_dict: Dict,
                               fusion_result: Dict) -> str:
        """
        Generate detailed analysis report
        
        Returns:
            Comprehensive report with all scores
        """
        explanation = self.generate_explanation(vision_dict, text_dict, fusion_result)
        
        # Add detailed breakdowns
        report = [
            "=" * 60,
            "MOODSYNCAI ANALYSIS REPORT",
            "=" * 60,
            "",
            explanation,
            "",
            "=" * 60,
            "DETAILED SCORES",
            "=" * 60,
            "",
            "Visual Emotion Breakdown:"
        ]
        
        # Sort emotions by confidence
        for emotion, prob in sorted(vision_dict.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 20)
            report.append(f"  {emotion:12s} {bar:20s} {prob*100:5.1f}%")
        
        report.append("\nTextual Sentiment Breakdown:")
        for sentiment, prob in sorted(text_dict.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 20)
            report.append(f"  {sentiment:12s} {bar:20s} {prob*100:5.1f}%")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


# Test function
if __name__ == "__main__":
    print("Testing EmotionExplainer...")
    
    # Initialize
    explainer = EmotionExplainer()
    
    # Test case: Mismatch scenario
    vision = {
        'Sad': 0.68, 'Fear': 0.15, 'Neutral': 0.08,
        'Happy': 0.04, 'Angry': 0.03, 'Disgust': 0.01, 'Surprise': 0.01
    }
    
    text = {
        'Positive': 0.81, 'Neutral': 0.12, 'Negative': 0.07
    }
    
    fusion = {
        'alignment_status': 'MISMATCH',
        'mismatch_confidence': 0.75,
        'details': {}
    }
    
    print("\n" + "=" * 60)
    explanation = explainer.generate_explanation(vision, text, fusion)
    print(explanation)
    
    print("\n" + "=" * 60)
    print("\nGenerating detailed report...")
    report = explainer.generate_detailed_report(vision, text, fusion)
    print(report)

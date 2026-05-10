# MoodSyncAI: Multi-Modal Emotion Analysis System

Detecting emotional mismatches between facial expressions and verbal communication using deep learning.

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 🎯 Project Overview

**MoodSyncAI** is a multi-modal AI system that analyzes facial emotions and text sentiment to detect when someone's words don't match their facial expressions. The system combines:

- **Vision Model**: Pre-trained facial emotion recognition (7 emotions)
- **Text Model**: DistilBERT for sentiment analysis (3 sentiments)
- **Custom Fusion Network**: Neural network achieving **95.75% validation accuracy** on mismatch detection

**Key Achievement**: Custom-designed fusion neural network trained from scratch with 95.75% accuracy ⭐

---

## 🌟 Features

- ✅ **Multi-Modal Analysis**: Combines visual and textual inputs
- ✅ **Real-Time Processing**: < 2 second response time
- ✅ **Intelligent Uncertainty**: Three-way classification (Aligned/Mismatch/Unclear)
- ✅ **Web Interface**: Gradio-based UI with webcam support
- ✅ **High Accuracy**: 95.75% validation accuracy on fusion network
- ✅ **Natural Language Explanations**: AI-generated analysis summaries

---

## 🏗️ Architecture

```
┌─────────────┐         ┌─────────────┐
│   IMAGE     │         │    TEXT     │
└──────┬──────┘         └──────┬──────┘
       │                       │
       ▼                       ▼
┌──────────────┐       ┌──────────────┐
│ Vision Model │       │  Text Model  │
└──────┬───────┘       └──────┬───────┘
       │                      │
       └──────────┬───────────┘
                  ▼
          ┌───────────────┐
          │ FUSION NETWORK│
          │  95.75% Acc ⭐ │
          └───────┬───────┘
                  ▼
          ┌───────────────┐
          │  PREDICTION   │
          └───────────────┘
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/[YOUR_USERNAME]/moodsync-ai.git
cd moodsync-ai

# Create environment
conda create -n moodsync python=3.10
conda activate moodsync

# Install dependencies
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

Open `http://127.0.0.1:7860` in your browser.

---

## 📊 Results

| Component | Performance |
|-----------|-------------|
| Vision Model | 40-80% confidence |
| Text Model | 100% accuracy |
| **Fusion Network** | **95.75% validation accuracy** ⭐ |

---

## 📁 Project Structure

```
moodsync_project/
├── app.py                    # Main application
├── train_fusion.py           # Training script
├── models/
│   └── fusion_trained.pth   # Trained model
└── src/
    ├── components/          # Model components
    └── fusion/             # Fusion network
```

---

## 👤 Author

**[Your Name]**  
Data Analytics-3 | May 2026

---

## 📄 License

MIT License

---

⭐ Star this repo if you find it helpful!

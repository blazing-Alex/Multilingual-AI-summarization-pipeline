# 📝 AI Text Summarizer & Translator

This is a project I built to explore the differences between **Extractive** and **Abstractive** text summarization. 

I wanted to see how "old school" statistical NLP compares to modern Transformer-based models when condensing long articles. To take it a step further, I also integrated a translation pipeline to convert the generated summaries into different languages on the go.

## 🚀 How It Works

This tool processes text using three distinct methods:

1.  **Extractive Summarization:** * Uses **NLTK** and **Spacy** to analyze word frequency and sentence ranking.
    * It identifies the most "important" sentences mathematically and pulls them out directly from the source.

2.  **Abstractive Summarization:**
    * Uses the **Facebook BART (Large-CNN)** model via the Hugging Face `transformers` library.
    * It reads the text and generates completely new sentences to capture the "gist," much like a human would.

3.  **Neural Translation:**
    * Takes the abstractive summary and translates it using **Helsinki-NLP** models.
    * Supports multiple target languages (e.g., French, German, Hindi).

## 🛠️ Tech Stack

* **Python 3.x**
* **Transformers (Hugging Face)** - Powering the BART and Opus-MT models.
* **Spacy** - For sentence segmentation and NLP tasks.
* **NLTK** - For tokenization and stopword filtering.
* **PyTorch/TensorFlow** - Backend for the transformer models.

## 📦 Installation

To run this locally, clone the repo and install the dependencies.

```bash
# Clone the repository
git clone [https://github.com/yourusername/text-summarizer.git](https://github.com/yourusername/text-summarizer.git)

# Navigate to the directory
cd text-summarizer

# Install required Python packages
pip install nltk spacy transformers torch

# Run the main script in your terminal
python main.py
```

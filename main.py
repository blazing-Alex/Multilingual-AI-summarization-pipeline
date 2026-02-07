import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from heapq import nlargest
from transformers import pipeline
import spacy

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Sample input
sample_text = """
The Avengers, a fictional team of superheroes from Marvel Comics, first assembled in 1963, created by Stan Lee and Jack Kirby. Iconic characters like Iron Man, Captain America, Thor, Hulk, Black Widow, and Hawkeye unite to combat threats too massive for any single hero, blending diverse powers and personalities.
Their stories, spanning comics, animated series, and the blockbuster Marvel Cinematic Universe (MCU) films, explore themes of teamwork, sacrifice, and resilience against villains like Loki, Ultron, and Thanos.
With a global fanbase, the Avengers symbolize hope and unity, their adventures continually evolving through new narratives and roster changes, cementing their cultural legacy.
"""

# Extractive summarization function
def extractive_summarization(text, num_sentences=3):
    stop_words = set(stopwords.words('english') + list(punctuation))
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    words = word_tokenize(text.lower())

    word_freq = {}
    for word in words:
        if word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1

    max_freq = max(word_freq.values(), default=1)
    word_freq = {word: freq / max_freq for word, freq in word_freq.items()}

    sentence_scores = {}
    for sent in sentences:
        for word in word_tokenize(sent.lower()):
            if word in word_freq:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + word_freq[word]

    summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary = ' '.join([sent for sent in sentences if sent in summary_sentences])
    return summary

# Abstractive summarization using BART
def abstractive_summarization(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    text = text[:1024]  # Limit input length
    summary = summarizer(text, max_length=120, min_length=40, do_sample=False)[0]['summary_text']
    return summary

# Translation function
def translate_summary(summary_text, target_lang="fr"):
    """
    Translate English summary to target language.
    Supported target_lang examples: "fr" (French), "de" (German), "hi" (Hindi), etc.
    """
    translator = pipeline("translation", model=f"Helsinki-NLP/opus-mt-en-{target_lang}")
    translation = translator(summary_text, max_length=400)[0]['translation_text']
    return translation

# Main function
def main():
    print("=== Text Summarization Project ===")
    print("\nOriginal Text:\n", sample_text)

    # Extractive Summary
    print("\nExtractive Summary:")
    extractive_summary = extractive_summarization(sample_text, num_sentences=3)
    print(extractive_summary)

    # Abstractive Summary
    print("\nAbstractive Summary:")
    abstractive_summary = abstractive_summarization(sample_text)
    print(abstractive_summary)

    # Translation option
    lang_code = input("\nEnter language code to translate the abstractive summary (e.g., fr for French, hi for Hindi, de for German): ").strip()
    if lang_code:
        translated_summary = translate_summary(abstractive_summary, target_lang=lang_code)
        print(f"\nTranslated Summary ({lang_code}):")
        print(translated_summary)

if __name__ == "__main__":
    main()

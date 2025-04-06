# 📝 Grammar Correction System using Fine-Tuned T5 Transformer

This project builds a **Grammar Correction System** using the **T5 Transformer model**, aimed specifically at correcting English-language text generated via voice input. The system supports training from scratch on grammar datasets, evaluates grammatical accuracy, and offers inference from both typed and spoken input.

We have fine-tuned a **T5 Transformer** for this task because:

- T5 (Text-to-Text Transfer Transformer) treats every NLP task as a text generation problem, making it naturally suited for grammar correction.
- It has shown strong performance on a wide range of language tasks with minimal task-specific modifications.
- Its encoder-decoder architecture allows effective mapping from incorrect to corrected text.

---

## 💽 Table of Contents

- [🔍 Introduction](#-introduction)
- [⚙️ Installation](#-installation)
- [📥 Data Collection](#-data-collection)
- [🔎 Data Examination](#-data-examination)
- [🧹 Dataset Preprocessing](#-dataset-preprocessing)
- [🏋️ Training](#-training)
- [📈 After Training Evaluating](#-after-training-evaluating)
- [🧠 Inference](#-inference)
- [🚀 Further Improvement](#-further-improvement)

---

## 🔍 Introduction

This project trains a **T5 Transformer model** to correct grammatical errors, especially in voice-generated English sentences. It includes:

- Data generation from `JFLEG` dataset.
- Grammar scoring.
- Fine-tuning the model using HuggingFace `transformers`.
- Voice-based input for real-time correction.

---

## ⚙️ Installation

```bash
# Clone repo
git clone https://github.com/your-username/grammar-correction-t5.git
cd grammar-correction-t5

# Install dependencies
# Tested on Python 3.9
pip install -r requirements.txt
```

**Key Libraries:**

- `transformers`
- `datasets`
- `torch`
- `pandas`
- `speechrecognition`, `sounddevice`, `scipy` (for voice input)
- `python-Levenshtein` (for grammar scoring)

---

## 📥 Data Collection

We use the [**JFLEG dataset**](https://huggingface.co/datasets/jfleg) via HuggingFace `datasets`.

Code Reference: [`data_utils.py`](./data_utils.py)

```python
from data_utils import load_jfleg_dataset
dataset = load_jfleg_dataset("validation")
```

---

## 🔎 Data Examination

The dataset contains sentences with multiple reference corrections. These are processed and formatted into a CSV file.

```python
from data_utils import generate_csv
generate_csv("train.csv", dataset, limit_to_one_target=True)
```

---

## 🧹 Dataset Preprocessing

Each sentence is prefixed with the `"grammar:"` tag, tokenized using the `T5Tokenizer`, and structured into HuggingFace `Dataset` objects.

Script: [`model_utils.py`](./model_utils.py)

---

## 🏋️ Training

Train the model from scratch (starting from `vennify/t5-base-grammar-correction`) using `Trainer`.

Run via: [`main.py`](./main.py)

```bash
python main.py
```

This includes:

- Splitting JFLEG dataset into train/eval
- Generating CSVs
- Training with early stopping and evaluation
- Saving the fine-tuned model

---

## 📈 After Training Evaluating

After training, the model is evaluated on unseen examples. Example output:

```
Original: This sentences, has bads grammar and spelling!
Corrected: This sentence has bad grammar and spelling!
Grammar score: 88.7
```

---

## 🧠 Inference

### 🐡 Text Input

Use `inference.py`:

```python
correct_text(model, tokenizer, "She no went there.")
```

### 🗣️ Voice Input

Run voice-based grammar correction via:

```python
from voice_app import voice_input_to_text
voice_input_to_text()  # Records and transcribes input
```

Runs automatic correction and scoring:

```python
corrected = correct_text(model, tokenizer, transcribed_text)
```

---

## 🚀 Further Improvement

- Transfer select evaluation examples into training set for more realistic learning.
- Tune hyperparameters with grid search (learning rate, batch size, epochs).
- Train on multilingual data to support grammar correction in other languages.
- Add custom decoder layers to refine correction quality.
- Try other models like `BART`, `GPT`, `T0`, or `mT5`.

---

## 📁 File Structure

```
.
├── main.py                # End-to-end pipeline
├── data_utils.py          # Load & preprocess JFLEG dataset
├── model_utils.py         # Train and evaluate T5 model
├── inference.py           # Text-based inference logic
├── grammer_score.py       # Grammar scoring using Levenshtein distance
├── voice_app.py           # Voice recording and transcription
├── train.csv / eval.csv   # Auto-generated CSVs
└── README.md              # You're here
```

---

## 🧠 Author

Built by T Rama Krishna. Inspired by the need to improve grammar in voice-based interfaces.


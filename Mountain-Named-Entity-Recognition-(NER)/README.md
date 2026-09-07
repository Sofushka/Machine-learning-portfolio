# Mountain Named Entity Recognition (NER)

## Model Weights

The fine-tuned model weights and tokenizer configs are hosted on Hugging Face Hub:
**[TerentievaSof/bert-base-mountain-ner](https://huggingface.co/TerentievaSof/bert-base-mountain-ner)**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange)](https://huggingface.co/docs/transformers/)

An end-to-end Machine Learning pipeline designed to automatically detect and extract mountain names (`B-MOUNTAIN`, `I-MOUNTAIN`) from unstructured English text using a fine-tuned Transformer model.

---

## Solution Overview

Token classification for entity extraction requires understanding fine-grained context and boundaries. This solution fine-tunes a pre-trained BERT architecture on a domain-specific tokenized dataset.

### Key Highlights:
* **Base Architecture:** `dslim/bert-base-NER` fine-tuned specifically for mountain entity extraction.
* **Tagging Scheme:** BIO (Beginning, Inside, Outside) format:
  * `O`: Non-entity tokens
  * `B-MOUNTAIN`: Beginning of a mountain entity
  * `I-MOUNTAIN`: Continuation of a mountain entity
* **Training Strategy:**
  * Optimized using AdamW with `learning_rate=2e-5` and a linear warmup ratio of `0.1`.
  * Evaluated on macro **F1-score** per epoch with `EarlyStoppingCallback` (patience = 3) to prevent overfitting.
  * Best weights automatically saved and pushed to the **Hugging Face Hub**.

---

## Repository Structure

```text
task1/
├── data/
│   ├── dataset.json                    # Train dataset with labeled entity spans
│   └── test.json                       # Test dataset for evaluation
├── dataset_creation.ipynb              # Notebook explaining dataset generation & formatting
├── demo.ipynb                          # Demo notebook showing interactive inference
├── inference.py                        # Pipeline for running predictions on new text
├── train.py                            # Model training script & HF upload pipeline
├── utils.py                            # Evaluation metrics & token alignment helpers
├── requirements.txt                    # Task dependencies
├── Report with potential improvements.pdf # Performance analysis and future ideas
└── README.md                           # Documentation

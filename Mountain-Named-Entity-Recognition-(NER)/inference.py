import os
from typing import List, Dict, Any
from transformers import pipeline

class MountainExtractor:
    """A wrapper class for convenient inference of the mountain search model."""

    def __init__(self, moedel_path: str ="task1/model_weights"):
        if not os.path.exists(moedel_path):
            raise FileNotFoundError(
                f"Model not found at {moedel_path}. Run train.py first"
            )

        # Use a pipeline to run the raw text.
        self.ner_pipeline = pipeline(
            'token-classification',
            model=moedel_path,
            tokenizer=moedel_path,
            aggregation_strategy='simple'
        )

    def predict_text(self, text: str):
        """Takes a single text as input and returns the entities found."""

        if not text.strip():
            return []

        prediction = self.ner_pipeline(text)
        
        results = []
        for pred in prediction:
            results.append({
                'entity': pred['entity_group'],
                'word': pred['word'].strip(),
                'score': round(float(pred['score']), 4),
                'start': pred['start'],
                'end': pred['end']
            })
        return results

    def predict_batch(self, texts : List[str]):
        """Gets a list of texts for batch processing."""

        return [self.predict_text(text) for text in texts]

def format_output(text: str, entities: List[Dict[str, Any]]):
    """Helper function for nice output to the console."""

    print(f"Text: {text}")
    if not entities:
        print('No mountains found.')
        return

    print('Mountains found:')
    for ent in entities:
        print(f"{ent['word']} (confidence: {ent['score'] * 100:.1f}%, position: {ent['start']}:{ent['end']})")
        

if __name__ == "__main__":
    MODEL_PATH = "TerentievaSof/bert-base-mountain-ner"

    extractor = MountainExtractor("task1/model_weights")

    test_samples = [
        "Yesterday we reached the summit of Mount Rainier after a long hike.",
        "We took amazing photos near Ben Nevis and Snowdon in Great Britain.",
        "They spent three days climbing Mount Dragonspire last weekend.",  # Fictional mountain
        "I just love hot coffee in the morning."  # A sentence without mountains
    ]

    print("Test sample")    
    batch_results = extractor.predict_batch(test_samples)

    for sample_text, entities in zip(test_samples, batch_results):
        format_output(sample_text, entities)

    print('Interactive mode')
    while True:
        user_input = input('\nEnter a sentence: ')
        if not user_input.strip():
            print('Exit')
            break

        found_mountains = extractor.predict_text(user_input)
        format_output(user_input, found_mountains)




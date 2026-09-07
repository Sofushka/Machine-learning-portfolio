import evaluate
import numpy as np

seqeval = evaluate.load("seqeval")

def tokenize_and_align_labels(examples, tokenizer):
    """
    BERT splits words into subwords.
    This function aligns labels to the first subword and assigns -100 
    to subsequent subwords so that the loss function ignores them.
    """

    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):        
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:            
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)             

            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs['labels'] = labels
    return tokenized_inputs


def compute_metrics(p, labels):
    """Calculates precision, recall, f1, and accuracy using seqeval."""
    predictions, labels_ref = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [labels[p_i] for (p_i, l_i) in zip(prediction, label) if l_i != -100]
        for prediction, label in zip(predictions, labels_ref)
    ]
    true_labels = [
        [labels[l_i] for (p_i, l_i) in zip(prediction, label) if l_i != -100]
        for prediction, label in zip(predictions, labels_ref)
    ]
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

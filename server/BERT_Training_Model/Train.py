from datasets import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import evaluate
import torch


def load_conll_file(filepath):
    """
    Load a CoNLL file with the specific format from your data
    """
    sentences = []
    current_sentence = []
    current_labels = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Skip document start markers
            if line.startswith("-DOCSTART-") or not line:
                if current_sentence:
                    sentences.append({"tokens": current_sentence, "ner_tags": current_labels})
                    current_sentence = []
                    current_labels = []
                continue

            parts = line.split()
            if len(parts) >= 1:
                token = parts[0]
                # Get the last part which should be the NER tag
                label = parts[-1] if len(parts) > 1 else "O"

                current_sentence.append(token)
                current_labels.append(label)

    # Add the last sentence if there is one
    if current_sentence:
        sentences.append({"tokens": current_sentence, "ner_tags": current_labels})

    return sentences


# Load the data
data = load_conll_file("project-11-at-2025-04-25-14-56-019ba00a.conll")
dataset = Dataset.from_list(data)

# Print a sample to verify it loaded correctly
print(f"Loaded {len(dataset)} sentences")
if len(dataset) > 0:
    print("Sample:", dataset[0])

# Get all unique labels and create mappings
unique_labels = sorted(set(label for example in data for label in example["ner_tags"]))
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}

print(f"Labels: {unique_labels}")
print(f"Label mappings: {label2id}")

# First convert the dataset to a list
data_list = dataset.to_list()

# Then perform the split
train_data, eval_data = train_test_split(data_list, test_size=0.2, random_state=42)

# Convert back to Dataset objects
train_dataset = Dataset.from_list(train_data)
eval_dataset = Dataset.from_list(eval_data)

print(f"Training samples: {len(train_dataset)}, Evaluation samples: {len(eval_dataset)}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")  # Using multilingual model for French text


def tokenize_and_align_labels(examples):
    """
    Tokenize all examples and align the labels with the tokens
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            # Special tokens have word_idx set to None
            if word_idx is None:
                label_ids.append(-100)
            # For the first token of a word
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            # For later subword tokens
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# Apply tokenization
tokenized_train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_and_align_labels, batched=True)

# Load the model
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# Load the evaluation metric
seqeval = evaluate.load("seqeval")


# 1. Adjust the training arguments for better learning
training_args = TrainingArguments(
    output_dir="./brand-ner-model-results",
    eval_strategy="steps",           # More frequent evaluation
    save_strategy="steps",
    eval_steps=5,                    # Evaluate every 5 steps
    save_steps=5,
    learning_rate=5e-5,             # Increased learning rate
    per_device_train_batch_size=12,  # Smaller batch size
    per_device_eval_batch_size=12,
    num_train_epochs=10,            # More epochs
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
    logging_dir="./logs",
    logging_steps=10
)

# 2. Add class weights to handle imbalanced data
class_weights = torch.tensor([
    1.0,                            # B-BRAND
    1.0,                            # I-BRAND
    0.1                             # O (reduce weight for non-brand tokens)
]).to(model.device)

# 3. Modify the compute_metrics function to include per-class metrics
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions,
                            references=true_labels,
                            zero_division=0)  # Add zero_division parameter

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
        "brand_f1": results.get("B-BRAND", {}).get("f1", 0.0)
    }

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
print("Starting training...")
trainer.train()

# Save the model
model_path = "./brand-ner-model"
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)
print(f"Model saved to {model_path}")


# Test the model on a few examples
def test_model_on_examples(examples):
    """
    Test the trained model on a few examples
    """
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    results = []

    for example in examples:
        tokens = example.split()
        # Get both the inputs and encoding object
        encoding = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=2)
        predicted_labels = [id2label[p.item()] for p in predictions[0]]

        # Use the tokenizer object to get word_ids
        word_ids = tokenizer(tokens, is_split_into_words=True).word_ids()
        aligned_preds = []
        current_word = None

        for word_idx, pred_label in zip(word_ids, predicted_labels):
            if word_idx is None:
                continue
            if word_idx != current_word:
                aligned_preds.append(pred_label)
                current_word = word_idx

        # Truncate predictions to match input length
        aligned_preds = aligned_preds[:len(tokens)]

        # Format the results
        result = []
        for token, label in zip(tokens, aligned_preds):
            result.append((token, label))

        results.append(result)

    return results

# Test examples
test_examples = [
    "Pc Portable Gamer Lenovo - Ecran : 15,6'' FHD (1920 x 1080) IPS 300 nits antireflet - Processeur : Intel Core i7-13650HX (2.6 up to 4.9GHz, 24 Mo de mémoire cache, 14C/20T) - Système d'exploitation : FreeDos - Mémoire RAM : 16 Go DDR5 - Disque Dur : 512 Go SSD - Carte Graphique : NVIDIA GeForce RTX 4050 6Go GDDR6 avec WI-FI6 et Bluetooth 5.2, 3x USB-A (USB 3.2 Gen 1), 1x USB-C (USB 3.2 Gen 2), 1x HDMI 2.1, 1x Prise combo casque/microphone (3,5 mm), 1x Ethernet (RJ-45) 1x Connecteur d'alimentation - Haut parleur & Microphone intégré - Clavier rétroéclaire blanc - Webcam FHD 1080p - Couleur : Gris - Garantie : 2 ans",
    "Écran 15.6'' Full HD - Processeur: Intel Celeron N4500 (1.10 GHz up to 2.80 GHz, 4Mo de mémoire cache, Dual-Core) - Système d'exploitation: Windows 11 Pro - Mémoire RAM: 16 Go - Disque Dur: 256 Go SSD - Carte Graphique: Intel UHD Graphics avec Wifi, Bluetooth, 1x USB 3.2 Gen 1 Type-A, 1x USB 3.2 Gen 1 Type-C, 2x USB 2.0 Type-A, 1x HDMI 1.4, 1x prise audio combinée 3,5 mm, 1x Entrée CC - Clavier chiclet - Couleur: Silver - Garantie: 1 an",
    "Pc Portable Gamer Lenovo LOQ 15IAX9 - Ecran : 15.6'' FHD (1920x1080) IPS 300nits Anti-reflet, 100% sRGB ,144Hz - Processeur : Intel Core i5-12450HX (up to 4.4 GHz, 12Mo de mémoire cache Smart, 8 cores) - Système d'exploitation : FreeDos - Mémoire RAM : 16 Go DDR5 - Disque Dur : 512 Go SSD - Carte Graphique : NVIDIA GeForce RTX 2050 4Go GDDR6 avec Bluetooth 5.2 et Wifi 6, 2x USB-A (USB 5Gbps / USB 3.2 Gen 1), 1x USB-C (USB 5Gbps / USB 3.2 Gen 1), 1x HDMI® 2.1, up to 8K/60Hz, 1x Headphone / microphone combo jack (3.5mm), 1x Ethernet (RJ-45), 1x Card reader, 1x Power connector - Clavier Non rétroéclairé français/arabe - Caméra HD 720p avec obturateur électronique - Couleur : Gris - Garantie : 2 ans",
    "Écran 15.6'' Full HD (1920 x 1080), IPS antireflet, 144 Hz - Processeur Intel Core i5-12450HX 12e génération, (jusqu'à 4.4 GHz, 12 Mo de mémoire cache) - Mémoire 16 Go DDR5 - Disque SSD NVMe 512 Go - Carte graphique NVIDIA GeForce RTX 2050, 4 Go de mémoire GDDR6 dédiée - Lecteur de cartes 3 en 1 - 2x Haut-parleurs stéréo 2W, optimisés avec Nahimic Audio - Caméra HD 720p avec obturateur de confidentialité - Ethernet (RJ-45) - Wi-Fi 6 - Bluetooth 5.2 - 2x USB-A 3.2 - 1x USB-C 3.2 - 1x HDMI 2.1 - 1x prise combinée casque/microphone (3,5 mm) - Couleur Gris Luna - Windows 11 - Garantie 2 ans Avec Souris RVB Lenovo LOQ M100"
]

print("\nTesting model on examples:")
predictions = test_model_on_examples(test_examples)

for i, example in enumerate(test_examples):
    print(f"\nExample {i + 1}: {example}")
    print("Predictions:")
    for token, label in predictions[i]:
        print(f"{token}: {label}")

# Get evaluation results
eval_results = trainer.evaluate()

print("\nModel Performance Metrics:")
print(f"Overall Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"Precision: {eval_results['eval_precision']:.4f}")
print(f"Recall: {eval_results['eval_recall']:.4f}")
print(f"F1 Score: {eval_results['eval_f1']:.4f}")
print(f"Brand F1 Score: {eval_results['eval_brand_f1']:.4f}")


from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Load model and tokenizer
model = AutoModelForTokenClassification.from_pretrained("../../Client_Module-3ec06e0e4600c2e048b8c69a55bc3d18569fabce/server/specs-ner-model")
tokenizer = AutoTokenizer.from_pretrained("../../Client_Module-3ec06e0e4600c2e048b8c69a55bc3d18569fabce/server/specs-ner-model")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval()
model.to(device)


def extract_entities(example):
    """
    Extract entities from text and join words with the same entity type
    """
    tokens = example.split()

    # Get predictions
    encoding = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)
    predicted_labels = [model.config.id2label[p.item()] for p in predictions[0]]

    # Align predictions with tokens
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

    # Group tokens by entity type (ignoring B-/I- prefixes)
    entities = {}
    current_entity = None
    current_tokens = []

    for token, label in zip(tokens, aligned_preds):
        if label == "O":
            # If we were building an entity, save it
            if current_entity and current_tokens:
                if current_entity not in entities:
                    entities[current_entity] = []
                entities[current_entity].append(" ".join(current_tokens))
                current_tokens = []
            current_entity = None
        else:
            # Extract the entity type without B-/I- prefix
            entity_type = label.split('-')[-1]

            # If we're starting a new entity type
            if current_entity != entity_type and current_entity is not None:
                if current_entity not in entities:
                    entities[current_entity] = []
                entities[current_entity].append(" ".join(current_tokens))
                current_tokens = []

            current_entity = entity_type
            current_tokens.append(token)

    # Don't forget the last entity
    if current_entity and current_tokens:
        if current_entity not in entities:
            entities[current_entity] = []
        entities[current_entity].append(" ".join(current_tokens))

    return entities


# Test function
if __name__ == "__main__":
    test_examples = [

        'Écran 16" WUXGA (1920 x 1200), rétroéclairage LED, 144 Hz, écran antireflet - Processeur Intel Core 5 210H, (jusqu’à 4.8 GHz, 12 Mo de mémoire cache) - Mémoire 8 Go DDR5 - Disque SSD M.2 NVMe 512 Go - Carte graphique NVIDIA GeForce RTX 3050, 6 Go de mémoire GDDR6 dédiée - 1x USB 3.2 Type-C - 2x USB 3.2 Type-A - 1x HDMI 2.1 - 1x prise audio combinée 3.5 mm - Clavier chiclet rétroéclairé avec touche numérique, course des touches de 1,5 mm, pavé tactile de précision - Caméra FHD 1080p avec obturateur de confidentialité - Audio SonicMaster - Haut-parleur / Microphone intégrés - Wi-Fi 6 - Bluetooth 5.3 - FreeDos - Couleur Noir - Garantie 1 an Avec Sac à dos ASUS',
        "Écran 15.6'' Full HD (1920 x 1080), antireflet - Taux de rafraîchissement: 144 Hz - Processeur AMD Ryzen 7 7435HS, (jusqu'à 4.5 GHz, 20 Mo de mémoire cache) - Mémoire 8 Go DDR5 - Disque SSD M.2 NVMe 512 Go - Carte graphique NVIDIA GeForce RTX 2050, 4 Go de mémoire GDDR6 dédiée - Wi-Fi 6 - Bluetooth 5.1 - Caméra HD 720P - Clavier Chiclet rétroéclairé 1 zone RVB - Système à 2 haut-parleurs - 1x port LAN RJ45 - 3x USB 3.2 - 1x USB-C 3.2 - 1x HDMI 2.0b - 1x prise audio combinée 3.5 mm - Windows 11 - Couleur Noir - Garantie 1 an",
        'Ecran 15.6" LED Full HD - Processeur Intel Core I5-13500H Up to 4.7 GHz, 18 Mo de mémoire cache - Mémoire 8 Go - Disque SSD 512 Go M.2 NVMe  - Carte graphique Intel Iris Xe -  WiFi - Bluetooth - HDMI - 1x USB 3.2 Type C - USB 3.2 - 1x HDMI - 1x Prise Jack 3.5 mm - Windows 11 - Couleur Bleu - Garantie 1 an'
    ]

    for i, example in enumerate(test_examples):
        print(f"\nExample {i + 1}:")
        entities = extract_entities(example)

        for entity_type, entity_mentions in entities.items():
            print(f"\n{entity_type}:")
            for mention in entity_mentions:
                print(f"  {mention}")
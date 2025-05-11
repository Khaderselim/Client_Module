import re
import pandas as pd  # Import pandas
from sentence_transformers import SentenceTransformer  # Import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity  # Import cosine_similarity
import numpy as np  # Import numpy
from transformers import  AutoTokenizer, AutoModelForTokenClassification
import torch

model = AutoModelForTokenClassification.from_pretrained("./specs-ner-model")
tokenizer = AutoTokenizer.from_pretrained("./specs-ner-model")
model_brand = AutoModelForTokenClassification.from_pretrained("./brand-ner-model")
tokenizer_brand = AutoTokenizer.from_pretrained("./brand-ner-model")
id2label_brand = model_brand.config.id2label

device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval()
model.to(device)

# Get the id2label mapping from the model config
id2label = model.config.id2label
def extract_entities(example, confidence_threshold=0.7):
    """
    Extract entities from text with a confidence threshold
    """
    tokens = example.split()

    # Tokenize the input
    encoding = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Get probabilities using softmax
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    max_probs, predictions = torch.max(probs, dim=2)

    # Convert predictions to labels
    predicted_labels = [model.config.id2label[p.item()] for p in predictions[0]]

    # Align predictions with tokens
    word_ids = tokenizer(tokens, is_split_into_words=True).word_ids()
    aligned_preds = []
    current_word = None

    for word_idx, (pred_label, prob) in zip(word_ids, zip(predicted_labels, max_probs[0])):
        if word_idx is None:
            continue
        if word_idx != current_word:
            aligned_preds.append((pred_label, prob.item()))
            current_word = word_idx

    aligned_preds = aligned_preds[:len(tokens)]

    # Group tokens by entity type
    entities = {}
    current_entity = None
    current_tokens = []

    for token, (label, prob) in zip(tokens, aligned_preds):
        if label == "O" or prob < confidence_threshold:
            # If we were building an entity, save it
            if current_entity and current_tokens:
                if current_entity not in entities:
                    entities[current_entity] = []
                entities[current_entity].append(" ".join(current_tokens))
                current_tokens = []
            current_entity = None
        else:
            entity_type = label.split('-')[-1]
            if current_entity != entity_type and current_entity is not None:
                if current_entity not in entities:
                    entities[current_entity] = []
                entities[current_entity].append(" ".join(current_tokens))
                current_tokens = []
            current_entity = entity_type
            current_tokens.append(token)

    # Save last entity
    if current_entity and current_tokens:
        if current_entity not in entities:
            entities[current_entity] = []
        entities[current_entity].append(" ".join(current_tokens))

    return entities

def create_weighted_embeddings(df, model, name_weight, brand_weight, component_weight, description_weight, price_weight):
        # Initialize the combined embedding
        embeddings = 0

        # Dictionary of fields and their weights
        field_weights = {
            'name': name_weight,
            'description': description_weight,
            'price': price_weight,
            'brands': brand_weight,
            'RAM': component_weight,
            'CPU': component_weight,
            'GPU': component_weight,
            'STORAGE': component_weight,
            'SCREEN': component_weight
        }

        # Clean and encode existing fields
        for field, weight in field_weights.items():
            if field in df.columns:
                if field in ['name', 'brands']:
                    cleaned_text = df[field].str.lower().apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x)))
                elif field == 'description':
                    cleaned_text = df[field].apply(remove_html_tags).str.lower().apply(
                        lambda x: re.sub(r'[^\w\s]', ' ', str(x)))
                elif field == 'price':
                    cleaned_text = df[field].astype(str).apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x)))
                else:
                    cleaned_text = df[field].fillna('')

                field_embedding = model.encode(cleaned_text.tolist(), convert_to_tensor=True)
                embeddings = embeddings + (weight * field_embedding)

        return embeddings

def remove_html_tags(text):
    if isinstance(text, str):  # Ensure the input is a string
        return re.sub(r'<[^>]*>', '', text)
    return text
def extract_brands(examples):
    """
    Test the trained model_brand on a few examples
    """
    model_brand.eval()
    model_brand.to("cuda" if torch.cuda.is_available() else "cpu")



    tokens = examples.split()
    # Get both the inputs and encoding object
    encoding = tokenizer_brand(tokens, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model_brand.device) for k, v in encoding.items()}
    with torch.no_grad():
        outputs = model_brand(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)
    predicted_labels = [id2label_brand[p.item()] for p in predictions[0]]
    # Use the tokenizer_brand object to get word_ids
    word_ids = tokenizer_brand(tokens, is_split_into_words=True).word_ids()
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
        if label != 'O':
            result.append(token)


    return result

def test_comparing(original, competitor):
        """
        Compare original and competitor products
        """
        # Initialize DataFrame and entity columns
        df_competitors = pd.DataFrame([competitor])
        entity_columns = {}

        # Extract brands and entities
        df_competitors['brands'] = df_competitors['name'].apply(lambda x: ", ".join(extract_brands(x)))

        for idx, entities in enumerate(df_competitors['description'].apply(extract_entities)):
            for entity_type, mentions in entities.items():
                if entity_type not in entity_columns:
                    entity_columns[entity_type] = [None] * len(df_competitors)
                entity_columns[entity_type][idx] = ", ".join(mentions) if mentions else None

        # Add entity columns to DataFrame
        for entity_type, values in entity_columns.items():
            df_competitors[entity_type] = values

        df_original = pd.DataFrame([original])
        df_original['brands'] = df_original['name'].apply(lambda x: ", ".join(extract_brands(x)))
        for idx, entities in enumerate(df_original['description'].apply(extract_entities)):
            for entity_type, mentions in entities.items():
                df_original[entity_type] = ", ".join(mentions) if mentions else None

        model = SentenceTransformer('./compare_model')

        df_embeddings = create_weighted_embeddings(
            df_original,
            model,
            name_weight=3.0,
            brand_weight=3.0,
            component_weight=4.0,
            description_weight=2.0,
            price_weight=1.0
        )

        dfs_embeddings = create_weighted_embeddings(
            df_competitors,
            model,
            name_weight=3.0,
            brand_weight=3.0,
            component_weight=4.0,
            description_weight=2.0,
            price_weight=1.0
        )

        # Move tensors to CPU and convert to numpy arrays
        similarity = cosine_similarity(
            df_embeddings.cpu().numpy(),
            dfs_embeddings.cpu().numpy()
        )

        return float(np.diag(similarity)[0])





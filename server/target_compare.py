import re
import mysql.connector
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
id2label = model.config.id2label
device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval()
model.to(device)

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

def extract_brands(text):
    """Extract brand names from text using NER model"""
    if not isinstance(text, str) or not text.strip():
        return []

    try:
        model_brand.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_brand.to(device)

        tokens = text.split()
        encoding = tokenizer_brand(tokens, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model_brand(**inputs)

        predictions = torch.argmax(outputs.logits, dim=2)
        predicted_labels = [model_brand.config.id2label[p.item()] for p in predictions[0]]
        word_ids = tokenizer_brand(tokens, is_split_into_words=True).word_ids()

        aligned_preds = []
        current_word = None
        for word_idx, pred_label in zip(word_ids, predicted_labels):
            if word_idx is None:
                continue
            if word_idx != current_word:
                aligned_preds.append(pred_label)
                current_word = word_idx

        aligned_preds = aligned_preds[:len(tokens)]
        result = []
        for token, label in zip(tokens, aligned_preds):
            if label != 'O':
                result.append(token)

        return result
    except Exception as e:
        print(f"Error extracting brands: {str(e)}")
        return []

def format_name(name):
    if not isinstance(name, str):
        return name
    # First lowercase everything
    name = name.lower()
    # Then title case each word, preserving brand names
    words = name.split()
    # Title case each word
    words = [w.title() for w in words]
    return ' '.join(words)

def remove_html_tags(text):
    if isinstance(text, str):  # Ensure the input is a string
        return re.sub(r'<[^>]*>', '', text)
    return text

def calculate_similarities(df, model):
    """Calculate similarities between target and competitor products"""
    try:
        # Extract brands
        df['target_brands'] = df['target_name'].fillna('').apply(lambda x: ", ".join(extract_brands(x)))
        df['competitor_brands'] = df['competitor_name'].fillna('').apply(lambda x: ", ".join(extract_brands(x)))

        # Define weights
        name_weight = 3
        description_weight = 2
        brands_weight = 3
        price_weight = 3
        components_weight = 4

        cleaned_target_names = df['target_name'].str.lower().apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x)))
        cleaned_target_descriptions = df['target_description'].apply(remove_html_tags).str.lower().apply(
            lambda x: re.sub(r'[^\w\s]', ' ', str(x)))
        cleaned_target_prices = df['target_price'].astype(str).apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x)))
        cleaned_target_brands = df['target_brands'].str.lower().apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x)))

        # Get target embeddings
        target_name_embedding = model.encode(cleaned_target_names.fillna('').tolist(), convert_to_tensor=True)
        target_description_embedding = model.encode(cleaned_target_descriptions.fillna('').tolist(), convert_to_tensor=True)
        target_brands_embedding = model.encode(cleaned_target_brands.fillna('').tolist(), convert_to_tensor=True)
        target_price_embedding = model.encode(cleaned_target_prices.fillna('').astype(str).tolist(), convert_to_tensor=True)
        target_CPU_embedding = model.encode(df['target CPU'].fillna('').tolist(), convert_to_tensor=True)
        target_GPU_embedding = model.encode(df['target GPU'].fillna('').tolist(), convert_to_tensor=True)
        target_RAM_embedding = model.encode(df['target RAM'].fillna('').tolist(), convert_to_tensor=True)
        target_STORAGE_embedding = model.encode(df['target STORAGE'].fillna('').tolist(), convert_to_tensor=True)
        target_SCREEN_embedding = model.encode(df['target SCREEN'].fillna('').tolist(), convert_to_tensor=True)

        cleaned_competitor_names = df['competitor_name'].str.lower().apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x)))
        cleaned_competitor_descriptions = df['competitor_description'].apply(remove_html_tags).str.lower().apply(
            lambda x: re.sub(r'[^\w\s]', ' ', str(x)))
        cleaned_competitor_prices = df['competitor_price'].astype(str).apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x)))
        cleaned_competitor_brands = df['competitor_brands'].str.lower().apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x)))
        # Get competitor embeddings
        competitor_name_embedding = model.encode(cleaned_competitor_names.fillna('').tolist(), convert_to_tensor=True)
        competitor_description_embedding = model.encode(cleaned_competitor_descriptions.fillna('').tolist(), convert_to_tensor=True)
        competitor_brands_embedding = model.encode(cleaned_competitor_brands.fillna('').tolist(), convert_to_tensor=True)
        competitor_price_embedding = model.encode(cleaned_competitor_prices.fillna('').astype(str).tolist(), convert_to_tensor=True)
        competitor_CPU_embedding = model.encode(df['competitor CPU'].fillna('').tolist(), convert_to_tensor=True)
        competitor_GPU_embedding = model.encode(df['competitor GPU'].fillna('').tolist(), convert_to_tensor=True)
        competitor_RAM_embedding = model.encode(df['competitor RAM'].fillna('').tolist(), convert_to_tensor=True)
        competitor_STORAGE_embedding = model.encode(df['competitor STORAGE'].fillna('').tolist(), convert_to_tensor=True)
        competitor_SCREEN_embedding = model.encode(df['competitor SCREEN'].fillna('').tolist(), convert_to_tensor=True)

        # Calculate weighted embeddings
        target_embedding = (
            target_name_embedding * name_weight +
            target_description_embedding * description_weight +
            target_brands_embedding * brands_weight +
            target_price_embedding * price_weight +
            target_CPU_embedding * components_weight +
            target_GPU_embedding * components_weight +
            target_RAM_embedding * components_weight +
            target_STORAGE_embedding * components_weight +
            target_SCREEN_embedding * components_weight
        )

        competitor_embedding = (
            competitor_name_embedding * name_weight +
            competitor_description_embedding * description_weight +
            competitor_brands_embedding * brands_weight +
            competitor_price_embedding * price_weight +
            competitor_CPU_embedding * components_weight +
            competitor_GPU_embedding * components_weight +
            competitor_RAM_embedding * components_weight +
            competitor_STORAGE_embedding * components_weight +
            competitor_SCREEN_embedding * components_weight
        )

        # Calculate similarity using the weighted embeddings
        similarity = cosine_similarity(
            target_embedding.numpy(),
            competitor_embedding.numpy()
        )

        # Add diagonal similarities to DataFrame
        df['similarity'] = np.diag(similarity)

        return df

    except Exception as e:
        print(f"Error calculating similarities: {str(e)}")
        return df

def compare_tracking_product(host , user , passwd , database , database_prefix):
    db = mysql.connector.connect(
        host=host,
        user=user,
        password=passwd,
        database=database)
    cursor = db.cursor()

    # Retrieve target products
    cursor.execute("SELECT c.id_product as id_target_product, c.name as target_name, c.description as target_description ,c.price as target_price ,cp.id_product as id_competitor_product, cp.name as competitor_name, cp.description as competitor_description, cp.price as competitor_price FROM "+database_prefix+"target_products_relation r LEFT JOIN "+database_prefix+"target_competitor_product cp ON r.id_product_competitor = cp.id_product LEFT JOIN "+database_prefix+"targets_products c ON r.id_product_target = c.id_target_product LEFT JOIN "+database_prefix+"target_competitor co ON cp.id_competitor = co.id_target_competitor ORDER BY c.id_target_product DESC; ")
    products = cursor.fetchall()

    df = pd.DataFrame(products, columns=[i[0] for i in cursor.description])
    # Process each row's description


    # Sentence transformer for similarity comparison
    similarity_model = SentenceTransformer('./compare_model')

    df['target_description'] = df['target_description'].apply(remove_html_tags)
    df['competitor_description'] = df['competitor_description'].apply(remove_html_tags)

    entity_columns = {}

    # Process each row's description
    for idx, entities in enumerate(df['target_description'].apply(extract_entities)):
        for entity_type, mentions in entities.items():
            if entity_type not in entity_columns:
                entity_columns[entity_type] = [None] * len(df)
            # Join multiple mentions with comma if there are multiple
            entity_columns[entity_type][idx] = ", ".join(mentions) if mentions else None

    # Add entity columns to DataFrame
    for entity_type, values in entity_columns.items():
        df["target " + entity_type] = values

    entity_columns = {}

    # Process each row's description
    for idx, entities in enumerate(df['competitor_description'].apply(extract_entities)):
        for entity_type, mentions in entities.items():
            if entity_type not in entity_columns:
                entity_columns[entity_type] = [None] * len(df)
            # Join multiple mentions with comma if there are multiple
            entity_columns[entity_type][idx] = ", ".join(mentions) if mentions else None

    # Add entity columns to DataFrame
    for entity_type, values in entity_columns.items():
        df["competitor " + entity_type] = values
    # Apply the calculations
    df = calculate_similarities(df, similarity_model)

    for index, row in df.iterrows():
        cursor.execute("""
                       INSERT INTO """+database_prefix+"""comparing_product
                       (id_product, id_competitor_product, product_brands, competitor_product_brands, similarity)
                       VALUES (%s, %s, %s, %s, %s) ON DUPLICATE KEY
                       UPDATE
                           product_brands =
                       VALUES (product_brands), competitor_product_brands =
                       VALUES (competitor_product_brands), similarity =
                       VALUES (similarity)
                       """, (int(row['id_target_product']), int(row['id_competitor_product']),
                             row['target_brands'], row['competitor_brands'], float(row['similarity'])))
        db.commit()


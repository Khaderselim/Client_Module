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
    # Clean the text fields first
    cleaned_names = df['name'].str.lower().apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x)))
    cleaned_descriptions = df['description'].apply(remove_html_tags).str.lower().apply(
        lambda x: re.sub(r'[^\w\s]', ' ', str(x)))
    cleaned_prices = df['price'].astype(str).apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x)))
    cleaned_brands = df['brands'].str.lower().apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x)))

    # Get embeddings for each field
    name_embeddings = model.encode(cleaned_names.tolist(), convert_to_tensor=True)
    ram_embeddings = model.encode(df['RAM'].fillna('').tolist(), convert_to_tensor=True)
    cpu_embeddings = model.encode(df['CPU'].fillna('').tolist(), convert_to_tensor=True)
    gpu_embeddings = model.encode(df['GPU'].fillna('').tolist(), convert_to_tensor=True)
    storage_embeddings = model.encode(df['STORAGE'].fillna('').tolist(), convert_to_tensor=True)
    screen_embeddings = model.encode(df['SCREEN'].fillna('').tolist(), convert_to_tensor=True)
    description_embeddings = model.encode(cleaned_descriptions.tolist(), convert_to_tensor=True)
    price_embeddings = model.encode(cleaned_prices.tolist(), convert_to_tensor=True)
    brand_embeddings = model.encode(cleaned_brands.tolist(), convert_to_tensor=True)

    # Apply weights and combine
    weighted_embeddings = (
        name_weight * name_embeddings +
        component_weight * ram_embeddings +
        component_weight * cpu_embeddings +
        component_weight * gpu_embeddings +
        component_weight * storage_embeddings +
        component_weight * screen_embeddings +
        description_weight * description_embeddings  +
        price_weight * price_embeddings +
        brand_weight * brand_embeddings
    )

    return weighted_embeddings

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
def compare_product(host , user , passwd , database , database_prefix):
    db = mysql.connector.connect(
        host=host,
        user=user,
        password=passwd,
        database=database)
    cursor = db.cursor()

    # Retrieve target products
    cursor.execute(
        "SELECT DISTINCT prl.name, prl.description, pr.price,pr.id_product FROM `"+database_prefix+"product` pr LEFT JOIN `"+database_prefix+"product_lang` prl ON prl.id_product = pr.id_product ")
    products = cursor.fetchall()

    # Convert to pandas DataFrame
    df = pd.DataFrame(products, columns=[i[0] for i in cursor.description])
    df['brands'] = df['name'].apply(lambda x: ", ".join(extract_brands(x)))
    df['description'] = df['description'].apply(remove_html_tags)

    # Create dictionary to store entities by type
    entity_columns = {}

    # Process each row's description
    for idx, entities in enumerate(df['description'].apply(extract_entities)):
        for entity_type, mentions in entities.items():
            if entity_type not in entity_columns:
                entity_columns[entity_type] = [None] * len(df)
            # Join multiple mentions with comma if there are multiple
            entity_columns[entity_type][idx] = ", ".join(mentions) if mentions else None

    # Add entity columns to DataFrame
    for entity_type, values in entity_columns.items():
        df[entity_type] = values
    # print(df)

    cursor.execute("SELECT * FROM `"+database_prefix+"target_competitor_product`")
    competitors = cursor.fetchall()
    # Convert to pandas DataFrame
    df_competitors = pd.DataFrame(competitors, columns=[i[0] for i in cursor.description])
    df_competitors['brands'] = df_competitors['name'].apply(lambda x: ", ".join(extract_brands(x)))
    df_competitors['description'] = df_competitors['description'].apply(remove_html_tags)

    entity_columns = {}

    # Process each row's description
    for idx, entities in enumerate(df_competitors['description'].apply(extract_entities)):
        for entity_type, mentions in entities.items():
            if entity_type not in entity_columns:
                entity_columns[entity_type] = [None] * len(df_competitors)
            # Join multiple mentions with comma if there are multiple
            entity_columns[entity_type][idx] = ", ".join(mentions) if mentions else None

    # Add entity columns to DataFrame
    for entity_type, values in entity_columns.items():
        df_competitors[entity_type] = values
    # print(df_competitors)
    model = SentenceTransformer('./compare_model')



    df_embeddings = create_weighted_embeddings(df, model, name_weight=3.0, brand_weight=3.0, component_weight=4.0,description_weight=2.0,
                                               price_weight=1.0)
    # Embed the product names, descriptions, and prices for df_competitors

    dfs_embeddings = create_weighted_embeddings(df_competitors, model,  name_weight=3.0, brand_weight=3.0, component_weight=4.0,description_weight=2.0,
                                               price_weight=1.0)
    similarities = cosine_similarity(df_embeddings.numpy(), dfs_embeddings.numpy())

    # Add the similarity scores to the DataFrame
    df_competitors['similarity'] = np.diag(similarities)

    # Find the most similar product for each product in df
    most_similar_indices = similarities.argmax(axis=0)
    assigned_indices = set()
    for i in range(len(most_similar_indices)):
        if most_similar_indices[i] in assigned_indices:
            sorted_similarities = np.argsort(-similarities[:, i])
            for idx in sorted_similarities:
                if idx not in assigned_indices:
                    most_similar_indices[i] = idx
                    assigned_indices.add(idx)
                    break
                else:
                    assigned_indices.add(most_similar_indices[i])

    x = pd.DataFrame({
        'most_similar_product_id': df['id_product'].iloc[most_similar_indices].values,
        'most_similar_history_id': df_competitors['id_product'].values,
        'product_name': df['name'].iloc[most_similar_indices].values,
        'history_name': df_competitors['name'].values,
        'product_price': df['price'].iloc[most_similar_indices].values,
        'history_price': df_competitors['price'].values,
        'history_url': df_competitors['url'].values,
        'product_brands': df['brands'].iloc[most_similar_indices].values,
        'history_brands': df_competitors['brands'].values,
        'similarity': similarities.max(axis=0)
    })
    for index, row in x.iterrows():
        cursor.execute("""
               INSERT INTO """+database_prefix+"""suggestion_product
               (id_product, id_competitor_product, product_brands, competitor_product_brands, similarity)
               VALUES (%s, %s, %s, %s, %s)
               ON DUPLICATE KEY UPDATE
               product_brands = VALUES(product_brands),
               competitor_product_brands = VALUES(competitor_product_brands),
               similarity = VALUES(similarity)
           """, (int(row['most_similar_product_id']), int(row['most_similar_history_id']),
                 row['product_brands'], row['history_brands'], float(row['similarity'])))
    db.commit()


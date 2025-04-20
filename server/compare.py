import re
import mysql.connector
import pandas as pd  # Import pandas
from sentence_transformers import SentenceTransformer  # Import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity  # Import cosine_similarity
import numpy as np  # Import numpy
import os
def remove_html_tags(text):
    if isinstance(text, str):  # Ensure the input is a string
        return re.sub(r'<[^>]*>', '', text)
    return text
def compare_product(host , user , passwd , database , database_prefix):
    db = mysql.connector.connect(
        host=host,
        port=3307,
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
    # print(df)

    cursor.execute("SELECT * FROM `"+database_prefix+"target_competitor_product`")
    competitors = cursor.fetchall()
    # Convert to pandas DataFrame
    df_competitors = pd.DataFrame(competitors, columns=[i[0] for i in cursor.description])
    # print(df_competitors)
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    df['name_description_price'] = (
            df['name'].str.lower().apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x))) + ' ' +
            df['name'].str.lower().apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x))) + ' ' +
            df['name'].str.lower().apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x))) + ' ' +
            df['description'].apply(remove_html_tags).str.lower().apply(
                lambda x: re.sub(r'[^\w\s]', ' ', str(x))) + ' ' +
            df['price'].astype(str)
    )
    df_embeddings = model.encode(df['name_description_price'].tolist())
    # Embed the product names, descriptions, and prices for df_competitors
    df_competitors['name_description_price'] = (
            df_competitors['name'].str.lower().apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x))) + ' ' +
            df_competitors['name'].str.lower().apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x))) + ' ' +
            df_competitors['name'].str.lower().apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x))) + ' ' +
            df_competitors['description'].str.lower().apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x))) + ' ' +
            df_competitors['price'].astype(str)
    )
    dfs_embeddings = model.encode(df_competitors['name_description_price'].tolist())
    similarities = cosine_similarity(df_embeddings, dfs_embeddings)

    # Add the similarity scores to the DataFrame
    df_competitors['similarity'] = similarities.max(axis=0)

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
        'most_similar_competitor_product_id': df_competitors['id_product'].values,
        'product_name': df['name'].iloc[most_similar_indices].values,
        'competitor_product_name': df_competitors['name'].values,
        'product_price': df['price'].iloc[most_similar_indices].values,
        'competitor_product_price': df_competitors['price'].values,
        'competitor_product_url': df_competitors['url'].values,
        'similarity': similarities.max(axis=0)
    })

    for index, row in x.iterrows():
        cursor.execute("""
               INSERT INTO """+database_prefix+"""comparing_product (id_product, id_competitor_product,similarity)
               VALUES (%s, %s, %s)
               ON DUPLICATE KEY UPDATE
               id_product = VALUES(id_product)
           """, (int(row['most_similar_product_id']), int(row['most_similar_competitor_product_id']), float(row['similarity'])))
    db.commit()



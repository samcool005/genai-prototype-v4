import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
from snowflake.core import Root

## Use this for Streamlit in Snowflake deployment
# from snowflake.snowpark.context import get_active_session

# Establish Snowflake session

## Use this for Streamlit in Snowflake deployment
# session = get_active_session()

## Use this for Streamlit Community Cloud deployment
session = st.connection("snowflake").session()

# Create tabs
tab1, tab2 = st.tabs(["Data & Plots", "RAG App"])

# Tab 1: Data and Plots
with tab1:
    st.title("Customer Sentiment and Delivery Analysis")

    # Data loading functions
    @st.cache_data
    def load_data():
        query_reviews = """
        SELECT
            *
        FROM
            REVIEWS_WITH_SENTIMENT
        """
        return session.sql(query_reviews).to_pandas()

    # Load data
    df = load_data()

    # Average sentiment by product
    st.header("Average Sentiment by Product")
    avg_sentiment_product = df.groupby("PRODUCT")["SENTIMENT_SCORE"].mean().sort_values()

    fig1, ax1 = plt.subplots(figsize=(8,5))
    avg_sentiment_product.plot(kind="barh", color="skyblue", ax=ax1)
    ax1.set_xlabel("Sentiment Score")
    ax1.set_ylabel("Product")
    st.pyplot(fig1)

    # Filter by product selection
    product = st.selectbox("Choose a product", ["All Products"] + list(df["PRODUCT"].unique()))

    if product != "All Products":
        filtered_data = df[df["PRODUCT"] == product]
    else:
        filtered_data = df

    # Display combined dataset
    st.subheader(f"📁 Reviews for {product}")
    st.dataframe(filtered_data)

    # Average sentiment by delivery status
    st.header(f"Average Sentiment by Delivery Status for {product}")
    avg_sentiment_status = filtered_data.groupby("STATUS")["SENTIMENT_SCORE"].mean().sort_values()

    fig2, ax2 = plt.subplots(figsize=(8,5))
    avg_sentiment_status.plot(kind="barh", color="slateblue", ax=ax2)
    ax2.set_xlabel("Sentiment Score")
    ax2.set_ylabel("Delivery Status")
    st.pyplot(fig2)


# Tab 2: RAG App
with tab2:
    st.title("RAG App")

    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    EMBED_MODEL = "text-embedding-3-small"

    # Load chunks from Snowflake (cached — only reruns if underlying query changes)
    @st.cache_data
    def load_chunks():
        query = """
        SELECT file_name, chunk
        FROM GENAI_PROTOTYPE_DB.GENAI_PROTOTYPE_SCHEMA.CHUNKED_CONTENT
        """
        return session.sql(query).to_pandas()

    # Compute embeddings for all chunks once, cache in memory for the session
    @st.cache_resource
    def build_chunk_embeddings():
        chunks_df = load_chunks()
        texts = chunks_df["CHUNK"].tolist()

        # Batch embed (OpenAI allows lists of inputs per call)
        embeddings = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
            embeddings.extend([item.embedding for item in resp.data])

        chunks_df = chunks_df.copy()
        chunks_df["EMBEDDING"] = embeddings
        return chunks_df

    chunks_df = build_chunk_embeddings()
    embedding_matrix = np.array(chunks_df["EMBEDDING"].tolist())

    def cosine_similarity(query_vec, matrix):
        query_vec = np.array(query_vec)
        matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        query_norm = query_vec / np.linalg.norm(query_vec)
        return matrix_norm @ query_norm

    def search_chunks(query, top_k=3):
        query_embedding = client.embeddings.create(
            model=EMBED_MODEL, input=[query]
        ).data[0].embedding
        scores = cosine_similarity(query_embedding, embedding_matrix)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return chunks_df.iloc[top_idx]

    # Input box for user prompt
    prompt = st.text_input("Enter your query:", value="Any goggles review?")

    if prompt:
        if st.button("Run Query"):
            top_chunks = search_chunks(prompt, top_k=3)

            for _, row in top_chunks.iterrows():
                st.write(f"**{row['CHUNK']}**")
                st.caption(row['FILE_NAME'])
                st.write('---')

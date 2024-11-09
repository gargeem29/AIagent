import streamlit as st
import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Function to handle file upload and create FAISS index dynamically
def handle_file_upload(uploaded_file):
    if uploaded_file is not None:
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)
        st.write("Preview of the uploaded CSV:")
        st.dataframe(df.head())
        
        # Dynamically create a description column by combining all string-based columns
        description_columns = df.select_dtypes(include=['object']).columns
        df['description'] = df[description_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        
        # Initialize SentenceTransformer model to get embeddings
        model = SentenceTransformer('all-MiniLM-L6-v2')
        descriptions = df['description'].tolist()
        embeddings = model.encode(descriptions)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))

        # Save FAISS index and data
        with open('faiss_index.pkl', 'wb') as f:
            pickle.dump(index, f)
        with open('columns.pkl', 'wb') as f:
            pickle.dump(df.columns.tolist(), f)

        st.success("FAISS index created and saved successfully!")
        return df, index

    return None, None

# Function to get FAISS search results
def get_faiss_search_results(query, faiss_index, df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])[0]

    # Perform the FAISS search
    D, I = faiss_index.search(np.array([query_embedding]).astype('float32'), k=5)  # k=5 for top 5 results

    results = []
    for idx in I[0]:
        if idx < len(df):
            results.append(df.iloc[idx])  # Append the row corresponding to the index

    return results

# Main logic for Streamlit interface
def main():
    st.set_page_config(page_title="Web Data Extraction Tool", layout="wide")

    # Allow CSV file upload to create FAISS index
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    df = None
    faiss_index = None

    if uploaded_file:
        df, faiss_index = handle_file_upload(uploaded_file)

    # If FAISS index is created, allow search
    if faiss_index is not None:
        query = st.text_input("Enter search query:")
        if query:
            search_results = get_faiss_search_results(query, faiss_index, df)
            if search_results:
                st.write("Search Results:")
                st.dataframe(pd.DataFrame(search_results))
            else:
                st.write("No results found.")

        # Download button for the CSV data
        st.download_button(
            label="Download Results",
            data=df.to_csv(index=False),
            file_name="extracted_data.csv",
            mime="text/csv"
        )

        # Download FAISS index and columns
        st.download_button(
            label="Download FAISS Index",
            data=open('faiss_index.pkl', 'rb').read(),
            file_name="faiss_index.pkl",
            mime="application/octet-stream"
        )
        st.download_button(
            label="Download Columns Info",
            data=open('columns.pkl', 'rb').read(),
            file_name="columns.pkl",
            mime="application/octet-stream"
        )


if __name__ == "__main__":
    main()

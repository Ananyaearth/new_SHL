import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai

# Configure Gemini API
API_KEY = st.secrets.get("GEMINI_API_KEY", "AIzaSyASKTzSNuMbJMdZWr81Xuw2hS1Poe3acZo")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')

# Test Type mapping
test_type_map = {
    'A': 'Ability & Aptitude',
    'B': 'Biodata & Situational Judgement',
    'C': 'Competencies',
    'D': 'Development & 360',
    'E': 'Assessment Exercises',
    'K': 'Knowledge & Skills',
    'P': 'Personality & Behaviour',
    'S': 'Simulations'
}

# Load data and index
try:
    st.write("Loading CSV...")
    df = pd.read_csv("shl_catalog_with_summaries.csv")
    st.write("Loading FAISS...")
    index = faiss.read_index("shl_assessments_index.faiss")
    st.write("Loading SentenceTransformer...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    st.write("Models loaded successfully!")
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# LLM preprocessing function (unchanged from reference)
def llm_shorten_query(query):
    prompt = "Extract all technical skills from query as space-separated list, max 10: "
    try:
        response = model.generate_content(prompt + query)
        shortened = response.text.strip()
        words = shortened.split()
        return " ".join(words[:10]) if words else query
    except Exception as e:
        st.error(f"Query LLM error: {e}")
        return query

# Simplified retrieval function with Test Type and Description
def retrieve_assessments(query, k=10):
    processed_query = llm_shorten_query(query)
    st.write(f"Processed Query: {processed_query}")  # Debug
    query_embedding = embedding_model.encode([processed_query], show_progress_bar=False)[0]
    query_embedding = np.array([query_embedding], dtype='float32')
    distances, indices = index.search(query_embedding, k)
    results = df.iloc[indices[0]].copy()
    results["similarity_score"] = 1 - distances[0] / 2
    # Process Test Type into a list of full names
    results["Test Type"] = results["Test Type"].apply(
        lambda x: ", ".join([test_type_map.get(abbrev.strip(), abbrev.strip()) for abbrev in str(x).split()])
    )
    # Rename and select columns
    results = results.rename(columns={
        "Pre-packaged Job Solutions": "Assessment Name",
        "Assessment Length": "Duration"
    })
    return results[["Assessment Name", "URL", "Description", "Remote Testing (y/n)", 
                    "Adaptive/IRT (y/n)", "Duration", "Test Type"]].head(k)

# Streamlit UI
st.title("SHL Assessment Recommendation Engine")
st.write("Enter a query (e.g., 'Java developers, 40 mins').")
query = st.text_input("Your Query", "")
if st.button("Get Recommendations"):
    if query:
        results = retrieve_assessments(query, k=10)
        st.write("### Recommended Assessments")
        # Enhanced visual with styled DataFrame
        styled_results = results.style.set_properties(**{
            'text-align': 'left',
            'border': '1px solid #ddd',
            'padding': '8px'
        }).set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#f2f2f2'), ('font-weight', 'bold'), ('text-align', 'left')]},
            {'selector': 'td', 'props': [('white-space', 'normal')]}
        ])
        st.dataframe(styled_results)
    else:
        st.warning("Please enter a query.")

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai

# Configure Gemini API
API_KEY = st.secrets.get("GEMINI_API_KEY", "AIzaSyASKTzSNuMbJMdZWr81Xuw2hS1Poe3acZo")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')  # Gemini model for LLM

# Test Type mapping (consistent with your FastAPI version)
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

# Load data and models
try:
    st.write("Loading CSV...")
    df = pd.read_csv("shl_catalog_detailed.csv")
    st.write("Loading FAISS...")
    index = faiss.read_index("shl_assessments_index.faiss")
    st.write("Loading SentenceTransformer...")
    # Explicitly fetch model with cache, renamed to avoid conflict
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    st.write("All loaded!")
except Exception as e:
    st.error(f"Failed to load: {e}")
    st.stop()

st.title("🔍 SHL Assessment Recommendation Engine")

st.markdown("""
Enter a job description, skill, or role and get the most relevant SHL assessments.
""")

query = st.text_input("💬 Enter your job description or keyword:")

top_k = st.slider("Number of recommendations", 1, 10, 5)

if query:
    # LLM preprocessing from your reference code
    def llm_shorten_query(query):
        prompt = "Extract all technical skills from query as space-separated list, max 10: "
        try:
            response = model.generate_content(prompt + query)  # Use Gemini model
            shortened = response.text.strip()
            words = shortened.split()
            return " ".join(words[:10]) if words else query
        except Exception as e:
            st.error(f"Query LLM error: {e}")
            return query

    processed_query = llm_shorten_query(query)

    query_embedding = embedding_model.encode([processed_query])[0].astype("float32")
    distances, indices = index.search(np.array([query_embedding]), top_k)

    results = []
    for idx in indices[0]:
        row = df.iloc[idx]
        # Process Test Type into a list of full names
        test_types = str(row['Test Type'])
        test_type = [test_type_map.get(abbrev.strip(), abbrev.strip()) for abbrev in test_types.split()]

        results.append({
            "Assessment Name": f"[{row['Individual Test Solutions']}]({row['URL']})",
            "Description": row['Description'],  # Added Description column
            "Remote Testing": row['Remote Testing (y/n)'],
            "Adaptive/IRT": row['Adaptive/IRT (y/n)'],
            "Duration": row['Assessment Length'],
            "Test Type": test_type  # Added Test Type as a list
        })

    st.markdown("### 📋 Top Recommendations")
    st.dataframe(pd.DataFrame(results))

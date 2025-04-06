import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import streamlit as st

# Load data and models
try:
    st.write("Loading CSV...")
    df = pd.read_csv("shl_catalog_detailed.csv")
    st.write("Loading FAISS...")
    index = faiss.read_index("shl_assessments_index.faiss")
    st.write("Loading SentenceTransformer...")
    # Explicitly fetch model with cache
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='./cache')
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
    query_embedding = model.encode([query])[0].astype("float32")
    distances, indices = index.search(np.array([query_embedding]), top_k)

    results = []
    for idx in indices[0]:
        row = df.iloc[idx]
        results.append({
            "Assessment Name": f"[{row['Individual Test Solutions']}]({row['URL']})",
            "Remote Testing": row['Remote Testing (y/n)'],
            "Adaptive/IRT": row['Adaptive/IRT (y/n)'],
            "Duration": row['Assessment Length']
        })

    st.markdown("### 📋 Top Recommendations")
    st.dataframe(pd.DataFrame(results))

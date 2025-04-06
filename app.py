# streamlit_app.py

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import streamlit as st

# Load the CSV
df = pd.read_csv("shl_catalog_detailed.csv")

# Load FAISS index
index = faiss.read_index("shl_assessments_index.faiss")

# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

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

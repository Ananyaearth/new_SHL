# SHL Assessment Recommendation Engine

## Introduction

This project aims to build a **SHL Assessment Recommendation Engine** using **Streamlit**, **FAISS**, **Sentence Transformers**, and **Pandas**. The recommendation engine provides personalized SHL assessment recommendations based on a user's input, such as a job description, skill, or role. The system performs semantic similarity matching and uses pre-indexed embeddings for fast retrieval.

## Objective

The goal of the project is to recommend the most relevant SHL assessments based on user-provided input. The assessments are selected based on the semantic similarity of their descriptions to the query, leveraging advanced machine learning techniques such as **Sentence Transformers** for text embedding and **FAISS** for fast nearest neighbor search.

## Technologies Used

- **Streamlit**: A Python framework for building interactive web applications.
- **Pandas**: For data manipulation and analysis (e.g., reading and processing the CSV file containing SHL assessment data).
- **NumPy**: For numerical computations and array manipulation.
- **FAISS**: A library for efficient similarity search, used here for fast nearest neighbor search over the pre-indexed embeddings.
- **Sentence Transformers**: A library for generating sentence embeddings, which allows text queries (e.g., job descriptions) to be represented as fixed-size vectors for similarity comparison.

## Workflow

### 1. Data Loading
The application loads the SHL assessment data from a CSV file (`shl_catalog_detailed.csv`) and a pre-built **FAISS** index (`shl_assessments_index.faiss`). The **Sentence Transformer** model (`all-MiniLM-L6-v2`) is also loaded to encode user queries into embeddings.

### 2. FAISS Index
The **FAISS index** (`shl_assessments_index.faiss`) contains the embeddings of SHL assessments and is used for efficient nearest neighbor search. **FAISS** enables fast similarity searches by indexing the embeddings in high-dimensional space.

### 3. User Input
Users can input a job description, skill, or role in the provided text field. A slider allows the user to select the number of top recommendations they wish to receive (from 1 to 10).

### 4. Embedding Generation
The user’s input is passed through the **Sentence Transformer** model to generate an embedding (a vector representation of the input text). This embedding is used for similarity comparison with the embeddings of SHL assessments stored in the FAISS index.

### 5. Similarity Search
The query embedding is compared against the embeddings in the **FAISS index**. The FAISS index performs a fast search and returns the top-k most similar assessments based on their distance from the query embedding.

### 6. Display Results
The system displays a dataframe of the top-k recommended assessments, including:
- **Assessment Name**: A clickable link to the individual test solution.
- **Remote Testing**: Whether remote testing is supported for the assessment.
- **Adaptive/IRT**: Whether the assessment uses adaptive techniques or Item Response Theory (IRT).
- **Duration**: The length of the assessment.

### 7. Error Handling
The app includes error handling to provide feedback in case of any issues during the loading process or while using the application.

## Features

- **Real-time Recommendations**: The system recommends the most relevant SHL assessments based on user input, which can be a job description, skill, or role.
- **Adjustable Output**: Users can control the number of recommendations (from 1 to 10) they receive.
- **Efficient Search with FAISS**: **FAISS** is used to efficiently search through a large number of assessment embeddings, ensuring fast and accurate recommendations.

## FAISS Index

The **FAISS index** (`shl_assessments_index.faiss`) is a pre-built index containing the embeddings of SHL assessments. These embeddings are used to search for the most relevant assessments based on semantic similarity to the input query. The use of FAISS enables fast similarity searches over large datasets, making the system both efficient and scalable.

## User Interface

The Streamlit app provides the following key components:

- **Title**: "SHL Assessment Recommendation Engine" with an engaging emoji.
- **Markdown Text**: A brief introduction explaining the tool and how users can interact with it.
- **Text Input**: A field for users to input a job description, skill, or role.
- **Slider**: A slider to specify the number of top recommendations (1 to 10).
- **Results Display**: A dataframe showing the top recommendations with details about the assessments.

## Challenges and Solutions

1. **Efficient Embedding Search**:
   - **Problem**: Searching for similar items in a large dataset can be slow.
   - **Solution**: The **FAISS** library was used to index the embeddings of SHL assessments and perform fast nearest neighbor searches, enabling real-time recommendations even with large datasets.

2. **Model Performance**:
   - **Problem**: The Sentence Transformer model needed to generate high-quality embeddings for accurate recommendations.
   - **Solution**: The **Sentence Transformer** model (`all-MiniLM-L6-v2`) was chosen for its ability to create accurate embeddings for sentences, allowing for effective similarity comparisons.

3. **Error Handling**:
   - **Problem**: Handling potential issues during the data loading or model inference process.
   - **Solution**: The app includes robust error handling to display helpful messages if any resources fail to load.

## Future Improvements

1. **Expand Dataset**: Incorporating more SHL assessments and additional metadata will improve the accuracy and variety of the recommendations.
2. **User Feedback**: Adding a mechanism for users to rate the relevance of recommendations could enhance the model over time.
3. **Advanced Search Filters**: Implementing additional search filters, such as difficulty level, industry-specific assessments, or skill categories, could provide more tailored results.

## Conclusion

The **SHL Assessment Recommendation Engine** provides a fast, efficient, and scalable solution for recommending SHL assessments based on user input. The system makes use of **FAISS** for fast nearest neighbor search and **Sentence Transformers** for high-quality sentence embeddings. By hosting the application on **Streamlit Cloud**, it is easily accessible for users to get real-time recommendations.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Ananyaearth/new_SHL.git
   cd new_SHL
Install the necessary dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py
Visit the app in your browser at the provided local address.

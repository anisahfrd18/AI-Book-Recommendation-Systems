import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="📚 AI Book Recommendation System",
    layout="wide"
)

st.title("📚 AI Book Recommendation System")
st.caption("Search • Filters • Explainable AI • Chat-based Recommender")

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(
        "data/Book.csv",
        dtype=str,
        low_memory=False
    )
    df.fillna("", inplace=True)

    df["Year-Of-Publication"] = pd.to_numeric(
        df["Year-Of-Publication"], errors="coerce"
    )

    df["combined_features"] = (
        df["Book-Title"] + " " +
        df["Book-Author"] + " " +
        df["Publisher"]
    )

    return df

books = load_data()

# --------------------------------------------------
# TF-IDF Vectorizer
# --------------------------------------------------
@st.cache_resource
def build_tfidf(df):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )
    matrix = vectorizer.fit_transform(df["combined_features"])
    return vectorizer, matrix

tfidf_vectorizer, tfidf_matrix = build_tfidf(books)

# --------------------------------------------------
# Explainable AI
# --------------------------------------------------
def explain_recommendation(book_idx, rec_idx, top_k=5):
    feature_names = tfidf_vectorizer.get_feature_names_out()

    base_vec = tfidf_matrix[book_idx].toarray()[0]
    rec_vec = tfidf_matrix[rec_idx].toarray()[0]

    base_terms = set(feature_names[i] for i in base_vec.argsort()[-top_k:])
    rec_terms = set(feature_names[i] for i in rec_vec.argsort()[-top_k:])

    common = base_terms.intersection(rec_terms)
    return ", ".join(common) if common else "Similar topic & writing style"

# --------------------------------------------------
# Recommendation Engine
# --------------------------------------------------
def recommend_books(book_title, top_n=5):
    if book_title not in books["Book-Title"].values:
        return pd.DataFrame()

    book_idx = books[books["Book-Title"] == book_title].index[0]

    similarity = cosine_similarity(
        tfidf_matrix[book_idx],
        tfidf_matrix
    ).flatten()

    indices = similarity.argsort()[::-1][1:top_n+1]

    results = []
    for idx in indices:
        results.append({
            "Book Title": books.iloc[idx]["Book-Title"],
            "Author": books.iloc[idx]["Book-Author"],
            "Publisher": books.iloc[idx]["Publisher"],
            "Year": books.iloc[idx]["Year-Of-Publication"],
            "Why Recommended": explain_recommendation(book_idx, idx)
        })

    return pd.DataFrame(results)

# --------------------------------------------------
# CHAT-BASED RECOMMENDER (UNIQUE FEATURE)
# --------------------------------------------------
def chat_recommend(user_query, top_n=5):
    query = user_query.lower()

    # Extract author intent
    for author in books["Book-Author"].unique():
        if author.lower() in query:
            return books[
                books["Book-Author"] == author
            ][["Book-Title", "Publisher", "Year-Of-Publication"]].head(top_n)

    # Extract publisher intent
    for pub in books["Publisher"].unique():
        if pub.lower() in query:
            return books[
                books["Publisher"] == pub
            ][["Book-Title", "Book-Author", "Year-Of-Publication"]].head(top_n)

    # TF-IDF semantic match
    query_vec = tfidf_vectorizer.transform([user_query])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()

    indices = similarity.argsort()[::-1][:top_n]

    return books.iloc[indices][
        ["Book-Title", "Book-Author", "Publisher", "Year-Of-Publication"]
    ]

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------
st.sidebar.header("🔍 Filters")

author_filter = st.sidebar.selectbox(
    "Author",
    ["All"] + sorted(books["Book-Author"].unique())
)

publisher_filter = st.sidebar.selectbox(
    "Publisher",
    ["All"] + sorted(books["Publisher"].unique())
)

filtered_books = books.copy()

if author_filter != "All":
    filtered_books = filtered_books[
        filtered_books["Book-Author"] == author_filter
    ]

if publisher_filter != "All":
    filtered_books = filtered_books[
        filtered_books["Publisher"] == publisher_filter
    ]

# --------------------------------------------------
# SEARCH & RECOMMEND
# --------------------------------------------------
st.subheader("🔎 Search & Recommend")

selected_book = st.selectbox(
    "Select a Book",
    sorted(books["Book-Title"].unique())
)

if st.button("📖 Recommend Similar Books"):
    recs = recommend_books(selected_book)
    st.dataframe(recs, use_container_width=True)

# --------------------------------------------------
# CHAT SECTION
# --------------------------------------------------
st.subheader("💬 Chat-Based Book Recommender")

user_input = st.text_input(
    "Ask me anything (e.g. 'Suggest books by Amy Tan' or 'history books')"
)

if st.button("🤖 Get Recommendations"):
    if user_input.strip() == "":
        st.warning("Please type a question.")
    else:
        chat_results = chat_recommend(user_input)
        st.success("Here are some recommendations:")
        st.dataframe(chat_results, use_container_width=True)

# --------------------------------------------------
# POPULAR BOOKS
# --------------------------------------------------
st.subheader("🔥 Popular Books")

st.dataframe(
    books.head(10)[
        ["Book-Title", "Book-Author", "Publisher", "Year-Of-Publication"]
    ],
    use_container_width=True
)

# --------------------------------------------------
# BROWSE BOOKS
# --------------------------------------------------
st.subheader("📚 Browse Books")

st.dataframe(
    filtered_books[
        ["Book-Title", "Book-Author", "Publisher", "Year-Of-Publication"]
    ],
    use_container_width=True
)

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==================================================================================================
# --- Configuration and Data Loading ---
# ==================================================================================================

# Set page configuration for a wide layout
st.set_page_config(layout="wide")

# --- Load the pre-trained model and data ---
# We use st.cache_data to load the data only once, which significantly improves performance.
@st.cache_data
def load_data():
    """
    Loads the pickled model and data files.
    
    Returns:
        tuple: A tuple containing the trained KNN model, the user-item matrix, and the books dataframe.
    """
    try:
        with open('model_knn.pkl', 'rb') as f:
            model_knn = pickle.load(f)
        with open('user_item_matrix.pkl', 'rb') as f:
            user_item_matrix = pickle.load(f)
        with open('df_books.pkl', 'rb') as f:
            df_books = pickle.load(f)
        return model_knn, user_item_matrix, df_books
    except FileNotFoundError:
        st.error("Model or data files not found. Please run the training script first.")
        # Return None for all variables to prevent TypeError
        return None, None, None
    except Exception as e:
        st.error(f"An error occurred while loading the model or data: {e}")
        # Return None for all variables to prevent TypeError
        return None, None, None

# Load the necessary components
model_knn, user_item_matrix, df_books = load_data()

# ==================================================================================================
# --- Helper Functions and UI Components ---
# ==================================================================================================

# Function to get book recommendations from the model
def get_book_recommendations(book_title: str, user_item_matrix: pd.DataFrame, model_knn, k: int = 5):
    """
    Generates book recommendations based on a given book title using the trained KNN model.
    """
    if book_title not in user_item_matrix.index:
        return pd.DataFrame()

    book_index = user_item_matrix.index.get_loc(book_title)
    distances, indices = model_knn.kneighbors(
        user_item_matrix.iloc[book_index, :].values.reshape(1, -1),
        n_neighbors=k + 1
    )
    
    recommended_books = []
    for i in range(1, len(distances.flatten())):
        recommended_title = user_item_matrix.index[indices.flatten()[i]]
        distance = distances.flatten()[i]
        
        # Look up the book details from the original books dataframe
        book_info = df_books[df_books['Book-Title'] == recommended_title].iloc[0]
        
        recommended_books.append({
            'Book-Title': book_info['Book-Title'],
            'Book-Author': book_info['Book-Author'],
            'Image-URL-M': book_info['Image-URL-M'],
            'Similarity': 1 - distance
        })
    
    return pd.DataFrame(recommended_books)

def display_recommendations(df: pd.DataFrame):
    """
    Displays book recommendations in an aesthetic card layout.
    """
    if df.empty:
        st.write("No recommendations available.")
        return

    # Filter out books with missing image URLs
    df = df[df['Image-URL-M'].notna()]
    
    # Use a grid layout for recommendations
    num_items = len(df)
    cols = st.columns(min(num_items, 5)) # Adjust column count for larger screens

    for i, item in enumerate(df.to_dict('records')):
        with cols[i % 5]:
            # Use a styled container for each book card
            st.markdown(
                f"""
                <div class="book-card">
                    <img src="{item['Image-URL-M']}" alt="{item['Book-Title']}">
                    <div class="book-details">
                        <b>{item['Book-Title']}</b>
                        <small>by {item['Book-Author']}</small>
                        <p class="similarity">Similarity: {item['Similarity']:.2f}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True
            )


# ==================================================================================================
# --- Streamlit App Layout and Logic ---
# ==================================================================================================

# Inject custom CSS for a more aesthetic UI
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;700&family=Lato:wght@300;400&display=swap');

body {
    background-color: #111418; /* Dark but soft */
    color: #e0e0e0; /* Light grey text for contrast */
    font-family: 'Lato', sans-serif;
    margin: 0;
    padding: 0;
}

/* ------------------------- HEADER SECTION ------------------------- */

.main-header {
    text-align: center;
    color: #ffffff;
    font-family: 'Cormorant Garamond', serif;
    font-size: 4rem;
    font-weight: 700;
    margin-top: 60px;
    margin-bottom: 10px;
    text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.6);
}

.subheader {
    text-align: center;
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.25rem;
    color: #aab4c2;
    margin-bottom: 50px;
    max-width: 800px;
    margin-left: auto;
    margin-right: auto;
    line-height: 1.6;
}

/* ------------------------- INPUT SECTION ------------------------- */

.stSelectbox label, h3 {
    font-weight: 700;
    color: #ffffff;
    font-size: 1.3rem;
    margin-bottom: 10px;
    display: block;
}

.stSelectbox > div {
    border-radius: 10px;
    background-color: #1e1e24;
    border: 1px solid #3a3f47;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
    color: #ffffff;
}

.stSelectbox > div:hover {
    border-color: #1f8ef1;
    box-shadow: 0 0 0 2px rgba(31, 142, 241, 0.3);
}

/* ------------------------- BUTTON ------------------------- */

.stButton > button {
    border-radius: 10px;
    border: none;
    background-color: #1f8ef1;
    color: #ffffff;
    padding: 14px 30px;
    font-size: 1.2rem;
    font-weight: bold;
    margin-top: 30px;
    box-shadow: 0 4px 12px rgba(31, 142, 241, 0.3);
    transition: all 0.3s ease-in-out;
}

.stButton > button:hover {
    background-color: #156ab7;
    transform: translateY(-3px);
    box-shadow: 0 6px 18px rgba(31, 142, 241, 0.45);
}

.stButton > button:active {
    background-color: #0f4c81;
}

/* ------------------------- BOOK CARD ------------------------- */

.book-card {
    border-radius: 14px;
    padding: 20px;
    margin: 15px;
    text-align: center;
    background-color: #1c1e22;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.book-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.35);
}

.book-card img {
    border-radius: 10px;
    object-fit: cover;
    height: 260px;
    width: 100%;
    margin-bottom: 10px;
}

.book-details b {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.25rem;
    color: #f1f1f1;
}

.book-details small {
    font-style: italic;
    color: #8898aa;
    font-size: 0.9rem;
}

.similarity {
    font-size: 0.9rem;
    color: #adb5bd;
    margin-top: 5px;
    font-family: 'Lato', sans-serif;
}

</style>
""", unsafe_allow_html=True)


# ==================================================================================================
# --- Main App Logic (Render UI) ---
# ==================================================================================================
# Check if all components loaded successfully before rendering the UI
if model_knn is not None and user_item_matrix is not None and isinstance(df_books, pd.DataFrame) and not user_item_matrix.empty:
    st.markdown(f'<h1 class="main-header">BookWorm Recommender</h1>', unsafe_allow_html=True)

    # Use a container for the input elements
    with st.container():
        st.subheader("Select a book you like:")
        book_titles = user_item_matrix.index.unique().tolist()
        book_titles.sort()
        selected_book = st.selectbox(
            "Choose a book:",
            options=book_titles,
            index=book_titles.index('1984') if '1984' in book_titles else 0
        )
    
    # Button to trigger the recommendation process
    st.write("---")
    if st.button("Get Recommendations", use_container_width=True):
        if selected_book:
            with st.spinner("Finding your next read..."):
                recommendations_df = get_book_recommendations(selected_book, user_item_matrix, model_knn)
            
            if not recommendations_df.empty:
                st.subheader(f"Because you liked '{selected_book}', you might also enjoy:")
                display_recommendations(recommendations_df)
            else:
                st.info("No recommendations found for this book. Please try another one.")
        else:
            st.warning("Please select a book to get recommendations.")
else:
    st.error("Application could not be loaded. Please ensure all data files and the model are correctly saved and accessible.")

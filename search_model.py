import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords from NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Sample product data
products = [
    {"id": 1, "name": "Red T-Shirt", "description": "A bright red t-shirt made of cotton.",
        "color": "red", "category": "clothing"},
    {"id": 2, "name": "Blue Jeans", "description": "Comfortable blue jeans.",
        "color": "blue", "category": "clothing"},
    {"id": 3, "name": "Green Hat", "description": "A stylish green hat.",
        "color": "green", "category": "accessories"},
    {"id": 4, "name": "Red Shoes", "description": "Red running shoes.",
        "color": "red", "category": "footwear"},
    {"id": 5, "name": "Yellow Jacket", "description": "A yellow rain jacket.",
        "color": "yellow", "category": "clothing"}
]


def preprocess(text):
    # Remove punctuation
    text = re.sub(r'\W', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text



# Preprocess product descriptions and names
for product in products:
    product['processed'] = preprocess(
        product['description'] + ' ' + product['name'] + ' ' + product['color'] + ' ' + product['category'])



# Create a list of all processed product texts
corpus = [product['processed'] for product in products]



# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)


def search_products(query):
    # Preprocess the query
    query_processed = preprocess(query)
    # Convert the query to a vector
    query_vec = vectorizer.transform([query_processed])
    # Calculate cosine similarity between query and product vectors
    similarity = cosine_similarity(query_vec, X).flatten()
    # Get top results based on similarity scores
    top_indices = similarity.argsort()[-5:][::-1]  # Top 5 results
    # Retrieve and print the top products
    results = [products[i] for i in top_indices]
    return results



if __name__ == "__main__":
    query = input("Enter search query: ")
    results = search_products(query)
    print(f"\nSearch results for '{query}':")
    for result in results:
        print(
            f"Product ID: {result['id']}, Name: {result['name']}, Description: {result['description']}")

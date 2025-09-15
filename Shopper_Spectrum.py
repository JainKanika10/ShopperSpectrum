import streamlit as st
import pandas as pd
import joblib

# --- Streamlit UI ---
st.set_page_config(page_title="Product & Customer Insights", layout="wide") 
# --- Load models and data ---
@st.cache_resource
def load_models():
    product_similarity = pd.read_pickle('product_similarity.pkl')
    scaler = joblib.load('scaler.joblib')
    kmeans = joblib.load('kmeans_model.joblib')
    return product_similarity, scaler, kmeans

product_similarity_df, scaler, kmeans = load_models()

# --- Cluster label mapping ---
cluster_labels = {
    0: 'High-Value',
    1: 'Regular',
    2: 'Occasional',
    3: 'At-Risk'
}

# --- Functions ---
def get_similar_products(product_name, top_n=5):
    if product_name not in product_similarity_df.index:
        return None
    similar_scores = product_similarity_df[product_name].sort_values(ascending=False).drop(product_name)
    return similar_scores.head(top_n).index.tolist()

def predict_cluster(recency, frequency, monetary):
    input_scaled = scaler.transform([[recency, frequency, monetary]])
    cluster_num = kmeans.predict(input_scaled)[0]
    return cluster_labels.get(cluster_num, "Unknown Cluster")




# Sidebar Navigation
page = st.sidebar.radio("Navigation", ["🏠 Home", "📦 Product Recommendation", "🧑‍💼 Customer Segmentation"])

st.sidebar.markdown("---")

# --- 🏠 Home ---
if page == "🏠 Home":
    st.title("📱 Product & Customer Insights App")
    st.header("Welcome!")
    st.markdown("""
        This app helps you:
        
        1. 🔍 **Find similar products** using collaborative filtering (based on what other customers bought).
        
        2. 📊 **Segment customers** into groups like High-Value, At-Risk, and Regular based on their purchase behavior.
        
        ---
        Use the sidebar to get started!
    """)

# --- 📦 Product Recommendation ---
elif page == "📦 Product Recommendation":
    st.title("📦 Product Recommendation")
    product_input = st.text_input("Enter Product Name:")

    if st.button("Get Recommendations"):
        if product_input.strip():
            recommendations = get_similar_products(product_input.strip())
            if recommendations is None:
                st.error(f"❌ Product '{product_input}' not found.")
            else:
                st.success("✅ Top 5 Similar Products:")
                for i, prod in enumerate(recommendations, 1):
                    st.markdown(f"""
                    <div style="border:1px solid #ddd; padding:10px; border-radius:5px; margin-bottom:10px; background-color:#f9f9f9;">
                        <strong>{i}. {prod}</strong>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter a product name.")

# --- 🧑‍💼 Customer Segmentation ---
elif page == "🧑‍💼 Customer Segmentation":
    st.title("🧑‍💼 Customer Segmentation")

    recency = st.number_input("Recency (in days)", min_value=0, max_value=1000, value=30)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0, max_value=1000, value=5)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, max_value=100000.0, value=100.0, format="%.2f")

    if st.button("Predict Cluster"):
        cluster = predict_cluster(recency, frequency, monetary)
        st.success(f"🧠 Predicted Customer Segment: **{cluster}**")

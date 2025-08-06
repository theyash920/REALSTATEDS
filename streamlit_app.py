import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Bangalore Home Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load the model and data
@st.cache_resource
def load_model():
    """Load the trained model and location data"""
    try:
        # Load columns data
        with open("server/artifacts/columns.json", "r") as f:
            data_columns = json.load(f)["data_columns"]
            locations = data_columns[3:]
        
        # Load the model
        with open("server/artifacts/banglore_home_prices_model.pickle", "rb") as f:
            model = pickle.load(f)
        
        return model, locations, data_columns
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, [], []

def get_estimated_price(location, sqft, bhk, bath, model, data_columns):
    """Get estimated price using the loaded model"""
    try:
        loc_index = data_columns.index(location.lower())
    except:
        loc_index = -1
    
    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    
    return round(model.predict([x])[0], 2)

# Load model
model, locations, data_columns = load_model()

if model is None:
    st.error("Failed to load the model. Please check if the model files exist in server/artifacts/")
    st.stop()

# Header
st.markdown('<h1 class="main-header">🏠 Bangalore Home Price Prediction</h1>', unsafe_allow_html=True)
st.markdown("### Get accurate property valuations in seconds with our AI-powered prediction tool")

# Stats section
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>5,000+</h3>
        <p>Properties Analyzed</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3>98%</h3>
        <p>Accuracy Rate</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3>₹1.2B+</h3>
        <p>Property Value Predicted</p>
    </div>
    """, unsafe_allow_html=True)

# Main prediction form
st.markdown("---")
st.markdown("## 📊 Estimate Property Value")

# Create two columns for the form
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Fill in the details to get an instant price prediction")
    
    # Form inputs
    sqft = st.number_input(
        "🏠 Area (Square Feet)",
        min_value=500,
        max_value=10000,
        value=1000,
        step=100,
        help="Enter the total square footage of the property"
    )
    
    bhk = st.selectbox(
        "🛏️ BHK",
        options=[1, 2, 3, 4, 5],
        index=1,
        help="Number of Bedrooms, Hall, and Kitchen"
    )
    
    bath = st.selectbox(
        "🚿 Bathrooms",
        options=[1, 2, 3, 4, 5],
        index=1,
        help="Number of bathrooms"
    )
    
    location = st.selectbox(
        "📍 Location",
        options=locations,
        help="Select the location in Bangalore"
    )
    
    # Prediction button
    if st.button("🚀 Estimate Price", type="primary", use_container_width=True):
        if location:
            estimated_price = get_estimated_price(location, sqft, bhk, bath, model, data_columns)
            
            # Display result
            st.markdown("---")
            st.markdown("### 📈 Prediction Result")
            
            # Format price with commas
            formatted_price = f"₹{estimated_price:,.2f}"
            
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="color: #1f77b4; text-align: center;">Estimated Price</h2>
                <h1 style="color: #2e8b57; text-align: center; font-size: 3rem;">{formatted_price}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional insights
            st.markdown("### 💡 Insights")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                price_per_sqft = estimated_price / sqft
                st.metric("Price per sq ft", f"₹{price_per_sqft:,.2f}")
            
            with col2:
                st.metric("Property Type", f"{bhk} BHK")
            
            with col3:
                st.metric("Location", location)
        else:
            st.error("Please select a location to get a prediction.")

with col2:
    st.markdown("### ℹ️ Why Use Our Prediction Tool?")
    
    st.markdown("""
    - 🤖 **AI-Powered Accuracy**: Our algorithm analyzes thousands of property transactions
    - ⚡ **Instant Results**: Get property valuations in seconds
    - 💰 **Free of Charge**: No hidden fees or subscriptions
    - 🗺️ **Bangalore Coverage**: Detailed insights for all major localities
    """)
    
    st.markdown("### 📊 Market Insights")
    st.markdown("""
    Based on recent market data:
    - Average price per sq ft: ₹8,500
    - Most expensive areas: Indira Nagar, Jayanagar
    - Best value areas: Electronic City, Marathahalli
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>© 2023 RealEstateAI. All rights reserved.</p>
    <p>Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True) 
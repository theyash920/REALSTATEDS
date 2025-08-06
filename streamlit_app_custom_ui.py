import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Bangalore Home Price Prediction | RealEstateAI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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

# Custom CSS to match your existing design
st.markdown("""
<style>
/* Import Font Awesome */
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

/* Base Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

body {
    background-color: #f8f9fa;
    color: #333;
    line-height: 1.6;
}

/* Header Styles */
.main .block-container {
    padding: 0;
    max-width: 100%;
}

/* Hero Section */
.hero-section {
    background: linear-gradient(rgba(26, 42, 108, 0.85), rgba(26, 42, 108, 0.85)), 
                url('https://images.unsplash.com/photo-1564013799919-ab600027ffc6?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80');
    background-size: cover;
    background-position: center;
    height: 500px;
    display: flex;
    align-items: center;
    padding: 0 5%;
    position: relative;
    margin-bottom: 50px;
}

.hero-content {
    max-width: 650px;
    color: white;
    z-index: 2;
}

.hero-content h1 {
    font-size: 3.2rem;
    margin-bottom: 20px;
    font-weight: 700;
    line-height: 1.2;
}

.hero-content p {
    font-size: 1.3rem;
    margin-bottom: 30px;
    max-width: 600px;
}

.stats {
    display: flex;
    gap: 40px;
    margin-top: 30px;
}

.stats div {
    text-align: center;
}

.stats span {
    display: block;
    font-size: 2.5rem;
    font-weight: 700;
    color: #a5dc86;
    margin-bottom: 5px;
}

.stats p {
    font-size: 0.9rem;
    opacity: 0.9;
}

/* Prediction Section */
.prediction-section {
    padding: 60px 5%;
    background: white;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 50px;
    align-items: start;
}

.form-container {
    background: white;
    padding: 40px;
    border-radius: 15px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
}

.form-container h2 {
    font-size: 2rem;
    margin-bottom: 10px;
    color: #1a2a6c;
    display: flex;
    align-items: center;
    gap: 15px;
}

.form-container p {
    color: #666;
    margin-bottom: 30px;
    font-size: 1.1rem;
}

.form-group {
    margin-bottom: 25px;
}

.form-group label {
    display: block;
    margin-bottom: 8px;
    font-weight: 600;
    color: #333;
    font-size: 1.1rem;
    display: flex;
    align-items: center;
    gap: 10px;
}

.switch-field {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}

.switch-field input[type="radio"] {
    display: none;
}

.switch-field label {
    padding: 12px 20px;
    background: #f8f9fa;
    border: 2px solid #e9ecef;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.3s ease;
    font-weight: 500;
    color: #666;
    margin: 0;
}

.switch-field label:hover {
    background: #e9ecef;
    border-color: #1a2a6c;
}

.switch-field input:checked + label {
    background: #1a2a6c;
    color: white;
    border-color: #1a2a6c;
}

.area, .location {
    width: 100%;
    padding: 15px;
    border: 2px solid #e9ecef;
    border-radius: 8px;
    font-size: 1rem;
    transition: all 0.3s ease;
    background: white;
}

.area:focus, .location:focus {
    outline: none;
    border-color: #1a2a6c;
    box-shadow: 0 0 0 3px rgba(26, 42, 108, 0.1);
}

.submit {
    width: 100%;
    padding: 15px;
    background: linear-gradient(135deg, #1a2a6c, #3a5fc5);
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 1.1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    margin-top: 20px;
}

.submit:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(26, 42, 108, 0.3);
}

.submit:active {
    transform: translateY(0);
}

.result {
    background: linear-gradient(135deg, #28a745, #20c997);
    color: white;
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    margin-top: 30px;
    box-shadow: 0 10px 30px rgba(40, 167, 69, 0.3);
}

.result h3 {
    font-size: 1.5rem;
    margin-bottom: 15px;
    font-weight: 600;
}

.result .price {
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 10px;
}

/* Info Panel */
.info-panel {
    background: #f8f9fa;
    padding: 30px;
    border-radius: 15px;
    height: fit-content;
}

.info-panel h3 {
    font-size: 1.5rem;
    margin-bottom: 20px;
    color: #1a2a6c;
}

.info-panel ul {
    list-style: none;
    margin-bottom: 30px;
}

.info-panel li {
    padding: 15px 0;
    border-bottom: 1px solid #e9ecef;
    display: flex;
    align-items: flex-start;
    gap: 15px;
}

.info-panel li:hover {
    background: rgba(26, 42, 108, 0.05);
    border-radius: 8px;
    padding: 15px;
    margin: 0 -15px;
}

.info-panel i {
    font-size: 1.5rem;
    color: #1a2a6c;
    margin-top: 2px;
    min-width: 20px;
}

.info-panel h4 {
    font-size: 1.1rem;
    margin-bottom: 5px;
    color: #333;
}

.info-panel p {
    color: #666;
    font-size: 0.95rem;
    line-height: 1.5;
}

.testimonial {
    background: white;
    padding: 25px;
    border-radius: 12px;
    margin-top: 30px;
    border-left: 4px solid #1a2a6c;
}

.testimonial p {
    font-style: italic;
    color: #666;
    margin-bottom: 15px;
    font-size: 1rem;
}

.author {
    display: flex;
    align-items: center;
    gap: 15px;
}

.author img {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    object-fit: cover;
}

.author div strong {
    display: block;
    color: #333;
    font-weight: 600;
}

.author div span {
    color: #666;
    font-size: 0.9rem;
}

/* Responsive Design */
@media (max-width: 768px) {
    .container {
        grid-template-columns: 1fr;
        gap: 30px;
    }
    
    .hero-content h1 {
        font-size: 2.5rem;
    }
    
    .stats {
        gap: 20px;
    }
    
    .stats span {
        font-size: 2rem;
    }
}

/* Hide Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="background: linear-gradient(135deg, #1a2a6c, #3a5fc5); color: white; padding: 15px 5%; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 15px rgba(0, 0, 0, 0.1); position: sticky; top: 0; z-index: 100;">
    <div style="display: flex; align-items: center; gap: 15px;">
        <i class="fas fa-home" style="font-size: 2.5rem; color: #a5dc86;"></i>
        <div>
            <span style="font-size: 1.8rem; font-weight: 700;">RealEstate</span><span style="font-size: 1.8rem; font-weight: 700; color: #a5dc86;">AI</span>
            <p style="font-size: 0.9rem; opacity: 0.8; margin-top: 3px;">Bangalore Property Experts</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section">
    <div class="hero-content">
        <h1>Bangalore Home Price Prediction</h1>
        <p>Get accurate property valuations in seconds with our AI-powered prediction tool</p>
        <div class="stats">
            <div>
                <span>5,000+</span>
                <p>Properties Analyzed</p>
            </div>
            <div>
                <span>98%</span>
                <p>Accuracy Rate</p>
            </div>
            <div>
                <span>₹1.2B+</span>
                <p>Property Value Predicted</p>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Prediction Section
st.markdown("""
<div class="prediction-section">
    <div class="container">
        <div class="form-container">
            <h2><i class="fas fa-calculator"></i> Estimate Property Value</h2>
            <p>Fill in the details to get an instant price prediction</p>
""", unsafe_allow_html=True)

# Form inputs using Streamlit components
col1, col2 = st.columns([2, 1])

with col1:
    # Area input
    sqft = st.number_input(
        "🏠 Area (Square Feet)",
        min_value=500,
        max_value=10000,
        value=1000,
        step=100,
        help="Enter the total square footage of the property"
    )
    
    # BHK selection
    bhk = st.selectbox(
        "🛏️ BHK",
        options=[1, 2, 3, 4, 5],
        index=1,
        help="Number of Bedrooms, Hall, and Kitchen"
    )
    
    # Bathrooms selection
    bath = st.selectbox(
        "🚿 Bathrooms",
        options=[1, 2, 3, 4, 5],
        index=1,
        help="Number of bathrooms"
    )
    
    # Location selection
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
            <div class="result">
                <h3>Estimated Price</h3>
                <div class="price">{formatted_price}</div>
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
    st.markdown("""
    <div class="info-panel">
        <h3>ℹ️ Why Use Our Prediction Tool?</h3>
        <ul>
            <li>
                <i class="fas fa-brain"></i>
                <div>
                    <h4>AI-Powered Accuracy</h4>
                    <p>Our algorithm analyzes thousands of property transactions for precise valuations</p>
                </div>
            </li>
            <li>
                <i class="fas fa-bolt"></i>
                <div>
                    <h4>Instant Results</h4>
                    <p>Get property valuations in seconds, not days</p>
                </div>
            </li>
            <li>
                <i class="fas fa-rupee-sign"></i>
                <div>
                    <h4>Free of Charge</h4>
                    <p>No hidden fees or subscriptions required</p>
                </div>
            </li>
            <li>
                <i class="fas fa-map-marked-alt"></i>
                <div>
                    <h4>Bangalore Coverage</h4>
                    <p>Detailed insights for all major Bangalore localities</p>
                </div>
            </li>
        </ul>
        
        <div class="testimonial">
            <p>"This tool helped me price my property correctly, resulting in a quick sale at the best possible price!"</p>
            <div class="author">
                <img src="https://randomuser.me/api/portraits/women/65.jpg" alt="Customer">
                <div>
                    <strong>Priya Sharma</strong>
                    <span>Property Owner, Koramangala</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Close the container divs
st.markdown("""
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; color: #666; padding: 30px; background: #f8f9fa; margin-top: 50px;">
    <p>© 2023 RealEstateAI. All rights reserved.</p>
    <p>Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True) 
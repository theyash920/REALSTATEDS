# 🏠 Bangalore Home Price Prediction

An AI-powered web application that predicts property prices in Bangalore using Machine Learning. Built with Flask backend and modern HTML/CSS/JavaScript frontend.

## ✨ Features

- **AI-Powered Predictions**: Uses a trained Linear Regression model for accurate price estimates
- **Real-time Results**: Get instant property valuations in seconds
- **Bangalore Coverage**: Supports 200+ locations across Bangalore
- **User-friendly Interface**: Modern, responsive design with intuitive form
- **Free to Use**: No hidden fees or subscriptions required

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- pip (Python package installer)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MLPROJECTREGRESSION
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the server**
   ```bash
   cd server
   python server.py
   ```

4. **Open the application**
   - Open `client/app1.html` in your web browser
   - Or navigate to the client folder and double-click `app1.html`

## 📁 Project Structure

```
MLPROJECTREGRESSION/
├── server/
│   ├── server.py          # Flask backend server
│   ├── util.py            # ML model utilities
│   └── artifacts/
│       ├── columns.json   # Feature columns data
│       └── banglore_home_prices_model.pickle  # Trained ML model
├── client/
│   ├── app1.html         # Main web interface
│   ├── app1.js           # Frontend JavaScript
│   └── app1.css          # Styling
├── requirements.txt       # Python dependencies
└── README.md            # This file
```

## 🛠️ Technology Stack

### Backend
- **Flask**: Web framework for API endpoints
- **Flask-CORS**: Cross-origin resource sharing
- **scikit-learn**: Machine Learning library
- **numpy**: Numerical computing
- **pandas**: Data manipulation

### Frontend
- **HTML5**: Structure and semantics
- **CSS3**: Modern styling with animations
- **JavaScript**: Interactive functionality
- **jQuery**: AJAX requests and DOM manipulation
- **Font Awesome**: Icons

### Machine Learning
- **Model**: Linear Regression
- **Features**: Area (sqft), BHK, Bathrooms, Location
- **Training Data**: Bangalore property dataset
- **Accuracy**: High precision predictions

## 🔧 API Endpoints

### GET `/get_location_names`
Returns all available locations in Bangalore.

**Response:**
```json
{
  "status": "success",
  "locations": ["Electronic City", "Rajaji Nagar", ...]
}
```

### POST `/predict_home_price`
Predicts property price based on input parameters.

**Request Body:**
```json
{
  "total_sqft": 1000,
  "bhk": 2,
  "bath": 2,
  "location": "Electronic City"
}
```

**Response:**
```json
{
  "status": "success",
  "estimated_price": 83.5
}
```

## 🎯 How to Use

1. **Fill the Form**:
   - Enter the area in square feet
   - Select number of bedrooms (BHK)
   - Select number of bathrooms
   - Choose location from dropdown

2. **Get Prediction**:
   - Click "Estimate Price" button
   - View the predicted price in Indian Rupees

3. **Result Display**:
   - Price is shown in formatted Indian currency (₹)
   - Results appear instantly after calculation

## 📊 Model Information

- **Algorithm**: Linear Regression
- **Features**: 200+ location encodings + area, BHK, bathrooms
- **Training**: Based on real Bangalore property data
- **Output**: Price in lakhs (converted to rupees for display)

## 🔍 Troubleshooting

### Common Issues

1. **Server not starting**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check if port 5000 is available

2. **Predictions not working**
   - Verify server is running on `http://127.0.0.1:5000`
   - Check browser console for error messages
   - Ensure all form fields are filled

3. **Locations not loading**
   - Check server logs for errors
   - Verify `artifacts/columns.json` exists

### Debug Mode

The server runs in debug mode by default. Check the terminal for:
- Server startup messages
- Request/response logs
- Error details

## 🎨 Customization

### Styling
- Modify `client/app1.css` for visual changes
- Update colors, fonts, and layout as needed

### Functionality
- Edit `client/app1.js` for frontend behavior
- Modify `server/server.py` for backend logic

### Model
- Replace `artifacts/banglore_home_prices_model.pickle` with your own model
- Update `server/util.py` for different prediction logic

## 📈 Performance

- **Response Time**: < 1 second for predictions
- **Accuracy**: High precision based on training data
- **Scalability**: Can handle multiple concurrent requests
- **Reliability**: Robust error handling and validation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👥 Authors

- **Your Name** - Initial work

## 🙏 Acknowledgments

- Bangalore property dataset providers
- scikit-learn community
- Flask development team

## 📞 Support

For issues and questions:
- Check the troubleshooting section above
- Review server logs for error details
- Ensure all dependencies are properly installed

---

**Happy Predicting! 🏠💰**

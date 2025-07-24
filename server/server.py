from flask import Flask, request, jsonify 
from flask_cors import CORS
import util

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    try:
        locations = util.get_location_names()
        response = jsonify({
            "status": "success",
            "locations": locations
        })
        return response
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/predict_home_price', methods=['POST', 'OPTIONS'])
def predict_home_price():
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({"status": "success"})
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response
        
    try:
        data = request.get_json()
        total_sqft = float(data.get('total_sqft', 0))
        location = data.get('location', '')
        bhk = int(data.get('bhk', 0))
        bath = int(data.get('bath', 0))
        
        estimated_price = util.get_estimated_price(location, total_sqft, bhk, bath)
        
        response = jsonify({
            "status": "success",
            "estimated_price": estimated_price
        })
        return response
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    print("Starting Python Flask Server")
    util.load_saved_artifacts()
    app.run(debug=True)
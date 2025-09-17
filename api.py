import requests
import json
import os

# --- 1. Define the API endpoint and the data to send ---
# The URL of your locally running Flask API
api_url = 'http://127.0.0.1:5000/predict'

# The date you want to get a forecast for
# You can change this date as needed
date_to_predict = '2025-05-01'

data = {
    "date": date_to_predict
}

# --- 2. Make the API call ---
try:
    print(f"Attempting to connect to API at {api_url}...")
    
    # Send a POST request with the JSON data
    response = requests.post(api_url, json=data)

    # Check if the request was successful
    if response.status_code == 200:
        # Get the JSON data from the response
        result = response.json()
        
        print("\nAPI Call Successful!")
        print("-" * 30)
        print(f"Date: {result['date']}")
        print(f"Forecasted Price: {result['forecasted_price']}")
        print("-" * 30)

    else:
        # Handle non-200 status codes (e.g., 400, 500)
        print(f"\nError: API returned status code {response.status_code}")
        print(f"Response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("\nError: Could not connect to the API server.")
    print("Please ensure your 'api.py' script is running in another terminal.")
    
except Exception as e:
    # Handle any other unexpected errors
    print(f"\nAn unexpected error occurred: {str(e)}")

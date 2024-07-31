from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
import json
app = Flask(__name__)
from HOUSE_PRICE_PREDICTION import predict_saleprice  # Import the prediction function

def load_model_and_columns():
    with open('columns.json', 'r') as file:
        model_columns = json.load(file)
    with open('HOUSE_PRICE_PREDICTION.pickle', 'rb') as file:
        model = pickle.load(file)
    return model, model_columns

model, model_columns = load_model_and_columns()

# Load actual data
train_data_path = 'train.csv'  # Make sure this file is in the correct path
df_train = pd.read_csv(train_data_path)

# Extract neighborhoods and house IDs
neighborhood_house_ids = df_train[['Neighborhood', 'Id']].groupby('Neighborhood')['Id'].apply(list).to_dict()

@app.route('/')
def home():
    return render_template('index.html', neighborhoods=neighborhood_house_ids.keys())

@app.route('/get_house_ids', methods=['GET'])
def get_house_ids():
    neighborhood = request.args.get('neighborhood')
    house_ids = neighborhood_house_ids.get(neighborhood, [])
    return {'house_ids': house_ids}


@app.route('/predict', methods=['POST'])
def predict():
    house_id = int(request.form['house_id'])
    
    try:
        # Use the prediction function from HOUSE_PRICE_PREDICTION.py
        predicted_price = predict_saleprice(house_id)
        predicted_price_formatted = f"${predicted_price:,.2f}"
    except Exception as e:
        print(f"Exception: {e}")
        predicted_price_formatted = "Error in prediction"
    
    return render_template('index.html', neighborhoods=neighborhood_house_ids.keys(), predicted_price=predicted_price_formatted)

if __name__ == '__main__':
    app.run(debug=True)



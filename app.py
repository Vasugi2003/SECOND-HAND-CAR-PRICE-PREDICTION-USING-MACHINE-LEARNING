# from flask import Flask, request, render_template
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# import pickle

# app = Flask(__name__)

# # Load the trained model
# with open("model.pkl", "rb") as f:
#     model = pickle.load(f)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get input from the user
#     present_price = float(request.form['Present_Price'])
#     kms_driven = float(request.form['Kms_Driven'])
#     fuel_type = request.form['Fuel_Type']
#     seller_type = request.form['Seller_Type']
#     transmission = request.form['Transmission']
#     owner = float(request.form['Owner'])
#     year = int(request.form['Year'])

#     # Convert categorical features to numerical
#     fuel_type_numeric = 0 if fuel_type == 'Petrol' else 1 if fuel_type == 'Diesel' else 2
#     seller_type_numeric = 0 if seller_type == 'Dealer' else 1
#     transmission_numeric = 0 if transmission == 'Manual' else 1

#     # Create a DataFrame from the user input
#     input_data = pd.DataFrame({
#         "Present_Price": [present_price],
#         "Kms_Driven": [kms_driven],
#         "Fuel_Type": [fuel_type_numeric],
#         "Seller_Type": [seller_type_numeric],
#         "Transmission": [transmission_numeric],
#         "Owner": [owner],
#         "Year": [year]
#     })

#     # Preprocess the input data
#     scaler = StandardScaler()
#     input_data[['Present_Price', 'Kms_Driven', 'Year']] = scaler.fit_transform(input_data[['Present_Price', 'Kms_Driven', 'Year']])

#     # Make predictions
#     prediction = model.predict(input_data)

#     # Print the predicted selling price
#     prediction_text = f'Predicted Selling Price: {prediction[0] * 10000}'

#     return render_template('index.html', prediction_text=prediction_text)

# if __name__ == "__main__":
#     app.run(debug=True)












import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def main():
    st.title("Car Price Prediction")

    # Get input from the user
    present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, format="%.2f")
    kms_driven = st.number_input("Kms Driven", min_value=0)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox("Owner", [0, 1, 2, 3])
    year = st.number_input("Year", min_value=1900, max_value=2024, step=1)

    if st.button("Predict"):
        # Convert categorical features to numerical
        fuel_type_numeric = 0 if fuel_type == 'Petrol' else 1 if fuel_type == 'Diesel' else 2
        seller_type_numeric = 0 if seller_type == 'Dealer' else 1
        transmission_numeric = 0 if transmission == 'Manual' else 1

        # Create a DataFrame from the user input
        input_data = pd.DataFrame({
            "Present_Price": [present_price],
            "Kms_Driven": [kms_driven],
            "Fuel_Type": [fuel_type_numeric],
            "Seller_Type": [seller_type_numeric],
            "Transmission": [transmission_numeric],
            "Owner": [owner],
            "Year": [year]
        })

        # Preprocess the input data
        scaler = StandardScaler()
        input_data[['Present_Price', 'Kms_Driven', 'Year']] = scaler.fit_transform(input_data[['Present_Price', 'Kms_Driven', 'Year']])

        # Make predictions
        prediction = model.predict(input_data)

        # Print the predicted selling price
        prediction_text = f'Predicted Selling Price: ₹ {prediction[0] * 100000}'

        st.success(prediction_text)

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("AgrcultureDataset.csv")
df['Production'] = pd.to_numeric(df['Production'], errors='coerce')
df = df.dropna()

# Clean the 'Crop' column to remove extra spaces
df['Crop'] = df['Crop'].str.strip()

# Automatically extract all unique crop names from the dataset
unique_crop_names = df['Crop'].unique().tolist()

# Fit the LabelEncoder with the unique crop names from the dataset
label_encoder = LabelEncoder()
label_encoder.fit(unique_crop_names)

# Encode the crop names in the dataset
df['Crop'] = label_encoder.transform(df['Crop'])

# Feature selection and target variable
X = df[['Crop_Year', 'Area', 'Crop']]
y = df['Production']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and fit the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

@app.route('/')
def home():
    # Pass the list of crop names to the frontend for the dropdown
    return render_template('index.html', crops=unique_crop_names)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            year = int(request.form['year'])
            area = float(request.form['area'])
            crop_name = request.form['crop_name'].strip()  # Remove extra spaces

            # Validate input
            if area < 0:
                return render_template('index.html', prediction='Area must be a positive number.')
            if year < 2000 or year > 2025:
                return render_template('index.html', prediction='Please enter a valid year.')

            # Encode the crop name input to match the training data
            if crop_name not in unique_crop_names:
                return render_template('index.html', prediction=f'Error: Crop {crop_name} not recognized.')
            
            crop_name_encoded = label_encoder.transform([crop_name])[0]

            # Create input DataFrame
            input_data = pd.DataFrame([[year, area, crop_name_encoded]], columns=['Crop_Year', 'Area', 'Crop'])

            # Make the prediction
            prediction = rf_model.predict(input_data)[0]
            return render_template('index.html', crops=unique_crop_names, prediction=f'Predicted Production: {prediction:.2f}')
        except Exception as e:
            return render_template('index.html', crops=unique_crop_names, prediction=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)




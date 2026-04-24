# House-Price-Predictor-XGBoost-Streamlit
House Price Prediction system built with Python, XGBoost, and Streamlit. Demonstrates data cleaning, feature engineering, and ML pipeline development.

**Project Overview**
This project is an end-to-end Machine Learning web application designed to predict house prices based on property features. It demonstrates the complete data science lifecycle, including data preprocessing, feature engineering, model training, and an interactive web dashboard.

The application utilizes an XGBoost Regressor trained on real estate data to provide instant price predictions in INR (Indian Rupees).

**Key Features**

Interactive User Interface: Built with Streamlit for seamless user input and visualization.

Data Processing: Custom parsing functions to handle complex formats (e.g., converting "1.5 Cr" to numerical values).

Model Performance: Real-time price estimation with R-squared metrics and feature importance visualization.

Modular Architecture: Separation of concerns between the training pipeline (ml_pipeline.py) and the web application (app.py).

**Tech Stack**
Languages: Python
Libraries: Pandas, NumPy, Scikit-Learn, XGBoost
Framework: Streamlit

**Tools**: Git, GitHub

**Project Structure**

├── app.py # Streamlit application script

├── ml_pipeline.py # Data processing and model training logic

├── train_evaluate.py # Script for model evaluation

├── requirements.txt # Required dependencies

├── house_prices.zip # Compressed dataset

└── README.md # Project documentation


## Installation and Usage

To run this project locally, follow these steps:

1. **Clone the repository**
   ```bash
   git clone https://github.com/ismailiqbal2773/House-Price-Predictor-XGBoost-Streamlit.git
   cd House-Price-Predictor-XGBoost-Streamlit
   
**Install dependencies**
pip install -r requirements.txt

**Run the application**
The application reads data directly from the compressed zip file.
streamlit run app.py

**Model Details**
Algorithm: XGBoost Regressor
Target Variable: House Price (INR)
Features: Location, Carpet Area, BHK, Transaction Type, Furnishing, etc.

**Author**
Muhammad Ismail Iqbal

GitHub: @ismailiqbal2773

**License**
This project is licensed under the MIT License.


# House Price Prediction Using Linear Regression

## Overview
This project implements a house price prediction model using Linear Regression, a fundamental machine learning algorithm. The goal is to predict house prices based on size of the house. The project is built with Python and uses popular libraries such as Pandas, NumPy, and Scikit-learn.

## Features
- Data preprocessing and cleaning
- Exploratory data analysis (EDA) to understand feature relationships
- Linear Regression model training and evaluation
- Visualization of results using Matplotlib/Seaborn
- streamlit for frontend

## Prerequisites
Before running the project, ensure you have the following installed:
- Python 3.x
- Required libraries (listed in `requirements.txt`)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Ravulaponnakrishnavamsi/house-price-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd house-price-prediction
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
The Dataset contains the 
-Size of the House
-Price of the house


## Usage
1. Ensure the dataset is placed in the project directory.
2. Run the main script:
   ```bash
   python house_price_prediction.py
   ```
3. The script will:
   - Load and preprocess the data
   - Train the Linear Regression model
   - Output predictions and performance metrics (e.g., Mean Squared Error, R² Score)

## Project Structure
```
house-price-prediction/
│
├── data/                 # Folder for dataset (e.g., house_prices.csv)
├── house_price_prediction.py  # Main script
├── requirements.txt      # List of dependencies
└── README.md             # Project documentation
```

## Results
The model outputs predictions and evaluation metrics. Visualizations (e.g., scatter plots of predicted vs. actual prices) are generated to assess performance.

## Dependencies
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- ploty

Install them using:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn ploty streamlit
```

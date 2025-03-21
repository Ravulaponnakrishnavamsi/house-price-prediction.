import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Generate synthetic house data
def genarate_house_data(n_samples=100):
    np.random.seed(50)
    size = np.random.normal(1400, 50, n_samples)
    price = size * 50 + np.random.normal(0, 50, n_samples)
    return pd.DataFrame({'size': size, 'price': price})

# Train a linear regression model
def train_model():
    df = genarate_house_data(n_samples=100)
    X = df[['size']]  # Features (house size)
    y = df['price']   # Target (price)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    # Optionally, calculate the mean absolute error
    predictions = model.predict(x_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f'Mean Absolute Error: {mae}')
    
    return model

# Main Streamlit app
def main():
    st.title('House Price Prediction App')
    st.write('Enter the size of the house, and this app will predict the price.')

    model = train_model()

    # Get user input
    size = st.number_input('House Size (sq ft)', min_value=100, max_value=10000, value=1000)

    if st.button('Predict Price'):
        predicted_price = model.predict([[size]])[0]  # Predict and get the first result
        st.success(f'Estimated Price: ${predicted_price:,.2f}')

        # Generate sample data for visualization
        df = genarate_house_data()

        # Create scatter plot with prediction
        fig = px.scatter(df, x='size', y='price', title='Size vs House Price')
        fig.add_scatter(x=[size], y=[predicted_price], mode='markers', marker=dict(size=15, color='red'), name='Prediction')
        st.plotly_chart(fig)

if __name__ == '__main__':
    main()

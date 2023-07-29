import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def readFinancialData(filePath):
    return pd.read_csv(filePath)

def preprocessData(data):
    # Perform data cleaning, normalization, feature engineering, etc.
    # For this example, we'll assume that the data is already cleaned and preprocessed.
    return data

def trainModel(data):
    # Assuming we have 'Date' and 'Revenue' columns in the CSV file
    X = data['Date'].values.reshape(-1, 1)
    y = data['Revenue'].values

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2

def generateReports(data, model, mse, r2):
    # Plotting the historical financial data
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data['Revenue'], label='Actual Revenue')
    plt.xlabel('Date')
    plt.ylabel('Revenue')
    plt.title('Historical Revenue Data')
    plt.legend()
    plt.savefig('historicalRevenue.png')
    plt.close()

    # Generating a forecast for the next quarter (assuming quarterly data)
    lastDate = pd.to_datetime(data['Date'].max())
    nextQuarter = lastDate + pd.DateOffset(months=3)
    forecastDate = pd.date_range(lastDate, nextQuarter, freq='M')
    forecastX = np.arange(len(data), len(data) + len(forecastDate)).reshape(-1, 1)
    forecastY = model.predict(forecastX)

    # Creating a DataFrame for the forecast
    forecastData = pd.DataFrame({'Date': forecastDate, 'Forecasted Revenue': forecastY})

    # Plotting the forecasted revenue
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data['Revenue'], label='Actual Revenue')
    plt.plot(forecastData['Date'], forecastData['Forecasted Revenue'], label='Forecasted Revenue')
    plt.xlabel('Date')
    plt.ylabel('Revenue')
    plt.title('Revenue Forecast for the Next Quarter')
    plt.legend()
    plt.savefig('forecastedRevenue.png')
    plt.close()

    # Generate a report with metrics and forecast details
    report = f"""Financial Report
-----------------
- Number of data points: {len(data)}
- Mean Squared Error: {mse:.2f}
- R-squared: {r2:.2f}

Forecasted Revenue for the Next Quarter:
{forecastData.to_string(index=False)}
"""
    with open('financialReport.txt', 'w') as reportFile:
        reportFile.write(report)

def main(filePath):
    # Step 1: Read financial data from CSV
    data = readFinancialData(filePath)

    # Step 2: Preprocess the data
    data = preprocessData(data)

    # Step 3: Train the machine learning model
    model, mse, r2 = trainModel(data)

    # Step 4: Generate financial reports
    generateReports(data, model, mse, r2)

if __name__ == "__main__":
    # Provide the path to the CSV file containing financial data
    csvFilePath = "path/to/your/financial_data.csv"
    main(csvFilePath)
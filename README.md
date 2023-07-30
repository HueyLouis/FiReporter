# FiReport

# README
This Python script demonstrates how to train a linear regression model using scikit-learn library and generate financial reports based on the trained model.

## Dependencies
- pandas
- numpy
- matplotlib
- scikit-learn

## Usage
1. Provide the path to the CSV file containing financial data in the **csvFilePath** variable
2. Run the script.

## Functions
**readFinancialData(filePath)**
This function reads financial data from a CSV file and returns a pandas DataFrame

**preprocessData(data)**
This function performs data cleaning, normalization, feature engineering, etc. on the input data. For example, it assumes that the data is already cleaned and processed.

**trainModel(data)**
This function trains a linear regression model on the input data and returns the trained model, mean squared error, and R-squared.

**generateReports(data, model, mse, r2)**
This function generates financial reports based on the input data, trained model, mean squared error, and R-squared. It plots the historical financial data, generates a forecast for the next quarter, and creates a report with metrics and forecast details.

**main(filePath)**
This function is the main function that calls the above functions in sequence to generate financial reports based on the input CSV file.

## Example
```python
csvFilePath = "path/to/your/financialData.csv"
main(csvFilePath)
```

## References
- Scikit-learn's train_test_split()
- Scikit-learn LinearRegression class
- Linear Regression in Python with Scikit-Learn
- Train/Test Split and Cross Validation in Python

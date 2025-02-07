# Housing Prices Regression Project

## Overview
This project is a machine learning regression model to predict housing prices using data from the Kaggle **Housing Prices Competition for Kaggle Learn Users**. The dataset includes various features such as lot size, number of rooms, and neighborhood details to predict the final house price.

## Dataset
The dataset consists of:
- **Training data**: Contains housing features and their corresponding sale prices.
- **Test data**: Contains only housing features, and the model predicts the sale prices.
- **Lasso Regression Output**: A CSV file containing predictions from a Lasso regression model.

## Project Workflow
1. **Data Preprocessing:**
   - Handle missing values
   - Feature encoding (One-Hot Encoding, Label Encoding)
   - Feature scaling (Standardization, Normalization)
2. **Exploratory Data Analysis (EDA):**
   - Visualizing distributions and correlations
   - Identifying and handling outliers
3. **Model Selection & Training:**
   - Linear Regression
   - Ridge & Lasso Regression
   - Decision Trees & Random Forest
   - Gradient Boosting Models (XGBoost, LightGBM)
4. **Model Evaluation:**
   - Root Mean Squared Error (RMSE)
   - Cross-validation to improve generalization
5. **Submission & Performance:**
   - Generating and submitting final predictions to Kaggle

## Results
The best model achieved an RMSE score of **17,659.86**, as per the Kaggle leaderboard.

## Installation
To run the project locally:
```bash
# Clone the repository
git clone https://github.com/EmmanuelA0428/HousingPricesRegression.git
cd HousingPricesRegression

# Install dependencies
pip install -r requirements.txt
```

## Usage
Run the Jupyter notebook for data preprocessing, training, and evaluation:
```bash
jupyter notebook housing_prices_final.ipynb
```

## Files
- `housing_prices_final.ipynb` - Final Jupyter notebook with the complete workflow
- `housing_prices_draft.ipynb` - Draft version of the analysis
- `lasso_output.csv` - Predictions from Lasso Regression
- `README.md` - Project documentation

## Future Improvements
- Fine-tuning hyperparameters for better model accuracy
- Experimenting with deep learning models for price prediction
- Adding more feature engineering techniques

## Author
Emmanuel Appiah

## Acknowledgments
This project was inspired by the Kaggle Housing Prices Competition for Kaggle Learn Users. Special thanks to the Kaggle community for valuable insights and datasets.


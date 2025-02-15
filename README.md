# Redeem Coupon Prediction

## Project Overview

This project aims to predict whether customers will redeem coupons received across various channels. The insights generated will help retailers design better-targeted marketing strategies and optimize promotional campaigns.

## Problem Statement

Coupon-based discount marketing is a common strategy to attract and retain customers. However, predicting coupon redemption behavior is crucial for assessing campaign effectiveness. The goal is to:

- Analyze customer insights based on demographics, income, and purchasing behavior.
- Build a machine learning model to predict the probability of coupon redemption.
- Rank customers based on their propensity to redeem.
- Identify the most effective types of coupons and campaigns.
- Recommend strategies to improve coupon redemption rates.

## Dataset

The dataset consists of multiple files:

- **train.csv** – Contains customer-coupon interactions and redemption status.
- **campaign_data.csv** – Information on different promotional campaigns.
- **coupon_item_mapping.csv** – Mapping of coupons to items eligible for discounts.
- **customer_demographics.csv** – Customer information (age range, income, family size, etc.).
- **customer_transaction_data.csv** – Customer purchase transactions.
- **item_data.csv** – Information about items sold by the retailer.
- **test.csv** – Data for which coupon redemption status needs to be predicted.

## Data Preprocessing

- **Handling Missing Data**:
  - `no_of_children`: Assumed as zero.
  - `marital_status`: Missing values set to "Unknown".
  - `age_range`: Missing values set to "Unknown".
  - `rented`: Filled missing values with 0 (majority class).
  - `family_size`: Missing values set to "Unknown".

- **Feature Engineering**:
  - Aggregated transaction data to extract customer and item purchase behavior.
  - Merged transaction data with item and customer datasets.
  - Created additional features to analyze customer spending habits.

## Exploratory Data Analysis (EDA)

Key findings:

- **Highly Imbalanced Dataset**: Only 0.94% of the data represents coupon redemption.
- **Customer Insights**:
  - Age group **46-55** had the highest coupon redemptions.
  - Higher-income customers are more likely to redeem coupons.
  - Small family sizes (1-2 members) show more redemption.
- **Product & Campaign Insights**:
  - Coupons for categories like *Garden, Salads, Travel, Vegetables, and Restaurants* were never redeemed.
  - Grocery (22%) and Pharmaceutical (14%) categories had the highest redemption rates.
  - **X-type campaigns** are more profitable than Y-type campaigns.

## Machine Learning Models

The following models were used:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

### Model Evaluation:

- Models were tuned for **recall** to maximize the detection of positive classes.
- Feature importance was analyzed to improve model interpretability.

## Key Recommendations

1. **Change Y-type campaigns to X-type campaigns** to improve coupon redemption.
2. **Make coupons applicable across multiple brands** to increase the likelihood of redemption.
3. **Encourage loyal/high-spending customers** by enhancing coupon distribution channels.
4. **Focus on categories with higher redemption rates** (e.g., Grocery, Pharmaceuticals).
5. **Discard ineffective coupons** (e.g., Garden, Travel, Salads).

## Installation & Usage

1. **Clone the repository**:
   ```sh
   git clone https://github.com/yourusername/redeem-coupon-prediction.git
   cd redeem-coupon-prediction

2. **Install dependencies**:
    ```sh
    pip install -r requirements.txt

3. **Run the notebook**:
    ```sh
    jupyter notebook data_analysis.ipynb

4. **Run the model script**:
    ```sh
    python model.py
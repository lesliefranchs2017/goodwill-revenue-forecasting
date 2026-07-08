# Revenue Forecasting Platform
### Comparative Machine Learning Using TensorFlow and PyTorch

## Overview

This project demonstrates the design and implementation of an end-to-end revenue forecasting platform using Python, TensorFlow, and PyTorch.

Beginning with publicly available financial and economic data, the project follows a complete data engineering workflow including data acquisition, cleaning, validation, feature engineering, model development, evaluation, and business interpretation.

The same forecasting problem was independently implemented using both TensorFlow and PyTorch to compare predictive performance, training behavior, and framework implementation.

---

## Business Problem

Can publicly available financial and economic information be used to forecast organizational revenue?

To answer this question, a complete forecasting platform was developed using machine learning techniques and modern data engineering practices.

---

## Data Sources

The project combines multiple public data sources including:

- IRS Form 990 Financial Data
- Consumer Price Index (CPI)
- U.S. Unemployment Rate
- Crude Oil Prices
- Organizational Net Assets

A synthetic dataset containing **3,000 observations** was generated for model training and evaluation.

---

# Project Workflow

```
Public Financial Data
          │
          ▼
 Data Cleaning & Validation
          │
          ▼
 Feature Engineering
          │
          ▼
 Machine Learning
      ┌──────────────┐
      ▼              ▼
 TensorFlow      PyTorch
      └──────┬───────┘
             ▼
 Model Evaluation
             ▼
 Revenue Prediction
             ▼
 Business Interpretation
```

---

# Technologies

- Python
- TensorFlow
- PyTorch
- Pandas
- NumPy
- Scikit-Learn
- Statsmodels
- Matplotlib

---

# Machine Learning Models

## TensorFlow

- Sequential Neural Network
- Hidden Layers: 16 → 8 → 1
- Adam Optimizer
- Mean Squared Error (MSE)

## PyTorch

- Feedforward Neural Network
- Hidden Layers: 16 → 8 → 1
- Adam Optimizer
- Mean Squared Error (MSE)

Both implementations used:

- 3,000 observations
- Five input features
- 80/20 Train/Test Split
- Standardized input variables
- Independent evaluation using held-out test data

---

# Results

| Metric | PyTorch | TensorFlow |
|---------|---------:|-----------:|
| Dataset | 3,000 | 3,000 |
| Training Rows | 2,400 | 2,400 |
| Testing Rows | 600 | 600 |
| Features | 5 | 5 |
| Optimizer | Adam | Adam |
| Hidden Layers | 16 → 8 → 1 | 16 → 8 → 1 |
| Test MAE | $2.07M | $2.19M |
| Test RMSE | $2.61M | $2.76M |
| Test R² | **0.9691** | **0.9656** |
| Training Time | **18.9 sec** | **71.3 sec** |
| Future Revenue Prediction | $68.36M | $58.21M |

---

# Key Findings

- Successfully built an end-to-end revenue forecasting platform.
- Implemented the same forecasting problem using both TensorFlow and PyTorch.
- Both models achieved excellent predictive performance with **R² values above 0.96**.
- In this CPU-based experiment, PyTorch completed training approximately **3.8× faster** than TensorFlow.
- Future revenue predictions differed by approximately **15–17%**, illustrating that independently trained neural networks can converge to different—but equally reasonable—solutions while maintaining similar predictive accuracy.

---

# Skills Demonstrated

### Data Engineering

- Data Acquisition
- ETL
- Data Cleaning
- Data Validation
- Feature Engineering
- Predictive Analytics

### Machine Learning

- TensorFlow
- PyTorch
- Neural Networks
- Model Evaluation
- Performance Metrics

### Programming

- Python
- Pandas
- NumPy
- SQL
- Scikit-Learn
- Statsmodels

---

# Repository Contents

- TensorFlow Implementation
- PyTorch Implementation
- Revenue Forecast Dataset
- Presentation Slides
- Architecture Diagrams
- Project Documentation

---

# Future Enhancements

- Additional historical financial data
- Hyperparameter tuning
- Cross-validation
- Additional forecasting algorithms
- GPU performance comparison
- Interactive dashboard

---

# About the Author

**Leslie Franchs**

Master of Economics

Senior Data Migration & Integration Engineer

SQL Developer | Data Engineer | Machine Learning Portfolio Project

This project demonstrates how modern data engineering practices and machine learning can be combined to solve real-world business forecasting problems while comparing two leading deep learning frameworks under comparable experimental conditions.

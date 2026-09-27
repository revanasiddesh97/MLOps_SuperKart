# 🛒 SuperKart Automated Sales Forecasting Pipeline

An end-to-end MLOps regression pipeline designed to forecast total product sales across various retail outlet types (`Product_Store_Sales_Total`) for SuperKart. This repository automates the Machine Learning lifecycle—from feature engineering and hyperparameter optimization to continuous integration (CI via GitHub Actions), model tracking (MLflow), and continuous deployment (CD via Streamlit).

---

## 📌 Project Overview & Objectives

Accurate sales forecasting enables SuperKart to optimize inventory management, minimize stockouts, reduce carrying costs, and improve regional supply chain allocation.

* **Task:** Supervised Regression
* **Target Variable:** `Product_Store_Sales_Total`
* **Primary Evaluation Metrics:** Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), $R^2$ Score
* **Key Features:** Product MRP, Item Weight, Sugar Content, Store Size, Store Location Tier, and Store Type.

---

## 🛠️ Architecture & Tech Stack

* **Language:** Python 3.11
* **Machine Learning:** `scikit-learn`, `xgboost`
* **Model Tracking & Experimentation:** `mlflow`
* **CI/CD & Automation:** GitHub Actions
* **Interactive Web App Deployment:** Streamlit / Streamlit Community Cloud
* **Serialization & Versioning:** `joblib`

---

## 📂 Repository Structure

```text
mlops_superkart/
├── .github/
│   └── workflows/
│       └── ml_pipeline.yml         # GitHub Actions CI/CD workflow
├── super_cart_project/
│   ├── data/
│   │   └── SuperKart.csv           # Raw dataset
│   └── deployment/
│       ├── app.py                  # Streamlit web application
│       └── sales_model.joblib      # Serialized production pipeline
├── models/
│   └── sales_model.joblib          # Trained pipeline artifact
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation

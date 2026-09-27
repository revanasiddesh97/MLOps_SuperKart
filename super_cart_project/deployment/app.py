import os
import joblib
import pandas as pd
import streamlit as st

# Load the trained regression model committed by the pipeline
model_path = os.path.join(
    os.path.dirname(__file__), "super_cart_package_model_v1.joblib"
)
model = joblib.load(model_path)

# Streamlit UI Configuration
st.set_page_config(
    page_title="SuperKart Sales Predictor", page_icon="🛒", layout="centered"
)

st.title("🛒 SuperKart Total Sales Predictor")
st.write(
    "Enter the product and store features below to forecast total store sales (`Product_Store_Sales_Total`)."
)

# Layout: Split inputs into Product and Store sections
col1, col2 = st.columns(2)

with col1:
    st.subheader("📦 Product Attributes")
    Product_Weight = st.number_input(
        "Product Weight (kg)", min_value=1.0, max_value=30.0, value=12.65, step=0.1
    )
    Product_Sugar_Content = st.selectbox(
        "Sugar Content", ["Low Sugar", "Regular", "No Sugar"]
    )
    Product_Allocated_Area = st.slider(
        "Allocated Area Ratio",
        min_value=0.001,
        max_value=0.300,
        value=0.068,
        step=0.005,
        format="%.3f",
    )
    Product_Type = st.selectbox(
        "Product Category",
        [
            "Fruits and Vegetables",
            "Snack Foods",
            "Frozen Foods",
            "Dairy",
            "Household",
            "Baking Goods",
            "Canned",
            "Health and Hygiene",
            "Meat",
            "Soft Drinks",
            "Breads",
            "Hard Drinks",
            "Starchy Foods",
            "Breakfast",
            "Seafood",
            "Others",
        ],
    )
    Product_MRP = st.number_input(
        "Product MRP ($)", min_value=10.0, max_value=300.0, value=147.0, step=1.0
    )

with col2:
    st.subheader("🏬 Store Attributes")
    Store_Establishment_Year = st.number_input(
        "Establishment Year",
        min_value=1980,
        max_value=2026,
        value=2009,
        step=1,
    )
    Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
    Store_Location_City_Type = st.selectbox(
        "City Tier", ["Tier 1", "Tier 2", "Tier 3"]
    )
    Store_Type = st.selectbox(
        "Store Type",
        [
            "Supermarket Type1",
            "Supermarket Type2",
            "Departmental Store",
            "Food Mart",
        ],
    )

# ----------------------------
# Prepare Input DataFrame
# ----------------------------
input_data = pd.DataFrame(
    [
        {
            "Product_Weight": Product_Weight,
            "Product_Sugar_Content": Product_Sugar_Content,
            "Product_Allocated_Area": Product_Allocated_Area,
            "Product_Type": Product_Type,
            "Product_MRP": Product_MRP,
            "Store_Establishment_Year": Store_Establishment_Year,
            "Store_Size": Store_Size,
            "Store_Location_City_Type": Store_Location_City_Type,
            "Store_Type": Store_Type,
        }
    ]
)

# Make Prediction
st.markdown("---")
if st.button("🚀 Forecast Total Sales"):
    predicted_sales = model.predict(input_data)[0]

    # Render results with metric visualization
    st.success("Sales Forecast Complete!")
    st.metric(
        label="Predicted Product Store Sales Total",
        value=f"${predicted_sales:,.2f}",
    )

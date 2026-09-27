RAW_PATH = "super_cart_project/data/SuperKart.csv"


import pandas as pd

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Validate that the expected columns are present before registering it
expected_columns = [
    "Product_Id", "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_Type",
    "Product_MRP", "Store_Id", "Store_Establishment_Year", "Store_Size",
    "Store_Location_City_Type", "Store_Type", "Product_Store_Sales_Total"
]
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

# for data manipulation
import pandas as pd
# for building the preprocessing and modeling pipeline
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score # Changed to regression metrics
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# for model serialization and experiment tracking
import joblib
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("SuperKart-Prediction")

# Xtrain/Xtest/ytrain/ytest are downloaded from the previous job's artifact
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

numeric_features = [
    'Product_Weight', 'Product_Allocated_Area', 'Product_MRP', 'Store_Establishment_Year'
]
categorical_features = [
    'Product_Sugar_Content', 'Product_Type', 'Store_Size',
    'Store_Location_City_Type', 'Store_Type'
]

# Removed class_weight as it's not applicable for regression

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)
# Define base XGBoost model - Changed to XGBRegressor for regression task
xgb_model = xgb.XGBRegressor(random_state=42) # Removed scale_pos_weight

# Define hyperparameter grid - Changed prefixes for XGBRegressor
param_grid = {
    'xgbregressor__n_estimators': [50, 75, 100, 125, 150],
    'xgbregressor__max_depth': [2, 3, 4],
    'xgbregressor__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbregressor__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbregressor__learning_rate': [0.01, 0.05, 0.1],
    'xgbregressor__reg_lambda': [0.4, 0.5, 0.6],
}
# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning with GridSearchCV
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error') # Added scoring for regression
    grid_search.fit(Xtrain, ytrain)

    # Log every parameter combination tried during the search as a nested run,
    # so all experiments can be compared side by side in the MLflow UI
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_test_score", results["mean_test_score"][i])
            mlflow.log_metric("std_test_score", results["std_test_score"][i])

    # Log the best hyperparameters in the main run
    mlflow.log_params(grid_search.best_params_)

    # Store the best model
    best_model = grid_search.best_estimator_;

    # Removed classification_threshold and probability predictions

    # Make predictions on the training and test data
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    # Evaluation - Changed to regression metrics
    train_rmse = mean_squared_error(ytrain, y_pred_train)**0.5
    test_rmse = mean_squared_error(ytest, y_pred_test)**0.5
    train_r2 = r2_score(ytrain, y_pred_train)
    test_r2 = r2_score(ytest, y_pred_test)

    # Log metrics
    mlflow.log_metrics({
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_r2_score": train_r2,
        "test_r2_score": test_r2
    })

    # Save the model next to app.py so the Streamlit app can load it directly,
    # and log it as an MLflow artifact for traceability
    model_path = "super_cart_project/deployment/super_cart_package_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved to {model_path}")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split,GridSearchCV,KFold,cross_val_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

risk_data = pd.read_csv("extracted_data_stocks.csv")
risk_data.dropna(inplace=True)

# Feature Selection
features = ["Volatility", "Beta", "Max_Drawdown", "Sharpe_Ratio", "Average_sentiment_score"]

# Heuristic risk score
def heuristic_risk_score(row):
    return (
        0.3 * row["Volatility"] +
        0.3 * abs(row["Max_Drawdown"]) +
        0.2 * row["Beta"] +
        -0.1 * row["Sharpe_Ratio"] + 
        -0.1 * row["Average_sentiment_score"]
    )

risk_data["Heuristic_Risk_Score"] = risk_data.apply(heuristic_risk_score, axis=1)

# Normalize heuristic scores
scaler = MinMaxScaler(feature_range=(0, 1))
risk_data["Heuristic_Risk_Score"] = scaler.fit_transform(risk_data[["Heuristic_Risk_Score"]])

# Normalize features
risk_data_scaled = scaler.fit_transform(risk_data[features])

# Apply K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
risk_data["Cluster_Label"] = kmeans.fit_predict(risk_data_scaled)

# Dynamically map cluster labels to risk scores based on cluster centroids
cluster_centers = kmeans.cluster_centers_.mean(axis=1)
sorted_clusters = np.argsort(cluster_centers)
cluster_risk_mapping = {sorted_clusters[i]: (i + 1) / 6 for i in range(5)}  # 0.166, 0.333, 0.5, 0.667, 0.833 range
risk_data["Cluster_Risk_Score"] = risk_data["Cluster_Label"].map(cluster_risk_mapping)

# Weighted Average
risk_data["Final_Risk_Score"] = (0.6 * risk_data["Heuristic_Risk_Score"] + 0.4 * risk_data["Cluster_Risk_Score"])

#XGBoost modeling
features = ['Volatility', 'Beta', 'Sharpe_Ratio', 'Max_Drawdown', 'PE_Ratio', 'EPS', '50_MA_Last', '200_MA_Last', 'Number_of_news_headlines', 'Average_sentiment_score', 'Beta_1', 'Beta_2', 'Beta_3', 'Beta_6', 'PE_Ratio_1', 'Volatility_1', 'PE_Ratio_2', 'Volatility_2', 'PE_Ratio_3', 'Volatility_3', 'PE_Ratio_6', 'Volatility_6', 'Heuristic_Risk_Score', 'Cluster_Label', 'Cluster_Risk_Score', 'Final_Risk_Score', 'Volatility_Beta', 'Max_Drawdown_Sharpe']
# Feature Engineering
risk_data["Volatility_Beta"] = risk_data["Volatility"] * risk_data["Beta"]
risk_data["Max_Drawdown_Sharpe"] = risk_data["Max_Drawdown"] * risk_data["Sharpe_Ratio"]
features.extend(["Volatility_Beta", "Max_Drawdown_Sharpe"])

target = "Final_Risk_Score"

# risk_data.to_csv("extracted_data_stocks.csv")

# Prepare data
X = risk_data[features]
y = risk_data[target]

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10]
}

xgb_model = XGBRegressor(objective="reg:squarederror", random_state=42)
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_xgb = grid_search.best_estimator_

# Perform K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cross_val_r2 = cross_val_score(best_xgb, X_scaled, y, cv=kf, scoring='r2')
print(f"Cross-Validation R² Scores: {cross_val_r2}")
print(f"Average CV R²: {cross_val_r2.mean():.4f}")

# Make predictions
y_pred = best_xgb.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save model
best_xgb.save_model("xgboost_risk_model.json")



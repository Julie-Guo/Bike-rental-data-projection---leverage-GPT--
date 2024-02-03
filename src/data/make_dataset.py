import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns



bikeshare_data = pd.read_csv("../../data/raw/Bikeshare data.csv")
bikeshare_data.columns


# Data Preprocessing

# Convert 'Date' to datetime and extract day of the week
# Convert 'Date' to datetime with the correct format
bikeshare_data['Date'] = pd.to_datetime(bikeshare_data['Date'], format='%d/%m/%Y')
bikeshare_data['DayOfWeek'] = bikeshare_data['Date'].dt.dayofweek

# Separate the features and the target variable
X = pd.get_dummies(bikeshare_data.drop(['Rented Bike Count', 'Date'], axis=1))
y = bikeshare_data['Rented Bike Count']

# List of categorical and numerical features
categorical_features = ['Seasons', 'Holiday', 'Functioning Day', 'DayOfWeek']
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
bikeshare_data
# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Create and evaluate the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit the model
pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mse, r2



# train data on all 5 different moels and and find the best model based on score

def evaluate_regression_models(X_train, X_test, y_train, y_test):
    # Initialize models
    models = {
         'XGBoost Regressor': xgb.XGBRegressor(objective ='reg:squarederror', n_estimators=100, seed=0),
        'Decision Tree': DecisionTreeRegressor(random_state=0),
        'Support Vector Regressor': SVR(),
        'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=0),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=0)
    }

    # Dictionary to store results
    results = {}

    # Train and evaluate each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        results[name] = {'MSE': mse, 'R-squared': r2}

    return results

# Use the function
model_scores = evaluate_regression_models(X_train, X_test, y_train, y_test)

# Display the results
for model, scores in model_scores.items():
    print(f"{model} - MSE: {scores['MSE']}, R-squared: {scores['R-squared']}")


# perform grid search to tune the XGBoost model to find best hyperparameters

def xgboost_grid_search(X_train, y_train):
    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5]
    }

    # Initialize the XGBoost regressor
    xgb_model = XGBRegressor(objective ='reg:squarederror')

    # Set up GridSearchCV with 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=xgb_model, 
        param_grid=param_grid, 
        cv=5, 
        scoring=['neg_mean_squared_error', 'r2'],
        refit='neg_mean_squared_error', # Refit on the best MSE score
        verbose=1
    )

    # Perform the grid search
    grid_search.fit(X_train, y_train)

    # Return the best parameters and the corresponding scores
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_r2 = grid_search.cv_results_['mean_test_r2'][grid_search.best_index_]

    return best_params, best_score, best_r2

# Call the function with your training data
best_params, best_mse, best_r2 = xgboost_grid_search(X_train, y_train)
print("Best Parameters:", best_params)
print("Best MSE (Negated):", best_mse)
print("Best R2 Score:", best_r2)


# Create the XGBoost model with the best parameters
best_xgb_model = xgb.XGBRegressor(learning_rate=0.1, max_depth=7, min_child_weight=1, n_estimators=200, objective='reg:squarederror')

# Fit the model
best_xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = best_xgb_model.predict(X_test)

# In the Actual vs Predicted Values Plot, points closer to the diagonal line indicate better predictions.

# Actual vs Predicted Values Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values')
plt.grid(True)
plt.show()


# In the Residuals Plot, a random dispersion of points around the horizontal line (y=0) usually suggests a good fit. 
# Systematic patterns might indicate a problem with the model.

# Residuals Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.axhline(y=0, color='r', linestyle='-')
plt.grid(True)
plt.show()


# The Feature Importance Plot shows which features are most influential in the model's predictions.
# This can be useful for understanding the model's decision-making process and for feature selection in future modeling.

# Feature Importance Plot
feature_importances = best_xgb_model.feature_importances_
sorted_idx = np.argsort(feature_importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
sns.barplot(x=feature_importances[sorted_idx], y=X_train.columns[sorted_idx])
plt.show()

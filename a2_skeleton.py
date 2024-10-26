import os
import sys
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Read in the datasets using pandas
wine_red_df = pd.read_csv(os.path.join(os.path.dirname(sys.argv[0]), "winequality-red.csv"), sep=';')
wine_white_df = pd.read_csv(os.path.join(os.path.dirname(sys.argv[0]), "winequality-white.csv"), sep=';')

# Define features and target variable for red and white wines
red_features = wine_red_df.columns[:-1]  # All columns except 'quality'
white_features = wine_white_df.columns[:-1]  # All columns except 'quality'

# Prepare the datasets
wineRed_X = wine_red_df[red_features].values
wineRed_y = wine_red_df['quality'].values
wineWhite_X = wine_white_df[white_features].values
wineWhite_y = wine_white_df['quality'].values

# Split the data into training/testing sets
wineRed_X_train, wineRed_X_test, wineRed_y_train, wineRed_y_test = train_test_split(wineRed_X, wineRed_y, test_size=0.3, random_state=42)
wineWhite_X_train, wineWhite_X_test, wineWhite_y_train, wineWhite_y_test = train_test_split(wineWhite_X, wineWhite_y, test_size=0.3, random_state=42)

# Create linear regression object for red wine
regr_red = linear_model.LinearRegression()
regr_red.fit(wineRed_X_train, wineRed_y_train)
wineRed_y_pred = regr_red.predict(wineRed_X_test)

# Report performance for red wine model
print("Red Wine Model:")
print("Coefficients: \n", regr_red.coef_)
print("MSE: %.2f" % mean_squared_error(wineRed_y_test, wineRed_y_pred))
print("R^2: %.2f" % r2_score(wineRed_y_test, wineRed_y_pred))

# Create linear regression object for white wine
regr_white = linear_model.LinearRegression()
regr_white.fit(wineWhite_X_train, wineWhite_y_train)
wineWhite_y_pred = regr_white.predict(wineWhite_X_test)

# Report performance for white wine model
print("\nWhite Wine Model:")
print("Coefficients: \n", regr_white.coef_)
print("MSE: %.2f" % mean_squared_error(wineWhite_y_test, wineWhite_y_pred))
print("R^2: %.2f" % r2_score(wineWhite_y_test, wineWhite_y_pred))

# Third model (you can customize this part)
# For the third model, you can select features that consumers might have access to.
# Example: let's use just 'alcohol', 'sulphates', and 'pH'
consumer_features = [
    'alcohol',
    'sulphates',
    'density',
    'pH',
    'volatile acidity'
]
# Prepare consumer dataset
wineConsumer_X = wine_red_df[consumer_features].values  # Use red wine as an example
wineConsumer_y = wine_red_df['quality'].values

# Split the consumer data
wineConsumer_X_train, wineConsumer_X_test, wineConsumer_y_train, wineConsumer_y_test = train_test_split(wineConsumer_X, wineConsumer_y, test_size=0.3, random_state=42)

# Create linear regression object for consumer model
regr_consumer = linear_model.LinearRegression()
regr_consumer.fit(wineConsumer_X_train, wineConsumer_y_train)
wineConsumer_y_pred = regr_consumer.predict(wineConsumer_X_test)

# Report performance for consumer model
print("\nConsumer Model:")
print("Coefficients: \n", regr_consumer.coef_)
print("MSE: %.2f" % mean_squared_error(wineConsumer_y_test, wineConsumer_y_pred))
print("R^2: %.2f" % r2_score(wineConsumer_y_test, wineConsumer_y_pred))

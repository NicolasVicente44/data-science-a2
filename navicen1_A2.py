import os
import sys
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

wine_red_df = pd.read_csv(os.path.join(os.path.dirname(sys.argv[0]), "winequality-red.csv"), sep=';')
wine_white_df = pd.read_csv(os.path.join(os.path.dirname(sys.argv[0]), "winequality-white.csv"), sep=';')

red_features = wine_red_df.columns[:-1]
white_features = wine_white_df.columns[:-1]

wineRed_X = wine_red_df[red_features].values
wineRed_y = wine_red_df['quality'].values
wineWhite_X = wine_white_df[white_features].values
wineWhite_y = wine_white_df['quality'].values

wineRed_X_train, wineRed_X_test, wineRed_y_train, wineRed_y_test = train_test_split(wineRed_X, wineRed_y, test_size=0.3, random_state=42)
wineWhite_X_train, wineWhite_X_test, wineWhite_y_train, wineWhite_y_test = train_test_split(wineWhite_X, wineWhite_y, test_size=0.3, random_state=42)

regr_red = linear_model.LinearRegression()
regr_red.fit(wineRed_X_train, wineRed_y_train)
wineRed_y_pred = regr_red.predict(wineRed_X_test)

print("\n********************************************")
print("Red Wine Model:")
print("Coefficients: \n", regr_red.coef_)
print("MSE: %.2f" % mean_squared_error(wineRed_y_test, wineRed_y_pred))
print("R^2: %.2f" % r2_score(wineRed_y_test, wineRed_y_pred))

regr_white = linear_model.LinearRegression()
regr_white.fit(wineWhite_X_train, wineWhite_y_train)
wineWhite_y_pred = regr_white.predict(wineWhite_X_test)

print("\nWhite Wine Model:")
print("Coefficients: \n", regr_white.coef_)
print("MSE: %.2f" % mean_squared_error(wineWhite_y_test, wineWhite_y_pred))
print("R^2: %.2f" % r2_score(wineWhite_y_test, wineWhite_y_pred))

consumer_features = ['alcohol', 'sulphates', 'density', 'pH', 'volatile acidity']
wineConsumer_X = wine_red_df[consumer_features].values  
wineConsumer_y = wine_red_df['quality'].values

wineConsumer_X_train, wineConsumer_X_test, wineConsumer_y_train, wineConsumer_y_test = train_test_split(wineConsumer_X, wineConsumer_y, test_size=0.3, random_state=42)

regr_consumer = linear_model.LinearRegression()
regr_consumer.fit(wineConsumer_X_train, wineConsumer_y_train)
wineConsumer_y_pred = regr_consumer.predict(wineConsumer_X_test)

print("\nConsumer Model:")
print("Coefficients: \n", regr_consumer.coef_)
print("MSE: %.2f" % mean_squared_error(wineConsumer_y_test, wineConsumer_y_pred))
print("R^2: %.2f" % r2_score(wineConsumer_y_test, wineConsumer_y_pred))
print("********************************************")

dummy_data = [9.0, 0.5, 1.005, 3.2, 0.03]  
predicted_quality = regr_consumer.predict([dummy_data])

print(f"Given the dummy data features: {dummy_data}, the predicted quality is: {predicted_quality[0]:.2f}")

fig, axs = plt.subplots(3, 2, figsize=(12, 12))

axs[0, 0].scatter(wineRed_y_test, wineRed_y_pred, alpha=0.6)
axs[0, 0].plot([wineRed_y_test.min(), wineRed_y_test.max()], [wineRed_y_test.min(), wineRed_y_test.max()], 'r--')
axs[0, 0].set_xlabel('Actual Quality')
axs[0, 0].set_ylabel('Predicted Quality')
axs[0, 0].set_title('Red Wine: Actual vs Predicted Quality')
axs[0, 0].set_xlim(1, 10)
axs[0, 0].set_ylim(1, 10)
axs[0, 0].grid()

axs[0, 1].scatter(wineWhite_y_test, wineWhite_y_pred, alpha=0.6)
axs[0, 1].plot([wineWhite_y_test.min(), wineWhite_y_test.max()], [wineWhite_y_test.min(), wineWhite_y_test.max()], 'r--')
axs[0, 1].set_xlabel('Actual Quality')
axs[0, 1].set_ylabel('Predicted Quality')
axs[0, 1].set_title('White Wine: Actual vs Predicted Quality')
axs[0, 1].set_xlim(1, 10)
axs[0, 1].set_ylim(1, 10)
axs[0, 1].grid()

axs[1, 0].scatter(wineConsumer_y_test, wineConsumer_y_pred, alpha=0.6)
axs[1, 0].plot([wineConsumer_y_test.min(), wineConsumer_y_test.max()], [wineConsumer_y_test.min(), wineConsumer_y_test.max()], 'r--')
axs[1, 0].set_xlabel('Actual Quality')
axs[1, 0].set_ylabel('Predicted Quality')
axs[1, 0].set_title('Consumer Model: Actual vs Predicted Quality')
axs[1, 0].set_xlim(1, 10)
axs[1, 0].set_ylim(1, 10)
axs[1, 0].grid()

axs[1, 1].barh(red_features, regr_red.coef_)
axs[1, 1].set_xlabel('Coefficient Value')
axs[1, 1].set_title('Red Wine Model Coefficients')
axs[1, 1].grid()

axs[2, 0].barh(white_features, regr_white.coef_)
axs[2, 0].set_xlabel('Coefficient Value')
axs[2, 0].set_title('White Wine Model Coefficients')
axs[2, 0].grid()

axs[2, 1].barh(consumer_features, regr_consumer.coef_)
axs[2, 1].set_xlabel('Coefficient Value')
axs[2, 1].set_title('Consumer Model Coefficients')
axs[2, 1].grid()

plt.tight_layout(pad=2.0)
plt.show()

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error

data_dict = {
    "Dosage(mg)": [1, 3, 5, 6, 10, 15, 20, 23, 25, 27, 29, 32, 36, 40, 42, 43, 44, 45],
    "Age": [50, 40, 20, 25, 47, 35, 50, 34, 45, 37, 38, 69, 49, 70, 80, 25, 20, 60],
    "Sex": [0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0],
    "Drug Effectiveness(%)": [0.5, 0.6, 0.7, 0.8, 5, 15, 98, 100, 100, 99, 78, 70, 66, 32, 5, 3, 1, 0]
}

df = pd.DataFrame(data_dict)

# Streamlit app
st.title("Drug Effectiveness Prediction App")

# Sidebar for user input
st.sidebar.header("Adjust Model Parameters")

# Choose max_depth using a slider
max_depth = st.sidebar.slider("Select Max Depth", min_value=1, max_value=10, value=2)

# Display the dataset
st.write("## Loaded Data:")
st.write(df.head())

# Train a Decision Tree Regressor with the specified max_depth
dtr = DecisionTreeRegressor(max_depth=max_depth)

# Prepare X and y
X = df.drop("Drug Effectiveness(%)", axis=1)
y = df["Drug Effectiveness(%)"]

# Fit the model
model = dtr.fit(X, y)

# Display the Decision Tree plot
st.write("## Decision Tree Plot:")
plt.figure(figsize=(10, 8))
plot_tree(model, feature_names=X.columns, filled=True)
st.pyplot()

# Make predictions
prediction = model.predict(X)

# Display scatter plot of actual vs predicted values
st.write("## Actual vs Predicted Values:")
plt.scatter(x=df["Dosage(mg)"], y=y, label="Actual")
plt.scatter(x=df["Dosage(mg)"], y=prediction, label="Predicted")
plt.xlabel("Dosage(mg)")
plt.ylabel("Drug Effectiveness(%)")
plt.legend()
st.pyplot()

# Display Mean Squared Error
mse = mean_squared_error(y, prediction)
st.write(f"Mean Squared Error: {mse}")

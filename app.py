import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from fastapi import FastAPI

# Load the data
data = pd.read_csv("data.csv")  # Replace 'data.csv' with your actual file name

# Extract the relevant columns
X = data.iloc[:, 1].values.reshape(-1, 1)  # Experience in years
y = data.iloc[:, 2].values  # Salary

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Create FastAPI app
app = FastAPI()

@app.get("/")
def greet_json():
    return {"Hello": "World!"}

@app.get("/predict/")
def predict(experience: float):
    """
    Predict the salary based on years of experience.
    :param experience: Years of experience (can be decimals)
    :return: Predicted salary
    """
    years_of_experience = [[experience]]
    predicted_salary = model.predict(years_of_experience)
    return {"message": f"Predicted Salary for {experience} years of experience: {predicted_salary[0]}"}
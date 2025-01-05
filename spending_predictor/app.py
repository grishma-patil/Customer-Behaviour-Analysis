from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd

app = Flask(__name__)

# Load the data
df = pd.read_csv("marketing_campaign.csv")

# Train the model
def train_model():
    X = df[["Age", "Income", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]]
    y = df["Total_Spending"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()

# Routes
@app.route("/")
def home():
    """Render the data analysis page."""
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Render the prediction page."""
    if request.method == "POST":
        data = request.json
        input_data = [
            [
                data.get("age", 0),
                data.get("income", 0),
                data.get("numwebpurchases", 0),
                data.get("numcatalogpurchases", 0),
                data.get("numstorepurchases", 0)
            ]
        ]
        prediction = model.predict(input_data)
        return jsonify({"total_spending": f"${prediction[0]:,.2f}"})
    return render_template("predict.html")

if __name__ == "__main__":
    app.run(debug=True)

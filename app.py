from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    amount = float(request.form["amount"])
    is_cod = int(request.form["is_cod"])
    past_returns = int(request.form["past_returns"])
    address_mismatch = int(request.form["address_mismatch"])

    features = np.array([[amount, is_cod, past_returns, address_mismatch]])
    risk_probability = model.predict_proba(features)[0][1] * 100

    if risk_probability < 30:
        risk_level = "LOW"
    elif risk_probability < 70:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    return render_template("index.html",
                           prediction_text=f"Fraud Risk: {risk_probability:.2f}% | Risk Level: {risk_level}")

if __name__ == "__main__":
    app.run(debug=True)

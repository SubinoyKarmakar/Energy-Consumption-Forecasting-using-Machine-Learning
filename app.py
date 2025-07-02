from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

features = ["Access to electricity (% of population)", "Access to clean fuels for cooking",
    "Renewable-electricity-generating-capacity-per-capita", "Financial flows to developing countries (US $)",
    "Renewable energy share in the total final energy consumption (%)", "Electricity from fossil fuels (TWh)",
    "Electricity from nuclear (TWh)", "Electricity from renewables (TWh)",
    "Low-carbon electricity (% electricity)", "Energy intensity level of primary energy (MJ/$2017 PPP GDP)",
    "Value_co2_emissions_kt_by_country", "Renewables (% equivalent primary energy)", "gdp_growth",
    "gdp_per_capita", "Land Area(Km2)", "Latitude", "Longitude"]

def categorize(value):
    if value <= 1000: return "Very Low"
    elif value <= 3000: return "Low"
    elif value <= 7000: return "Moderate"
    elif value <= 10000: return "High"
    else: return "Very High"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(request.form.get(f)) for f in features]
        df = pd.DataFrame([data], columns=features)
        scaled = scaler.transform(df)

        y_scaler = pickle.load(open('y_scaler.pkl', 'rb'))
        pred_scaled = model.predict(scaled)[0]
        pred = y_scaler.inverse_transform([[pred_scaled]])[0][0]

        return render_template('index.html', result=round(pred, 2), category=categorize(pred))
    except Exception as e:
        print("Error:", e)
        return render_template('index.html', result="Invalid input! Enter numbers only.", category="")

if __name__ == '__main__':
    app.run(debug=True)

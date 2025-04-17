from flask import Flask, request, render_template
import pickle
import joblib
import numpy as np

# Load your saved pipeline
with open('taxi_fare_pipeline.pkl', 'rb') as f:
    model = joblib.load(f)

app = Flask(__name__)

# Just a list of IDs 1–264
location_ids = list(range(1, 265))

@app.route('/')
def home():
    return render_template(
        'index.html',
        location_ids=location_ids,
        form_data={},               # ← here
        prediction_text=None        # ← optional, since you also guard on it in the template
    )


@app.route('/predict', methods=['POST'])
def predict():
    # Turn ImmutableMultiDict into a plain dict
    form_data = request.form.to_dict()

    # Build your feature array exactly as before
    user_input = [[
        int(form_data['VendorID']),
        float(form_data['passenger_count']),
        float(form_data['trip_distance']),
        float(form_data['RatecodeID']),
        form_data['store_and_fwd_flag'],
        int(form_data['PULocationID']),
        int(form_data['DOLocationID']),
        form_data['payment_type'],
        float(form_data['extra']),
        float(form_data['tip_amount']),
        float(form_data['tolls_amount']),
        float(form_data['improvement_surcharge']),
        float(form_data['congestion_surcharge']),
        float(form_data['Airport_fee'])
    ]]

    pred = model.predict(user_input)[0]
    prediction_text = f"Estimated fare: ₹{pred:.2f}"

    return render_template(
        'index.html',
        location_ids=location_ids,
        form_data=form_data,
        prediction_text=prediction_text
    )


if __name__ == '__main__':
    app.run(debug=True)

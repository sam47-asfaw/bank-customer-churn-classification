from flask import Flask, jsonify, request
from waitress import serve

import pickle

with open('./model_C=1.0.bin', 'rb') as f_in:
    (dv, model)= pickle.load(f_in)

app = Flask('Bank Churn')

@app.route('/predict', methods= ['POST'])
def predict():
    customer = request.get_json()
    X = dv.transform(customer)
    prediction = float(model.predict_proba(X)[0,1])
    churn = bool(prediction >= 0.36)

    result = {
        'Churn Probablility': prediction,
        'churn': churn
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)





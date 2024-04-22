from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

loaded_model = joblib.load('lstm_model.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    open_price = float(request.form['open'])
    high_price = float(request.form['high'])
    low_price = float(request.form['low'])
    volume = float(request.form['volume'])
    change_percent = float(request.form['change'])
    company = int(request.form['company'])

    new_data = pd.DataFrame({
        'Open': [open_price],
        'High': [high_price],
        'Low': [low_price],
        'Vol.': [volume],
        'Change %': [change_percent],
        'Company': [company]
    })

    predictions = loaded_model.predict(new_data)

    return jsonify({'predictions': predictions.tolist()})


from joblib import load

loaded_model_anomaly = load('best_model.joblib')
label_encoder = load('label_encoder.joblib')


@app.route('/anomaly')
def anomaly():
    return render_template('anomaly.html')


@app.route('/predictAnomaly', methods=['POST'])
def predictAnomaly():
    cpu_usage = float(request.form.get('cpu_usage'))
    memory_usage = float(request.form.get('memory_usage'))
    response_time = float(request.form.get('response_time'))
    network_traffic = int(request.form.get('network_traffic'))

    # result = {'cpu_usage': cpu_usage, 'memory_usage': memory_usage, 'response_time': response_time, 'network_traffic': network_traffic}
    #
    # return jsonify(result)

    user_df = pd.DataFrame([[cpu_usage, memory_usage, response_time, network_traffic]],
                           columns=['cpu_usage', 'memory_usage', 'response_time', 'network_traffic'])

    user_pred = loaded_model_anomaly.predict(user_df)

    user_pred_labels = label_encoder.inverse_transform(user_pred)

    return jsonify('Prediction: {}'.format(user_pred_labels[0]))


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')

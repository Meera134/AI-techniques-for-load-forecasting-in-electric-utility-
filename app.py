from flask import Flask, render_template, request
import matlab.engine
import numpy as np

app = Flask(__name__)

# Start MATLAB engine
print("Starting MATLAB Engine...")
eng = matlab.engine.start_matlab()
eng.addpath(r'C:\Users\yash2\OneDrive\Desktop\AI Load Forecast', nargout=0)
print("MATLAB Engine started.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        city = request.form['city']
        model = request.form['model']
        T2M = float(request.form['T2M'])
        QV2M = float(request.form['QV2M'])
        W2M = float(request.form['W2M'])
        TQL = float(request.form['TQL'])

        # Prepare input array for MATLAB
        weather_input = matlab.double([T2M, QV2M, W2M, TQL])

        # Call MATLAB function
        prediction = eng.predict_load(city, model, weather_input)

        # Convert MATLAB output to Python list
        prediction_list = [round(p[0], 2) for p in prediction]  # Each 'p' is a list like [value]


        # Generate hours of the day for display
        hours = [f"{h}:00" for h in range(24)]

        # Return result template
        return render_template('result.html', prediction=zip(hours, prediction_list))

    except Exception as e:
        print("Error:", str(e))
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=8000)

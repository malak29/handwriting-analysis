from flask import Flask, request, jsonify
import json
# Import your modules that contain EDA, training, and testing functions
# from your_module import eda_function, train_model, test_model

app = Flask(__name__)

@app.route('/eda', methods=['POST'])
def run_eda():
    # Assuming the EDA function doesn't need input or can get it from a file
    try:
        results = eda_function()
        return jsonify({"status": "success", "results": results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/train', methods=['POST'])
def train():
    # Assuming the training data and parameters are sent in the request
    data = request.json
    try:
        model_info = train_model(data)
        return jsonify({"status": "success", "model_info": model_info})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/test', methods=['POST'])
def test():
    # Assuming the test data and model information are sent in the request
    data = request.json
    try:
        test_results = test_model(data)
        return jsonify({"status": "success", "test_results": test_results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

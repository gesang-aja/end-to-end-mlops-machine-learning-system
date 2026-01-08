from flask import Flask, request, jsonify
import pandas as pd
import pickle
import traceback

# Load model pickle
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    return "pong"

@app.route("/invocations", methods=["POST"])
def invocations():
    try:
        data = request.get_json()

        if "dataframe_records" not in data:
            return jsonify({"error": "Invalid input format"}), 400

        df = pd.DataFrame(data["dataframe_records"])
        preds = model.predict(df)

        return jsonify(preds.tolist())

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

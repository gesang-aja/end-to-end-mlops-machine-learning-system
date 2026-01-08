import requests
import json

data = {
    "dataframe_records": [
        {
            "SeniorCitizen": 0,
            "tenure": 12,
            "MonthlyCharges": 70.5,
            "TotalCharges": 845.0,
            "gender_Male": 1,
            "Partner_Yes": 1,
            "Dependents_Yes": 0,
            "PhoneService_Yes": 1,
            "PaperlessBilling_Yes": 1,
            "PaymentMethod_Electronic check": 1
        }
    ]
}

url = "http://localhost:8000/invocations"
headers = {"Content-Type": "application/json"}

response = requests.post(url, data=json.dumps(data), headers=headers)

print("Status:", response.status_code)
print("Result:", response.text)

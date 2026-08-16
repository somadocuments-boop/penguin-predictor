import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={
        "bill_length_mm": 45.0,
        "bill_depth_mm": 17.0,
        "flipper_length_mm": 200.0,
        "body_mass_g": 4000.0
    }
)
print(response.json())
import requests
import json

# URL de l'API
api_url = "https://githubactions-atsfyczita-od.a.run.app/predict"

def test_predict_positive():
    """
    Teste l'endpoint /predict avec un tweet positif.
    """
    tweet = "I love this product! It's amazing and works perfectly."
    response = requests.post(api_url, headers={"Content-Type": "application/json"}, data=json.dumps({"tweet": tweet}))
    
    assert response.status_code == 200, f"Expected status code 200 but got {response.status_code}"
    data = response.json()
    assert "sentiment" in data, "Response JSON does not contain 'sentiment' key"
    assert data["sentiment"] == "positive", f"Expected sentiment 'positive' but got {data['sentiment']}"

def test_predict_negative():
    """
    Teste l'endpoint /predict avec un tweet négatif.
    """
    tweet = "I am extremely disappointed with the experience. The product quality was far below expectations, and the customer support was unhelpful and slow to respond. I wasted both my time and money. I would not recommend this to anyone."
    response = requests.post(api_url, headers={"Content-Type": "application/json"}, data=json.dumps({"tweet": tweet}))
    
    assert response.status_code == 200, f"Expected status code 200 but got {response.status_code}"
    data = response.json()
    assert "sentiment" in data, "Response JSON does not contain 'sentiment' key"
    assert data["sentiment"] == "negative", f"Expected sentiment 'negative' but got {data['sentiment']}"

if __name__ == "__main__":
    test_predict_positive()
    test_predict_negative()
    print("All tests passed!")
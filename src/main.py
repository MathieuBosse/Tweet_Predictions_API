from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import re
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
import logging


# Initialiser Flask
app = Flask(__name__)

# Configurer le logging de base
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Télécharger la liste des stopwords si nécessaire
nltk.download('stopwords')

# Liste des stopwords en anglais
stop_words = set(stopwords.words('english'))

# Dictionnaire pour les emojis
emojis = {
    ':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
    ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
    ':-@': 'shocked', ':@': 'shocked', ':$': 'confused', ':\\': 'annoyed', 
    ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
    '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
    '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
    ';-)': 'wink', 'O:-)': 'angel', 'O*-)': 'angel', '(:-D': 'gossip', '=^.^=': 'cat'
}

def clean_tweet(tweet):
    """
    Nettoie le texte du tweet en supprimant les liens, les mentions, les caractères spéciaux et en convertissant en minuscules.
    """
    tweet = re.sub(r"http\S+", "", tweet)    # Supprime les liens
    tweet = re.sub(r"@\w+", "", tweet)       # Supprime les mentions
    tweet = re.sub(r"[^a-zA-Z\s]", "", tweet)  # Supprime les caractères spéciaux et les chiffres
    tweet = tweet.lower()                   # Convertir en minuscules
    return tweet

def replacing_emojis(data):
    """
    Remplace les emojis par des mots correspondants.
    """
    text = []
    for word in data.split():
        text.append(emojis.get(word, word))
    return ' '.join(text)

def remove_stopwords(tweet):
    """
    Supprime les mots vides du texte.
    """
    return " ".join([word for word in tweet.split() if word not in stop_words])

def preprocess_tweet(tweet, tokenizer, max_length):
    """
    Prétraite le tweet et le convertit en séquence de tokens avec padding.
    """
    tweet = clean_tweet(tweet)
    tweet = replacing_emojis(tweet)
    tweet = remove_stopwords(tweet)
    
    # Convertir le texte en séquence de tokens
    tweet_seq = tokenizer.texts_to_sequences([tweet])
    
    # Appliquer le padding
    tweet_padded = pad_sequences(tweet_seq, maxlen=max_length)
    
    return tweet_padded

# Charger le modèle et le tokenizer
model_path = 'models/model6.h5'
tokenizer_path = 'models/tokenizer.pkl'
model_path_tflite = 'models/model.tflite'


# Charger le tokenizerr
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

# Initialiser l'interpréteur TFLite
interpreter = tf.lite.Interpreter(model_path=model_path_tflite)
interpreter.allocate_tensors()

# Obtenir les informations sur les entrées et les sorties du modèle
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Déterminer la longueur maximale des séquences
max_length = 35  # Remplacer par la longueur maximale utilisée lors du prétraitementt

@app.route('/')
def home():
    return "API is working!"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint pour prédire le sentiment d'un tweet.
    """
    # Récupérer le texte du tweet depuis la requête
    data = request.json
    tweet = data.get('tweet', '')

    # Prétraiter le tweet
    processed_tweet = preprocess_tweet(tweet, tokenizer, max_length)
    
    # Assurez-vous que le format de l'entrée correspond à celui attendu par le modèle
    processed_tweet = np.array(processed_tweet, dtype=np.float32)  # TFLite utilise souvent float32

    # Définir l'entrée pour l'interpréteur TFLite
    interpreter.set_tensor(input_details[0]['index'], processed_tweet)

    # Effectuer l'inférence
    interpreter.invoke()

    # Récupérer le résultat de la prédiction
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    sentiment = 'positive' if prediction > 0.5 else 'negative'
    
    return jsonify({'sentiment': sentiment})

@app.route('/feedback', methods=['POST'])
def feedback():
    """
    Endpoint pour recueillir les feedbacks utilisateurs concernant les prédictions.
    """
    data = request.json
    tweet = data.get('tweet', '')
    predicted_sentiment = data.get('prediction', '')
    feedback = data.get('feedback', '')

    if feedback == 'incorrect':
        # Log le tweet et la prédiction mal prédite
        logger.error(f"Incorrect prediction: tweet='{tweet}', predicted_sentiment='{predicted_sentiment}'")
    
    return jsonify({'status': 'feedback received'})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

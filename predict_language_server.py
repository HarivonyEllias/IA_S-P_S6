from typing import Counter
from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
from math import log, log2
import random

app = Flask(__name__)

def generate_random_binary_word(min_length=1, max_length=7):
    # Générer une longueur aléatoire pour le mot binaire
    length = random.randint(min_length, max_length)
    # Générer un mot binaire de la longueur spécifiée
    word = ''.join(random.choice('01') for _ in range(length))
    return word

def generate_random_language(min_words=1, max_words=10, min_length=1, max_length=7):
    num_words = random.randint(min_words, max_words)
    language = [generate_random_binary_word(min_length, max_length) for _ in range(num_words)]
    return language

# Charger le modèle
model = joblib.load('SP_model.pkl')

def generate_features(language):
    features = {
        'nombre_mots': len(language),
        'longueur_moyenne': sum(len(word) for word in language) / len(language),
        'longueur_minimale': min(len(word) for word in language),
        'longueur_maximale': max(len(word) for word in language),
        'distribution_longueur': [len([word for word in language if len(word) == i]) for i in range(8)],
        'complexité_shannon': shannon_entropy(language),
        'redondance': redundancy(language),
        'proportion_0': ''.join(language).count('0') / len(''.join(language)),
        'proportion_1': ''.join(language).count('1') / len(''.join(language)),
        'entropie_des_caracteres': character_entropy(language),
        'distance_de_hamming_moyenne': average_hamming_distance(language),
    }
    return features

def character_entropy(language):
    if len(language) == 0:
        return 0
    concatenated = ''.join(language)
    frequency = Counter(concatenated)
    total_chars = len(concatenated)
    entropy = -sum((count / total_chars) * log2(count / total_chars) for count in frequency.values())
    return entropy

def shannon_entropy(language):
    concatenated = ''.join(language)
    frequency = pd.Series(list(concatenated)).value_counts() / len(concatenated)
    entropy = -(frequency * frequency.apply(lambda x: log(x, 2))).sum()
    return entropy

def redundancy(language):
    redundancy_count = 0
    for word in language:
        for other_word in language:
            if word != other_word and (word.startswith(other_word) or word.endswith(other_word)):
                redundancy_count += 1
                break
    return redundancy_count

def predict_language(model, language):
    features = generate_features(language)
    df = pd.DataFrame([features])
    max_length = 7
    for i in range(max_length + 1):
        df[f'longueur_{i + 1}'] = df['distribution_longueur'].apply(lambda x: x[i] if len(x) > i else 0)
    df = df.drop('distribution_longueur', axis=1)
    prediction = model.predict(df)
    return prediction[0]

def average_hamming_distance(language):
    if len(language) < 2:
        return 0
    def hamming_distance(s1, s2):
        return sum(c1 != c2 for c1, c2 in zip(s1, s2)) + abs(len(s1) - len(s2))
    
    distances = []
    for i in range(len(language)):
        for j in range(i + 1, len(language)):
            distances.append(hamming_distance(language[i], language[j]))
    
    return np.mean(distances)


## 
# Enlever le prefixe et s'il est vide on remplace par e
def removeprefixe(language, prefixe):
    result = [
        (str[len(prefixe):] if str.startswith(prefixe) else str) or "e"
        for str in language
    ]
    return result

# Enlever le epsilone
def removeepsilone(language):
    result = [
        str[1:] if str.startswith("e") else str
        for str in language
    ]
    return result

# Residuel
def residuel(language, mot):
    residuel = [s for s in language if s.startswith(mot)]
    if not residuel:
        residuel.append("vide")
    residuel = removeprefixe(residuel, mot)
    return residuel

# Quotient c'est à dire l'union des languages 
def quotient(*arrays):
    result_set = set()
    for array in arrays:
        for item in array:
            if item != "vide": 
                result_set.add(item) 
    return sorted(result_set)

# Vérifier si le mot est déjà passer
def ifexist(residuel):
    seen = set()
    for r in residuel:
        r_tuple = tuple(sorted(r))
        if r_tuple in seen:
            return True
        seen.add(r_tuple)
    return False

# Enlever le string vide dans la language
def residuel_func(language, mot):
    residuel = [s for s in language if s.startswith(mot)]
    if not residuel:
        residuel.append("vide")
    return removeprefixe(residuel, mot)

# Sardinas
def sardinas(language):
    residuel = []
    L1 = []
    for i in range(len(language)):
        L11 = residuel_func(language, language[i])
        L1.append(L11)

    LUnion = quotient(*L1)
    LUnion = removeepsilone(LUnion)

    residuel.append(LUnion)

    for i in range(len(residuel[0]) - 1, -1, -1):
        if residuel[0][i] == "":
            residuel[0].pop(i)

    i = 0
    while i < len(residuel):
        if "e" not in residuel[i]:
            residuelL = [residuel[i][j] for j in range(len(residuel[i]))]
            firstL = []
            secondL = []
            for k in range(len(language)):
                firstL.append(residuel_func(residuelL, language[k]))
            for l in range(len(residuelL)):
                secondL.append(residuel_func(language, residuelL[l]))
            firstUnion = quotient(*firstL)
            secondUnion = quotient(*secondL)
            residuel.append(quotient(firstUnion, secondUnion))
        i += 1
        if ifexist(residuel):
            break

    return residuel

# Code or not
def verificationcode(language):
    residuel = sardinas(language)
    for res in residuel:
        if "e" in res:
            return False
    return True

##




@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        language = request.form['language'].split(',')
        prediction = predict_language(model, language) #Boolean from model
        algo = verificationcode(language)         # Boolean from Sardinas algorithm
        
        return render_template('result.html', prediction=prediction , algo=algo)
    return render_template('index.html')

@app.route('/test')
def test():
    # Initialize counters
    correct_predictions = 0
    total_tests = 5000
    
    for _ in range(total_tests):
        language = generate_random_language()
        prediction_a = predict_language(model, language)  # Boolean from model
        prediction_b = verificationcode(language)         # Boolean from Sardinas algorithm
        print( str(correct_predictions) + " / " + str(_) )
        if prediction_a == prediction_b:
            correct_predictions += 1
    
    # Calculate efficiency percentage
    efficiency = (correct_predictions / total_tests) * 100
    
    return render_template('result_test.html', efficiency=efficiency)

if __name__ == "__main__":
    app.run(debug=True)

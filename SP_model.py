from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
from ast import literal_eval  # Utilisé pour convertir la chaîne de caractères en liste

# Charger les données
data = pd.read_csv('./training_data.csv')

# Convertir la colonne 'distribution_longueur' de la chaîne de caractères en liste
data['distribution_longueur'] = data['distribution_longueur'].apply(literal_eval)

# Créer de nouvelles colonnes pour chaque longueur de mot possible
max_length = max(data['distribution_longueur'].apply(len))
for i in range(max_length):
    data[f'longueur_{i+1}'] = data['distribution_longueur'].apply(lambda x: x[i] if len(x) > i else 0)

# Supprimer la colonne 'distribution_longueur' d'origine
data.drop('distribution_longueur', axis=1, inplace=True)

# Diviser les données en variables explicatives (X) et cible (y)
X = data.drop('est_code', axis=1)
y = data['est_code']

# Créer et entraîner le modèle RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Sauvegarder le modèle dans un fichier
joblib.dump(model, 'SP_model.pkl')

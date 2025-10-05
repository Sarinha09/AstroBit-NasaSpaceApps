import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree

import joblib

df = pd.read_csv('master_exoplanet_data.csv', comment='#')

features_selecionadas = [
    'orbital_period','transit_duration','transit_depth_ppm','planet_radius','stellar_temp','stellar_logg','stellar_radius'
]
alvo = 'disposition'
df_modelo = df[features_selecionadas + [alvo]].copy()

df_modelo.dropna(inplace=True)

X = df_modelo[features_selecionadas]
y = df_modelo[alvo]
le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded)



scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, max_depth=12)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAcurácia Geral do Modelo: {accuracy:.4f} (ou {accuracy*100:.2f}%)")
class_names = le.classes_
print(classification_report(y_test, y_pred, target_names=class_names))

print("Matriz de confusao")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusão', fontsize=16)
plt.ylabel('Classe Verdadeira (Real)')
plt.xlabel('Classe Prevista (Modelo)')
plt.savefig('matriz_confusao.png')

report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)


importances = model.feature_importances_
feature_names_original = X.columns

feature_importance_pairs = sorted(zip(feature_names_original, importances), key=lambda x: x[1], reverse=True)
sorted_labels = [name for name, score in feature_importance_pairs]
sorted_scores = [score for name, score in feature_importance_pairs]

FEATURE_NAME_MAP = {
    'orbital_period': 'Orbital Period',
    'transit_duration': 'Transit Duration',
    'transit_depth_ppm': 'Transit Depth',
    'planet_radius': 'Planet Radius',
    'stellar_temp': 'Stellar Temperature',
    'stellar_logg': 'Stellar Gravity (log g)',
    'stellar_radius': 'Stellar Radius'
}
friendly_labels = [FEATURE_NAME_MAP.get(name, name) for name in sorted_labels]

metrics = {
    "accuracy": accuracy,
    "classification_report": report_dict,
    "confusion_matrix": cm.tolist(),
    "class_names": class_names.tolist(),
    "feature_importances": {
        "features": friendly_labels,
        "importances": sorted_scores
    }
}
with open('model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

joblib.dump(model, 'exoplanet_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(le, 'label_encoder.joblib')

plt.close()
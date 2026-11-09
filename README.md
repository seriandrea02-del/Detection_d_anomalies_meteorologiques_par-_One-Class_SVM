# Détection d'anomalies météorologiques — One-Class SVM

## Présentation
Projet Python de détection non supervisée d'observations météorologiques inhabituelles à partir du jeu de données `weatherHistory.csv`. Le modèle apprend les caractéristiques générales des observations historiques, puis signale les observations qui s'écartent de ce comportement.

**Attention :** une anomalie statistique n'est pas nécessairement une erreur de capteur ni un événement météorologique dangereux. Les résultats doivent être interprétés et vérifiés.

## Objectifs
- Explorer et préparer des données météorologiques réelles.
- Appliquer un modèle One-Class SVM (noyau RBF).
- Détecter des observations atypiques sur une période de test ultérieure.
- Visualiser et exporter les résultats pour leur analyse.

## Données et variables
**Source :** fichier `weatherHistory.csv` fourni par l'enseignant dans le cadre d'un TP (observations météo horaires, 2006–2016). Ce jeu de données semble également circuler publiquement : vérifier son origine et sa licence avant toute réutilisation ou redistribution.

Le fichier contient 96 453 observations et 12 colonnes, couvrant la période 2005–2016. Le modèle utilise : température, température ressentie, humidité, vitesse du vent, visibilité et pression. La date est conservée pour l'analyse temporelle.

## Méthode
1. Lecture, conversion des dates et tri chronologique.
2. Remplacement des pressions nulles par des valeurs manquantes, puis imputation médiane.
3. Séparation chronologique : premières 80 % des observations pour l'apprentissage, dernières 20 % pour le test.
4. Standardisation ajustée uniquement sur l'apprentissage.
5. Entraînement du One-Class SVM sur un échantillon de 6 000 observations d'apprentissage (pour maîtriser le coût de calcul du noyau RBF).
6. Prédiction sur la période de test.

Le modèle renvoie `1` pour une observation acceptée et `-1` pour une observation détectée comme atypique. Le paramètre `nu=0.05` règle notamment une borne sur la proportion d'observations rejetées pendant l'apprentissage ; il ne garantit pas un taux de 5 % sur le test.

## Installation et exécution
Python 3.10 ou plus récent recommandé.

```bash
python -m venv .venv
# Windows :
.venv\Scripts\activate
# macOS / Linux :
source .venv/bin/activate

pip install -r requirements.txt
python run_project.py
```

Le fichier CSV doit rester dans `data/weatherHistory.csv`.

## Résultats générés
Après exécution :
- `results/anomalies.csv` : observations de test détectées comme atypiques ;
- `results/predictions.csv` : prédiction et score de décision pour chaque observation test ;
- `results/figures/01_temperature_anomalies.png` : température dans le temps ;
- `results/figures/02_temperature_humidity.png` : projection température/humidité ;
- `results/figures/03_anomaly_distribution.png` : répartition normale/anomalie ;
- `results/figures/04_feature_comparison.png` : comparaison des variables.

Le rapport détaillé au format Word est fourni dans `rapport/rapport_projet.docx`.

## Structure
```text
weather-anomaly-detection/
├── data/
│   └── weatherHistory.csv
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── model.py
│   └── visualization.py
├── results/
│   └── figures/
├── rapport/
│   └── rapport_projet.docx
├── .gitignore
├── run_project.py
├── requirements.txt
└── README.md
```

## Limites
Le jeu de données ne contient pas d'étiquettes de référence indiquant les vraies anomalies. Il n'est donc pas possible de calculer directement une exactitude, une précision ou un rappel fiables. Le modèle identifie des observations atypiques au regard de l'apprentissage, pas des événements dangereux confirmés. Une validation métier ou des anomalies artificiellement injectées et connues seraient nécessaires pour une évaluation quantitative plus complète.

## Pistes d'amélioration
- Comparer One-Class SVM à Isolation Forest et Local Outlier Factor.
- Étudier l'effet de `nu` et `gamma`.
- Analyser les anomalies par saison et par variable.
- Tester des stratégies de validation avec anomalies artificielles.

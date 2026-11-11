# 🌦️ Détection des anomalies météorologiques

**Auteure :** Laurette Seri

## 1. Présentation du projet

La météo évolue constamment : la température augmente ou diminue, l'humidité varie, le vent se renforce et la pression atmosphérique change.

Mais que se passe-t-il lorsqu'une combinaison de ces mesures devient inhabituelle ? Comment pourrait-on détecter automatiquement ces changements à partir d'un grand nombre de données ?

C'est à partir de ce constat qu'est née l'idée de ce projet.

Ce travail est également un cas pratique réalisé dans le but de mettre en application les connaissances acquises en cours d'apprentissage automatique (Machine Learning). L'objectif était de passer de la théorie à la pratique : découvrir comment un ordinateur peut apprendre à partir de données réelles, reconnaître un comportement habituel et repérer les observations qui s'en éloignent.

À travers ce projet, j'ai donc exploré une application concrète de l'intelligence artificielle dans l'analyse et la surveillance de données météorologiques.

## 2. Pourquoi ce projet ?

Dans de nombreux domaines, il est important de pouvoir repérer rapidement un fonctionnement inhabituel.

Par exemple :
- Dans une usine, détecter qu'une machine commence à fonctionner anormalement.
- Dans un bâtiment, repérer une température ou une humidité inhabituelle.
- Dans l'environnement, surveiller les variations de certaines mesures météorologiques.

L'idée est donc de créer un système capable d'apprendre ce qui est habituel afin de signaler ce qui s'en éloigne.

## 3. Comment fonctionne le programme ?

Pour réaliser ce projet, j'ai utilisé un historique de données météorologiques contenant notamment :
- La température extérieure ;
- La température ressentie ;
- Le taux d'humidité ;
- La vitesse du vent ;
- La visibilité ;
- La pression atmosphérique.

Le programme suit quatre grandes étapes :
1. **Collecter les données :** utiliser un fichier contenant des mesures météorologiques réelles.
2. **Préparer les informations :** vérifier et organiser les données pour qu'elles puissent être analysées.
3. **Apprendre le comportement habituel :** utiliser un modèle d'intelligence artificielle pour reconnaître les situations courantes.
4. **Repérer les anomalies :** comparer les observations avec ce que le modèle a appris et signaler les situations inhabituelles.

## 4. Résultats obtenus

Après l'analyse des données, le programme a étudié 19 286 observations réservées aux tests.

Il a identifié :
- 17 526 observations comme étant habituelles ;
- 1 760 observations comme étant inhabituelles.

Les graphiques suivants permettent de visualiser les résultats de l'analyse :
- **Graphique 1 – Détection des anomalies de température :** Permet de visualiser les températures étudiées et de distinguer les observations considérées comme habituelles de celles signalées comme inhabituelles.
- **Graphique 2 – Relation entre température et humidité :** Permet d'observer comment la température et l'humidité varient ensemble et de visualiser les observations signalées par le modèle.
- **Graphique 3 – Répartition des observations :** Présente la répartition entre les observations considérées comme normales et celles détectées comme inhabituelles.
- **Graphique 4 – Comparaison des différentes mesures :** Permet de comparer plusieurs paramètres météorologiques et de mieux comprendre les différences entre les observations normales et inhabituelles.

> **À retenir :** Les observations signalées ne sont pas nécessairement des phénomènes météorologiques dangereux. Le fichier ne contient pas de liste d'anomalies confirmées : le programme identifie simplement les mesures qui s'écartent du comportement appris.

## 5. Ce que ce projet m'a permis d'apprendre

Ce projet m'a permis de passer de la théorie à la pratique et de développer plusieurs compétences :
- L'analyse et la préparation de données réelles ;
- La programmation en Python ;
- La mise en œuvre d'une méthode d'apprentissage automatique ;
- La détection de comportements inhabituels ;
- La création et l'interprétation de graphiques ;
- L'organisation d'un projet informatique complet.

## 6. Technologies et environnement utilisés

- **Python :** développement du programme.
- **Pandas et NumPy :** manipulation des données.
- **Scikit-learn :** création du modèle d'intelligence artificielle.
- **Matplotlib :** réalisation des graphiques.
- **Spyder :** environnement de développement (IDE) utilisé pour écrire, exécuter et tester le programme.

## 7. Organisation du projet

```text
weather-anomaly-detection/
├── data/              # Données météorologiques utilisées
├── src/               # Programmes qui réalisent l'analyse
├── results/           # Résultats et graphiques générés
├── rapport/           # Rapport détaillé du projet
├── run_project.py     # Programme principal pour lancer l'analyse
├── requirements.txt   # Dépendances du projet
└── README.md          # Documentation du projet
```

## 8. Comment lancer le projet ?

Après avoir installé Python et les bibliothèques nécessaires :

```bash
pip install -r requirements.txt
python run_project.py
```

Le programme réalise automatiquement l'analyse et génère les résultats ainsi que les graphiques.



## 9. Auteure

**Laurette Seri**

*Projet réalisé dans le cadre de la mise en pratique des connaissances acquises en apprentissage automatique (Machine Learning).*

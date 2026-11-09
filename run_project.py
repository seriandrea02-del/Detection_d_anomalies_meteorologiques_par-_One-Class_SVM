import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from config import (
    DATA_PATH, RESULTS_DIR, FIGURES_DIR, DATE_COLUMN, FEATURES,
    TRAIN_RATIO, TRAIN_SAMPLE_SIZE, RANDOM_STATE, NU, GAMMA
)
from preprocessing import (
    load_and_prepare, chronological_split, transform_features
)
from model import train_one_class_svm, predict
from visualization import create_figures


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("1/5 - Lecture et préparation des données...")
    data, X = load_and_prepare(DATA_PATH, DATE_COLUMN, FEATURES)

    print("2/5 - Séparation chronologique...")
    train_data, test_data, X_train, X_test = chronological_split(
        data, X, DATE_COLUMN, TRAIN_RATIO
    )

    print("3/5 - Imputation et standardisation...")
    X_train_scaled, X_test_scaled, imputer, scaler = transform_features(
        X_train, X_test
    )

    print("4/5 - Entraînement du One-Class SVM...")
    model, fitted_size = train_one_class_svm(
        X_train_scaled, TRAIN_SAMPLE_SIZE, NU, GAMMA, RANDOM_STATE
    )

    print("5/5 - Détection et génération des résultats...")
    labels, scores = predict(model, X_test_scaled)

    predictions = test_data[[DATE_COLUMN] + FEATURES].copy()
    predictions["Prediction"] = labels
    predictions["Decision_Score"] = scores
    predictions.to_csv(RESULTS_DIR / "predictions.csv", index=False)

    anomalies = predictions[predictions["Prediction"] == -1].copy()
    anomalies.to_csv(RESULTS_DIR / "anomalies.csv", index=False)

    create_figures(test_data, labels, FEATURES, FIGURES_DIR)

    total = len(labels)
    anomaly_count = int((labels == -1).sum())
    normal_count = int((labels == 1).sum())
    rate = 100 * anomaly_count / total if total else 0

    print("\n===== BILAN =====")
    print(f"Observations totales après préparation : {len(data)}")
    print(f"Observations d'apprentissage : {len(train_data)}")
    print(f"Observations utilisées pour ajuster le SVM : {fitted_size}")
    print(f"Observations de test : {total}")
    print(f"Observations normales détectées : {normal_count}")
    print(f"Anomalies détectées : {anomaly_count}")
    print(f"Taux d'anomalies sur le test : {rate:.2f} %")
    print(f"Résultats enregistrés dans : {RESULTS_DIR}")


if __name__ == "__main__":
    main()

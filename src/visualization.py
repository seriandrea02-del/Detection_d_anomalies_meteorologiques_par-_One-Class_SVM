from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def create_figures(test_data, predictions, features, figures_dir):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_data = test_data.copy()
    plot_data["Prediction"] = predictions
    normal = plot_data["Prediction"] == 1
    anomaly = plot_data["Prediction"] == -1

    # 1. Température au cours du temps.
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(plot_data["Formatted Date"], plot_data["Temperature (C)"],
            linewidth=0.55, alpha=0.65, label="Mesures")
    ax.scatter(plot_data.loc[anomaly, "Formatted Date"],
               plot_data.loc[anomaly, "Temperature (C)"],
               s=12, label="Anomalies détectées")
    ax.set_title("Température au cours du temps et anomalies détectées")
    ax.set_xlabel("Date")
    ax.set_ylabel("Température (°C)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figures_dir / "01_temperature_anomalies.png", dpi=160)
    plt.close(fig)

    # 2. Projection 2D : utile pour explorer, mais elle ne montre que deux
    # variables alors que le modèle utilise six variables.
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(plot_data.loc[normal, "Temperature (C)"],
               plot_data.loc[normal, "Humidity"], s=8, alpha=0.35,
               label="Observations acceptées")
    ax.scatter(plot_data.loc[anomaly, "Temperature (C)"],
               plot_data.loc[anomaly, "Humidity"], s=12, alpha=0.7,
               label="Anomalies détectées")
    ax.set_title("Observations selon la température et l'humidité")
    ax.set_xlabel("Température (°C)")
    ax.set_ylabel("Humidité (0 à 1)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "02_temperature_humidity.png", dpi=160)
    plt.close(fig)

    # 3. Nombre de prédictions par classe.
    counts = pd.Series(predictions).map({1: "Normales", -1: "Anomalies"}).value_counts()
    fig, ax = plt.subplots(figsize=(7, 5))
    counts.reindex(["Normales", "Anomalies"], fill_value=0).plot(
        kind="bar", ax=ax
    )
    ax.set_title("Répartition des observations détectées")
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Nombre d'observations")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "03_anomaly_distribution.png", dpi=160)
    plt.close(fig)

    # 4. Comparaison des distributions des variables.
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, feature in zip(axes.ravel(), features):
        values_normal = plot_data.loc[normal, feature].dropna()
        values_anomaly = plot_data.loc[anomaly, feature].dropna()
        ax.boxplot([values_normal, values_anomaly],
                   tick_labels=["Normales", "Anomalies"],
                   showfliers=False)
        ax.set_title(feature, fontsize=9)
        ax.tick_params(axis="x", labelrotation=15)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Comparaison des variables selon la classe prédite")
    fig.tight_layout()
    fig.savefig(figures_dir / "04_feature_comparison.png", dpi=160)
    plt.close(fig)

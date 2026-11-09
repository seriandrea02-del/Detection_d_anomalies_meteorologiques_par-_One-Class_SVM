import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def load_and_prepare(path, date_column, features):
    """Charge le CSV, convertit les dates et prépare les variables numériques."""
    data = pd.read_csv(path)

    data[date_column] = pd.to_datetime(
        data[date_column], errors="coerce", utc=True
    )
    data = data.dropna(subset=[date_column]).sort_values(date_column)
    data = data.drop_duplicates(subset=[date_column], keep="first").reset_index(drop=True)

    # Une pression nulle n'est pas physiquement plausible ici : on la traite
    # comme une donnée manquante plutôt que comme une vraie mesure.
    data["Pressure (millibars)"] = data["Pressure (millibars)"].replace(0, float("nan"))

    X = data[features].apply(pd.to_numeric, errors="coerce")
    return data, X


def chronological_split(data, X, date_column, train_ratio):
    """Sépare les observations selon le temps, sans mélanger passé et futur."""
    split_index = int(len(data) * train_ratio)
    train_data = data.iloc[:split_index].copy()
    test_data = data.iloc[split_index:].copy()
    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()
    return train_data, test_data, X_train, X_test


def transform_features(X_train, X_test):
    """Impute les valeurs manquantes et standardise sans fuite de données."""
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    train_imputed = imputer.fit_transform(X_train)
    test_imputed = imputer.transform(X_test)

    train_scaled = scaler.fit_transform(train_imputed)
    test_scaled = scaler.transform(test_imputed)

    return train_scaled, test_scaled, imputer, scaler

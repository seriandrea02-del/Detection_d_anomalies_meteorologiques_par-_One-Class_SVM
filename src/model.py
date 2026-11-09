import numpy as np
from sklearn.svm import OneClassSVM


def train_one_class_svm(X_train, sample_size, nu, gamma, random_state):
    """Entraîne le modèle sur un échantillon de la période d'apprentissage."""
    rng = np.random.default_rng(random_state)
    if len(X_train) > sample_size:
        indices = rng.choice(len(X_train), size=sample_size, replace=False)
        X_fit = X_train[indices]
    else:
        X_fit = X_train

    model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    model.fit(X_fit)
    return model, len(X_fit)


def predict(model, X_test):
    """Retourne les classes (1 normal, -1 atypique) et les scores."""
    labels = model.predict(X_test)
    scores = model.decision_function(X_test).ravel()
    return labels, scores

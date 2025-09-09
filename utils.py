import pandas as pd
import os
import re
import tempfile
import pickle
import json
import sys
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow
import mlflow.sklearn

import platform


def save_dataset_version(dataset_name: str, df: pd.DataFrame):
    """Salva il dataframe in una cartella "datasets/<dataset_name>/"
    con un nome di file che segue lo schema vXX.csv, dove XX è un numero
    incrementale che rappresenta la versione del dataset"""

    # Percorso della cartella dataset
    base_dir = "datasets"
    dataset_dir = os.path.join(base_dir, dataset_name)

    # Crea la cartella se non esiste
    os.makedirs(dataset_dir, exist_ok=True)

    # Trova i file già presenti che matchano lo schema vXX.csv
    existing_files = [f for f in os.listdir(dataset_dir) if re.match(r"v\d{2}\.csv", f)]

    if not existing_files:
        # Se non ci sono file, la prima versione è v01
        version_number = 1
    else:
        # Estrai i numeri delle versioni dai file
        versions = [int(re.findall(r"\d{2}", f)[0]) for f in existing_files]
        version_number = max(versions) + 1

    # Nome del file da salvare
    filename = f"v{version_number:02d}.csv"
    filepath = os.path.join(dataset_dir, filename)

    # Salva il dataframe
    df.to_csv(filepath, index=False)

    return filepath, version_number



def load_dataset(dataset_name):
    """
    Carica un dataset in base al nome fornito.
    Supporta "iris_dataset" e "breast_cancer_dataset".
    Restituisce X_train, X_test, y_train, y_test, dataset_version, dataset_path, df
    """
    if dataset_name == "iris_dataset":
        # Carichiamo il dataset
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target

        # Salviamo il dataset versionato localmente (puoi anche usare DVC o Git LFS)
        dataset_path, dataset_version = save_dataset_version(dataset_name, df)

        # Split train/test
        X = df[iris.feature_names]
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    elif dataset_name == "breast_cancer_dataset":
        # Carichiamo il dataset Breast Cancer
        cancer = load_breast_cancer()
        df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
        df['target'] = cancer.target

        # Salviamo il dataset versionato localmente
        dataset_path, dataset_version = save_dataset_version(dataset_name, df)

        # Split train/test
        X = df[cancer.feature_names]
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

    return X_train, X_test, y_train, y_test, dataset_version, dataset_path, df


# Funzione helper per loggare esperimenti
def train_and_log_model(model_name="IrisClassifier",
                        model_dict = {
                            "model": "RandomForest",
                            "n_estimators": 100,
                            "max_depth": None
                        },
                        X_train=None,
                        X_test=None,
                        y_train=None,
                        y_test=None,
                        dataset_version=None,
                        dataset_name=None,
                        ):
    """
    Funzione per addestrare un modello, calcolare metriche e loggare tutto su MLflow.
    model_dict: dizionario con i parametri del modello
    model_name: nome del modello registrato in MLflow
    X_train, X_test, y_train, y_test: dati di addestramento e test
    dataset_version: versione del dataset (se disponibile)
    dataset_name: nome del dataset (se disponibile)
    """

    with mlflow.start_run() as run:
        if model_dict["model"] == "RandomForest":
            # 1. Crea il modello
            model = RandomForestClassifier(n_estimators=model_dict["n_estimators"], max_depth=model_dict["max_depth"], random_state=42)
            model.fit(X_train, y_train)

            mlflow.log_param("n_estimators", model_dict["n_estimators"])
            mlflow.log_param("max_depth", model_dict["max_depth"])

            # Log feature importance (se utile per analisi)
            feature_importances = dict(zip(X_train.columns, model.feature_importances_))
            mlflow.log_dict(feature_importances, "feature_importances.json")

        elif model_dict["model"] == "LogisticRegression":
            model = LogisticRegression(max_iter=model_dict["max_iter"], random_state=42)
            model.fit(X_train, y_train)

            mlflow.log_param("max_iter", model_dict["max_iter"])


        # 2. Previsioni e metriche
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted")
        rec = recall_score(y_test, preds, average="weighted")
        f1 = f1_score(y_test, preds, average="weighted")

        # 3. Log parametri principali del modello
        mlflow.log_param("model_class", model_dict["model"])
        if dataset_version is not None:
            mlflow.log_param("dataset_version", dataset_version)
        if dataset_name is not None:
            mlflow.log_param("dataset_name", dataset_name)


        # 4. Log metriche
        mlflow.log_param("accuracy", round( float(acc), 2))
        mlflow.log_param("precision_weighted", round( float(prec), 2))
        mlflow.log_param("recall_weighted", round( float(rec), 2))
        mlflow.log_param("f1_weighted", round( float(f1), 2))

        # ---  Salvataggio locale + upload negli artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, f"{model_name}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            # Scrivi info ambiente
            env_info = {
                "python_version": sys.version,
                "platform": platform.platform(),
                "mlflow_version": mlflow.__version__,
                "sklearn_version": model.__module__.split('.')[0]
            }
            env_path = os.path.join(tmpdir, "environment_info.json")
            with open(env_path, "w") as f:
                json.dump(env_info, f, indent=2)

            # Scrivi requirements standardizzati
            reqs_path = os.path.join(tmpdir, "requirements.txt")
            with open(reqs_path, "w") as f:
                f.write("scikit-learn\nmlflow\npandas\nnumpy\n")

            # Log degli artifacts (modello + env)
            mlflow.log_artifacts(tmpdir, artifact_path="model_package")

        # 5. Log del modello e dell’ambiente
        mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)

        conda_env = mlflow.sklearn.get_default_conda_env()
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            conda_env=conda_env
        )

        # N.B. Non per forza tutti i modelli vanno registrati.
        # Volendo, si può decidere di non registrare i modelli dentro questo codice, ma decidere a posteriori quali registrare e quali no

        # 6. Log di due righe di esempio dal dataset
        sample_input = X_train.head(2).copy()
        sample_input["target"] = y_train.iloc[:2].values
        mlflow.log_table(sample_input, "sample_input.parquet")

        # 7. Stampa riassunto
        print(f"Run {run.info.run_id} - Acc: {acc:.4f}")

        return run.info.run_id

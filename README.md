# Demo MLflow - MLOps Formazione

Questo progetto è una demo che ha l'obiettivo di spiegare le funzionalità base di mlflow in Azure Machine Learning

## Come usare la repository

1. **Creare un ambiente virtuale**
	```powershell
	python3 -m venv .venv
	source .venv/bin/activate
	```

2. **Installare i requirements**
	```powershell
	pip install -r requirements.txt
	```

3. **Configurare pre-commit**
	```powershell
	pip install pre-commit
	pre-commit install
	```

## Flusso di lavoro

Tutto il flusso di lavoro è contenuto nel notebook `main.ipynb`, dove è possibile scegliere se eseguire l'esempio con il dataset *iris* o con il dataset *breast cancer* tramite il parametro "example_name"

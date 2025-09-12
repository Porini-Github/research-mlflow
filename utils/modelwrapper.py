from mlflow.pyfunc import PythonModel

class ModelWrapper(PythonModel):
    def __init__(self, model, target_col, date_col):
        self._model = model
        self.target_col = target_col
        self.date_col = date_col
    
    def predict(self, data_input):
        data = data_input.copy() # Here, we should add the preprocessing step or something similar.
        data['prediction'] = self._model.predict(data)
        return data
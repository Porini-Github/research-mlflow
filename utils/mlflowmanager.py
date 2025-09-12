import mlflow
import re
from datetime import datetime
from lib.modelwrapper import ModelWrapper

class MlFlowManager():
    def getTimeStamp(self):
        return re.sub("[^0-9]", "", str(datetime.now()))[:14]

    def setExperimentName(self, experiment_name):
        mlflow.set_experiment(experiment_name)
    
    def setStartRun(self, run_name):
        timestamp = self.getTimeStamp()
        mlflow.start_run(run_name = f'{run_name}_{timestamp}', nested=True)
    
    def setAutoLog(self):
         mlflow.autolog(log_datasets=False, log_models=False)
    
    def setLogParam(self, param_value):
        mlflow.log_params(param_value)
    
    def setLogMetric(self, metric_name, metric_value):
        mlflow.log_metric(metric_name, metric_value)
    
    def setLogModel(self, model, model_name, target_col, date_col):
        wrappedModel = ModelWrapper(model, target_col, date_col)
        mlflow.pyfunc.log_model(model_name, python_model=wrappedModel, registered_model_name=model_name, code_paths=['./']) 

    def setEndRun(self):
        mlflow.end_run()
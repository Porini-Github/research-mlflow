from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import ManagedIdentityCredential
import pandas as pd
import azureml.dataprep.fuse
import os

class DatasetManager():
    def __init__(self):
        from config.secret import SUBSCRIPTION_ID, RESOURCE_GROUP_NAME, WORKSPACE_NAME, DATASTORE_URI
        self.datastore_uri = DATASTORE_URI
        client_id = os.environ.get('DEFAULT_IDENTITY_CLIENT_ID')
        credential = ManagedIdentityCredential(client_id=client_id)
        self.ml_client = MLClient(credential=credential,
            subscription_id=SUBSCRIPTION_ID,
            resource_group_name=RESOURCE_GROUP_NAME,
            workspace_name=WORKSPACE_NAME
        )
    
    def RegisterDataset(self, input_file_path, data_asset_name):
        paths = [{"file": self.datastore_uri + input_file_path}]
        tbl = Data(
                    path=self.datastore_uri + input_file_path,
                    type=AssetTypes.URI_FILE,
                    name=data_asset_name
                )
        self.ml_client.data.create_or_update(tbl)
        version = self.ml_client.data._get_latest_version(data_asset_name).version
        return version
    
    def ReadDataset(self, data_asset_name, data_asset_version):
        data_asset = self.ml_client.data.get(data_asset_name, data_asset_version)
        df = pd.read_csv(data_asset.path)
        return df
    
    def GetDatasetList(self, data_asset_name):
        return self.ml_client.data.list(name=data_asset_name)
    
    def GetDatasetLastVersion(self, data_asset_name):
        return max([int(x.version) for x in self.GetDatasetList(data_asset_name)])
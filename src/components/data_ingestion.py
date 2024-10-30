import os 
import sys
import pandas as pd
from src.exxeption import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")
    raw_data_path:str=os.path.join("artifacts","data.csv")
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):
        try:
            df=pd.read_csv("notebook/data/data.csv")
            
            logging.info("Data Ingestion is successful")
            os.makedirs(os.path.driname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("trian test split is successful")
            trian_set, test_set =train_test_split(df,test_size=0.2,random_state=42)
            df.to_csv(self.ingestion_config.train_data_path,index=False,header=True)       
            df.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Data Ingestion is successful")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                
            )
        except Exception as e:
            raise CustomException(e,sys.exc_info())
if __name__ =="__main__":
   obj=DataIngestion()
   obj.initiate_data_ingestion
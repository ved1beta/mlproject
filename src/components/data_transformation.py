import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.exxeption import CustomException
from src.logger import logging
from src.utils import save_object

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder 

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        try:
            numerical_features = ["writing_score", "reading_score"]
            
            categorical_features = [
            "gender" ,
            "race_ethnicity",
            "parental_level_of_education",
            "lunch",
            "test_preparation_course",
        ]
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("std_scaler", StandardScaler()),
                ]
            )
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder()),
                    ("std_scaler", StandardScaler(with_mean=False)),
                ]
            )
            logging.info("Created numerical pipeline")
            logging.info("Created categorical pipeline")
            preprocessor = ColumnTransformer(
                [
                    ("numpipeline", num_pipeline, numerical_features),
                    ("catpipeline", categorical_pipeline, categorical_features),
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
            
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Starting data transformation")
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            logging.info(f"Read train data with shape: {train_data.shape}")
            logging.info(f"Read test data with shape: {test_data.shape}")
            
            preprocessor = self.get_data_transformer_object()
            logging.info("Created preprocessor object")
            
            transformer_column_name = "math_score"
            numerical_features = ["writing_score", "reading_score"]
            
            input_feature_train_df = train_data.drop(columns=[transformer_column_name], axis=1)
            target_feature_train_df = train_data[transformer_column_name]    

            input_feature_test_df = test_data.drop(columns=[transformer_column_name], axis=1)
            target_feature_test_df = test_data[transformer_column_name]
            logging.info("applying processing on train and test data")
            input_feature_train_df = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_df = preprocessor.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_df, np.array(target_feature_train_df)
                ]
            test_arr = np.c_[input_feature_test_df, np.array(target_feature_test_df)]   
            logging.info(f"Transformed train array shape: {train_arr.shape}")
            logging.info(f"Transformed test array shape: {test_arr.shape}")
            logging.info(f"Saved preprocessor object to {self.data_transformation_config.preprocessor_obj_file_path}")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor)
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
        except Exception as e:
            logging.error(f"Error in data transformation: {str(e)}")
            raise CustomException(e,sys)

if __name__ == "__main__":
    try:
        obj = DataTransformation()
        train_path = os.path.join("artifacts", "train.csv")
        test_path = os.path.join("artifacts", "test.csv")
        train_arr, test_arr, _ = obj.initiate_data_transformation(train_path, test_path)
        logging.info("Data transformation completed successfully")
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        raise CustomException(e, sys)
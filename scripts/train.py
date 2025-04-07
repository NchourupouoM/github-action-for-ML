import pandas as pd
import numpy as np
import skops.io as sio
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
import os

from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression 

import pickle

class ChurnModelPipeline:
    def __init__(self, data_path, target_column,output_dir_data, index_column=None, random_state=125):
        self.data_path = data_path
        self.target_column = target_column
        self.output_dir_data = output_dir_data
        self.index_column = index_column
        self.random_state = random_state
        self.model_pipeline = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.preprocessor = None 

    def load_and_prepare_dataset(self, drop_columns=None, nrows=None):
        print("Loading data...")
        df = pd.read_csv(self.data_path,index_col=self.index_column, nrows=nrows)
        if drop_columns:
            df = df.drop(drop_columns, axis=1)
        df = df.sample(frac=1)
        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )

        print(f"{type(self.X_train)}")

        print("Data load and split successfully.")
        # Sauvegarde des données au format NumPy
        # self.X_train.to_csv(os.path.join(self.output_dir, 'X_train.csv'),  index=False)
        self.X_test.to_csv(os.path.join(self.output_dir_data, 'X_test.csv'), index=False)
        # self.y_train.to_csv(os.path.join(self.output_dir, 'y_train.csv'),  index=False, header=True)
        self.y_test.to_csv(os.path.join(self.output_dir_data, 'y_test.csv'), index=False, header=True)

        print(f"test data saved to '{self.output_dir_data}' directory.")

    def build_processor(self, cat_cols, num_cols):
        print("building preprocessing pipeline...")
        numerical_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
        )
        categorical_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OrdinalEncoder())]
        )
        self.preprocessor = ColumnTransformer(
            transformers= [
                ("num", numerical_transformer, num_cols),
                ('cat', categorical_transformer, cat_cols)
            ]
        )
        print("preprocessing pipeline built.")

    def build_model_pipeline(self, k_best=5):
        print("Building model Pipeline...")
        feature_selector = SelectFromModel(
            LogisticRegression(max_iter=1000)
        )
        model = GradientBoostingClassifier(
            n_estimators=100, 
            random_state=self.random_state
        )
        train_pipeline = Pipeline(
            steps=[
                ("feature selection", feature_selector),
                ("GBmodel", model)           
            ]
        )

        self.model_pipeline = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("train", train_pipeline)
            ]
        )
        print("Model pipeline built.")

    def train_model(self):
        if self.model_pipeline is None:
            raise ValueError("Model pipeline is not initiate. Build the model pipeline first")
        print("Training the model...")
        self.model_pipeline.fit(self.X_train, self.y_train)

        # Enregistrer le modèle dans un fichier

        with open('../models/modele_logistic.pkl', 'wb') as file:
            pickle.dump(self.model_pipeline, file)

        print("Model training completed and saved successfuly.")

if __name__ == "__main__":
    #configuration
    data_file = "data/Churn_Modelling.csv"
    target_col = "Exited"
    drop_cols = ["RowNumber","CustomerId","Surname"]
    output_dir_data = "data/split datas"
    
    cat_columns = [1,2] # indices after dropping
    num_columns = [0,3,4,5,6,7,8,9]

    # Initialize and build pipeline 
    churn_pipeline = ChurnModelPipeline(data_path=data_file, target_column=target_col, output_dir_data=output_dir_data)
    churn_pipeline.load_and_prepare_dataset(drop_columns=drop_cols, nrows=1000)
    churn_pipeline.build_processor(cat_cols=cat_columns, num_cols=num_columns)
    churn_pipeline.build_model_pipeline()

    # Train and evaluate model 
    churn_pipeline.train_model()


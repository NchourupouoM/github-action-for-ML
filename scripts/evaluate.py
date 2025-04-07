import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
import skops.io as sio
from train import ChurnModelPipeline
import pickle
import numpy as np
import os
import pandas as pd

output_dir_data = "data/split datas"

class EvaluateChurnModel:
    def __init__(self, model_trained):
       self.model_pipeline = model_trained
       self.X_test = pd.read_csv(os.path.join(output_dir_data, 'X_test.csv'))
       self.y_test = pd.read_csv(os.path.join(output_dir_data, 'y_test.csv'))

    def evaluate_model(self):
        print("Evaluating the model...")
        predictions = self.model_pipeline.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        f1 = f1_score(self.y_test, predictions, average='macro')
        print(f"Accuracy: {round(accuracy*100,2)}%, F1 score: {round(f1, 2)}")

        return accuracy, f1 
    def plot_confusion_matrix(self):
        print("ploting confusion matrix...")
        prediction = self.model_pipeline.predict(self.X_test)
        cm = confusion_matrix(self.y_test, prediction, labels=self.model_pipeline.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.model_pipeline.classes_)
        disp.plot()
        plt.savefig("metrics/model_result.png", dpi=120)
        print("Confusion matrix saved as model_results.png")

    def save_matrics(self, accuracy, f1):
        print("saving metrics to file...")
        with open("metrics/metrics.txt", "w") as outfile:
            outfile.write(f"accuracy = {round(accuracy, 2)}, \n F1 score: {round(f1, 2)}\n")
        print("Metrics saved in metrics.txt")

    def plot_roc_curve(self):
        """Plot and save ROC curve for the classifier."""
        print("Plotting ROC curve...")
        y_probs = self.model_pipeline.predict_proba(self.X_test)[:, 1]  # Probabilities for class 1
        fpr, tpr, _ = roc_curve(self.y_test, y_probs)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_score(self.y_test, y_probs):.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig("metrics/roc_curve.png", dpi=120)
        print("ROC curve saved as 'roc_curve.png'.")

    def save_pipeline(self):
        print("saving pipeline to file...")
        sio.dump(self.model_pipeline, "models/churn_pipeline.skops")
        print("Pipeline saved as 'churn_prediction.skops'.")

if __name__ == "__main__":

    # Pour charger le modèle ultérieurement :
    filename = "models/modele_logistic.pkl"
    with open(filename, 'rb') as file:
        model = pickle.load(file)

    evaluate_model = EvaluateChurnModel(model_trained = model)

    accuracy, f1 = evaluate_model.evaluate_model()

    #plot CM and saved the metrics 
    evaluate_model.plot_confusion_matrix()
    evaluate_model.save_matrics(accuracy=accuracy, f1=f1)
    evaluate_model.plot_roc_curve()
    # save the pipeline 
    evaluate_model.save_pipeline()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score
import mlflow
import mlflow.sklearn
import joblib
import os

def run_training_pipeline(data_path):
    df=pd.read_csv(data_path)
    X=df.drop('churn',axis=1)
    y=df['churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("Customer_Churn_Analysis")
    with mlflow.start_run(run_name="Gradient_Boosting_Baseline"):
        params={
            "n_estimators" : 100,
            "learning_rate" : 0.01,
            "max_depth" : 3,
            "random_state" : 42
        }
        model=GradientBoostingClassifier(**params)
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred)
        }
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model,name="churn_model_v1")

        os.makedirs('models',exist_ok=True)
        joblib.dump(model, 'models/churn_model.pkl')
        print(f"Training Complete. Metrics: {metrics}")


if __name__ == "__main__":
    if not os.path.exists('data/churn_data.csv'):
        os.makedirs('data', exist_ok=True)
        data = pd.DataFrame(np.random.rand(200, 5),
                            columns=['tenure', 'monthly_charges', 'total_charges', 'support_calls', 'usage'])
        data['churn'] = np.random.randint(0, 2, 200)
        data.to_csv('data/churn_data.csv', index=False)

    run_training_pipeline('data/churn_data.csv')
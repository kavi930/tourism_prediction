
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from huggingface_hub import (
    hf_hub_download,
    HfApi
)

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# -------------------------------
# Hugging Face Repository
# -------------------------------

DATASET_REPO = "kavita89/tourism-dataset"

MODEL_REPO = "kavita89/tourism-model"

HF_TOKEN = os.getenv("HF_TOKEN")

# -------------------------------
# Download Dataset
# -------------------------------
X_train = pd.read_csv(
    hf_hub_download(
        repo_id=DATASET_REPO,
        filename="X_train_resampled.csv",
        repo_type="dataset",
        token=HF_TOKEN
    )
)

y_train = pd.read_csv(
    hf_hub_download(
        repo_id=DATASET_REPO,
        filename="y_train_resampled.csv",
        repo_type="dataset",
        token=HF_TOKEN
    )
).squeeze()

X_test = pd.read_csv(
    hf_hub_download(
        repo_id=DATASET_REPO,
        filename="X_test_processed.csv",
        repo_type="dataset",
        token=HF_TOKEN
    )
)

y_test = pd.read_csv(
    hf_hub_download(
        repo_id=DATASET_REPO,
        filename="y_test.csv",
        repo_type="dataset",
        token=HF_TOKEN
    )
).squeeze()

# -------------------------------
# ML Models
# -------------------------------

models = {

    "DecisionTree": (
        DecisionTreeClassifier(),
        {
            "max_depth":[5,10,20],
            "criterion":["gini","entropy"]
        }
    ),

    "RandomForest": (
        RandomForestClassifier(random_state=42),
        {
            "n_estimators":[100,200],
            "max_depth":[10,20]
        }
    ),

    "Bagging": (
        BaggingClassifier(random_state=42),
        {
            "n_estimators":[50,100]
        }
    ),

    "AdaBoost": (
        AdaBoostClassifier(random_state=42),
        {
            "n_estimators":[50,100],
            "learning_rate":[0.01,0.1,1]
        }
    ),

    "GradientBoosting": (
        GradientBoostingClassifier(random_state=42),
        {
            "n_estimators":[100,200],
            "learning_rate":[0.01,0.1],
            "max_depth":[3,5]
        }
    )
}

mlflow.set_experiment("Tourism_Project")

best_model = None
best_accuracy = 0

for model_name,(model,param_grid) in models.items():

    with mlflow.start_run(run_name=model_name):

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )

        grid.fit(X_train,y_train)

        model = grid.best_estimator_

        pred = model.predict(X_test)

        accuracy = accuracy_score(y_test,pred)

        precision = precision_score(y_test,pred)

        recall = recall_score(y_test,pred)

        f1 = f1_score(y_test,pred)

        mlflow.log_params(grid.best_params_)

        mlflow.log_metric("accuracy",accuracy)

        mlflow.log_metric("precision",precision)

        mlflow.log_metric("recall",recall)

        mlflow.log_metric("f1",f1)

        mlflow.sklearn.log_model(model,model_name)

        print(model_name)
        print(grid.best_params_)
        print("Accuracy:",accuracy)

        if accuracy>best_accuracy:

            best_accuracy=accuracy

            best_model=model

            best_model_name=model_name

# -------------------------------
# Save Best Model
# -------------------------------

# Ensure the model is saved in the correct directory within the project structure using an absolute path
MODEL_SAVE_DIR = "/content/tourism_project/model_building"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
LOCAL_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.pkl")
joblib.dump(best_model, LOCAL_MODEL_PATH)

# -------------------------------
# Upload Model
# -------------------------------

api=HfApi(token=HF_TOKEN)

api.upload_file(

    path_or_fileobj=LOCAL_MODEL_PATH, # Use the correctly saved local path

    path_in_repo="best_model.pkl", # Name in the Hugging Face repo

    repo_id=MODEL_REPO,

    repo_type="model"

)

print("="*50)
print("Best Model :",best_model_name)
print("Accuracy :",best_accuracy)
print("Uploaded to Hugging Face Model Hub")
print("="*50)

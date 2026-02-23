import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

the_dataset = pd.read_csv("./train.csv")

the_dataset["ST_interaction"] = the_dataset["ST depres sion"] * the_dataset["Exercise angina"]
the_dataset = pd.get_dummies(the_dataset, columns=["Heart Disease"], drop_first=True)
the_dataset.rename(columns={"Heart Disease_Presence": "Heart Disease"}, inplace=True)

heart_disease = the_dataset["Heart Disease"]
the_dataset.drop(columns=["Heart Disease", "BP", "id", "FBS over 120"], inplace=True)

num_cols = the_dataset.select_dtypes(include=['int64', 'float64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("scale", StandardScaler(), num_cols)
    ],
    remainder='passthrough'
)

# Models Creation
model = XGBClassifier(subsample= 1.0, scale_pos_weight= 1, n_estimators= 500, min_child_weight= 5, max_depth= 3, learning_rate= 0.1, gamma= 0.1, colsample_bytree= 1.0)

pipeline = Pipeline(steps=[
    ('scaling', preprocessor),
    ("model", model)
])

pipeline.fit(the_dataset, heart_disease)

test_dataset = pd.read_csv("./test.csv")
test_dataset["ST_interaction"] = test_dataset["ST depression"] * test_dataset["Exercise angina"]
test_dataset2 = test_dataset.copy()
test_dataset2.drop(columns=["id", "BP", "FBS over 120"], inplace =True)


heart_d = pipeline.predict(test_dataset2)
id = test_dataset[["id"]]
heart = pd.DataFrame(heart_d)
submit = pd.concat([id, heart], axis=1)
submit.rename(columns={0: "Heart Disease"}, inplace=True)
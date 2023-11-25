
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore")

output_file = f'model_C=1.0.bin'

df = pd.read_csv("Customer-Churn-Records.csv")
df.columns = df.columns.str.lower().str.lower().str.replace(' ','_')
df = df.drop(columns = ['rownumber', 'surname'],axis = 1)
categorical = list(df.select_dtypes(include=["object"]).columns)
numeric = list(df.select_dtypes(exclude=['object']).columns)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

#model training function
def train(df_train, y_train, lr):
    
    dicts = df_train[categorical + numeric].to_dict(orient='records')

    dv = DictVectorizer(sparse= True)
    X_train = dv.fit_transform(dicts)

    model = lr
    model.fit(X_train, y_train)

    return dv, model

#prediction function
def predict(df, dv, model):
    dicts = df[categorical + numeric].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


#Final model training and prediction
rfc = RandomForestClassifier(
        n_estimators=50,
        max_depth=3,
        random_state=1,
        n_jobs=-1,
        warm_start=True
    )

dv, model = train(df_full_train, df_full_train.exited.values, rfc)

#predict with the model
y_pred = predict(df_test, dv, model)
   
y_test = df_test.exited.values    

auc = roc_auc_score(y_test, y_pred)
print(f'auc={auc}')

# Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'The model saved to {output_file}')




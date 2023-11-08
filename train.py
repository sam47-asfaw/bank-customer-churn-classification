
import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


C= 1.0

n_splits = 5
output_file = f'model_C={C}.bin'



df = pd.read_csv("Customer-Churn-Records.csv")

df.columns = df.columns.str.lower().str.lower().str.replace(' ','_')


df = df.drop(columns = ['rownumber', 'surname'],axis = 1)


# In[29]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)


# In[30]:


numerical = [
    'customerid','creditscore','age',
    'tenure', 'balance', 'numofproducts', 'hascrcard',
    'isactivemember', 'estimatedsalary',
    'complain','satisfaction_score','point_earned'
]

categorical = ['geography','gender','card_type']


def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model =  RandomForestClassifier(
        n_estimators=10,
        max_depth= 10,
        random_state=1,
        n_jobs=-1,
        warm_start=True
    )

    model.fit(X_train, y_train)

    return dv, model

def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred



print('Final model training')

dv, model = train(df_full_train, df_full_train.exited.values, C=1.0)
y_pred = predict(df_test, dv, model)

y_test = df_test.exited.values
auc = roc_auc_score(y_test, y_pred)

print(f'auc={auc}')


# Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'The model saved to {output_file}')





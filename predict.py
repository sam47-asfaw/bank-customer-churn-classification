from flask import Flask, jsonify, request
from waitress import serve

import pickle

with open('./model_C=1.0.bin', 'rb') as f_in:
    (dv, model)= pickle.load(f_in)

app = Flask('Bank Churn')

@app.route('/predict', methods= ['POST'])
def predict():
    customer = request.get_json()
    X = dv.transform(customer)
    prediction = float(model.predict_proba(X)[0,1])
    churn = bool(prediction >= 0.36)

    result = {
        'Churn Probablility': prediction,
        'churn': churn
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)






# ### Data Preparation and Wrangling

# In[226]:


df.info()


# In[227]:


df.shape


# In[228]:


#change the column headers into lower case
df.columns = df.columns.str.lower().str.lower().str.replace(' ','_')


# In[229]:


df.columns


# In[230]:


#drop columns rownumber and surname
df = df.drop(columns = ['rownumber', 'surname'],axis = 1)


# In[231]:


df.nunique()


# In[232]:


#display the descriptive statistics of the numeric data
df.describe().T


# In[ ]:





# In[233]:


#check for duplicate and null values
df.duplicated().sum()


# In[234]:


df.isna().sum()


# In[235]:


#separate the dataset into numeric and categorical
numeric = df.select_dtypes(include=[np.number]).columns
categorical = df.select_dtypes(include=['object']).columns


# In[ ]:





# ### Exploratory Data Analysis (EDA)

# In[236]:


#plotting the dtatistical description of the numerical columns
description = df.describe().T
sns.barplot(data=description)
plt.xticks(rotation=45)
plt.show()


# In[ ]:





# In[237]:


#plotting the feature distributions
cols = [
    'creditscore','age','tenure','balance',
        'numofproducts','hascrcard','isactivemember',
        'estimatedsalary','complain','satisfaction_score','point_earned'
       ]
for col in cols:
    plt.figure(figsize=(15,5))
    plt.subplot(121)
    sns.distplot(df[col])
    plt.title(col)

    plt.show()


# In[ ]:





# In[238]:


plt.figure(figsize=(15, 8))
ax = sns.barplot(x=df['exited'].value_counts().index, y=df['exited'].value_counts().values)
plt.title("Non-Churned vs Churned Customer")
plt.ylabel('Count')
plt.xticks()
plt.show()


# ##### Insight: Number of Customers retained outweighs Churned Customers

# In[ ]:





# In[239]:


plt.figure(figsize=(10,5))
corr_matrix= df.corr(numeric_only=True).round(3)
sns.heatmap(corr_matrix, annot=True, cmap='Greens', vmin=-1, vmax= 1, fmt=".1f")


# #### Insight: There seems to be no high correlation between the features

# In[ ]:





# #### Split the Dataset into train(60%), test(20%), and val(20%)

# In[240]:


#set SEED value = 42
SEED = 42
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=SEED)

df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=SEED)


# In[241]:


len(df), len(df_full_train),len(df_test) ,len(df_train), len(df_val)


# In[242]:


# reset the index
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# In[243]:


#split the churn column from the rest of the dataframe
y_train = df_train.exited.values
y_val = df_val.exited.values
y_test = df_test.exited.values


# In[244]:


del df_train['exited']
del df_val['exited']
del df_test['exited']


# #### Feature Importance: Decline Rate and Risk ratio 
# ##### Identifying features that have higher effect on the target value
# ##### Decline Rate
# ##### Risk Ration
# ##### Mutual information

# In[245]:


#global decline rate
total_churn_rate = df_full_train.exited.mean().round(2)
total_churn_rate


# ##### Risk Ratio
# ##### if the group_decline_rate / total_churn_rate > 1 -> the group(feature) is more likely to exit
# ##### if the group_decline_rate / total_churn_rate < 1 -> the group(feature) is less likely to exit

# In[246]:


#create a new dataframe to perform Decline rate and risk ratio for the categorical columns
for c in categorical:
    print(c)
    df_group = df_full_train.groupby(c).exited.agg(['mean', 'count'])
    df_group['diff'] = total_churn_rate - df_group['mean']
    df_group['risk'] =  df_group['mean'] /total_churn_rate 
    display(df_group)
    print()
    print()


# In[ ]:





# In[247]:


# separate the columns as categorical and numeical
numerical = [
    'customerid','creditscore','age',
    'tenure', 'balance', 'numofproducts', 'hascrcard',
    'isactivemember', 'estimatedsalary',
    'complain','satisfaction_score','point_earned'
]

categorical = ['geography','gender','card_type']


# #### Feature Importance: Mutual information

# In[248]:


def mutual_info_decline_rate(series):
    return mutual_info_score(series, df_full_train.exited)


# In[249]:


mutual_info = df_full_train[categorical].apply(mutual_info_decline_rate)
mutual_info.sort_values(ascending=False)


# In[ ]:





# #### train and prediction function 

# In[250]:


def train(df_train, y_train,current_model, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = current_model
    model.fit(X_train, y_train)

    return dv, model

def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# In[ ]:





# ### Train Logistic Regression

# In[251]:


lr = LogisticRegression(C=1.0, max_iter=1000)

#train the model
dv, model = train(df_train, y_train, lr)

#predict with the model
prediction = predict(df_val, dv, model)
prediction


# ##### Find the optimal threshold to use

# In[252]:


thresholds = np.linspace(0,1,15)
thresholds
scores = []
for t in thresholds:
    churn_decision = (prediction >= t)
    model_acc = (y_val == churn_decision).mean()
    scores.append(model_acc)
    print("For threshold %.2f accuracy of model is %.3f" %(t,model_acc))


# #### Insight: the accuracy of the model stays constant starting at threshold 0.36
# #### t = 0.36 will be the threshold

# In[253]:


plt.plot(thresholds, scores)


# In[ ]:





# #### Confusion Matrix

# In[254]:


def confusion_matrix_dataframe(y_val, y_pred):
    scores = []

    thresholds = np.linspace(0, 1, 101)

    for t in thresholds:
        actual_positive = (y_val == 1)
        actual_negative = (y_val == 0)

        predict_positive = (y_pred >= t)
        predict_negative = (y_pred < t)

        tp = (predict_positive & actual_positive).sum()
        tn = (predict_negative & actual_negative).sum()

        fp = (predict_positive & actual_negative).sum()
        fn = (predict_negative & actual_positive).sum()

        scores.append((t, tp, fp, fn, tn))

    columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
    df_scores = pd.DataFrame(scores, columns=columns)

    return df_scores


# In[255]:


df_scores = confusion_matrix_dataframe(y_val, prediction)


# In[256]:


df_scores[::10]


# In[257]:


df_scores['p'] = df_scores.tp / (df_scores.tp + df_scores.fp)
df_scores['r'] = df_scores.tp / (df_scores.tp + df_scores.fn)


# In[258]:


plt.plot(df_scores.threshold, df_scores.p, label='precision')
plt.plot(df_scores.threshold, df_scores.r, label='recall')

plt.vlines(0.48, 0, 1, color='grey', linestyle='--', alpha=0.5)

plt.legend()
plt.show()


# #### F1 Score

# In[259]:


df_scores['f1'] = 2 * df_scores.p * df_scores.r / (df_scores.p + df_scores.r)


# In[260]:


df_scores.loc[df_scores.f1.argmax()]


# In[261]:


plt.figure(figsize=(10, 5))

plt.plot(df_scores.threshold, df_scores.f1)
plt.vlines(0.52, 0.45, 0.9, color='grey', linestyle='--', alpha=0.5)

plt.xticks(np.linspace(0, 1, 11))
plt.show()


# In[ ]:





# #### Cross-Validation

# In[262]:


kfold = KFold(n_splits=10, shuffle=True , random_state=1)


# In[263]:


for C in [0,0.001, 0.01, 0.1, 0.5, 1, 5, 10]:
    
    scores = []
    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.exited.values
        y_val = df_val.exited.values

        dv,model = train(df_train, y_train, lr)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)
        
    print('%s %.3f +.. %.2f'%(C, np.mean(scores), np.std(scores)))


# In[264]:


dv,model = train(df_full_train, df_full_train.exited.values,lr)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
auc


# In[ ]:





# ##### Hyperparameter Tuning

# In[265]:


kfold = KFold(n_splits=5, shuffle=True, random_state=1)

for C in [0.01, 0.1, 1, 10]:
    scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.exited.values
        y_val = df_val.exited.values

        dv, model = train(df_train, y_train,lr, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print('C=%4s, %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# #### Insight: Logistic Regression doesn't perform well

# In[266]:


len(y_val),len (y_test)


# ####  RandomForestClassifier() 

# In[220]:


train_dicts = df_train.to_dict(orient='records')
dv_new = DictVectorizer(sparse=True)
X_train = dv_new.fit_transform(train_dicts)

val_dicts = df_val.to_dict(orient='records')
X_val = dv_new.transform(val_dicts)


# In[278]:


scores = []

for d in tqdm([10, 15, 20, 25]):
    rfc = RandomForestClassifier(n_estimators=0,
                               max_depth=d,
                               random_state=1,
                                n_jobs=-1,
                               warm_start=True)

    for n in tqdm(range(10, 201, 10)):
        rfc.n_estimators = n
        rfc.fit(X_train, y_train)

        y_pred = rfc.predict(X_val)
        score = roc_auc_score(y_val, y_pred)
        
        scores.append((d, n, score))

columns = ['max_depth', 'n_estimators','roc_auc_score']
df_scores = pd.DataFrame(scores, columns=columns)


# In[279]:


df_scores


# ##### Insight: RandomForestClassififer perfoms better than LogisticRegression

# In[ ]:





# In[ ]:





# In[ ]:





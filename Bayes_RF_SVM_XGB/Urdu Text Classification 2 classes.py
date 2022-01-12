#!/usr/bin/env python
# coding: utf-8

# ### imports

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import  svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from matplotlib import pyplot as plt
import seaborn as sns
import xgboost as xgb


# In[17]:


# !pip install xlrd


# ### read the data

# In[2]:


data1  = pd.read_excel("data_files/dramas.xlsx")
data2 = pd.read_excel("data_files/Food and receip.xlsx")
data3 = pd.read_excel("data_files/Politics.xlsx")
data4 = pd.read_excel("data_files/software blog and forum reviews.xlsx")
data5 = pd.read_excel("data_files/sports.xlsx")


# In[3]:


total = len(data1)+len(data2)+len(data3)+len(data4)+len(data5)
total


# In[4]:


frames = [data1, data2, data3, data4, data5]
data = pd.concat(frames, sort=False)


# In[5]:


# data.isna().any()

pd.isnull(data).sum()


# ### Deciding annotation based on majority voting

# In[6]:


data_annotated = {}

for row in data.values:
    review =  row[3]
    a1, a2, a3 = row[5], row[6], row[7]
    if(a2 is np.nan):
        print("skipping....")
        continue
    else:
        data_annotated[review] = {'pos':0, 'neg':0, 'neu':0}
        data_annotated[review][a1]+=1
        data_annotated[review][a2]+=1
        data_annotated[review][a3]+=1
print("done")


# ###  Sorting

# In[7]:


data_annotated_sorted = {}
for k, v in data_annotated.items():
    data_annotated_sorted[k]= {k1: v1 for k1, v1 in sorted(v.items(), key=lambda item: item[1], reverse=True)}
    


# In[8]:


data_final = {}
for k, v in data_annotated_sorted.items():
    data_final[k] = list(v)[0]


# #### Filtering for binary classes only

# In[23]:


data2 ={}
for k, v in data_final.items():
    if(v=='neu'):
        continue
    else:
        data2[k]=v


# #### save to csv file

# In[37]:


df = pd.DataFrame(list(data2.items()),columns = ['text','class'])
df.to_excel("data_binary.xlsx")


# #### Special Cells here

# In[25]:


# documents = []
# y = []
# for k, v in data_final.items():
#     documents.append(k)
#     y.append(v)
    
# from sklearn.feature_extraction.text import CountVectorizer

# vectorizer = CountVectorizer(max_features=4000, min_df=5, max_df=0.7)

# X = vectorizer.fit_transform(documents).toarray()


# In[26]:


corpus  = {}

for text, tag in data2.items():
    splitted  =  text.split(" ")
    for w in splitted:
        if w not in corpus:
            corpus[w]=len(corpus)


# In[27]:


X = []
y=[]

for text, tag in data2.items():
    splitted  =  text.split(" ")
    tf = {}
    for w in splitted:
        if w not in tf:
            tf[w]=1
        else:
            tf[w]+=1
    vec = np.zeros(len(corpus))
    for w in splitted:
        vec[corpus[w]]= tf[w]
    X.append(vec)
    y.append(tag)


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.207, random_state=42)


# In[29]:


print("training set:\t", len(X_train))
print("test set:\t", len(X_test))


# In[30]:


def evaluate_model(y_test, y_pred, labels, title='Confusion Matrix', name="123", width=5, height=5):
    f1_macro = f1_score(y_test, y_pred, average='macro')*100
    f1_micro = f1_score(y_test, y_pred, average='micro')*100
    f1_weighted = f1_score(y_test, y_pred, average='weighted')*100
    accuracy = accuracy_score(y_test, y_pred)*100

    f1_macro = np.round(f1_macro, 2)
    f1_micro = np.round(f1_micro, 2)
    f1_weighted = np.round(f1_weighted, 2)
    accuracy = np.round(accuracy, 2)

    macro_p = precision_score(y_test, y_pred, average='macro')*100
    micro_p = precision_score(y_test, y_pred, average='micro')*100
    macro_r = recall_score(y_test, y_pred, average='macro')*100
    micro_r = recall_score(y_test, y_pred, average='micro')*100

    macro_p = np.round(macro_p, 2)
    micro_p = np.round(micro_p, 2)
    macro_r = np.round(macro_r, 2)
    micro_r = np.round(micro_r, 2)

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig = plt.figure(figsize=(width, height))
    ax= plt.subplot()
    ax = sns.heatmap(cm, annot=True, ax = ax, fmt='d'); #annot=True to annotate cells
    # labels, title and ticks
    ax.set_xlabel('Predicted labels\n'
                  +'F1=[macro:'+str(f1_macro) +"  micro:" +str(f1_micro)+"]   "\
                  +"P = [macro:" +str(macro_p) + "  micro:" +str(micro_p)+"]\n"\
                  +"R = [macro:" +str(macro_r) + "  micro:" +str(micro_r)+"]                   "\
                  +"Accuracy = " +str(accuracy)
                 )
    ax.set_ylabel('True labels'); 
    ax.set_title(title); 
    ax.xaxis.set_ticklabels(labels);
    ax.yaxis.set_ticklabels(labels);
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.savefig(name+".pdf", bbox_inches='tight')
    return  ax


# #### Multinomial Naive Bayes

# In[32]:


clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

labels=["neg", "pos"]
title = "Naive Bayes Confusion Matrix"
ax = evaluate_model(y_test, y_pred,labels,title=title,name="Naive Bayes",width=7, height=5)
plt.show()


# ####  Random Forest

# In[34]:


# clf = RandomForestClassifier(max_depth=200, n_estimators =15, random_state=0)
clf = RandomForestClassifier(max_depth=200, n_estimators =400, random_state=0) ## 80.68 %
# clf = RandomForestClassifier(max_depth=200, n_estimators =1000, random_state=0)
clf.fit(X_train, y_train)
y_pred  = clf.predict(X_test)


labels=["neg", "pos"]
title = "Random Forest Confusion Matrix"

ax = evaluate_model(y_test, y_pred,labels,title=title,name="Random Forest",width=7, height=5)
plt.show()


# #### xgboost

# In[38]:


X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)


model=xgb.XGBClassifier(random_state=1,learning_rate=0.01)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# model.score(X_test, y_test)
labels=["neg", "pos"]
title = "XGBOOST Confusion Matrix"

ax = evaluate_model(y_test, y_pred,labels,title=title,name="XGBOOST",width=7, height=5)
plt.show()


# ### SVM

# In[39]:


SVM = svm.SVC(C=4.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(X_train, y_train)
y_pred = SVM.predict(X_test)
acc = (y_test == y_pred).sum()*100/(len(X_test))
print("accuracy =  %.3f %% " % acc)

labels=["neg", "pos"]
title = "SVM Confusion Matrix"

ax = evaluate_model(y_test, y_pred,labels,title=title,name="SVM",width=7, height=5)
plt.show()


# In[ ]:





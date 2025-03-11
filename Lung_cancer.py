#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing and EDA

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time


# In[2]:


df=pd.read_csv("Lung_cancer_detection.csv")
df.head()


# In[3]:


df.columns


# In[4]:


df.info()


# # Changing the categorical values into numerical data

# In[5]:


df["GENDER"]=df["GENDER"].map({"M":1,"F":0})
df["LUNG_CANCER"]=df["LUNG_CANCER"].map({"YES":1,"NO":0})


# In[6]:


df.head()


# In[7]:


df.isnull().sum()


# # Checking Duplicates

# In[8]:


df.duplicated().sum()


# # Removing Duplicates

# In[9]:


df = df.drop_duplicates()


# In[10]:


df.head()


# In[11]:


df.info()


# # Correlation Heatmap

# In[12]:


plt.figure(figsize=(15,15))
ax=sns.heatmap(df.corr(),annot=True)
plt.show()


# In[13]:


df.describe()


# # Independent and Dependent Variable

# In[14]:


X=df.drop(columns="LUNG_CANCER",axis=1)
y=df["LUNG_CANCER"]


# In[15]:


X


# In[16]:


y


# # BoxPlot and Outliers

# In[17]:


fig,ax=plt.subplots(figsize=(15,15))
sns.boxplot(data=X,ax=ax)


# ### their is no need to remove outliers in age columns

# In[18]:


y.value_counts()


# In[19]:


df.shape


# # Splitting data into Train and Test

# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=42)


# In[21]:


X_train


# In[22]:


X_test


# In[23]:


y_train


# In[24]:


y_test


# # Applying smote technique

# In[25]:


from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\nResampled class distribution:")
print(pd.Series(y_train_resampled).value_counts())


# In[26]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)


# In[27]:


training_times = {}
accuracies = {}


# In[28]:


from sklearn.metrics import accuracy_score, confusion_matrix,roc_curve, auc


# # Logistic Regression

# In[29]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
start_time=time.time()
lr.fit(X_train_scaled,y_train_resampled)
end_time= time.time()-start_time
training_times["Logistic Regression"]=end_time


# In[30]:


y_pred_lr = lr.predict(X_test_scaled)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
print("Logistic Regression Accuracy:", accuracy_lr)
print("Confusion Matrix for Logistic Regression:\n", conf_matrix_lr)
accuracies["Logistic Regression"] = accuracy_score(y_test, y_pred_lr)


# # KNN

# In[31]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
start_time=time.time()
knn.fit(X_train_scaled,y_train_resampled)
end_time= time.time()-start_time
training_times["KNN"]=end_time


# In[32]:


y_pred_knn = knn.predict(X_test_scaled)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
print("KNN Accuracy:", accuracy_knn)
print("Confusion Matrix for KNN:\n", conf_matrix_knn)
accuracies["KNN"] = accuracy_score(y_test, y_pred_knn)


# # RandomForest

# In[33]:


from sklearn.ensemble import RandomForestClassifier
rdc=RandomForestClassifier()
start_time=time.time()
rdc.fit(X_train_resampled,y_train_resampled)
end_time= time.time()-start_time
training_times["Random Forest Classifier"]=end_time



# In[34]:


y_pred_rf = rdc.predict(X_test)  
accuracy_rf = accuracy_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)
print("Confusion Matrix for Random Forest:\n", conf_matrix_rf)
accuracies["Random Forest"] = accuracy_score(y_test, y_pred_rf)


# # GradientBoosting

# In[35]:


from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
start_time=time.time()
gbc.fit(X_train_resampled,y_train_resampled)
end_time= time.time()-start_time
training_times["Gradient Boosting Classifier"]=end_time


# In[36]:


y_pred_gbc = gbc.predict(X_test) 
accuracy_gbc = accuracy_score(y_test, y_pred_gbc)
conf_matrix_gbc = confusion_matrix(y_test, y_pred_gbc)
print("Gradient Boosting Accuracy:", accuracy_gbc)
print("Confusion Matrix for Gradient Boosting:\n", conf_matrix_gbc)
accuracies["Gradient Boosting"] = accuracy_score(y_test, y_pred_gbc)


# In[37]:



# # ADABOOSTClassifier

# In[39]:


from sklearn.ensemble import AdaBoostClassifier
abc=AdaBoostClassifier()
start_time=time.time()
abc.fit(X_train_resampled,y_train_resampled)
end_time= time.time()-start_time
training_times["ADA Boost Classifier"]=end_time


# In[40]:


y_pred_ada = abc.predict(X_test)
accuracy_ada = accuracy_score(y_test, y_pred_ada)
conf_matrix_ada = confusion_matrix(y_test, y_pred_ada)

print("AdaBoost Accuracy:", accuracy_ada)
print("Confusion Matrix for AdaBoost:\n", conf_matrix_ada)
accuracies["AdaBoost"] = accuracy_score(y_test, y_pred_ada)


# In[41]:


print("Training Times:")
for model_name, t in training_times.items():
    print(f"{model_name}: {t:.4f} seconds")

print("\nAccuracies:")
for model_name, acc in accuracies.items():
    print(f"{model_name}: {acc:.4f}")


# In[42]:


from sklearn.model_selection import GridSearchCV


# In[43]:


knn_param_grid = {
    'n_neighbors': list(range(1, 31)),
    'weights': ['uniform', 'distance']
}

knn_grid = GridSearchCV(estimator=KNeighborsClassifier(),
                        param_grid=knn_param_grid,
                        scoring='accuracy',
                        cv=5,
                        n_jobs=-1)
# Note: KNN uses scaled data
knn_grid.fit(X_train_scaled, y_train_resampled)

print("Best Parameters for KNN:", knn_grid.best_params_)
print("Best Accuracy for KNN:", knn_grid.best_score_)


# In[44]:


lr_param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],   # 'l2' is generally used; for 'l1', you need to specify solver='liblinear'
    'solver': ['lbfgs'], # lbfgs supports only l2 penalty
    'max_iter': [100, 200, 500]
}

lr_grid = GridSearchCV(estimator=LogisticRegression(random_state=42),
                       param_grid=lr_param_grid,
                       scoring='accuracy',
                       cv=5,
                       n_jobs=-1)
# Logistic Regression uses scaled data
lr_grid.fit(X_train_scaled, y_train_resampled)

print("Best Parameters for Logistic Regression:", lr_grid.best_params_)
print("Best Accuracy for Logistic Regression:", lr_grid.best_score_)


# In[45]:


ada_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.5, 1.0, 1.5, 2.0]
}

# Initialize GridSearchCV with AdaBoostClassifier using SAMME algorithm
ada_grid = GridSearchCV(
    estimator=AdaBoostClassifier(random_state=42, algorithm="SAMME"),
    param_grid=ada_param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)

# Fit on resampled training data
ada_grid.fit(X_train_resampled, y_train_resampled)

print("Best parameters for AdaBoost:", ada_grid.best_params_)
print("Best Accuracy for AdaBoost:", ada_grid.best_score_)


# In[46]:




plt.figure(figsize=(8, 6))

# Logistic Regression ROC
y_pred_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_prob_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')

# KNN ROC
y_pred_prob_knn = knn.predict_proba(X_test_scaled)[:, 1]
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_pred_prob_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)
plt.plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {roc_auc_knn:.2f})')

# AdaBoost ROC
y_pred_prob_ada = abc.predict_proba(X_test)[:, 1]
fpr_ada, tpr_ada, _ = roc_curve(y_test, y_pred_prob_ada)
roc_auc_ada = auc(fpr_ada, tpr_ada)
plt.plot(fpr_ada, tpr_ada, label=f'AdaBoost (AUC = {roc_auc_ada:.2f})')

# Plot the random chance line
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Top 3 Models')
plt.legend(loc="lower right")
plt.show()



import pickle

best_lr_model = lr_grid.best_estimator_

# Making Pickle File For Further Use
with open('best_logistic_regression.pkl', 'wb') as file:
    pickle.dump(best_lr_model, file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)


from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the Neural Network
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),  # Input Layer
    Dense(32, activation='relu'),  # Hidden Layer
    Dense(1, activation='sigmoid')  # Output Layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train_resampled, epochs=20, batch_size=16, validation_data=(X_test_scaled, y_test))

# Evaluate the model
loss, accuracy_nn = model.evaluate(X_test_scaled, y_test)
print(f"Neural Network Accuracy: {accuracy_nn:.4f}")

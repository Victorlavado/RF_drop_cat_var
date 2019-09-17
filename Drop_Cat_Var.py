#!/usr/bin/env python
# coding: utf-8

# In[23]:


# Import pandas and train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split

# Store the data from the csv file
data = pd.read_csv("melb_data.csv")


# In[24]:


# Generate X as the dataframe and y as the prediction target
y = data.Price
X = data.drop(["Price"], axis=1)


# In[25]:


# Break the dataframe into 2 pieces by applying train_test_split
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Drop columns with missing values
cols_with_missing = [col for col in X_train_full.columns if X_train_full[col].isnull().any()]
print(cols_with_missing)
X_train_reduced = X_train_full.drop(cols_with_missing, axis=1)
X_valid_reduced = X_valid_full.drop(cols_with_missing, axis=1)


# In[37]:


# Determine which columns have text elements and low cardinality
low_cardinality_cols = [cname for cname in X_train_reduced.columns if X_train_reduced[cname].nunique() < 10 and X_train_reduced[cname].dtype =="object"]


# In[38]:


# Determine which columns have floating and integers elements
numerical_cols = [cname for cname in X_train_reduced.columns if X_train_reduced[cname].dtype in ["int64", "float64"]]
my_cols = low_cardinality_cols + numerical_cols
X_train = X_train_reduced[my_cols].copy()
X_valid = X_valid_reduced[my_cols].copy()


# In[41]:


X_train.head()


# In[43]:


# Exclude all columns in the dataframe that have text elements
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])


# In[44]:


# Generate several Random Forest model to asses which one
# produces the lowest mean_absolute_error in the predictions
from sklearn.ensemble import RandomForestRegressor

model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion="mae", random_state=0)
model_4 = RandomForestRegressor(n_estimators=100, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]


# In[45]:


# Import mean_absolute_error as the criteria to assess
# model´s accuracy
from sklearn.metrics import mean_absolute_error

# Define score_model function to compare the actual prices
# with the predictions 
def score_model(model, X_train, X_valid, y_train, y_valid):
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    return mean_absolute_error(y_valid, pred)

# Iterate through the list of models to assess which gives
# the best accurary for the database
for i in range(0, len(models)):
    mae = score_model(models[i], drop_X_train, drop_X_valid, y_train, y_valid)
    print("Model %d MAE: %d" %(i+1, mae))


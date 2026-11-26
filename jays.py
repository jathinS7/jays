import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


DATA_PATH = "ml_practice_dataset_100MB.csv"  # place the CSV next to this script
data = pd.read_csv(DATA_PATH)

print(data.head())
print(data.tail())
data.info()
print(data.describe())
print(data.columns)







data = data.fillna(data.mean(numeric_only=True))

data[['city','device_type']] = data[['city','device_type']].fillna(data[['city','device_type']].mode().iloc[0])

print(data.isna().sum())


# Impute numeric columns once, with KNN (the earlier mean/median passes were overwritten)
from sklearn.impute import KNNImputer
data[['credit_score', 'annual_income']] = KNNImputer(n_neighbors=10).fit_transform(data[['credit_score', 'annual_income']])


print(data['city'].unique())
print(data[['device_type','subscription_type','gender']].drop_duplicates())
print(data['device_type'].unique())
print(data['subscription_type'].unique())
print(data['gender'].unique())

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder,LabelEncoder
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
		
X = data.drop(['customer_id', 'churn', 'review_text'],axis = 1)
y = data['churn']

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 32)


Onehot = ['device_type', 'subscription_type', 'preferred_language' , 'payment_method', 'marital_status']
Ordinal = ['city', 'owns_car', 'email_verified', 'is_student','gender']

preprocessor = ColumnTransformer(
	transformers = [
		('onehot',OneHotEncoder(drop = 'first' , sparse_output = False, handle_unknown = 'ignore'),
		Onehot),
		('ordinal' , OrdinalEncoder(handle_unknown = 'use_encoded_value' , unknown_value = -1),
		Ordinal)
	],
	remainder = 'passthrough'
)
pipeline = Pipeline([('preprocessor' ,preprocessor), ('scaler' , StandardScaler())])
print('pipeline')

X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)
print(X_train.shape)
print(X_test.shape)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score 


knn =  KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train ,y_train)
y_pred = knn.predict(X_test)
print(accuracy_score(y_test,y_pred))

nb = GaussianNB()
nb.fit(X_train , y_train)
y_pred = nb.predict(X_test)
print(accuracy_score(y_test,y_pred))


from sklearn.tree import DecisionTreeClassifier, plot_tree
dt = DecisionTreeClassifier(criterion = 'entropy', max_depth = 4, random_state=13, min_samples_split = 2)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print(accuracy_score(y_test,y_pred))

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=150)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test,y_pred))




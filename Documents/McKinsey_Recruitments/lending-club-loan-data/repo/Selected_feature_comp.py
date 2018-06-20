import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Final Feature set selection

#  top10_RFE
# Importing the data
df = pd.read_csv('top10_RFE.csv',
                 index_col=0,
                 low_memory=False)

array = df.values
X = array[:, 0:10]
y = array[:, 10]

# Fature scaling
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=0)

# Fitting XGBoost Classification to the Training set
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy top10_RFE: %.2f%%" % (accuracy * 100.0))
report_top10_RFE = classification_report(y_test, y_pred)

#   top10_ETC
# Importing the data
df = pd.read_csv('top10_ETC.csv',
                 index_col=0,
                 low_memory=False)
array = df.values
X = array[:, 0:10]
y = array[:, 10]

# Fature scaling
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=0)

# Fitting XGBoost Classification to the Training set
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy top10_ETC: %.2f%%" % (accuracy * 100.0))
report_top10_ETC = classification_report(y_test, y_pred)

#   top10_XGB
# Importing the data
df = pd.read_csv('top10_XGB.csv',
                 index_col=0,
                 low_memory=False)

array = df.values
X = array[:, 0:10]
y = array[:, 10]

# Fature scaling
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=0)

# Fitting XGBoost Classification to the Training set
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy top10_XGB: %.2f%%" % (accuracy * 100.0))
report_top10_XGB = classification_report(y_test, y_pred)

#   top7_RFE
# Importing the data
df = pd.read_csv('top7_RFE.csv',
                 index_col=0,
                 low_memory=False)
array = df.values
X = array[:, 0:7]
y = array[:, 7]

# Fature scaling
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=0)

# Fitting XGBoost Classification to the Training set
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy top7_RFE: %.2f%%" % (accuracy * 100.0))
report_top7_RFE = classification_report(y_test, y_pred)

#   top7_ETC
# Importing the data
df = pd.read_csv('top7_ETC.csv',
                 index_col=0,
                 low_memory=False)

array = df.values
X = array[:, 0:7]
y = array[:, 7]

# Fature scaling
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=0)

# Fitting XGBoost Classification to the Training set
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy top7_ETC: %.2f%%" % (accuracy * 100.0))
report_top7_ETC = classification_report(y_test, y_pred)

#   top7_XGB
# Importing the data
df = pd.read_csv('top7_XGB.csv',
                 index_col=0,
                 low_memory=False)

array = df.values
X = array[:, 0:7]
y = array[:, 7]

# Fature scaling
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=0)

# Fitting XGBoost Classification to the Training set
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy top7_XGB: %.2f%%" % (accuracy * 100.0))
report_top7_XGB = classification_report(y_test, y_pred)

#   top5_RFE
# Importing the data
df = pd.read_csv('top5_RFE.csv',
                 index_col=0,
                 low_memory=False)

array = df.values
X = array[:, 0:5]
y = array[:, 5]

# Fature scaling
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=0)

# Fitting XGBoost Classification to the Training set
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy top5_RFE: %.2f%%" % (accuracy * 100.0))
report_top5_RFE = classification_report(y_test, y_pred)

#   top5_ETC
# Importing the data
df = pd.read_csv('top5_ETC.csv',
                 index_col=0,
                 low_memory=False)

array = df.values
X = array[:, 0:5]
y = array[:, 5]

# Fature scaling
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=0)

# Fitting XGBoost Classification to the Training set
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy top5_ETC: %.2f%%" % (accuracy * 100.0))
report_top5_ETC = classification_report(y_test, y_pred)

#   top5_XGB
# Importing the data
df = pd.read_csv('top5_XGB.csv',
                 index_col=0,
                 low_memory=False)

array = df.values
X = array[:, 0:5]
y = array[:, 5]

# Fature scaling
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=0)

# Fitting XGBoost Classification to the Training set
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy top5_XGB: %.2f%%" % (accuracy * 100.0))
report_top5_XGB = classification_report(y_test, y_pred)

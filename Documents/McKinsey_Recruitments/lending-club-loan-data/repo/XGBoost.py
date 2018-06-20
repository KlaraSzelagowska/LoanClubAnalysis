# XGBoost model
import pandas as pd
from sklearn.metrics import accuracy_score
import time
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix

# Importing the data
df = pd.read_csv('top5_RFE.csv',
                 index_col=0,
                 low_memory=False)

array = df.values
X = array[:, 0:5]
y = array[:, 5]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=0)
# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting XGBoost Classification to the Training set
classifier = XGBClassifier()
t0 = time.time()
classifier.fit(X_train, y_train)
print("XGBoost model training time:", round(time.time() - t0, 3), "s")

# Predicting the Test set results
t1 = time.time()
y_pred = classifier.predict(X_test)
print("XGBoost model prediction time:", round(time.time() - t1, 3), "s")

# Evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy XGBoost model: %.2f%%" % (accuracy * 100.0))

# Making the Confusion Matrix
xgb_cm = confusion_matrix(y_test, y_pred)

# Receiver operating characteristic(ROC) and The Area Under ROC curve
y_predict_probabilities = classifier.predict_proba(X_test)[:, 1]
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, y_predict_probabilities)
xgb_roc_auc = auc(xgb_fpr, xgb_tpr)
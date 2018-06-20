# Random Forest
import pandas as pd
from sklearn.metrics import accuracy_score
import time
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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

# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(n_estimators=10,
                                    criterion='entropy',
                                    random_state=0)

t0 = time.time()
classifier.fit(X_train, y_train)
print("Random Forest model training time:", round(time.time() - t0, 3), "s")

# Predicting the Test set results
t1 = time.time()
y_pred = classifier.predict(X_test)
print("Random Forest model prediction time:", round(time.time() - t1, 3), "s")

# Evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Random Forest model: %.2f%%" % (accuracy * 100.0))

# Making the Confusion Matrix
rf_cm = confusion_matrix(y_test, y_pred)

# Receiver operating characteristic(ROC) and The Area Under ROC curve
y_predict_probabilities = classifier.predict_proba(X_test)[:, 1]
rf_fpr, rf_tpr, _ = roc_curve(y_test, y_predict_probabilities)
rf_roc_auc = auc(rf_fpr, rf_tpr)

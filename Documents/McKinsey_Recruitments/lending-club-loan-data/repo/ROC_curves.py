import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from matplotlib import pyplot as plt

# Naive Bayes
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

# Fitting Naive Bayes to the Training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Receiver operating characteristic(ROC) and The Area Under ROC curve
y_predict_probabilities = classifier.predict_proba(X_test)[:, 1]
nb_fpr, nb_tpr, _ = roc_curve(y_test, y_predict_probabilities)
nb_roc_auc = auc(nb_fpr, nb_tpr)

# Random Forest
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

classifier.fit(X_train, y_train)

# XGBoost model
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
classifier.fit(X_train, y_train)

# Receiver operating characteristic(ROC) and The Area Under ROC curve
y_predict_probabilities = classifier.predict_proba(X_test)[:, 1]
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, y_predict_probabilities)
xgb_roc_auc = auc(xgb_fpr, xgb_tpr)

# Receiver operating characteristic(ROC) and The Area Under ROC curve
y_predict_probabilities = classifier.predict_proba(X_test)[:, 1]
rf_fpr, rf_tpr, _ = roc_curve(y_test, y_predict_probabilities)
rf_roc_auc = auc(rf_fpr, rf_tpr)

# ROC Curv
plt.figure()
plt.plot(nb_fpr,
         nb_tpr,
         color='darkorange',
         lw=2,
         label='Naive Bayes (area = %0.2f)' % nb_roc_auc)

plt.plot(rf_fpr,
         rf_tpr,
         color='darkgreen',
         lw=2,
         label='Random Forest (area = %0.2f)' % rf_roc_auc)

plt.plot(xgb_fpr,
         xgb_tpr,
         color='purple',
         lw=2,
         label='Random Forest (area = %0.2f)' % xgb_roc_auc)

plt.plot([0, 1],
         [0, 1],
         color='navy',
         lw=2,
         linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

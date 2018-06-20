import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier

# Importing the data
df = pd.read_csv('pre_sele_feature_out.csv',
                 index_col=0,
                 low_memory=False)

print("Features list:", df.columns.tolist())

'''print('~~~ETC~~~')
# Feature Selection by Feature Importance with Extra Trees Classifier
array = df.values
X = array[:, 0:63]
y = array[:, 63]

# Feature Scaling
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Feature extraction
classifier = ExtraTreesClassifier()
classifier.fit(X, y)
print("Fature importance by ExtraTreesClassifier:", classifier.feature_importances_)
'''
'''
# Fature selection by Recursive Feature Elimination for top 10 features
array = df.values
X = array[:, 0:63]
y = array[:, 63]

# Fature Scaling
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Feature extraction
regressor = LogisticRegression()

rfe = RFE(regressor, 10)
fit = rfe.fit(X, y)
print("Recursive Feature Elimination for top 10")
print("top10 Num Features: %d" % (fit.n_features_,))
print("top10 Selected Features: %s" % (fit.support_,))
print("top10 Feature Ranking: %s" % (fit.ranking_,))
'''
'''print('~~~ETC top7~~~')
# Fature selection by Recursive Feature Elimination for top 7 features
array = df.values
X = array[:, 0:63]
y = array[:, 63]

# Fature Scaling
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Feature extraction
regressor = LogisticRegression()

rfe = RFE(regressor, 7)
fit = rfe.fit(X, y)
print("Recursive Feature Elimination for top 7")
print("top7 Num Features: %d" % (fit.n_features_,))
print("top7 Selected Features: %s" % (fit.support_,))
print("top7 Feature Ranking: %s" % (fit.ranking_,))
'''
print('~~~ETC top5~~~')
# Fature selection by Recursive Feature Elimination for top 5 features
array = df.values
X = array[:, 0:63]
y = array[:, 63]

# Fature Scaling
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Feature extraction
regressor = LogisticRegression()

rfe = RFE(regressor, 5)
fit = rfe.fit(X, y)
print("Recursive Feature Elimination for top 5")
print("top5 Num Features: %d" % (fit.n_features_,))
print("top5 Selected Features: %s" % (fit.support_,))
print("top5 Feature Ranking: %s" % (fit.ranking_,))
'''
print('~~~XGB~~~')
# XGBoost use feature importance for feature selection
array = df.values
X = array[:, 0:63]
y = array[:, 63]

# Feature scaling
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Fit model on training data
classifier = XGBClassifier()
classifier.fit(X, y)

# Plot feature importance
print("XGBoost Feature importance", classifier.feature_importances_)
'''
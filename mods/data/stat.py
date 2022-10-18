import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

data = pd.read_csv('hsja4_0.csv')
data2 = pd.read_csv('hsja4eval_0.csv')
data = data.sample(frac=0.1)
data2 = data2.sample(frac=0.1)


# The target variable is 'score'
y = data.iloc[:,-2]
X = data.iloc[:,:77]

y1 = data2.iloc[:,-2]
X1 = data2.iloc[:,:77]

mean = X.mean()
std = X.std()

mean1 = X1.mean()
std1 = X1.std()

res = mean - mean1
res2 = std - std1

# Split the data into train and test data:
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# # model = RandomForestRegressor(random_state=0, n_estimators=300, max_depth=100, criterion='mse', max_features='auto', min_samples_leaf=1, min_samples_split=2)
# model = RandomForestClassifier(n_estimators=300, verbose=1)
# model.fit(X_train, y_train)
# print('Fitted...')

# # ..evaluate
# # y_pred = model.predict(X_test)6
# # r2 = r2_score(y_test, y_pred)

# r2 = model.score(X_test, y_test)
# print('R2 is:', r2)

# # print shapley values
# shap_values = shap.TreeExplainer(model).shap_values(X_train)
# print('done')
# shap.summary_plot(shap_values, X_train, plot_type="bar")
# # shap.plots.beeswarm(shap_values)

# # joblib.dump(model, "rfc.joblib")





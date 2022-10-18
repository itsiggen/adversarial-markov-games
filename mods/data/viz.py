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
data = data.sample(frac=0.001)
data2 = data2.sample(frac=0.001)


# L = [0,1,2,3,4,5,25,26,27,28,29,30,108,109]
# x = data.iloc[:,:110]
# f = data.iloc[:, L]
# y = data.iloc[:,-1]

# tsne = TSNE(n_components=2, verbose=1)
# z = tsne.fit_transform(x)

# df = pd.DataFrame()
# df["y"] = y
# df["comp-1"] = z[:,0]
# df["comp-2"] = z[:,1]

# sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
#                 palette=sns.color_palette("hls", 2),
#                 data=df).set(title="T-SNE projection")


# ben = data.loc[data['label'] == 0]
# mal = data.loc[data['label'] == 1]
# ben2 = data2.loc[data2['label'] == 0]
# mal2 = data2.loc[data2['label'] == 1]

# The target variable is 'score'
y = data.iloc[:,-2]
X = data.iloc[:,:77]

# Split the data into train and test data:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# model = RandomForestRegressor(random_state=0, n_estimators=300, max_depth=100, criterion='mse', max_features='auto', min_samples_leaf=1, min_samples_split=2)
model = RandomForestClassifier(n_estimators=300, verbose=1)
model.fit(X_train, y_train)
print('Fitted...')

# ..evaluate
# y_pred = model.predict(X_test)
# r2 = r2_score(y_test, y_pred)

r2 = model.score(X_test, y_test)
print('R2 is:', r2)

# print shapley values
shap_values = shap.TreeExplainer(model).shap_values(X_train)
print('done')
shap.summary_plot(shap_values, X_train, plot_type="bar")
# shap.plots.beeswarm(shap_values)

# joblib.dump(model, "rfc.joblib")





import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA, TruncatedSVD

newhousing = pd.read_csv("newhousing.csv")


newhousing

newhousing.head(10)

len(newhousing)

newhousing.info()

newhousing.duplicated().sum()

newhousing.eq(0).sum()

newhousing.describe().T

newhousing.columns

# Split data into features and labels
newhousing.columns
features = ['area', 'bedrooms', 'bathrooms','stories', 'mainroad']

X = newhousing[features]
y = newhousing.price

X.head()

y.head()


X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=5, test_size=0.15)

lr_model = LinearRegression().fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

MSE = mean_squared_error(y_test,y_pred)
MSE = np.sqrt(MSE)
print('Mean Squared Error:', MSE)

mean_absolute_error(y_test, y_pred)


explained_variance_score(y_test, y_pred)

newhousing['price'].describe()

# Linear Regression Model without Dimensionality Reduction
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)

lr = LinearRegression()

# Train the model
model = lr.fit(X_train, y_train)

# Prediction
y_pred =  lr.predict(X_test)

# Accuracy Score
lr.score(X_test, y_test).round(10)

#  Linear Regression Model with PCA
pca = PCA(n_components = 5, whiten = True)
pca.fit(X)
pca_X = pca.transform(X)

pca_X_train, pca_X_test, pca_y_train, pca_y_test = train_test_split(pca_X, y, test_size = 0.2, random_state = 4)

pca_lr = LinearRegression()

# Train the model
pca_model = pca_lr.fit(pca_X_train, pca_y_train)

# Prediction
pca_y_pred =  pca_lr.predict(pca_X_test)

# Accuracy Score
pca_lr.score(pca_X_test, pca_y_test).round(4)

# Linear Regression Model with SVD
svd = TruncatedSVD(n_components = 5)
svd.fit(X)
svd_X = pca.transform(X)

svd_X_train, svd_X_test, svd_y_train, svd_y_test = train_test_split(svd_X, y, test_size = 0.2, random_state = 4)

svd_lr = LinearRegression()

# Train the model
svd_model = svd_lr.fit(svd_X_train, svd_y_train)

# Prediction
pca_y_pred =  svd_lr.predict(svd_X_test)

# Accuracy Score
svd_lr.score(svd_X_test, svd_y_test).round(4)





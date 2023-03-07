import pandas as pd
from sklearn.tree import DecisionTreeRegressor
MELBOURNE_FILE_PATH = 'input/melbourne-housing-snapshot/melb_data.csv'

melbourne_data = pd.read_csv(MELBOURNE_FILE_PATH)
# DataFrame.columns returns Index object which contains list of columns and dtype
print(melbourne_data.columns)
# now we drop houses whith missing values
melbourne_data = melbourne_data.dropna(axis=0)

# select specific column with DataFrame.name_of_column
y = melbourne_data.Price
print(y)

# select features
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
print(X.describe())
print(X.head())

# Define model using scikit-learn
melbourne_model = DecisionTreeRegressor(random_state=1)
# Fit model
melbourne_model.fit(X, y)

print('Предсказание для следующих 5 домов:')
print(X.head())
print('Вот мое предсказание')
print(melbourne_model.predict(X.head()))
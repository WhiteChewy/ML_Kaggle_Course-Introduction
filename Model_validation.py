import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

MELBOURNE_FILE_PATH = 'input/melbourne-housing-snapshot/melb_data.csv'

melbourne_data = pd.read_csv(MELBOURNE_FILE_PATH)
filtered_melbourne_data = melbourne_data.dropna(axis=0)
y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]

# Fit the model
melbourn_model = DecisionTreeRegressor()
melbourn_model.fit(X, y)

#  Calculating MAO In-split score
predicted_home_prcies = melbourn_model.predict(X)
mae = mean_absolute_error(y, predicted_home_prcies)
print(mae)

# split data
train_x, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# define the model
melbourn_model = DecisionTreeRegressor()
# fit model
melbourn_model.fit(train_x, train_y)
#get predicted prices on validation data
val_predictions = melbourn_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
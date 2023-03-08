import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

MELBOURNE_FILE_PATH = 'input/melbourne-housing-snapshot/melb_data.csv'


def get_mae(max_leaf_nodes: int, train_X, val_X, train_y, val_y) -> int:
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae


melbourne_data = pd.read_csv(MELBOURNE_FILE_PATH)
melbourne_data = melbourne_data.dropna(axis=0)
y = melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
train_x, val_x, train_y, val_y = train_test_split(X, y, random_state=1)
for max_leaf in [5, 50, 500, 5000]:
    mae_val = get_mae(max_leaf, train_x, val_x, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf, mae_val))

import pandas as pd
MELBOURNE_FILE_PATH = 'input/melbourne-housing-snapshot/melb_data.csv'

melbourne_data = pd.read_csv(MELBOURNE_FILE_PATH)
# .describe() make DataFrame from cvs
print(melbourne_data.describe())
# [column_name] - gets column, u can get avg, max, min, std, 25%, 50%, 75% values of column as described below
print(round(melbourne_data.Price.mean()))

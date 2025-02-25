# The TableVectorizer
####################################
# %%
# Load the data and split out the column to predict
import numpy as np
import pandas as pd

df = pd.read_csv('employees_salaries.csv')
y = df['salary']
df = df.drop('salary', axis=1)

results = {} # We will store the results here
# %%
# The default TableVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from skrub import TableVectorizer
model = make_pipeline(TableVectorizer(), RandomForestRegressor())
results[model] = cross_validate(model, df, y, scoring='r2', n_jobs=-1)
print(f"R2 score {np.mean(results[model]['test_score']):.2f} in {np.mean(results[model]['fit_time']):.2f} seconds")
model

# %%
# The default tabular_learner
from skrub import tabular_learner
model = tabular_learner(RandomForestRegressor())
results[model] = cross_validate(model, df, y, scoring='r2', n_jobs=-1)
print(f"R2 score {np.mean(results[model]['test_score']):.2f} in {np.mean(results[model]['fit_time']):.2f} seconds")
model

# %%
# Let us vary the high_cardinality parameter
from skrub import StringEncoder
model['tablevectorizer'].high_cardinality = StringEncoder()
results[model] = cross_validate(model, df, y, scoring='r2', n_jobs=-1)
print(f"R2 score {np.mean(results[model]['test_score']):.2f} in {np.mean(results[model]['fit_time']):.2f} seconds")
model

# %%
# And if we want, we can use LLMs to embed the high cardinality strings
# Only 2 cores, to avoid blowing up the memory
from skrub import TextEncoder
model['tablevectorizer'].high_cardinality = TextEncoder()
results[model] = cross_validate(model, df, y, scoring='r2', n_jobs=-2)
print(f"R2 score {np.mean(results[model]['test_score']):.2f} in {np.mean(results[model]['fit_time']):.2f} seconds")
model

# Beyond the string encoders, we also have the DatetimeEncoder

# %%


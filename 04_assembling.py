# Data operations, cross-validated
####################################
#
# Actual data preparation involves multiple steps, across multiple tables
# "pipeline" API is tedious
#
# WIP: https://github.com/skrub-data/skrub/pull/1233

# %%
# Import a more complex dataset, with multiple tables
import skrub
from skrub.datasets import fetch_credit_fraud

dataset = fetch_credit_fraud()
# A first table, linking baskets to fraud
baskets = dataset.baskets
skrub.TableReport(baskets)
# %%
# A second table, which gives the products in each basket
products = dataset.products
skrub.TableReport(products)
# %%
# We need to 1) group the products by basket, 2) join the two tables
#
# An example of basket looks like this
next(iter(dataset.products.groupby('basket_ID')))[1]

# %%
# A groupby calls for an aggregation. How to aggregate the items, models: strings?
# We'll vectorize the table
vectorizer = skrub.TableVectorizer(high_cardinality=skrub.StringEncoder())
vectorized_products = vectorizer.fit_transform(products)

# %%
# We can now aggregate the products and join the tables: pandas operations
aggregated_products = (
   vectorized_products.groupby("basket_ID").agg("mean").reset_index()
)
aggregated_products
# %%
baskets = baskets.merge(
    aggregated_products, left_on="ID", right_on="basket_ID"
).drop(columns=["ID", "basket_ID"])
baskets

# %%
# And we can now train a model
y = baskets['fraud_flag']
X = baskets.drop('fraud_flag', axis=1)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_jobs=-1)
from sklearn.model_selection import cross_val_score
cross_val_score(model, X, y, scoring='roc_auc')
# %%
# But we are back to the same problem as in the first section:
# We're in pandas' land. When comes new data, how to apply the same transformations?
# How to cross-validate, or tune the data-preparation steps?


# skrub comes with a way to change ever so slightly what we did above
# We define our inputs as "variables"
products = skrub.var("products", dataset.products)
products
# %%
# Now we define our "x" and "y" variables
baskets = skrub.var("baskets", dataset.baskets[["ID"]]).skb.mark_as_x()
fraud_flags = skrub.var("fraud", dataset.baskets["fraud_flag"]).skb.mark_as_y()

# %%
# We can now proceed almost as above to prepare the data
# We vectorize the products, with a slightly different API
from skrub import selectors as s
vectorized_products = products.skb.apply(vectorizer, cols=s.all() - "basket_ID")
vectorized_products
# %%
# We aggregate the products
aggregated_products = vectorized_products.groupby("basket_ID").agg("mean").reset_index()
aggregated_products
# %%
# And we join the tables
baskets = baskets.merge(aggregated_products, left_on="ID", right_on="basket_ID")
baskets = baskets.drop(columns=["ID", "basket_ID"])
baskets
# %%
# And we do the prediction
predictions = baskets.skb.apply(RandomForestClassifier(), y=fraud_flags)
predictions

# %%
# What's the big deal? We now have a graph of computations, that we can optimize
# Or apply to new data

# For instance, let's optimize a bit our vectorization of the products.
# We just need to change a bit the above code
encoder = skrub.choose_from(
    {
        "MinHash": skrub.MinHashEncoder(),
        "LSA": skrub.StringEncoder(),
    },
    name="encoder",
)
vectorizer = skrub.TableVectorizer(high_cardinality=encoder)
vectorized_products = products.skb.apply(vectorizer, cols=s.all() - "basket_ID")
aggregated_products = vectorized_products.groupby("basket_ID").agg("mean").reset_index()
baskets = baskets.merge(aggregated_products, left_on="ID", right_on="basket_ID")
baskets = baskets.drop(columns=["ID", "basket_ID"])
predictions = baskets.skb.apply(RandomForestClassifier(), y=fraud_flags)

search = predictions.skb.get_grid_search(n_jobs=-1)
search.get_cv_results_table()
# %%

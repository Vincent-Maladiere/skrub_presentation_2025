# srkub for tables: Less wrangling, more machine learning

## Preparing one table

- TableVectorizer and many helpers

- Importance of distinguish train-time and test-time operations
    - Things like detection of column type might give different answers

- Very robust part of skrub, still improving


## Open-ended sequence of operations

- Modern data preparation for machine learning
    - Machine-learning steps in the middle of the pipeline
    - Vectorization of tables (easier aggregations)
    - But also useful to impute

- API: versatile, but standard-looking
    - Wraps standards objects such as dataframe, and records
    - Can be applied to polars, ibis...

- Benefits 
    - Can be put in production (applied to new data)
    - Every aspect is tunable
    - Visual debugging helpers

- Long term
    - Subsampling for faster interactive work
    - Connecting directly to databases
    - Out-of-core execution (from a database engine)

- skrub is community driven
    - https://github.com/skrub-data/skrub
    - Try it out... help us
    - Every help is useful, including using and communicating

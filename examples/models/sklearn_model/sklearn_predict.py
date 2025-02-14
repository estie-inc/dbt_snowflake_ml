import pandas as pd
from sklearn.impute import SimpleImputer
from snowflake.ml.registry import registry


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
  df = df[["PCLASS", "SEX", "AGE", "SIBSP", "PARCH", "FARE", "EMBARKED"]]
  df["PCLASS"] = pd.Categorical(df["PCLASS"], categories=[1, 2, 3])
  df["SEX"] = pd.Categorical(df["SEX"], categories=["male", "female"])
  df["EMBARKED"] = pd.Categorical(df["EMBARKED"], categories=["C", "Q", "S"])
  return pd.get_dummies(df, columns=["PCLASS", "SEX", "EMBARKED"])


def model(dbt, session):
  dbt.config(
    materialized="table",
    python_version="3.11",
    packages=["snowflake-ml-python", "pandas", "scikit-learn"],
  )

  dataset = dbt.ref("titanic3")

  data = dataset.to_pandas()

  x = preprocess(data)

  imputer = SimpleImputer()
  x = imputer.fit_transform(x)

  reg = registry.Registry(session=session)

  model_ref = dbt.ref("sklearn_model")
  mv = reg.get_model(model_ref.table_name).default
  data["PREDICTED"] = mv.run(x, function_name="PREDICT")

  return data

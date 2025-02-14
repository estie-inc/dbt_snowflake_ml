import datetime
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from snowflake.ml.model import model_signature


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
  df = df[["PCLASS", "SEX", "AGE", "SIBSP", "PARCH", "FARE", "EMBARKED"]]
  df["PCLASS"] = pd.Categorical(df["PCLASS"], categories=[1, 2, 3])
  df["SEX"] = pd.Categorical(df["SEX"], categories=["male", "female"])
  df["EMBARKED"] = pd.Categorical(df["EMBARKED"], categories=["C", "Q", "S"])
  return pd.get_dummies(df, columns=["PCLASS", "SEX", "EMBARKED"])


def model(dbt, session):
  dbt.config(
    materialized="model",
    python_version="3.11",
    packages=["snowflake-ml-python", "pandas", "scikit-learn"],
  )

  dataset = dbt.ref("titanic3")

  data = dataset.to_pandas()

  x = preprocess(data)
  y = data["SURVIVED"]

  imputer = SimpleImputer()
  x = imputer.fit_transform(x)

  model = SVC()
  model.fit(x, y)

  return {
    "model": model,
    "signatures": {"predict": model_signature.infer_signature(x, y)},
    "version_name": datetime.datetime.today().strftime("V%Y%m%d"),
    "metrics": {"r2_score": model.score(x, y)},
    "comment": f"r2_score: {model.score(x, y)}",
    "set_default": True,
  }

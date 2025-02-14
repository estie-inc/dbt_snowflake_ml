import datetime
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.svm import SVC
from snowflake.ml.model import custom_model, model_signature


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

  imputer = IterativeImputer()
  x = imputer.fit_transform(x)

  model = SVC()
  model.fit(x, y)

  mc = custom_model.ModelContext(
    models={
      "model": model,
      "imputer": imputer,
    },
  )

  class CustomModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
      super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
      model = self.context.model_ref("model")
      imputer = self.context.model_ref("imputer")

      x = preprocess(input)

      x = imputer.transform(x)

      return pd.DataFrame({'output': model.predict(x)})

  return {
    "model": CustomModel(mc),
    "signatures": {"predict": model_signature.infer_signature(dataset.drop("SURVIVED"), y)},
    "version_name": datetime.datetime.today().strftime("V%Y%m%d"),
    "metrics": {"r2_score": model.score(x, y)},
    "comment": f"r2_score: {model.score(x, y)}",
    "set_default": True,
  }

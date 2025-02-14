# dbt_snowflake_ml
dbt package integrates Snowflake ML


## Requirements
- https://github.com/dbt-labs/dbt-snowflake adapter


## Installation

Refer this repository from `packages.yml` of your dbt project.
```yaml
packages:
  - git: "https://github.com/estie-inc/dbt_snowflake_ml.git"
    revision: main
```


## Usage

Examples are also available under `/examples` directory.


### Store trained ML model

Define a dbt Python model with `materialized="model"`.
dict returned from the model will be fed into `snowflake.ml.registry.Registry.log_model` as arguments and model will be stored in Snowflake ML Model Registry.
See https://docs.snowflake.com/en/developer-guide/snowpark-ml/reference/latest/api/registry/snowflake.ml.registry.Registry for available arguments.

There are some special arguments:
- `model_name`: unavailable and forced to `dbt.this.identifier`
- `conda_dependencies`: unavailable and generated from dbt.config.packages
- `set_default`: additionally available and stored model version will be set to default version if this argument is set to true

```python
import datetime
from snowflake.ml.model import model_signature


def model(dbt, session):
  dbt.config(
    materialized="model",
    python_version="3.11",
    packages=["snowflake-ml-python", "pandas", "scikit-learn"],
  )

  x = ... # training features
  y = ... # training labels
  model = ... # train your model

  return {
    "model": model,
    "signatures": {"predict": model_signature.infer_signature(x, y)},
    "version_name": datetime.datetime.today().strftime("V%Y%m%d"),
    "metrics": {"r2_score": model.score(x, y)},
    "comment": f"r2_score: {model.score(x, y)}",
    "set_default": True,
  }
```

### Using stored ML model
There are two ways to use stored ML models.

#### SQL model
You can directly use stored model on SQL by using model methods.
https://docs.snowflake.com/en/sql-reference/commands-model-function#calling-model-methods

```sql
select
  feature_source.id,
  {{ ref('trained_model') }}!predict(feature_source.* exclude (id)):output as prediction
from {{ ref('feature_source') }} as feature_source
```

#### Python model
You can use stored model through Python model by using Snowflake Model Registry API.
https://docs.snowflake.com/en/developer-guide/snowflake-ml/model-registry/overview
Note that `dbt.ref` returns `snowflake.snowpark.Table` constructed with model identifier. `snowflake.snowpark.Table` is just a wrapper and you can unwrap the identifier with `table_name` field.

```python
from snowflake.ml.registry import registry


def model(dbt, session):
  dbt.config(
    materialized="table",
    python_version="3.11",
    packages=["snowflake-ml-python", "pandas", "scikit-learn"],
  )

  data = dbt.ref("feature_source").to_pandas()
  result = data[["ID"]]

  model_ref = dbt.ref("trained_model")
  mv = reg.get_model(model_ref.table_name).default
  result["PREDICTION"] = mv.run(data.drop("ID"), function_name="PREDICT")

  return data
```

### Best practices

- You can package model weights, imputer parameters, and feature preprocessing logics into Snowflake ML model by using `snowflake.ml.model.custom_model.CustomModel`. This will modularize ML model well and model usage become simpler. You can find an example under `/examples/models/custom_model`. 
- You can split training dbt model from regular dbt run by using tags and `--select` or `--exclude` dbt options.
- If your model depends on user modules, use `ext_modules` argument to pickle your module with model object.

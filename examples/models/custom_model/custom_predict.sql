{{
  config(
    materialized='table',
  )
}}

select
  titanic3.*,
  {{ ref('custom_model') }}!predict(titanic3.* exclude (survived)):output as prediction
from {{ ref('titanic3') }} as titanic3

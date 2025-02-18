{% materialization model, adapter = "snowflake", supported_languages = ["python"] -%}
    {% set original_query_tag = set_query_tag() %}

    {%- set identifier = model["alias"] -%}
    {%- set language = model["language"] -%}

    {% set grant_config = config.get("grants") %}

    {%- set existing_relation = adapter.get_relation(
        database=database, schema=schema, identifier=identifier
    ) -%}
    {%- set target_relation = api.Relation.create(
        identifier=identifier,
        schema=schema,
        database=database,
        type="external",
    ) -%}

    {{ run_hooks(pre_hooks) }}

    {# dropping model relation is not supported yet
    {% if target_relation.needs_to_drop(existing_relation) %}
        {{ drop_relation_if_exists(existing_relation) }}
    {% endif %}#}
    {% call statement("main", language=language) -%}
        {{ dbt_snowflake_ml.snowflake__create_model_as(target_relation, compiled_code, language) }}
    {%- endcall %}

    {{ run_hooks(post_hooks) }}

    {% set should_revoke = should_revoke(existing_relation, full_refresh_mode=True) %}
    {% do apply_grants(target_relation, grant_config, should_revoke=should_revoke) %}

    {% do persist_docs(target_relation, model) %}

    {% do unset_query_tag(original_query_tag) %}

    {{ return({"relations": [target_relation]}) }}
{% endmaterialization %}

{% macro snowflake__create_model_as(relation, compiled_code, language="sql") -%}

    {# Generate DDL/DML #}
    {%- if language == "python" -%}
        {%- if relation.is_iceberg_format %}
            {% do exceptions.raise_compiler_error(
                "Iceberg is incompatible with Python models. Please use a SQL model for the iceberg format."
            ) %}
        {%- endif %}
        {{ dbt_snowflake_ml.py_write_model(compiled_code=compiled_code, target_relation=relation) }}
    {%- else -%}
        {% do exceptions.raise_compiler_error(
            "snowflake__create_model_as macro didn't get supported language, it got %s"
            % language
        ) %}
    {%- endif -%}

{% endmacro %}

-- fmt: off
{% macro py_write_model(compiled_code, target_relation) %}
import importlib.util
if importlib.util.find_spec("snowflake.ml") is None:
    raise ImportError("snowflake.ml is not found. Add snowflake-ml-python to package dependencies.")
from snowflake.ml.registry import Registry
{{ compiled_code }}
def main(session):
    dbt = dbtObj(session.table)
    model_dict = model(dbt, session)
    reg = Registry(
        session = session,
        database_name = dbt.this.database,
        schema_name = dbt.this.schema
    )
    set_default = model_dict.pop("set_default", False)
    assert "model_name" not in model_dict, "model_name cannot be overridden"
    assert "conda_dependencies" not in model_dict, "conda_dependencies cannot be overridden"
    mv = reg.log_model(
        **model_dict,
        model_name = dbt.this.identifier,
        conda_dependencies = ["{{ config.get('packages') | join('", "') }}"],
    )
    if set_default:
        reg.get_model(dbt.this.identifier).default = mv
    return "OK"
{% endmacro %}
-- fmt: on

{% macro list_model(model_name) %}
    {% do run_query("show versions in model " ~ ref(model_name)).print_table(
        **kwargs
    ) %}
{% endmacro %}

{% macro set_model_default_version(model_name, version_name) %}
    {% do run_query(
        "alter model "
        ~ ref(model_name)
        ~ " set default_version = '"
        ~ version_name
        ~ "'"
    ).print_table(**kwargs) %}
{% endmacro %}

{% macro drop_model_version(model_name, version_name) %}
    {% do run_query(
        "alter model " ~ ref(model_name) ~ " drop version " ~ version_name
    ).print_table(**kwargs) %}
{% endmacro %}

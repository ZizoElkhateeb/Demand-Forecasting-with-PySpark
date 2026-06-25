import glob
import json
import math
import os
import re
import sys
from datetime import date as date_type
from typing import cast

import boto3
import streamlit as st
from pyspark.ml import PipelineModel
from pyspark.ml.feature import StringIndexerModel
from pyspark.ml.regression import (
    GBTRegressionModel,
    LinearRegressionModel,
    RandomForestRegressionModel,
)
from pyspark.sql import DataFrame, SparkSession, functions as F

MODEL_DIRS = {
    "LinearRegression": "linear_regression",
    "RandomForest": "random_forest",
    "GradientBoosting": "gradient_boosting",
}

MODEL_LOADERS = {
    "LinearRegression": LinearRegressionModel,
    "RandomForest": RandomForestRegressionModel,
    "GradientBoosting": GBTRegressionModel,
}

STRING_COLS = {
    "Category",
    "Region",
    "Weather Condition",
    "Seasonality",
}

INT_COLS = {
    "Promotion",
    "Epidemic",
    "Year",
    "Month",
    "DayOfWeek",
}


def _normalize_s3a_duration(value: str) -> str | None:
    match = re.match(r"^(\d+)(ms|s|m|h|d)$", value.strip())
    if not match:
        return None
    number = int(match.group(1))
    unit = match.group(2)
    multipliers = {"ms": 1, "s": 1000, "m": 60000, "h": 3600000, "d": 86400000}
    return str(number * multipliers[unit])


def _normalize_s3a_durations(spark: SparkSession) -> None:
    jsc = spark.sparkContext._jsc
    if jsc is None:
        return
    hconf_fn = getattr(jsc, "hadoopConfiguration", None)
    if hconf_fn is None:
        return
    hconf = hconf_fn()
    if hconf is None:
        return

    hconf.set("fs.s3a.vectored.reads.enabled", "false")

    it = hconf.iterator()
    while it.hasNext():
        entry = it.next()
        key = entry.getKey()
        value = entry.getValue()
        if key.startswith("fs.s3a.") and isinstance(value, str):
            normalized = _normalize_s3a_duration(value)
            if normalized is not None:
                hconf.set(key, normalized)


@st.cache_resource(show_spinner=False)
def get_spark_session(
    enable_s3a: bool,
    endpoint: str,
    access_key: str,
    secret_key: str,
    use_ssl: bool,
) -> SparkSession:
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    builder = (
        SparkSession.builder.appName("DemandForecastingApp")
        .master("local[*]")
        .config("spark.ui.enabled", "false")
        .config("spark.python.worker.faulthandler.enabled", "true")
        .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
    )

    if enable_s3a:
        os.makedirs("C:/temp/s3a", exist_ok=True)
        builder = (
            builder.config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.6")
            .config("spark.hadoop.fs.s3a.endpoint", endpoint)
            .config("spark.hadoop.fs.s3a.access.key", access_key)
            .config("spark.hadoop.fs.s3a.secret.key", secret_key)
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .config(
                "spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem"
            )
            .config(
                "spark.hadoop.fs.s3a.connection.ssl.enabled",
                "true" if use_ssl else "false",
            )
            .config(
                "spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
            )
            .config("spark.hadoop.fs.s3a.vectored.reads.enabled", "false")
            .config("spark.hadoop.fs.s3a.fast.upload", "true")
            .config("spark.hadoop.fs.s3a.fast.upload.buffer", "bytebuffer")
            .config("spark.hadoop.fs.s3a.buffer.dir", "C:/temp/s3a")
            .config("spark.hadoop.fs.s3a.connection.timeout", "60000")
            .config("spark.hadoop.fs.s3a.connection.establish.timeout", "60000")
            .config("spark.hadoop.fs.s3a.socket.timeout", "60000")
            .config("spark.hadoop.fs.s3a.threads.keepalivetime", "60000")
            .config("spark.hadoop.fs.s3a.retry.interval", "500")
            .config("spark.hadoop.fs.s3a.retry.throttle.interval", "100")
            .config("spark.hadoop.fs.s3a.signing.session.duration", "86400000")
        )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    if enable_s3a:
        _normalize_s3a_durations(spark)

    return spark


@st.cache_resource(show_spinner=False)
def load_models(preprocess_path: str, model_path: str, model_type: str):
    preprocess_model = PipelineModel.load(preprocess_path)
    model_loader = MODEL_LOADERS[model_type]
    model = model_loader.load(model_path)
    return preprocess_model, model


def spark_day_of_week(input_date: date_type) -> int:
    # Spark dayofweek: Sunday=1 ... Saturday=7
    return ((input_date.weekday() + 1) % 7) + 1


def build_feature_row(
    input_date: date_type,
    category: str,
    region: str,
    weather_condition: str,
    seasonality: str,
    inventory_level: float,
    units_sold: float,
    units_ordered: float,
    price: float,
    discount: float,
    promotion: int,
    competitor_pricing: float,
    epidemic: int,
    lag_1_demand: float,
    lag_7_demand: float,
    rolling_7_avg_demand: float,
    expected_demand_for_gap: float,
):
    year = input_date.year
    month = input_date.month
    day_of_week = spark_day_of_week(input_date)

    price_gap = price - competitor_pricing
    inventory_gap = inventory_level - expected_demand_for_gap
    sell_through_ratio = units_sold / inventory_level if inventory_level > 0 else 0.0
    order_fill_ratio = units_sold / units_ordered if units_ordered > 0 else 0.0

    return {
        "Category": category,
        "Region": region,
        "Weather Condition": weather_condition,
        "Seasonality": seasonality,
        "Inventory Level": float(inventory_level),
        "Units Sold": float(units_sold),
        "Units Ordered": float(units_ordered),
        "Price": float(price),
        "Discount": float(discount),
        "Promotion": int(promotion),
        "Competitor Pricing": float(competitor_pricing),
        "Epidemic": int(epidemic),
        "Year": int(year),
        "Month": int(month),
        "DayOfWeek": int(day_of_week),
        "lag_1_demand": float(lag_1_demand),
        "lag_7_demand": float(lag_7_demand),
        "rolling_7_avg_demand": float(rolling_7_avg_demand),
        "inventory_gap": float(inventory_gap),
        "price_gap": float(price_gap),
        "sell_through_ratio": float(sell_through_ratio),
        "order_fill_ratio": float(order_fill_ratio),
    }


def _is_s3a_path(path: str) -> bool:
    return path.strip().lower().startswith("s3a://")


def _parse_s3a_path(path: str) -> tuple[str, str]:
    cleaned = path.strip()
    if not cleaned.lower().startswith("s3a://"):
        raise ValueError("Expected s3a:// path")
    without_scheme = cleaned[6:]
    parts = without_scheme.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


@st.cache_resource(show_spinner=False)
def _download_minio_prefix(
    endpoint: str,
    access_key: str,
    secret_key: str,
    bucket: str,
    prefix: str,
    local_dir: str,
) -> str:
    if os.path.isdir(local_dir):
        try:
            if any(os.scandir(local_dir)):
                return local_dir
        except OSError:
            pass

    os.makedirs(local_dir, exist_ok=True)
    s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            if not key or key.endswith("/"):
                continue
            rel_path = os.path.relpath(key, prefix) if prefix else key
            local_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3_client.download_file(bucket, key, local_path)

    return local_dir


def build_feature_df(spark: SparkSession, feature_row: dict) -> DataFrame:
    cols = []
    for name, value in feature_row.items():
        if name in STRING_COLS:
            cols.append(F.lit(value).cast("string").alias(name))
        elif name in INT_COLS:
            cols.append(F.lit(value).cast("int").alias(name))
        else:
            cols.append(F.lit(value).cast("double").alias(name))
    return spark.range(1).select(*cols)


def _default_index(options: list[str], default_value: str) -> int:
    if default_value in options:
        return options.index(default_value)
    return 0


def get_categorical_options(preprocess_model: PipelineModel) -> dict[str, list[str]]:
    options: dict[str, list[str]] = {}
    for stage in preprocess_model.stages:
        if not isinstance(stage, StringIndexerModel):
            continue
        input_col = stage.getInputCol()
        if input_col not in STRING_COLS:
            continue
        labels = [label for label in stage.labels if label]
        if labels:
            options[input_col] = labels
    return options


def main() -> None:
    st.set_page_config(
        page_title="Demand Forecasting", page_icon="📈", layout="centered"
    )
    st.title("Demand Forecasting - Best Model")
    st.write(
        "Load the preprocessing pipeline and best model, collect inputs, and predict demand."
    )

    project_root = os.path.dirname(os.path.abspath(__file__))
    default_local_preprocess = os.path.join(
        project_root, "local_models", "preprocessing_pipeline"
    )
    default_local_model = os.path.join(
        project_root, "local_models", "linear_regression"
    )

    with st.sidebar:
        st.header("Model Settings")
        storage_mode = st.radio(
            "Model source", ("MinIO (s3a://)", "Local filesystem"), index=0
        )
        model_type = cast(
            str, st.selectbox("Model type", list(MODEL_DIRS.keys()), index=0)
        )

        if storage_mode == "MinIO (s3a://)":
            endpoint_default = "http://127.0.0.1:9000"
            endpoint = st.text_input("MinIO endpoint", value=endpoint_default)
            use_ssl = endpoint.lower().startswith("https://")
            use_ssl = st.checkbox("Use SSL", value=use_ssl)
            bucket = st.text_input("Bucket", value="demand-lake")
            access_key = st.text_input(
                "Access key",
                value=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            )
            secret_key = st.text_input(
                "Secret key",
                value=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
                type="password",
            )
            default_preprocess_path = f"s3a://{bucket}/models/preprocessing-pipeline/"
            default_model_path = f"s3a://{bucket}/models/{MODEL_DIRS[model_type]}/"
        else:
            endpoint = ""
            access_key = ""
            secret_key = ""
            use_ssl = False
            default_preprocess_path = default_local_preprocess
            default_model_path = default_local_model

        if st.session_state.get("storage_mode") != storage_mode:
            st.session_state.storage_mode = storage_mode
            st.session_state.preprocess_path = default_preprocess_path
            st.session_state.model_path = default_model_path
            st.cache_resource.clear()
            st.rerun()

        if st.session_state.get("model_type") != model_type:
            st.session_state.model_type = model_type
            st.session_state.model_path = default_model_path
            st.cache_resource.clear()
            st.rerun()

        preprocess_path = st.text_input(
            "Preprocessing pipeline path",
            value=default_preprocess_path,
            key="preprocess_path",
        )
        model_path = st.text_input(
            "Model path", value=default_model_path, key="model_path"
        )

        if st.button("Reset Spark session"):
            st.cache_resource.clear()
            st.rerun()

    enable_s3a = storage_mode == "MinIO (s3a://)"

    if storage_mode == "Local filesystem":
        if _is_s3a_path(preprocess_path) or _is_s3a_path(model_path):
            st.error("Local mode selected, but an s3a:// path is provided.")
            st.info("Switch to MinIO mode or use local paths from local_models/.")
            st.stop()
    else:
        if not _is_s3a_path(preprocess_path) or not _is_s3a_path(model_path):
            st.error("MinIO mode requires s3a:// paths for both pipeline and model.")
            st.stop()

        bucket_p, prefix_p = _parse_s3a_path(preprocess_path)
        bucket_m, prefix_m = _parse_s3a_path(model_path)
        cache_root = os.path.join(project_root, ".cache", "minio_models")
        preprocess_local = os.path.join(
            cache_root, bucket_p, prefix_p.replace("/", os.sep)
        )
        model_local = os.path.join(cache_root, bucket_m, prefix_m.replace("/", os.sep))

        with st.spinner("Downloading model artifacts from MinIO..."):
            preprocess_path = _download_minio_prefix(
                endpoint,
                access_key,
                secret_key,
                bucket_p,
                prefix_p,
                preprocess_local,
            )
            model_path = _download_minio_prefix(
                endpoint,
                access_key,
                secret_key,
                bucket_m,
                prefix_m,
                model_local,
            )

        enable_s3a = False

    if not _is_s3a_path(preprocess_path) and not os.path.isdir(preprocess_path):
        st.error("Preprocessing pipeline path not found.")
        st.stop()

    if not _is_s3a_path(model_path) and not os.path.isdir(model_path):
        st.error("Model path not found.")
        st.stop()

    spark = get_spark_session(enable_s3a, endpoint, access_key, secret_key, use_ssl)

    try:
        preprocess_model, model = load_models(preprocess_path, model_path, model_type)
    except Exception as exc:  # pragma: no cover - streamlit displays the error
        st.exception(exc)
        st.stop()

    categorical_options = get_categorical_options(preprocess_model)

    st.subheader("Input Features")

    col1, col2 = st.columns(2)
    with col1:
        input_date = st.date_input("Date", value=date_type.today())
        input_date_value = cast(date_type, input_date)
        category_options = categorical_options.get("Category", ["Groceries"])
        region_options = categorical_options.get("Region", ["North"])
        weather_options = categorical_options.get("Weather Condition", ["Sunny"])
        seasonality_options = categorical_options.get("Seasonality", ["Summer"])

        category = st.selectbox(
            "Category",
            category_options,
            index=_default_index(category_options, "Groceries"),
        )
        region = st.selectbox(
            "Region",
            region_options,
            index=_default_index(region_options, "North"),
        )
        weather_condition = st.selectbox(
            "Weather Condition",
            weather_options,
            index=_default_index(weather_options, "Sunny"),
        )
        seasonality = st.selectbox(
            "Seasonality",
            seasonality_options,
            index=_default_index(seasonality_options, "Summer"),
        )

    with col2:
        inventory_level = st.number_input(
            "Inventory Level", min_value=0.0, value=1000.0, step=1.0
        )
        units_sold = st.number_input("Units Sold", min_value=0.0, value=220.0, step=1.0)
        units_ordered = st.number_input(
            "Units Ordered", min_value=0.0, value=250.0, step=1.0
        )
        price = st.number_input("Price", min_value=0.0, value=20.0, step=0.1)
        competitor_pricing = st.number_input(
            "Competitor Pricing", min_value=0.0, value=19.5, step=0.1
        )

    col3, col4 = st.columns(2)
    with col3:
        discount = st.number_input("Discount", min_value=0.0, value=0.1, step=0.01)
        promotion = st.checkbox("Promotion", value=False)
        epidemic = st.checkbox("Epidemic", value=False)
        expected_demand_for_gap = st.number_input(
            "Expected Demand (for inventory gap)", min_value=0.0, value=200.0, step=1.0
        )

    with col4:
        lag_1_demand = st.number_input(
            "lag_1_demand", min_value=0.0, value=210.0, step=1.0
        )
        lag_7_demand = st.number_input(
            "lag_7_demand", min_value=0.0, value=205.0, step=1.0
        )
        rolling_7_avg_demand = st.number_input(
            "rolling_7_avg_demand", min_value=0.0, value=208.0, step=1.0
        )

    if st.button("Predict Demand", type="primary"):
        feature_row = build_feature_row(
            input_date=input_date_value,
            category=category,
            region=region,
            weather_condition=weather_condition,
            seasonality=seasonality,
            inventory_level=inventory_level,
            units_sold=units_sold,
            units_ordered=units_ordered,
            price=price,
            discount=discount,
            promotion=int(promotion),
            competitor_pricing=competitor_pricing,
            epidemic=int(epidemic),
            lag_1_demand=lag_1_demand,
            lag_7_demand=lag_7_demand,
            rolling_7_avg_demand=rolling_7_avg_demand,
            expected_demand_for_gap=expected_demand_for_gap,
        )

        input_df = build_feature_df(spark, feature_row)
        prepared_df = preprocess_model.transform(input_df)
        prediction_df = model.transform(prepared_df)
        try:
            prediction_rows = prediction_df.select("prediction").head(1)
            if not prediction_rows:
                st.error(
                    "No prediction was returned. Check the input values and model files."
                )
                st.stop()
            prediction_value = prediction_rows[0]["prediction"]
        except Exception:
            temp_dir = os.path.join(project_root, ".cache", "prediction_output")
            (
                prediction_df.select("prediction")
                .coalesce(1)
                .write.mode("overwrite")
                .json(temp_dir)
            )
            json_files = glob.glob(os.path.join(temp_dir, "part-*.json"))
            if not json_files:
                st.error("Prediction output file not found.")
                st.stop()
            with open(json_files[0], "r", encoding="utf-8") as handle:
                first_line = handle.readline().strip()
            if not first_line:
                st.error("Prediction output was empty.")
                st.stop()
            payload = json.loads(first_line)
            prediction_value = payload.get("prediction")
            if prediction_value is None:
                st.error("Prediction value was missing in output.")
                st.stop()
        st.subheader("Prediction")
        st.metric("Predicted Demand", f"{math.ceil(prediction_value)}")

        with st.expander("View model inputs"):
            st.write(feature_row)


if __name__ == "__main__":
    main()

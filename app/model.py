from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset.csv"

FEATURES = [
    "Year", "Month", "Day", "DayOfWeek", "Hour",
    "DayOfYear", "Temperature", "Rain", "Sun",
]
TARGET = "Consumption"
REQUIRED_COLUMNS = ["Date", "Time", "Consumption", "Temperature", "Rain", "Sun"]


def load_dataset(path=DATA_PATH):
    """Read and prepare the WaterWise data; inspect outliers but never remove them automatically."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required column(s): " + ", ".join(missing)
            + ". Expected columns: Date, Time, Consumption, Temperature, Rain, Sun."
        )

    # Parse dates while allowing different date formats in the source file.
    date_values = df["Date"].astype(str).str.strip()
    try:
        parsed_dates = pd.to_datetime(date_values, errors="coerce", dayfirst=True, format="mixed")
    except TypeError:  # Compatibility with older pandas versions.
        parsed_dates = pd.to_datetime(date_values, errors="coerce", dayfirst=True)

    # Time values in this dataset are often ranges such as "00:00-01:00".
    # Use the beginning of the interval as the observation hour.
    time_values = df["Time"].astype(str).str.strip()
    start_time = time_values.str.extract(r"(\d{1,2}:\d{2})", expand=False)
    parsed_clock = pd.to_datetime(start_time, format="%H:%M", errors="coerce")
    df["DateTime"] = (
        parsed_dates.dt.normalize()
        + pd.to_timedelta(parsed_clock.dt.hour, unit="h")
        + pd.to_timedelta(parsed_clock.dt.minute, unit="m")
    )

    for column in ["Consumption", "Temperature", "Rain", "Sun"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Fill missing Rain values using the median; keep rows and do not auto-remove outliers.
    if df["Rain"].notna().any():
        df["Rain"] = df["Rain"].fillna(df["Rain"].median())

    df["Year"] = df["DateTime"].dt.year
    df["Month"] = df["DateTime"].dt.month
    df["Day"] = df["DateTime"].dt.day
    df["DayOfWeek"] = df["DateTime"].dt.dayofweek
    df["Hour"] = df["DateTime"].dt.hour
    df["DayOfYear"] = df["DateTime"].dt.dayofyear
    return df


def prepare_features(df):
    """Return rows that have all required model fields, sorted chronologically."""
    usable = df.dropna(subset=FEATURES + [TARGET, "DateTime"]).copy()
    return usable.sort_values("DateTime").reset_index(drop=True)


def train_model(path=DATA_PATH):
    """Train Random Forest with chronological 80/20 split; return model, data, metrics, test results."""
    df = load_dataset(path)
    usable = prepare_features(df)
    if len(usable) < 10:
        raise ValueError(f"Not enough complete rows to train the model. Usable rows: {len(usable)}")

    split_index = int(len(usable) * 0.8)
    if split_index <= 0 or split_index >= len(usable):
        raise ValueError("Unable to create an 80/20 chronological train/test split.")

    train_df = usable.iloc[:split_index]
    test_df = usable.iloc[split_index:]

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train_df[FEATURES], train_df[TARGET])

    predictions = model.predict(test_df[FEATURES])
    actual = test_df[TARGET].to_numpy()
    metrics = {
        "MAE": float(mean_absolute_error(actual, predictions)),
        "RMSE": float(mean_squared_error(actual, predictions) ** 0.5),
        "R2": float(r2_score(actual, predictions)) if len(test_df) > 1 else None,
        "training_rows": int(len(train_df)),
        "testing_rows": int(len(test_df)),
    }
    test_results = pd.DataFrame({
        "DateTime": test_df["DateTime"].to_numpy(),
        "Actual": actual,
        "Predicted": predictions,
    })
    return model, usable, metrics, test_results


def forecast_next_hours(model, df=None, hours=24):
    """Forecast future hours using the latest available temperature, rain and sun values."""
    if hours < 1:
        raise ValueError("hours must be at least 1")
    if df is None:
        df = load_dataset()
    dated = df.dropna(subset=["DateTime"]).sort_values("DateTime")
    if dated.empty:
        raise ValueError("No valid date/time rows are available for forecasting.")
    last = dated.iloc[-1]
    start = last["DateTime"]

    def last_number(column, default=0.0):
        value = pd.to_numeric(pd.Series([last.get(column)]), errors="coerce").iloc[0]
        return float(value) if pd.notna(value) else default

    future_rows = []
    future_times = []
    for step in range(1, hours + 1):
        dt = start + pd.Timedelta(hours=step)
        future_times.append(dt)
        future_rows.append({
            "Year": dt.year, "Month": dt.month, "Day": dt.day,
            "DayOfWeek": dt.dayofweek, "Hour": dt.hour, "DayOfYear": dt.dayofyear,
            "Temperature": last_number("Temperature"),
            "Rain": last_number("Rain"), "Sun": last_number("Sun"),
        })

    future = pd.DataFrame(future_rows, columns=FEATURES)
    return pd.DataFrame({
        "DateTime": future_times,
        "PredictedConsumption": model.predict(future[FEATURES]),
    })


def get_dataset_summary(path=DATA_PATH):
    df = load_dataset(path)
    consumption = pd.to_numeric(df[TARGET], errors="coerce").dropna()
    return {
        "total_records": int(len(df)),
        "average_consumption": float(consumption.mean()) if not consumption.empty else 0.0,
        "maximum_consumption": float(consumption.max()) if not consumption.empty else 0.0,
        "minimum_consumption": float(consumption.min()) if not consumption.empty else 0.0,
    }


def inspect_outliers(df, column="Consumption"):
    """Report IQR outlier boundaries/count without modifying or deleting any rows."""
    if column not in df.columns:
        raise ValueError(f"Column not found: {column}")
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return {"column": column, "q1": None, "q3": None, "lower_bound": None,
                "upper_bound": None, "outlier_count": 0, "outlier_percent": 0.0}
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((values < lower) | (values > upper)).sum())
    return {
        "column": column, "q1": q1, "q3": q3, "lower_bound": lower,
        "upper_bound": upper, "outlier_count": count,
        "outlier_percent": float(count / len(values) * 100),
    }
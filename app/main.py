
from pathlib import Path
from datetime import datetime
import os
import sqlite3
import secrets
import hashlib
import hmac
import math

import pandas as pd

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.model import train_model, FEATURES, forecast_next_hours


# =========================================================
# 1. PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATA_PATH = BASE_DIR / "dataset.csv"
TEMPLATES_DIR = PROJECT_DIR / "templates"
STATIC_DIR = PROJECT_DIR / "static"
USER_DB = BASE_DIR / "users.db"


# =========================================================
# 2. FASTAPI APPLICATION
# =========================================================

app = FastAPI(title="WaterWise - Water Consumption Forecasting")

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# =========================================================
# 3. SESSION CONFIGURATION
# =========================================================


# =========================================================
# 3. SESSION CONFIGURATION
# =========================================================

is_production = (
    os.getenv("RENDER") is not None
    or os.getenv("WATERWISE_HTTPS_ONLY", "").lower() == "true"
)

secret_key = os.getenv("WATERWISE_SECRET_KEY")

if is_production and not secret_key:
    raise RuntimeError(
        "Set WATERWISE_SECRET_KEY in your deployment environment."
    )

if not secret_key:
    # Local development only
    secret_key = secrets.token_urlsafe(32)



# =========================================================
# 4. DATABASE AND PASSWORD HELPERS
# =========================================================

def get_connection():
    connection = sqlite3.connect(str(USER_DB))
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    USER_DB.parent.mkdir(parents=True, exist_ok=True)

    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL
            )
            """
        )


def hash_password(password: str, salt: str) -> str:
    result = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        310000
    )
    return result.hex()


def verify_password(password: str, salt: str, stored_hash: str) -> bool:
    candidate = hash_password(password, salt)
    return hmac.compare_digest(candidate, stored_hash)


initialize_database()


# =========================================================
# 5. LOGIN PROTECTION
# =========================================================


# =========================================================
# 5. LOGIN PROTECTION AND SESSION MIDDLEWARE
# =========================================================

PUBLIC_PATHS = {
    "/",
    "/login",
    "/signup",
    "/logout",
    "/logout-page",
    "/favicon.ico"
}


class LoginProtectionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        is_public = (
            path in PUBLIC_PATHS
            or path.startswith("/static/")
        )

        if not is_public and not request.session.get("user_id"):
            if path.startswith("/api/"):
                return JSONResponse(
                    {"detail": "Please log in to continue."},
                    status_code=401
                )

            return RedirectResponse(
                url="/login",
                status_code=303
            )

        return await call_next(request)


# Add login protection first
app.add_middleware(LoginProtectionMiddleware)

# Add session middleware last so sessions are available
# when login protection checks the request.
app.add_middleware(
    SessionMiddleware,
    secret_key=secret_key,
    same_site="lax",
    https_only=is_production,
    max_age=60 * 60 * 8
)


# =========================================================
# 6. DATASET HELPERS
# =========================================================

def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required = {
        "Date",
        "Time",
        "Consumption",
        "Temperature",
        "Rain",
        "Sun"
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )

    return df


def statistics(df: pd.DataFrame) -> dict:
    consumption = pd.to_numeric(
        df["Consumption"],
        errors="coerce"
    ).dropna()

    if consumption.empty:
        average = maximum = minimum = None
    else:
        average = float(consumption.mean())
        maximum = float(consumption.max())
        minimum = float(consumption.min())

    return {
        "total_records": int(len(df)),
        "average_consumption": average,
        "maximum_consumption": maximum,
        "minimum_consumption": minimum
    }


# =========================================================
# 7. MODEL CACHE
# =========================================================

_model_cache = None
_model_error = None


def get_trained_model():
    """
    Train the model once per application process and reuse it.

    This expects train_model() to return:
    model, usable_dataframe, metrics, test_results
    as in the existing model implementation.
    """
    global _model_cache, _model_error

    if _model_cache is not None:
        return _model_cache

    if _model_error is not None:
        raise RuntimeError(_model_error)

    try:
        _model_cache = train_model()
        return _model_cache
    except Exception as exc:
        _model_error = str(exc)
        raise


def get_model_parts():
    result = get_trained_model()

    if not isinstance(result, (tuple, list)) or len(result) < 3:
        raise RuntimeError(
            "train_model() returned an unexpected result. "
            "Check the return statement in app/model.py."
        )

    model = result[0]
    usable_df = result[1]
    metrics = result[2]

    test_results = result[3] if len(result) > 3 else None

    return model, usable_df, metrics, test_results


def clean_number(value):
    try:
        number = float(value)

        if not math.isfinite(number):
            return None

        return number
    except (TypeError, ValueError):
        return None


# =========================================================
# 8. HOME PAGE
# =========================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    if not request.session.get("user_id"):
        return RedirectResponse("/login", status_code=303)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "user": request.session.get("user_name"),
            "name": request.session.get("user_name"),
            "email": request.session.get("user_email")
        }
    )


# =========================================================
# 9. LOGIN
# =========================================================

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if request.session.get("user_id"):
        return RedirectResponse("/", status_code=303)

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={
            "error": None,
            "registered": request.query_params.get("registered"),
            "logged_out": request.query_params.get("logged_out")
        }
    )


@app.post("/login", response_class=HTMLResponse)
async def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...)
):
    email = email.strip().lower()

    with get_connection() as connection:
        user = connection.execute(
            """
            SELECT id, name, email, password_hash, salt
            FROM users
            WHERE email = ?
            """,
            (email,)
        ).fetchone()

    if user is None or not verify_password(
        password,
        user["salt"],
        user["password_hash"]
    ):
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={
                "error": "Invalid email or password.",
                "registered": None,
                "logged_out": None
            },
            status_code=401
        )

    request.session.clear()
    request.session["user_id"] = user["id"]
    request.session["user_name"] = user["name"]
    request.session["user_email"] = user["email"]

    return RedirectResponse("/", status_code=303)


# =========================================================
# 10. SIGNUP
# =========================================================

@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    if request.session.get("user_id"):
        return RedirectResponse("/", status_code=303)

    return templates.TemplateResponse(
        request=request,
        name="signuppage.html",
        context={
            "error": None,
            "success": None
        }
    )


@app.post("/signup", response_class=HTMLResponse)
async def signup(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...)
):
    name = name.strip()
    email = email.strip().lower()

    if not name or not email or not password:
        return templates.TemplateResponse(
            request=request,
            name="signuppage.html",
            context={
                "error": "Please complete all required fields.",
                "success": None
            },
            status_code=400
        )

    if len(password) < 8:
        return templates.TemplateResponse(
            request=request,
            name="signuppage.html",
            context={
                "error": "Password must contain at least 8 characters.",
                "success": None
            },
            status_code=400
        )

    salt = secrets.token_hex(16)
    password_hash = hash_password(password, salt)

    try:
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO users (name, email, password_hash, salt)
                VALUES (?, ?, ?, ?)
                """,
                (name, email, password_hash, salt)
            )
    except sqlite3.IntegrityError:
        return templates.TemplateResponse(
            request=request,
            name="signuppage.html",
            context={
                "error": "An account with this email already exists.",
                "success": None
            },
            status_code=400
        )

    with get_connection() as connection:
        user = connection.execute(
            "SELECT id, name, email FROM users WHERE email = ?",
            (email,)
        ).fetchone()

    request.session.clear()
    request.session["user_id"] = user["id"]
    request.session["user_name"] = user["name"]
    request.session["user_email"] = user["email"]

    return RedirectResponse("/", status_code=303)


# =========================================================
# 11. LOGOUT
# =========================================================

@app.get("/logout")
@app.post("/logout")
async def logout(request: Request):
    request.session.clear()

    return RedirectResponse(
        url="/login?logged_out=1",
        status_code=303
    )


@app.get("/logout-page", response_class=HTMLResponse)
async def logout_page(request: Request):
    # Compatibility route for older links.
    if request.session.get("user_id"):
        return RedirectResponse("/", status_code=303)

    return RedirectResponse(
        "/login?logged_out=1",
        status_code=303
    )


# =========================================================
# 12. DASHBOARD
# =========================================================

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    try:
        df = load_dataset()
        stats = statistics(df)
        error = None
    except Exception as exc:
        stats = {
            "total_records": 0,
            "average_consumption": None,
            "maximum_consumption": None,
            "minimum_consumption": None
        }
        error = str(exc)

    context = {
        **stats,
        "user": request.session.get("user_name"),
        "name": request.session.get("user_name"),
        "email": request.session.get("user_email"),
        "error": error
    }

    try:
        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context=context
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return HTMLResponse(
            content=(
                f"Dashboard rendering error: "
                f"{type(exc).__name__}: {exc}"
            ),
            status_code=500
        )

# =========================================================
# 13. PREDICTION FEATURE HELPERS
# =========================================================

def build_prediction_features(
    date_value: str,
    time_value: str,
    temperature: float,
    rain: float,
    sun: float
) -> pd.DataFrame:
    dt = pd.to_datetime(
        f"{date_value} {time_value}",
        errors="coerce"
    )

    if pd.isna(dt):
        raise ValueError("Please enter a valid date and time.")

    row = {
        "Year": dt.year,
        "Month": dt.month,
        "Day": dt.day,
        "DayOfWeek": dt.dayofweek,
        "Hour": dt.hour,
        "DayOfYear": dt.dayofyear,
        "Temperature": temperature,
        "Rain": rain,
        "Sun": sun
    }

    return pd.DataFrame(
        [[row[column] for column in FEATURES]],
        columns=FEATURES
    )


def predict_consumption(
    date_value: str,
    time_value: str,
    temperature: float,
    rain: float,
    sun: float
) -> float:
    model, _, _, _ = get_model_parts()

    features = build_prediction_features(
        date_value,
        time_value,
        temperature,
        rain,
        sun
    )

    prediction = float(model.predict(features)[0])

    if not math.isfinite(prediction):
        raise ValueError("The model returned an invalid prediction.")

    return prediction


# =========================================================
# 14. PREDICTION PAGE
# =========================================================

@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="predict.html",
        context={
            "user": request.session.get("user_name"),
            "error": None,
            "prediction": None
        }
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict_page_submit(
    request: Request,
    date: str = Form(...),
    time: str = Form(...),
    temperature: float = Form(...),
    rain: float = Form(...),
    sun: float = Form(...)
):
    try:
        prediction = predict_consumption(
            date,
            time,
            temperature,
            rain,
            sun
        )

        return templates.TemplateResponse(
            request=request,
            name="predict.html",
            context={
                "user": request.session.get("user_name"),
                "error": None,
                "prediction": round(prediction, 3)
            }
        )

    except Exception as exc:
        return templates.TemplateResponse(
            request=request,
            name="predict.html",
            context={
                "user": request.session.get("user_name"),
                "error": str(exc),
                "prediction": None
            },
            status_code=400
        )


# =========================================================
# 15. PREDICTION API
# =========================================================

@app.post("/api/predict")
async def api_predict(
    date: str = Form(...),
    time: str = Form(...),
    temperature: float = Form(...),
    rain: float = Form(...),
    sun: float = Form(...)
):
    try:
        prediction = predict_consumption(
            date,
            time,
            temperature,
            rain,
            sun
        )

        return {
            "prediction": round(prediction, 3),
            "predicted_consumption": round(prediction, 3)
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )


# =========================================================
# 16. FORECAST API
# =========================================================

@app.get("/api/forecast")
async def api_forecast(hours: int = 24):
    if hours < 1 or hours > 168:
        raise HTTPException(
            status_code=400,
            detail="Hours must be between 1 and 168."
        )

    try:
        model, usable_df, _, _ = get_model_parts()

        forecast_df = forecast_next_hours(
            model,
            usable_df,
            hours
        )

        if forecast_df is None or forecast_df.empty:
            raise ValueError("The model returned no forecast records.")

        forecast_df = forecast_df.copy()

        # The existing forecasting function is expected to return
        # DateTime and PredictedConsumption columns.
        required = {"DateTime", "PredictedConsumption"}

        if not required.issubset(forecast_df.columns):
            raise ValueError(
                "Unexpected forecast output columns. "
                "Check forecast_next_hours() in app/model.py."
            )

        forecast_rows = []

        for _, row in forecast_df.iterrows():
            timestamp = pd.to_datetime(
                row["DateTime"],
                errors="coerce"
            )

            value = clean_number(row["PredictedConsumption"])

            if pd.isna(timestamp) or value is None:
                continue

            forecast_rows.append({
                "DateTime": timestamp.isoformat(),
                "PredictedConsumption": round(value, 3)
            })

        if not forecast_rows:
            raise ValueError("No valid forecast values were generated.")

        values = [
            row["PredictedConsumption"]
            for row in forecast_rows
        ]

        return {
            "hours": len(forecast_rows),
            "forecast": forecast_rows,
            "summary": {
                "average": round(sum(values) / len(values), 3),
                "minimum": round(min(values), 3),
                "maximum": round(max(values), 3)
            },
            "source": "trained_random_forest_model"
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Forecast generation failed: {exc}"
        )


# =========================================================
# 17. PERFORMANCE PAGE
# =========================================================

@app.get("/performance", response_class=HTMLResponse)
async def performance_page(request: Request):
    context = {
        "user": request.session.get("user_name"),
        "mae": None,
        "rmse": None,
        "r2": None,
        "total_records": 0,
        "train_records": 0,
        "test_records": 0,
        "feature_importance": [],
        "error": None
    }

    try:
        model, usable_df, metrics, test_results = get_model_parts()

        context["total_records"] = int(len(usable_df))

        context["mae"] = clean_number(metrics.get("MAE"))
        context["rmse"] = clean_number(metrics.get("RMSE"))
        context["r2"] = clean_number(metrics.get("R2"))

        split_index = int(len(usable_df) * 0.8)
        context["train_records"] = split_index
        context["test_records"] = len(usable_df) - split_index

        if hasattr(model, "feature_importances_"):
            context["feature_importance"] = [
                {
                    "feature": feature,
                    "importance": float(importance)
                }
                for feature, importance in zip(
                    FEATURES,
                    model.feature_importances_
                )
            ]

    except Exception as exc:
        context["error"] = str(exc)

    return templates.TemplateResponse(
        request=request,
        name="performance.html",
        context=context
    )


# =========================================================
# 18. DATASET PAGE
# =========================================================

# This route remains available for compatibility.
# You can remove its navigation links without deleting the dataset
# needed by the forecasting model.

@app.get("/dataset", response_class=HTMLResponse)
async def dataset_page(request: Request):
    try:
        df = load_dataset()

        preview = df.head(100).where(
            pd.notna(df.head(100)),
            None
        ).to_dict(orient="records")

        context = {
            "user": request.session.get("user_name"),
            "columns": list(df.columns),
            "rows": preview,
            "total_records": len(df),
            "error": None
        }

    except Exception as exc:
        context = {
            "user": request.session.get("user_name"),
            "columns": [],
            "rows": [],
            "total_records": 0,
            "error": str(exc)
        }

    return templates.TemplateResponse(
        request=request,
        name="dataset.html",
        context=context
    )


# =========================================================
# 19. ABOUT PAGE
# =========================================================

@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="about.html",
        context={
            "user": request.session.get("user_name"),
            "name": request.session.get("user_name")
        }
    )


# =========================================================
# 20. SUMMARY API
# =========================================================

@app.get("/api/summary")
async def api_summary():
    try:
        df = load_dataset()
        return statistics(df)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to load dataset summary: {exc}"
        )


# =========================================================
# 21. HISTORICAL TREND API
# =========================================================

@app.get("/api/trend")
async def api_trend():
    try:
        df = load_dataset()

        df["Consumption"] = pd.to_numeric(
            df["Consumption"],
            errors="coerce"
        )

        parsed_dates = pd.to_datetime(
            df["Date"],
            errors="coerce",
            dayfirst=True,
            format="mixed"
        )

        df["ParsedDate"] = parsed_dates

        df = df.dropna(
            subset=["ParsedDate", "Consumption"]
        )

        if df.empty:
            return {
                "dates": [],
                "consumption": []
            }

        daily = (
            df.groupby(df["ParsedDate"].dt.strftime("%Y-%m-%d"))
            ["Consumption"]
            .sum()
            .sort_index()
            .tail(90)
        )

        return {
            "dates": daily.index.tolist(),
            "consumption": [
                round(float(value), 3)
                for value in daily.tolist()
            ]
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to load historical trend: {exc}"
        )


# =========================================================
# 22. DATASET API
# =========================================================

@app.get("/api/dataset")
async def api_dataset():
    try:
        df = load_dataset()

        preview = df.head(500).copy()
        preview = preview.astype(object).where(
            pd.notna(preview),
            None
        )

        return {
            "columns": list(df.columns),
            "rows": preview.to_dict(orient="records"),
            "total_records": int(len(df))
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to load dataset: {exc}"
        )


# =========================================================
# 23. PERFORMANCE API
# =========================================================

@app.get("/api/performance")
async def api_performance():
    try:
        model, usable_df, metrics, test_results = get_model_parts()

        split_index = int(len(usable_df) * 0.8)

        feature_importance = []

        if hasattr(model, "feature_importances_"):
            feature_importance = [
                {
                    "feature": feature,
                    "importance": round(float(importance), 6)
                }
                for feature, importance in zip(
                    FEATURES,
                    model.feature_importances_
                )
            ]

        return {
            "MAE": clean_number(metrics.get("MAE")),
            "RMSE": clean_number(metrics.get("RMSE")),
            "R2": clean_number(metrics.get("R2")),
            "total_records": int(len(usable_df)),
            "train_records": split_index,
            "test_records": len(usable_df) - split_index,
            "feature_importance": feature_importance
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to load model performance: {exc}"
        )

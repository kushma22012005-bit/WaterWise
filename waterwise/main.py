
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
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from model import train_model, FEATURES


# --------------------------------------------------
# PATHS AND APPLICATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_PATH = BASE_DIR / "dataset.csv"
TEMPLATES_DIR = PROJECT_DIR / "templates"
STATIC_DIR = PROJECT_DIR / "static"
USER_DB = BASE_DIR / "users.db"

app = FastAPI(
    title="WaterWise",
    description="Water Consumption Analysis and Forecasting"
)

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR), check_dir=False),
    name="static"
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

PUBLIC_PATHS = {
    "/", "/login", "/signup", "/logout",
    "/logout-page", "/favicon.ico"
}


# --------------------------------------------------
# DATABASE
# --------------------------------------------------

def create_user_table():
    with sqlite3.connect(USER_DB) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL DEFAULT '',
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL
            )
        """)

        existing = {
            row[1]
            for row in con.execute(
                "PRAGMA table_info(users)"
            ).fetchall()
        }

        if "name" not in existing:
            con.execute("""
                ALTER TABLE users
                ADD COLUMN name TEXT NOT NULL DEFAULT ''
            """)

        required = {"email", "password_hash", "salt"}

        if not required.issubset(existing):
            raise RuntimeError(
                "Unexpected users database schema. "
                "Back up the database before making schema changes."
            )


create_user_table()


# --------------------------------------------------
# PASSWORD SECURITY
# --------------------------------------------------

def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        310000
    ).hex()


# --------------------------------------------------
# DATASET AND STATISTICS
# --------------------------------------------------

def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


def statistics(df: pd.DataFrame) -> dict:
    if "Consumption" not in df.columns:
        raise ValueError(
            "Dataset must contain a Consumption column."
        )

    values = pd.to_numeric(
        df["Consumption"],
        errors="coerce"
    ).dropna()

    if values.empty:
        raise ValueError(
            "No valid Consumption values were found."
        )

    return {
        "total_records": int(len(df)),
        "average_consumption": round(float(values.mean()), 3),
        "maximum_consumption": round(float(values.max()), 3),
        "minimum_consumption": round(float(values.min()), 3),
    }


# --------------------------------------------------
# MACHINE LEARNING MODEL
# --------------------------------------------------

_model_cache = None


def get_trained_model():
    global _model_cache

    if _model_cache is None:
        _model_cache = train_model(path=DATA_PATH)

    return _model_cache


def find_metric(metrics, *names):
    normalized = {
        str(key).lower().replace("_", "").replace(" ", ""): value
        for key, value in metrics.items()
    }

    for name in names:
        key = name.lower().replace("_", "").replace(" ", "")

        if key in normalized:
            return normalized[key]

    return None


def predict_consumption(date, time, temperature, rain, sun):
    try:
        dt = datetime.strptime(
            f"{date} {time}",
            "%Y-%m-%d %H:%M"
        )
    except ValueError as exc:
        raise ValueError(
            "Choose a valid date and time."
        ) from exc

    values = [temperature, rain, sun]

    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError(
            "Temperature, rain, and sun must be valid numbers."
        )

    row = pd.DataFrame([{
        "Year": dt.year,
        "Month": dt.month,
        "Day": dt.day,
        "DayOfWeek": dt.weekday(),
        "Hour": dt.hour,
        "DayOfYear": dt.timetuple().tm_yday,
        "Temperature": float(temperature),
        "Rain": float(rain),
        "Sun": float(sun),
    }])

    model, prepared_df, metrics, test_results = get_trained_model()

    return float(model.predict(row[FEATURES])[0])


# --------------------------------------------------
# SESSION AND LOGIN PROTECTION
# --------------------------------------------------

@app.middleware("http")
async def require_login(request: Request, call_next):
    path = request.url.path

    if (
        path in PUBLIC_PATHS
        or path.startswith("/static/")
    ):
        return await call_next(request)

    if not request.session.get("user"):
        return RedirectResponse(
            url="/login",
            status_code=303
        )

    return await call_next(request)


# Keep WATERWISE_SECRET_KEY configured on Render.
# The fallback is only for local development.
app.add_middleware(
    SessionMiddleware,
    secret_key=(
        os.environ.get("WATERWISE_SECRET_KEY")
        or "waterwise-local-development-key"
    ),
    same_site="lax",
    https_only=(
        os.environ.get("WATERWISE_HTTPS_ONLY", "true").lower()
        == "true"
    )
)


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    user = request.session.get("user")

    if not user:
        return RedirectResponse(
            url="/login",
            status_code=303
        )

    success = request.session.pop("flash_message", None)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "user": user,
            "user_name": user.get("name", ""),
            "user_email": user.get("email", ""),
            "success": success
        }
    )


# --------------------------------------------------
# LOGIN
# --------------------------------------------------

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if request.session.get("user"):
        return RedirectResponse(
            url="/",
            status_code=303
        )

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={
            "error": None,
            "registered": False,
            "logged_out": (
                request.query_params.get("logged_out") == "1"
            )
        }
    )


@app.post("/login", response_class=HTMLResponse)
async def login_user(
    request: Request,
    email: str = Form(...),
    password: str = Form(...)
):
    email = email.strip().lower()

    with sqlite3.connect(USER_DB) as con:
        user = con.execute(
            """
            SELECT id, name, email, password_hash, salt
            FROM users
            WHERE email = ?
            """,
            (email,)
        ).fetchone()

    if (
        user is None
        or not hmac.compare_digest(
            hash_password(password, user[4]),
            user[3]
        )
    ):
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={
                "error": "Invalid email or password. Please try again.",
                "registered": False,
                "logged_out": False
            },
            status_code=401
        )

    request.session.clear()

    request.session["user"] = {
        "id": user[0],
        "name": user[1],
        "email": user[2]
    }

    request.session["flash_message"] = (
        "Login successful. Welcome to WaterWise!"
    )

    return RedirectResponse(
        url="/",
        status_code=303
    )


# --------------------------------------------------
# SIGNUP
# --------------------------------------------------

@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    if request.session.get("user"):
        return RedirectResponse(
            url="/",
            status_code=303
        )

    return templates.TemplateResponse(
        request=request,
        name="signuppage.html",
        context={"error": None}
    )


@app.post("/signup", response_class=HTMLResponse)
async def signup_user(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...)
):
    name = name.strip()
    email = email.strip().lower()

    error = None

    if not name:
        error = "Please enter your name."
    elif len(name) > 100:
        error = "Name must be 100 characters or fewer."
    elif (
        len(email) > 254
        or "@" not in email
        or email.startswith("@")
        or email.endswith("@")
    ):
        error = "Please enter a valid email address."
    elif len(password) < 8:
        error = "Password must contain at least 8 characters."
    elif password != confirm_password:
        error = "Passwords do not match. Please try again."

    if error:
        return templates.TemplateResponse(
            request=request,
            name="signuppage.html",
            context={"error": error},
            status_code=400
        )

    salt = secrets.token_hex(16)
    digest = hash_password(password, salt)

    try:
        with sqlite3.connect(USER_DB) as con:
            con.execute(
                """
                INSERT INTO users
                (name, email, password_hash, salt)
                VALUES (?, ?, ?, ?)
                """,
                (name, email, digest, salt)
            )

    except sqlite3.IntegrityError:
        return templates.TemplateResponse(
            request=request,
            name="signuppage.html",
            context={
                "error": (
                    "This email is already registered. Please log in."
                )
            },
            status_code=409
        )

    with sqlite3.connect(USER_DB) as con:
        user = con.execute(
            """
            SELECT id, name, email
            FROM users
            WHERE email = ?
            """,
            (email,)
        ).fetchone()

    request.session.clear()

    request.session["user"] = {
        "id": user[0],
        "name": user[1],
        "email": user[2]
    }

    request.session["flash_message"] = (
        "Account created successfully! Welcome to WaterWise."
    )

    return RedirectResponse(
        url="/",
        status_code=303
    )


# --------------------------------------------------
# LOGOUT
# --------------------------------------------------

@app.get("/logout-page", response_class=HTMLResponse)
async def logout_page(request: Request):
    # The session has already been cleared before this page is opened.
    return templates.TemplateResponse(
        request=request,
        name="logout.html",
        context={"user": None}
    )


@app.post("/logout")
async def logout_post(request: Request):
    # Clear all session data.
    request.session.clear()

    # Redirect to the logout confirmation page.
    return RedirectResponse(
        url="/logout-page",
        status_code=303
    )


@app.get("/logout")
async def logout_get(request: Request):
    # Support navigation links using href="/logout".
    request.session.clear()

    return RedirectResponse(
        url="/logout-page",
        status_code=303
    )


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    try:
        stats = statistics(load_dataset())
        error = None
    except Exception as exc:
        stats = {
            "total_records": 0,
            "average_consumption": 0,
            "maximum_consumption": 0,
            "minimum_consumption": 0
        }
        error = str(exc)

    user = request.session.get("user") or {}
    success = request.session.pop("flash_message", None)

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "user": user,
            "user_name": user.get("name", ""),
            "user_email": user.get("email", ""),
            "success": success,
            "error": error,
            **stats
        }
    )


# --------------------------------------------------
# PREDICTION PAGE
# --------------------------------------------------

@app.get("/predict", response_class=HTMLResponse)
async def prediction_page(request: Request):
    user = request.session.get("user") or {}

    return templates.TemplateResponse(
        request=request,
        name="predict.html",
        context={
            "user": user,
            "user_email": user.get("email", ""),
            "prediction": None,
            "error": None
        }
    )


@app.post("/predict", response_class=HTMLResponse)
async def prediction_form(
    request: Request,
    date: str = Form(...),
    time: str = Form(...),
    temperature: float = Form(...),
    rain: float = Form(...),
    sun: float = Form(...)
):
    user = request.session.get("user") or {}

    base_context = {
        "user": user,
        "user_email": user.get("email", "")
    }

    try:
        value = predict_consumption(
            date, time, temperature, rain, sun
        )

        return templates.TemplateResponse(
            request=request,
            name="predict.html",
            context={
                **base_context,
                "prediction": round(value, 3),
                "error": None,
                "input_date": date,
                "input_time": time
            }
        )

    except Exception as exc:
        return templates.TemplateResponse(
            request=request,
            name="predict.html",
            context={
                **base_context,
                "prediction": None,
                "error": str(exc)
            },
            status_code=400
        )


# --------------------------------------------------
# PREDICTION API
# --------------------------------------------------

@app.post("/api/predict")
async def api_predict(
    date: str = Form(...),
    time: str = Form(...),
    temperature: float = Form(...),
    rain: float = Form(...),
    sun: float = Form(...)
):
    try:
        value = predict_consumption(
            date, time, temperature, rain, sun
        )

        return {
            "prediction": round(value, 3),
            "predicted_consumption": round(value, 3)
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

@app.get("/performance", response_class=HTMLResponse)
async def performance(request: Request):
    user = request.session.get("user") or {}

    try:
        model, df, metrics, test_results = get_trained_model()

        context = {
            "user": user,
            "user_email": user.get("email", ""),
            "metrics": metrics,
            "mae": find_metric(
                metrics, "MAE", "Mean Absolute Error"
            ),
            "rmse": find_metric(
                metrics, "RMSE", "Root Mean Squared Error"
            ),
            "r2": find_metric(
                metrics, "R2", "R²", "r2_score"
            ),
            "total_records": len(df),
            "train_records": int(len(df) * 0.8),
            "test_records": len(df) - int(len(df) * 0.8),
            "error": None,
            "feature_importance": sorted(
                [
                    {
                        "name": name,
                        "value": float(value)
                    }
                    for name, value in zip(
                        FEATURES,
                        model.feature_importances_
                    )
                ],
                key=lambda item: item["value"],
                reverse=True
            )
        }

    except Exception as exc:
        context = {
            "user": user,
            "user_email": user.get("email", ""),
            "metrics": {},
            "mae": None,
            "rmse": None,
            "r2": None,
            "total_records": 0,
            "train_records": 0,
            "test_records": 0,
            "error": str(exc),
            "feature_importance": []
        }

    return templates.TemplateResponse(
        request=request,
        name="performance.html",
        context=context
    )


# --------------------------------------------------
# DATASET PAGE
# --------------------------------------------------

@app.get("/dataset", response_class=HTMLResponse)
async def dataset_page(request: Request):
    user = request.session.get("user") or {}

    try:
        df = load_dataset()
        preview_df = df.head(25)

        rows = preview_df.astype(object).where(
            pd.notna(preview_df),
            None
        ).to_dict(orient="records")

        context = {
            "user": user,
            "user_email": user.get("email", ""),
            "columns": list(df.columns),
            "rows": rows,
            "total_records": len(df),
            "error": None
        }

    except Exception as exc:
        context = {
            "user": user,
            "user_email": user.get("email", ""),
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


# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    user = request.session.get("user") or {}

    return templates.TemplateResponse(
        request=request,
        name="about.html",
        context={
            "user": user,
            "user_email": user.get("email", "")
        }
    )


# --------------------------------------------------
# SUMMARY API
# --------------------------------------------------

@app.get("/api/summary")
async def api_summary():
    try:
        return statistics(load_dataset())
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# --------------------------------------------------
# TREND API
# --------------------------------------------------

@app.get("/api/trend")
async def api_trend():
    try:
        df = load_dataset()

        df["Date"] = pd.to_datetime(
            df["Date"].astype(str),
            errors="coerce",
            dayfirst=True,
            format="mixed"
        )

        df["Consumption"] = pd.to_numeric(
            df["Consumption"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["Date", "Consumption"]
        )

        daily = (
            df.assign(
                Date=df["Date"].dt.strftime("%Y-%m-%d")
            )
            .groupby("Date", as_index=False)["Consumption"]
            .sum()
            .tail(90)
        )

        return {
            "dates": daily["Date"].tolist(),
            "consumption": [
                round(float(value), 3)
                for value in daily["Consumption"]
            ]
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# --------------------------------------------------
# DATASET API
# --------------------------------------------------

@app.get("/api/dataset")
async def api_dataset():
    try:
        df = load_dataset()
        preview_df = df.head(500)

        preview = preview_df.astype(object).where(
            pd.notna(preview_df),
            None
        )

        return {
            "columns": list(df.columns),
            "rows": preview.to_dict(orient="records"),
            "total_records": len(df)
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# --------------------------------------------------
# PERFORMANCE API
# --------------------------------------------------

@app.get("/api/performance")
async def api_performance():
    try:
        model, df, metrics, test_results = get_trained_model()

        return {
            "mae": find_metric(
                metrics, "MAE", "Mean Absolute Error"
            ),
            "rmse": find_metric(
                metrics, "RMSE", "Root Mean Squared Error"
            ),
            "r2": find_metric(
                metrics, "R2", "R²", "r2_score"
            ),
            "metrics": metrics,
            "total_records": len(df),
            "train_records": int(len(df) * 0.8),
            "test_records": len(df) - int(len(df) * 0.8)
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )

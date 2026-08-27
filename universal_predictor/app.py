"""
Customer Lifetime Value (CLV) Predictor - Professional Edition
Advanced ML platform for CLV prediction with business insights
"""
import io
import json
import base64
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, html, dcc, dash_table, Input, Output, State, no_update
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Optional dependencies
try:
    import lightgbm as lgb
    HAVE_LGBM = True
except ImportError:
    HAVE_LGBM = False

try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

try:
    import diskcache
    cache = diskcache.Cache("./cache")
    HAVE_CACHE = True
except ImportError:
    HAVE_CACHE = False

# Configuration
APP_VERSION = "1.0.0"
TEST_SIZE = 0.2
RANDOM_STATE = 42
LGBM_N_ESTIMATORS = 300
XGB_N_ESTIMATORS = 400
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.08
MAX_ROWS_IN_MEMORY = 50000
CV_FOLDS = 5
# A categorical feature (gender, category, payment method, etc.) is only
# useful if it has a small, repeating set of values. Above this many
# distinct values it's more likely a free-text or ID-like column, which
# one-hot encoding would turn into hundreds of near-useless 0/1 columns.
CATEGORICAL_MAX_UNIQUE = 30

CLV_SEGMENTS = {
    "High Value": {"min": 0.75, "color": "#28a745"},
    "Medium Value": {"min": 0.40, "color": "#ffc107"},
    "Low Value": {"min": 0.0, "color": "#dc3545"}
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clv_predictor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Dash app
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
    ],
    title="CLV Predictor Pro",
    suppress_callback_exceptions=True
)
server = app.server

def create_header():
    return dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-chart-line fa-2x text-primary me-3"),
                        html.Div([
                            html.H3("CLV Predictor Pro", className="mb-0"),
                            html.Small("Customer Lifetime Value Analytics Platform", className="text-muted")
                        ])
                    ], className="d-flex align-items-center")
                ], width="auto"),
                dbc.Col([
                    dbc.RadioItems(
                        id="theme-toggle",
                        options=[
                            {"label": html.I(className="fas fa-sun"), "value": "light"},
                            {"label": html.I(className="fas fa-moon"), "value": "dark"}
                        ],
                        value="light",
                        inline=True,
                        className="theme-toggle"
                    )
                ], width="auto", className="ms-auto")
            ], className="w-100 align-items-center")
        ], fluid=True),
        color="white",
        className="shadow-sm mb-4",
        style={"padding": "1rem 0"}
    )

app.layout = html.Div(id="main-container", className="", children=[
    dcc.Store(id="store-df"),
    dcc.Store(id="store-preds"),
    dcc.Store(id="store-metrics"),
    dcc.Store(id="store-importance"),
    dcc.Store(id="store-segments"),
    dcc.Store(id="store-theme", data="light"),
    dcc.Store(id="store-model"),
    
    create_header(),
    
    dbc.Container([
        dbc.Tabs([
            dbc.Tab(label="📊 Data & Training", tab_id="tab-train", children=[
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.I(className="fas fa-upload me-2"),
                                    "Step 1: Upload Dataset"
                                ]),
                                dbc.CardBody([
                                    dcc.Upload(
                                        id="upload",
                                        children=html.Div([
                                            html.I(className="fas fa-cloud-upload-alt fa-3x mb-3 text-primary"),
                                            html.H5("Drag & Drop or Click to Upload"),
                                            html.P("Supported: CSV, Excel (.xlsx, .xls)", className="text-muted mb-0")
                                        ], className="text-center py-4"),
                                        style={
                                            "borderWidth": "2px",
                                            "borderStyle": "dashed",
                                            "borderRadius": "10px",
                                            "borderColor": "#007bff"
                                        }
                                    ),
                                    html.Div(id="upload-msg", className="mt-3"),
                                    html.Div(id="data-preview", className="mt-3")
                                ])
                            ], className="shadow-sm mb-4"),
                            
                            dbc.Card([
                                dbc.CardHeader([
                                    html.I(className="fas fa-cog me-2"),
                                    "Step 2: Configure Model"
                                ]),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label([
                                                html.I(className="fas fa-id-card me-2"),
                                                "Customer ID Column"
                                            ], className="fw-bold"),
                                            dcc.Dropdown(id="col-id", placeholder="Select ID column")
                                        ], md=4),
                                        dbc.Col([
                                            html.Label([
                                                html.I(className="fas fa-bullseye me-2"),
                                                "Target Column (CLV/Revenue)"
                                            ], className="fw-bold"),
                                            dcc.Dropdown(id="col-target", placeholder="Select target")
                                        ], md=4),
                                        dbc.Col([
                                            html.Label([
                                                html.I(className="fas fa-sliders-h me-2"),
                                                "Feature Columns"
                                            ], className="fw-bold"),
                                            dcc.Dropdown(id="col-feats", multi=True, placeholder="Select features")
                                        ], md=4),
                                    ], className="mb-3"),
                                    
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Model Algorithm", className="fw-bold"),
                                            dcc.Dropdown(
                                                id="model-selector",
                                                options=[
                                                    {"label": "🌲 LightGBM (Recommended)", "value": "lightgbm", "disabled": not HAVE_LGBM},
                                                    {"label": "🚀 XGBoost", "value": "xgboost", "disabled": not HAVE_XGB},
                                                    {"label": "📊 Linear Regression", "value": "linear"},
                                                ],
                                                value="lightgbm" if HAVE_LGBM else "linear"
                                            )
                                        ], md=4),
                                        dbc.Col([
                                            html.Label("Advanced Options", className="fw-bold"),
                                            dbc.Checklist(
                                                id="opt-log1p",
                                                options=[{"label": " Log transform (for skewed CLV)", "value": "log"}],
                                                value=["log"],
                                                switch=True
                                            ),
                                            dbc.Checklist(
                                                id="opt-cv",
                                                options=[{"label": " Cross-validation", "value": "cv"}],
                                                value=["cv"],
                                                switch=True
                                            ),
                                            dbc.Checklist(
                                                id="opt-aggregate",
                                                options=[{"label": " Aggregate rows to one-per-customer first", "value": "agg"}],
                                                value=[],
                                                switch=True
                                            ),
                                            html.Small(
                                                "Turn this on if your file has multiple rows per customer "
                                                "(e.g. one row per purchase). It totals up the target column "
                                                "per customer and summarizes each feature automatically.",
                                                className="text-muted"
                                            )
                                        ], md=4),
                                        dbc.Col([
                                            html.Label("Test Split", className="fw-bold"),
                                            dcc.Slider(
                                                id="test-size-slider",
                                                min=0.1,
                                                max=0.4,
                                                step=0.05,
                                                value=0.2,
                                                marks={0.1: "10%", 0.2: "20%", 0.3: "30%", 0.4: "40%"},
                                                tooltip={"placement": "bottom", "always_visible": True}
                                            )
                                        ], md=4)
                                    ]),
                                    
                                    html.Hr(),
                                    
                                    dcc.Loading(
                                        dbc.Button([
                                            html.I(className="fas fa-rocket me-2"),
                                            "Train CLV Model"
                                        ], id="btn-train", color="success", size="lg", className="w-100", disabled=True),
                                        type="default"
                                    ),
                                    
                                    html.Div(id="train-status", className="mt-3")
                                ])
                            ], className="shadow-sm")
                        ], md=12)
                    ])
                ], className="py-4")
            ]),
            
            dbc.Tab(label="📈 Results & Analytics", tab_id="tab-results", children=[
                html.Div([
                    html.Div(id="metrics-cards", className="mb-4"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([html.I(className="fas fa-chart-bar me-2"), "CLV Distribution Analysis"]),
                                dbc.CardBody(dcc.Loading(dcc.Graph(id="pred-hist", style={"height": 400}), type="graph"))
                            ], className="shadow-sm mb-4"),
                            
                            dbc.Card([
                                dbc.CardHeader([html.I(className="fas fa-chart-scatter me-2"), "Model Performance: Actual vs Predicted"]),
                                dbc.CardBody(dcc.Loading(dcc.Graph(id="scatter-plot", style={"height": 400}), type="graph"))
                            ], className="shadow-sm mb-4"),
                            
                            dbc.Card([
                                dbc.CardHeader([html.I(className="fas fa-star me-2"), "Feature Importance Analysis"]),
                                dbc.CardBody(dcc.Loading(dcc.Graph(id="importance-chart", style={"height": 400}), type="graph"))
                            ], className="shadow-sm", id="importance-card", style={"display": "none"})
                        ], md=8),
                        
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([html.I(className="fas fa-users me-2"), "Customer Segmentation"]),
                                dbc.CardBody(html.Div(id="segment-chart"))
                            ], className="shadow-sm mb-4"),
                            
                            dbc.Card([
                                dbc.CardHeader([html.I(className="fas fa-dollar-sign me-2"), "Business Impact"]),
                                dbc.CardBody(html.Div(id="business-metrics"))
                            ], className="shadow-sm mb-4"),
                            
                            dbc.Card([
                                dbc.CardHeader([html.I(className="fas fa-download me-2"), "Export & Actions"]),
                                dbc.CardBody([
                                    dbc.Button([html.I(className="fas fa-file-csv me-2"), "Download Predictions"], 
                                              id="btn-dl", color="primary", className="w-100 mb-2", disabled=True),
                                    dcc.Download(id="dl-preds"),
                                    dbc.Button([html.I(className="fas fa-file-excel me-2"), "Download Detailed Report"], 
                                              id="btn-dl-report", color="info", className="w-100 mb-2", disabled=True),
                                    dcc.Download(id="dl-report"),
                                    html.Hr(),
                                    html.Div(id="stats-panel", className="small")
                                ])
                            ], className="shadow-sm")
                        ], md=4)
                    ])
                ], className="py-4")
            ]),
            
            dbc.Tab(label="👑 Top Customers", tab_id="tab-customers", children=[
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.H4([html.I(className="fas fa-crown me-2 text-warning"), "High-Value Customer Identification"], className="mb-3"),
                            html.P("Focus marketing efforts on customers with highest predicted lifetime value.", className="text-muted")
                        ])
                    ], className="mb-4"),
                    
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Filter by Segment:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id="segment-filter",
                                        options=[
                                            {"label": "🟢 All Customers", "value": "all"},
                                            {"label": "🟢 High Value", "value": "high"},
                                            {"label": "🟡 Medium Value", "value": "medium"},
                                            {"label": "🔴 Low Value", "value": "low"}
                                        ],
                                        value="all"
                                    )
                                ], md=4),
                                dbc.Col([
                                    html.Label("Show Top:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id="top-n",
                                        options=[
                                            {"label": "Top 10", "value": 10},
                                            {"label": "Top 20", "value": 20},
                                            {"label": "Top 50", "value": 50},
                                            {"label": "Top 100", "value": 100}
                                        ],
                                        value=20
                                    )
                                ], md=4),
                                dbc.Col([
                                    html.Label("Actions:", className="fw-bold"),
                                    dbc.Button([html.I(className="fas fa-download me-2"), "Export List"], 
                                              id="btn-export-top", color="success", className="w-100", disabled=True)
                                ], md=4)
                            ], className="mb-3"),
                            
                            dash_table.DataTable(
                                id="top-table",
                                columns=[
                                    {"name": "Rank", "id": "rank"},
                                    {"name": "Customer ID", "id": "customer_id"},
                                    {"name": "Predicted CLV", "id": "predicted_value", "type": "numeric", "format": {"specifier": "$,.2f"}},
                                    {"name": "Segment", "id": "segment"},
                                    {"name": "Percentile", "id": "percentile", "type": "numeric", "format": {"specifier": ".1f"}}
                                ],
                                page_size=20,
                                style_table={"overflowX": "auto"},
                                style_header={"fontWeight": "600", "backgroundColor": "rgb(230, 230, 230)", "textAlign": "center"},
                                style_data_conditional=[
                                    {"if": {"row_index": "odd"}, "backgroundColor": "rgb(248, 248, 248)"},
                                    {"if": {"filter_query": "{segment} = 'High Value'"}, "backgroundColor": "#d4edda", "color": "#155724"},
                                    {"if": {"filter_query": "{segment} = 'Low Value'"}, "backgroundColor": "#f8d7da", "color": "#721c24"}
                                ],
                                style_cell={"textAlign": "left", "padding": "10px"},
                                style_cell_conditional=[
                                    {"if": {"column_id": "rank"}, "width": "80px", "textAlign": "center"},
                                    {"if": {"column_id": "predicted_value"}, "textAlign": "right"},
                                    {"if": {"column_id": "percentile"}, "textAlign": "center"}
                                ],
                                sort_action="native",
                                filter_action="native"
                            )
                        ])
                    ], className="shadow-sm")
                ], className="py-4")
            ]),
            
            dbc.Tab(label="🔍 Model Insights", tab_id="tab-insights", children=[
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Model Performance Metrics"),
                                dbc.CardBody(html.Div(id="detailed-metrics"))
                            ], className="shadow-sm mb-4")
                        ], md=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Model Recommendations"),
                                dbc.CardBody(html.Div(id="recommendations"))
                            ], className="shadow-sm mb-4")
                        ], md=6)
                    ]),
                    
                    dbc.Card([
                        dbc.CardHeader("Residual Analysis"),
                        dbc.CardBody(dcc.Loading(dcc.Graph(id="residual-plot", style={"height": 400}), type="graph"))
                    ], className="shadow-sm")
                ], className="py-4")
            ])
        ], id="main-tabs", active_tab="tab-train")
    ], fluid=True)
])

def parse_upload(contents: str, filename: str) -> pd.DataFrame:
    if not contents or "," not in contents:
        raise ValueError("Invalid file content")
    try:
        _, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)
        if filename and filename.lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            df = pd.read_csv(io.BytesIO(decoded))
        # Duplicate column names make df[col] return a DataFrame instead of a
        # Series, which crashes model .fit(). Rename duplicates before use.
        if df.columns.duplicated().any():
            dupes = df.columns[df.columns.duplicated()].unique().tolist()
            counts = {}
            new_cols = []
            for c in df.columns:
                if list(df.columns).count(c) > 1:
                    counts[c] = counts.get(c, 0) + 1
                    new_cols.append(c if counts[c] == 1 else f"{c}_{counts[c]}")
                else:
                    new_cols.append(c)
            df.columns = new_cols
            logger.warning(f"Renamed duplicate columns in {filename}: {dupes}")
        logger.info(f"Parsed {filename}: {len(df)} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"Failed to parse {filename}: {str(e)}")
        raise

def validate_model_inputs(df, id_col, target_col, feat_cols):
    missing = set([id_col, target_col] + feat_cols) - set(df.columns)
    if missing:
        return False, f"Missing columns: {missing}"
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        return False, "Target column must be numeric"
    valid_rows = df[[id_col, target_col] + feat_cols].dropna()
    if len(valid_rows) < 30:
        return False, f"Insufficient data: only {len(valid_rows)} valid rows"
    # Non-numeric features are now allowed (they get one-hot encoded in
    # train_model), but only if they look like genuine CATEGORIES rather
    # than free text -- e.g. "Male"/"Female" or "Clothing"/"Shoes", not an
    # address, a review comment, or a near-unique ID-like column. A column
    # with too many distinct values would blow up into hundreds of useless
    # feature columns and isn't something the model can learn from anyway.
    bad_text_cols = []
    for f in feat_cols:
        if pd.api.types.is_numeric_dtype(df[f]):
            continue
        n_unique = df[f].nunique(dropna=True)
        if n_unique > CATEGORICAL_MAX_UNIQUE or n_unique > 0.5 * len(df):
            bad_text_cols.append(f)
    if bad_text_cols:
        return False, (f"These columns have too many distinct values to use as categories "
                        f"(free text or ID-like, not a real category): {bad_text_cols}")
    return True, "Valid"

def segment_customers(predictions):
    percentiles = predictions.rank(pct=True)
    segments = pd.Series(index=predictions.index, dtype=str)
    for segment_name, config in CLV_SEGMENTS.items():
        mask = percentiles >= config["min"]
        if segment_name == "Low Value":
            mask = percentiles < CLV_SEGMENTS["Medium Value"]["min"]
        elif segment_name == "Medium Value":
            mask = (percentiles >= config["min"]) & (percentiles < CLV_SEGMENTS["High Value"]["min"])
        segments[mask] = segment_name
    return segments

def aggregate_to_customer_level(df, id_col, target_col, feat_cols):
    """
    Collapses transaction-level data (many rows per customer, e.g. one row
    per purchase) into one row per customer, which is what CLV prediction
    actually needs.

    Defaults, chosen to match how these columns are normally used:
      - target column   -> SUM (e.g. total amount spent = CLV)
      - numeric features -> MEAN (e.g. average age is just age; makes sense
                             for anything that describes the customer rather
                             than a single transaction)
      - text/category features -> MODE (the customer's most common value,
                             e.g. their most-purchased category)
    A new column, "n_transactions", is added automatically -- how many rows
    (purchases) each customer had -- since that's a genuinely useful CLV
    signal that wouldn't otherwise exist after aggregating.
    """
    df = df.loc[:, ~df.columns.duplicated()].copy()
    numeric_feats = [f for f in feat_cols if pd.api.types.is_numeric_dtype(df[f])]
    categorical_feats = [f for f in feat_cols if f not in numeric_feats]

    agg_spec = {target_col: "sum"}
    agg_spec.update({f: "mean" for f in numeric_feats})

    def _mode(series):
        m = series.mode(dropna=True)
        return m.iloc[0] if len(m) else np.nan

    agg_spec.update({f: _mode for f in categorical_feats})

    grouped = df.groupby(id_col, as_index=False).agg(agg_spec)
    counts = df.groupby(id_col, as_index=False).size().rename(columns={"size": "n_transactions"})
    grouped = grouped.merge(counts, on=id_col, how="left")

    logger.info(f"Aggregated {len(df):,} rows -> {len(grouped):,} customers "
                f"(target='{target_col}' summed, {len(numeric_feats)} numeric features averaged, "
                f"{len(categorical_feats)} categorical features set to most-common value)")
    return grouped, feat_cols + ["n_transactions"]

def train_model(df, id_col, target_col, feat_cols, model_type, use_log1p, test_size=0.2, use_cv=True):
    # Drop duplicate column names first: df[col] would otherwise return a
    # DataFrame rather than a Series and break both validation and .fit().
    if df.columns.duplicated().any():
        logger.warning("Dropping duplicate columns before training")
        df = df.loc[:, ~df.columns.duplicated()]
    is_valid, msg = validate_model_inputs(df, id_col, target_col, feat_cols)
    if not is_valid:
        raise ValueError(msg)
    
    work = df[[id_col, target_col] + feat_cols].dropna().copy().reset_index(drop=True)
    work = work.loc[:, ~work.columns.duplicated()]

    # Split requested features into numeric (used as-is) and categorical
    # (text values like "Male"/"Female" or "Clothing"/"Shoes"). A model can
    # only do arithmetic on numbers, so each category gets turned into its
    # own 0/1 column -- e.g. gender becomes "gender_Male" (1 if Male, else
    # 0). This is called one-hot encoding: it's how real-world tools turn
    # words into something a model can learn from.
    numeric_feats = [f for f in feat_cols if pd.api.types.is_numeric_dtype(work[f])]
    categorical_feats = [f for f in feat_cols if f not in numeric_feats]

    X_num = work[numeric_feats] if numeric_feats else pd.DataFrame(index=work.index)
    if categorical_feats:
        X_cat = pd.get_dummies(work[categorical_feats].astype(str), prefix=categorical_feats)
        X = pd.concat([X_num, X_cat], axis=1)
    else:
        X = X_num
    encoded_feat_cols = X.columns.tolist()

    y = work[target_col]
    if isinstance(y, pd.DataFrame):          # duplicate target name
        y = y.iloc[:, 0]
    y = y.astype(float)
    ids = work[id_col].astype(str)
    
    logger.info(f"Training with {len(work)} samples, {len(feat_cols)} requested features "
                f"({len(numeric_feats)} numeric + {len(categorical_feats)} categorical -> "
                f"{len(encoded_feat_cols)} model columns after encoding)")
    
    if use_log1p:
        y_transformed = np.log1p(y)
        inverse_transform = np.expm1
    else:
        y_transformed = y
        inverse_transform = lambda a: a
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=test_size, random_state=RANDOM_STATE)
    
    if model_type == "lightgbm" and HAVE_LGBM:
        model = lgb.LGBMRegressor(random_state=RANDOM_STATE, n_estimators=LGBM_N_ESTIMATORS, verbose=-1)
        model_name = "LightGBM"
    elif model_type == "xgboost" and HAVE_XGB:
        model = XGBRegressor(n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH, learning_rate=XGB_LEARNING_RATE,
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=RANDOM_STATE,
                            objective="reg:squarederror", tree_method="hist", verbosity=0)
        model_name = "XGBoost"
    else:
        model = LinearRegression()
        model_name = "Linear Regression"
    
    logger.info(f"Training {model_name} model...")
    model.fit(X_train, y_train)
    
    y_pred_train = inverse_transform(model.predict(X_train))
    y_pred_test = inverse_transform(model.predict(X_test))
    y_train_orig = inverse_transform(y_train)
    y_test_orig = inverse_transform(y_test)
    preds_all = inverse_transform(model.predict(X))
    if use_log1p:
        # expm1 of a negative prediction yields a negative value; customer value
        # cannot be negative, so floor at zero.
        preds_all = np.clip(preds_all, 0, None)
        y_pred_test = np.clip(y_pred_test, 0, None)
        y_pred_train = np.clip(y_pred_train, 0, None)
    
    train_rmse = float(np.sqrt(mean_squared_error(y_train_orig, y_pred_train)))
    test_rmse = float(np.sqrt(mean_squared_error(y_test_orig, y_pred_test)))
    train_mae = float(mean_absolute_error(y_train_orig, y_pred_train))
    test_mae = float(mean_absolute_error(y_test_orig, y_pred_test))
    train_r2 = float(r2_score(y_train_orig, y_pred_train))
    test_r2 = float(r2_score(y_test_orig, y_pred_test))
    
    cv_scores = None
    if use_cv and len(X_train) > CV_FOLDS * 10:
        try:
            # Score in the SAME units as train/test R2. Without this, CV R2 is
            # computed in log space and is not comparable to the headline R2.
            if use_log1p:
                def _r2_original_units(est, X_cv, y_cv):
                    return r2_score(np.expm1(y_cv), np.expm1(est.predict(X_cv)))
                scorer = _r2_original_units
            else:
                scorer = 'r2'
            cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring=scorer, n_jobs=-1)
            logger.info(f"CV scores: {cv_scores}")
        except Exception as e:
            logger.warning(f"CV failed: {e}")
    
    predictions = pd.DataFrame({"customer_id": ids, "predicted_value": preds_all, "segment": segment_customers(pd.Series(preds_all))})
    test_results = pd.DataFrame({"actual": y_test_orig, "predicted": y_pred_test, "residual": y_test_orig - y_pred_test})
    
    metrics = {
        "model": model_name,
        "use_log1p": use_log1p,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "cv_mean_r2": float(cv_scores.mean()) if cv_scores is not None else None,
        "cv_std_r2": float(cv_scores.std()) if cv_scores is not None else None,
        "rows": len(work),
        "features": feat_cols,
        "test_size": test_size
    }
    
    importance_df = None
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({'feature': encoded_feat_cols, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
    
    logger.info(f"Training complete. Test RMSE: {test_rmse:.3f}, Test R²: {test_r2:.3f}")
    
    return predictions, metrics, importance_df, test_results

def store_dataframe(df):
    if HAVE_CACHE and len(df) > MAX_ROWS_IN_MEMORY:
        import uuid
        cache_key = f"df_{uuid.uuid4()}"
        cache.set(cache_key, df)
        return json.dumps({"type": "cache", "key": cache_key})
    return df.to_json(date_format="iso", orient="split")

def load_dataframe(data):
    try:
        metadata = json.loads(data)
        if isinstance(metadata, dict) and metadata.get("type") == "cache":
            df = cache.get(metadata["key"])
            if df is None:
                raise ValueError("Cached data not found")
            return df
    except (json.JSONDecodeError, ValueError):
        pass
    return pd.read_json(data, orient="split")

@app.callback(
    Output("upload-msg", "children"),
    Output("data-preview", "children"),
    Output("col-id", "options"),
    Output("col-target", "options"),
    Output("col-feats", "options"),
    Output("col-id", "value"),
    Output("col-target", "value"),
    Output("col-feats", "value"),
    Output("btn-train", "disabled"),
    Output("store-df", "data"),
    Input("upload", "contents"),
    State("upload", "filename"),
    prevent_initial_call=True
)
def handle_upload(contents, filename):
    if not contents:
        raise PreventUpdate
    try:
        df = parse_upload(contents, filename)
        cols = df.columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(num_cols) < 2:
            return (dbc.Alert("Need at least 2 numeric columns", color="warning"), None, [], [], [], None, None, None, True, None)
        
        id_keywords = ["customer_id", "customerid", "cust_id", "id", "customer", "user_id"]
        guess_id = next((c for c in cols if any(kw in c.lower() for kw in id_keywords)), cols[0])
        
        target_keywords = ["clv", "ltv", "lifetime_value", "monetary", "revenue", "amount", "sales", "spend", "value", "total"]
        guess_target = next((c for c in num_cols if any(kw in c.lower() for kw in target_keywords)), num_cols[0])
        
        priority_features = ["recency", "frequency", "monetary", "rfm_score", "avg_order_value", "purchase_count", "days_since_last", "cluster", "age", "tenure"]
        default_feats = [c for c in priority_features if c in num_cols]
        if not default_feats:
            default_feats = [c for c in num_cols if c != guess_target][:5]
        
        opts_all = [{"label": c, "value": c} for c in cols]
        opts_num = [{"label": c, "value": c} for c in num_cols]
        
        preview = dbc.Card([
            dbc.CardHeader("Data Preview (First 5 rows)"),
            dbc.CardBody([
                dash_table.DataTable(
                    data=df.head(5).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in df.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '8px'},
                    style_header={'fontWeight': 'bold', 'backgroundColor': 'rgb(230, 230, 230)'}
                )
            ])
        ], className="mt-3")
        
        msg = dbc.Alert([
            html.I(className="fas fa-check-circle me-2"),
            html.Strong(f"✅ {filename} uploaded successfully"),
            html.Hr(className="my-2"),
            html.Div([
                html.I(className="fas fa-database me-2"),
                f"{len(df):,} rows × {len(cols)} columns",
                html.Br(),
                html.I(className="fas fa-calculator me-2"),
                f"{len(num_cols)} numeric columns available"
            ])
        ], color="success")
        
        return (msg, preview, opts_all, opts_num, opts_all, guess_id, guess_target, default_feats, False, store_dataframe(df))
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return (dbc.Alert(f"Error: {str(e)}", color="danger"), None, [], [], [], None, None, None, True, None)

@app.callback(
    Output("train-status", "children"),
    Output("store-preds", "data"),
    Output("store-metrics", "data"),
    Output("store-importance", "data"),
    Output("store-model", "data"),
    Output("btn-dl", "disabled"),
    Output("btn-dl-report", "disabled"),
    Output("btn-export-top", "disabled"),
    Output("main-tabs", "active_tab"),
    Input("btn-train", "n_clicks"),
    State("store-df", "data"),
    State("col-id", "value"),
    State("col-target", "value"),
    State("col-feats", "value"),
    State("model-selector", "value"),
    State("opt-log1p", "value"),
    State("opt-cv", "value"),
    State("opt-aggregate", "value"),
    State("test-size-slider", "value"),
    prevent_initial_call=True
)
def handle_training(n, df_data, id_col, target_col, feats, model_type, logopt, cvopt, aggopt, test_size):
    if not n:
        raise PreventUpdate
    if not df_data:
        return (dbc.Alert("No data", color="danger"), None, None, None, None, True, True, True, "tab-train")
    try:
        df = load_dataframe(df_data)
        use_log = "log" in (logopt or [])
        use_cv = "cv" in (cvopt or [])
        feats = feats or []
        if "agg" in (aggopt or []):
            df, feats = aggregate_to_customer_level(df, id_col, target_col, feats)
        predictions, metrics, importance, test_results = train_model(df, id_col, target_col, feats, model_type, use_log, test_size, use_cv)
        status = dbc.Alert([
            html.I(className="fas fa-check-circle me-2"),
            html.Strong("🎉 Model Training Complete!"),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.Strong("Model:"), f" {metrics['model']}",
                    html.Br(),
                    html.Strong("Features:"), f" {len(feats)}",
                    html.Br(),
                    html.Strong("Samples:"), f" {metrics['rows']:,}",
                ], md=6),
                dbc.Col([
                    html.Strong("Test R²:"), f" {metrics['test_r2']:.3f}",
                    html.Br(),
                    html.Strong("Test RMSE:"), f" {metrics['test_rmse']:,.2f}",
                    html.Br(),
                    html.Strong("Test MAE:"), f" {metrics['test_mae']:,.2f}",
                ], md=6)
            ])
        ], color="success")
        importance_json = importance.to_json(orient="split") if importance is not None else None
        return (status, predictions.to_json(orient="split"), json.dumps(metrics), importance_json, test_results.to_json(orient="split"), False, False, False, "tab-results")
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return (dbc.Alert(f"Training failed: {str(e)}", color="danger"), None, None, None, None, True, True, True, "tab-train")

@app.callback(
    Output("metrics-cards", "children"),
    Input("store-metrics", "data"),
    Input("store-preds", "data"),
    Input("store-theme", "data")
)
def update_metrics(metrics_json, preds_json, theme):
    if not metrics_json or not preds_json:
        return html.Div()
    try:
        metrics = json.loads(metrics_json)
        preds_df = pd.read_json(preds_json, orient="split")
        total_clv = preds_df["predicted_value"].sum()
        avg_clv = preds_df["predicted_value"].mean()
        segments = preds_df["segment"].value_counts()
        high_value_count = segments.get("High Value", 0)
        high_value_pct = (high_value_count / len(preds_df)) * 100 if len(preds_df) > 0 else 0
        cards = dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody([html.I(className="fas fa-chart-line fa-2x text-primary mb-2"), html.H6("Total Predicted CLV", className="text-muted"), html.H3(f"${total_clv:,.0f}", className="mb-0")])], className="shadow-sm text-center"), md=3),
            dbc.Col(dbc.Card([dbc.CardBody([html.I(className="fas fa-user fa-2x text-success mb-2"), html.H6("Avg CLV per Customer", className="text-muted"), html.H3(f"${avg_clv:,.2f}", className="mb-0")])], className="shadow-sm text-center"), md=3),
            dbc.Col(dbc.Card([dbc.CardBody([html.I(className="fas fa-star fa-2x text-warning mb-2"), html.H6("High-Value Customers", className="text-muted"), html.H3(f"{high_value_count:,}", className="mb-0"), html.Small(f"{high_value_pct:.1f}% of total", className="text-muted")])], className="shadow-sm text-center"), md=3),
            dbc.Col(dbc.Card([dbc.CardBody([html.I(className="fas fa-check-circle fa-2x text-info mb-2"), html.H6("Model Accuracy (R²)", className="text-muted"), html.H3(f"{metrics['test_r2']:.3f}", className="mb-0"), html.Small(f"RMSE: {metrics['test_rmse']:,.0f}", className="text-muted")])], className="shadow-sm text-center"), md=3),
        ], className="g-3")
        return cards
    except Exception as e:
        logger.error(f"Metrics error: {str(e)}")
        return html.Div()

@app.callback(
    Output("pred-hist", "figure"),
    Output("stats-panel", "children"),
    Input("store-preds", "data"),
    Input("store-theme", "data")
)
def update_histogram(pred_json, theme):
    if not pred_json:
        return no_update, html.Div()
    try:
        dfp = pd.read_json(pred_json, orient="split")
        template = "plotly_dark" if theme == "dark" else "plotly"
        stats = html.Div([
            html.H6([html.I(className="fas fa-info-circle me-2"), "Quick Stats"], className="mb-3"),
            html.P([html.Strong("Mean: "), f"${dfp['predicted_value'].mean():,.2f}"]),
            html.P([html.Strong("Median: "), f"${dfp['predicted_value'].median():,.2f}"]),
            html.P([html.Strong("Std Dev: "), f"${dfp['predicted_value'].std():,.2f}"]),
            html.P([html.Strong("Range: "), f"${dfp['predicted_value'].min():,.2f} - ${dfp['predicted_value'].max():,.2f}"]),
            html.P([html.Strong("Total Customers: "), f"{len(dfp):,}"])
        ])
        fig = px.histogram(dfp, x="predicted_value", color="segment", nbins=50, template=template, 
                          labels={"predicted_value": "Predicted CLV ($)", "count": "Number of Customers"},
                          color_discrete_map={"High Value": "#28a745", "Medium Value": "#ffc107", "Low Value": "#dc3545"})
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), bargap=0.05, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        return fig, stats
    except Exception as e:
        logger.error(f"Histogram error: {str(e)}")
        return no_update, html.Div()

@app.callback(Output("scatter-plot", "figure"), Input("store-model", "data"), Input("store-theme", "data"))
def update_scatter(model_json, theme):
    if not model_json:
        return go.Figure()
    try:
        test_results = pd.read_json(model_json, orient="split")
        template = "plotly_dark" if theme == "dark" else "plotly"
        fig = px.scatter(test_results, x="actual", y="predicted", labels={"actual": "Actual CLV ($)", "predicted": "Predicted CLV ($)"}, template=template, opacity=0.6)
        max_val = max(test_results["actual"].max(), test_results["predicted"].max())
        fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', name='Perfect Prediction', line=dict(dash='dash', color='red')))
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        return fig
    except Exception as e:
        logger.error(f"Scatter error: {str(e)}")
        return go.Figure()

@app.callback(Output("residual-plot", "figure"), Input("store-model", "data"), Input("store-theme", "data"))
def update_residuals(model_json, theme):
    if not model_json:
        return go.Figure()
    try:
        test_results = pd.read_json(model_json, orient="split")
        template = "plotly_dark" if theme == "dark" else "plotly"
        fig = px.scatter(test_results, x="predicted", y="residual", labels={"predicted": "Predicted CLV ($)", "residual": "Residual"}, template=template, opacity=0.6)
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        return fig
    except Exception as e:
        logger.error(f"Residual error: {str(e)}")
        return go.Figure()

@app.callback(Output("importance-chart", "figure"), Output("importance-card", "style"), Input("store-importance", "data"), Input("store-theme", "data"))
def update_importance(importance_json, theme):
    if not importance_json:
        return go.Figure(), {"display": "none"}
    try:
        importance_df = pd.read_json(importance_json, orient="split")
        template = "plotly_dark" if theme == "dark" else "plotly"
        fig = px.bar(importance_df.head(10), x="importance", y="feature", orientation="h", template=template, labels={"importance": "Importance Score", "feature": "Feature"})
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), yaxis={"categoryorder": "total ascending"})
        return fig, {"display": "block"}
    except Exception as e:
        logger.error(f"Importance error: {str(e)}")
        return go.Figure(), {"display": "none"}

@app.callback(Output("segment-chart", "children"), Output("business-metrics", "children"), Input("store-preds", "data"))
def update_segments(pred_json):
    if not pred_json:
        return html.Div(), html.Div()
    try:
        dfp = pd.read_json(pred_json, orient="split")
        segments = dfp.groupby("segment").agg({"predicted_value": ["count", "sum", "mean"]}).round(2)
        segments.columns = ["count", "total_clv", "avg_clv"]
        segments = segments.reset_index()
        fig = px.pie(segments, values="count", names="segment", color="segment", color_discrete_map={"High Value": "#28a745", "Medium Value": "#ffc107", "Low Value": "#dc3545"})
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=True)
        high_value = segments[segments["segment"] == "High Value"]
        if len(high_value) > 0:
            high_clv_total = high_value["total_clv"].values[0]
            high_clv_pct = (high_clv_total / dfp["predicted_value"].sum()) * 100
        else:
            high_clv_total = 0
            high_clv_pct = 0
        business = html.Div([
            html.H6("💡 Key Insights", className="mb-3"),
            dbc.Alert([html.Strong("Top 25% customers generate:"), html.Br(), html.H4(f"${high_clv_total:,.0f}", className="mb-0 mt-2"), html.Small(f"({high_clv_pct:.1f}% of total CLV)")], color="info"),
            html.P([html.Strong("Recommendation: "), "Focus retention efforts on high-value segment."], className="small text-muted")
        ])
        return dcc.Graph(figure=fig, style={"height": 250}), business
    except Exception as e:
        logger.error(f"Segment error: {str(e)}")
        return html.Div(), html.Div()

@app.callback(Output("top-table", "data"), Input("store-preds", "data"), Input("segment-filter", "value"), Input("top-n", "value"))
def update_top_table(pred_json, segment, top_n):
    if not pred_json:
        return []
    try:
        dfp = pd.read_json(pred_json, orient="split")
        if segment != "all":
            segment_map = {"high": "High Value", "medium": "Medium Value", "low": "Low Value"}
            dfp = dfp[dfp["segment"] == segment_map[segment]]
        dfp = dfp.sort_values("predicted_value", ascending=False).head(top_n or 20)
        dfp["rank"] = range(1, len(dfp) + 1)
        dfp["percentile"] = dfp["predicted_value"].rank(pct=True) * 100
        return dfp[["rank", "customer_id", "predicted_value", "segment", "percentile"]].to_dict("records")
    except Exception as e:
        logger.error(f"Table error: {str(e)}")
        return []

@app.callback(Output("detailed-metrics", "children"), Output("recommendations", "children"), Input("store-metrics", "data"))
def update_insights(metrics_json):
    if not metrics_json:
        return html.Div(), html.Div()
    try:
        metrics = json.loads(metrics_json)
        detailed = html.Div([
            dbc.Table([
                html.Thead(html.Tr([html.Th("Metric"), html.Th("Train"), html.Th("Test")])),
                html.Tbody([
                    html.Tr([html.Td("R² Score"), html.Td(f"{metrics['train_r2']:.4f}"), html.Td(f"{metrics['test_r2']:.4f}")]),
                    html.Tr([html.Td("RMSE"), html.Td(f"{metrics['train_rmse']:,.2f}"), html.Td(f"{metrics['test_rmse']:,.2f}")]),
                    html.Tr([html.Td("MAE"), html.Td(f"{metrics['train_mae']:,.2f}"), html.Td(f"{metrics['test_mae']:,.2f}")])
                ])
            ], bordered=True, striped=True),
            html.P([html.Strong("Model: "), metrics["model"]]),
            html.P([html.Strong("Features: "), f"{len(metrics['features'])}"])
        ])
        r2 = metrics["test_r2"]
        recommendations = []
        if r2 < 0.5:
            recommendations.append(("warning", "⚠️ Low R² score. Consider adding more relevant features or using a different model."))
        elif r2 < 0.7:
            recommendations.append(("info", "📊 Moderate performance. Model is reasonable but could be improved."))
        else:
            recommendations.append(("success", "✅ Excellent model performance! Deploy with confidence."))
        if abs(metrics["train_r2"] - metrics["test_r2"]) > 0.1:
            recommendations.append(("warning", "⚠️ Signs of overfitting detected. Consider regularization or more data."))
        rec_div = html.Div([dbc.Alert(msg, color=color) for color, msg in recommendations])
        return detailed, rec_div
    except Exception as e:
        logger.error(f"Insights error: {str(e)}")
        return html.Div(), html.Div()

@app.callback(Output("dl-preds", "data"), Input("btn-dl", "n_clicks"), State("store-preds", "data"), prevent_initial_call=True)
def download_predictions(n, pred_json):
    if not n or not pred_json:
        raise PreventUpdate
    dfp = pd.read_json(pred_json, orient="split")
    return dcc.send_data_frame(dfp.to_csv, f"clv_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)

@app.callback(Output("dl-report", "data"), Input("btn-dl-report", "n_clicks"), State("store-preds", "data"), State("store-metrics", "data"), prevent_initial_call=True)
def download_report(n, pred_json, metrics_json):
    if not n or not pred_json or not metrics_json:
        raise PreventUpdate
    try:
        dfp = pd.read_json(pred_json, orient="split")
        metrics = json.loads(metrics_json)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            dfp.to_excel(writer, sheet_name='Predictions', index=False)
            segments = dfp.groupby("segment").agg({"predicted_value": ["count", "sum", "mean", "min", "max"]}).round(2)
            segments.to_excel(writer, sheet_name='Segments')
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_excel(writer, sheet_name='Model_Metrics', index=False)
        output.seek(0)
        return dcc.send_bytes(output.getvalue(), f"clv_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    except Exception as e:
        logger.error(f"Report download error: {str(e)}")
        raise PreventUpdate

@app.callback(Output("store-theme", "data"), Input("theme-toggle", "value"))
def toggle_theme(theme):
    return theme or "light"

@app.callback(Output("main-container", "className"), Input("store-theme", "data"))
def apply_page_theme(theme):
    # This is what was missing: the toggle updated the charts (via store-theme)
    # but nothing ever told the PAGE itself (header, cards, background) to
    # repaint. This callback puts a "theme-dark" class on the whole page,
    # which the CSS in assets/dark-mode.css then styles.
    return "theme-dark" if theme == "dark" else ""

if __name__ == "__main__":
    logger.info(f"Starting CLV Predictor Pro v{APP_VERSION}...")
    logger.info(f"LightGBM available: {HAVE_LGBM}")
    logger.info(f"XGBoost available: {HAVE_XGB}")
    logger.info(f"Disk cache available: {HAVE_CACHE}")
    # 127.0.0.1 keeps the app on this machine. "0.0.0.0" exposes it to the
    # whole local network, which is not wanted for a local analysis tool.
    app.run(host="127.0.0.1", port=3100, debug=False)
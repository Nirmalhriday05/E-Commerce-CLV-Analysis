"""
CLV/RFM Dashboard - Production Ready Version
================================================
Customer Lifetime Value and RFM Analysis Dashboard

Features:
- Interactive 3D RFM visualization
- Customer segmentation and clustering
- Predictive analytics integration
- Multi-filter capabilities
- CSV export functionality

Author: Optimized for production use
Version: 2.0 (Fixed)
"""

from pathlib import Path
import json
import os
import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

# ==================== CONFIGURATION ====================

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    """
    Application configuration - all settings in one place
    Override via environment variables for different deployments
    """
    # Paths
    ROOT = Path(os.getenv("PROJECT_ROOT", str(Path(__file__).resolve().parents[1])))
    OUTPUTS = ROOT / "Notebooks" / "Outputs"
    
    # URLs and Ports
    PREDICTOR_URL = os.getenv("PREDICTOR_URL", "http://127.0.0.1:3100")
    DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", 3000))
    DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    
    # App Settings
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    PAGE_SIZE = int(os.getenv("PAGE_SIZE", 15))
    TOP_N_PREDICTIONS = int(os.getenv("TOP_N_PREDICTIONS", 20))
    
    # Visualization
    PLOT_HEIGHT_3D = 420
    HISTOGRAM_BINS = 30


# ==================== FILE PATHS ====================

FILE_RFM = Config.OUTPUTS / "rfm_with_clusters_and_segments.csv"
FILE_SEG_SUM = Config.OUTPUTS / "segment_summary.csv"
FILE_CLU_SUM = Config.OUTPUTS / "cluster_summary.csv"
FILE_VIP = Config.OUTPUTS / "targets_vip.csv"
FILE_CORE = Config.OUTPUTS / "targets_core.csv"
FILE_AR = Config.OUTPUTS / "targets_atrisk.csv"
FILE_PRED_OUT = Config.OUTPUTS / "predicted_customer_value.csv"
FILE_METRICS = Config.OUTPUTS / "model_metrics.json"


# ==================== HELPER FUNCTIONS ====================

def safe_load_csv(filepath, description="file"):
    """
    Safely load CSV with comprehensive error handling
    
    Args:
        filepath: Path to CSV file
        description: Human-readable description for error messages
        
    Returns:
        pandas.DataFrame: Loaded data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: For other loading errors
    """
    try:
        if not filepath.exists():
            raise FileNotFoundError(
                f"{description} not found at: {filepath}\n"
                f"Expected location: {filepath.absolute()}"
            )
        
        df = pd.read_csv(filepath)
        
        if df.empty:
            logger.warning(f"⚠️ {description} is empty: {filepath}")
        else:
            logger.info(f"✅ Loaded {description}: {len(df):,} rows")
        
        return df
        
    except pd.errors.EmptyDataError:
        logger.error(f"❌ {description} is empty or corrupted: {filepath}")
        raise
    except Exception as e:
        logger.error(f"❌ Error loading {description}: {e}")
        raise


def safe_json_load(filepath, description="JSON file"):
    """
    Safely load JSON file with error handling
    
    Args:
        filepath: Path to JSON file
        description: Human-readable description
        
    Returns:
        dict: Loaded JSON data or empty dict on error
    """
    try:
        if not filepath.exists():
            logger.info(f"ℹ️ {description} not found (optional): {filepath}")
            return {}
        
        data = json.loads(filepath.read_text(encoding="utf-8"))
        logger.info(f"✅ Loaded {description}")
        return data or {}
        
    except json.JSONDecodeError as e:
        logger.warning(f"⚠️ Could not parse {description}: {e}")
        return {}
    except Exception as e:
        logger.warning(f"⚠️ Error loading {description}: {e}")
        return {}


def create_empty_figure(message="No data available"):
    """
    Create an empty plotly figure with a message
    
    Args:
        message: Message to display in the empty plot
        
    Returns:
        plotly.graph_objects.Figure: Empty figure with message
    """
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
    )
    return fig


# ==================== DATA LOADING ====================

logger.info("=" * 60)
logger.info("Starting CLV/RFM Dashboard")
logger.info("=" * 60)
logger.info(f"📁 Data directory: {Config.OUTPUTS}")

try:
    # Load required data files
    logger.info("Loading required data files...")
    
    rfm = safe_load_csv(FILE_RFM, "RFM data")
    targets_vip = safe_load_csv(FILE_VIP, "VIP targets")
    targets_core = safe_load_csv(FILE_CORE, "Core targets")
    targets_ar = safe_load_csv(FILE_AR, "At-Risk targets")
    
    # Optional files - we can continue without these
    # seg_summary = safe_load_csv(FILE_SEG_SUM, "Segment summary")
    # cluster_summary = safe_load_csv(FILE_CLU_SUM, "Cluster summary")
    
    logger.info("✅ All required data loaded successfully")
    
except FileNotFoundError as e:
    logger.error("=" * 60)
    logger.error("❌ CRITICAL ERROR: Required files missing")
    logger.error("=" * 60)
    logger.error(str(e))
    logger.error("")
    logger.error("Please ensure the following files exist:")
    logger.error(f"  1. {FILE_RFM}")
    logger.error(f"  2. {FILE_VIP}")
    logger.error(f"  3. {FILE_CORE}")
    logger.error(f"  4. {FILE_AR}")
    logger.error("")
    logger.error("Run your data preparation pipeline first to generate these files.")
    logger.error("=" * 60)
    exit(1)
    
except Exception as e:
    logger.error(f"❌ Unexpected error loading data: {e}")
    exit(1)


# ==================== DATA PROCESSING ====================

logger.info("Processing data...")

# Validate required columns
required_columns = ["customer_id", "recency", "frequency", "monetary", "segment"]
missing_columns = [col for col in required_columns if col not in rfm.columns]
if missing_columns:
    logger.error(f"❌ Missing required columns in RFM data: {missing_columns}")
    exit(1)

# Process cluster column
if "cluster" in rfm.columns:
    try:
        rfm["cluster"] = pd.to_numeric(rfm["cluster"], errors="coerce").astype("Int64")
        logger.info(f"✅ Cluster column processed: {rfm['cluster'].nunique()} clusters")
    except Exception as e:
        logger.warning(f"⚠️ Could not convert cluster to numeric: {e}")
        rfm["cluster"] = pd.NA
else:
    logger.info("ℹ️ No cluster column found in data")

# Clean segment column
rfm["segment"] = (
    rfm["segment"]
    .astype(str)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)
rfm["segment"] = rfm["segment"].replace({"nan": None, "NaN": None, "None": None, "": None})
rfm["segment"] = rfm["segment"].fillna("Unsegmented")
logger.info(f"✅ Segments cleaned: {rfm['segment'].nunique()} unique segments")

# Merge predictions if available
PRED_AVAILABLE = False
if FILE_PRED_OUT.exists():
    try:
        logger.info("Loading predictions...")
        preds = pd.read_csv(FILE_PRED_OUT)
        
        # Handle different column names
        if "predicted_value" not in preds.columns and "predicted_monetary" in preds.columns:
            preds = preds.rename(columns={"predicted_monetary": "predicted_value"})
        
        if {"customer_id", "predicted_value"}.issubset(preds.columns):
            # Ensure consistent data types for merge
            preds["customer_id"] = pd.to_numeric(preds["customer_id"], errors="coerce").astype("Int64").astype(str)
            rfm["customer_id"] = pd.to_numeric(rfm["customer_id"], errors="coerce").astype("Int64").astype(str)
            
            # Merge predictions
            rfm = rfm.merge(
                preds[["customer_id", "predicted_value"]], 
                on="customer_id", 
                how="left"
            )
            
            PRED_AVAILABLE = True
            pred_count = rfm["predicted_value"].notna().sum()
            logger.info(f"✅ Predictions merged: {pred_count:,} customers have predictions")
        else:
            logger.warning("⚠️ Predictions file missing required columns")
            rfm["predicted_value"] = pd.NA
            
    except Exception as e:
        logger.warning(f"⚠️ Could not load predictions: {e}")
        rfm["predicted_value"] = pd.NA
else:
    logger.info("ℹ️ No predictions file found - predictions will not be available")
    rfm["predicted_value"] = pd.NA

# Load model metrics
METRICS = safe_json_load(FILE_METRICS, "Model metrics")
if METRICS:
    r2 = METRICS.get("r2", "N/A")
    rmse = METRICS.get("rmse", "N/A")
    logger.info(f"✅ Model metrics: R²={r2}, RMSE={rmse}")


# ==================== PREPARE DROPDOWN OPTIONS ====================

# Get unique segments (excluding invalid values)
segments = sorted([
    s for s in pd.unique(rfm["segment"].dropna())
    if str(s).strip().lower() not in ["", "nan", "none", "unsegmented"]
])
logger.info(f"✅ Available segments: {len(segments)}")

# Get unique clusters
clusters = []
if "cluster" in rfm.columns:
    try:
        clusters = sorted([
            int(c) for c in rfm["cluster"].dropna().unique().tolist()
        ])
        logger.info(f"✅ Available clusters: {len(clusters)}")
    except (ValueError, TypeError) as e:
        logger.warning(f"⚠️ Could not convert cluster values: {e}")
        clusters = []

# Prepare target cohorts
TARGETS = {
    "vip": set(targets_vip["customer_id"].astype(str)),
    "core": set(targets_core["customer_id"].astype(str)),
    "ar": set(targets_ar["customer_id"].astype(str)),
}

target_options = [
    {"label": f"VIP \u2014 top 20% by value ({len(TARGETS['vip']):,})", "value": "vip"},
    {"label": f"Loyal Core \u2014 K-Means cluster ({len(TARGETS['core']):,})", "value": "core"},
    {"label": f"At-Risk \u2014 valuable & lapsed ({len(TARGETS['ar']):,})", "value": "ar"},
]

logger.info(f"✅ Target cohorts: VIP={len(TARGETS['vip'])}, Core={len(TARGETS['core'])}, At-Risk={len(TARGETS['ar'])}")


# ==================== INITIALIZE DASH APP ====================

external_stylesheets = [dbc.themes.LUX, dbc.icons.BOOTSTRAP]
app = Dash(
    __name__, 
    external_stylesheets=external_stylesheets, 
    title="CLV / RFM Dashboard",
    suppress_callback_exceptions=True
)


# ==================== UI COMPONENTS ====================

def kpi_card(title, value):
    """Create a KPI card component"""
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, className="text-muted fw-semibold mb-1"),
            html.H4(
                f"{value:,}" if isinstance(value, (int, float)) else value,
                className="mb-0 fw-bold"
            ),
        ]),
        className="shadow-sm border-0",
        style={"borderRadius": "14px"}
    )


def fig_card(title, fig_component, icon="bi-graph-up"):
    """Wrap a figure in a styled card"""
    return dbc.Card(
        [
            dbc.CardHeader([
                html.I(className=f"{icon} me-2"),
                html.Span(title, className="fw-semibold")
            ], className="bg-white"),
            dbc.CardBody(dcc.Loading(fig_component, type="dot"), className="pt-2")
        ],
        className="shadow-sm border-0",
        style={"borderRadius": "14px"}
    )


# ==================== FILTER CONTROLS ====================

controls = dbc.Card(
    [
        html.Label("Filter by Segment", className="fw-bold"),
        dcc.Dropdown(
            id="segment-filter",
            options=[{"label": s, "value": s} for s in segments],
            multi=True,
            placeholder="All segments",
        ),
        dbc.Tooltip(
            "Filter by behavioral segments (e.g., Champions, Loyal, At-Risk)", 
            target="segment-filter"
        ),

        html.Br(),
        html.Label("Filter by Cluster", className="fw-bold"),
        dcc.Dropdown(
            id="cluster-filter",
            options=[{"label": f"Cluster {c}", "value": c} for c in clusters],
            multi=True,
            placeholder="All clusters",
        ),
        dbc.Tooltip(
            "K-means clusters computed from RFM metrics", 
            target="cluster-filter"
        ),

        html.Br(),
        html.Label("Filter by Target Cohort", className="fw-bold"),
        dcc.Dropdown(
            id="target-filter",
            options=target_options,
            multi=True,
            placeholder="All customers",
        ),
        dbc.Tooltip(
            "Pre-defined customer cohorts for targeted marketing", 
            target="target-filter"
        ),

        html.Br(),
        html.Label("Color 3D Plot By", className="fw-bold"),
        dcc.Dropdown(
            id="color-by",
            options=(
                [{"label": "Cluster", "value": "cluster"},
                 {"label": "Segment", "value": "segment"}]
                + ([{"label": "Predicted 6-Mo Value", "value": "predicted_value"}] 
                   if PRED_AVAILABLE else [])
            ),
            value=("cluster" if "cluster" in rfm.columns else "segment"),
            clearable=False,
        ),
        dbc.Tooltip(
            "Choose the dimension for color coding in the 3D scatter plot", 
            target="color-by"
        ),

        html.Br(),
        dbc.Button(
            [html.I(className="bi-arrow-counterclockwise me-2"), "Reset Filters"], 
            id="btn-reset", 
            color="secondary", 
            size="sm", 
            className="mt-1 w-100"
        ),
    ],
    className="p-3 shadow-sm border-0",
    style={"borderRadius": "14px"}
)


# ==================== KPI CARDS ====================

def _fmt_pct(x):
    try:
        return f"{float(x)*100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_model(raw):
    """Turn a raw metrics model key (e.g. 'lightgbm_log1p') into a display label."""
    if not raw:
        return "Unknown"
    names = {
        "lightgbm": "LightGBM",
        "lgbm": "LightGBM",
        "xgboost": "XGBoost",
        "xgb": "XGBoost",
        "catboost": "CatBoost",
        "randomforest": "Random Forest",
        "rf": "Random Forest",
        "linreg": "Linear Regression",
        "linear": "Linear Regression",
        "linearregression": "Linear Regression",
    }
    base = str(raw).strip().split("_")[0].lower()
    return names.get(base, str(raw))

def _fmt_r2(x): 
    """Format R² value"""
    return f"{float(x):.3f}" if x or x == 0 else "–"

def _fmt_rmse(x): 
    """Format RMSE value"""
    return f"{float(x):,.0f}" if x or x == 0 else "–"

# Main KPI row
kpi_cards = dbc.Row([
    dbc.Col(kpi_card("Total Customers", rfm["customer_id"].nunique()), md=3),
    dbc.Col(kpi_card("Avg Frequency", round(rfm["frequency"].mean(), 2)), md=3),
    dbc.Col(kpi_card("Median Monetary", f"${int(rfm['monetary'].median()):,}"), md=3),
    dbc.Col(kpi_card("Median Recency", f"{int(rfm['recency'].median())} days"), md=3),
], className="g-3 mb-3")

# Additional KPIs
extra_kpis = dbc.Row([
    dbc.Col(kpi_card("Total Revenue", f"${rfm['monetary'].sum():,.0f}"), md=3),
    dbc.Col(kpi_card("Avg Customer Value", f"${rfm['monetary'].mean():,.0f}"), md=3),
    dbc.Col(kpi_card("Segments", rfm["segment"].nunique()), md=3),
    dbc.Col(kpi_card("Clusters", rfm["cluster"].nunique() if "cluster" in rfm.columns else "–"), md=3),
], className="g-3 mb-3")

# Model performance KPIs (if metrics available)
model_kpis = (
    dbc.Row([
        dbc.Col(kpi_card("Return AUC", _fmt_r2(METRICS.get("return_auc"))), md=3),
        dbc.Col(kpi_card("Top-20% Capture", _fmt_pct(METRICS.get("top20pct_revenue_capture"))), md=3),
        dbc.Col(kpi_card("Customers Scored", rfm["predicted_value"].notna().sum()), md=3),
        dbc.Col(kpi_card("Model Type", _fmt_model(METRICS.get("model"))), md=3),
    ], className="g-3 mb-3")
    if METRICS else html.Div()
)


# ==================== TABLE COLUMNS ====================

TABLE_COLS = ["customer_id", "segment", "cluster", "recency", "frequency", "monetary"]
if PRED_AVAILABLE:
    TABLE_COLS.append("predicted_value")


# ==================== NAVBAR ====================

launch_button = html.A(
    [html.I(className="bi-rocket-takeoff me-2"), "Launch Predictor"],
    href=Config.PREDICTOR_URL,
    target="_blank",
    className="btn btn-primary ms-2"
)

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                html.Span("CLV / RFM Dashboard", className="navbar-brand fw-bold mb-0"),
                href="#",
                style={"textDecoration": "none"}
            ),
            html.Div([
                dbc.Badge("Live", color="success", pill=True, className="me-2"),
                dbc.Badge(f"{len(rfm):,} customers", color="info", pill=True, className="me-2"),
                launch_button,
            ], className="ms-auto d-flex align-items-center")
        ],
        fluid=True
    ),
    color="light",
    dark=False,
    className="mb-3 shadow-sm"
)


# ==================== APP LAYOUT ====================

app.layout = dbc.Container([
    # Navbar
    navbar,
    
    # KPI Cards
    kpi_cards,
    extra_kpis,
    model_kpis,
    
    # Main content with filters and 3D plot
    dbc.Row([
        dbc.Col(controls, md=4),
        dbc.Col(
            fig_card(
                "RFM 3D Scatter Plot", 
                dcc.Graph(id="scatter3d", style={"height": Config.PLOT_HEIGHT_3D}), 
                "bi-view-list"
            ), 
            md=8
        ),
    ], className="g-3 mb-3"),
    
    # Charts row
    dbc.Row([
        dbc.Col(
            fig_card("Customers by Segment", dcc.Graph(id="seg_bar"), "bi-diagram-3"), 
            md=6
        ),
        dbc.Col(
            fig_card("Monetary by Cluster", dcc.Graph(id="cluster_box"), "bi-box"), 
            md=6
        ),
    ], className="g-3 mb-3"),
    
    # Distribution plot
    dbc.Row([
        dbc.Col(
            fig_card(
                "Customer Value Distribution", 
                dcc.Graph(id="clv_hist"), 
                "bi-bar-chart-steps"
            ), 
            md=12
        )
    ], className="g-3 mb-4"),
    
    # Data table section
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.H4("Customer Data Table", className="mb-3"),
            html.P("Click rows in the 3D plot to filter this table", className="text-muted small"),
        ], md=8),
        dbc.Col([
            dbc.Button(
                [html.I(className="bi-download me-2"), "Download CSV"],
                id="dl-btn", 
                color="primary",
                className="float-end"
            ),
        ], md=4),
    ], className="mb-2"),
    
    dash_table.DataTable(
        id="rfm-table",
        columns=[{"name": c.replace("_", " ").title(), "id": c} for c in TABLE_COLS],
        page_size=Config.PAGE_SIZE,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_header={
            "fontWeight": "600",
            "backgroundColor": "rgb(248, 249, 250)",
            "borderBottom": "2px solid #dee2e6"
        },
        style_cell={
            "padding": "12px",
            "textAlign": "left",
            "fontSize": "14px"
        },
        style_data_conditional=[
            {"if": {"column_id": "predicted_value"}, "backgroundColor": "#fff8e1"},
            {"if": {"row_index": "odd"}, "backgroundColor": "rgb(248, 249, 250)"}
        ],
    ),
    dcc.Download(id="dl-csv"),
    
    # Top predictions table (if available)
    html.Hr(className="mt-4"),
    dbc.Row([
        dbc.Col([
            html.H4(f"Top {Config.TOP_N_PREDICTIONS} Customers by Predicted Next-6-Month Value", className="mb-3"),
            html.P("Customers most likely to generate high future value", className="text-muted small"),
        ], md=12),
    ]),
    
    dash_table.DataTable(
        id="top-preds-table",
        columns=[
            {"name": n.replace("_", " ").title(), "id": n}
            for n in ["customer_id", "segment", "cluster", "predicted_value"]
        ],
        page_size=Config.TOP_N_PREDICTIONS,
        style_table={"overflowX": "auto"},
        style_header={
            "fontWeight": "600",
            "backgroundColor": "rgb(248, 249, 250)",
            "borderBottom": "2px solid #dee2e6"
        },
        style_cell={
            "padding": "12px",
            "textAlign": "left",
            "fontSize": "14px"
        },
        style_data_conditional=[
            {"if": {"column_id": "predicted_value"}, "backgroundColor": "#eef9ff", "fontWeight": "600"},
            {"if": {"row_index": "odd"}, "backgroundColor": "rgb(248, 249, 250)"}
        ],
    ),
    
    # Footer
    html.Hr(className="mt-5"),
    html.Footer([
        html.P([
            "CLV/RFM Dashboard • ",
            html.A("Documentation", href="#", className="text-decoration-none"),
            " • ",
            html.A("Support", href="#", className="text-decoration-none")
        ], className="text-center text-muted small mb-4")
    ])
    
], fluid=True, className="px-4 py-3")


# ==================== HELPER FUNCTIONS ====================

def _filtered(df, seg, clu, tgt):
    """
    Apply filters to dataframe
    
    Args:
        df: DataFrame to filter
        seg: List of segments to include
        clu: List of clusters to include
        tgt: List of target cohorts to include
        
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    out = df.copy()
    
    if seg:
        out = out[out["segment"].isin(seg)]
    
    if clu:
        out = out[out["cluster"].isin(clu)]
    
    if tgt:
        selected_ids = set().union(*[TARGETS.get(t, set()) for t in tgt])
        out = out[out["customer_id"].astype(str).isin(selected_ids)]
    
    return out


# ==================== CALLBACKS ====================

@app.callback(
    Output("scatter3d", "figure"),
    Output("seg_bar", "figure"),
    Output("cluster_box", "figure"),
    Output("clv_hist", "figure"),
    Output("rfm-table", "data"),
    Input("segment-filter", "value"),
    Input("cluster-filter", "value"),
    Input("target-filter", "value"),
    Input("color-by", "value"),
    Input("scatter3d", "selectedData"),
)
def update_charts(seg, clu, tgt, color_by, selected):
    """Update all charts based on filter selections"""
    
    # Apply filters
    df = _filtered(rfm, seg, clu, tgt)
    
    # Handle empty dataframe
    if df.empty:
        empty_fig = create_empty_figure("No data matches current filters. Try adjusting your selections.")
        return empty_fig, empty_fig, empty_fig, empty_fig, []
    
    # Validate color_by column
    valid_color_fields = [c for c in ["cluster", "segment", "predicted_value"] if c in df.columns]
    if color_by not in valid_color_fields:
        color_by = "cluster" if "cluster" in valid_color_fields else (
            "segment" if "segment" in valid_color_fields else None
        )
    
    # === 3D Scatter Plot ===
    fig3d = px.scatter_3d(
        df, 
        x="frequency", 
        y="recency", 
        z="monetary",
        color=color_by if color_by in df.columns else None,
        hover_data=["customer_id", "segment"],
        height=Config.PLOT_HEIGHT_3D,
        labels={
            "frequency": "Purchase Frequency",
            "recency": "Days Since Last Purchase",
            "monetary": "Total Spending ($)"
        }
    )
    fig3d.update_traces(
        marker=dict(
            size=5, 
            opacity=0.85, 
            line=dict(width=0.3, color="rgba(0,0,0,0.35)")
        )
    )
    fig3d.update_layout(
        template="plotly_white",
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis_title="Frequency",
            yaxis_title="Recency (days)",
            zaxis_title="Monetary ($)"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # === Segment Bar Chart ===
    seg_counts = df.groupby("segment")["customer_id"].nunique().sort_values(ascending=True)
    if len(seg_counts) > 0:
        fig_seg = px.bar(
            seg_counts,
            orientation='h',
            labels={"value": "Customers", "segment": "Segment"}
        )
        fig_seg.update_layout(
            template="plotly_white",
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
            xaxis_title="Number of Customers",
            yaxis_title=""
        )
        fig_seg.update_traces(marker_color='#636EFA')
    else:
        fig_seg = create_empty_figure("No segment data available")
    
    # === Cluster Box Plot ===
    if "cluster" in df.columns and not df["cluster"].isna().all():
        df_with_cluster = df.dropna(subset=["cluster"])
        if len(df_with_cluster) > 0:
            fig_box = px.box(
                df_with_cluster,
                x="cluster",
                y="monetary",
                labels={"cluster": "Cluster", "monetary": "Monetary Value ($)"}
            )
            fig_box.update_layout(
                template="plotly_white",
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Cluster",
                yaxis_title="Monetary Value ($)"
            )
        else:
            fig_box = create_empty_figure("No cluster data available")
    else:
        fig_box = create_empty_figure("Clustering not available")
    
    # === Monetary Histogram ===
    if len(df) > 0:
        # Remove outliers for better visualization
        lo = float(df["monetary"].quantile(0.01))
        hi = float(df["monetary"].quantile(0.99))
        df_hist = df[(df["monetary"] >= lo) & (df["monetary"] <= hi)]
        
        if len(df_hist) > 0:
            fig_hist = px.histogram(
                df_hist,
                x="monetary",
                nbins=Config.HISTOGRAM_BINS,
                labels={"monetary": "Customer Value ($)"}
            )
            fig_hist.update_layout(
                template="plotly_white",
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Customer Value ($)",
                yaxis_title="Number of Customers",
                showlegend=False
            )
            fig_hist.update_traces(marker_color='#00CC96')
        else:
            fig_hist = create_empty_figure("Insufficient data for histogram")
    else:
        fig_hist = create_empty_figure("No monetary data available")
    
    # === Table Data ===
    table_df = df
    
    # Handle selected points from 3D plot
    try:
        if selected and "points" in selected and len(selected["points"]) > 0:
            ids = []
            for p in selected["points"]:
                idx = p.get("pointIndex", None)
                if idx is not None and 0 <= idx < len(df):
                    ids.append(df.iloc[idx]["customer_id"])
            
            if ids:
                table_df = df[df["customer_id"].astype(str).isin([str(x) for x in ids])]
                logger.info(f"Selected {len(table_df)} customers from 3D plot")
    except Exception as e:
        logger.warning(f"⚠️ Error processing selected data: {e}")
        table_df = df
    
    # Prepare table data
    table_cols = [c for c in TABLE_COLS if c in table_df.columns]
    table_data = table_df[table_cols].to_dict("records")
    
    return fig3d, fig_seg, fig_box, fig_hist, table_data


@app.callback(
    Output("dl-csv", "data"),
    Input("dl-btn", "n_clicks"),
    State("segment-filter", "value"),
    State("cluster-filter", "value"),
    State("target-filter", "value"),
    prevent_initial_call=True
)
def download_table(n, seg, clu, tgt):
    """Download filtered data as CSV"""
    if not n:
        raise PreventUpdate
    
    df = _filtered(rfm, seg, clu, tgt)
    logger.info(f"Downloading {len(df)} rows as CSV")
    
    return dcc.send_data_frame(df.to_csv, "clv_filtered_export.csv", index=False)


@app.callback(
    Output("segment-filter", "value"),
    Output("cluster-filter", "value"),
    Output("target-filter", "value"),
    Output("color-by", "value"),
    Input("btn-reset", "n_clicks"),
    prevent_initial_call=True
)
def reset_filters(n):
    """Reset all filters to default"""
    if not n:
        raise PreventUpdate
    
    default_color = "cluster" if "cluster" in rfm.columns else "segment"
    logger.info("Filters reset to default")
    
    return None, None, None, default_color


@app.callback(
    Output("top-preds-table", "data"),
    Input("segment-filter", "value"),
    Input("cluster-filter", "value"),
    Input("target-filter", "value")
)
def fill_top_preds(seg, clu, tgt):
    """Fill top predictions table"""
    if not PRED_AVAILABLE or "predicted_value" not in rfm.columns:
        return []
    
    df = _filtered(rfm, seg, clu, tgt).dropna(subset=["predicted_value"])
    
    if df.empty:
        return []
    
    keep = [c for c in ["customer_id", "segment", "cluster", "predicted_value"] if c in df.columns]
    top_customers = df.sort_values("predicted_value", ascending=False).head(Config.TOP_N_PREDICTIONS)
    
    return top_customers[keep].to_dict("records")


# ==================== MAIN ====================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 Starting CLV/RFM Dashboard")
    logger.info("=" * 60)
    logger.info(f"📊 Customers loaded: {len(rfm):,}")
    logger.info(f"📈 Predictions available: {'Yes' if PRED_AVAILABLE else 'No'}")
    logger.info(f"🎯 Debug mode: {Config.DEBUG}")
    logger.info(f"🌐 Dashboard URL: http://{Config.DASHBOARD_HOST}:{Config.DASHBOARD_PORT}")
    logger.info(f"🔗 Predictor URL: {Config.PREDICTOR_URL}")
    logger.info("=" * 60)
    
    try:
        # app.run() is correct for Dash 2.0+; run_server was removed in Dash 3.0.
        # getattr fallback keeps this working on pre-2.0 installs.
        _run = getattr(app, "run", None) or getattr(app, "run_server")
        _run(
            host=Config.DASHBOARD_HOST,
            port=Config.DASHBOARD_PORT,
            debug=Config.DEBUG
        )
    except Exception as e:
        logger.error(f"❌ Failed to start dashboard: {e}")
        exit(1)
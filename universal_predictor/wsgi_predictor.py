"""
wsgi_predictor.py - Launch the UniversalPredictor app
Loads the Predictor app from ../UniversalPredictor/app.py and serves it on port 3100
"""

from waitress import serve
from pathlib import Path
import importlib.util
import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
HERE = Path(__file__).resolve().parent                          # .../Desktop/CLV_project
DESKTOP = HERE.parent                                           # .../Desktop
PRED_PATH = DESKTOP / "UniversalPredictor" / "app.py"          # .../Desktop/UniversalPredictor/app.py

logger.info("=" * 60)
logger.info("Starting Universal Predictor")
logger.info("=" * 60)
logger.info(f"Current directory: {HERE}")
logger.info(f"Looking for predictor at: {PRED_PATH}")

# Check if predictor exists
if not PRED_PATH.exists():
    logger.error("=" * 60)
    logger.error("ERROR: Predictor app not found!")
    logger.error("=" * 60)
    logger.error(f"Expected location: {PRED_PATH}")
    logger.error(f"Absolute path: {PRED_PATH.absolute()}")
    logger.error("")
    logger.error("Please ensure the following:")
    logger.error("  1. UniversalPredictor folder exists on Desktop")
    logger.error("  2. app.py exists inside UniversalPredictor")
    logger.error("")
    logger.error("Expected structure:")
    logger.error("  Desktop/")
    logger.error("    ├── CLV_project/")
    logger.error("    │   ├── wsgi_predictor.py (this file)")
    logger.error("    │   └── start_app.bat")
    logger.error("    └── UniversalPredictor/")
    logger.error("        └── app.py")
    logger.error("=" * 60)
    input("Press Enter to exit...")
    sys.exit(1)

try:
    # Load the UniversalPredictor/app.py as a module
    logger.info("Loading predictor app module...")
    spec = importlib.util.spec_from_file_location("universal_predictor_app", str(PRED_PATH))
    module = importlib.util.module_from_spec(spec)
    sys.modules["universal_predictor_app"] = module
    spec.loader.exec_module(module)
    logger.info("✅ Predictor module loaded successfully")
    
    # Find the Dash app instance
    logger.info("Looking for Dash app instance...")
    predict_app = None
    
    # Try common Dash instance names
    for attr_name in ["app", "predictor_app", "dash_app", "application"]:
        if hasattr(module, attr_name):
            predict_app = getattr(module, attr_name)
            logger.info(f"✅ Found Dash app: {attr_name}")
            break
    
    if predict_app is None:
        logger.error("=" * 60)
        logger.error("ERROR: Could not find Dash app in predictor")
        logger.error("=" * 60)
        logger.error("Expected variable names: 'app', 'predictor_app', 'dash_app', 'application'")
        logger.error("")
        logger.error("Please ensure your UniversalPredictor/app.py has:")
        logger.error("  app = Dash(__name__)")
        logger.error("  or")
        logger.error("  predictor_app = Dash(__name__)")
        logger.error("=" * 60)
        input("Press Enter to exit...")
        sys.exit(1)
    
    # Get the server object for Waitress
    if hasattr(predict_app, 'server'):
        server = predict_app.server
    else:
        server = predict_app
    
    logger.info("=" * 60)
    logger.info("🚀 Starting Predictor Server")
    logger.info("=" * 60)
    logger.info("URL: http://127.0.0.1:3100")
    logger.info("Press CTRL+C to stop")
    logger.info("=" * 60)
    
    # Serve the predictor on port 3100
    serve(server, host="0.0.0.0", port=3100, threads=4)
    
except KeyboardInterrupt:
    logger.info("\n⚠️ Predictor stopped by user")
    sys.exit(0)
    
except Exception as e:
    logger.error("=" * 60)
    logger.error("ERROR: Failed to start predictor")
    logger.error("=" * 60)
    logger.error(f"Error: {e}")
    logger.error("")
    logger.error("Traceback:")
    import traceback
    traceback.print_exc()
    logger.error("=" * 60)
    input("Press Enter to exit...")
    sys.exit(1)

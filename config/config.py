from pathlib import Path
import mlflow
import pretty_errors

# Repo
AUTHOR = "NuwanCW"
REPO = "iris_mlops"

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")
STORES_DIR = Path(BASE_DIR, "store")

# Local stores
MODEL_REGISTRY = Path(STORES_DIR, "model")

# create dirs
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)

# Mlflow model registry
mlflow.set_tracking_uri("http://192.168.1.136:5000")
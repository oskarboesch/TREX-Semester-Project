from pathlib import Path

# DATA PATHS
BASEPATH = Path(__file__).parent.parent
DATA_FOLDER = BASEPATH / "data/"
RAW_DATA_FOLDER = DATA_FOLDER / "raw/"
PROCESSED_DATA_FOLDER = DATA_FOLDER / "processed/"
PAPER_EXPERIMENT_DATA_FOLDER = RAW_DATA_FOLDER / "Paper_experiments/"
RESULTS_FOLDER = BASEPATH / "results/"
FIGURES_FOLDER = BASEPATH / "figures/"
MODELS_FOLDER = BASEPATH / "models/"
GRU_RESULTS_FOLDER = RESULTS_FOLDER / "gru_results/"


def ensure_directories():
    """Ensure that all necessary directories exist."""
    for folder in [
        DATA_FOLDER,
        RAW_DATA_FOLDER,
        PROCESSED_DATA_FOLDER,
        PAPER_EXPERIMENT_DATA_FOLDER,
        RESULTS_FOLDER,
        FIGURES_FOLDER,
        MODELS_FOLDER,
        GRU_RESULTS_FOLDER,
    ]:
        folder.mkdir(parents=True, exist_ok=True)


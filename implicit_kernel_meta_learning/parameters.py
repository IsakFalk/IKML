from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "exp_raw"
PROCESSED_DATA_DIR = DATA_DIR / "exp_pro"
FIGURES_DIR = PROJECT_DIR / "plots"
GUILD_HOME_DIR = PROJECT_DIR / ".guild"
GUILD_RUNS_DIR = PROJECT_DIR / ".guild" / "runs"

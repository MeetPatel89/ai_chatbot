from pathlib import Path

from dotenv import load_dotenv

# Project root directory
ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / "data" / "raw"
CHUNKED_DATA_DIR = ROOT_DIR / "data" / "chunked"
VECTOR_DB_DIR = ROOT_DIR / "vectorstore"


def load_config() -> None:
    load_dotenv()

    # Create necessary directories
    for dir_path in [DATA_DIR, CHUNKED_DATA_DIR, VECTOR_DB_DIR]:
        dir_path.mkdir(exist_ok=True, parents=True)

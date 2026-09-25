import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.ingest.indexer import build_index
from app.models.embedder import Embedder


def main() -> None:
    settings = get_settings()
    stats = build_index(settings, Embedder(settings))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

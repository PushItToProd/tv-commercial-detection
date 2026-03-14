from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppConfig:
    matrix_url: str = "http://localhost:5000"
    save_dir: Path = field(default_factory=lambda: Path("frames"))
    load_examples: bool = False
    enable_debounce: bool = False
    output_settings: dict = field(default_factory=lambda: {"ad": {}, "content": {}})


app_config = AppConfig()

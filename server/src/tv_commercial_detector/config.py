from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppConfig:
    matrix_url: str = "http://localhost:5000"
    save_dir: Path = field(default_factory=lambda: Path("frames"))
    enable_debounce: bool = False
    output_settings: dict = field(default_factory=lambda: {"ad": {}, "content": {}})
    classifier_profile: str = "nascar_on_fox"


app_config = AppConfig()

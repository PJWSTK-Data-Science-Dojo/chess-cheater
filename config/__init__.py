from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Self
import dacite as _dct
import tomllib
import json

@dataclass(frozen=True)
class LichessDB:
    catalog: Path
    zst_compressed_db: Path
    lichess_db_file: Path

@dataclass
class Config:
    stockfish: Path
    lichess_db: LichessDB

    @classmethod
    def from_toml(cls, file_name: str | Path) -> Self:
        with open(file_name, "rb") as toml_file:
            data: dict = tomllib.load(toml_file)
        
        return _dct.from_dict(data_class=cls, data=data, config=_dct.Config(cast=[Path]))

    @classmethod
    def from_json(cls, file_name: str | Path) -> Self:
        with open(file_name) as json_file:
            data: dict = json.load(json_file)
        
        return _dct.from_dict(data_class=cls, data=data, config=_dct.Config(cast=[Path]))
    
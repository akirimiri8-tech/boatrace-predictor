"""場コード定義とアプリ設定。"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# stadium_number -> 場名(北から順、boatrace.jp の jcd と同一の並び)
STADIUMS: dict[int, str] = {
    1: "桐生",
    2: "戸田",
    3: "江戸川",
    4: "平和島",
    5: "多摩川",
    6: "浜名湖",
    7: "蒲郡",
    8: "常滑",
    9: "津",
    10: "三国",
    11: "びわこ",
    12: "住之江",
    13: "尼崎",
    14: "鳴門",
    15: "丸亀",
    16: "児島",
    17: "宮島",
    18: "徳山",
    19: "下関",
    20: "若松",
    21: "芦屋",
    22: "福岡",
    23: "唐津",
    24: "大村",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BOATRACE_")

    project_root: Path = Path(__file__).resolve().parents[2]
    db_path: Path = project_root / "data" / "boatrace.db"
    api_base_url: str = "https://boatraceopenapi.github.io"
    api_version: str = "v1"  # 統合api/v1(racers+preview+result)、2026-01-01以降対応
    request_timeout_seconds: float = 15.0

    # 検証対象の会場(stadium_number)。
    # 24=大村(内枠有利で安定・波の影響小さい) / 3=江戸川(汽水・強風・干満差が大きく難解)
    # 天候/波の効果を検証するには対照的な2場にした方が差が出やすい
    # 9=津(2026-08-01時点で直近30日の開催頻度が最も高く、大村・江戸川が
    #   両方休みの日を全てカバーできることを確認済み。日次運用の空振り防止に追加)
    # 8=常滑(2026-08-02追加。津と並んで開催頻度が最も高い(30日中21日)。
    #   大村・江戸川・津が3日連続で全休みになったため追加)
    target_stadiums: list[int] = [24, 3, 9, 8]

    tide_api_base_url: str = "https://tide736.net/api/get_tide.php"


# 場コード -> 潮汐736 API の(都道府県コード, 港コード)。
# 江戸川は競艇場に隣接する河口(市川)、大村は競艇場と同じ大村湾、
# 津・常滑は競艇場と同名の観測点(三重県・愛知県)を採用。
STADIUM_TIDE_STATIONS: dict[int, tuple[int, int]] = {
    24: (42, 50),  # 大村(長崎県, 大村)
    3: (12, 17),  # 江戸川(千葉県, 市川)
    9: (24, 2),  # 津(三重県, 津)
    8: (23, 15),  # 常滑(愛知県, 常滑)
}


settings = Settings()

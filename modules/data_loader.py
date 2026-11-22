
"""Bước 1: tải và đọc UCI Adult, giữ nguyên hai tập dữ liệu gốc."""
from pathlib import Path
from io import BytesIO
import hashlib
import json
from zipfile import ZipFile, BadZipFile

import pandas as pd
import requests

DATA_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/2/adult"
NUMERIC_COLUMNS = [
    "age", "fnlwgt", "education-num", "capital-gain", "capital-loss",
    "hours-per-week",
]
CATEGORICAL_COLUMNS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]
REQUIRED_FILES = ("adult.data", "adult.test", "adult.names")


def _validate_zip(payload):
    """Kiểm tra ZIP trước khi giải nén; chỉ chấp nhận các tệp cần dùng."""
    with ZipFile(BytesIO(payload)) as archive:
        missing = set(REQUIRED_FILES) - set(archive.namelist())
        if missing:
            raise ValueError(f"ZIP thiếu các tệp: {sorted(missing)}")
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError(f"Tệp trong ZIP bị hỏng: {corrupt}")


def download_adult(data_dir="data", force=False):
    """Tự tải dữ liệu công khai; tái sử dụng ZIP hợp lệ khi chạy lại."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / "adult.zip"
    source = "download"
    payload = None
    if archive_path.is_file() and not force:
        try:
            payload = archive_path.read_bytes()
            _validate_zip(payload)
            source = "cache"
        except (BadZipFile, ValueError):
            payload = None
    if payload is None:
        response = requests.get(DATA_URL, timeout=(15, 120))
        response.raise_for_status()
        payload = response.content
        _validate_zip(payload)
        temporary = archive_path.with_suffix(".zip.download")
        temporary.write_bytes(payload)
        temporary.replace(archive_path)
    with ZipFile(BytesIO(payload)) as archive:
        # Tên thành viên cố định: không giải nén các đường dẫn tùy ý trong ZIP.
        for filename in REQUIRED_FILES:
            (data_dir / filename).write_bytes(archive.read(filename))
    return {
        "url": DATA_URL,
        "archive": str(archive_path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "source": source,
    }


def read_adult_file(path, is_test=False):
    """Đọc cột có kiểu rõ ràng, chuẩn hóa chuỗi; không loại mẫu dữ liệu."""
    types = {column: "Int64" for column in NUMERIC_COLUMNS}
    types.update({column: "string" for column in CATEGORICAL_COLUMNS + ["income"]})
    frame = pd.read_csv(
        path, header=None, names=COLUMNS, sep=",", dtype=types,
        skiprows=1 if is_test else 0, skipinitialspace=True,
        na_values=["?"], keep_default_na=False,
    )
    for column in CATEGORICAL_COLUMNS + ["income"]:
        frame[column] = frame[column].str.strip().replace("?", pd.NA)
    if is_test:
        frame["income"] = frame["income"].str.replace(r"\.$", "", regex=True)
    return frame


def validate_original_splits(train, test):
    """Dừng rõ ràng nếu tệp nguồn, số mẫu hoặc nhãn không đúng dự kiến."""
    for name, frame, expected_rows in (
        ("train", train, 32561), ("test", test, 16281),
    ):
        if frame.shape != (expected_rows, 15):
            raise ValueError(f"{name}: kích thước không đúng: {frame.shape}")
        if list(frame.columns) != COLUMNS:
            raise ValueError(f"{name}: tên hoặc thứ tự cột không đúng")
        if frame["income"].isna().any():
            raise ValueError(f"{name}: có nhãn bị thiếu")
        if set(frame["income"].unique()) != {"<=50K", ">50K"}:
            raise ValueError(f"{name}: có nhãn ngoài hai lớp quy định")


def load_adult(data_dir="data"):
    """Đọc hai tệp gốc đã tải, trả về train và test riêng biệt."""
    data_dir = Path(data_dir)
    train = read_adult_file(data_dir / "adult.data")
    test = read_adult_file(data_dir / "adult.test", is_test=True)
    validate_original_splits(train, test)
    return train, test


def export_step01(train, test, download_info, project_dir="."):
    """Lưu bảng đã đọc và biên bản kiểm tra; chưa xuất đặc trưng mô hình."""
    validate_original_splits(train, test)
    root = Path(project_dir)
    loaded_dir = root / "data" / "loaded"
    result_dir = root / "results" / "step_01"
    loaded_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(loaded_dir / "adult_train.csv", index=False, encoding="utf-8")
    test.to_csv(loaded_dir / "adult_test.csv", index=False, encoding="utf-8")
    inventory = {
        "step": 1,
        "source_page": SOURCE_PAGE,
        "download_url": DATA_URL,
        "archive_sha256": download_info["sha256"],
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "feature_count": 14,
        "target": "income",
        "labels": ["<=50K", ">50K"],
        "columns": COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "dtypes": {c: str(t) for c, t in train.dtypes.items()},
        "changes": ["strip_whitespace", "recognize_question_mark_as_missing", "remove_test_comment", "normalize_test_label_suffix"],
        "model_features_exported": False,
    }
    target = result_dir / "data_inventory.json"
    target.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    return inventory

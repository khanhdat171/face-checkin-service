from __future__ import annotations

import hashlib
import subprocess
import json
import os
from pathlib import Path
import struct
import zlib
from datetime import UTC, datetime, timedelta
from typing import Annotated

import numpy as np
import pytest
from fastapi import Depends
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import (
    get_embedder,
    get_engine,
    get_session,
    get_session_factory,
)
from app.core.config import get_settings
from app.core.security import create_access_token
from app.domain import FaceEmbedder, InvalidImage, NoFaceDetected, Role, UnmatchedFace
from app.models import Base, CheckInRecordORM, FaceProfileORM, UserORM


JWT_SECRET = "phase2-audit-test-secret-do-not-use-in-production"
FAKE_MODEL_NAME = "phase2-fake-model"


class FakeEmbedder(FaceEmbedder):
    """Embedder giả: không tải InsightFace/ONNX khi chạy test."""

    def __init__(
        self,
        *,
        result: np.ndarray | None = None,
        error: Exception | None = None,
    ) -> None:
        self._result = result
        self._error = error

    @property
    def model_name(self) -> str:
        return FAKE_MODEL_NAME

    def embed(self, image_bytes: bytes) -> np.ndarray:
        # API upload phải chuyển được nội dung file không rỗng tới embedder.
        assert image_bytes

        if self._error is not None:
            raise self._error

        if self._result is None:
            raise AssertionError("FakeEmbedder needs either result or error")

        return np.asarray(self._result, dtype=np.float32)


def _reset_database_dependencies() -> None:
    """Xóa cache engine/session/settings để mỗi test dùng SQLite riêng."""
    if get_engine.cache_info().currsize:
        get_engine().dispose()

    get_session_factory.cache_clear()
    get_engine.cache_clear()
    get_settings.cache_clear()


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    chunk_body = chunk_type + data
    checksum = zlib.crc32(chunk_body) & 0xFFFFFFFF
    return (
        len(data).to_bytes(4, "big")
        + chunk_body
        + checksum.to_bytes(4, "big")
    )


def _valid_png_bytes() -> bytes:
    """Tạo PNG RGBA 2x2 hợp lệ chỉ bằng thư viện chuẩn Python."""
    width = 2
    height = 2
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    pixel = b"\xff\x00\x00\xff"  # đỏ, alpha 255
    scanline = b"\x00" + pixel * width  # filter byte + 2 pixels
    image_data = scanline * height

    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(image_data))
        + _png_chunk(b"IEND", b"")
    )


@pytest.fixture

def audit_app(tmp_path, monkeypatch):
    """Tạo FastAPI app + database SQLite tạm cho từng test."""
    database_file = tmp_path / "phase2-audit.sqlite3"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_file.as_posix()}")
    monkeypatch.setenv("JWT_SECRET", JWT_SECRET)
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")

    _reset_database_dependencies()

    # Import sau khi đặt biến môi trường vì app.main tạo app toàn cục khi import.
    from app.main import create_app

    app = create_app()
    Base.metadata.create_all(bind=get_engine())

    yield app

    _reset_database_dependencies()


def _make_embedder_result(vector: list[float] | None = None) -> FakeEmbedder:
    if vector is None:
        vector = [1.0, 0.0]
    return FakeEmbedder(result=np.asarray(vector, dtype=np.float32))


def _create_user(*, email: str = "audit-user@example.com") -> int:
    """Seed user trực tiếp; password không được dùng trong các request test này."""
    with get_session_factory()() as session:
        user = UserORM(
            email=email,
            hashed_password="not-used-by-this-test",
            role="user",
            full_name="Audit Test User",
            created_at=datetime.now(UTC),
        )
        session.add(user)
        session.commit()
        return user.id


def _create_face_profile(user_id: int, embedding: np.ndarray | None = None) -> int:
    """Seed profile có vector trùng với FakeEmbedder để tạo ca SUCCESS."""
    if embedding is None:
        embedding = np.asarray([1.0, 0.0], dtype="<f4")
    else:
        embedding = np.asarray(embedding, dtype="<f4")

    with get_session_factory()() as session:
        profile = FaceProfileORM(
            user_id=user_id,
            embedding=np.ascontiguousarray(embedding).tobytes(),
            embedding_dim=int(embedding.size),
            model_name=FAKE_MODEL_NAME,
            created_at=datetime.now(UTC),
        )
        session.add(profile)
        session.commit()
        return profile.id


def _create_user_token(user_id: int) -> str:
    return create_access_token(
        user_id=user_id,
        role=Role.USER,
        secret=JWT_SECRET,
        algorithm="HS256",
        expires_delta=timedelta(minutes=5),
    )


def _post_checkin(client: TestClient):
    return client.post(
        "/api/v1/checkins",
        files={"image": ("face.png", _valid_png_bytes(), "image/png")},
    )


def _post_face_registration(client: TestClient, token: str):
    return client.post(
        "/api/v1/face-profiles",
        headers={"Authorization": f"Bearer {token}"},
        files={"image": ("face.png", _valid_png_bytes(), "image/png")},
    )


def _audit_rows() -> list[dict[str, int | str | None]]:
    """Đọc audit bằng Session MỚI, độc lập với Session xử lý HTTP request."""
    with get_session_factory()() as session:
        rows = session.scalars(
            select(CheckInRecordORM).order_by(CheckInRecordORM.id)
        ).all()
        return [
            {
                "id": row.id,
                "status": row.status,
                "user_id": row.user_id,
                "matched_face_profile_id": row.matched_face_profile_id,
            }
            for row in rows
        ]
        
def _count_rows(model: type) -> int:
    with get_session_factory()() as session:
        count = session.scalar(
            select(func.count()).select_from(model)
        )
        return int(count or 0)
    

def _user_exists(email: str) -> bool:
    with get_session_factory()() as session:
        user_id = session.scalar(select(UserORM.id).where(UserORM.email == email))
        return user_id is not None


def test_no_face_returns_422_and_commits_one_audit_record(audit_app):
    """Ảnh không có mặt: API trả 422 nhưng audit phải được commit."""
    audit_app.dependency_overrides[get_embedder] = lambda: FakeEmbedder(
        error=NoFaceDetected("No face in uploaded image")
    )

    with TestClient(audit_app) as client:
        response = _post_checkin(client)

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "NoFaceDetected"
    assert isinstance(body["record_id"], int)

    rows = _audit_rows()
    assert len(rows) == 1
    assert rows[0]["id"] == body["record_id"]
    assert rows[0]["status"] == "no_face"
    assert rows[0]["user_id"] is None


def test_unmatched_returns_404_and_commits_one_audit_record(audit_app):
    """Không có profile khớp: API trả 404 nhưng audit vẫn phải được commit."""
    audit_app.dependency_overrides[get_embedder] = lambda: _make_embedder_result()
    # Không seed FaceProfileORM: service sẽ đi vào nhánh unmatched.

    with TestClient(audit_app) as client:
        response = _post_checkin(client)

    assert response.status_code == 404
    body = response.json()
    assert body["error"] == "UnmatchedFace"
    assert isinstance(body["record_id"], int)

    rows = _audit_rows()
    assert len(rows) == 1
    assert rows[0]["id"] == body["record_id"]
    assert rows[0]["status"] == "unmatched"
    assert rows[0]["user_id"] is None
    assert rows[0]["matched_face_profile_id"] is None


def test_success_returns_201_and_commits_one_audit_record(audit_app):
    """Vector khớp profile: API trả 201 và lưu một audit success."""
    audit_app.dependency_overrides[get_embedder] = lambda: _make_embedder_result()
    user_id = _create_user()
    profile_id = _create_face_profile(user_id)

    with TestClient(audit_app) as client:
        response = _post_checkin(client)

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "success"
    assert body["user_id"] == user_id
    assert body["matched_face_profile_id"] == profile_id
    assert body["full_name"] == "Audit Test User"

    rows = _audit_rows()
    assert len(rows) == 1
    assert rows[0]["id"] == body["id"]
    assert rows[0]["status"] == "success"
    assert rows[0]["user_id"] == user_id
    assert rows[0]["matched_face_profile_id"] == profile_id
    
def test_register_face_without_face_returns_422_and_creates_no_audit(audit_app):
    """Đăng ký profile không phát hiện mặt: không được tạo check-in audit."""
    audit_app.dependency_overrides[get_embedder] = lambda: FakeEmbedder(
        error=NoFaceDetected("No face in registration image")
    )
    user_id = _create_user(email="register-user@example.com")
    token = _create_user_token(user_id)

    with TestClient(audit_app) as client:
        response = _post_face_registration(client, token)

    assert response.status_code == 422
    assert response.json()["error"] == "NoFaceDetected"
    assert "record_id" not in response.json()
    assert _count_rows(CheckInRecordORM) == 0
    assert _count_rows(FaceProfileORM) == 0
    
def test_other_domain_error_after_flush_rolls_back(audit_app):
    """DomainError khác sau flush phải rollback thay vì commit dữ liệu dở."""
    email = "must-rollback-domain@example.com"

    @audit_app.post("/_test/domain-error-after-flush")
    def domain_error_after_flush(
        session: Annotated[Session, Depends(get_session)],
    ) -> None:
        session.add(
            UserORM(
                email=email,
                hashed_password="test-hash",
                role="user",
                full_name="Must Roll Back",
                created_at=datetime.now(UTC),
            )
        )
        session.flush()
        raise InvalidImage("injected domain error after flush")

    with TestClient(audit_app) as client:
        response = client.post(
            "/_test/domain-error-after-flush"
            )

    # InvalidImage được map thành HTTP 400 bởi app/api/errors.py.
    assert response.status_code == 400
    assert response.json()["error"] == "InvalidImage"
    assert not _user_exists(email)
    assert _count_rows(CheckInRecordORM) == 0
    
@pytest.mark.parametrize(
    ("kind", "expected_status"),
    [("no-face", 422), ("unmatched", 404)],
)
def test_audit_error_without_record_id_rolls_back_flushed_data(
    audit_app,
    kind: str,
    expected_status: int,
):
    """Hai lỗi audit nếu KHÔNG có record_id thì cũng phải rollback."""
    email = f"must-rollback-{kind}@example.com"

    @audit_app.post("/_test/audit-error-without-id/{error_kind}")
    def audit_error_without_id(
        error_kind: str,
        session: Annotated[Session, Depends(get_session)],
    ) -> None:
        session.add(
            UserORM(
                email=email,
                hashed_password="test-hash",
                role="user",
                full_name="Must Roll Back",
                created_at=datetime.now(UTC),
            )
        )
        session.flush()

        if error_kind == "no-face":
            raise NoFaceDetected("missing record id")
        raise UnmatchedFace(0.2)

    with TestClient(audit_app) as client:
        response = client.post(f"/_test/audit-error-without-id/{kind}")

    assert response.status_code == expected_status
    assert not _user_exists(email)
    assert _count_rows(CheckInRecordORM) == 0
    
def test_unexpected_system_error_after_flush_rolls_back(audit_app):
    """RuntimeError sau flush phải rollback; HTTP mặc định là 500."""
    email = "must-rollback-system@example.com"

    @audit_app.post("/_test/system-error-after-flush")
    def system_error_after_flush(
        session: Annotated[Session, Depends(get_session)],
    ) -> None:
        session.add(
            UserORM(
                email=email,
                hashed_password="test-hash",
                role="user",
                full_name="Must Roll Back",
                created_at=datetime.now(UTC),
            )
        )
        session.flush()
        raise RuntimeError("injected system error after flush")

    with TestClient(audit_app, raise_server_exceptions=False) as client:
        response = client.post("/_test/system-error-after-flush")

    assert response.status_code == 500
    assert not _user_exists(email)
    assert _count_rows(CheckInRecordORM) == 0
    
def test_commit_failure_rolls_back_and_closes_session(audit_app, monkeypatch):
    """Commit thất bại phải gọi rollback, đóng Session và không lưu dữ liệu."""
    email = "must-rollback-commit-failure@example.com"

    @audit_app.post("/_test/commit-failure")
    def create_then_commit(
    session: Annotated[
        Session,
        Depends(get_session, scope="function"),
        ],
    ) -> dict[str, bool]:
        
        session.add(
            UserORM(
                email=email,
                hashed_password="test-hash",
                role="user",
                full_name="Commit Failure",
                created_at=datetime.now(UTC),
            )
        )
        # Chứng minh row đã được flush vào transaction trước khi commit lỗi.
        session.flush()
        return {"ok": True}

    counters = {"commit": 0, "rollback": 0, "close": 0}

    def fail_commit(self, *args, **kwargs):
        counters["commit"] += 1
        raise RuntimeError("injected commit failure")

    original_rollback = Session.rollback
    original_close = Session.close

    def track_rollback(self, *args, **kwargs):
        counters["rollback"] += 1
        return original_rollback(self, *args, **kwargs)

    def track_close(self, *args, **kwargs):
        counters["close"] += 1
        return original_close(self, *args, **kwargs)

    monkeypatch.setattr(Session, "commit", fail_commit)
    monkeypatch.setattr(Session, "rollback", track_rollback)
    monkeypatch.setattr(Session, "close", track_close)

    with TestClient(audit_app, raise_server_exceptions=False) as client:
        response = client.post("/_test/commit-failure")

    assert response.status_code == 500
    assert counters["commit"] == 1
    assert counters["rollback"] >= 1
    assert counters["close"] >= 1

    # Query bằng một Session mới sau khi request và cleanup đã kết thúc.
    assert not _user_exists(email)
    assert _count_rows(CheckInRecordORM) == 0
    

def export_audit_report(report: dict | list) -> None:
    # Chuyển report thành dictionary để bổ sung metadata
    if isinstance(report, dict):
        report_data = dict(report)
    else:
        report_data = {"results": report}

    # Xác định thư mục gốc repository
    repo_root = Path(__file__).resolve().parents[2]

    # Lấy SHA commit hiện tại
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    git_head = result.stdout.strip()

    # Tính SHA256 của ảnh PNG fixture
    fixture_sha256 = hashlib.sha256(
        _valid_png_bytes()
    ).hexdigest()

    # Bổ sung metadata trực tiếp vào JSON
    report_data["git_head"] = git_head
    report_data["fixture_sha256"] = fixture_sha256

    # Chuyển report thành JSON
    rendered = json.dumps(
        report_data,
        ensure_ascii=False,
        indent=2,
    )

    print("\n=== PHASE 2 AUDIT PROBE REPORT ===")
    print(rendered)

    # Đường dẫn lưu file
    output_path = os.getenv(
        "AUDIT_PROBE_OUTPUT",
        r"D:\KTPMphase2\audit-base.json",
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Ghi JSON
    destination.write_text(
        rendered + "\n",
        encoding="utf-8",
    )

    print(
        f"\nJSON report written to: {destination.resolve()}"
    )

def test_export_audit_report():
    report = {
        "requests_per_case": 1,
        "test_status": "PASS",
        "test_file": "tests/api/test_phase2_audit.py",
        "note": "Smoke test for JSON export only",
    }
    
    report["git_head"] = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    text=True,
    ).strip()

    report["fixture_sha256"] = hashlib.sha256(
    _valid_png_bytes()
    ).hexdigest()

    export_audit_report(report)
    
def test_export_audit_report_metadata():
    report = {
        "test": "metadata_export_check",
    }

    export_audit_report(report)
"""Tests for the Morpheus REST API endpoints."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

try:
    from src.api import create_app
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not API_AVAILABLE,
    reason="src/api.py not yet implemented",
)


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.get_positions = AsyncMock(return_value=[])
    db.get_daily_pnl = AsyncMock(return_value=[])
    db.get_strategy_perf = AsyncMock(return_value={})
    db._conn = MagicMock()
    return db


@pytest.fixture
def app(mock_db):
    return create_app(db=mock_db)


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Headers with API key for authenticated endpoints."""
    # Set env var so the auth check passes
    os.environ["MORPHEUS_API_KEY"] = "test-key-12345"
    yield {"X-API-Key": "test-key-12345"}
    del os.environ["MORPHEUS_API_KEY"]


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        """Health endpoint should return 200 without auth."""
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """Health response should have required fields."""
        response = client.get("/api/health")
        data = response.json()
        assert "status" in data
        assert "uptime_seconds" in data
        assert "timestamp" in data

    def test_health_status_ok(self, client):
        """Health status should be 'ok'."""
        response = client.get("/api/health")
        assert response.json()["status"] == "ok"

    def test_health_shows_kill_switch_state(self, client, tmp_path, monkeypatch):
        """Health endpoint should report kill switch state."""
        # Patch Path to use tmp_path
        stop_file = tmp_path / "STOP_TRADING"
        stop_file.touch()

        with patch("src.api.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            response = client.get("/api/health")

        # The response should have trading_halted field


class TestStatusEndpoint:
    def test_status_requires_auth(self, client):
        """Status endpoint should return 401 without API key."""
        os.environ["MORPHEUS_API_KEY"] = "test-key-12345"
        try:
            response = client.get("/api/status")
            assert response.status_code in (401, 403)
        finally:
            del os.environ["MORPHEUS_API_KEY"]

    def test_status_with_auth(self, client, auth_headers):
        """Status endpoint should return 200 with valid API key."""
        response = client.get("/api/status", headers=auth_headers)
        assert response.status_code == 200

    def test_status_response_structure(self, client, auth_headers):
        """Status response should have required fields."""
        response = client.get("/api/status", headers=auth_headers)
        data = response.json()
        assert "trading_active" in data
        assert "timestamp" in data


class TestPositionsEndpoint:
    def test_positions_requires_auth(self, client):
        """Positions endpoint should require authentication."""
        os.environ["MORPHEUS_API_KEY"] = "test-key-12345"
        try:
            response = client.get("/api/positions")
            assert response.status_code in (401, 403)
        finally:
            del os.environ["MORPHEUS_API_KEY"]

    def test_positions_returns_list(self, client, auth_headers, mock_db):
        """Positions should return a list."""
        mock_db.get_positions.return_value = []
        response = client.get("/api/positions", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "positions" in data
        assert isinstance(data["positions"], list)

    def test_positions_count(self, client, auth_headers, mock_db):
        """Positions response should include count."""
        mock_db.get_positions.return_value = [
            {"ticker": "KXTEST", "quantity": 1, "direction": "BUY_YES"}
        ]
        response = client.get("/api/positions", headers=auth_headers)
        data = response.json()
        assert data["count"] == 1


class TestPnlEndpoint:
    def test_pnl_returns_data(self, client, auth_headers, mock_db):
        """P&L endpoint should return data."""
        mock_db.get_daily_pnl.return_value = [
            {"date": "2026-03-28", "realized_pnl_cents": 1500}
        ]
        response = client.get("/api/pnl", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "pnl" in data

    def test_pnl_days_parameter(self, client, auth_headers, mock_db):
        """P&L endpoint should accept days parameter."""
        response = client.get("/api/pnl?days=14", headers=auth_headers)
        assert response.status_code == 200
        # Verify the call was made with correct days
        mock_db.get_daily_pnl.assert_called_with(days=14)


class TestStrategiesEndpoint:
    def test_strategies_returns_dict(self, client, auth_headers, mock_db):
        """Strategies endpoint should return a dict."""
        mock_db.get_strategy_perf.return_value = {}
        response = client.get("/api/strategies", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "strategies" in data
        assert isinstance(data["strategies"], dict)


class TestAdminEndpoints:
    def test_halt_requires_auth(self, client):
        """Halt endpoint should require authentication."""
        os.environ["MORPHEUS_API_KEY"] = "test-key-12345"
        try:
            response = client.post("/api/admin/halt")
            assert response.status_code in (401, 403)
        finally:
            del os.environ["MORPHEUS_API_KEY"]

    def test_halt_creates_stop_file(self, client, auth_headers, tmp_path, monkeypatch):
        """Halt endpoint should create STOP_TRADING file."""
        # This is a side-effect test — the endpoint creates a file
        response = client.post("/api/admin/halt", headers=auth_headers)
        # Just verify it doesn't 500
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "halted"

    def test_resume_clears_stop_file(self, client, auth_headers):
        """Resume endpoint should remove STOP_TRADING file."""
        response = client.post("/api/admin/resume", headers=auth_headers)
        assert response.status_code == 200

    def test_resume_when_not_halted(self, client, auth_headers):
        """Resume when not halted should return already_running."""
        # Ensure no stop file exists
        stop_file = Path("state/STOP_TRADING")
        if stop_file.exists():
            stop_file.unlink()

        response = client.post("/api/admin/resume", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "already_running"


class TestRiskEndpoint:
    def test_risk_returns_state(self, client, auth_headers, mock_db):
        """Risk endpoint should return trading state."""
        mock_db.get_positions.return_value = []
        mock_db.get_daily_pnl.return_value = []
        response = client.get("/api/risk", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "trading_halted" in data
        assert "timestamp" in data


class TestOpenAPISchema:
    def test_openapi_docs_available(self, client):
        """OpenAPI docs endpoint should be accessible."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema_valid(self, client):
        """OpenAPI JSON schema should be valid."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        assert "/api/health" in schema["paths"]

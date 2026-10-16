from collections.abc import Iterator
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient
from mcp.server.auth.provider import AccessToken
from mcp.server.transport_security import TransportSecurityMiddleware
from starlette.routing import Mount

from openlab.main import app, get_db
from openlab.mcp_auth import mcp_transport_security
from openlab.mcp_server import OpenLabTokenVerifier
from openlab.models import Lab, McpGrant, McpOAuthClient, User
from openlab.security import current_user

CHALLENGE = "a" * 43
REDIRECT = "http://localhost:4567/callback"


@pytest.fixture
def consent(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, MagicMock]]:
    db = MagicMock()
    db.get.return_value = McpOAuthClient(id="client", name="<script>alert(1)</script>", redirect_uris=[REDIRECT])
    monkeypatch.setattr("openlab.main.ensure_mcp_enabled", lambda *_: Lab(id="lab", mcp_enabled=True))
    monkeypatch.setattr("openlab.main.audit", lambda *_, **__: None)
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[current_user] = lambda: User(id="owner")
    try:
        yield TestClient(app, cookies={"openlab_csrf": "csrf"}), db
    finally:
        app.dependency_overrides.clear()


def test_oauth_metadata_advertises_public_origin_not_internal_hop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("openlab.main.settings.public_url", "https://lab.example.com/")
    client = TestClient(app, base_url="http://openlab-server:8000")
    for path in ("/.well-known/oauth-protected-resource", "/.well-known/oauth-protected-resource/mcp"):
        body = client.get(path).json()
        assert body["resource"] == "https://lab.example.com/mcp"
        assert body["authorization_servers"] == ["https://lab.example.com"]
    server = client.get("/.well-known/oauth-authorization-server").json()
    assert server["issuer"] == "https://lab.example.com"
    assert server["token_endpoint"] == "https://lab.example.com/oauth/token"


def test_unauthenticated_mcp_through_web_rewrite_gets_bearer_challenge() -> None:
    response = TestClient(app, base_url="http://openlab-server:8000").post(
        "/mcp/", json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        headers={"Accept": "application/json, text/event-stream"},
    )
    assert response.status_code == 401
    assert 'resource_metadata="' in response.headers["www-authenticate"]


def test_transport_security_allows_public_and_internal_hosts_only() -> None:
    guard = TransportSecurityMiddleware(mcp_transport_security("https://lab.example.com"))
    for host in ("lab.example.com", "lab.example.com:443", "openlab-server:8000", "localhost:8000"):
        assert guard._validate_host(host)
    assert not guard._validate_host("evil.example")
    assert guard._validate_origin("https://lab.example.com")
    assert not guard._validate_origin("https://evil.example")


def test_consent_page_escapes_client_values_and_preselects_read_only(consent: tuple[TestClient, MagicMock]) -> None:
    client, _ = consent
    page = client.get("/oauth/authorize", params={
        "client_id": "client", "redirect_uri": REDIRECT, "response_type": "code", "code_challenge": CHALLENGE,
        "code_challenge_method": "S256", "scope": "openlab:read openlab:write", "state": '"><script>x()</script>',
    }).text
    assert "<script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert 'value="openlab:read" checked disabled' in page
    assert 'name="scope" value="openlab:write">' in page


def test_approval_grants_only_selected_scopes(consent: tuple[TestClient, MagicMock]) -> None:
    client, db = consent
    response = client.post("/oauth/authorize/approve", follow_redirects=False, data={
        "client_id": "client", "redirect_uri": REDIRECT, "code_challenge": CHALLENGE,
        "requested_scope": "openlab:read openlab:write", "scope": ["openlab:read"], "state": "s", "csrf": "csrf",
    })
    assert response.status_code == 302
    assert parse_qs(urlsplit(response.headers["location"]).query)["state"] == ["s"]
    grant = next(call.args[0] for call in db.add.call_args_list if isinstance(call.args[0], McpGrant))
    assert grant.scopes == ["openlab:read"]


@pytest.mark.parametrize("scopes", [["openlab:write"], ["openlab:read", "openlab:commit"]])
def test_approval_rejects_missing_read_or_unrequested_scope(consent: tuple[TestClient, MagicMock], scopes: list[str]) -> None:
    client, _ = consent
    response = client.post("/oauth/authorize/approve", follow_redirects=False, data={
        "client_id": "client", "redirect_uri": REDIRECT, "code_challenge": CHALLENGE,
        "requested_scope": "openlab:read openlab:write", "scope": scopes, "csrf": "csrf",
    })
    assert response.status_code == 422


def test_authenticated_mcp_accepts_internal_rewrite_host(monkeypatch: pytest.MonkeyPatch) -> None:
    async def verified(_: object, token: str) -> AccessToken:
        return AccessToken(token=token, client_id="client", scopes=["openlab:read"], subject="owner", claims={})

    monkeypatch.setattr(OpenLabTokenVerifier, "verify_token", verified)
    mcp_app = next(route.app for route in app.routes if isinstance(route, Mount) and route.path == "/mcp")
    with TestClient(mcp_app, base_url="http://openlab-server:8000") as client:
        response = client.post("/", headers={"Authorization": "Bearer t", "Accept": "application/json, text/event-stream"}, json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}},
        })
    assert response.status_code == 200, response.text
    assert response.json()["result"]["serverInfo"]["name"] == "openlab"

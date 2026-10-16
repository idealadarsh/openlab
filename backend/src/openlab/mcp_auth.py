"""OAuth credential helpers shared by OpenLab's product MCP routes and server."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from secrets import token_urlsafe
from urllib.parse import urlsplit

from fastapi import HTTPException
from mcp.server.transport_security import TransportSecuritySettings
from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Lab, McpGrant, McpOAuthClient, User
from .services import lab_for_user

MCP_SCOPES = {"openlab:read", "openlab:write", "openlab:commit", "openlab:ai"}
ACCESS_LIFETIME = timedelta(minutes=15)
REFRESH_LIFETIME = timedelta(days=30)


def hash_credential(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_mcp_url(public_url: str | None) -> str | None:
    if not public_url:
        return None
    value = public_url.rstrip("/")
    return f"{value}/mcp" if value.startswith("https://") else None


def oauth_origin(public_url: str | None, request_origin: str) -> str:
    """Public origin for OAuth metadata; the request origin is the internal hop behind the web proxy."""
    return (public_url or request_origin).rstrip("/")


def mcp_transport_security(public_url: str | None) -> TransportSecuritySettings:
    # Requests arrive through the Next.js rewrite (internal service host) or a
    # local tunnel; the public hostname is allowed for direct proxy setups.
    hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*", "openlab-server", "openlab-server:*"]
    origins = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]
    if public_url:
        parsed = urlsplit(public_url)
        if parsed.hostname:
            hosts += [parsed.netloc, f"{parsed.hostname}:*"]
            origins.append(f"{parsed.scheme}://{parsed.netloc}")
    return TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=hosts, allowed_origins=origins)


def ensure_enabled(db: Session, user: User) -> Lab:
    lab_id = lab_for_user(db, user)
    lab = db.get(Lab, lab_id)
    if not lab or not lab.mcp_enabled:
        raise HTTPException(status_code=403, detail="MCP access is not enabled for this lab")
    return lab


def register_public_client(db: Session, client_id: str, name: str, redirect_uris: list[str]) -> McpOAuthClient:
    if not client_id or len(client_id) > 120 or not redirect_uris:
        raise HTTPException(status_code=422, detail="Invalid MCP public client")
    if any(not uri.startswith(("http://127.0.0.1:", "http://localhost:", "https://")) for uri in redirect_uris):
        raise HTTPException(status_code=422, detail="MCP redirect URIs must be loopback HTTP or HTTPS")
    client = db.get(McpOAuthClient, client_id)
    if client:
        if client.redirect_uris != redirect_uris:
            raise HTTPException(status_code=409, detail="MCP client redirect URIs do not match registration")
        return client
    client = McpOAuthClient(id=client_id, name=name[:200] or "MCP client", redirect_uris=redirect_uris, grant_types=["authorization_code", "refresh_token"])
    db.add(client)
    db.flush()
    return client


def issue_grant(db: Session, *, user: User, client: McpOAuthClient, scopes: list[str]) -> tuple[McpGrant, str, str]:
    selected = sorted(set(scopes))
    if not selected or not set(selected).issubset(MCP_SCOPES):
        raise HTTPException(status_code=422, detail="Invalid MCP scopes")
    now = datetime.now(UTC)
    access = token_urlsafe(32)
    refresh = token_urlsafe(40)
    grant = McpGrant(
        lab_id=lab_for_user(db, user), user_id=user.id, client_id=client.id, scopes=selected,
        access_token_hash=hash_credential(access), access_expires_at=now + ACCESS_LIFETIME,
        refresh_token_hash=hash_credential(refresh), refresh_expires_at=now + REFRESH_LIFETIME,
    )
    db.add(grant)
    db.flush()
    return grant, access, refresh


def rotate_refresh(db: Session, grant: McpGrant) -> tuple[str, str]:
    now = datetime.now(UTC)
    if grant.revoked_at or not grant.refresh_expires_at or grant.refresh_expires_at <= now:
        raise HTTPException(status_code=401, detail="MCP refresh credential is expired or revoked")
    access, refresh = token_urlsafe(32), token_urlsafe(40)
    grant.access_token_hash = hash_credential(access)
    grant.access_expires_at = now + ACCESS_LIFETIME
    grant.refresh_token_hash = hash_credential(refresh)
    grant.last_used_at = now
    return access, refresh


def actor_for_access_token(db: Session, raw_token: str) -> tuple[User, McpGrant] | None:
    now = datetime.now(UTC)
    grant = db.scalar(select(McpGrant).where(McpGrant.access_token_hash == hash_credential(raw_token)))
    if not grant or grant.revoked_at or not grant.access_expires_at or grant.access_expires_at <= now:
        return None
    user = db.get(User, grant.user_id)
    if not user:
        return None
    lab = db.get(Lab, grant.lab_id)
    if not lab or not lab.mcp_enabled:
        return None
    grant.last_used_at = now
    db.commit()
    return user, grant

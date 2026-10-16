import base64
import html
import logging
import re
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from http import HTTPStatus
from io import BytesIO
from secrets import compare_digest, token_urlsafe
from urllib.parse import urlencode

import qrcode  # type: ignore[import-untyped]
from argon2.exceptions import VerifyMismatchError
from fastapi import (
    Cookie,
    Depends,
    FastAPI,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.responses import Response as FastAPIResponse
from qrcode.image.svg import SvgPathImage  # type: ignore[import-untyped]
from sqlalchemy import func, or_, select, text
from sqlalchemy.orm import Session, aliased

from .alternatives import create_build_from_alternative
from .config import get_settings
from .db import get_db
from .enrichment import enrich_thing, queue_thing_enrichment
from .host_setup import (
    HostSetupInput,
    HostSetupOperation,
    HostSetupOut,
    host_setup_status,
    queue_host_setup,
)
from .installation import (
    InstallationOverview,
    InstallationPolicy,
    NetworkInput,
    NetworkOut,
    OnboardingOut,
    ReadinessReport,
    application_readiness,
    installation_overview,
    network_out,
    normalize_public_url,
    provider_fingerprint,
    save_installation_policy,
)
from .intelligence import accept_build_plan, queue_thing_embedding, search_inventory
from .mcp_auth import (
    ACCESS_LIFETIME,
    MCP_SCOPES,
    canonical_mcp_url,
    hash_credential,
    mcp_transport_security,
    oauth_origin,
    register_public_client,
    rotate_refresh,
)
from .mcp_auth import (
    ensure_enabled as ensure_mcp_enabled,
)
from .mcp_server import product_mcp
from .models import (
    Allocation,
    Capability,
    CommunicationConsent,
    InboxCandidate,
    InboxItem,
    Job,
    Lab,
    Location,
    McpAuthorizationCode,
    McpGrant,
    McpOAuthClient,
    Membership,
    Pin,
    Project,
    ProviderConfig,
    Requirement,
    SessionToken,
    StockBalance,
    StockMovement,
    TechnicalFact,
    TelemetryOutbox,
    TelemetryState,
    Thing,
    ThingAlias,
    ThingInterface,
    User,
)
from .providers import (
    OpenAICompatibleProvider,
    ProviderError,
    decrypt_secret,
    encrypt_secret,
    is_local_endpoint,
)
from .schemas import (
    AIAnswer,
    AIQuery,
    AllocationCreate,
    AllocationRecover,
    AlternativeSearchRequest,
    BalanceOut,
    BuildPlanAccept,
    BuildPlanRequest,
    CompatibilityRequest,
    CompatibilityResult,
    FactCreate,
    InboxCandidateBatchConfirm,
    InboxCandidateConfirm,
    InboxCandidateInput,
    InboxCandidateOut,
    InboxCandidatePatch,
    InboxCandidateReceive,
    InboxCapture,
    InboxEnrichURL,
    InboxOut,
    JobOut,
    KicadSettingsInput,
    KicadSettingsOut,
    KnowledgeSearchRequest,
    KnowledgeSearchResult,
    LabOut,
    LabSettingsInput,
    LocationCreate,
    LocationOut,
    LocationQRInfo,
    LoginRequest,
    McpIntegrationInput,
    McpIntegrationOut,
    McpRevokeInput,
    NewsletterSettingsInput,
    PinOut,
    PinoutReplace,
    ProjectCreate,
    ProjectDetailOut,
    ProjectOut,
    ProjectUpdate,
    ProviderConfigInput,
    ProviderConfigOut,
    ProviderModelsOut,
    RequirementCreate,
    SchematicAccept,
    SchematicRequest,
    SettingsOverviewOut,
    SetupRequest,
    StockAdjustment,
    StockMovementDetailOut,
    StockMovementOut,
    StockMutation,
    TelemetrySettingsInput,
    TelemetrySettingsOut,
    ThingCreate,
    ThingKnowledgeReplace,
    ThingOut,
    ThingPatch,
    UserOut,
)
from .schematics import accept_schematic, export_kicad_schematic
from .security import create_session, current_user, hasher, require_csrf
from .services import (
    active_provider,
    adjust_inventory,
    apply_movement,
    audit,
    available_quantity,
    compatible_things,
    create_thing,
    enrich_product_url,
    get_lab_location,
    get_lab_thing,
    lab_for_user,
    location_capture_url,
    refresh_inbox_status,
    save_upload,
)
from .system_settings import effective_kicad_cli, normalize_kicad_cli
from .telemetry import DISCLOSURE_VERSION, ensure_state, queue_outbox, queue_preference

settings = get_settings()
bootstrap_token = settings.setup_token or token_urlsafe(24)
logger = logging.getLogger("openlab")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    if not settings.setup_token:
        logger.warning("OpenLab one-time owner setup token: %s", bootstrap_token)
    # Starlette never runs a mounted app's lifespan, so the MCP session manager must be started here.
    async with product_mcp.session_manager.run():
        yield


app = FastAPI(
    lifespan=lifespan,
    title="OpenLab API",
    version="0.1.0",
    openapi_url="/api/v1/openapi.json",
    docs_url=None,
    redoc_url=None,
)
app.mount(
    "/mcp",
    product_mcp.streamable_http_app(
        streamable_http_path="/",
        json_response=True,
        stateless_http=True,
        max_request_body_size=64 * 1024,
        transport_security=mcp_transport_security(settings.public_url),
    ),
)


def problem(code: int, title: str, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=code,
        content={"type": "about:blank", "title": title, "status": code, "detail": detail},
        media_type="application/problem+json",
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return problem(exc.status_code, HTTPStatus(exc.status_code).phrase, str(exc.detail))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    # Pydantic's default errors include raw request inputs (passwords/API keys).
    errors = [{"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
              for error in exc.errors()]
    return problem(422, "Validation failed", str(errors))


def require_idempotency(value: str | None = Header(default=None, alias="Idempotency-Key")) -> str:
    if not value or len(value) > 128:
        raise HTTPException(status_code=400, detail="A valid Idempotency-Key header is required")
    return value


def require_owner(user: User = Depends(current_user)) -> User:
    if not user.is_owner:
        raise HTTPException(status_code=403, detail="Only the lab owner may change settings")
    return user


def mcp_grant_out(db: Session, grant: McpGrant) -> dict[str, object]:
    client = db.get(McpOAuthClient, grant.client_id)
    return {
        "id": grant.id,
        "client_id": grant.client_id,
        "client_name": client.name if client else "Unknown MCP client",
        "scopes": grant.scopes,
        "created_at": grant.created_at,
        "last_used_at": grant.last_used_at,
        "refresh_expires_at": grant.refresh_expires_at,
    }


@app.get("/api/v1/integrations/mcp", response_model=McpIntegrationOut)
def get_mcp_integration(user: User = Depends(require_owner), db: Session = Depends(get_db)) -> dict[str, object]:
    lab = db.get(Lab, lab_for_user(db, user))
    assert lab is not None
    grants = db.scalars(
        select(McpGrant)
        .where(McpGrant.lab_id == lab.id, McpGrant.revoked_at.is_(None))
        .order_by(McpGrant.created_at.desc())
    ).all()
    return {
        "enabled": lab.mcp_enabled,
        "direct_http_ready": canonical_mcp_url(lab.public_url or settings.public_url) is not None
            and bool(lab.public_url_verified_at if lab.public_url else settings.public_url),
        "mcp_url": canonical_mcp_url(lab.public_url or settings.public_url),
        "grants": [mcp_grant_out(db, grant) for grant in grants],
    }


@app.put(
    "/api/v1/integrations/mcp",
    response_model=McpIntegrationOut,
    dependencies=[Depends(require_csrf)],
)
def save_mcp_integration(
    payload: McpIntegrationInput,
    user: User = Depends(require_owner),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    lab = db.get(Lab, lab_for_user(db, user))
    assert lab is not None
    lab.mcp_enabled = payload.enabled
    audit(db, user, "mcp.integration_updated", "lab", lab.id, enabled=payload.enabled)
    db.commit()
    return get_mcp_integration(user, db)


@app.post("/api/v1/integrations/mcp/revoke", status_code=204, dependencies=[Depends(require_csrf)])
def revoke_mcp_grant(
    payload: McpRevokeInput,
    user: User = Depends(require_owner),
    db: Session = Depends(get_db),
) -> None:
    grant = db.scalar(
        select(McpGrant).where(McpGrant.id == payload.grant_id, McpGrant.lab_id == lab_for_user(db, user))
    )
    if not grant:
        raise HTTPException(status_code=404, detail="MCP grant not found")
    grant.revoked_at = datetime.now(UTC)
    audit(db, user, "mcp.grant_revoked", "mcp_grant", grant.id)
    db.commit()


@app.get("/.well-known/oauth-protected-resource", include_in_schema=False)
@app.get("/.well-known/oauth-protected-resource/mcp", include_in_schema=False)
def mcp_protected_resource_metadata(request: Request) -> dict[str, object]:
    # RFC 9728 path-specific URL is what the MCP SDK advertises in WWW-Authenticate.
    origin = oauth_origin(settings.public_url, str(request.base_url))
    return {
        "resource": f"{origin}/mcp",
        "authorization_servers": [origin],
        "scopes_supported": sorted(MCP_SCOPES),
    }


@app.get("/.well-known/oauth-authorization-server", include_in_schema=False)
def mcp_authorization_metadata(request: Request) -> dict[str, object]:
    origin = oauth_origin(settings.public_url, str(request.base_url))
    return {
        "issuer": origin,
        "authorization_endpoint": f"{origin}/oauth/authorize",
        "token_endpoint": f"{origin}/oauth/token",
        "registration_endpoint": f"{origin}/oauth/register",
        "revocation_endpoint": f"{origin}/oauth/revoke",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "scopes_supported": sorted(MCP_SCOPES),
        "token_endpoint_auth_methods_supported": ["none"],
    }


@app.post("/oauth/register", include_in_schema=False)
def register_mcp_oauth_client(payload: dict[str, object], db: Session = Depends(get_db)) -> dict[str, object]:
    redirect_uris = payload.get("redirect_uris")
    if not isinstance(redirect_uris, list) or not all(isinstance(value, str) for value in redirect_uris):
        raise HTTPException(status_code=422, detail="redirect_uris must be a non-empty string list")
    client_id = token_urlsafe(24)
    name = str(payload.get("client_name", "MCP client"))
    client = register_public_client(db, client_id, name, list(redirect_uris))
    db.commit()
    return {"client_id": client.id, "client_name": client.name, "redirect_uris": client.redirect_uris, "grant_types": client.grant_types, "token_endpoint_auth_method": "none"}


@app.get("/oauth/authorize", response_class=HTMLResponse, include_in_schema=False)
def authorize_mcp_oauth(
    request: Request,
    client_id: str,
    redirect_uri: str,
    response_type: str,
    code_challenge: str,
    code_challenge_method: str,
    scope: str = "openlab:read",
    state: str | None = None,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    if response_type != "code" or code_challenge_method != "S256" or len(code_challenge) < 43:
        raise HTTPException(status_code=422, detail="MCP OAuth requires authorization code with PKCE S256")
    ensure_mcp_enabled(db, user)
    client = db.get(McpOAuthClient, client_id)
    requested = sorted(set(scope.split()))
    if not client or redirect_uri not in client.redirect_uris or not requested or not set(requested).issubset(MCP_SCOPES):
        raise HTTPException(status_code=422, detail="Invalid MCP OAuth authorization request")
    csrf = request.cookies.get("openlab_csrf")
    if not csrf:
        raise HTTPException(status_code=403, detail="CSRF cookie is required for MCP approval")
    fields = {"client_id": client_id, "redirect_uri": redirect_uri, "code_challenge": code_challenge, "requested_scope": " ".join(requested), "state": state or "", "csrf": csrf}
    hidden = "".join(f'<input type="hidden" name="{html.escape(key)}" value="{html.escape(value)}">' for key, value in fields.items())
    # Only read access is preselected; broader scopes must be opted into here.
    choices = "".join(
        f'<label><input type="checkbox" name="scope" value="{html.escape(value)}"'
        + (" checked disabled" if value == "openlab:read" else "")
        + f"> {html.escape(value)}</label><br>"
        for value in requested
    )
    if "openlab:read" in requested:
        choices += '<input type="hidden" name="scope" value="openlab:read">'
    return HTMLResponse(f"""<!doctype html><title>Authorize OpenLab MCP</title><main><h1>Authorize {html.escape(client.name)}</h1><p>Choose the access to grant:</p><form method=post action=/oauth/authorize/approve>{hidden}{choices}<button type=submit>Approve OpenLab access</button></form></main>""", headers={"Cache-Control": "no-store"})


@app.post("/oauth/authorize/approve", include_in_schema=False)
def approve_mcp_oauth(
    client_id: str = Form(),
    redirect_uri: str = Form(),
    code_challenge: str = Form(),
    requested_scope: str = Form(),
    scope: list[str] = Form(),
    state: str = Form(default=""),
    csrf: str = Form(),
    csrf_cookie: str | None = Cookie(default=None, alias="openlab_csrf"),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> Response:
    if not csrf_cookie or not compare_digest(csrf, csrf_cookie):
        raise HTTPException(status_code=403, detail="CSRF validation failed")
    lab = ensure_mcp_enabled(db, user)
    client = db.get(McpOAuthClient, client_id)
    scopes = sorted(set(scope))
    requested = set(requested_scope.split())
    if not client or redirect_uri not in client.redirect_uris or "openlab:read" not in scopes or not set(scopes).issubset(requested & MCP_SCOPES):
        raise HTTPException(status_code=422, detail="Invalid MCP OAuth approval")
    grant = McpGrant(lab_id=lab.id, user_id=user.id, client_id=client.id, scopes=scopes)
    db.add(grant)
    db.flush()
    raw_code = token_urlsafe(32)
    db.add(McpAuthorizationCode(
        grant_id=grant.id,
        code_hash=hash_credential(raw_code),
        code_challenge=code_challenge,
        redirect_uri=redirect_uri,
        expires_at=datetime.now(UTC).replace(microsecond=0) + timedelta(minutes=5),
    ))
    audit(db, user, "mcp.grant_authorized", "mcp_grant", grant.id, client_id=client.id, scopes=scopes)
    db.commit()
    destination = f"{redirect_uri}{'&' if '?' in redirect_uri else '?'}{urlencode({'code': raw_code, **({'state': state} if state else {})})}"
    return RedirectResponse(destination, status_code=302, headers={"Cache-Control": "no-store"})


@app.post("/oauth/token", include_in_schema=False)
def exchange_mcp_oauth_token(
    grant_type: str = Form(),
    client_id: str = Form(),
    code: str | None = Form(default=None),
    redirect_uri: str | None = Form(default=None),
    code_verifier: str | None = Form(default=None),
    refresh_token: str | None = Form(default=None),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    client = db.get(McpOAuthClient, client_id)
    if not client:
        raise HTTPException(status_code=401, detail="Unknown MCP client")
    if grant_type == "authorization_code":
        if not code or not redirect_uri or not code_verifier:
            raise HTTPException(status_code=422, detail="Authorization code, redirect URI, and PKCE verifier are required")
        authorization = db.scalar(select(McpAuthorizationCode).where(McpAuthorizationCode.code_hash == hash_credential(code)).with_for_update())
        if not authorization or authorization.consumed_at or authorization.expires_at <= datetime.now(UTC) or authorization.redirect_uri != redirect_uri:
            raise HTTPException(status_code=401, detail="Invalid or expired authorization code")
        verifier_digest = base64.urlsafe_b64encode(sha256(code_verifier.encode()).digest()).decode().rstrip("=")
        if not compare_digest(verifier_digest, authorization.code_challenge):
            raise HTTPException(status_code=401, detail="PKCE verification failed")
        grant = db.get(McpGrant, authorization.grant_id)
        if not grant or grant.client_id != client.id:
            raise HTTPException(status_code=401, detail="Invalid MCP authorization grant")
        access, refresh = token_urlsafe(32), token_urlsafe(40)
        grant.access_token_hash = hash_credential(access)
        grant.access_expires_at = datetime.now(UTC) + ACCESS_LIFETIME
        grant.refresh_token_hash = hash_credential(refresh)
        grant.refresh_expires_at = datetime.now(UTC) + timedelta(days=30)
        authorization.consumed_at = datetime.now(UTC)
        db.commit()
        return {"access_token": access, "token_type": "Bearer", "expires_in": int(ACCESS_LIFETIME.total_seconds()), "refresh_token": refresh, "scope": " ".join(grant.scopes)}
    if grant_type == "refresh_token":
        if not refresh_token:
            raise HTTPException(status_code=422, detail="Refresh token is required")
        grant = db.scalar(select(McpGrant).where(McpGrant.refresh_token_hash == hash_credential(refresh_token)).with_for_update())
        if not grant or grant.client_id != client.id:
            raise HTTPException(status_code=401, detail="Invalid MCP refresh token")
        access, refresh = rotate_refresh(db, grant)
        db.commit()
        return {"access_token": access, "token_type": "Bearer", "expires_in": int(ACCESS_LIFETIME.total_seconds()), "refresh_token": refresh, "scope": " ".join(grant.scopes)}
    raise HTTPException(status_code=422, detail="Unsupported MCP OAuth grant type")


@app.post("/oauth/revoke", status_code=204, include_in_schema=False)
def revoke_mcp_oauth_token(token: str = Form(), db: Session = Depends(get_db)) -> None:
    value = hash_credential(token)
    grant = db.scalar(select(McpGrant).where((McpGrant.access_token_hash == value) | (McpGrant.refresh_token_hash == value)))
    if grant:
        grant.revoked_at = datetime.now(UTC)
        db.commit()


def provider_out(config: ProviderConfig) -> dict[str, object]:
    return {
        "id": config.id,
        "provider": config.provider,
        "base_url": config.base_url,
        "model": config.model,
        "embedding_model": config.embedding_model,
        "enabled": config.enabled,
        "embeddings_enabled": config.embeddings_enabled,
        "has_api_key": config.secret_ciphertext is not None,
        "egress": "local" if is_local_endpoint(config.base_url) else "external",
    }


def deployment_environment() -> list[dict[str, object]]:
    def item(
        name: str,
        category: str,
        description: str,
        *,
        configured: bool | None = None,
        value: object | None = None,
        secret: bool = False,
        editable: bool = False,
        restart_required: bool = True,
    ) -> dict[str, object]:
        status = (
            "deployment_managed"
            if configured is None
            else "configured"
            if configured
            else "not_configured"
        )
        return {
            "name": name,
            "category": category,
            "status": status,
            "value": None if secret or value is None else str(value),
            "secret": secret,
            "editable": editable,
            "restart_required": restart_required,
            "description": description,
        }

    return [
        item("OPENLAB_KICAD_CLI", "application", "Optional worker-container fallback for KiCad ERC.", configured=bool(settings.kicad_cli), value=settings.kicad_cli, editable=True, restart_required=False),
        item("OPENLAB_PUBLIC_URL", "application", "Stable browser URL used by drawer QR labels.", configured=bool(settings.public_url), value=settings.public_url),
        item("OPENLAB_DATA_DIR", "infrastructure", "Persistent attachment storage inside the server and worker.", configured=True, value=settings.data_dir),
        item("SESSION_HOURS", "application", "Session lifetime for newly created login sessions.", configured=True, value=settings.session_hours),
        item("UPLOAD_MAX_BYTES", "application", "Maximum accepted Inbox attachment size in bytes.", configured=True, value=settings.upload_max_bytes),
        item("DATABASE_URL", "infrastructure", "PostgreSQL connection string.", configured=bool(settings.database_url), secret=True),
        item("OPENLAB_SECRET_KEY", "security", "Session-signing secret generated during bootstrap.", configured=settings.secret_key != "development-only-change-me", secret=True),
        item("OPENLAB_ENCRYPTION_KEY", "security", "Encryption key for stored provider credentials.", configured=bool(settings.encryption_key), secret=True),
        item("OPENLAB_SETUP_TOKEN", "security", "Optional stable first-owner setup token.", configured=bool(settings.setup_token), secret=True),
        item("OPENLAB_API_INTERNAL_URL", "infrastructure", "Next.js-to-API routing configured by the web deployment."),
        item("POSTGRES_DB / USER / PASSWORD", "infrastructure", "PostgreSQL container bootstrap values managed by Compose.", secret=True),
    ]


def kicad_settings_out(db: Session, lab: Lab, *, include_latest: bool = True) -> dict[str, object]:
    cli, source = effective_kicad_cli(lab, settings)
    status = "unknown"
    version = None
    error = None
    if include_latest:
        latest = db.scalar(
            select(Job)
            .where(Job.lab_id == lab.id, Job.kind == "system.kicad_check")
            .order_by(Job.created_at.desc())
        )
        if latest:
            if latest.status in {"queued", "running"}:
                status = latest.status
            elif latest.status == "completed" and latest.result:
                status = str(latest.result.get("status", "unavailable"))
                version = latest.result.get("version")
                error = latest.result.get("error")
            else:
                status = "unavailable"
                error = latest.last_error or "KiCad capability check did not complete"
    return {
        "cli_path": lab.kicad_cli,
        "effective_cli": cli,
        "source": source,
        "check_status": status,
        "version": version,
        "error": error,
    }


def settings_overview_out(db: Session, lab: Lab) -> dict[str, object]:
    return {
        "lab": lab,
        "kicad": kicad_settings_out(db, lab),
        "environment": deployment_environment(),
    }


@app.get("/api/v1/health")
def health(db: Session = Depends(get_db)) -> dict[str, object]:
    db.execute(select(1))
    return {"status": "healthy", "time": datetime.now(UTC).isoformat()}


@app.get("/api/v1/setup")
def setup_status(db: Session = Depends(get_db)) -> dict[str, bool]:
    return {"setup_required": db.scalar(select(func.count(User.id))) == 0}


@app.post("/api/v1/setup", response_model=LabOut, status_code=201)
def setup(payload: SetupRequest, response: Response, db: Session = Depends(get_db)) -> Lab:
    # Serialize competing first-owner requests across processes, not only within Uvicorn.
    db.execute(text("SELECT pg_advisory_xact_lock(762104913)"))
    if db.scalar(select(func.count(User.id))) != 0:
        raise HTTPException(status_code=409, detail="OpenLab is already configured")
    if not compare_digest(payload.token.encode(), bootstrap_token.encode()):
        raise HTTPException(status_code=403, detail="Invalid setup token")
    lab = Lab(name=payload.lab_name)
    owner = User(
        email=payload.email.lower(),
        display_name=payload.display_name,
        password_hash=hasher.hash(payload.password),
        is_owner=True,
    )
    db.add_all([lab, owner])
    db.flush()
    db.add(Membership(lab_id=lab.id, user_id=owner.id, role="owner"))
    consent = CommunicationConsent(
        user_id=owner.id,
        newsletter_opt_in=payload.newsletter_opt_in,
        notice_version=DISCLOSURE_VERSION,
        consented_at=datetime.now(UTC) if payload.newsletter_opt_in else None,
        status="pending_delivery" if payload.newsletter_opt_in else "not_subscribed",
    )
    db.add(consent)
    db.flush()
    if payload.newsletter_opt_in and consent.consented_at:
        queue_outbox(db, "subscribe", f"subscribe:{consent.id}:{int(consent.consented_at.timestamp())}", consent_id=consent.id)
    raw, csrf = create_session(db, owner)
    db.commit()
    set_session_cookies(response, raw, csrf)
    return lab


def set_session_cookies(response: Response, raw: str, csrf: str) -> None:
    response.headers["Cache-Control"] = "private, no-store"
    response.set_cookie(
        "openlab_session", raw, httponly=True, samesite="lax", max_age=settings.session_hours * 3600
    )
    response.set_cookie(
        "openlab_csrf", csrf, httponly=False, samesite="lax", max_age=settings.session_hours * 3600
    )


@app.post("/api/v1/session", response_model=UserOut)
def login(payload: LoginRequest, response: Response, db: Session = Depends(get_db)) -> User:
    user = db.scalar(select(User).where(User.email == payload.email.lower()))
    try:
        valid = bool(user and hasher.verify(user.password_hash, payload.password))
    except VerifyMismatchError:
        valid = False
    if not valid or not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    raw, csrf = create_session(db, user)
    db.commit()
    set_session_cookies(response, raw, csrf)
    return user


@app.delete("/api/v1/session", status_code=204, dependencies=[Depends(require_csrf)])
def logout(
    response: Response, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> Response:
    for item in db.scalars(select(SessionToken).where(SessionToken.user_id == user.id)).all():
        db.delete(item)
    db.commit()
    response.delete_cookie("openlab_session")
    response.delete_cookie("openlab_csrf")
    return response


@app.get("/api/v1/session", response_model=UserOut)
def me(user: User = Depends(current_user)) -> User:
    return user


@app.get("/api/v1/lab", response_model=LabOut)
def lab(user: User = Depends(current_user), db: Session = Depends(get_db)) -> Lab:
    value = db.get(Lab, lab_for_user(db, user))
    if value is None:
        raise HTTPException(status_code=404, detail="Lab not found")
    return value


@app.get("/api/v1/settings", response_model=SettingsOverviewOut)
def get_settings_overview(
    user: User = Depends(require_owner), db: Session = Depends(get_db)
) -> dict[str, object]:
    value = db.get(Lab, lab_for_user(db, user))
    if value is None:
        raise HTTPException(status_code=404, detail="Lab not found")
    return settings_overview_out(db, value)


def telemetry_settings_out(db: Session, user: User) -> TelemetrySettingsOut:
    try:
        state = ensure_state(db, settings)
    except ProviderError as exc:
        raise HTTPException(status_code=409, detail="Telemetry identity needs a valid encryption key") from exc
    consent = db.scalar(select(CommunicationConsent).where(CommunicationConsent.user_id == user.id))
    pending = db.scalar(select(func.count(TelemetryOutbox.id)).where(
        TelemetryOutbox.status == "queued"
    )) or 0
    return TelemetrySettingsOut(
        usage_enabled=state.usage_enabled,
        installation_id=state.installation_id,
        disclosure_version=state.disclosure_version,
        onboarding_seen_at=state.onboarding_seen_at,
        last_reported_day=state.last_reported_day,
        pending_delivery_count=int(pending),
        newsletter_status=consent.status if consent else "not_subscribed",
    )


@app.get("/api/v1/settings/telemetry", response_model=TelemetrySettingsOut)
def get_telemetry_settings(user: User = Depends(require_owner), db: Session = Depends(get_db)) -> TelemetrySettingsOut:
    return telemetry_settings_out(db, user)


@app.put("/api/v1/settings/telemetry", response_model=TelemetrySettingsOut,
         dependencies=[Depends(require_csrf)])
def save_telemetry_settings(payload: TelemetrySettingsInput, user: User = Depends(require_owner),
                            db: Session = Depends(get_db)) -> TelemetrySettingsOut:
    state = ensure_state(db, settings)
    state.usage_enabled = payload.usage_enabled
    if not payload.usage_enabled:
        state.last_queued_day = datetime.now(UTC) - timedelta(days=1)
        for item in db.scalars(select(TelemetryOutbox).where(
            TelemetryOutbox.kind == "activity", TelemetryOutbox.status == "queued"
        )).all():
            item.status = "cancelled"
    queue_preference(db, state)
    audit(db, user, "telemetry.preference_updated", "installation", state.id, usage_enabled=payload.usage_enabled)
    db.commit()
    return telemetry_settings_out(db, user)


@app.delete("/api/v1/settings/telemetry/history", status_code=202,
            dependencies=[Depends(require_csrf)])
def delete_telemetry_history(user: User = Depends(require_owner), db: Session = Depends(get_db)) -> dict[str, str]:
    state = ensure_state(db, settings)
    queue_outbox(db, "history_delete", f"history-delete:{state.installation_id}:{int(datetime.now(UTC).timestamp())}")
    audit(db, user, "telemetry.history_delete_requested", "installation", state.id)
    db.commit()
    return {"status": "queued"}


@app.put("/api/v1/settings/communications", response_model=TelemetrySettingsOut,
         dependencies=[Depends(require_csrf)])
def save_communications(payload: NewsletterSettingsInput, user: User = Depends(require_owner),
                        db: Session = Depends(get_db)) -> TelemetrySettingsOut:
    consent = db.scalar(select(CommunicationConsent).where(CommunicationConsent.user_id == user.id))
    if consent is None:
        consent = CommunicationConsent(user_id=user.id, notice_version=DISCLOSURE_VERSION)
        db.add(consent)
        db.flush()
    consent.newsletter_opt_in = payload.newsletter_opt_in
    consent.notice_version = DISCLOSURE_VERSION
    if payload.newsletter_opt_in:
        consent.consented_at = datetime.now(UTC)
        consent.status = "pending_delivery"
        queue_outbox(db, "subscribe", f"subscribe:{consent.id}:{int(consent.consented_at.timestamp())}", consent_id=consent.id)
    else:
        pending_subscribes = db.scalars(select(TelemetryOutbox).where(
            TelemetryOutbox.consent_id == consent.id, TelemetryOutbox.kind == "subscribe",
            TelemetryOutbox.status == "queued",
        )).all()
        for item in pending_subscribes:
            item.status = "cancelled"
        if consent.subscription_token_ciphertext:
            consent.status = "pending_unsubscribe"
            queue_outbox(db, "unsubscribe", f"unsubscribe:{consent.id}:{int(datetime.now(UTC).timestamp())}", consent_id=consent.id)
        else:
            consent.status = "not_subscribed"
    audit(db, user, "communication.newsletter_updated", "user", user.id, newsletter_opt_in=payload.newsletter_opt_in)
    db.commit()
    return telemetry_settings_out(db, user)


def owner_lab(db: Session, user: User) -> Lab:
    value = db.get(Lab, lab_for_user(db, user))
    if value is None:
        raise HTTPException(status_code=404, detail="Lab not found")
    return value


@app.get("/api/v1/readiness", response_model=ReadinessReport, operation_id="get_readiness")
def get_readiness(response: Response, user: User = Depends(require_owner),
                  db: Session = Depends(get_db)) -> ReadinessReport:
    response.headers["Cache-Control"] = "private, no-store"
    return application_readiness(db, owner_lab(db, user), settings)


@app.get("/api/v1/onboarding", response_model=OnboardingOut, operation_id="get_onboarding")
def get_onboarding(response: Response, user: User = Depends(require_owner),
                   db: Session = Depends(get_db)) -> OnboardingOut:
    value = owner_lab(db, user)
    response.headers["Cache-Control"] = "private, no-store"
    return OnboardingOut(completed_at=value.onboarding_completed_at,
        network=network_out(value, settings), readiness=application_readiness(db, value, settings))


@app.post("/api/v1/onboarding/complete", response_model=OnboardingOut,
          dependencies=[Depends(require_csrf)], operation_id="complete_onboarding")
def complete_onboarding(user: User = Depends(require_owner),
                        db: Session = Depends(get_db)) -> OnboardingOut:
    value = owner_lab(db, user)
    report = application_readiness(db, value, settings)
    if report.overall == "blocked":
        raise HTTPException(status_code=409, detail="Required checks have not passed; refresh readiness and fix the listed errors")
    value.onboarding_completed_at = datetime.now(UTC)
    state = db.get(TelemetryState, "installation")
    if state:
        state.onboarding_seen_at = datetime.now(UTC)
    audit(db, user, "installation.onboarding_completed", "lab", value.id)
    db.commit()
    return OnboardingOut(completed_at=value.onboarding_completed_at,
                         network=network_out(value, settings), readiness=report)


@app.get("/api/v1/settings/network", response_model=NetworkOut, operation_id="get_network_settings")
def get_network_settings(user: User = Depends(require_owner),
                         db: Session = Depends(get_db)) -> NetworkOut:
    return network_out(owner_lab(db, user), settings)


@app.put("/api/v1/settings/network", response_model=NetworkOut,
         dependencies=[Depends(require_csrf)], operation_id="save_network_settings")
def save_network_settings(payload: NetworkInput, request: Request,
                          user: User = Depends(require_owner), db: Session = Depends(get_db)) -> NetworkOut:
    value = owner_lab(db, user)
    value.public_url = payload.public_url
    # Verify through an authenticated browser at the actual origin, never an SSRF-prone
    # server-side fetch to an owner-supplied address. Re-saving elsewhere revokes verification.
    try:
        origin = normalize_public_url(request.headers.get("origin", ""))
    except ValueError:
        origin = ""
    value.public_url_verified_at = datetime.now(UTC) if origin == payload.public_url else None
    audit(db, user, "installation.network_updated", "lab", value.id,
          verified=value.public_url_verified_at is not None)
    db.commit()
    return network_out(value, settings)


@app.get("/api/v1/settings/installation", response_model=InstallationOverview,
         operation_id="get_installation_settings")
def get_installation_settings(response: Response, user: User = Depends(require_owner)) -> InstallationOverview:
    response.headers["Cache-Control"] = "private, no-store"
    return installation_overview(settings)


@app.get("/api/v1/settings/host-setup", response_model=HostSetupOut, operation_id="get_host_setup")
def get_host_setup(response: Response, user: User = Depends(require_owner)) -> HostSetupOut:
    response.headers["Cache-Control"] = "private, no-store"
    return host_setup_status(settings)


@app.post("/api/v1/settings/host-setup", response_model=HostSetupOperation, status_code=202,
          dependencies=[Depends(require_csrf)], operation_id="request_host_setup")
def request_host_setup(payload: HostSetupInput, user: User = Depends(require_owner),
                       db: Session = Depends(get_db)) -> HostSetupOperation:
    try:
        operation = queue_host_setup(settings, payload)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=409, detail="The host setup queue is not writable. Check installer diagnostics.") from exc
    audit(db, user, "installation.host_setup_requested", "lab", lab_for_user(db, user),
          setup_action=payload.action, request_id=operation.id)
    db.commit()
    return operation


@app.put("/api/v1/settings/installation", response_model=InstallationOverview,
         dependencies=[Depends(require_csrf)], operation_id="save_installation_settings")
def save_installation_settings(payload: InstallationPolicy, user: User = Depends(require_owner),
                               db: Session = Depends(get_db)) -> InstallationOverview:
    try:
        save_installation_policy(settings, payload)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=409, detail="Installer policy is not writable; run openlabctl doctor on the host") from exc
    audit(db, user, "installation.policy_updated", "lab", lab_for_user(db, user), **payload.model_dump())
    db.commit()
    return installation_overview(settings)


@app.put(
    "/api/v1/settings/lab", response_model=LabOut, dependencies=[Depends(require_csrf)]
)
def save_lab_settings(
    payload: LabSettingsInput,
    user: User = Depends(require_owner),
    db: Session = Depends(get_db),
) -> Lab:
    value = db.get(Lab, lab_for_user(db, user))
    if value is None:
        raise HTTPException(status_code=404, detail="Lab not found")
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Lab name cannot be blank")
    value.name = name
    value.units = payload.units
    audit(db, user, "settings.lab_updated", "lab", value.id, units=value.units)
    db.commit()
    db.refresh(value)
    return value


@app.put(
    "/api/v1/settings/kicad",
    response_model=KicadSettingsOut,
    dependencies=[Depends(require_csrf)],
)
def save_kicad_settings(
    payload: KicadSettingsInput,
    user: User = Depends(require_owner),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    value = db.get(Lab, lab_for_user(db, user))
    if value is None:
        raise HTTPException(status_code=404, detail="Lab not found")
    try:
        value.kicad_cli = normalize_kicad_cli(payload.cli_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    audit(
        db,
        user,
        "settings.kicad_updated",
        "lab",
        value.id,
        configured=value.kicad_cli is not None,
    )
    db.commit()
    db.refresh(value)
    return kicad_settings_out(db, value, include_latest=False)


@app.post(
    "/api/v1/settings/kicad/check",
    response_model=JobOut,
    status_code=202,
    dependencies=[Depends(require_csrf)],
)
def queue_kicad_check(
    user: User = Depends(require_owner), db: Session = Depends(get_db)
) -> Job:
    lab_id = lab_for_user(db, user)
    pending = db.scalar(
        select(Job).where(
            Job.lab_id == lab_id,
            Job.kind == "system.kicad_check",
            Job.status.in_(["queued", "running"]),
        )
    )
    if pending:
        return pending
    job = Job(lab_id=lab_id, kind="system.kicad_check", payload={})
    db.add(job)
    db.flush()
    audit(db, user, "settings.kicad_check_queued", "lab", lab_id, job_id=job.id)
    db.commit()
    return job


@app.get("/api/v1/ai/provider", response_model=ProviderConfigOut | None)
def get_provider_config(
    user: User = Depends(require_owner), db: Session = Depends(get_db)
) -> dict[str, object] | None:
    config = db.scalar(
        select(ProviderConfig)
        .where(ProviderConfig.lab_id == lab_for_user(db, user))
        .order_by(ProviderConfig.updated_at.desc())
    )
    return provider_out(config) if config else None


@app.put(
    "/api/v1/ai/provider", response_model=ProviderConfigOut, dependencies=[Depends(require_csrf)]
)
def save_provider_config(
    payload: ProviderConfigInput,
    user: User = Depends(require_owner),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    if not payload.base_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=422, detail="Provider endpoint must use http or https")
    if payload.embeddings_enabled and not payload.embedding_model:
        raise HTTPException(
            status_code=422, detail="Choose an embedding model before enabling semantic retrieval"
        )
    lab_id = lab_for_user(db, user)
    config = db.scalar(
        select(ProviderConfig)
        .where(ProviderConfig.lab_id == lab_id)
        .order_by(ProviderConfig.updated_at.desc())
    )
    if not config:
        config = ProviderConfig(
            lab_id=lab_id,
            provider="openai-compatible",
            base_url=payload.base_url.rstrip("/"),
            model=payload.model,
            embedding_model=payload.embedding_model,
            capabilities={"chat": True, "vision": "model-dependent", "audio": "endpoint-dependent"},
            enabled=payload.enabled,
            embeddings_enabled=payload.embeddings_enabled,
        )
        db.add(config)
    else:
        config.base_url = payload.base_url.rstrip("/")
        config.model = payload.model
        config.embedding_model = payload.embedding_model
        config.enabled = payload.enabled
        config.embeddings_enabled = payload.embeddings_enabled
    if payload.api_key is not None:
        try:
            config.secret_ciphertext = (
                encrypt_secret(payload.api_key, settings.encryption_key)
                if payload.api_key
                else None
            )
        except ProviderError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    db.flush()
    if config.embeddings_enabled and config.embedding_model:
        for thing_id in db.scalars(
            select(Thing.id).where(Thing.lab_id == lab_id, Thing.archived_at.is_(None))
        ).all():
            queue_thing_embedding(db, lab_id, thing_id)
    audit(db, user, "ai.provider_configured", "provider_config", config.id, enabled=config.enabled)
    db.commit()
    return provider_out(config)


@app.get("/api/v1/ai/provider/models", response_model=ProviderModelsOut)
def provider_models(
    user: User = Depends(require_owner), db: Session = Depends(get_db)
) -> dict[str, object]:
    try:
        config = db.scalar(
            select(ProviderConfig)
            .where(ProviderConfig.lab_id == lab_for_user(db, user))
            .order_by(ProviderConfig.updated_at.desc())
        )
        if not config:
            raise HTTPException(status_code=409, detail="Save an AI provider first")
        provider = OpenAICompatibleProvider(
            base_url=config.base_url,
            model=config.model,
            api_key=decrypt_secret(config.secret_ciphertext, settings.encryption_key),
        )
        models = provider.list_models()
        value = owner_lab(db, user)
        value.integration_checks = {**(value.integration_checks or {}), "ai": {
            "fingerprint": provider_fingerprint(config), "checked_at": datetime.now(UTC).isoformat(),
        }}
        db.commit()
        return {
            "models": models,
            "egress": "local" if is_local_endpoint(config.base_url) else "external",
        }
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail="AI endpoint check failed. Verify the endpoint, credentials, network access, and model permissions.") from exc


@app.get("/api/v1/things", response_model=list[ThingOut])
def list_things(
    q: str | None = None, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> list[Thing]:
    query = (
        select(Thing)
        .where(Thing.lab_id == lab_for_user(db, user), Thing.archived_at.is_(None))
        .order_by(Thing.name)
    )
    if q:
        term = f"%{q}%"
        query = (
            query.outerjoin(ThingAlias)
            .where(Thing.name.ilike(term) | Thing.mpn.ilike(term) | ThingAlias.value.ilike(term))
            .distinct()
        )
    return list(db.scalars(query).all())


@app.post(
    "/api/v1/things", response_model=ThingOut, status_code=201, dependencies=[Depends(require_csrf)]
)
def add_thing(
    payload: ThingCreate, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> Thing:
    item = create_thing(
        db,
        user,
        name=payload.name,
        category=payload.category,
        manufacturer=payload.manufacturer,
        mpn=payload.mpn,
        metadata=payload.metadata,
        aliases=payload.aliases,
    )
    queue_thing_embedding(db, item.lab_id, item.id)
    db.commit()
    return item


@app.get("/api/v1/things/{thing_id}", response_model=ThingOut)
def get_thing(
    thing_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> Thing:
    return get_lab_thing(db, user, thing_id)


@app.patch(
    "/api/v1/things/{thing_id}", response_model=ThingOut, dependencies=[Depends(require_csrf)]
)
def patch_thing(
    thing_id: str,
    payload: ThingPatch,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> Thing:
    item = get_lab_thing(db, user, thing_id)
    if item.revision != payload.revision:
        raise HTTPException(status_code=409, detail="Thing was updated elsewhere; reload and retry")
    for field, value in payload.model_dump(exclude={"revision"}, exclude_unset=True).items():
        setattr(item, "metadata_json" if field == "metadata" else field, value)
    item.revision += 1
    queue_thing_embedding(db, item.lab_id, item.id)
    audit(db, user, "thing.updated", "thing", item.id)
    db.commit()
    return item


@app.delete("/api/v1/things/{thing_id}", status_code=204, dependencies=[Depends(require_csrf)])
def archive_thing(
    thing_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> None:
    item = get_lab_thing(db, user, thing_id)
    stock_total = db.scalar(
        select(func.coalesce(func.sum(StockBalance.quantity), 0)).where(
            StockBalance.thing_id == item.id,
            StockBalance.quantity > 0,
        )
    )
    active_allocations = db.scalar(
        select(func.count(Allocation.id)).where(
            Allocation.thing_id == item.id,
            Allocation.state.in_(["reserved", "in_use", "recoverable"]),
        )
    )
    if stock_total or active_allocations:
        raise HTTPException(
            status_code=409,
            detail="Move or consume all stock and clear active allocations before archiving",
        )
    item.archived_at = datetime.now(UTC)
    item.revision += 1
    audit(db, user, "thing.archived", "thing", item.id)
    db.commit()


def location_out(db: Session, location: Location) -> dict[str, object]:
    thing_count, total_quantity = db.execute(
        select(
            func.count(func.distinct(StockBalance.thing_id)),
            func.coalesce(func.sum(StockBalance.quantity), 0),
        )
        .select_from(StockBalance)
        .join(Thing, Thing.id == StockBalance.thing_id)
        .where(
            StockBalance.location_id == location.id,
            StockBalance.quantity > 0,
            Thing.archived_at.is_(None),
        )
    ).one()
    return {
        "id": location.id,
        "name": location.name,
        "parent_id": location.parent_id,
        "public_code": location.public_code,
        "revision": location.revision,
        "thing_count": thing_count,
        "total_quantity": total_quantity,
    }


@app.get("/api/v1/locations", response_model=list[LocationOut])
def list_locations(
    user: User = Depends(current_user), db: Session = Depends(get_db)
) -> list[dict[str, object]]:
    locations = list(
        db.scalars(
            select(Location)
            .where(Location.lab_id == lab_for_user(db, user), Location.archived_at.is_(None))
            .order_by(Location.name)
        ).all()
    )
    return [location_out(db, location) for location in locations]


@app.post(
    "/api/v1/locations",
    response_model=LocationOut,
    status_code=201,
    dependencies=[Depends(require_csrf)],
)
def add_location(
    payload: LocationCreate, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> Location:
    item = Location(lab_id=lab_for_user(db, user), name=payload.name, parent_id=None)
    db.add(item)
    db.flush()
    audit(db, user, "location.created", "location", item.id)
    db.commit()
    return item


@app.get("/api/v1/locations/code/{public_code}", response_model=LocationOut)
def lookup_location(
    public_code: str, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> dict[str, object]:
    item = db.scalar(
        select(Location).where(
            Location.public_code == public_code, Location.lab_id == lab_for_user(db, user)
        )
    )
    if not item:
        raise HTTPException(status_code=404, detail="Location not found")
    return location_out(db, item)


@app.get("/api/v1/locations/{location_id}", response_model=LocationOut)
def get_location(
    location_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> dict[str, object]:
    return location_out(db, get_lab_location(db, user, location_id))


def qr_target(location: Location, request: Request, base_url: str | None,
              public_url: str | None = None) -> str:
    try:
        return location_capture_url(
            location.public_code,
            configured_url=public_url or settings.public_url,
            request_url=base_url or str(request.base_url),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/v1/locations/{location_id}/qr-info", response_model=LocationQRInfo)
def location_qr_info(
    location_id: str,
    request: Request,
    base_url: str | None = None,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    location = get_lab_location(db, user, location_id)
    value = db.get(Lab, location.lab_id)
    public_url = value.public_url if value else None
    target_url = qr_target(location, request, base_url, public_url)
    query = f"?{urlencode({'base_url': base_url})}" if base_url and not (public_url or settings.public_url) else ""
    return {
        "target_url": target_url,
        "svg_url": f"/api/v1/locations/{location.id}/qr.svg{query}",
    }


@app.get("/api/v1/locations/{location_id}/qr.svg", response_class=FastAPIResponse)
def location_qr(
    location_id: str,
    request: Request,
    base_url: str | None = None,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> FastAPIResponse:
    location = get_lab_location(db, user, location_id)
    output = BytesIO()
    value = db.get(Lab, location.lab_id)
    qrcode.make(qr_target(location, request, base_url, value.public_url if value else None), image_factory=SvgPathImage, border=2).save(
        output
    )
    return FastAPIResponse(
        content=output.getvalue(),
        media_type="image/svg+xml",
        headers={"Content-Disposition": f'inline; filename="openlab-{location.public_code}.svg"'},
    )


@app.get("/api/v1/inventory/balances", response_model=list[BalanceOut])
def balances(
    location_id: str | None = None,
    thing_id: str | None = None,
    user: User = Depends(current_user), db: Session = Depends(get_db)
) -> list[dict[str, object]]:
    lab_id = lab_for_user(db, user)
    if location_id:
        get_lab_location(db, user, location_id)
    if thing_id:
        get_lab_thing(db, user, thing_id)
    query = (
        select(StockBalance, Thing, Location)
        .join(Thing, Thing.id == StockBalance.thing_id)
        .join(Location, Location.id == StockBalance.location_id)
        .where(
            Thing.lab_id == lab_id,
            Thing.archived_at.is_(None),
            Location.lab_id == lab_id,
            Location.archived_at.is_(None),
            StockBalance.quantity > 0,
        )
        .order_by(Thing.name, Location.name)
    )
    if location_id:
        query = query.where(StockBalance.location_id == location_id)
    if thing_id:
        query = query.where(StockBalance.thing_id == thing_id)
    return [
        {
            "thing_id": balance.thing_id,
            "location_id": balance.location_id,
            "quantity": balance.quantity,
            "revision": balance.revision,
            "thing_name": thing.name,
            "thing_category": thing.category,
            "thing_manufacturer": thing.manufacturer,
            "thing_mpn": thing.mpn,
            "location_name": location.name,
        }
        for balance, thing, location in db.execute(query).all()
    ]


@app.get("/api/v1/inventory/movements", response_model=list[StockMovementDetailOut])
def movements(
    location_id: str | None = None,
    thing_id: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> list[dict[str, object]]:
    lab_id = lab_for_user(db, user)
    if location_id:
        get_lab_location(db, user, location_id)
    if thing_id:
        get_lab_thing(db, user, thing_id)
    from_location = aliased(Location)
    to_location = aliased(Location)
    query = (
        select(StockMovement, Thing, from_location, to_location)
        .join(Thing, Thing.id == StockMovement.thing_id)
        .outerjoin(from_location, from_location.id == StockMovement.from_location_id)
        .outerjoin(to_location, to_location.id == StockMovement.to_location_id)
        .where(StockMovement.lab_id == lab_id, Thing.lab_id == lab_id)
        .order_by(StockMovement.created_at.desc())
        .limit(limit)
    )
    if location_id:
        query = query.where(
            or_(
                StockMovement.from_location_id == location_id,
                StockMovement.to_location_id == location_id,
            )
        )
    if thing_id:
        query = query.where(StockMovement.thing_id == thing_id)
    return [
        {
            "id": movement.id,
            "thing_id": movement.thing_id,
            "thing_name": thing.name,
            "from_location_id": movement.from_location_id,
            "from_location_name": source.name if source else None,
            "to_location_id": movement.to_location_id,
            "to_location_name": destination.name if destination else None,
            "quantity": movement.quantity,
            "movement_type": movement.movement_type,
            "note": movement.note,
            "created_at": movement.created_at,
        }
        for movement, thing, source, destination in db.execute(query).all()
    ]


def stock_endpoint(kind: str) -> Callable[..., StockMovement]:
    def endpoint(
        payload: StockMutation,
        idempotency_key: str = Depends(require_idempotency),
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> StockMovement:
        if kind == "receive" and not payload.to_location_id:
            raise HTTPException(status_code=422, detail="Receiving stock needs a destination")
        if kind == "move" and (not payload.from_location_id or not payload.to_location_id):
            raise HTTPException(
                status_code=422, detail="Moving stock needs a source and destination"
            )
        if kind == "move" and payload.from_location_id == payload.to_location_id:
            raise HTTPException(status_code=422, detail="Source and destination must be different")
        if kind == "consume" and not payload.from_location_id:
            raise HTTPException(status_code=422, detail="Consuming stock needs a source")
        item = apply_movement(
            db,
            user,
            thing_id=payload.thing_id,
            quantity=payload.quantity,
            movement_type=kind,
            idempotency_key=idempotency_key,
            from_location_id=payload.from_location_id,
            to_location_id=payload.to_location_id,
            note=payload.note,
        )
        db.commit()
        return item

    return endpoint


@app.post(
    "/api/v1/inventory/adjust",
    response_model=StockMovementOut,
    status_code=201,
    dependencies=[Depends(require_csrf)],
)
def adjust_stock(
    payload: StockAdjustment,
    idempotency_key: str = Depends(require_idempotency),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> StockMovement:
    movement = adjust_inventory(
        db,
        user,
        thing_id=payload.thing_id,
        location_id=payload.location_id,
        counted_quantity=payload.counted_quantity,
        revision=payload.revision,
        note=payload.note,
        idempotency_key=idempotency_key,
    )
    db.commit()
    return movement


app.post(
    "/api/v1/inventory/receive",
    response_model=StockMovementOut,
    status_code=201,
    dependencies=[Depends(require_csrf)],
)(stock_endpoint("receive"))
app.post(
    "/api/v1/inventory/move",
    response_model=StockMovementOut,
    status_code=201,
    dependencies=[Depends(require_csrf)],
)(stock_endpoint("move"))
app.post(
    "/api/v1/inventory/consume",
    response_model=StockMovementOut,
    status_code=201,
    dependencies=[Depends(require_csrf)],
)(stock_endpoint("consume"))


@app.get("/api/v1/inbox", response_model=list[InboxOut])
def list_inbox(
    user: User = Depends(current_user), db: Session = Depends(get_db)
) -> list[InboxItem]:
    return list(
        db.scalars(
            select(InboxItem)
            .where(InboxItem.lab_id == lab_for_user(db, user))
            .order_by(InboxItem.created_at.desc())
        ).all()
    )


@app.get("/api/v1/inbox/{inbox_id}/candidates", response_model=list[InboxCandidateOut])
def list_inbox_candidates(
    inbox_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> list[InboxCandidate]:
    inbox = db.scalar(
        select(InboxItem).where(
            InboxItem.id == inbox_id, InboxItem.lab_id == lab_for_user(db, user)
        )
    )
    if not inbox:
        raise HTTPException(status_code=404, detail="Inbox item not found")
    return list(
        db.scalars(select(InboxCandidate).where(InboxCandidate.inbox_item_id == inbox.id)).all()
    )


@app.post(
    "/api/v1/inbox", response_model=InboxOut, status_code=201, dependencies=[Depends(require_csrf)]
)
def capture_inbox(
    payload: InboxCapture, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> InboxItem:
    item = InboxItem(
        lab_id=lab_for_user(db, user),
        created_by=user.id,
        input_type=payload.input_type,
        text=payload.text,
        status="captured",
    )
    db.add(item)
    db.flush()
    audit(db, user, "inbox.captured", "inbox_item", item.id, input_type=payload.input_type)
    db.commit()
    return item


def project_detail(db: Session, project: Project) -> dict[str, object]:
    return {
        "id": project.id,
        "name": project.name,
        "description": project.description,
        "status": project.status,
        "revision": project.revision,
        "created_at": project.created_at,
        "updated_at": project.updated_at,
        "design_json": project.design_json,
        "requirements": list(
            db.scalars(select(Requirement).where(Requirement.project_id == project.id)).all()
        ),
        "allocations": list(
            db.scalars(select(Allocation).where(Allocation.project_id == project.id)).all()
        ),
    }


@app.get("/api/v1/projects/{project_id}", response_model=ProjectDetailOut)
def get_project(
    project_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> dict[str, object]:
    project = db.scalar(
        select(Project).where(Project.id == project_id, Project.lab_id == lab_for_user(db, user))
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project_detail(db, project)


@app.post(
    "/api/v1/inbox/{inbox_id}/attachments", status_code=201, dependencies=[Depends(require_csrf)]
)
async def upload_inbox_attachment(
    inbox_id: str,
    upload: UploadFile,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    item = db.scalar(
        select(InboxItem).where(
            InboxItem.id == inbox_id, InboxItem.lab_id == lab_for_user(db, user)
        )
    )
    if not item:
        raise HTTPException(status_code=404, detail="Inbox item not found")
    attachment = await save_upload(db, user, inbox_id, upload)
    db.commit()
    return {"id": attachment.id, "sha256": attachment.sha256, "size_bytes": attachment.size_bytes}


@app.post(
    "/api/v1/inbox/{inbox_id}/process",
    response_model=InboxOut,
    status_code=202,
    dependencies=[Depends(require_csrf)],
)
def process_inbox(
    inbox_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> InboxItem:
    item = db.scalar(
        select(InboxItem).where(
            InboxItem.id == inbox_id, InboxItem.lab_id == lab_for_user(db, user)
        )
    )
    if not item:
        raise HTTPException(status_code=404, detail="Inbox item not found")
    if item.status in {"committed", "cancelled"}:
        raise HTTPException(status_code=409, detail="This Inbox item cannot be processed again")
    item.status = "queued"
    item.error = None
    job = Job(lab_id=item.lab_id, kind="inbox.process", payload={"inbox_id": item.id})
    db.add(job)
    audit(db, user, "inbox.queued", "inbox_item", item.id, job_id=job.id)
    db.commit()
    return item


def inbox_candidate_for_user(
    db: Session, user: User, inbox_id: str, candidate_id: str
) -> tuple[InboxItem, InboxCandidate]:
    item = db.scalar(
        select(InboxItem).where(
            InboxItem.id == inbox_id, InboxItem.lab_id == lab_for_user(db, user)
        )
    )
    if not item:
        raise HTTPException(status_code=404, detail="Inbox item not found")
    candidate = db.scalar(
        select(InboxCandidate).where(
            InboxCandidate.id == candidate_id, InboxCandidate.inbox_item_id == item.id
        )
    )
    if not candidate:
        raise HTTPException(status_code=404, detail="Inbox candidate not found")
    return item, candidate


def update_candidate_proposal(
    candidate: InboxCandidate, payload: InboxCandidatePatch
) -> InboxCandidate:
    if candidate.status != "proposed":
        raise HTTPException(
            status_code=409,
            detail="Only proposed candidates can be edited; edit the inventory item instead",
        )
    if payload.name is not None:
        candidate.name = payload.name
    if payload.quantity is not None:
        candidate.quantity = payload.quantity
    if payload.category is not None:
        candidate.category = payload.category
    if "description" in payload.model_fields_set:
        provenance = dict(candidate.provenance)
        if payload.description is None:
            provenance.pop("description", None)
        else:
            provenance["description"] = payload.description
        candidate.provenance = provenance
    candidate.revision = (candidate.revision or 0) + 1
    return candidate


@app.patch(
    "/api/v1/inbox/{inbox_id}/candidates/{candidate_id}",
    response_model=InboxCandidateOut,
    dependencies=[Depends(require_csrf)],
)
def patch_inbox_candidate(
    inbox_id: str,
    candidate_id: str,
    payload: InboxCandidatePatch,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> InboxCandidate:
    _item, candidate = inbox_candidate_for_user(db, user, inbox_id, candidate_id)
    update_candidate_proposal(candidate, payload)
    audit(db, user, "inbox.candidate_updated", "inbox_candidate", candidate.id)
    db.commit()
    return candidate


def confirm_candidate_identity(
    db: Session,
    user: User,
    item: InboxItem,
    candidate: InboxCandidate,
    payload: InboxCandidateConfirm,
) -> InboxCandidate:
    if candidate.status == "ignored":
        raise HTTPException(status_code=409, detail="Ignored candidates cannot be confirmed")
    supplied_values = any(
        value is not None
        for value in (
            payload.name,
            payload.description,
            payload.quantity,
            payload.category,
            payload.existing_thing_id,
        )
    )
    if candidate.status in {"confirmed", "received"}:
        if not supplied_values:
            return candidate
        raise HTTPException(
            status_code=409,
            detail="This candidate is already confirmed; edit the inventory item instead",
        )
    if (
        candidate.status == "proposed"
        and candidate.identity_confidence == "unresolved"
        and not candidate.product_url
    ):
        raise HTTPException(
            status_code=409,
            detail="Unable to parse this item; provide a product link before confirmation",
        )
    if payload.name is not None:
        candidate.name = payload.name
    if payload.quantity is not None:
        candidate.quantity = payload.quantity
    if payload.category is not None:
        candidate.category = payload.category
    provenance = dict(candidate.provenance)
    if "description" in payload.model_fields_set:
        if payload.description is None:
            provenance.pop("description", None)
        else:
            provenance["description"] = payload.description
        candidate.provenance = provenance
    description = provenance.get("description")
    thing = (
        get_lab_thing(db, user, payload.existing_thing_id)
        if payload.existing_thing_id
        else create_thing(
            db,
            user,
            name=candidate.name,
            category=candidate.category,
            manufacturer=None,
            mpn=None,
            metadata={"description": description} if isinstance(description, str) else {},
            aliases=[],
        )
    )
    candidate.thing_id = thing.id
    candidate.status = "confirmed"
    queue_thing_enrichment(db, thing.lab_id, thing.id)
    queue_thing_embedding(db, thing.lab_id, thing.id)
    refresh_inbox_status(db, item)
    audit(db, user, "inbox.candidate_confirmed", "inbox_candidate", candidate.id, thing_id=thing.id)
    return candidate


@app.post(
    "/api/v1/inbox/{inbox_id}/candidates/{candidate_id}/confirm",
    response_model=InboxCandidateOut,
    dependencies=[Depends(require_csrf)],
)
def confirm_inbox_candidate(
    inbox_id: str,
    candidate_id: str,
    payload: InboxCandidateConfirm,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> InboxCandidate:
    item, candidate = inbox_candidate_for_user(db, user, inbox_id, candidate_id)
    result = confirm_candidate_identity(db, user, item, candidate, payload)
    db.commit()
    return result


@app.post(
    "/api/v1/inbox/{inbox_id}/confirm-batch",
    response_model=list[InboxCandidateOut],
    dependencies=[Depends(require_csrf)],
)
def confirm_inbox_batch(
    inbox_id: str,
    payload: InboxCandidateBatchConfirm,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> list[InboxCandidate]:
    item = db.scalar(
        select(InboxItem).where(
            InboxItem.id == inbox_id, InboxItem.lab_id == lab_for_user(db, user)
        )
    )
    if not item:
        raise HTTPException(status_code=404, detail="Inbox item not found")
    candidates = list(
        db.scalars(
            select(InboxCandidate).where(
                InboxCandidate.inbox_item_id == item.id,
                InboxCandidate.id.in_(payload.candidate_ids),
            )
        ).all()
    )
    if len(candidates) != len(set(payload.candidate_ids)):
        raise HTTPException(status_code=404, detail="One or more Inbox candidates were not found")
    for candidate in candidates:
        confirm_candidate_identity(db, user, item, candidate, InboxCandidateConfirm())
    db.commit()
    return candidates


@app.post(
    "/api/v1/inbox/{inbox_id}/candidates/{candidate_id}/ignore",
    response_model=InboxCandidateOut,
    dependencies=[Depends(require_csrf)],
)
def ignore_inbox_candidate(
    inbox_id: str,
    candidate_id: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> InboxCandidate:
    item, candidate = inbox_candidate_for_user(db, user, inbox_id, candidate_id)
    if candidate.status == "received":
        raise HTTPException(status_code=409, detail="Received candidates cannot be ignored")
    candidate.status = "ignored"
    refresh_inbox_status(db, item)
    audit(db, user, "inbox.candidate_ignored", "inbox_candidate", candidate.id)
    db.commit()
    return candidate


@app.delete(
    "/api/v1/inbox/{inbox_id}/candidates/{candidate_id}",
    status_code=204,
    dependencies=[Depends(require_csrf)],
)
def delete_inbox_candidate(
    inbox_id: str,
    candidate_id: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> Response:
    item, candidate = inbox_candidate_for_user(db, user, inbox_id, candidate_id)
    if candidate.status != "proposed":
        raise HTTPException(
            status_code=409,
            detail="Only proposed candidates can be removed; edit the inventory item instead",
        )
    candidate.status = "ignored"
    refresh_inbox_status(db, item)
    audit(db, user, "inbox.candidate_removed", "inbox_candidate", candidate.id)
    db.commit()
    return Response(status_code=204)


@app.post(
    "/api/v1/inbox/{inbox_id}/candidates/{candidate_id}/receive",
    response_model=StockMovementOut,
    status_code=201,
    dependencies=[Depends(require_csrf)],
)
def receive_inbox_candidate(
    inbox_id: str,
    candidate_id: str,
    payload: InboxCandidateReceive,
    idempotency_key: str = Depends(require_idempotency),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> StockMovement:
    item, candidate = inbox_candidate_for_user(db, user, inbox_id, candidate_id)
    if candidate.status not in {"confirmed", "received"} or not candidate.thing_id:
        raise HTTPException(
            status_code=409, detail="Confirm this candidate's identity before receiving it"
        )
    quantity = payload.quantity or candidate.quantity
    movement = apply_movement(
        db,
        user,
        thing_id=candidate.thing_id,
        quantity=quantity,
        movement_type="receive",
        idempotency_key=idempotency_key,
        to_location_id=payload.location_id,
        note=f"Inbox {item.id} candidate {candidate.id}",
    )
    candidate.status = "received"
    refresh_inbox_status(db, item)
    audit(
        db,
        user,
        "inbox.candidate_received",
        "inbox_candidate",
        candidate.id,
        thing_id=candidate.thing_id,
    )
    db.commit()
    return movement


@app.post(
    "/api/v1/inbox/{inbox_id}/candidates/{candidate_id}/enrich-url",
    response_model=InboxCandidateOut,
    dependencies=[Depends(require_csrf)],
)
def enrich_inbox_candidate_url(
    inbox_id: str,
    candidate_id: str,
    payload: InboxEnrichURL,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> InboxCandidate:
    item, candidate = inbox_candidate_for_user(db, user, inbox_id, candidate_id)
    if candidate.status != "proposed":
        raise HTTPException(status_code=409, detail="This candidate cannot be enriched")
    result = enrich_product_url(payload.url)
    candidate.product_url = str(result["normalized_url"])
    provenance = dict(candidate.provenance)
    provenance["product_link"] = result
    candidate.provenance = provenance
    proposal = result.get("proposal", {})
    if isinstance(proposal, dict):
        page_text = "\n".join(
            str(value) for value in (proposal.get("name"), proposal.get("description")) if value
        )
        provider, config = active_provider(db, item.lab_id)
        if provider and page_text:
            try:
                raw = provider.extract_inbox(page_text)
                values = raw.get("candidates", [])
                if not isinstance(values, list) or not values:
                    raise ValueError("candidates is empty")
                normalized = InboxCandidateInput.model_validate(values[0])
            except (ProviderError, ValueError) as first_error:
                try:
                    repaired = provider.extract_inbox(
                        page_text, repair_error=str(first_error)[:1200]
                    )
                    values = repaired.get("candidates", [])
                    if not isinstance(values, list) or not values:
                        raise ValueError("candidates is empty")
                    normalized = InboxCandidateInput.model_validate(values[0])
                except (ProviderError, ValueError):
                    normalized = None
            if normalized:
                candidate.name = normalized.name
                candidate.category = normalized.category
                candidate.identity_confidence = normalized.identity_confidence
                provenance["description"] = normalized.description
                provenance["observations"] = normalized.observations
                provenance["product_link_model"] = config.model if config else None
                candidate.provenance = provenance
    audit(db, user, "inbox.candidate_enriched", "inbox_candidate", candidate.id)
    db.commit()
    return candidate


@app.get("/api/v1/projects", response_model=list[ProjectOut])
def list_projects(
    user: User = Depends(current_user), db: Session = Depends(get_db)
) -> list[Project]:
    return list(
        db.scalars(
            select(Project).where(Project.lab_id == lab_for_user(db, user)).order_by(Project.name)
        ).all()
    )


@app.post(
    "/api/v1/projects",
    response_model=ProjectOut,
    status_code=201,
    dependencies=[Depends(require_csrf)],
)
def create_project(
    payload: ProjectCreate, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> Project:
    item = Project(
        lab_id=lab_for_user(db, user),
        name=payload.name,
        description=payload.description,
        status="pending",
    )
    db.add(item)
    db.flush()
    audit(db, user, "project.created", "project", item.id)
    db.commit()
    return item


@app.patch(
    "/api/v1/projects/{project_id}",
    response_model=ProjectOut,
    dependencies=[Depends(require_csrf)],
)
def update_project_status(
    project_id: str,
    payload: ProjectUpdate,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> Project:
    project = db.scalar(
        select(Project)
        .where(Project.id == project_id, Project.lab_id == lab_for_user(db, user))
        .with_for_update(of=Project)
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    changed_fields: list[str] = []
    if payload.name is not None:
        name = payload.name.strip()
        if not name:
            raise HTTPException(status_code=422, detail="Project name cannot be blank")
        project.name = name
        changed_fields.append("name")
    if "description" in payload.model_fields_set:
        project.description = (
            payload.description.strip() if payload.description and payload.description.strip() else None
        )
        changed_fields.append("description")
    if payload.status is not None:
        project.status = payload.status
        changed_fields.append("status")
    if not changed_fields:
        raise HTTPException(status_code=400, detail="At least one project field is required")
    project.revision += 1
    if changed_fields == ["status"]:
        audit(db, user, "project.status_updated", "project", project.id, status=payload.status)
    else:
        audit(db, user, "project.updated", "project", project.id, fields=changed_fields)
    db.commit()
    return project


@app.post(
    "/api/v1/projects/{project_id}/requirements",
    status_code=201,
    dependencies=[Depends(require_csrf)],
)
def add_requirement(
    project_id: str,
    payload: RequirementCreate,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    project = db.scalar(
        select(Project)
        .where(Project.id == project_id, Project.lab_id == lab_for_user(db, user))
        .with_for_update(of=Project)
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    item = Requirement(project_id=project.id, **payload.model_dump())
    db.add(item)
    project.revision += 1
    db.commit()
    return {"id": item.id}


@app.post(
    "/api/v1/projects/{project_id}/allocations",
    status_code=201,
    dependencies=[Depends(require_csrf)],
)
def allocate_project_stock(
    project_id: str,
    payload: AllocationCreate,
    idempotency_key: str = Depends(require_idempotency),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    project = db.scalar(
        select(Project)
        .where(Project.id == project_id, Project.lab_id == lab_for_user(db, user))
        .with_for_update(of=Project)
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    existing = db.scalar(
        select(Allocation).where(
            Allocation.project_id == project.id, Allocation.idempotency_key == idempotency_key
        )
    )
    if existing:
        return {"id": existing.id, "state": existing.state}
    get_lab_thing(db, user, payload.thing_id)
    get_lab_location(db, user, payload.location_id)
    if available_quantity(db, payload.thing_id, lock=True) < payload.quantity:
        raise HTTPException(status_code=409, detail="Insufficient available stock")
    allocation = Allocation(
        project_id=project.id, idempotency_key=idempotency_key, **payload.model_dump()
    )
    db.add(allocation)
    db.flush()
    if payload.state in {"in_use", "recoverable"}:
        apply_movement(
            db,
            user,
            thing_id=payload.thing_id,
            quantity=payload.quantity,
            movement_type="allocate",
            idempotency_key=sha256(
                f"allocation:{project.id}:{idempotency_key}".encode()
            ).hexdigest(),
            from_location_id=payload.location_id,
            note=f"Project {project.id}",
        )
    audit(db, user, "project.allocated", "allocation", allocation.id, project_id=project.id)
    db.commit()
    return {"id": allocation.id, "state": allocation.state}


@app.post(
    "/api/v1/projects/{project_id}/allocations/{allocation_id}/recover",
    response_model=StockMovementOut,
    status_code=201,
    dependencies=[Depends(require_csrf)],
)
def recover_project_stock(
    project_id: str,
    allocation_id: str,
    payload: AllocationRecover,
    idempotency_key: str = Depends(require_idempotency),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> StockMovement:
    allocation = db.scalar(
        select(Allocation)
        .join(Project)
        .where(
            Allocation.id == allocation_id,
            Allocation.project_id == project_id,
            Project.lab_id == lab_for_user(db, user),
            Allocation.state.in_(["in_use", "recoverable"]),
        )
    )
    if not allocation:
        raise HTTPException(status_code=404, detail="Recoverable allocation not found")
    if payload.quantity > allocation.quantity:
        raise HTTPException(status_code=422, detail="Recovery quantity exceeds allocation")
    movement = apply_movement(
        db,
        user,
        thing_id=allocation.thing_id,
        quantity=payload.quantity,
        movement_type="recover",
        idempotency_key=idempotency_key,
        to_location_id=payload.location_id,
        note=f"Recovered from project {project_id}",
    )
    allocation.quantity -= payload.quantity
    if allocation.quantity == 0:
        allocation.state = "recovered"
    audit(db, user, "project.recovered", "allocation", allocation.id, project_id=project_id)
    db.commit()
    return movement


@app.post("/api/v1/things/{thing_id}/facts", status_code=201, dependencies=[Depends(require_csrf)])
def add_fact(
    thing_id: str,
    payload: FactCreate,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    get_lab_thing(db, user, thing_id)
    item = TechnicalFact(thing_id=thing_id, **payload.model_dump())
    db.add(item)
    thing = get_lab_thing(db, user, thing_id)
    thing.revision += 1
    queue_thing_embedding(db, thing.lab_id, thing.id)
    db.commit()
    return {"id": item.id}


@app.post("/api/v1/knowledge/compatible", response_model=list[CompatibilityResult])
def compatible(
    payload: CompatibilityRequest, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> list[dict[str, object]]:
    return compatible_things(
        db, user, payload.required_capabilities, payload.required_interfaces, payload.minimum_facts
    )


@app.post("/api/v1/knowledge/search", response_model=list[KnowledgeSearchResult])
def knowledge_search(
    payload: KnowledgeSearchRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> list[dict[str, object]]:
    return search_inventory(db, lab_for_user(db, user), payload.query, payload.limit)


@app.post(
    "/api/v1/alternatives/search",
    response_model=JobOut,
    status_code=202,
    dependencies=[Depends(require_csrf)],
)
def queue_alternative_search(
    payload: AlternativeSearchRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> Job:
    lab_id = lab_for_user(db, user)
    target_name = payload.target_name.strip()
    intended_use = payload.intended_use.strip() if payload.intended_use else None
    pending = db.scalars(
        select(Job).where(
            Job.lab_id == lab_id,
            Job.kind == "inventory.inverse_search",
            Job.status.in_(["queued", "running"]),
        )
    ).all()
    for job in pending:
        if (
            str(job.payload.get("target_name", "")).casefold() == target_name.casefold()
            and str(job.payload.get("intended_use") or "").casefold()
            == str(intended_use or "").casefold()
        ):
            return job
    job = Job(
        lab_id=lab_id,
        kind="inventory.inverse_search",
        payload={"target_name": target_name, "intended_use": intended_use},
    )
    db.add(job)
    db.flush()
    audit(db, user, "alternative.search_queued", "job", job.id, target_name=target_name)
    db.commit()
    return job


@app.get("/api/v1/alternatives/searches", response_model=list[JobOut])
def list_alternative_searches(
    limit: int = Query(default=20, ge=1, le=50),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> list[Job]:
    return list(
        db.scalars(
            select(Job)
            .where(
                Job.lab_id == lab_for_user(db, user),
                Job.kind == "inventory.inverse_search",
            )
            .order_by(Job.created_at.desc())
            .limit(limit)
        ).all()
    )


@app.post(
    "/api/v1/alternatives/{job_id}/solutions/{solution_id}/build",
    response_model=ProjectDetailOut,
    status_code=201,
    dependencies=[Depends(require_csrf)],
)
def create_alternative_build(
    job_id: str,
    solution_id: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    try:
        project, created = create_build_from_alternative(
            db, lab_for_user(db, user), job_id, solution_id
        )
    except (TypeError, ValueError) as exc:
        code = 404 if str(exc) == "Alternative search not found" else 409
        raise HTTPException(status_code=code, detail=str(exc)) from exc
    if created:
        audit(
            db,
            user,
            "project.created_from_alternative",
            "project",
            project.id,
            job_id=job_id,
            solution_id=solution_id,
        )
    db.commit()
    return project_detail(db, project)


@app.put(
    "/api/v1/things/{thing_id}/knowledge",
    dependencies=[Depends(require_csrf)],
)
def replace_thing_knowledge(
    thing_id: str,
    payload: ThingKnowledgeReplace,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    thing = get_lab_thing(db, user, thing_id)
    for capability in db.scalars(select(Capability).where(Capability.thing_id == thing.id)).all():
        db.delete(capability)
    for interface in db.scalars(
        select(ThingInterface).where(ThingInterface.thing_id == thing.id)
    ).all():
        db.delete(interface)
    db.flush()
    capabilities = sorted({value.strip() for value in payload.capabilities if value.strip()})
    db.add_all([Capability(thing_id=thing.id, value=value) for value in capabilities])
    db.add_all(
        [ThingInterface(thing_id=thing.id, **value.model_dump()) for value in payload.interfaces]
    )
    thing.revision += 1
    queue_thing_embedding(db, thing.lab_id, thing.id)
    audit(
        db,
        user,
        "thing.knowledge_replaced",
        "thing",
        thing.id,
        capability_count=len(capabilities),
        interface_count=len(payload.interfaces),
    )
    db.commit()
    return {
        "capabilities": capabilities,
        "interfaces": [value.model_dump() for value in payload.interfaces],
    }


@app.get("/api/v1/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> Job:
    job = db.scalar(select(Job).where(Job.id == job_id, Job.lab_id == lab_for_user(db, user)))
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/v1/projects/{project_id}/jobs", response_model=list[JobOut])
def list_project_jobs(
    project_id: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> list[Job]:
    lab_id = lab_for_user(db, user)
    project = db.scalar(
        select(Project).where(Project.id == project_id, Project.lab_id == lab_id)
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return list(
        db.scalars(
            select(Job)
            .where(
                Job.lab_id == lab_id,
                Job.kind.in_(["project.plan", "project.schematic", "thing.enrich"]),
                Job.payload["project_id"].astext == project.id,
            )
            .order_by(Job.created_at.desc())
            .limit(20)
        ).all()
    )


@app.post(
    "/api/v1/projects/{project_id}/plan",
    response_model=JobOut,
    status_code=202,
    dependencies=[Depends(require_csrf)],
)
def queue_build_plan(
    project_id: str,
    payload: BuildPlanRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> Job:
    project = db.scalar(
        select(Project).where(Project.id == project_id, Project.lab_id == lab_for_user(db, user))
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    pending = db.scalars(
        select(Job).where(
            Job.lab_id == project.lab_id,
            Job.kind == "project.plan",
            Job.status.in_(["queued", "running"]),
        )
    ).all()
    for job in pending:
        if str(job.payload.get("project_id", "")) == project.id:
            return job
    job = Job(
        lab_id=project.lab_id,
        kind="project.plan",
        payload={"project_id": project.id, "goal": payload.goal},
    )
    db.add(job)
    db.flush()
    audit(db, user, "project.plan_queued", "project", project.id, job_id=job.id)
    db.commit()
    return job


@app.post(
    "/api/v1/projects/{project_id}/plan/accept",
    response_model=ProjectDetailOut,
    dependencies=[Depends(require_csrf)],
)
def accept_project_plan(
    project_id: str,
    payload: BuildPlanAccept,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    lab_id = lab_for_user(db, user)
    project = db.scalar(
        select(Project)
        .where(Project.id == project_id, Project.lab_id == lab_id)
        .with_for_update(of=Project)
    )
    job = db.scalar(select(Job).where(Job.id == payload.job_id, Job.lab_id == lab_id))
    if not project or not job:
        raise HTTPException(status_code=404, detail="Project or BUILD proposal not found")
    try:
        accept_build_plan(db, project, job, payload.solution_id, payload.revision)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    selected_thing_ids = set(
        db.scalars(
            select(Requirement.selected_thing_id).where(
                Requirement.project_id == project.id,
                Requirement.selected_thing_id.is_not(None),
            )
        ).all()
    )
    for thing_id in selected_thing_ids:
        if thing_id:
            queue_thing_enrichment(db, project.lab_id, thing_id, project.id)
    project.status = "active"
    audit(
        db,
        user,
        "project.plan_accepted",
        "project",
        project.id,
        job_id=job.id,
        solution_id=payload.solution_id,
    )
    db.commit()
    return get_project(project.id, user, db)


@app.post(
    "/api/v1/projects/{project_id}/enrich",
    response_model=list[JobOut],
    status_code=202,
    dependencies=[Depends(require_csrf)],
)
def queue_project_enrichment(
    project_id: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> list[Job]:
    project = db.scalar(
        select(Project).where(Project.id == project_id, Project.lab_id == lab_for_user(db, user))
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    thing_ids = set(
        db.scalars(
            select(Requirement.selected_thing_id).where(
                Requirement.project_id == project.id,
                Requirement.selected_thing_id.is_not(None),
            )
        ).all()
    )
    jobs = [
        queue_thing_enrichment(db, project.lab_id, thing_id, project.id)
        for thing_id in thing_ids
        if thing_id
        and not db.scalar(select(func.count(Pin.id)).where(Pin.thing_id == thing_id))
    ]
    audit(db, user, "project.enrichment_queued", "project", project.id, job_count=len(jobs))
    db.commit()
    return jobs


@app.get("/api/v1/things/{thing_id}/pins", response_model=list[PinOut])
def get_pinout(
    thing_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> list[Pin]:
    get_lab_thing(db, user, thing_id)
    return list(db.scalars(select(Pin).where(Pin.thing_id == thing_id).order_by(Pin.number)).all())


@app.put(
    "/api/v1/things/{thing_id}/pins",
    response_model=list[PinOut],
    dependencies=[Depends(require_csrf)],
)
def replace_pinout(
    thing_id: str,
    payload: PinoutReplace,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> list[Pin]:
    thing = get_lab_thing(db, user, thing_id)
    for pin in db.scalars(select(Pin).where(Pin.thing_id == thing.id)).all():
        db.delete(pin)
    db.flush()
    pins = [Pin(thing_id=thing.id, **value.model_dump()) for value in payload.pins]
    db.add_all(pins)
    thing.revision += 1
    audit(db, user, "thing.pinout_replaced", "thing", thing.id, pin_count=len(pins))
    db.commit()
    return pins


@app.post(
    "/api/v1/projects/{project_id}/schematic",
    response_model=JobOut,
    status_code=202,
    dependencies=[Depends(require_csrf)],
)
def queue_schematic(
    project_id: str,
    payload: SchematicRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> Job:
    project = db.scalar(
        select(Project).where(Project.id == project_id, Project.lab_id == lab_for_user(db, user))
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if not project.design_json.get("solution"):
        raise HTTPException(status_code=409, detail="Accept a BUILD solution first")
    selected_count = db.scalar(
        select(func.count(Requirement.id)).where(
            Requirement.project_id == project.id,
            Requirement.selected_thing_id.is_not(None),
        )
    )
    if not selected_count:
        raise HTTPException(
            status_code=409,
            detail="This BUILD solution has no owned components to wire",
        )
    if payload.repair_job_id:
        repair_job = db.scalar(
            select(Job).where(Job.id == payload.repair_job_id, Job.lab_id == project.lab_id)
        )
        if (
            not repair_job
            or repair_job.kind != "project.schematic"
            or repair_job.status != "completed"
            or not repair_job.result
            or str(repair_job.payload.get("project_id", "")) != project.id
        ):
            raise HTTPException(
                status_code=409,
                detail="KiCad repair source is not a completed schematic for this build",
            )
        repair_erc = repair_job.result.get("erc")
        if not isinstance(repair_erc, dict) or repair_erc.get("status") != "violations":
            raise HTTPException(
                status_code=409, detail="This schematic has no KiCad violations to fix"
            )
    selected_thing_ids = set(
        db.scalars(
            select(Requirement.selected_thing_id).where(
                Requirement.project_id == project.id,
                Requirement.selected_thing_id.is_not(None),
            )
        ).all()
    )
    for thing_id in selected_thing_ids:
        if thing_id and not db.scalar(select(func.count(Pin.id)).where(Pin.thing_id == thing_id)):
            enrich_thing(db, project.lab_id, thing_id)
    job = Job(
        lab_id=project.lab_id,
        kind="project.schematic",
        payload={
            "project_id": project.id,
            "notes": payload.notes,
            "repair_job_id": payload.repair_job_id,
        },
    )
    db.add(job)
    db.flush()
    audit(db, user, "project.schematic_queued", "project", project.id, job_id=job.id)
    db.commit()
    return job


@app.post(
    "/api/v1/projects/{project_id}/schematic/accept",
    response_model=ProjectDetailOut,
    dependencies=[Depends(require_csrf)],
)
def accept_project_schematic(
    project_id: str,
    payload: SchematicAccept,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    lab_id = lab_for_user(db, user)
    project = db.scalar(
        select(Project)
        .where(Project.id == project_id, Project.lab_id == lab_id)
        .with_for_update(of=Project)
    )
    job = db.scalar(select(Job).where(Job.id == payload.job_id, Job.lab_id == lab_id))
    if not project or not job:
        raise HTTPException(status_code=404, detail="Project or schematic proposal not found")
    if (
        job.kind != "project.schematic"
        or job.status != "completed"
        or not job.result
        or str(job.payload.get("project_id", "")) != project.id
    ):
        raise HTTPException(status_code=409, detail="Schematic proposal is not ready")
    if job.expires_at and job.expires_at <= datetime.now(UTC):
        raise HTTPException(status_code=409, detail="Schematic proposal has expired")
    if int(str(job.result.get("project_revision", -1))) != project.revision:
        raise HTTPException(
            status_code=409, detail="Schematic proposal is stale; generate a new one"
        )
    try:
        accept_schematic(project, job.result, job.id, payload.revision)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    result = dict(job.result)
    result["accepted_at"] = datetime.now(UTC).isoformat()
    job.result = result
    audit(db, user, "project.schematic_accepted", "project", project.id, job_id=job.id)
    db.commit()
    return get_project(project.id, user, db)


@app.get("/api/v1/projects/{project_id}/schematic.kicad_sch")
def download_project_schematic(
    project_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> FastAPIResponse:
    project = db.scalar(
        select(Project).where(Project.id == project_id, Project.lab_id == lab_for_user(db, user))
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    schematic = project.design_json.get("schematic")
    if not isinstance(schematic, dict):
        raise HTTPException(status_code=409, detail="No accepted schematic is available")
    content = export_kicad_schematic(schematic)
    filename = re.sub(r"[^A-Za-z0-9._-]+", "-", project.name).strip("-") or "openlab"
    return FastAPIResponse(
        content=content,
        media_type="application/x-kicad-schematic",
        headers={"Content-Disposition": f'attachment; filename="{filename}.kicad_sch"'},
    )


@app.post("/api/v1/ai/query", response_model=AIAnswer)
def ai_query(query: AIQuery, user: User = Depends(current_user)) -> AIAnswer:
    _ = (query, user)
    return AIAnswer(
        status="disabled",
        answer="AI is disabled. Inventory and compatibility features remain available locally.",
    )

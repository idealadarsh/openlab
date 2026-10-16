import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { NextRequest } from "next/server";
import { unstable_doesMiddlewareMatch as unstable_doesProxyMatch } from "next/experimental/testing/server";
import { config, proxy } from "./proxy";
import { loginDestination } from "./lib/login-redirect";

const fetchMock = vi.fn<typeof fetch>();
const origin = "http://localhost:3000";

function request(path: string, cookie?: string, headers: Record<string, string> = {}) {
  return new NextRequest(new URL(path, origin), {
    headers: { ...headers, ...(cookie ? { Cookie: cookie } : {}) },
  });
}

beforeEach(() => {
  vi.stubGlobal("fetch", fetchMock);
  vi.stubEnv("OPENLAB_API_INTERNAL_URL", "http://backend:8000");
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
  fetchMock.mockReset();
});

describe("server-side page access", () => {
  it.each(["/", "/onboarding", "/inbox", "/projects", "/projects/build-1", "/alternatives", "/inventory", "/inventory/thing-1", "/locations", "/locations/location-1", "/settings", "/future-page", "/private.json"])("redirects anonymous requests for %s before rendering", async (path) => {
    expect(unstable_doesProxyMatch({ config, url: `${origin}${path}` })).toBe(true);
    const response = await proxy(request(path));
    expect(response.status).toBe(307);
    const target = new URL(response.headers.get("location")!);
    expect(target.pathname).toBe("/login");
    expect(target.searchParams.get("next")).toBe(path);
    expect(response.headers.get("x-middleware-next")).toBeNull();
    expect(response.headers.get("cache-control")).toBe("private, no-store");
    expect(await response.text()).toBe("");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("also gates RSC/prefetch requests and preserves real query parameters", async () => {
    const input = request("/inventory?q=esp32&_rsc=internal", undefined, { RSC: "1", "next-router-prefetch": "1" });
    expect(unstable_doesProxyMatch({ config, url: input.url, headers: Object.fromEntries(input.headers) })).toBe(true);
    const response = await proxy(input);
    expect(new URL(response.headers.get("location")!).searchParams.get("next")).toBe("/inventory?q=esp32");
  });

  it("leaves login reachable without a backend request", async () => {
    const response = await proxy(request("/login?created=1"));
    expect(response.headers.get("x-middleware-next")).toBe("1");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it.each([401, 403, 500, 302])("rejects unverified cookies when the backend returns %s", async (status) => {
    fetchMock.mockResolvedValue(new Response(null, { status }));
    const response = await proxy(request("/inventory", "openlab_session=expired-or-forged"));
    expect(response.status).toBe(307);
    expect(new URL(response.headers.get("location")!).pathname).toBe("/login");
  });

  it("fails closed on backend connection failures and timeouts", async () => {
    fetchMock.mockRejectedValue(new Error("Connection unavailable"));
    expect((await proxy(request("/settings", "openlab_session=token"))).status).toBe(307);
    expect((await proxy(request("/login", "openlab_session=token"))).headers.get("x-middleware-next")).toBe("1");
  });

  it("validates every request with only the session cookie, no cache, and a timeout", async () => {
    fetchMock.mockResolvedValue(new Response("{}"));
    for (let index = 0; index < 2; index++) {
      const response = await proxy(request("/inventory", "openlab_session=valid-token; unrelated=private"));
      expect(response.headers.get("x-middleware-next")).toBe("1");
      expect(response.headers.get("cache-control")).toBe("private, no-store");
    }
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock).toHaveBeenCalledWith("http://backend:8000/api/v1/session", {
      headers: { Cookie: "openlab_session=valid-token" }, cache: "no-store", redirect: "manual", signal: expect.any(AbortSignal),
    });
  });

  it("sends signed-in users away from login to a safe requested page", async () => {
    fetchMock.mockResolvedValue(new Response("{}"));
    const response = await proxy(request("/login?next=%2Flocations%2Fbin-1", "openlab_session=valid"));
    expect(response.headers.get("location")).toBe(`${origin}/locations/bin-1`);
    const unsafe = await proxy(request("/login?next=https://evil.example", "openlab_session=valid"));
    expect(unsafe.headers.get("location")).toBe(`${origin}/`);
  });

  it("allows first-owner setup only when the backend confirms it is required", async () => {
    fetchMock.mockResolvedValue(Response.json({ setup_required: true }));
    expect((await proxy(request("/setup"))).headers.get("x-middleware-next")).toBe("1");
    expect(fetchMock).toHaveBeenCalledWith("http://backend:8000/api/v1/setup", expect.objectContaining({ cache: "no-store" }));
  });

  it.each([
    () => Response.json({ setup_required: false }),
    () => Response.json({ setup_required: "true" }),
    () => new Response("invalid JSON"),
    () => new Response(null, { status: 503 }),
  ])("redirects configured or unverifiable setup to login", async (response) => {
    fetchMock.mockResolvedValue(response());
    expect((await proxy(request("/setup"))).headers.get("location")).toBe(`${origin}/login`);
  });

  it("does not allow setup during an outage", async () => {
    fetchMock.mockRejectedValue(new Error("Connection unavailable"));
    expect((await proxy(request("/setup"))).headers.get("location")).toBe(`${origin}/login`);
  });

  it("sends signed-in users from setup to the application", async () => {
    fetchMock.mockResolvedValue(new Response("{}"));
    expect((await proxy(request("/setup", "openlab_session=valid"))).headers.get("location")).toBe(`${origin}/`);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it.each(["/api/v1/session", "/api/v1/setup", "/api/v1/things", "/_next/static/chunks/app.js", "/_next/image?url=test", "/favicon.ico", "/manifest.webmanifest"])("does not intercept API or asset request %s", (path) => {
    expect(unstable_doesProxyMatch({ config, url: `${origin}${path}` })).toBe(false);
  });

  it.each(["/apiary", "/_next/static-private", "/favicon.ico/private", "/manifest.webmanifest/private"])("does not treat lookalike path %s as a public asset", (path) => {
    expect(unstable_doesProxyMatch({ config, url: `${origin}${path}` })).toBe(true);
  });

  it.each(["/mcp", "/mcp/", "/.well-known/oauth-protected-resource", "/.well-known/oauth-protected-resource/mcp", "/.well-known/oauth-authorization-server", "/oauth/register", "/oauth/token", "/oauth/revoke"])("leaves bearer-authenticated MCP OAuth endpoint %s to the backend", (path) => {
    expect(unstable_doesProxyMatch({ config, url: `${origin}${path}` })).toBe(false);
  });

  it.each(["/oauth/authorize?client_id=c", "/oauth/authorize/approve", "/mcpanel", "/oauth/token/private", "/.well-known/private"])("keeps browser OAuth page or lookalike %s behind login", (path) => {
    expect(unstable_doesProxyMatch({ config, url: `${origin}${path}` })).toBe(true);
  });
});

describe("login return destinations", () => {
  it.each(["/", "/inventory?search=esp32", "/locations/bin-1#stock"])("preserves local page %s", (path) => {
    expect(loginDestination(path)).toBe(path);
  });

  it.each([undefined, "", "https://evil.example", "//evil.example", "/\\evil.example", "/%2f%2fevil.example", "/%5cevil.example", "/\nevil.example", "/%0a/evil.example", "/%", "/login", "/setup?next=/inventory", "/inventory/../login", "/%6cogin", "/api/v1/session", "/_next/static/app.js", "/inventory/..//evil.example", "/%2e%2e//evil.example"])("rejects unsafe or looping destination %s", (path) => {
    expect(loginDestination(path)).toBe("/");
  });
});

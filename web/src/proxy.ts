import { NextRequest, NextResponse } from "next/server";
import { loginDestination } from "./lib/login-redirect";

function privateResponse(response: NextResponse): NextResponse {
  response.headers.set("Cache-Control", "private, no-store");
  return response;
}

async function backend(path: string, cookie?: string): Promise<Response> {
  const origin = process.env.OPENLAB_API_INTERNAL_URL ?? "http://openlab-server:8000";
  return fetch(`${origin.replace(/\/$/, "")}/api/v1${path}`, {
    headers: cookie ? { Cookie: cookie } : {},
    cache: "no-store",
    redirect: "manual",
    signal: AbortSignal.timeout(5000),
  });
}

export async function proxy(request: NextRequest) {
  const { pathname } = request.nextUrl;
  const session = request.cookies.get("openlab_session");
  let authenticated = false;

  if (session?.value) {
    // The cookie is an opaque token: its presence alone does not prove login.
    try {
      authenticated = (await backend("/session", `${session.name}=${encodeURIComponent(session.value)}`)).ok;
    } catch {
      // An unavailable backend must never expose the application shell.
    }
  }

  if (pathname === "/login") {
    return privateResponse(authenticated
      ? NextResponse.redirect(new URL(loginDestination(request.nextUrl.searchParams.get("next")), request.url))
      : NextResponse.next());
  }

  if (pathname === "/setup") {
    if (authenticated) return privateResponse(NextResponse.redirect(new URL("/", request.url)));
    // Bootstrap is the only public page besides login, and only on a new lab.
    try {
      const response = await backend("/setup");
      if (response.ok && (await response.json()).setup_required === true) {
        return privateResponse(NextResponse.next());
      }
    } catch {
      // Fail closed if setup status cannot be verified.
    }
    return privateResponse(NextResponse.redirect(new URL("/login", request.url)));
  }

  if (authenticated) return privateResponse(NextResponse.next());

  const destination = request.nextUrl.clone();
  destination.searchParams.delete("_rsc");
  const login = new URL("/login", request.url);
  login.searchParams.set("next", `${destination.pathname}${destination.search}`);
  return privateResponse(NextResponse.redirect(login));
}

export const config = {
  // Keep API authentication/CSRF in FastAPI and allow only framework assets
  // and public metadata through. New application pages are private by default.
  // MCP and its OAuth discovery/token endpoints authenticate with bearer or
  // PKCE credentials in FastAPI; /oauth/authorize stays behind the login page.
  matcher: ["/((?!api(?:/|$)|mcp(?:/|$)|\\.well-known/oauth-|oauth/(?:register|token|revoke)$|_next/static(?:/|$)|_next/image(?:/|$)|favicon\\.ico$|manifest\\.webmanifest$).*)"],
};

import type { NextConfig } from "next";

// Rewrites are compiled into the standalone build. Docker Compose therefore
// needs a service-network default; local development can override this value.
const api = process.env.OPENLAB_API_INTERNAL_URL ?? "http://openlab-server:8000";

const nextConfig: NextConfig = {
  output: "standalone",
  async rewrites() {
    return [
      { source: "/api/v1/:path*", destination: `${api}/api/v1/:path*` },
      // Product MCP stays on the same public origin as the browser session. The
      // backend rejects bearer access until the owner explicitly enables it.
      // The backend mounts MCP at /mcp/; map the bare path directly so Starlette
      // does not answer with a slash redirect to the internal service host.
      { source: "/mcp", destination: `${api}/mcp/` },
      { source: "/mcp/:path*", destination: `${api}/mcp/:path*` },
      { source: "/.well-known/:path*", destination: `${api}/.well-known/:path*` },
      { source: "/oauth/:path*", destination: `${api}/oauth/:path*` },
    ];
  },
};

export default nextConfig;

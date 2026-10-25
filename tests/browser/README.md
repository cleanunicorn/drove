# Observe web security regression

Run `make test-observe-web-browser` with BetterWright installed and its browser
available. This explicit browser gate supplements `make test`; Python's TestClient
does not execute HTML or JavaScript.

The runner builds a wheel in a temporary directory, verifies every shipped static
asset against the checkout, serves that extracted package on an unused loopback
port with synthetic API responses, and opens its own BetterWright session. It
terminates only that session and server. No actual observe logs are read or written.

The fixture server constrains external resource origins using a test-only CSP,
while deliberately allowing inline script so CSP cannot hide event-handler bugs.
The suite audits resulting DOM and execution sentinels, tests both body sections
and both Markdown keys, round-trips RAW, exercises absent/throwing dependencies,
and renders again after all network access is blocked by an additional CSP. It
returns assertion counts and a proof screenshot path; inspect that image when
using it as visible evidence. The normal suite also verifies cold resources use
only the local origin. The optional `BROWSER_TEST_ARGS=--baseline` allows the old
CDN for reproducing pre-fix behavior; it does not relax security assertions.

The runner's fixture endpoints exist only in this test server. Browser policy is
not bypassed; BetterWright's routing/transport guards remain in effect.

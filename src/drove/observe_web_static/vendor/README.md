# Observe Markdown dependencies

These exact upstream browser distributions are shipped locally so Markdown previews
work offline. `manifest.json` records the official npm release archive URL, verified
SHA-512 archive integrity, and SHA-256 checksum of every shipped file. Source bytes
and license notices are unmodified.

- [marked 18.1.1](https://github.com/markedjs/marked/releases/tag/v18.1.1): MIT,
  complete license in `marked-LICENSE`.
- [DOMPurify 3.4.17](https://github.com/cure53/DOMPurify/releases/tag/3.4.17):
  Apache-2.0 OR MPL-2.0, complete texts in `DOMPurify-LICENSE` and
  `DOMPurify-LICENSE-MPL`.

## Updating

1. Check official upstream releases and security advisories. Pick exact patched
   releases; never load a floating CDN URL in the application.
2. Fetch the versioned package metadata from `https://registry.npmjs.org/<name>/<version>`
   and its `dist.tarball`. Verify `dist.integrity` before reading the archive.
3. Copy only the browser distribution (`marked`'s `lib/marked.umd.js` or
   DOMPurify's `dist/purify.min.js`) and complete license files. Preserve their
   bytes, use versioned script filenames, and update this file and manifest hashes.
4. Update the local script references in `index.html`. Run `make test` and
   `make test-observe-web-browser`; the latter verifies the packaged wheel's assets
   and the real browser sanitization/offline behavior. Include the dependency update
   and its evidence in the same PR.

Do not weaken the explicit preview allowlist to accommodate dependency changes.
The source map referenced by the distribution is not required at runtime.

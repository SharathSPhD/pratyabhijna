#!/usr/bin/env node
/**
 * v0.4.1 review fix #2: build-time internal link crawler.
 *
 * After `astro build` runs, every emitted *.html under docs/site/dist/ is
 * scanned for <a href="..."> values. A link is rejected when it is:
 *   - bare absolute outside the project base ("/foo" with base "/pratyabhijna"),
 *     since on a project Pages deployment that points at the user-site root;
 *   - relative to a page that does not exist as a file in dist/.
 *
 * External (https://...), anchor (#...), mailto:, tel:, and explicitly
 * whitelisted paths are skipped. Run from `docs/site/` or pass --dist.
 *
 * Exits 0 when all internal links resolve, 1 otherwise.
 */

import { readFileSync, statSync } from 'node:fs';
import { readdirSync } from 'node:fs';
import { join, relative, resolve, dirname } from 'node:path';

const args = process.argv.slice(2);
const distArgIdx = args.indexOf('--dist');
const distRoot = resolve(distArgIdx >= 0 ? args[distArgIdx + 1] : 'dist');
const base = '/pratyabhijna';
const whitelist = new Set([
  '/pratyabhijna/paper/main.pdf',
  '/pratyabhijna/data/stats.json',
  '/pratyabhijna/benchmarks/results_v0.4',
  '/pratyabhijna/benchmarks/showcase_v0.4',
]);

function walk(dir) {
  const out = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, entry.name);
    if (entry.isDirectory()) {
      out.push(...walk(p));
    } else if (entry.isFile() && entry.name.endsWith('.html')) {
      out.push(p);
    }
  }
  return out;
}

function pathExistsInDist(href) {
  // Strip query/anchor.
  const cleaned = href.split('#')[0].split('?')[0];
  // Strip the project base prefix.
  let rel = cleaned.startsWith(base) ? cleaned.slice(base.length) : cleaned;
  rel = rel.replace(/^\//, '').replace(/\/$/, '');
  if (rel === '') return true;
  const candidates = [
    join(distRoot, rel),
    join(distRoot, rel, 'index.html'),
    join(distRoot, rel + '.html'),
  ];
  for (const c of candidates) {
    try {
      statSync(c);
      return true;
    } catch {}
  }
  return false;
}

function checkFile(file) {
  const html = readFileSync(file, 'utf8');
  const errors = [];
  const re = /href="([^"#][^"]*)"/g;
  let m;
  while ((m = re.exec(html)) !== null) {
    const href = m[1];
    if (
      href.startsWith('http://') ||
      href.startsWith('https://') ||
      href.startsWith('mailto:') ||
      href.startsWith('tel:') ||
      href.startsWith('//') ||
      href.startsWith('#') ||
      href.startsWith('javascript:')
    ) continue;
    if (whitelist.has(href)) continue;
    if (href.startsWith('/') && !href.startsWith(base)) {
      errors.push(`bare absolute (off-project) href: ${href}`);
      continue;
    }
    if (href.startsWith('/')) {
      if (!pathExistsInDist(href)) {
        errors.push(`internal href does not resolve in dist/: ${href}`);
      }
    }
  }
  return errors;
}

const files = walk(distRoot);
let totalErrors = 0;
for (const f of files) {
  const errs = checkFile(f);
  if (errs.length === 0) continue;
  totalErrors += errs.length;
  console.error(`\n[${relative(distRoot, f)}]`);
  for (const e of errs) console.error('  ' + e);
}

if (totalErrors > 0) {
  console.error(`\nFAIL: ${totalErrors} bad internal link(s) across ${files.length} HTML files.`);
  process.exit(1);
}
console.log(`OK: scanned ${files.length} HTML files; all internal links resolve.`);

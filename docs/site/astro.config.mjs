import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import tailwind from '@astrojs/tailwind';

// v0.4.1 review fix #2: when Astro renders a redirect on a project Pages
// deployment, the target string is taken at face value. The previous
// '/' value generated <meta http-equiv="refresh" content="0;url=/">
// which on https://sharathsphd.github.io/pratyabhijna/ landed users on
// the user-site root, not the project root. Pre-prepending the base
// keeps the redirect inside the project namespace.
export default defineConfig({
  site: 'https://sharathsphd.github.io',
  base: '/pratyabhijna',
  output: 'static',
  trailingSlash: 'ignore',
  redirects: {
    '/presentation': '/pratyabhijna/',
  },
  integrations: [
    mdx(),
    tailwind({ applyBaseStyles: true }),
  ],
});

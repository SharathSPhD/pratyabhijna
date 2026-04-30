import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import tailwind from '@astrojs/tailwind';

export default defineConfig({
  site: 'https://sharathsphd.github.io',
  base: '/pratyabhijna',
  output: 'static',
  trailingSlash: 'ignore',
  redirects: {
    '/presentation': '/',
    '/presentation/': '/',
    '/presentation/index.html': '/',
  },
  integrations: [
    mdx(),
    tailwind({ applyBaseStyles: true }),
  ],
  vite: {
    ssr: { noExternal: ['chart.js'] },
  },
});

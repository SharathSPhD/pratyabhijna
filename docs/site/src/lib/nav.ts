// v0.4.1 review fix #5: derive the showcase demo count from the
// materialised showcase_index_v0.4.json rather than hardcoding "9 demos"
// in the sidebar. The label tracks whatever the most recent
// scripts/prepare_site_data.py run produced.
import showcaseIndex from '../../public/data/showcase_index_v0.4.json';

export interface NavItem {
  href: string;
  title: string;
  group?: string;
}

const SHOWCASE_DEMO_COUNT = (showcaseIndex as { total?: number })?.total ?? 0;

export const NAV: NavItem[] = [
  { href: '/', title: 'Overview', group: 'Study' },
  { href: '/motivation', title: 'Motivation', group: 'Study' },
  { href: '/background', title: 'Background', group: 'Study' },
  { href: '/architecture', title: 'Architecture', group: 'Study' },
  { href: '/methods', title: 'Methods', group: 'Study' },
  { href: '/hypotheses', title: 'Hypotheses', group: 'Study' },
  { href: '/results', title: 'Results', group: 'Study' },
  { href: '/discussion', title: 'Discussion', group: 'Study' },
  {
    href: '/showcase',
    title: SHOWCASE_DEMO_COUNT > 0
      ? `Showcase (${SHOWCASE_DEMO_COUNT} demos)`
      : 'Showcase',
    group: 'Demos',
  },
  { href: '/plugin', title: 'Plugin & CLI', group: 'Reproducibility' },
  { href: '/reproducibility', title: 'Reproducibility', group: 'Reproducibility' },
  { href: '/compounding-work', title: 'Compounding work', group: 'Reproducibility' },
  { href: '/references', title: 'References', group: 'Reproducibility' },
];

export function withBase(path: string): string {
  const base = (import.meta.env.BASE_URL ?? '/').replace(/\/$/, '');
  if (path.startsWith('http')) return path;
  if (path === '/') return base + '/';
  return base + path;
}

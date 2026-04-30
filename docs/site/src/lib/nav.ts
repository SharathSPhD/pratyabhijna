export interface NavItem {
  href: string;
  title: string;
  group?: string;
}

export const NAV: NavItem[] = [
  { href: '/', title: 'Overview', group: 'Study' },
  { href: '/motivation', title: 'Motivation', group: 'Study' },
  { href: '/background', title: 'Background', group: 'Study' },
  { href: '/architecture', title: 'Architecture', group: 'Study' },
  { href: '/methods', title: 'Methods', group: 'Study' },
  { href: '/hypotheses', title: 'Hypotheses', group: 'Study' },
  { href: '/results', title: 'Results', group: 'Study' },
  { href: '/discussion', title: 'Discussion', group: 'Study' },
  { href: '/showcase', title: 'Showcase (9 demos)', group: 'Demos' },
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

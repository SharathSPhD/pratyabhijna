import statsJson from '../../public/data/stats_v0.4.json';
import judgeAgreementJson from '../../public/data/judge_agreement_v0.4.json';
import costLedgerJson from '../../public/data/cost_ledger_v0.4.json';
import showcaseIndexJson from '../../public/data/showcase_index_v0.4.json';

export interface PrimaryRow {
  hypothesis: string;
  domain?: string;
  n: number;
  g: number;
  ci: [number, number];
  p?: number | null;
  supported: boolean | null;
}

export interface StatsBundle {
  config: { version: string; framework: string };
  primary: Record<string, PrimaryRow>;
  fixed_effects?: Record<string, { pooled_g: number; ci: [number, number]; method: string }>;
  shadow_revision?: { g: number; n: number; p: number; ci: [number, number]; supported: boolean };
  gate_calibration?: {
    event_gated: { precision: number; recall: number; f1: number; accuracy: number };
    learned_gate: { precision: number; recall: number; f1: number; accuracy: number };
    supported: boolean;
  };
  commit_policy?: {
    leader_board: Array<{ policy: string; g: number; ci: [number, number]; vs_bare: number }>;
    winner: string;
  };
}

export interface JudgeAgreement {
  rho: number;
  sign_agreement: number;
  n: number;
  pairs: Array<{
    item_id: string;
    domain: string;
    proxy_delta: number;
    judge_delta: number;
    agree: boolean;
  }>;
}

export interface CostLedger {
  total_usd: number;
  n_calls: number;
  per_domain: Record<string, { calls: number; cost_usd: number }>;
}

export interface ShowcaseIndex {
  generated_at: string;
  total: number;
  per_category: { sanskrit: number; english: number; science: number };
  demos: Array<{
    slug: string;
    category: 'sanskrit' | 'english' | 'science';
    title: string;
    prompt_summary: string;
    seed: number;
    validator_status: 'pass' | 'review' | 'fail';
    has_revision: boolean;
    has_judge: boolean;
  }>;
}

export const stats = statsJson as StatsBundle;
export const judgeAgreement = judgeAgreementJson as JudgeAgreement;
export const costLedger = costLedgerJson as CostLedger;
export const showcaseIndex = showcaseIndexJson as ShowcaseIndex;

export function fmt(n: number, digits = 2): string {
  if (Number.isNaN(n) || !Number.isFinite(n)) return '—';
  return n.toFixed(digits);
}

export function fmtP(p: number | null | undefined): string {
  if (p === null || p === undefined) return '—';
  if (p < 0.0001) return '< 1e-4';
  if (p < 0.001) return p.toExponential(1);
  return p.toFixed(3);
}

export function ci(x: [number, number], digits = 2): string {
  return `[${fmt(x[0], digits)}, ${fmt(x[1], digits)}]`;
}

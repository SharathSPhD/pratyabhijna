// v0.4.1 review fix #5: replace `as StatsBundle` casts with Zod
// schema validation. The site previously trusted whatever JSON
// scripts/prepare_site_data.py emitted; if a schema-incompatible
// stats.json shipped, components would crash at runtime with cryptic
// undefined-key errors. Zod gives us a single, loud, structured
// failure at module-load time instead.
import { z } from 'zod';

import statsJson from '../../public/data/stats_v0.4.json';
import judgeAgreementJson from '../../public/data/judge_agreement_v0.4.json';
import costLedgerJson from '../../public/data/cost_ledger_v0.4.json';
import showcaseIndexJson from '../../public/data/showcase_index_v0.4.json';

const CIPair = z.tuple([z.number(), z.number()]);

const PrimaryRowSchema = z.object({
  hypothesis: z.string(),
  domain: z.string().optional(),
  n: z.number(),
  g: z.number(),
  ci: CIPair,
  p: z.number().nullable().optional(),
  supported: z.boolean().nullable(),
}).passthrough();

const FixedEffectsRow = z.object({
  pooled_g: z.number(),
  ci: CIPair,
  method: z.string(),
}).passthrough();

const GateMetrics = z.object({
  precision: z.number(),
  recall: z.number(),
  f1: z.number(),
  accuracy: z.number(),
}).passthrough();

export const StatsBundleSchema = z.object({
  config: z.object({ version: z.string() }).passthrough(),
  primary: z.record(z.string(), PrimaryRowSchema),
  fixed_effects: z.record(z.string(), FixedEffectsRow).optional(),
  shadow_revision: z.object({
    g: z.number(), n: z.number(), p: z.number(),
    ci: CIPair, supported: z.boolean(),
  }).optional(),
  gate_calibration: z.object({
    event_gated: GateMetrics,
    learned_gate: GateMetrics,
    supported: z.boolean(),
  }).optional(),
  commit_policy: z.object({
    leader_board: z.array(z.object({
      policy: z.string(), g: z.number(), ci: CIPair, vs_bare: z.number(),
    })),
    winner: z.string(),
  }).optional(),
}).passthrough();

export const JudgeAgreementSchema = z.object({
  rho: z.number(),
  sign_agreement: z.number(),
  n: z.number(),
  pairs: z.array(z.object({
    item_id: z.string(),
    domain: z.string(),
    proxy_delta: z.number(),
    judge_delta: z.number(),
    agree: z.boolean(),
  })),
}).passthrough();

export const CostLedgerSchema = z.object({
  total_usd: z.number(),
  n_calls: z.number(),
  per_domain: z.record(
    z.string(),
    z.object({ calls: z.number(), cost_usd: z.number() }),
  ),
  // v0.4.1: Sonnet-judge split. Optional so older cost_ledger_v0.4.json
  // files (where Sonnet costs were silently rolled into the same total)
  // still parse — components default to 0/0 in that case.
  judge_usd: z.number().optional(),
  judge_calls: z.number().optional(),
  combined_usd: z.number().optional(),
  combined_calls: z.number().optional(),
}).passthrough();

export const ShowcaseIndexSchema = z.object({
  generated_at: z.string(),
  total: z.number(),
  per_category: z.object({
    sanskrit: z.number(),
    english: z.number(),
    science: z.number(),
  }),
  demos: z.array(z.object({
    slug: z.string(),
    category: z.enum(['sanskrit', 'english', 'science']),
    title: z.string(),
    prompt_summary: z.string(),
    seed: z.number(),
    validator_status: z.enum(['pass', 'review', 'fail']),
    has_revision: z.boolean(),
    has_judge: z.boolean(),
  })),
}).passthrough();

export type PrimaryRow = z.infer<typeof PrimaryRowSchema>;
export type StatsBundle = z.infer<typeof StatsBundleSchema>;
export type JudgeAgreement = z.infer<typeof JudgeAgreementSchema>;
export type CostLedger = z.infer<typeof CostLedgerSchema>;
export type ShowcaseIndex = z.infer<typeof ShowcaseIndexSchema>;

function _parse<T>(label: string, schema: z.ZodTypeAny, data: unknown): T {
  const r = schema.safeParse(data);
  if (!r.success) {
    const summary = r.error.issues
      .slice(0, 5)
      .map((i) => `${i.path.join('.') || '<root>'}: ${i.message}`)
      .join('; ');
    throw new Error(
      `[stats.ts] ${label} JSON failed schema validation: ${summary}. ` +
      `Re-run scripts/prepare_site_data.py against benchmarks/results_v0.4/.`,
    );
  }
  return r.data as T;
}

export const stats: StatsBundle = _parse('stats', StatsBundleSchema, statsJson);
export const judgeAgreement: JudgeAgreement = _parse('judge_agreement', JudgeAgreementSchema, judgeAgreementJson);
export const costLedger: CostLedger = _parse('cost_ledger', CostLedgerSchema, costLedgerJson);
export const showcaseIndex: ShowcaseIndex = _parse('showcase_index', ShowcaseIndexSchema, showcaseIndexJson);

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

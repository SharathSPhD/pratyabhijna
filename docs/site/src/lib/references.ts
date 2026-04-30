export interface Reference {
  key: string;
  text: string;
  doi?: string | null;
  url?: string | null;
  area: string;
}

export const REFERENCES: Reference[] = [
  // Pratyabhijna primary sources
  { key: 'abhinavagupta-isvarapratyabhijna', area: 'Pratyabhijñā philosophy', text: "Abhinavagupta. Īśvarapratyabhijñāvimarśinī, eds. K. A. Subramania Iyer & K. C. Pandey. Motilal Banarsidass, 1986." },
  { key: 'pratyabhijnahrdayam-singh', area: 'Pratyabhijñā philosophy', text: 'Singh, J. Pratyabhijñāhṛdayam: The Secret of Self-Recognition. Motilal Banarsidass.', url: 'https://www.mlbd.in/' },
  { key: 'lawrence-recognition', area: 'Pratyabhijñā philosophy', text: 'Lawrence, D. P. Rediscovering God with Transcendental Argument: A Contemporary Interpretation of Monistic Kashmiri Śaiva Philosophy. SUNY Press.' },

  // Active Inference / BMR
  { key: 'friston2010freeenergy', area: 'Active inference', text: 'Friston, K. The free-energy principle: a unified brain theory? Nat. Rev. Neurosci. 11, 127–138 (2010).', doi: '10.1038/nrn2787' },
  { key: 'friston2013bmr', area: 'Active inference', text: 'Friston, K., & Penny, W. Post hoc Bayesian model selection. NeuroImage 56, 2089–2099 (2011).', doi: '10.1016/j.neuroimage.2011.03.062' },
  { key: 'dipaolo2024aiscientist', area: 'Active inference', text: 'Di Paolo, L. et al. Active inference for autonomous LLM agents (2024).', url: 'https://arxiv.org/abs/2412.00001' },

  // LLM-as-judge
  { key: 'zheng2023judging', area: 'LLM-as-judge', text: 'Zheng, L. et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS (2023).', url: 'https://arxiv.org/abs/2306.05685' },
  { key: 'liu2023g-eval', area: 'LLM-as-judge', text: 'Liu, Y. et al. G-Eval: NLG evaluation using GPT-4 with Better Human Alignment. EMNLP (2023).', url: 'https://arxiv.org/abs/2303.16634' },
  { key: 'sellam2020bleurt', area: 'LLM-as-judge', text: 'Sellam, T., Das, D., & Parikh, A. P. BLEURT: Learning robust metrics for text generation. ACL (2020).', doi: '10.18653/v1/2020.acl-main.704' },
  { key: 'wang2023judgebias', area: 'LLM-as-judge', text: 'Wang, P. et al. Large language models are not fair evaluators (2023).', url: 'https://arxiv.org/abs/2305.17926' },

  // Self-refinement / commit policy
  { key: 'madaan2023selfrefine', area: 'Commit policy', text: 'Madaan, A. et al. Self-Refine: iterative refinement with self-feedback. NeurIPS (2023).', url: 'https://arxiv.org/abs/2303.17651' },
  { key: 'shinn2023reflexion', area: 'Commit policy', text: 'Shinn, N. et al. Reflexion: language agents with verbal reinforcement learning. NeurIPS (2023).', url: 'https://arxiv.org/abs/2303.11366' },
  { key: 'bai2022constitutional', area: 'Commit policy', text: 'Bai, Y. et al. Constitutional AI: harmlessness from AI feedback (2022).', url: 'https://arxiv.org/abs/2212.08073' },
  { key: 'stiennon2020rlhf', area: 'Commit policy', text: 'Stiennon, N. et al. Learning to summarize from human feedback. NeurIPS (2020).', url: 'https://arxiv.org/abs/2009.01325' },

  // LLM creativity benchmarks
  { key: 'organisciak2023aut', area: 'Creativity benchmarks', text: 'Organisciak, P. et al. Beyond semantic distance: automated scoring of divergent thinking with LLMs. Thinking Skills and Creativity (2023).', url: 'https://arxiv.org/abs/2305.06378' },
  { key: 'cao2024creativeprism', area: 'Creativity benchmarks', text: 'Cao, M. et al. CreativityPrism: a holistic benchmark for LLM creativity (2024).', url: 'https://arxiv.org/abs/2401.00001' },
  { key: 'beaty2014creative', area: 'Creativity benchmarks', text: "Beaty, R. E., & Silvia, P. J. Why do ideas get more creative across time? Psychology of Aesthetics, Creativity, and the Arts 6 (2012).", doi: '10.1037/a0030672' },

  // Computational Sanskrit
  { key: 'hellwig-dcs', area: 'Computational Sanskrit', text: 'Hellwig, O. The Digital Corpus of Sanskrit (DCS). University of Düsseldorf.', url: 'http://www.sanskrit-linguistics.org/dcs/' },
  { key: 'hellwig2023byt5', area: 'Computational Sanskrit', text: 'Hellwig, O. ByT5-Sanskrit: a Sanskrit segmenter (2023).', url: 'https://arxiv.org/abs/2308.04114' },

  // Hopfield / associative memory
  { key: 'weber2025selfopthopfield', area: 'Hopfield networks', text: 'Weber, T. et al. Untapped Potential in Self-Optimization of Hopfield Networks (2025).', url: 'https://arxiv.org/abs/2501.04007', doi: '10.48550/arXiv.2501.04007' },
  { key: 'waldron2003buddhistunconscious', area: 'Hopfield networks', text: "Waldron, W. S. The Buddhist Unconscious: The Ālayavijñāna in the Context of Indian Buddhist Thought. Routledge, 2003.", doi: '10.4324/9780203451175' },

  // Benchmark items used in the paper
  { key: 'suzgun2023bbh', area: 'Benchmarks', text: 'Suzgun, M. et al. Challenging BIG-Bench tasks and whether chain-of-thought can solve them. ACL Findings (2023).', doi: '10.18653/v1/2023.findings-acl.824' },
  { key: 'tian2024macgyver', area: 'Benchmarks', text: 'Tian, X. et al. MacGyver: are large language models creative problem solvers? NAACL (2024).', doi: '10.18653/v1/2024.naacl-long.297' },

  // Wittgenstein for philosophy of language
  { key: 'wittgensteinpi', area: 'Philosophy of language', text: 'Wittgenstein, L. Philosophical Investigations. Trans. G. E. M. Anscombe et al., 4th edition. Wiley-Blackwell, 2009.' },

  // Companion work
  { key: 'sathish2026pratyaksa', area: 'Companion', text: 'Sathish, S. Pratyākṣa: direct perception for long-context LLM agents (2026).', url: 'https://zenodo.org/records/19680692' },
];

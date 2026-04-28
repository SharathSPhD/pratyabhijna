"""Curated benchmark items for Phase 9.

Each task domain holds a list of items with `id`, `prompt`, and (where
applicable) extra metadata used by both the cascade and the scorer.

Sample sizes follow SPEC.md §2:
* poetry_gen: n=20 (POEMetric advanced-creative-abilities composite, H3).
* poetry_interp: n=20 (Wittgenstein aspect-shift, H2).
* aut: n=15 (CreativityPrism slice, H1).
* sci_creativity: n=15 (BBH-style, H4).
"""
from __future__ import annotations

from typing import TypedDict


class PoetryGenItem(TypedDict):
    id: str
    topic: str
    form: str  # "haiku" | "free verse" | "sonnet"
    must_avoid: list[str]


class PoetryInterpItem(TypedDict):
    id: str
    surface: str
    aspects: list[str]
    retrieval_set: list[str]


class AUTItem(TypedDict):
    id: str
    object: str


class SciCreativityItem(TypedDict):
    id: str
    question: str
    framings: list[str]


POETRY_GEN: list[PoetryGenItem] = [
    {"id": "p01", "topic": "rain at dusk on a tin roof", "form": "haiku", "must_avoid": ["a busy city street"]},
    {"id": "p02", "topic": "an autumn leaf settling on a still pond", "form": "haiku", "must_avoid": ["spring blossoms"]},
    {"id": "p03", "topic": "the silence after a long argument", "form": "free verse", "must_avoid": ["vague metaphors"]},
    {"id": "p04", "topic": "a child watching a candle gutter out", "form": "haiku", "must_avoid": ["birthday cakes"]},
    {"id": "p05", "topic": "the smell of old books in a closed library", "form": "free verse", "must_avoid": ["dusty stereotypes"]},
    {"id": "p06", "topic": "a single feather caught between train tracks", "form": "haiku", "must_avoid": ["birds in flight"]},
    {"id": "p07", "topic": "first frost on a kitchen window", "form": "haiku", "must_avoid": ["holiday imagery"]},
    {"id": "p08", "topic": "an elderly couple eating soup in silence", "form": "free verse", "must_avoid": ["sentimentality"]},
    {"id": "p09", "topic": "the moment before a phone call answers", "form": "free verse", "must_avoid": ["explicit anxiety"]},
    {"id": "p10", "topic": "an empty playground at midnight", "form": "haiku", "must_avoid": ["children laughing"]},
    {"id": "p11", "topic": "a moth circling a porch light", "form": "haiku", "must_avoid": ["death imagery"]},
    {"id": "p12", "topic": "footprints disappearing in fresh snow", "form": "free verse", "must_avoid": ["solitary walker cliché"]},
    {"id": "p13", "topic": "a clock ticking in a hospital waiting room", "form": "free verse", "must_avoid": ["overt grief"]},
    {"id": "p14", "topic": "tide pulling stones across a beach", "form": "haiku", "must_avoid": ["seagulls", "sunset"]},
    {"id": "p15", "topic": "an unread letter on a kitchen table", "form": "free verse", "must_avoid": ["love letter cliché"]},
    {"id": "p16", "topic": "the first cicada of summer at 4 a.m.", "form": "haiku", "must_avoid": ["cliché summer scenes"]},
    {"id": "p17", "topic": "a dog watching its owner pack a suitcase", "form": "free verse", "must_avoid": ["explicit sadness"]},
    {"id": "p18", "topic": "a dropped pencil rolling under a desk", "form": "haiku", "must_avoid": ["school imagery"]},
    {"id": "p19", "topic": "sunlight on the back of a stranger's neck", "form": "free verse", "must_avoid": ["romantic longing"]},
    {"id": "p20", "topic": "a coffee cup forgotten on a windowsill", "form": "haiku", "must_avoid": ["morning routines"]},
]

POETRY_INTERP: list[PoetryInterpItem] = [
    {
        "id": "i01",
        "surface": "I have measured out my life with coffee spoons",
        "aspects": ["a life of small repetitive rituals", "a quiet despair at unlived possibilities"],
        "retrieval_set": ["coffee in the morning is essential", "I drink coffee every day"],
    },
    {
        "id": "i02",
        "surface": "Two roads diverged in a wood, and I— / I took the one less traveled by",
        "aspects": ["a celebration of nonconformity", "an ironic shrug at how memory rewrites choice"],
        "retrieval_set": ["I went hiking in a forest", "trails are well-marked"],
    },
    {
        "id": "i03",
        "surface": "Do not go gentle into that good night",
        "aspects": ["a cry against death", "advice given by a son to a dying father"],
        "retrieval_set": ["bedtime is at 9pm", "good night messages"],
    },
    {
        "id": "i04",
        "surface": "Hope is the thing with feathers that perches in the soul",
        "aspects": ["hope as a delicate persistent bird", "the soul as a place where wild things live"],
        "retrieval_set": ["I saw a sparrow today", "feathers are light"],
    },
    {
        "id": "i05",
        "surface": "Because I could not stop for Death, He kindly stopped for me",
        "aspects": ["death as a courteous gentleman caller", "an inversion of who has agency in the encounter"],
        "retrieval_set": ["traffic stops at red lights", "everyone dies eventually"],
    },
    {
        "id": "i06",
        "surface": "April is the cruellest month, breeding lilacs out of the dead land",
        "aspects": ["spring's renewal as violence against winter's rest", "memory's intrusion on dormancy"],
        "retrieval_set": ["April has 30 days", "lilacs bloom in spring"],
    },
    {
        "id": "i07",
        "surface": "The river is a clock and a clock is a river",
        "aspects": ["time flowing as water", "a clock face that ripples"],
        "retrieval_set": ["clocks tell time", "rivers flow downhill"],
    },
    {
        "id": "i08",
        "surface": "We are such stuff as dreams are made on, and our little life is rounded with a sleep",
        "aspects": ["life as a dream interrupted by waking", "a meta-theatrical farewell from Prospero"],
        "retrieval_set": ["sleep is essential for health", "dreams happen during REM"],
    },
    {
        "id": "i09",
        "surface": "And miles to go before I sleep, and miles to go before I sleep",
        "aspects": ["the literal weariness of a long journey", "the obligations that keep one alive"],
        "retrieval_set": ["sleep regulates mood", "highway distances are measured in miles"],
    },
    {
        "id": "i10",
        "surface": "Shall I compare thee to a summer's day? Thou art more lovely and more temperate",
        "aspects": ["the beloved exceeds nature in beauty", "an ironic acknowledgment that no comparison works"],
        "retrieval_set": ["summer days are warm", "weather varies"],
    },
    {
        "id": "i11",
        "surface": "The fog comes on little cat feet",
        "aspects": ["fog as a small soft animal", "the silence of weather encroaching"],
        "retrieval_set": ["cats are quiet", "fog reduces visibility"],
    },
    {
        "id": "i12",
        "surface": "All the world's a stage, and all the men and women merely players",
        "aspects": ["life as performance", "an existential observation about identity"],
        "retrieval_set": ["theater requires actors", "the world has 8 billion people"],
    },
    {
        "id": "i13",
        "surface": "I wandered lonely as a cloud / That floats on high o'er vales and hills",
        "aspects": ["solitary observation of nature", "the speaker dissolved into the landscape"],
        "retrieval_set": ["clouds form from condensation", "wandering is unstructured walking"],
    },
    {
        "id": "i14",
        "surface": "Stopping by woods on a snowy evening",
        "aspects": ["a literal pause on a winter ride", "a meditation on the temptation of rest"],
        "retrieval_set": ["snow falls in winter", "woods have many trees"],
    },
    {
        "id": "i15",
        "surface": "Tyger Tyger, burning bright, in the forests of the night",
        "aspects": ["a literal predator", "an emblem of God's terrible creativity"],
        "retrieval_set": ["tigers are large cats", "forests are dark at night"],
    },
    {
        "id": "i16",
        "surface": "The woods are lovely, dark and deep",
        "aspects": ["pastoral beauty", "the seductive pull of oblivion"],
        "retrieval_set": ["woods contain trees", "lovely means pleasing"],
    },
    {
        "id": "i17",
        "surface": "Whose woods these are I think I know",
        "aspects": ["identifying a property owner", "claiming intimacy with a place"],
        "retrieval_set": ["land has owners", "thinking is mental activity"],
    },
    {
        "id": "i18",
        "surface": "Out of the cradle endlessly rocking",
        "aspects": ["a lullaby's eternal return", "the sea as origin and end"],
        "retrieval_set": ["cradles are for babies", "rocking is back-and-forth motion"],
    },
    {
        "id": "i19",
        "surface": "I am the master of my fate, I am the captain of my soul",
        "aspects": ["a defiant individualism", "a Stoic resignation dressed as agency"],
        "retrieval_set": ["captains command ships", "fate is the future"],
    },
    {
        "id": "i20",
        "surface": "Let us go then, you and I, when the evening is spread out against the sky",
        "aspects": ["an invitation to a walk", "a self-address by a divided psyche"],
        "retrieval_set": ["evening is between afternoon and night", "sky is overhead"],
    },
]

AUT: list[AUTItem] = [
    {"id": "a01", "object": "brick"},
    {"id": "a02", "object": "paperclip"},
    {"id": "a03", "object": "rubber band"},
    {"id": "a04", "object": "shoebox"},
    {"id": "a05", "object": "wire coat hanger"},
    {"id": "a06", "object": "candle"},
    {"id": "a07", "object": "newspaper"},
    {"id": "a08", "object": "tennis ball"},
    {"id": "a09", "object": "spoon"},
    {"id": "a10", "object": "umbrella"},
    {"id": "a11", "object": "rope"},
    {"id": "a12", "object": "pillowcase"},
    {"id": "a13", "object": "broken mirror shard"},
    {"id": "a14", "object": "shoelace"},
    {"id": "a15", "object": "ice cube tray"},
]

SCI_CREATIVITY: list[SciCreativityItem] = [
    {
        "id": "s01",
        "question": "Why do mass extinctions sometimes accelerate evolutionary innovation?",
        "framings": ["niche-vacuum dynamics", "regulatory-network release"],
    },
    {
        "id": "s02",
        "question": "Why does ice float on water?",
        "framings": ["hydrogen-bond geometry", "thermodynamic stability"],
    },
    {
        "id": "s03",
        "question": "Why are leaves green and not black?",
        "framings": ["photon-absorption efficiency", "evolutionary path-dependence"],
    },
    {
        "id": "s04",
        "question": "Why does sleep exist if it makes us vulnerable?",
        "framings": ["synaptic homeostasis", "memory consolidation", "energy metabolism"],
    },
    {
        "id": "s05",
        "question": "Why do galaxies have spiral arms?",
        "framings": ["density-wave theory", "self-propagating star formation"],
    },
    {
        "id": "s06",
        "question": "Why is mathematics unreasonably effective in physics?",
        "framings": ["selection bias on what we call 'physics'", "an unexplained ontological alignment"],
    },
    {
        "id": "s07",
        "question": "Why do crows recognize individual human faces?",
        "framings": ["co-evolutionary pressure with humans", "general object-individuation cognition"],
    },
    {
        "id": "s08",
        "question": "Why does the placebo effect work even when patients know it's a placebo?",
        "framings": ["expectation conditioning", "ritual and meaning-making"],
    },
    {
        "id": "s09",
        "question": "Why do some neural networks generalize beyond their training distribution?",
        "framings": ["implicit-bias of optimizer", "feature-reuse across distributions"],
    },
    {
        "id": "s10",
        "question": "Why do octopuses appear to have intelligence so different from vertebrates?",
        "framings": ["distributed peripheral cognition", "convergent vs divergent evolutionary architecture"],
    },
    {
        "id": "s11",
        "question": "Why does the entropy of the universe seem to be increasing despite local pockets of order?",
        "framings": ["thermodynamic accounting", "gravitational entropy", "information-processing perspective"],
    },
    {
        "id": "s12",
        "question": "Why do humans have such an enlarged prefrontal cortex compared to other primates?",
        "framings": ["social-brain hypothesis", "tool-use feedback loop", "language coevolution"],
    },
    {
        "id": "s13",
        "question": "Why does antibiotic resistance evolve so quickly?",
        "framings": ["selection pressure intensity", "horizontal gene transfer"],
    },
    {
        "id": "s14",
        "question": "Why are prime numbers distributed irregularly?",
        "framings": ["zeta-function zeros", "sieve heuristics"],
    },
    {
        "id": "s15",
        "question": "Why does anesthesia produce loss of consciousness even though we don't fully understand consciousness?",
        "framings": ["integrated-information disruption", "thalamocortical loop interruption"],
    },
]

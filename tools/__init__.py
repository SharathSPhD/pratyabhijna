"""PCE v0.4 surface validators — chandas, English meter, scientific lint.

These tools are deliberately self-contained: zero scientific-stack imports
(no numpy, no transformers). They run in 50 ms on commodity hardware so
the Astro site can call them in a build step and the Ralph gate stack can
call them in a CI hook. They are what the showcase pages display next to
each cascade trace.
"""

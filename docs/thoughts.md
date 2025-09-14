## Why i built this

Was trying to implement rbergomi for a side project and realized there's basically nothing good out there. academic repos are mostly broken matlab, and the few py implementations take forever to run.

meanwhile.. we have jax and pytorch that can do gpu acceleration and auto-diff.

so i built what i wanted to use..
- actual rough volatility models (rbergomi, rough heston) that work
- gpu acceleration via jax/pytorch backends
- nets for microsecond results
- pathwise greeks through auto-diff
- proper python packaging (cause why not open source this for others to use)

so, to sum it up.. this really just started with me wanting decent tooling for rough vol computations, and ended up being a PyPI package.

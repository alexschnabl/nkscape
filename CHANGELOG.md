# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-11

### Added

- `examples/levinthal_1997`: a notebook reproducing Figures 1 to 9 of Levinthal (1997) with the
  library, with the figures saved alongside it.
- `InteractionMatrix`: an N×N matrix of 0/1 built from user-supplied rows, validated to be
  square, to contain only 0 and 1, and to have an all-1 diagonal.
- `InteractionMatrix.k_per_row` and `InteractionMatrix.k`: K is read from the matrix; `k` raises
  if the rows do not share one K.
- `InteractionMatrix.depends_on(i)` and `InteractionMatrix.affects(j)`: read a row (decisions that
  contribution i depends on) or a column (contributions that change when decision j is flipped).
- `InteractionMatrix.next_k(n, k)` and `InteractionMatrix.random_k(n, k, rng)`: build a matrix in
  which each contribution depends on the next k decisions, or on k randomly chosen others. `rng`
  is a seed or a NumPy generator and is required.
- `Landscape`: an interaction matrix together with its contribution tables. `fitness(config)` is
  the average of `contributions(config)`. Built from existing parts, or with
  `Landscape.random_k(n, k, rng)`, `Landscape.next_k(n, k, rng)` and
  `Landscape.with_random_tables(matrix, rng)`. `random_k` draws the matrix and the tables from
  separate streams spawned from `rng`.
- `Landscape.fitness_cube()`: the fitness of all 2^N configurations as an array with N axes of
  length 2 (`cube[1, 0, 1]` is the fitness of 101), built by broadcasting, computed once,
  read-only, identical bit for bit to `fitness(config)`. Refused above N = 24.
- `nk.Agent`: base class for anything that walks a landscape. Subclass it and write
  `step(config, landscape, rng, population)`, which returns the configuration wanted next - or
  several, best first. Optional: `before_period(...)` runs for every agent before any agent
  steps; `after_period(...)` is told what the agent really ended up with once the period is over;
  `record()` logs extra values per period.
- `nk.run(landscape, agents, periods, rng, starts=None, rules=(), build_cube=None)` runs a
  population in lockstep. Every period: every agent's `before_period`; every agent's `step`, each
  seeing the same read-only `nk.Population` snapshot (everybody's configuration and fitness, who
  is `alive`, and `period`, the period being played); then the `nk.PopulationRule`s, applied one
  after the other, get the proposals and `decide` what is really adopted (without rules, the
  first choice); then every agent's `after_period`. Every agent and every rule has its own random
  stream, so without rules an agent walks the same way alone as in a crowd. The fitness cube is
  built first when the run is long enough to repay it (agents x periods >= 2^N x N / 40,000);
  `build_cube=True/False` overrides that.
- Births and deaths: a rule may return `nk.PopulationDecision(configs, dies=[...], born=[(agent, start)])`
  instead of plain configurations. Both take effect at the end of the period. An agent that dies
  is never called again; a newcomer gets a new row and its own random stream, and is first called
  in the next period. The population may grow without limit, shrink, or die out.
- `nk.RunResult`: `configs[agent, period]`, `fitness[agent, period]`, `alive[agent, period]`,
  `extra` (from the agents' `record()`), `rule_extra` (from the rules' `record()`) and `agents`
  (row a belongs to `agents[a]`). Every agent that ever existed keeps its row; where it did not
  exist the configuration is -1 and the fitness NaN.
- `nk.agents.HillClimber(pick, long_jumps=False)`: the first ready-made agent. Once per period it
  moves to a strictly better neighbour and stays if there is none. `pick` is required:
  `"first"` tries the decisions in order 0, 1, 2, ... and takes the first better neighbour (no
  randomness), `"random"` takes any better neighbour with equal probability, `"best"` takes the
  highest one (exact ties broken at random). With `long_jumps=True` it also draws one
  configuration uniformly at random each period and takes that instead if it is strictly better
  than the neighbour picked. `pick="first"` with `long_jumps=True` is the search of Levinthal
  (1997), checked against the original program.
- `dtype=np.float32` on `ContributionTables` and the `Landscape` builders: tables, contributions,
  fitness and the fitness cube use 4 instead of 8 bytes per number. Numbers are drawn as float64
  and then rounded, so a seed gives the same landscape in either size. Default stays float64.
- `ContributionTables` takes its tables one at a time (any iterable, including a generator), and
  `ContributionTables.random` draws each table only when it is stored. Creating a landscape with
  large tables peaks at about the size of the tables instead of twice that.
- Reproducibility guards (`tests/test_reproducibility.py`): pinned results for fixed seeds, and
  the rule that entry p of table i is a fixed draw of the tables stream (reachable by jumping
  ahead), which any future backend has to reproduce.
- Look-around helpers on `Landscape`: `neighbours(config)`, `neighbour_fitness(config)`,
  `is_peak(config)` and `random_configuration(rng)`. `fitness` and `neighbour_fitness` read from
  the fitness cube when it has been built and look the values up otherwise; both give identical
  results, and the helpers never build the cube themselves.
- `nk.set_enumeration_limit(n)` and `nk.get_enumeration_limit()`: the largest N for which every
  configuration may be looked at (default 24). The refusal states the memory needed and how to
  raise the limit.
- `Landscape.peaks()`, `count_peaks()` and `global_max()`: the local optima (no single-decision
  change improves fitness; ties count), their number, and the highest fitness.
- `ContributionTables.contribution_cube(i)`: contribution i of every configuration, as a
  zero-copy view of the stored table.
- `ContributionTables`: one fitness-contribution table per decision, shaped to fit its row of the
  matrix (axes follow `depends_on`). Built from hand-made tables or with
  `ContributionTables.random(matrix, rng)`; validated and read-only.
- `ContributionTables.contributions(config)`: the N fitness contributions of one configuration
  (any sequence of N zeros and ones).
- The grid of an `InteractionMatrix` is read-only after creation and independent of the array
  it was built from.
- `print(matrix)` shows an `InteractionMatrix` as a labelled grid of `X` and `.` with K at the
  end of each row.

## [0.0.1] - 2026-09-18

### Added

- Project skeleton: packaging, tests, linting, type checking, CI.

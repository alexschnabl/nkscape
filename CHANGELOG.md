# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `InteractionMatrix`: an N×N matrix of 0/1 built from user-supplied rows, validated to be
  square, to contain only 0 and 1, and to have an all-1 diagonal.
- `InteractionMatrix.k_per_row` and `InteractionMatrix.k`: K is read from the matrix; `k` raises
  if the rows do not share one K.
- `InteractionMatrix.depends_on(i)` and `InteractionMatrix.affects(j)`: read a row (decisions that
  contribution i depends on) or a column (contributions that change when decision j is flipped).
- `InteractionMatrix.next_k(n, k)` and `InteractionMatrix.random_k(n, k, rng)`: build a matrix in
  which each contribution depends on the next k decisions, or on k randomly chosen others. `rng`
  is a seed or a NumPy generator and is required.
- `ContributionTables`: one fitness-contribution table per decision, shaped to fit its row of the
  matrix (axes follow `depends_on`). Built from hand-made tables or with
  `ContributionTables.random(matrix, rng)`; validated and read-only.
- The grid of an `InteractionMatrix` is read-only after creation and independent of the array
  it was built from.
- `print(matrix)` shows an `InteractionMatrix` as a labelled grid of `X` and `.` with K at the
  end of each row.

## [0.0.1] - 2026-09-18

### Added

- Project skeleton: packaging, tests, linting, type checking, CI.

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
- `print(matrix)` shows an `InteractionMatrix` as a labelled grid of `X` and `.` with K at the
  end of each row.

## [0.0.1] - 2026-09-18

### Added

- Project skeleton: packaging, tests, linting, type checking, CI.

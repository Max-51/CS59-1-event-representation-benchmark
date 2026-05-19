# Contribution Guide

## Directory Rules

- Put task code under `tasks/{classification,detection,optical_flow}/`.
- Put shared reusable code under `src/`.
- Put new experiment outputs under `artifacts/` task-specific paths.
- Keep old experiment snapshots under `artifacts/archive/`.

## Naming

- Use canonical dataset names: `nmnist`, `ncaltech101`, `cifar10dvs`.
- Aliases (`minist`, `ncar101`, `cifa`, `cifar`) are compatibility-only.

## Legacy Compatibility

If moving an entrypoint, keep a root wrapper to avoid breaking existing commands.

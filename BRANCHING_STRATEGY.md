# AutonomousVehiclePerception/BRANCHING_STRATEGY.md
# GitFlow Branching Strategy

## Branches

| Branch | Purpose | Merges Into |
|---|---|---|
| `main` | Production-ready releases | — |
| `develop` | Integration branch for features | `main` (via release PR) |
| `feature/*` | New features | `develop` (via PR) |
| `bugfix/*` | Bug fixes | `develop` (via PR) |
| `hotfix/*` | Critical production fixes | `main` + `develop` |
| `release/*` | Release preparation | `main` + `develop` |

## Workflow

1. Branch off `develop`: `git checkout develop && git checkout -b feature/my-feature`
2. Develop + commit + push to feature branch
3. Open PR from `feature/*` → `develop`
4. Solo projects: self-merge PRs without approval
5. Team projects: require 1+ review before merge
6. Periodically merge `develop` → `main` via release PR
7. Tag releases on `main`: `git tag -a v1.0.0 -m "Release 1.0.0"`

## Naming Conventions

- `feature/short-description` — new functionality
- `bugfix/issue-number-description` — bug fixes
- `hotfix/critical-fix` — urgent production patches
- `release/v1.0.0` — release candidates

## Commit Messages

Format: `type: description`

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`

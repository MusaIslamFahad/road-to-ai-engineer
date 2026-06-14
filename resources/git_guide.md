# Git & GitHub Guide for ML Engineers

Complete Git reference — from basics to team workflows used in ML projects.

---

## Setup

```bash
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
git config --global core.editor "code --wait"   # VS Code as default editor
git config --global init.defaultBranch main
git config --list                                # Verify config
```

---

## Core Concepts

| Term | What It Means |
|---|---|
| **Repository (repo)** | Folder tracked by Git; contains all files + history |
| **Working directory** | Your local files as they currently are |
| **Staging area (index)** | Changes marked to be included in the next commit |
| **Commit** | A snapshot of staged changes saved to history |
| **Branch** | A lightweight pointer to a specific commit |
| **Remote** | A version of the repo hosted elsewhere (GitHub, GitLab) |
| **HEAD** | Pointer to the current branch / commit you're on |

---

## Daily Workflow

```bash
# ─── Start a new day ─────────────────────────────────────────────────────────
git status                    # What's changed?
git pull origin main          # Get latest from remote

# ─── Make changes ────────────────────────────────────────────────────────────
git add file.py               # Stage specific file
git add src/                  # Stage entire directory
git add -p                    # Interactively stage hunks (recommended)
git add .                     # Stage everything (use carefully)

git status                    # Review what's staged vs. unstaged

git commit -m "feat: add SHAP explainability to model API"
git commit --amend            # Edit the last commit message (before pushing)

# ─── Sync with remote ────────────────────────────────────────────────────────
git push origin feature/shap-api      # Push current branch
git push -u origin feature/shap-api  # First push — sets upstream
```

---

## Branching Strategy for ML Projects

```bash
# ─── Branch naming conventions ───────────────────────────────────────────────
# feature/add-rag-pipeline
# fix/broken-preprocessing-step
# experiment/try-llama3-embeddings
# chore/update-dependencies
# refactor/clean-training-loop

# Create and switch to new branch
git checkout -b feature/add-rag-pipeline   # Create + switch
git switch -c feature/add-rag-pipeline     # Modern syntax (Git 2.23+)

# List branches
git branch          # Local branches
git branch -r       # Remote branches
git branch -a       # All branches

# Switch between branches
git checkout main
git switch main     # Modern syntax

# Delete branches
git branch -d feature/done-branch     # Safe delete (merged only)
git branch -D feature/force-delete    # Force delete
git push origin --delete feature/done-branch  # Delete remote branch
```

---

## Viewing History

```bash
git log                          # Full history
git log --oneline                # Compact — one line per commit
git log --oneline --graph --all  # Visual branch graph
git log --author="Name"          # Filter by author
git log --since="2 weeks ago"    # Filter by time
git log -- path/to/file.py       # History of specific file

git show abc1234                 # Show specific commit
git diff                         # Unstaged changes
git diff --staged                # Staged changes (ready to commit)
git diff main feature/new        # Diff between branches

git blame file.py                # Who changed each line
```

---

## Undoing Mistakes

```bash
# ─── Unstage files (keep changes in working dir) ─────────────────────────────
git restore --staged file.py    # Unstage file (Git 2.23+)
git reset HEAD file.py          # Old syntax

# ─── Discard changes in working directory (IRREVERSIBLE) ─────────────────────
git restore file.py             # Discard local changes to file
git checkout -- file.py         # Old syntax

# ─── Undo last commit (keep changes staged) ──────────────────────────────────
git reset --soft HEAD~1

# ─── Undo last commit (keep changes unstaged) ────────────────────────────────
git reset --mixed HEAD~1        # Default

# ─── Undo last commit (DISCARD changes — IRREVERSIBLE) ───────────────────────
git reset --hard HEAD~1

# ─── Create a new commit that reverses a previous commit (safe for shared branches)
git revert abc1234              # Revert specific commit

# ─── Fix last commit message (before pushing) ────────────────────────────────
git commit --amend -m "Corrected message"

# ─── Temporarily save work (when you need to switch branches) ────────────────
git stash                       # Stash all changes
git stash push -m "WIP: RAG pipeline"
git stash list                  # View stashes
git stash pop                   # Apply latest stash + delete it
git stash apply stash@{1}       # Apply specific stash (keep it)
git stash drop stash@{0}        # Delete specific stash
```

---

## Merging & Rebasing

```bash
# ─── Merge (creates a merge commit — preserves history) ──────────────────────
git checkout main
git merge feature/new-model
git merge --no-ff feature/new-model   # Always create merge commit

# ─── Rebase (rewrites history — cleaner, linear) ─────────────────────────────
git checkout feature/new-model
git rebase main                        # Replay feature commits on top of main

# After rebase: push requires force (because history changed)
git push --force-with-lease            # Safer than --force: fails if remote changed

# ─── Interactive rebase (squash, reorder, edit commits) ──────────────────────
git rebase -i HEAD~3                   # Interactive rebase of last 3 commits
# In the editor: pick → squash (combine), reword (edit message), drop (delete)

# ─── Cherry-pick (apply specific commit to current branch) ───────────────────
git cherry-pick abc1234
```

---

## Working with Remotes

```bash
git remote -v                               # List remotes
git remote add origin https://github.com/user/repo.git
git remote add upstream https://github.com/original/repo.git  # For forks

git fetch origin                            # Download changes (don't merge)
git pull origin main                        # Fetch + merge
git pull --rebase origin main               # Fetch + rebase (cleaner)

git push origin main
git push --force-with-lease                 # Push after rebase (safe force)

# ─── Sync fork with upstream ─────────────────────────────────────────────────
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

---

## GitHub-Specific Workflows

### Pull Request (PR) Workflow

```bash
# 1. Create feature branch
git checkout -b feature/add-monitoring

# 2. Make commits
git add .
git commit -m "feat: add Evidently AI monitoring dashboard"

# 3. Push branch
git push -u origin feature/add-monitoring

# 4. Open PR on GitHub
# - Clear title: "Add Evidently AI monitoring to churn model API"
# - Description: What changed, why, how to test, screenshots
# - Link related issues: "Closes #42"
# - Request specific reviewers

# 5. After review feedback — update the PR
git add .
git commit -m "fix: address code review feedback on dashboard layout"
git push origin feature/add-monitoring

# 6. After approval — merge (prefer squash merge for clean history)
# Squash and Merge on GitHub UI
```

### GitHub Actions Triggers

```yaml
# Trigger CI on PR
on:
  pull_request:
    branches: [main]
    paths:
      - 'src/**'         # Only run if src/ changes
      - 'requirements.txt'

# Trigger on push to main
on:
  push:
    branches: [main]

# Scheduled (daily model retraining)
on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM UTC every day

# Manual trigger
on:
  workflow_dispatch:
    inputs:
      model_version:
        description: 'Model version to deploy'
        required: true
```

---

## Git for ML: DVC Integration

```bash
# Initialize DVC alongside Git
git init
dvc init
git add .dvc/ .gitignore
git commit -m "Initialize DVC"

# Track large data file with DVC (not Git)
dvc add data/train.csv              # Creates data/train.csv.dvc
git add data/train.csv.dvc data/.gitignore
git commit -m "Add training data"

# Push data to remote storage
dvc remote add -d myremote s3://my-bucket/dvc-cache
dvc push                            # Upload data to S3

# Team member pulls data
git clone https://github.com/user/repo
cd repo
dvc pull                            # Download data from S3
```

---

## Commit Message Convention

Use **Conventional Commits** for clear, searchable history:

```
<type>(<scope>): <short description>

[optional body]

[optional footer]
```

| Type | When to Use |
|---|---|
| `feat` | New feature |
| `fix` | Bug fix |
| `chore` | Tooling, dependencies, build |
| `docs` | Documentation only |
| `refactor` | Code restructure (no behavior change) |
| `test` | Adding or fixing tests |
| `perf` | Performance improvement |
| `experiment` | ML experiment (specific to AI projects) |

```bash
# Examples:
git commit -m "feat(rag): add hybrid retrieval with BM25 + dense search"
git commit -m "fix(training): correct validation split to prevent data leakage"
git commit -m "experiment(embeddings): test nomic-embed vs. OpenAI ada-002"
git commit -m "chore: upgrade transformers to 4.38.0"
git commit -m "docs(readme): add QuickStart section with 30-min guide"
```

---

## .gitignore for ML Projects

```
# Python
__pycache__/
*.py[cod]
.venv/
*.egg-info/

# Data (track with DVC instead)
data/raw/
data/processed/
*.csv
*.parquet
*.jsonl

# Models (track with DVC/MLflow)
*.pkl
*.pt
*.onnx
models/

# Experiment tracking
mlruns/
wandb/
.dvc/tmp/

# Jupyter
.ipynb_checkpoints/

# Secrets
.env
*.pem
secrets.yaml

# IDE
.vscode/
.idea/
.DS_Store
```

---

## Quick Reference

```bash
# Most-used commands
git status                     # Current state
git add -p                     # Stage interactively (best practice)
git commit -m "message"        # Commit
git pull --rebase origin main  # Pull cleanly
git push                       # Push

git log --oneline -10          # Last 10 commits
git diff --staged              # Review before committing

git stash / git stash pop      # Temporarily shelve changes
git reset --soft HEAD~1        # Undo last commit, keep changes staged
git revert abc1234             # Safe undo (creates new commit)

# Danger zone
git reset --hard HEAD~1        # Undo + DISCARD changes (irreversible)
git push --force-with-lease    # Only after rebase; never on shared branches
```

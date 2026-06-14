# Contributing to Road to AI Engineer

Thank you for your interest in contributing! This repository is community-driven and we welcome contributions of all kinds.

---

## Code of Conduct

Be respectful, inclusive, and constructive. We're all here to learn and help each other grow.

---

## What We Welcome

### High-Priority Contributions
- **Corrections**: Fix errors in explanations, code, or math
- **Clarity improvements**: Make explanations clearer or more beginner-friendly
- **New examples**: Add code examples that illustrate concepts
- **New exercises**: Add practice problems with solutions
- **Resource additions**: Books, papers, courses, or tools we've missed

### Module Content
- New or improved module guides (`.md` files)
- Additional Jupyter notebooks with worked examples
- Better diagrams or visualizations

### Projects
- New beginner/intermediate/advanced project ideas
- Improved project README files
- Working code solutions for existing projects

### Resources
- New cheatsheets or reference guides
- Career/interview preparation content
- Links to high-quality free learning resources

---

## How to Contribute

### 1. Fork the Repository
```bash
# On GitHub, click "Fork" in the top right
# Then clone your fork:
git clone https://github.com/YOUR_USERNAME/road-to-ai-engineer.git
cd road-to-ai-engineer
```

### 2. Create a Branch
```bash
# Use a descriptive branch name
git checkout -b fix/typo-in-module-05
git checkout -b feature/add-transformer-exercise
git checkout -b improve/clearer-backprop-explanation
```

### 3. Make Your Changes
- Follow the existing file structure and naming conventions
- Match the writing style of the existing content (friendly, practical, example-driven)
- Test all code examples before submitting

### 4. Commit with a Clear Message
```bash
git add .
git commit -m "Fix: correct gradient descent formula in module 09"
git commit -m "Add: new RAG evaluation exercise to module 25"
git commit -m "Improve: clearer explanation of attention mechanism"
```

### 5. Push and Open a Pull Request
```bash
git push origin your-branch-name
# Then go to GitHub and click "Compare & pull request"
```

---

## Content Guidelines

### Writing Style
- **Beginner-friendly first**: Assume the reader is motivated but new to the topic
- **Show, don't just tell**: Always accompany explanations with working code
- **Practical over theoretical**: Favor examples over proofs (link to proofs when needed)
- **Be honest about difficulty**: Don't oversimplify; acknowledge when something is hard

### Code Standards
- Python 3.10+
- Use descriptive variable names (not `x`, `a`, `b` — use `house_price`, `learning_rate`)
- Include comments explaining **why**, not just **what**
- All code must run without errors
- Include expected output as comments where helpful

### Markdown Standards
- Use ATX-style headers (`#`, `##`, `###`)
- Include a table of contents for files longer than 500 lines
- Use fenced code blocks with language identifier (` ```python `)
- Tables for comparisons; bullets for lists; prose for explanations

### File Naming
- Module guides: `topic-name.md` (e.g., `neural-networks.md`)
- Advanced topics: `topic-name-advanced-topics.md`
- Quick references: `topic-name-quick-reference.md`
- Project tutorials: `topic-name-project-tutorial.md`

---

## Reporting Issues

Found a bug, error, or area for improvement?

1. **Check existing issues** first to avoid duplicates
2. **Open an issue** with a clear title and description
3. Use the appropriate label:
   - `bug` — incorrect code or wrong information
   - `enhancement` — suggestions for new content
   - `question` — clarification needed
   - `good first issue` — suitable for new contributors

### Issue Template
```
**Module/File**: [e.g., 09-neural-networks-basics/neural-networks.md]
**Issue type**: [bug / incorrect explanation / missing content]
**Description**: [What's wrong or missing]
**Suggested fix**: [Optional: how you'd fix it]
```

---

## Module Structure Reference

Each module follows this structure:

```
XX-module-name/
├── README.md                          # Overview, learning objectives, prerequisites
├── module-name.md                     # Main guide with explanations and code
├── module-name-advanced-topics.md     # Deeper dives, edge cases, research connections
├── module-name-quick-reference.md     # Cheatsheet / quick lookup
├── module-name-project-tutorial.md    # Step-by-step project walkthrough
└── exercises/
    └── README.md                      # Practice problems with solutions
```

When adding content to a module, match this structure.

---

## Recognition

All contributors are listed in the project's contributor graph on GitHub. Significant contributions may be highlighted in the README.

Thank you for helping make this resource better for everyone on the AI Engineer journey! 🚀

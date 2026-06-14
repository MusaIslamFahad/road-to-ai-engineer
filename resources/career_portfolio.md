# Career & Portfolio Guide for AI Engineers

How to build a compelling AI Engineer portfolio, land your first (or next) role, and grow your career.

---

## Part 1: Your Portfolio Strategy

### What Recruiters Look For (in order)

1. **GitHub with active, real projects** — not tutorials, not empty repos
2. **Deployed demos** — something they can click and actually use
3. **Clear READMEs** — what the project does, why, how, and results
4. **Breadth + depth** — shows you're a Generalist AI Engineer (breadth) with deep knowledge in at least 2–3 areas
5. **Impact framing** — "Reduced churn by 18%" beats "Built a churn model"

### Portfolio Structure (for AI Engineer)

| Project Type | What to Show | Priority |
|---|---|---|
| **Classical ML pipeline** | End-to-end tabular: cleaning → features → model → deployed API | High |
| **Deep Learning (CV or NLP)** | Fine-tuned model or trained from scratch, with eval | High |
| **LLM / GenAI project** | RAG system, AI agent, or fine-tuned LLM with eval | Very High (2025+) |
| **MLOps / Deployment** | Docker + CI/CD + monitoring — at least one project should show this | High |
| **Data analysis / visualization** | Insightful EDA with a real dataset; Streamlit or Plotly dash | Medium |
| **Advanced domain** | RL agent, GNN, or audio project | Optional |

**Minimum viable portfolio**: 3–4 strong projects in the High-priority categories above.

---

## Part 2: Writing Project READMEs That Get Noticed

Every project README should answer:

### Structure

```markdown
# Project Name

One-line tagline that describes what it does and who it's for.

## Problem Statement
What real-world problem does this solve? Why does it matter?

## Demo
[Live demo link] | [Screenshot/GIF]

## Results
- Achieved 94.2% accuracy on CIFAR-10 (vs. 91.5% baseline ResNet-18)
- Latency: <50ms p99 on AWS t3.medium
- Processed 10,000 predictions in production with zero downtime

## Architecture
[Diagram or clear description of how it works end-to-end]

## Tech Stack
Python 3.10 | PyTorch 2.0 | FastAPI | Docker | AWS ECS | MLflow

## How to Run
```bash
git clone ...
pip install -r requirements.txt
uvicorn app:app --reload
```

## Project Structure
[Brief explanation of folder layout]

## Dataset
[Source, size, preprocessing steps]

## Methodology
[Why did you choose this approach? What alternatives did you consider?]

## Future Improvements
[Show you think critically — what would you do with more time/data?]
```

### The "So What?" Rule

Every metric you report should have context:

❌ "Accuracy: 89%"
✅ "Accuracy: 89% on held-out test set (vs. 85% baseline logistic regression)"

❌ "Built a recommendation system"
✅ "Built a hybrid collaborative + content-based recommender; NDCG@10 of 0.73 on MovieLens-1M, outperforming pure MF baseline (0.68)"

---

## Part 3: Deploying Your Projects (Free Options)

| Platform | Best For | Free Tier |
|---|---|---|
| **Hugging Face Spaces** | ML demos, Gradio/Streamlit apps | Free CPU; free GPU (limited) |
| **Streamlit Community Cloud** | Streamlit dashboards | Free for public repos |
| **Railway** | FastAPI/Flask REST APIs | $5/month free credit |
| **Render** | APIs and web services | Free tier (sleeps after idle) |
| **Vercel** | Frontend + serverless | Generous free tier |
| **GitHub Pages** | Static sites, portfolio | Free |
| **Google Colab** | Notebooks with GPU | Free T4 GPU |
| **AWS Free Tier** | EC2, Lambda, SageMaker | 12 months free |

**Recommended stack for a demo-able portfolio project**:
- Frontend: Streamlit or Gradio (deployed to Hugging Face Spaces)
- Backend API: FastAPI on Railway
- Model: loaded from Hugging Face Hub or S3

---

## Part 4: Job Search Strategy

### Where to Apply

| Platform | Best For |
|---|---|
| **LinkedIn Jobs** | General, all company sizes |
| **Levels.fyi** | Big tech, compensation data |
| **Y Combinator Jobs** | Startups, cutting-edge AI work |
| **Greenhouse / Lever** | Direct company career pages |
| **Wellfound (AngelList)** | Startups, equity-focused |
| **ML/AI newsletters** | TheSequence, Import AI, AlphaSignal often list jobs |

### Application Tactics

1. **Apply directly** to company career pages (not via LinkedIn Easy Apply) — higher conversion rate
2. **Network first**: try to get a referral — can 5–10× your interview rate
3. **Tailor your resume** for each role: match keywords from the JD
4. **Follow up**: email recruiter 5–7 days after applying if no response
5. **Track everything**: spreadsheet with company, role, date applied, status, notes

### The Funnel (Realistic Expectations)

```
100 applications → 15–20 recruiter screens → 8–12 technical screens
→ 3–5 virtual onsites → 1–2 offers
```

This is normal. Rejection is information, not failure.

---

## Part 5: Resume Tips

### Format
- 1 page for < 5 years experience; 1–2 pages for senior
- Clean, minimal formatting (PDF)
- No photos, no objectives section
- ATS-friendly: no tables, no text boxes, no columns

### Sections (in order)

1. **Header**: Name, email, GitHub, LinkedIn, portfolio site
2. **Skills**: Grouped by category — avoid listing every library you've touched
3. **Experience**: Reverse chronological; 3–4 bullet points per role
4. **Projects** (if early career or strong portfolio): 2–3 highlighted projects
5. **Education**: Degree, institution, GPA if > 3.5

### Writing Experience Bullets

Use the formula: **[Action verb] + [what you did] + [result/impact]**

❌ "Responsible for building ML models for fraud detection"
✅ "Built XGBoost fraud detection model deployed to 2M daily transactions; reduced false positive rate by 23% vs. rule-based baseline"

❌ "Used LangChain to build a chatbot"
✅ "Designed and deployed RAG-based customer support chatbot; deflected 34% of Tier-1 tickets, saving ~$120K/year"

**Strong action verbs for AI Engineers**:
Architected, Designed, Built, Implemented, Trained, Deployed, Optimized, Reduced, Improved, Automated, Integrated, Led, Mentored, Shipped

---

## Part 6: Building Your Online Presence

### GitHub
- Keep your profile README updated (pinned repos, brief bio)
- Commit consistently — the green squares matter to some recruiters
- Star interesting repos to show your interests
- Contribute to at least 1–2 open-source ML projects

### LinkedIn
- Headline: "AI Engineer | ML, LLMs, GenAI | Python, PyTorch, LangChain" (not just "Software Engineer")
- About section: 3–5 sentences about what you do and what you're building
- Feature your best GitHub repos and demos in the Featured section
- Post 1–2x per week about things you're learning or building

### Writing & Thought Leadership (Optional but Powerful)
- **Blog**: dev.to, Medium, Substack, or your own site
- **Format ideas**: "I rebuilt X from scratch in PyTorch", "Why RAG failed on our production system and how we fixed it", "5 things I learned deploying my first ML model"
- One good technical blog post can generate more recruiter inbound than 50 applications

### Twitter / X
- Follow: @karpathy, @ylecun, @jeremyphoward, @ilyasut, @emollick
- Share your projects, experiments, and learning journey

---

## Part 7: Salary Negotiation

### Research First
- Levels.fyi for total compensation by company/role/level
- Glassdoor for base salary ranges
- Ask people in your network (normalize this!)
- LinkedIn Salary

### Negotiation Principles
1. **Never give a number first** if you can avoid it; ask for their range
2. **Use competing offers** as leverage — even a recruiter screen counts
3. **Negotiate everything**: base, equity, signing bonus, start date, remote policy
4. **Get it in writing** before giving notice anywhere

### Example Script
> "I'm very excited about this opportunity. Based on my research and a competing offer I have, I was hoping we could get to [X]. Is there flexibility there?"

The worst they say is no. Most companies have some flexibility, especially on equity or signing bonus.

---

## Part 8: Continuous Learning & Career Growth

### Staying Current
- **Papers**: arxiv.org (cs.LG, cs.AI, cs.CL) — follow @Papers_with_Code
- **Newsletters**: TheSequence (weekly), Import AI (Brockman), The Batch (DeepLearning.AI)
- **Podcasts**: Lex Fridman, Machine Learning Street Talk, Latent Space
- **YouTube**: Andrej Karpathy, Yannic Kilcher, AI Explained

### Career Ladders (Typical Titles)

```
Junior ML Engineer / ML Engineer I
    ↓ (2–3 years)
ML Engineer II / Mid-level
    ↓ (2–3 years)
Senior ML Engineer
    ↓ (2–4 years, often a fork)
    ├── Staff ML Engineer (IC track)
    │       ↓
    │   Principal / Distinguished
    └── ML Engineering Manager
            ↓
        Director of ML / VP of AI
```

### Specialization vs. Generalization Over Time
- **Years 1–3**: Build breadth as a Generalist AI Engineer (exactly what this repo covers)
- **Years 3–5**: Go deep in 2–3 domains that excite you most AND are marketable
- **Years 5+**: Leverage your combination of breadth + depth; lead technical direction

---

## Resources

| Resource | Use |
|---|---|
| [Chip Huyen - Building a Career in AI](https://huyenchip.com/2018/10/08/career-advice-recent-cs-graduates.html) | Practical career advice from ML expert |
| [Levels.fyi](https://www.levels.fyi/) | Compensation data |
| [ML News Job Board](https://mlnews.dev/jobs) | Curated ML/AI roles |
| [Hugging Face Jobs](https://huggingface.co/jobs) | AI-specific roles |
| [AI Safety / Research Orgs](https://80000hours.org/job-board/) | Research-focused positions |

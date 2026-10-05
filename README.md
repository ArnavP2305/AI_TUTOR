# 🎓 AI Exam Prep Tutor

An AI-powered interactive quiz app built with **LangGraph + Groq + Streamlit**.

Upload your notes or paste text → get a personalised quiz → track weak topics → improve!

---

## 🏗️ Architecture

## 🏗️ Architecture & LangGraph Pipelines

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           LANGGRAPH PIPELINES                            │
├──────────────────┬────────────────────────────────┬──────────────────────┤
│   setup_graph    │           eval_graph           │    report_graph      │
│  ─────────────  │         ──────────────         │  ───────────────────  │
│  extract_topics  │        evaluate_answer         │  identify_weak_      │
│       ↓          │               ↓                │    topics ↓          │
│  generate_       │  🔀 route_difficulty           │  generate_final_     │
│   questions      │      (Conditional Edge)        │    report            │
│  (easy/med/hard) │     ┌─────────┴─────────┐      │                      │
│                  │  [Correct]          [Wrong]    │  (Adaptive Tier      │
│                  │     ↓                  ↓       │   Breakdown)         │
│                  │ level_up_          remedial_   │                      │
│                  │ challenge          guidance    │                      │
└──────────────────┴────────────────────────────────┴──────────────────────┘
              ↑                    ↑                         ↑
         (on upload)        (per question)              (at the end)
              │                    │                         │
┌──────────────────────────────────────────────────────────────────────────┐
│                          STREAMLIT SESSION STATE                         │
│  phase → upload | loading | quiz | answered | report                     │
│  adaptive state: current_difficulty (Easy/Medium/Hard) + trajectory     │
└──────────────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
exam_tutor/
├── app.py              ← Streamlit UI with adaptive quiz runner
├── graph.py            ← 3 LangGraph sub-graphs with Conditional Edges
├── requirements.txt    ← Dependencies
├── .env                ← GROQ_API_KEY
├── sample_notes.txt    ← Biology notes for instant testing
├── README.md
└── .streamlit/
    └── config.toml     ← Dark violet custom theme
```

## 🚀 Local Setup

```bash
# 1. Navigate to project
cd D:\langgraph_learning\exam_tutor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Groq API key in .env
# GROQ_API_KEY=gsk_xxxxxxxxxx

# 4. Run!
streamlit run app.py
```

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this folder to a **GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → select your repo → set `app.py` as the main file
4. In **"Advanced settings" → Secrets**, add:
   ```toml
   GROQ_API_KEY = "gsk_your_actual_key_here"
   ```
5. Click **Deploy** — live in ~2 minutes! 🎉

## 🧠 LangGraph Concepts Used

| Concept | Implementation in Code |
|---|---|
| **TypedDict State** | `SetupState`, `EvalState` (with `current_difficulty`, `next_difficulty`, `adaptive_message`), `ReportState` |
| **Normal Edges** | `START ➔ extract_topics ➔ generate_questions ➔ END` |
| **Conditional Edges** | `add_conditional_edges("evaluate_answer", route_difficulty, {...})` |
| **Branching Subgraphs** | `level_up_challenge` (promotes difficulty) vs. `remedial_guidance` (remedial foundation) |
| **Adaptive Trajectory** | Dynamic question reordering based on runtime graph state updates |

## ✨ Features
- 📄 Upload PDF or paste text directly
- 🎯 Continuous question limit slider (1 to 15 questions)
- 🔀 **Dynamic Adaptive Difficulty**:
  - Automatically assesses student ability tier (`🟢 Easy`, `🟡 Medium`, `🔴 Hard`)
  - Correct answer triggers **Level Up Challenge** node with advanced pro insights
  - Incorrect answer triggers **Remedial Guidance** node with misconception correction
- 📊 Real-time LangGraph routing decision box shown after every submission
- 📈 Full adaptive trajectory visualization and difficulty accuracy matrix in final report
- 🔄 Retry quiz with clean state management

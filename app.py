"""
╔══════════════════════════════════════════════════════════════════╗
║              🎓 AI EXAM PREP TUTOR — STREAMLIT APP              ║
╚══════════════════════════════════════════════════════════════════╝
Run: streamlit run app.py
"""

import streamlit as st
import fitz  # PyMuPDF
from graph import setup_graph, eval_graph, report_graph

# ─────────────────────────────────────────────────────────────────────
# ⚙️  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Exam Prep Tutor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────
# 💅  CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── General ── */
.stApp { max-width: 800px; margin: 0 auto; }

/* ── Progress bar color ── */
.stProgress > div > div { background-color: #6C63FF; }

/* ── Question card ── */
.question-card {
    background: #1A1D27;
    border: 1px solid #2E3250;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}

/* ── Topic badges ── */
.topic-badge {
    display: inline-block;
    background: #6C63FF22;
    border: 1px solid #6C63FF55;
    color: #A09BF8;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.8rem;
    margin: 3px;
}

/* ── Correct / Wrong banners ── */
.correct-banner {
    background: #00C85322;
    border: 1px solid #00C85355;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    color: #00C853;
    font-weight: 600;
}
.wrong-banner {
    background: #FF525222;
    border: 1px solid #FF525255;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    color: #FF5252;
    font-weight: 600;
}

/* ── Difficulty badges ── */
.diff-badge-easy {
    display: inline-block;
    background: #00C85322;
    border: 1px solid #00C85388;
    color: #00E676;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 3px;
}
.diff-badge-medium {
    display: inline-block;
    background: #FFD60022;
    border: 1px solid #FFD60088;
    color: #FFEA00;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 3px;
}
.diff-badge-hard {
    display: inline-block;
    background: #FF174422;
    border: 1px solid #FF174488;
    color: #FF5252;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 3px;
}
.adaptive-decision-box {
    background: #181B26;
    border: 1px solid #6C63FF66;
    border-radius: 10px;
    padding: 1.1rem;
    margin: 1rem 0;
}

/* ── Stat cards ── */
.stat-box {
    background: #1A1D27;
    border: 1px solid #2E3250;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# 🗂️  SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "phase":               "upload",   # upload | loading | quiz | answered | report
        "questions":           [],
        "topics":              [],
        "current_idx":         0,
        "topic_scores":        {},         # {topic: {correct, total}}
        "all_answers":         [],
        "last_eval":           None,       # result of last eval_graph run
        "raw_text":            "",
        "final_report":        "",
        "weak_topics":         [],
        "num_questions":       5,
        "current_difficulty":  "medium",   # starts at medium, adapts via LangGraph conditional edges
        "difficulty_scores":   {
            "easy":   {"correct": 0, "total": 0},
            "medium": {"correct": 0, "total": 0},
            "hard":   {"correct": 0, "total": 0},
        },
        "difficulty_history":  [],         # trajectory of adaptations
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────────────────────────────
# 🛠️  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────
def extract_pdf_text(uploaded_file) -> str:
    """Extract all text from an uploaded PDF file using PyMuPDF."""
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()


def extract_txt_text(uploaded_file) -> str:
    """Extract text from a plain .txt file."""
    return uploaded_file.read().decode("utf-8", errors="ignore")


def get_progress():
    total = len(st.session_state.questions)
    done  = st.session_state.current_idx
    return done, total


def update_score(topic: str, is_correct: bool):
    """Update topic-wise score in session state."""
    if topic not in st.session_state.topic_scores:
        st.session_state.topic_scores[topic] = {"correct": 0, "total": 0}
    st.session_state.topic_scores[topic]["total"] += 1
    if is_correct:
        st.session_state.topic_scores[topic]["correct"] += 1


def reset_app():
    """Full reset to start over."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_state()


# ─────────────────────────────────────────────────────────────────────
# 📄  PHASE 0 — UPLOAD
# ─────────────────────────────────────────────────────────────────────
def show_upload():
    st.markdown("## 🎓 AI Exam Prep Tutor")
    st.markdown("Upload your **notes or textbook chapter** — I'll generate a personalised quiz and track your weak areas.")

    st.markdown("---")

    uploaded = st.file_uploader(
        "📎 Upload your study material",
        type=["pdf", "txt"],
        help="PDF or plain text file — max 50 MB",
    )

    # OR paste text directly
    st.markdown("**— or paste your notes directly —**")
    pasted = st.text_area("✏️ Paste notes here", height=180, placeholder="Paste your lecture notes, chapter summaries, etc...")

    st.markdown("---")

    # Sequential question count slider (1 to 15)
    num_questions = st.slider(
        "🎯 Total Number of Questions",
        min_value=1,
        max_value=15,
        value=int(st.session_state.get("num_questions", 5)),
        step=1,
        help="Select any number of questions between 1 and 15"
    )
    st.session_state["num_questions"] = num_questions

    if st.button("🚀 Generate My Quiz", use_container_width=True, type="primary"):
        if uploaded:
            if uploaded.name.endswith(".pdf"):
                raw_text = extract_pdf_text(uploaded)
            else:
                raw_text = extract_txt_text(uploaded)
        elif pasted.strip():
            raw_text = pasted.strip()
        else:
            st.error("⚠️ Please upload a file or paste your notes first.")
            return

        if len(raw_text) < 100:
            st.error("⚠️ Content too short. Please provide at least a paragraph of study material.")
            return

        st.session_state.raw_text = raw_text
        st.session_state.phase = "loading"
        st.rerun()


# ─────────────────────────────────────────────────────────────────────
# ⏳  PHASE 1 — LOADING (Run LangGraph Setup)
# ─────────────────────────────────────────────────────────────────────
def show_loading():
    st.markdown("## 🧠 Preparing Your Quiz...")

    with st.status("Running AI pipeline...", expanded=True) as status:
        st.write("📖 Extracting topics from your notes...")
        result = setup_graph.invoke({
            "raw_text": st.session_state.raw_text,
            "topics":   [],
            "questions": [],
            "num_questions": st.session_state.get("num_questions", 5),
        })
        st.write(f"✅ Found {len(result['topics'])} topics: **{', '.join(result['topics'])}**")
        st.write(f"✅ Generated {len(result['questions'])} questions")
        status.update(label="Quiz ready! 🎉", state="complete")

    st.session_state.topics    = result["topics"]
    st.session_state.questions = result["questions"]

    # Initialise scores for each topic
    for topic in result["topics"]:
        st.session_state.topic_scores[topic] = {"correct": 0, "total": 0}

    st.session_state.phase = "quiz"
    st.rerun()


# ─────────────────────────────────────────────────────────────────────
# 🎯  PHASE 2 — QUIZ
# ─────────────────────────────────────────────────────────────────────
def show_quiz():
    questions = st.session_state.questions
    idx       = st.session_state.current_idx
    done, total = get_progress()

    # ── Header ──────────────────────────────────────────────
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown(f"### 🎓 Exam Prep Quiz")
    with col2:
        total_correct = sum(s["correct"] for s in st.session_state.topic_scores.values())
        st.metric("✅ Correct", total_correct)
    with col3:
        st.metric("📝 Q Progress", f"{done}/{total}")

    st.progress(done / total if total > 0 else 0)

    # ── Adaptive Engine Status Bar ──────────────────────────
    curr_diff = st.session_state.get("current_difficulty", "medium").lower()
    diff_labels = {
        "easy":   "🟢 Easy (Foundational Review)",
        "medium": "🟡 Medium (Core Proficiency)",
        "hard":   "🔴 Hard (Advanced Mastery)",
    }
    st.caption(f"🧠 **Adaptive Difficulty Engine:** Student Tier is **{diff_labels.get(curr_diff, curr_diff.title())}**")
    st.markdown("---")

    # ── Current question ─────────────────────────────────────
    if idx >= total:
        st.session_state.phase = "report"
        st.rerun()
        return

    q = questions[idx]

    # Badges: Topic, Difficulty, and Type
    q_diff = q.get("difficulty", "medium").lower()
    diff_badge_class = f"diff-badge-{q_diff}"
    diff_label = {"easy": "🟢 Easy", "medium": "🟡 Medium", "hard": "🔴 Hard"}.get(q_diff, q_diff.title())

    st.markdown(
        f'<span class="topic-badge">📚 {q["topic"]}</span> '
        f'<span class="{diff_badge_class}">{diff_label}</span> '
        f'<span class="topic-badge">{"🔵 MCQ" if q["type"] == "mcq" else "✍️ Short Answer"}</span>',
        unsafe_allow_html=True
    )
    st.markdown("")

    # Question text
    st.markdown(f"#### Q{idx+1}. {q['question']}")

    # ── MCQ ────────────────────────────────────────────────
    if q["type"] == "mcq":
        options = q.get("options", ["A. -", "B. -", "C. -", "D. -"])
        choice = st.radio(
            "Select your answer:",
            options=options,
            key=f"mcq_{idx}",
            label_visibility="collapsed",
        )
        student_answer = choice[0] if choice else ""  # Just the letter

        if st.button("✅ Submit Answer", key=f"submit_mcq_{idx}", use_container_width=True, type="primary"):
            _run_evaluation(q, student_answer)

    # ── Short Answer ────────────────────────────────────────
    else:
        student_answer = st.text_area(
            "Your answer:",
            key=f"sa_{idx}",
            height=120,
            placeholder="Write your answer here...",
        )
        if st.button("✅ Submit Answer", key=f"submit_sa_{idx}", use_container_width=True, type="primary"):
            if not student_answer.strip():
                st.warning("⚠️ Please write something before submitting.")
                return
            _run_evaluation(q, student_answer)


def _run_evaluation(q: dict, student_answer: str):
    """Invoke eval_graph with conditional edges and store adaptive outcome."""
    with st.spinner("Evaluating & adapting difficulty via LangGraph..."):
        result = eval_graph.invoke({
            "question":           q,
            "student_answer":     student_answer,
            "is_correct":         False,
            "score":              0,
            "feedback":           "",
            "correct_answer":     "",
            "current_difficulty": st.session_state.get("current_difficulty", "medium"),
            "next_difficulty":    "",
            "adaptive_message":   "",
            "concept_hint":       "",
        })

    # Update topic scores
    update_score(q["topic"], result["is_correct"])

    # Update difficulty scores
    q_diff = q.get("difficulty", "medium").lower()
    if q_diff not in st.session_state.difficulty_scores:
        st.session_state.difficulty_scores[q_diff] = {"correct": 0, "total": 0}
    st.session_state.difficulty_scores[q_diff]["total"] += 1
    if result["is_correct"]:
        st.session_state.difficulty_scores[q_diff]["correct"] += 1

    # Record in history
    st.session_state.difficulty_history.append({
        "qid":       q["id"],
        "topic":     q["topic"],
        "q_diff":    q_diff,
        "is_correct": result["is_correct"],
        "from_tier": st.session_state.get("current_difficulty", "medium"),
        "to_tier":   result.get("next_difficulty", "medium"),
    })

    # Save to answer history
    st.session_state.all_answers.append({
        "question_id":      q["id"],
        "topic":            q["topic"],
        "type":             q["type"],
        "difficulty":       q_diff,
        "question":         q["question"],
        "student_answer":   student_answer,
        "correct_answer":   result["correct_answer"],
        "is_correct":       result["is_correct"],
        "score":            result["score"],
        "feedback":         result["feedback"],
        "adaptive_message": result.get("adaptive_message", ""),
        "concept_hint":     result.get("concept_hint", ""),
    })

    st.session_state.last_eval = result
    st.session_state.phase = "answered"
    st.rerun()


# ─────────────────────────────────────────────────────────────────────
# ✅  PHASE 2b — SHOW ANSWER FEEDBACK
# ─────────────────────────────────────────────────────────────────────
def show_answered():
    idx  = st.session_state.current_idx
    q    = st.session_state.questions[idx]
    eval_result = st.session_state.last_eval
    done, total = get_progress()

    # Progress
    st.markdown(f"### 🎓 Exam Prep Quiz")
    st.progress(done / total if total > 0 else 0)
    st.markdown("---")

    # Badges
    q_diff = q.get("difficulty", "medium").lower()
    diff_badge_class = f"diff-badge-{q_diff}"
    diff_label = {"easy": "🟢 Easy", "medium": "🟡 Medium", "hard": "🔴 Hard"}.get(q_diff, q_diff.title())

    st.markdown(
        f'<span class="topic-badge">📚 {q["topic"]}</span> '
        f'<span class="{diff_badge_class}">{diff_label}</span>',
        unsafe_allow_html=True
    )
    st.markdown("")
    st.markdown(f"#### Q{idx+1}. {q['question']}")

    # Result banner
    if eval_result["is_correct"]:
        st.markdown('<div class="correct-banner">✅ Correct! Well done.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="wrong-banner">❌ Incorrect.</div>', unsafe_allow_html=True)
        st.info(f"📌 **Correct Answer:** {eval_result['correct_answer']}")

    # Feedback
    st.markdown("**💬 Feedback:**")
    st.markdown(f"> {eval_result['feedback']}")

    # Short answer score
    if q["type"] == "short_answer":
        score = eval_result.get("score", 0)
        color = "#00C853" if score >= 60 else "#FF5252"
        st.markdown(f"**Score:** <span style='color:{color}; font-size:1.2rem; font-weight:bold'>{score}/100</span>", unsafe_allow_html=True)

    # 🔀 LangGraph Adaptive Routing Decision Box
    route_name = "level_up_challenge" if eval_result["is_correct"] else "remedial_guidance"
    next_tier = eval_result.get("next_difficulty", "medium").upper()
    hint = eval_result.get("concept_hint", "")

    st.markdown(f"""
    <div class="adaptive-decision-box">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
        <span style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em; color:#A09BF8; font-weight:700;">
          🔀 LangGraph Conditional Edge ➔ Node: <code>{route_name}</code>
        </span>
        <span style="font-size:0.8rem; font-weight:bold; color:#FAFAFA;">
          Next Tier: {next_tier}
        </span>
      </div>
      <div style="font-size:0.92rem; margin-bottom:8px; color:#EEE;">
        {eval_result.get('adaptive_message', '')}
      </div>
      <div style="background:#13151D; border-left:3px solid #6C63FF; padding:8px 12px; border-radius:4px; font-size:0.85rem; color:#DDD;">
        <strong>💡 AI Tutor Insight:</strong> {hint}
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Navigation buttons
    col1, col2 = st.columns(2)
    is_last = (idx + 1 >= len(st.session_state.questions))

    with col1:
        if st.button("⬅️ Review Question", use_container_width=True):
            st.session_state.phase = "quiz"
            st.rerun()

    with col2:
        if is_last:
            if st.button("📊 See Final Report", use_container_width=True, type="primary"):
                st.session_state.current_idx += 1
                st.session_state.phase = "report"
                st.rerun()
        else:
            if st.button("➡️ Next Question", use_container_width=True, type="primary"):
                # Update current_difficulty to the adapted next_difficulty
                target_diff = eval_result.get("next_difficulty", st.session_state.current_difficulty).lower()
                st.session_state.current_difficulty = target_diff

                # Adaptively swap in an unattempted question matching target_diff if available
                next_pos = idx + 1
                questions = st.session_state.questions
                match_pos = None
                for cand_pos in range(next_pos, len(questions)):
                    if questions[cand_pos].get("difficulty", "medium").lower() == target_diff:
                        match_pos = cand_pos
                        break

                if match_pos is not None and match_pos != next_pos:
                    questions[next_pos], questions[match_pos] = questions[match_pos], questions[next_pos]

                st.session_state.current_idx += 1
                st.session_state.phase = "quiz"
                st.rerun()


# ─────────────────────────────────────────────────────────────────────
# 📊  PHASE 3 — REPORT
# ─────────────────────────────────────────────────────────────────────
def show_report():
    # Run report_graph once
    if not st.session_state.final_report:
        with st.spinner("📊 Generating your performance report..."):
            result = report_graph.invoke({
                "topic_scores":      st.session_state.topic_scores,
                "difficulty_scores": st.session_state.difficulty_scores,
                "all_answers":       st.session_state.all_answers,
                "weak_topics":       [],
                "final_report":      "",
            })
        st.session_state.final_report = result["final_report"]
        st.session_state.weak_topics  = result.get("weak_topics", [])

    # ── Display report ─────────────────────────────────────────
    st.markdown(st.session_state.final_report)

    # ── Adaptive Trajectory Visualization ──────────────────────
    if st.session_state.difficulty_history:
        st.markdown("### 🔀 Adaptive Learning Trajectory")
        st.caption("How LangGraph conditional edges dynamically adapted your test experience in real-time:")
        traj_items = []
        for step in st.session_state.difficulty_history:
            icon = "✅" if step["is_correct"] else "❌"
            tier_badge = {"easy": "🟢 Easy", "medium": "🟡 Med", "hard": "🔴 Hard"}.get(step["q_diff"], step["q_diff"])
            next_badge = {"easy": "🟢 Easy", "medium": "🟡 Med", "hard": "🔴 Hard"}.get(step["to_tier"], step["to_tier"])
            traj_items.append(f"Q{step['qid']} ({tier_badge}) {icon} ➔ {next_badge}")
        st.info(" ➔ ".join(traj_items))

    st.markdown("---")

    # ── Wrong answers review ────────────────────────────────────
    wrong = [a for a in st.session_state.all_answers if not a["is_correct"]]
    if wrong:
        with st.expander(f"🔍 Review {len(wrong)} Incorrect Answer(s)"):
            for a in wrong:
                q_d = a.get("difficulty", "medium").upper()
                st.markdown(f"**Q:** {a['question']} `[{q_d}]`")
                st.markdown(f"- 🔴 Your answer: `{a['student_answer']}`")
                st.markdown(f"- ✅ Correct answer: `{a['correct_answer']}`")
                st.markdown(f"- 💬 {a['feedback']}")
                if a.get("concept_hint"):
                    st.markdown(f"- 💡 *Remedial Advice:* {a['concept_hint']}")
                st.markdown("---")

    # ── Action buttons ──────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Retry Quiz", use_container_width=True, type="primary"):
            questions = st.session_state.questions
            topics    = st.session_state.topics
            raw_text  = st.session_state.raw_text
            reset_app()
            st.session_state.questions = questions
            st.session_state.topics    = topics
            st.session_state.raw_text  = raw_text
            for t in topics:
                st.session_state.topic_scores[t] = {"correct": 0, "total": 0}
            st.session_state.phase = "quiz"
            st.rerun()
    with col2:
        if st.button("📄 Upload New Notes", use_container_width=True):
            reset_app()
            st.rerun()


# ─────────────────────────────────────────────────────────────────────
# 🚦  MAIN ROUTER
# ─────────────────────────────────────────────────────────────────────
phase = st.session_state.phase

if phase == "upload":
    show_upload()
elif phase == "loading":
    show_loading()
elif phase == "quiz":
    show_quiz()
elif phase == "answered":
    show_answered()
elif phase == "report":
    show_report()

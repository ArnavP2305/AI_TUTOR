"""
╔══════════════════════════════════════════════════════════════════╗
║              🧠 AI EXAM PREP TUTOR — LANGGRAPH PIPELINE         ║
╠══════════════════════════════════════════════════════════════════╣
║  3 Sub-Graphs:                                                    ║
║  1. setup_graph   → PDF text → Topics → Questions                 ║
║  2. eval_graph    → Question + Student Answer → Evaluation        ║
║  3. report_graph  → All Scores → Final Performance Report         ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import json
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

load_dotenv()

# ─────────────────────────────────────────────────────────────────────
# 🤖  LLM SETUP
# ─────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
)


# ─────────────────────────────────────────────────────────────────────
# 🗂️  STATE SCHEMAS
# ─────────────────────────────────────────────────────────────────────

class SetupState(TypedDict):
    """State for Graph 1: PDF processing & question generation."""
    raw_text: str           # Full extracted PDF text
    topics: List[str]       # Extracted topic list
    questions: List[dict]   # Generated question bank
    num_questions: int      # Target question count (1 to 15)


class EvalState(TypedDict):
    """State for Graph 2: Per-question answer evaluation with dynamic adaptive routing."""
    question: dict          # The question dict (includes difficulty: easy/medium/hard)
    student_answer: str     # What the student answered
    is_correct: bool        # Pass/Fail
    score: int              # 0-100 score for short answers
    feedback: str           # Detailed LLM feedback
    correct_answer: str     # The right answer (to show student)
    current_difficulty: str # Current difficulty level: "easy", "medium", "hard"
    next_difficulty: str    # Adjusted difficulty after conditional routing: "easy", "medium", "hard"
    adaptive_message: str   # Reason for level up or remedial calibration
    concept_hint: str       # Targeted remedial hint or pro tip


class ReportState(TypedDict):
    """State for Graph 3: Final performance report generation."""
    topic_scores: dict      # {topic: {"correct": int, "total": int}}
    difficulty_scores: dict # {difficulty: {"correct": int, "total": int}}
    all_answers: List[dict] # Full Q&A history
    weak_topics: List[str]  # Topics scoring below 60%
    final_report: str       # Formatted markdown report


# ─────────────────────────────────────────────────────────────────────
# ⚙️  GRAPH 1 NODES — Setup (PDF → Questions)
# ─────────────────────────────────────────────────────────────────────

def extract_topics(state: SetupState) -> dict:
    """
    Node 1: Reads raw PDF text → identifies main topics/chapters.
    Returns a list of topic strings.
    """
    prompt = f"""Analyze this educational content and extract the MAIN topics or chapters.
Return ONLY a JSON array of topic names (2 to 5 topics).
Example: ["Photosynthesis", "Cell Division", "DNA Structure"]

Content:
{state['raw_text'][:3500]}

Return ONLY the JSON array, no explanation, no markdown:"""

    response = llm.invoke(prompt)
    try:
        text = _clean_json(response.content)
        topics = json.loads(text)
        if not isinstance(topics, list) or len(topics) == 0:
            raise ValueError("Not a list")
        # Limit to 5 topics max
        topics = [str(t) for t in topics[:5]]
    except Exception:
        topics = ["Main Concepts", "Key Principles"]

    return {"topics": topics}


def generate_questions(state: SetupState) -> dict:
    """
    Node 2: Generates the requested number of questions (1 to 15),
    distributed evenly across topics, each tagged with difficulty (easy, medium, hard).
    """
    all_questions = []
    target_total = state.get("num_questions", 5)
    topics = state.get("topics", ["Main Concepts"])
    if not topics:
        topics = ["Main Concepts"]

    num_topics = len(topics)
    base_per_topic = target_total // num_topics
    remainder = target_total % num_topics

    difficulties = ["easy", "medium", "hard"]

    qid = 1
    for i, topic in enumerate(topics):
        count_for_topic = base_per_topic + (1 if i < remainder else 0)
        if count_for_topic <= 0:
            continue

        prompt = f"""You are an expert exam paper setter. Create exactly {count_for_topic} questions about "{topic}".
Use ONLY information from this content:
{state['raw_text'][:4000]}

Format — return ONLY a JSON array with exactly {count_for_topic} question objects:
[
  {{
    "topic": "{topic}",
    "type": "mcq",
    "difficulty": "easy",
    "question": "A clear question about {topic}?",
    "options": ["A. Option 1", "B. Option 2", "C. Option 3", "D. Option 4"],
    "correct_answer": "A",
    "explanation": "Why this answer is correct."
  }}
]

Requirements:
- Total questions in array must be exactly {count_for_topic}.
- Include a mix of 'mcq' (with 4 options and correct letter) and 'short_answer' (with expected correct_answer and explanation).
- Assign a 'difficulty' level to each question:
  * "easy" (fundamental facts, basic definitions)
  * "medium" (conceptual understanding, application)
  * "hard" (in-depth analysis, multi-step reasoning)
- Provide a balanced spread across easy, medium, and hard difficulty tiers.
- Return ONLY valid JSON array, no conversational text, no markdown backticks."""

        response = llm.invoke(prompt)
        try:
            text = _clean_json(response.content)
            questions = json.loads(text)
            if not isinstance(questions, list):
                raise ValueError
            for q in questions:
                q["id"] = qid
                if q.get("difficulty") not in ["easy", "medium", "hard"]:
                    q["difficulty"] = difficulties[(qid - 1) % len(difficulties)]
                all_questions.append(q)
                qid += 1
        except Exception:
            # Fallback questions if LLM response format fails
            for _ in range(count_for_topic):
                diff = difficulties[(qid - 1) % len(difficulties)]
                all_questions.append({
                    "id": qid,
                    "topic": topic,
                    "type": "mcq",
                    "difficulty": diff,
                    "question": f"What is a core principle of {topic}?",
                    "options": [f"A. Fundamental concept of {topic}", "B. Secondary aspect", "C. Unrelated concept", "D. None of the above"],
                    "correct_answer": "A",
                    "explanation": f"Key principle of {topic}.",
                })
                qid += 1

    # Guarantee exactly target_total questions and sequential numbering 1..target_total
    all_questions = all_questions[:target_total]
    for idx, q in enumerate(all_questions, 1):
        q["id"] = idx
        if q.get("difficulty") not in ["easy", "medium", "hard"]:
            q["difficulty"] = difficulties[(idx - 1) % len(difficulties)]

    return {"questions": all_questions}


# ─────────────────────────────────────────────────────────────────────
# ⚙️  GRAPH 2 NODES — Evaluate (Answer → Feedback)
# ─────────────────────────────────────────────────────────────────────

def evaluate_answer(state: EvalState) -> dict:
    """
    Node: Evaluates student's answer.
    - MCQ: exact letter match
    - Short answer: LLM semantic evaluation (0-100 score)
    """
    q = state["question"]
    student_ans = state["student_answer"].strip()

    if q["type"] == "mcq":
        # Extract just the letter from student's choice
        letter = student_ans.upper()
        if len(letter) > 1:
            letter = letter[0]
        correct_letter = q["correct_answer"].strip().upper()
        if len(correct_letter) > 1:
            correct_letter = correct_letter[0]

        is_correct = letter == correct_letter
        score = 100 if is_correct else 0
        feedback = q.get("explanation", "Review the correct answer.")
        correct_answer = q["correct_answer"]

    else:  # short_answer
        prompt = f"""You are a strict but fair exam evaluator.

Question: {q['question']}
Expected Answer: {q['correct_answer']}
Key Points to Cover: {q.get('explanation', 'Main concepts of the topic')}
Student Answer: {student_ans}

Evaluate the student's answer. Return ONLY this JSON:
{{
  "score": <integer 0-100>,
  "is_correct": <true if score >= 60 else false>,
  "feedback": "2-3 sentence specific feedback: what was correct, what was missing."
}}"""

        response = llm.invoke(prompt)
        try:
            result = json.loads(_clean_json(response.content))
            score = int(result.get("score", 0))
            is_correct = score >= 60
            feedback = result.get("feedback", "Review the expected answer.")
        except Exception:
            score = 0
            is_correct = False
            feedback = "Unable to auto-evaluate. Please review the expected answer manually."
        correct_answer = q["correct_answer"]

    return {
        "is_correct": is_correct,
        "score": score,
        "feedback": feedback,
        "correct_answer": correct_answer,
    }


def route_difficulty(state: EvalState) -> str:
    """
    🔀 LANGGRAPH CONDITIONAL EDGE ROUTER
    Inspects whether the student answered correctly:
    - True  ➔ routes to 'level_up_challenge' (promotes difficulty tier)
    - False ➔ routes to 'remedial_guidance'  (calibrates down/reinforces)
    """
    if state.get("is_correct", False):
        return "level_up_challenge"
    return "remedial_guidance"


def level_up_challenge(state: EvalState) -> dict:
    """
    Adaptive Node A: Promotes student difficulty tier and generates an advanced pro tip.
    """
    curr = state.get("current_difficulty", "medium").lower()

    if curr == "easy":
        next_diff = "medium"
        msg = "🟢 ➔ 🟡 **Level Up!** You mastered the core fundamentals. Promoting to **Medium Difficulty**."
    elif curr == "medium":
        next_diff = "hard"
        msg = "🟡 ➔ 🔴 **Outstanding!** Demonstrating strong proficiency. Promoting to **Hard / Advanced Difficulty**."
    else:
        next_diff = "hard"
        msg = "🔴 **Mastery Confirmed!** You solved an Advanced level problem with precision. Retaining **Hard Difficulty**."

    topic = state.get("question", {}).get("topic", "the concept")
    prompt = f"""Provide a concise 1-sentence advanced 'Pro Exam Tip' or deep-dive takeaway about {topic} for an exam candidate who just answered correctly:
Question: {state.get('question', {}).get('question', '')}
Correct Answer: {state.get('correct_answer', '')}

Return ONLY the 1 sentence tip:"""
    try:
        resp = llm.invoke(prompt)
        tip = resp.content.strip()
    except Exception:
        tip = f"Pro Tip: Focus on how {topic} connects with related biological processes and multi-step mechanisms."

    return {
        "next_difficulty": next_diff,
        "adaptive_message": msg,
        "concept_hint": tip,
    }


def remedial_guidance(state: EvalState) -> dict:
    """
    Adaptive Node B: Calibrates difficulty down or reinforces fundamentals
    with targeted remedial advice explaining the exact misconception.
    """
    curr = state.get("current_difficulty", "medium").lower()

    if curr == "hard":
        next_diff = "medium"
        msg = "🔴 ➔ 🟡 **Difficulty Calibrated:** Shifting back to **Medium** to consolidate concepts before advancing."
    elif curr == "medium":
        next_diff = "easy"
        msg = "🟡 ➔ 🟢 **Foundation Check:** Stepping down to **Easy / Core Concept** to lock in foundational principles."
    else:
        next_diff = "easy"
        msg = "🟢 **Core Reinforcement:** Staying at **Easy** to build confidence and reinforce fundamental facts."

    topic = state.get("question", {}).get("topic", "the topic")
    prompt = f"""The student answered this question incorrectly.
Question: {state.get('question', {}).get('question', '')}
Student Answer: {state.get('student_answer', '')}
Correct Answer: {state.get('correct_answer', '')}
Topic: {topic}

Provide a concise 1-2 sentence remedial hint clarifying the exact misconception or key fact they need to remember.
Return ONLY the hint:"""
    try:
        resp = llm.invoke(prompt)
        hint = resp.content.strip()
    except Exception:
        hint = f"Review the essential definitions and key steps of {topic} before re-attempting."

    return {
        "next_difficulty": next_diff,
        "adaptive_message": msg,
        "concept_hint": hint,
    }


# ─────────────────────────────────────────────────────────────────────
# ⚙️  GRAPH 3 NODES — Report (Scores → Final Report)
# ─────────────────────────────────────────────────────────────────────

def identify_weak_topics(state: ReportState) -> dict:
    """Node 1: Flag any topic scoring below 60%."""
    weak = [
        topic
        for topic, s in state["topic_scores"].items()
        if s["total"] > 0 and (s["correct"] / s["total"]) < 0.6
    ]
    return {"weak_topics": weak}


def generate_final_report(state: ReportState) -> dict:
    """Node 2: Build a rich markdown performance report with adaptive breakdown."""
    topic_scores = state["topic_scores"]
    total_correct = sum(s["correct"] for s in topic_scores.values())
    total_q = sum(s["total"] for s in topic_scores.values())
    pct = (total_correct / total_q * 100) if total_q > 0 else 0

    # Grade
    if pct >= 85:
        grade, badge = "A+", "🏆 Outstanding!"
    elif pct >= 70:
        grade, badge = "A", "🎉 Excellent!"
    elif pct >= 55:
        grade, badge = "B", "👍 Good Work!"
    elif pct >= 40:
        grade, badge = "C", "📚 Needs Improvement"
    else:
        grade, badge = "D", "🔄 Revise Thoroughly"

    # Topic rows
    topic_rows = []
    for topic, s in topic_scores.items():
        if s["total"] > 0:
            tp = s["correct"] / s["total"] * 100
            bar = "🟩" * int(tp // 20) + "⬜" * (5 - int(tp // 20))
            topic_rows.append(
                f"| {topic} | {s['correct']}/{s['total']} | {tp:.0f}% | {bar} |"
            )

    # Adaptive Difficulty Breakdown
    diff_scores = state.get("difficulty_scores", {})
    diff_rows = []
    diff_icons = {"easy": "🟢 Easy", "medium": "🟡 Medium", "hard": "🔴 Hard"}
    for diff_key in ["easy", "medium", "hard"]:
        ds = diff_scores.get(diff_key, {"correct": 0, "total": 0})
        if ds["total"] > 0:
            dp = (ds["correct"] / ds["total"] * 100)
            diff_rows.append(f"| {diff_icons[diff_key]} | {ds['correct']}/{ds['total']} | {dp:.0f}% |")

    diff_table = ""
    if diff_rows:
        diff_table = f"""
## 🎚️ Adaptive Difficulty Performance

| Difficulty Tier | Score | Accuracy |
|-----------------|-------|----------|
{chr(10).join(diff_rows)}
"""

    weak_section = ""
    if state["weak_topics"]:
        weak_list = "\n".join(f"- 🔴 **{t}** — spend extra time here" for t in state["weak_topics"])
        weak_section = f"\n## 📌 Topics to Revise\n{weak_list}\n"

    report = f"""# 📊 Exam Prep Performance Report

## 🎯 Overall Score: {total_correct}/{total_q} ({pct:.1f}%) — Grade {grade}
### {badge}

---

## 📈 Topic-wise Breakdown

| Topic | Score | % | Progress |
|-------|-------|---|----------|
{chr(10).join(topic_rows)}
{diff_table}
{weak_section}
---

## 💡 Recommended Next Steps
{"- **Keep it up!** Review any questions you got wrong to solidify your understanding." if pct >= 70 else "- **Focus on weak topics first.** Spend 30 minutes on each red topic before retrying."}
- Re-attempt this quiz in **24 hours** — spaced repetition boosts retention by 40%.
- For wrong answers, re-read that section of your notes carefully.
- {"You're exam-ready! 🚀" if pct >= 80 else "Consistent practice = exam success. You've got this! 💪"}
"""

    return {"final_report": report}


# ─────────────────────────────────────────────────────────────────────
# 🔧  HELPER
# ─────────────────────────────────────────────────────────────────────

def _clean_json(text: str) -> str:
    """Strip markdown code fences from LLM JSON responses."""
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        # Take the content between first pair of ```
        text = parts[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return text


# ─────────────────────────────────────────────────────────────────────
# 🏗️  BUILD & COMPILE GRAPHS
# ─────────────────────────────────────────────────────────────────────

def _build_setup_graph():
    g = StateGraph(SetupState)
    g.add_node("extract_topics",    extract_topics)
    g.add_node("generate_questions", generate_questions)
    g.add_edge(START,               "extract_topics")
    g.add_edge("extract_topics",    "generate_questions")
    g.add_edge("generate_questions", END)
    return g.compile()


def _build_eval_graph():
    g = StateGraph(EvalState)
    g.add_node("evaluate_answer",    evaluate_answer)
    g.add_node("level_up_challenge", level_up_challenge)
    g.add_node("remedial_guidance",  remedial_guidance)

    g.add_edge(START, "evaluate_answer")

    # 🔀 CONDITIONAL EDGE: Dynamically routes based on student performance
    g.add_conditional_edges(
        "evaluate_answer",
        route_difficulty,
        {
            "level_up_challenge": "level_up_challenge",
            "remedial_guidance":  "remedial_guidance",
        }
    )

    g.add_edge("level_up_challenge", END)
    g.add_edge("remedial_guidance",  END)
    return g.compile()


def _build_report_graph():
    g = StateGraph(ReportState)
    g.add_node("identify_weak_topics",  identify_weak_topics)
    g.add_node("generate_final_report", generate_final_report)
    g.add_edge(START,                   "identify_weak_topics")
    g.add_edge("identify_weak_topics",  "generate_final_report")
    g.add_edge("generate_final_report", END)
    return g.compile()


# ── Pre-compiled instances (imported by app.py) ────────────────────
setup_graph  = _build_setup_graph()
eval_graph   = _build_eval_graph()
report_graph = _build_report_graph()

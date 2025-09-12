import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# ======================
# Setup
# ======================

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

UNIFIED_GRAPH_PATH = OUTPUTS_DIR / "phase4_unified.json"
STUDENT_MODEL_PATH = OUTPUTS_DIR / "student_model.json"
CURRICULUM_PLAN_PATH = OUTPUTS_DIR / "curriculum_plan.json"


# ======================
# OpenAI Call Helper
# ======================

def call_openai_json(prompt, max_retries=2):
    """
    Calls OpenAI with a prompt, retries if JSON is malformed.
    Falls back to dummy JSON if still invalid.
    """
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful teaching assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )

            # FIX: use .content, not ["content"]
            content = response.choices[0].message.content.strip()

            # Strip code fences if present
            if content.startswith("```"):
                content = "\n".join(content.split("\n")[1:-1])

            return json.loads(content)

        except Exception as e:
            if attempt < max_retries:
                print(f"⚠️ JSON parsing failed, retrying... (attempt {attempt+1})")
                continue
            else:
                # Final fallback
                print("⚠️ Falling back to dummy JSON.")
                return {
                    "explanation": "Content unavailable due to parsing error.",
                    "quiz": [
                        {
                            "question": f"What does the concept '{prompt.split()[-1]}' relate to?",
                            "options": ["Option A", "Option B", "Option C"],
                            "answer": "Option A"
                        }
                    ]
                }


# ======================
# Core Functions
# ======================

def load_unified_graph():
    with open(UNIFIED_GRAPH_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_or_init_student_model():
    if STUDENT_MODEL_PATH.exists():
        with open(STUDENT_MODEL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_student_model(student_model):
    with open(STUDENT_MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(student_model, f, indent=2)


def update_student_model(student_model, student_id, concept, score):
    """
    Update mastery score for a concept based on quiz result.
    """
    if student_id not in student_model:
        student_model[student_id] = {}

    prev_score = student_model[student_id].get(concept, 0)
    new_score = min(1.0, (prev_score + score) / 2)  # smoothing
    student_model[student_id][concept] = new_score

    return student_model


def generate_next_concepts(unified_graph, student_model, student_id, max_concepts=2):
    """
    Select next concepts for the student.
    """
    taught = set(student_model.get(student_id, {}).keys())
    candidates = [c["text"] for c in unified_graph if c.get("class") == "concept"]

    next_concepts = [c for c in candidates if c not in taught]
    return next_concepts[:max_concepts]


def generate_lesson(concept):
    """
    Use OpenAI to generate a structured lesson for a concept.
    """
    prompt = f"""
    Generate a short lesson for the concept '{concept}'.
    Return JSON with the following keys:
    - explanation: a clear and simple explanation
    - quiz: a list of MCQs with fields 'question', 'options', and 'answer'
    """

    return call_openai_json(prompt)


# ======================
# Main
# ======================

def main(student_id, max_concepts=2):
    unified_graph = load_unified_graph()
    student_model = load_or_init_student_model()

    # Decide next concepts
    next_concepts = generate_next_concepts(unified_graph, student_model, student_id, max_concepts)

    curriculum_plan = {}
    for concept in next_concepts:
        lesson = generate_lesson(concept)

        # Log curriculum plan
        curriculum_plan[concept] = lesson

        # Dummy: assume student scores 0.7
        student_model = update_student_model(student_model, student_id, concept, score=0.7)

    # Save outputs
    with open(CURRICULUM_PLAN_PATH, "w", encoding="utf-8") as f:
        json.dump(curriculum_plan, f, indent=2)

    save_student_model(student_model)

    print(f"✅ Curriculum plan saved to {CURRICULUM_PLAN_PATH}")
    print(f"✅ Student model saved to {STUDENT_MODEL_PATH}")


# ======================
# CLI
# ======================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-id", required=True, help="Unique ID for the student")
    parser.add_argument("--max-concepts", type=int, default=2, help="Max number of concepts to generate")
    args = parser.parse_args()

    main(args.student_id, args.max_concepts)

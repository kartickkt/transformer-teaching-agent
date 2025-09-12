## Phase 5 – Adaptive Curriculum

In this phase, the agent dynamically generates personalized learning paths for each student, based on their current mastery and the unified knowledge graph.

### What happens
1. **Student model** (`outputs/student_model.json`)  
   - Tracks each student’s mastery per concept.  
   - Updates after every quiz.  

2. **Curriculum planner** (`outputs/curriculum_plan.json`)  
   - Selects next concepts to teach (using dependencies + mastery).  
   - Generates lessons and quizzes with OpenAI.  

3. **Lesson generation**  
   - OpenAI generates simple explanations + worked examples.  

4. **Quiz generation**  
   - OpenAI generates multiple-choice questions (JSON with exact answer keys).  
   - Student scores update mastery.  
   - Currently simulated, can be replaced with real inputs later.  

---

### Requirements
- Ensure you have an OpenAI API key in your `.env` file:

```bash
OPENAI_API_KEY=sk-...

"""Custom extraction instructions for personal agent memory.

Ported from ~/Develop/OpenClaw/rekollect/rekollect/extraction_config.py
"""

PERSONAL_MEMORY_INSTRUCTIONS = """
Focus on extracting:
1. DECISIONS: What was decided and why
2. PREFERENCES: User preferences, likes, dislikes, interests, passions (infer from behavior, not just explicit statements)
3. BEHAVIORAL RULES: Lessons learned, mistakes to avoid, workflow rules
4. PROJECT FACTS: Project names, repos, tech stacks, status
5. PEOPLE: Names, roles, relationships
6. TEMPORAL CHANGES: When something changed from one state to another
7. INTERESTS & OPINIONS: What the user cares about, finds exciting, or has strong opinions on

Entity naming rules (CRITICAL):
- Use the FULL canonical name for people: "Elton Faggett" not "Elton", "Cooper Flagg" not "Flagg"
- If you see a first name only and can infer the full name from context, use the full name
- Use proper project names: "Rekollect" not "the memory engine", "DFS Cheatsheet" not "the app"
- Team abbreviations are OK for sports: "DAL", "GS", "MEM"
- Merge aliases: "OpenClaw" and "Jarvis" refer to the same system -- use "OpenClaw"

Entity summary rules (CRITICAL):
- Every entity MUST have a meaningful one-sentence summary
- If you cannot describe what an entity IS or DOES from the conversation context, DO NOT extract it
- Generic/tangential mentions are NOT worth extracting as entities
- Good summary: "FastAPI backend serving DFS Cheatsheet v3 endpoints"
- Bad summary: "" (empty) or "mentioned in conversation"

DO NOT extract as entities:
- File paths, URLs, code variable/function/class names
- API endpoints, configuration keys, environment variables
- Git commit hashes
- Generic terms that aren't meaningful entities

DO NOT extract:
- Code snippets or raw technical output
- API keys, passwords, or secrets
- Transient UI/UX details
- System messages or heartbeat content
- Routine acknowledgments ("ok", "sounds good", "yes")
"""

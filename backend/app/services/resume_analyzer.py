import re

from app.schemas import ResumeProfile


PROGRAMMING_LANGUAGES = [
    "Python",
    "JavaScript",
    "TypeScript",
    "Java",
    "C++",
    "SQL",
]

FRAMEWORKS = [
    "FastAPI",
    "Django",
    "Flask",
    "React",
    "Next.js",
    "Tailwind",
    "SQLAlchemy",
]

TOOLS = [
    "SQLAlchemy",
    "SQLite",
    "PostgreSQL",
    "ChromaDB",
    "Docker",
    "Git",
    "Pandas",
    "NumPy",
    "Scikit-learn",
]

DOMAINS = [
    "RAG",
    "Machine Learning",
    "Deep Learning",
    "NLP",
    "Backend",
    "API",
    "Vector Search",
    "Data Engineering",
]

KNOWN_SKILLS = PROGRAMMING_LANGUAGES + FRAMEWORKS + TOOLS + DOMAINS


def extract_name(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if len(stripped.split()) <= 4 and not any(char.isdigit() for char in stripped):
            return stripped
    return None


def extract_skills(text: str) -> list[str]:
    found: list[str] = []
    lower_text = text.lower()
    for skill in KNOWN_SKILLS:
        if skill.lower() in lower_text and skill not in found:
            found.append(skill)
    return found


def extract_known_terms(text: str, terms: list[str]) -> list[str]:
    lower_text = text.lower()
    return [term for term in terms if term.lower() in lower_text]


def extract_bullets(text: str, keywords: list[str], limit: int = 4) -> list[str]:
    results: list[str] = []
    for line in text.splitlines():
        lower_line = line.lower()
        if any(keyword in lower_line for keyword in keywords):
            cleaned = re.sub(r"^[\-*•\s]+", "", line).strip()
            if cleaned and cleaned not in results:
                results.append(cleaned[:240])
        if len(results) >= limit:
            break
    return results


def infer_seniority(text: str, skills: list[str], projects: list[str]) -> str:
    lower_text = text.lower()
    if any(term in lower_text for term in ["senior", "lead", "architect", "principal"]):
        return "advanced"
    if any(term in lower_text for term in ["intern", "student", "fresher", "entry-level", "entry level"]):
        return "beginner"
    if len(skills) >= 8 or len(projects) >= 3:
        return "advanced"
    if len(skills) >= 4 or projects:
        return "intermediate"
    return "beginner"


def build_topics(
    skills: list[str],
    programming_languages: list[str],
    frameworks: list[str],
    tools: list[str],
    domains: list[str],
) -> list[str]:
    ordered = domains + frameworks + tools + programming_languages + skills
    topics: list[str] = []
    for topic in ordered:
        if topic not in topics:
            topics.append(topic)
        if len(topics) >= 6:
            break
    return topics or ["technical fundamentals", "project discussion"]


def analyze_resume(text: str) -> ResumeProfile:
    if len(text.strip()) < 30:
        raise ValueError("Resume text is too short to analyze.")

    skills = extract_skills(text)
    programming_languages = extract_known_terms(text, PROGRAMMING_LANGUAGES)
    frameworks = extract_known_terms(text, FRAMEWORKS)
    tools = extract_known_terms(text, TOOLS)
    domains = extract_known_terms(text, DOMAINS)
    projects = extract_bullets(text, ["project", "built", "developed", "created", "implemented"])
    seniority_level = infer_seniority(text, skills, projects)
    topics = build_topics(skills, programming_languages, frameworks, tools, domains)
    summary_parts = []
    if skills:
        summary_parts.append(f"Candidate shows signals for {', '.join(skills[:5])}.")
    if projects:
        summary_parts.append("Resume includes project evidence suitable for deep-dive questions.")
    summary = " ".join(summary_parts) or "Basic resume profile extracted with limited technical evidence."

    return ResumeProfile(
        candidate_name=extract_name(text),
        skills=skills,
        programming_languages=programming_languages,
        frameworks=frameworks,
        tools=tools,
        projects=projects,
        domains=domains,
        seniority_level=seniority_level,
        suggested_topics=topics,
        summary=summary,
    )

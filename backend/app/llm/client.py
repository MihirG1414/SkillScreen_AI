from abc import ABC, abstractmethod
import json
import re
from urllib import request
from urllib.error import URLError

from app.schemas import ResumeProfile, SourceTrace


class LLMProvider(ABC):
    @abstractmethod
    def generate_question(
        self,
        role: str,
        profile: ResumeProfile,
        traces: list[SourceTrace],
        difficulty: str,
        previous_questions: list[dict] | None = None,
        context_mode: str = "kb",
    ) -> dict:
        raise NotImplementedError

    @abstractmethod
    def evaluate_answer(self, question: str, answer: str) -> dict:
        raise NotImplementedError

    @abstractmethod
    def generate_report(self, profile: ResumeProfile, evaluations: list[dict]) -> dict:
        raise NotImplementedError


class FallbackProvider(LLMProvider):
    def generate_question(
        self,
        role: str,
        profile: ResumeProfile,
        traces: list[SourceTrace],
        difficulty: str,
        previous_questions: list[dict] | None = None,
        context_mode: str = "kb",
    ) -> dict:
        previous_questions = previous_questions or []
        question_index = len(previous_questions) + 1
        primary_skill = first_signal(profile.skills, "your project experience")
        project_signal = first_signal(profile.projects, "the resume project you described")
        concept = (
            extract_resume_concept(profile, role, previous_questions)
            if context_mode != "kb" or not traces
            else extract_concept_from_chunks(traces)
        )
        topic = derive_topic_label(concept, profile)
        style = choose_question_style(role, difficulty, previous_questions)
        prompt = build_question_from_style(
            style=style,
            role=role,
            primary_skill=primary_skill,
            project_focus=summarize_project_signal(project_signal),
            concept=concept,
            difficulty=difficulty,
            question_index=question_index,
        )
        is_valid, reason = validate_generated_question(prompt, previous_questions)
        if not is_valid:
            fallback_style = choose_question_style(
                role,
                difficulty,
                previous_questions,
                excluded_styles={style},
            )
            prompt = build_safe_fallback_question(
                style=fallback_style,
                role=role,
                primary_skill=primary_skill,
                project_focus=summarize_project_signal(project_signal),
                concept=concept,
                difficulty=difficulty,
                question_index=question_index,
            )
            style = fallback_style
        tier = resolved_source_tier(context_mode, concept["source_tier"])
        why = build_question_reason(
            topic=topic,
            primary_skill=primary_skill,
            concept_summary=concept["clean_context_summary"],
            tier=tier,
            context_mode=context_mode,
        )
        return {
            "question_text": prompt,
            "topic": topic,
            "difficulty": difficulty,
            "question_type": style,
            "why_this_question": why,
            "expected_points": build_expected_points(
                primary_skill=primary_skill,
                concept=concept,
                style=style,
                tier=tier,
                context_mode=context_mode,
            ),
            "source_tier_used": tier,
        }

    def evaluate_answer(self, question: str, answer: str) -> dict:
        words = [word for word in answer.split() if word.strip()]
        answer_lower = answer.lower()
        keyword_hits = sum(
            1
            for keyword in ["validate", "persist", "error", "test", "security", "api", "data", "model"]
            if keyword in answer_lower
        )
        score = min(95, max(20, len(words) * 3 + keyword_hits * 8))
        if not answer.strip():
            score = 0
        band = "strong" if score >= 80 else "acceptable" if score >= 65 else "weak" if score >= 40 else "insufficient"
        strengths = ["Mentions practical implementation details"] if keyword_hits else ["Provides a direct response"]
        gaps = [] if score >= 80 else ["Add more concrete tradeoffs, edge cases, and implementation detail"]
        return {
            "score": float(score),
            "band": band,
            "rationale": "Deterministic fallback evaluation based on answer length and technical keyword coverage.",
            "strengths": strengths,
            "gaps": gaps,
        }

    def generate_report(self, profile: ResumeProfile, evaluations: list[dict]) -> dict:
        if evaluations:
            overall = sum(item["score"] for item in evaluations) / len(evaluations)
        else:
            overall = 0.0
        recommendation = "advance" if overall >= 8 else "hold" if overall >= 5 else "needs_review"
        return {
            "overall_score": round(overall, 1),
            "recommendation": recommendation,
            "summary": profile.summary or "Candidate completed the MVP screening flow.",
            "strengths": sorted({s for item in evaluations for s in item.get("strengths", [])}),
            "gaps": sorted({g for item in evaluations for g in item.get("gaps", [])}),
        }


class OpenAIProvider(FallbackProvider):
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    def generate_question(
        self,
        role: str,
        profile: ResumeProfile,
        traces: list[SourceTrace],
        difficulty: str,
        previous_questions: list[dict] | None = None,
        context_mode: str = "kb",
    ) -> dict:
        if not self.api_key:
            return super().generate_question(role, profile, traces, difficulty, previous_questions, context_mode)
        prompt = build_llm_prompt(role, profile, traces, difficulty, previous_questions or [], context_mode)
        try:
            payload = {
                "model": "gpt-5-nano",
                "input": prompt,
                "text": {"format": {"type": "json_object"}},
            }
            response = post_json(
                url="https://api.openai.com/v1/responses",
                payload=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            text = response.get("output_text") or extract_openai_text(response)
            return normalize_question_json(
                json.loads(text), role, profile, traces, difficulty, previous_questions or [], context_mode
            )
        except Exception:
            return super().generate_question(role, profile, traces, difficulty, previous_questions, context_mode)


class GeminiProvider(FallbackProvider):
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    def generate_question(
        self,
        role: str,
        profile: ResumeProfile,
        traces: list[SourceTrace],
        difficulty: str,
        previous_questions: list[dict] | None = None,
        context_mode: str = "kb",
    ) -> dict:
        if not self.api_key:
            return super().generate_question(role, profile, traces, difficulty, previous_questions, context_mode)
        prompt = build_llm_prompt(role, profile, traces, difficulty, previous_questions or [], context_mode)
        try:
            response = post_json(
                url="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
                payload={"contents": [{"parts": [{"text": prompt}]}]},
                headers={"x-goog-api-key": self.api_key},
            )
            text = response["candidates"][0]["content"]["parts"][0]["text"]
            return normalize_question_json(
                json.loads(text), role, profile, traces, difficulty, previous_questions or [], context_mode
            )
        except Exception:
            return super().generate_question(role, profile, traces, difficulty, previous_questions, context_mode)


def first_signal(items: list[str], fallback: str) -> str:
    return items[0] if items else fallback


def summarize_trace_for_question(trace: SourceTrace | None) -> str:
    if trace is None:
        return "the selected role requirements"

    concept_phrase = derive_clean_concept_phrase(trace.text)
    if concept_phrase and not is_low_quality_summary(concept_phrase):
        return concept_phrase

    candidates = split_into_sentences(trace.text)
    for sentence in candidates:
        normalized = normalize_sentence(sentence)
        if is_readable_context_sentence(normalized):
            summary = generalize_context_sentence(normalized)
            if not is_low_quality_summary(summary):
                return summary

    if trace.display_name:
        keyword_phrase = extract_keyword_context(trace.text)
        if keyword_phrase and not is_low_quality_summary(keyword_phrase):
            return keyword_phrase
        return fallback_trace_summary(trace.display_name)
    return "the retrieved knowledge base context"


def split_into_sentences(text: str) -> list[str]:
    cleaned = " ".join(text.replace("\n", " ").split())
    if not cleaned:
        return []
    return re.split(r"(?<=[.!?])\s+", cleaned)


def normalize_sentence(sentence: str) -> str:
    return " ".join(sentence.strip().split())


def is_readable_context_sentence(sentence: str) -> bool:
    if len(sentence) < 40:
        return False
    if len(sentence.split()) < 7:
        return False
    lowered = sentence.lower()
    if any(term in lowered for term in ["index", "references", "bibliography", "contents"]):
        return False
    if re.search(r"(figure|fig\.)\s*\d+(\.\d+)?", lowered):
        return False
    if "=" in sentence:
        return False
    digit_count = sum(char.isdigit() for char in sentence)
    alpha_count = sum(char.isalpha() for char in sentence)
    if alpha_count == 0:
        return False
    if digit_count > max(3, alpha_count // 6):
        return False
    symbol_count = sum(not char.isalnum() and not char.isspace() for char in sentence)
    if symbol_count > max(8, alpha_count // 5):
        return False
    return True


def generalize_context_sentence(sentence: str) -> str:
    working = sentence.strip().rstrip(".")
    lowered = working.lower()
    subject_phrase = ""
    for marker in [" should ", " must ", " need to ", " needs to ", " can "]:
        if marker in lowered:
            marker_index = lowered.index(marker)
            subject_phrase = working[:marker_index].strip()
            clause = working[marker_index + len(marker):].strip()
            if clause:
                working = clause
                break
    working = re.sub(r"^(reliable|effective|good|strong|robust)\s+", "", working, flags=re.IGNORECASE)
    working = re.sub(r"^(backend|machine learning|ml|data science)\s+(systems?|workflows?|pipelines?|services?)\s+", "", working, flags=re.IGNORECASE)
    working = re.sub(r"^(the system|the workflow|the service)\s+", "", working, flags=re.IGNORECASE)
    working = re.sub(r"\bshould\b", "", working, flags=re.IGNORECASE)
    working = " ".join(working.split()).strip(" ,.;:")
    if not working:
        return "the retrieved knowledge base context"
    working = gerundify_leading_phrase(working)
    subject_keywords = summarize_subject_phrase(subject_phrase)
    if subject_keywords:
        working = f"{subject_keywords} with {working}"
    words = working.split()
    if len(words) > 16:
        working = " ".join(words[:16]).rstrip(",")
    if words and words[0].lower().endswith("ing"):
        return working
    if re.search(r"\b(validate|persist|handle|manage|compare|monitor|design|evaluate|debug|retrieve|rank)\b", working, re.IGNORECASE):
        return working[0].lower() + working[1:] if len(working) > 1 else working.lower()
    return working.lower()


def summarize_subject_phrase(subject_phrase: str) -> str:
    cleaned = normalize_sentence(subject_phrase).lower()
    cleaned = re.sub(r"^(a|an|the|reliable|effective|good|strong|robust)\s+", "", cleaned)
    cleaned = cleaned.replace("workflow", "").replace("system", "").replace("service", "").replace("training", "").strip()
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return ""
    words = [word for word in cleaned.split() if word not in {"backend", "machine", "learning"} or len(cleaned.split()) == 1]
    if not words:
        words = cleaned.split()
    summary = " ".join(words[:3]).strip()
    return summary


def gerundify_leading_phrase(text: str) -> str:
    replacements = {
        "validate ": "validating ",
        "persist ": "persisting ",
        "make ": "making ",
        "handle ": "handling ",
        "manage ": "managing ",
        "compare ": "comparing ",
        "monitor ": "monitoring ",
        "design ": "designing ",
        "evaluate ": "evaluating ",
        "debug ": "debugging ",
        "retrieve ": "retrieving ",
        "rank ": "ranking ",
    }
    lowered = text.lower()
    for prefix, replacement in replacements.items():
        if lowered.startswith(prefix):
            return replacement + text[len(prefix):]
    return text


def extract_keyword_context(text: str) -> str:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", text.lower())
    stopwords = {
        "this", "that", "with", "from", "into", "your", "have", "using", "used", "when",
        "where", "which", "their", "there", "these", "those", "page", "chapter", "figure",
        "algorithm", "system", "systems", "model", "models", "question", "questions", "source",
        "context", "role", "candidate", "resume", "knowledge", "book", "books", "data",
        "introduction", "artificial", "intelligence", "machine", "learning", "general",
        "probably", "analysis", "purpose", "sequential", "pandas", "keras", "fortunately",
        "trained", "apply", "using", "python", "videos", "later", "scientists", "generally",
        "sufficient", "moments", "test",
    }
    interesting = [token for token in tokens if token not in stopwords]
    if not interesting:
        return ""
    ranked: list[str] = []
    for token in interesting:
        if token not in ranked:
            ranked.append(token)
        if len(ranked) >= 3:
            break
    if not ranked:
        return ""
    if len(ranked) == 1:
        return ranked[0]
    if len(ranked) == 2:
        return f"{ranked[0]} and {ranked[1]}"
    return f"{ranked[0]}, {ranked[1]}, and {ranked[2]}"


def summarize_project_signal(project_signal: str) -> str:
    normalized = normalize_sentence(project_signal).rstrip(".")
    lowered = normalized.lower()
    for prefix in ["built ", "developed ", "implemented ", "created "]:
        if lowered.startswith(prefix):
            summary = normalized[len(prefix):]
            return trim_phrase(summary, 96)
    return trim_phrase(normalized, 96)


def extract_concept_from_chunks(chunks: list[SourceTrace]) -> dict:
    if not chunks:
        return {
            "concept": "role-aligned engineering tradeoffs",
            "learning_objective": "Explain a practical solution and how to validate it.",
            "clean_context_summary": "role-aligned engineering tradeoffs and validation",
            "usable_keywords": ["validation", "tradeoffs"],
            "source_tier": "none",
        }

    summaries: list[tuple[float, str]] = []
    keywords: list[str] = []
    highest_tier = chunks[0].tier
    for chunk in chunks[:3]:
        summary = summarize_trace_for_question(chunk)
        if summary:
            quality = concept_summary_quality(summary)
            if quality > 0 and all(existing_summary != summary for _, existing_summary in summaries):
                summaries.append((quality, summary))
        for keyword in extract_ranked_keywords(chunk.text):
            if keyword not in keywords:
                keywords.append(keyword)
        highest_tier = max_tier(highest_tier, chunk.tier)

    summaries.sort(key=lambda item: item[0], reverse=True)
    concept = summaries[0][1] if summaries else "retrieved engineering context"
    if len(summaries) > 1 and len(tokenize_phrase(concept)) <= 4 and summaries[1][0] >= summaries[0][0] - 0.5:
        concept = f"{summaries[0][1]} and {summaries[1][1]}"
    clean_context_summary = concept.strip().rstrip(".")
    learning_objective = build_learning_objective(clean_context_summary)
    usable_keywords = tokenize_phrase(clean_context_summary)[:5] or keywords[:5]
    return {
        "concept": clean_context_summary,
        "learning_objective": learning_objective,
        "clean_context_summary": clean_context_summary,
        "usable_keywords": usable_keywords,
        "source_tier": highest_tier,
    }


def extract_resume_concept(
    profile: ResumeProfile,
    role: str,
    previous_questions: list[dict] | None = None,
) -> dict:
    previous_questions = previous_questions or []
    profile_text = " ".join(
        profile.skills + profile.projects + profile.suggested_topics + profile.domains + profile.frameworks + profile.tools
    ).lower()
    candidates = resume_concept_candidates(profile_text, role)
    recent_topics = {normalize_topic_key(str(question.get("topic") or "")) for question in previous_questions[-3:]}
    selected = candidates[0]
    for candidate in candidates:
        if normalize_topic_key(candidate["topic"]) not in recent_topics:
            selected = candidate
            break

    concept_summary = selected["concept"]
    keywords = selected["keywords"] or extract_ranked_keywords(profile_text)[:5] or tokenize_phrase(concept_summary)[:4]
    return {
        "concept": concept_summary,
        "learning_objective": build_learning_objective(concept_summary),
        "clean_context_summary": concept_summary,
        "usable_keywords": keywords,
        "source_tier": "resume_only",
        "topic_hint": selected["topic"],
    }


def role_default_resume_concept(role: str) -> str:
    return {
        "BACKEND_ENGINEER": "request validation, persistence, and reliable API behavior",
        "AI_ML_ENGINEER": "model design, evaluation, and production tradeoffs",
        "DATA_SCIENCE_APPLIED_ML": "data preparation, model evaluation, and production readiness",
    }.get(role, "role-aligned engineering tradeoffs and validation")


def resume_concept_candidates(profile_text: str, role: str) -> list[dict]:
    candidates: list[dict] = []
    if role == "AI_ML_ENGINEER":
        if any(term in profile_text for term in ["validation", "error analysis", "confusion", "metrics"]):
            candidates.append(
                {
                    "topic": "Model Validation",
                    "concept": "model validation, failure analysis, and reliable evaluation",
                    "keywords": ["validation", "metrics", "failure", "evaluation"],
                }
            )
        if any(term in profile_text for term in ["tracking", "computer vision", "opencv", "object tracking", "kalman"]):
            candidates.append(
                {
                    "topic": "Computer Vision Systems",
                    "concept": "designing robust computer vision pipelines under latency and accuracy constraints",
                    "keywords": ["tracking", "opencv", "latency", "robustness"],
                }
            )
        if any(term in profile_text for term in ["drift", "data quality", "class imbalance", "preprocessing"]):
            candidates.append(
                {
                    "topic": "Data Quality",
                    "concept": "data quality checks, drift detection, and failure isolation",
                    "keywords": ["drift", "quality", "imbalance", "preprocessing"],
                }
            )
        if any(term in profile_text for term in ["deployment", "latency", "production", "monitoring"]):
            candidates.append(
                {
                    "topic": "Production Tradeoffs",
                    "concept": "accuracy, latency, and deployment tradeoffs in ML systems",
                    "keywords": ["latency", "deployment", "monitoring", "tradeoffs"],
                }
            )
        if any(term in profile_text for term in ["uncertainty", "probabilistic", "tracking systems"]):
            candidates.append(
                {
                    "topic": "Uncertainty",
                    "concept": "uncertainty handling and state estimation in ML systems",
                    "keywords": ["uncertainty", "state", "probabilistic", "tracking"],
                }
            )
    elif role == "BACKEND_ENGINEER":
        if any(term in profile_text for term in ["validation", "api", "fastapi"]):
            candidates.append(
                {
                    "topic": "API Reliability",
                    "concept": "request validation, safe errors, and reliable API behavior",
                    "keywords": ["validation", "api", "errors", "reliability"],
                }
            )
        if any(term in profile_text for term in ["persist", "sql", "metadata", "storage"]):
            candidates.append(
                {
                    "topic": "Persistence Design",
                    "concept": "metadata persistence, consistency, and recovery tradeoffs",
                    "keywords": ["persistence", "metadata", "consistency", "recovery"],
                }
            )
    elif role == "DATA_SCIENCE_APPLIED_ML":
        candidates.extend(
            [
                {
                    "topic": "Model Evaluation",
                    "concept": "data preparation, model evaluation, and practical tradeoffs",
                    "keywords": ["evaluation", "validation", "tradeoffs", "features"],
                },
                {
                    "topic": "Feature Quality",
                    "concept": "feature quality, data drift, and reliable experimentation",
                    "keywords": ["features", "drift", "experimentation", "quality"],
                },
            ]
        )

    if not candidates:
        candidates.append(
            {
                "topic": role.replace("_", " ").title(),
                "concept": role_default_resume_concept(role),
                "keywords": tokenize_phrase(role_default_resume_concept(role))[:4],
            }
        )
    return dedupe_resume_candidates(candidates)


def dedupe_resume_candidates(candidates: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = normalize_topic_key(candidate["topic"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def normalize_topic_key(text: str) -> str:
    return " ".join(re.findall(r"[a-zA-Z]+", text.lower())[:4])


def derive_clean_concept_phrase(text: str) -> str:
    lowered = " ".join(text.lower().split())
    pattern_map = [
        (
            ["upload", "metadata", "idempotent"],
            "upload validation, metadata persistence, and idempotent retries",
        ),
        (
            ["validate requests", "persist metadata"],
            "request validation, metadata persistence, and safe API errors",
        ),
        (
            ["safe api errors", "persist metadata"],
            "request validation, metadata persistence, and safe API errors",
        ),
        (
            ["missing data"],
            "handling missing data in training pipelines",
        ),
        (
            ["cost function", "hyper"],
            "hyperparameter tuning and objective-function tradeoffs",
        ),
        (
            ["initializing hyperparameters"],
            "hyperparameter initialization and tuning strategy",
        ),
        (
            ["weighted sum"],
            "weighted feature contributions and model scoring",
        ),
        (
            ["hypotheses", "population"],
            "searching a population of candidate hypotheses",
        ),
        (
            ["inductive learning"],
            "inductive learning under limited prior knowledge",
        ),
        (
            ["bias", "variance"],
            "bias-variance tradeoffs and model generalization",
        ),
        (
            ["hidden", "state"],
            "hidden-state estimation and sequence uncertainty",
        ),
        (
            ["posterior", "observation"],
            "posterior updates under uncertain observations",
        ),
        (
            ["decision", "boundary"],
            "decision boundaries and classification tradeoffs",
        ),
        (
            ["model", "selection"],
            "model selection and generalization checks",
        ),
        (
            ["hypothesis", "space"],
            "search strategy, model complexity, and generalization",
        ),
        (
            ["id3", "hypothesis"],
            "search strategy and decision tree generalization",
        ),
        (
            ["graphical", "model"],
            "probabilistic graphical models for uncertainty and state estimation",
        ),
        (
            ["gaussian", "process"],
            "uncertainty estimation with Gaussian processes",
        ),
        (
            ["cross-validation"],
            "cross-validation and model selection tradeoffs",
        ),
        (
            ["resampling"],
            "model selection and resampling-based validation",
        ),
        (
            ["regularization"],
            "regularization and generalization tradeoffs",
        ),
        (
            ["boosting"],
            "ensemble methods and bias-variance tradeoffs",
        ),
        (
            ["mixture", "model"],
            "mixture models and latent structure",
        ),
        (
            ["misclassified", "classification"],
            "classification decision boundaries and error tradeoffs",
        ),
        (
            ["missing data"],
            "handling missing data in training pipelines",
        ),
        (
            ["data leakage", "baseline"],
            "model validation, baseline comparison, and data leakage prevention",
        ),
        (
            ["latent state", "uncertainty"],
            "uncertainty-aware probabilistic tracking",
        ),
        (
            ["drift", "feature audit"],
            "debugging regressions through drift checks and feature audits",
        ),
        (
            ["deterministic", "probabilistic"],
            "comparing deterministic and probabilistic modeling approaches",
        ),
        (
            ["upload", "metadata", "idempotent"],
            "upload validation, metadata persistence, and idempotent retries",
        ),
        (
            ["validate requests", "persist metadata"],
            "request validation, metadata persistence, and safe API errors",
        ),
        (
            ["safe api errors", "persist metadata"],
            "request validation, metadata persistence, and safe API errors",
        ),
        (
            ["failure", "retry"],
            "failure recovery and safe retry behavior",
        ),
        (
            ["source trace", "metadata"],
            "source traceability and metadata integrity",
        ),
        (
            ["validation", "failure case"],
            "model validation and failure-case review",
        ),
    ]
    for required_terms, phrase in pattern_map:
        if all(term in lowered for term in required_terms):
            return phrase

    keywords = extract_ranked_keywords(text)
    if "validation" in keywords and "leakage" in keywords:
        return "model validation and leakage prevention"
    if "tracking" in keywords and "uncertainty" in keywords:
        return "uncertainty-aware tracking"
    if "drift" in keywords and "audits" in keywords:
        return "debugging regressions through drift checks"
    if keywords:
        phrase = " ".join(keywords[:4]).replace("pipelines", "pipeline")
        return trim_phrase(phrase, 72)
    return ""


def trim_phrase(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text.rstrip(", ")
    trimmed = text[:limit].rsplit(" ", 1)[0].rstrip(", ")
    trimmed = re.sub(r"\b(and|or|with)$", "", trimmed, flags=re.IGNORECASE).rstrip(", ")
    return trimmed or text[:limit].rstrip(", ")


def build_learning_objective(clean_context_summary: str) -> str:
    lowered = clean_context_summary.lower()
    if "compare" in lowered or "versus" in lowered:
        return f"Compare practical options for {clean_context_summary}."
    if "debug" in lowered or "failure" in lowered or "drift" in lowered:
        return f"Explain how to diagnose and recover from issues related to {clean_context_summary}."
    return f"Explain how to apply {clean_context_summary} in a practical system and validate the result."


def choose_question_style(
    role: str,
    difficulty: str,
    previous_questions: list[dict],
    excluded_styles: set[str] | None = None,
) -> str:
    excluded_styles = excluded_styles or set()
    recent_styles = [
        str(question.get("question_type") or "").strip()
        for question in previous_questions[-2:]
        if str(question.get("question_type") or "").strip()
    ]
    if not previous_questions:
        candidates = ["resume_grounding", "applied_scenario", "conceptual"]
    elif difficulty == "advanced":
        candidates = ["advanced_reasoning", "system_design", "comparison", "debugging"]
    elif difficulty in {"beginner", "probing"}:
        candidates = ["conceptual", "resume_grounding", "applied_scenario", "debugging"]
    else:
        candidates = ["applied_scenario", "debugging", "system_design", "comparison", "conceptual"]

    for style in candidates:
        if style not in recent_styles and style not in excluded_styles:
            return style
    for style in candidates:
        if style not in excluded_styles:
            return style
    return "conceptual"


def build_question_from_style(
    *,
    style: str,
    role: str,
    primary_skill: str,
    project_focus: str,
    concept: dict,
    difficulty: str,
    question_index: int,
) -> str:
    role_label = role_context_label(role)
    role_phrase = role_context_noun_phrase(role)
    context_focus = sanitize_concept_phrase(concept["clean_context_summary"])
    topic_hint = str(concept.get("topic_hint") or "").strip()
    topic_lead = topic_hint.lower() if topic_hint else context_focus
    if style == "resume_grounding":
        if question_index == 1:
            return (
                f"Your resume includes {project_focus}. How would you use {primary_skill} to handle {context_focus} "
                "in a real workflow, and what would you validate first?"
            )
        return (
            f"Staying with your {primary_skill} experience, how would you handle {topic_lead} in {role_phrase} workflow, "
            "and what would you validate first?"
        )
    if style == "conceptual":
        return (
            f"What does {context_focus} change in how you design a {role_label} workflow? "
            f"What signal would tell you the approach is reliable?"
        )
    if style == "applied_scenario":
        return (
            f"Suppose you are shipping {project_focus}. How would you design the workflow around {context_focus}, "
            "and what practical checks would you add before rollout?"
        )
    if style == "debugging":
        return (
            f"If a deployed system starts failing around {context_focus}, how would you isolate the cause "
            "and what evidence would you collect first?"
        )
    if style == "system_design":
        return (
            f"Design {role_phrase} workflow for {context_focus} using {primary_skill}. "
            "What components and safeguards would you include?"
        )
    if style == "comparison":
        return (
            f"Compare two ways to handle {context_focus} in {role_phrase} workflow. "
            "When would you prefer one approach over the other?"
        )
    return (
        f"For a harder version of this problem, how would you reason about {context_focus} in {role_phrase} system? "
        "Defend the tradeoffs and validation plan you would use."
    )


def role_context_label(role: str) -> str:
    return {
        "AI_ML_ENGINEER": "AI/ML engineering",
        "BACKEND_ENGINEER": "backend engineering",
        "DATA_SCIENCE_APPLIED_ML": "applied ML",
    }.get(role, role.replace("_", " ").lower())


def role_context_noun_phrase(role: str) -> str:
    label = role_context_label(role)
    article = "an" if label[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {label}"


def build_safe_fallback_question(
    *,
    style: str,
    role: str,
    primary_skill: str,
    project_focus: str,
    concept: dict,
    difficulty: str,
    question_index: int,
) -> str:
    safe_concept = sanitize_concept_phrase(concept["clean_context_summary"])
    return build_question_from_style(
        style=style,
        role=role,
        primary_skill=primary_skill,
        project_focus=project_focus,
        concept={**concept, "clean_context_summary": safe_concept},
        difficulty=difficulty,
        question_index=question_index,
    )


def sanitize_concept_phrase(text: str) -> str:
    cleaned = re.sub(r"\b(index|figure|references|bibliography|contents|page)\b", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b\d+\b", "", cleaned)
    cleaned = cleaned.replace("consider difference between hypothesis", "search strategy and model complexity")
    cleaned = cleaned.replace("building and validating computer vision workflows", "computer vision pipelines and their validation strategy")
    cleaned = " ".join(cleaned.split()).strip(" ,.;:")
    if is_low_quality_summary(cleaned):
        return "reliable model design and validation"
    return cleaned or "reliable system design and validation"


def concept_summary_quality(summary: str) -> float:
    if is_low_quality_summary(summary):
        return 0.0
    tokens = tokenize_phrase(summary)
    score = float(len(tokens))
    if 3 <= len(tokens) <= 8:
        score += 4.0
    if len(tokens) > 12:
        score -= 4.0
    preferred = {"validation", "tradeoffs", "uncertainty", "generalization", "tracking", "missing", "regularization", "classification", "debugging"}
    score += len(preferred & set(tokens)) * 1.5
    return score


def is_low_quality_summary(summary: str) -> bool:
    lowered = summary.lower().strip()
    if not lowered:
        return True
    if lowered[0].isdigit() or lowered.startswith(("•", "-", "chapter ", "introduction to ")):
        return True
    bad_phrases = [
        "introduction keras", "probably sequential", "purpose analysis", "artificial intelligence machine learning",
        "apply trained", "generally sufficient", "scientists videos", "while training test", "after moments",
        "expert systems", "information science and statistics", "solution path selecting",
        "deep learning introduction", "inductive learning analytical", "collection hypotheses called current",
    ]
    if any(phrase in lowered for phrase in bad_phrases):
        return True
    generic_only = {"artificial", "intelligence", "machine", "learning", "general", "analysis", "purpose"}
    tokens = tokenize_phrase(lowered)
    if len(tokens) < 3:
        return True
    if "introduction" in tokens[:2]:
        return True
    weak_leads = {"however", "there", "need", "since", "preceding", "current", "plentiful"}
    strong_ml_terms = {"validation", "generalization", "regularization", "tracking", "uncertainty", "classification", "regression", "drift", "cross", "missing", "tradeoffs"}
    if tokens and tokens[0] in weak_leads and not (set(tokens) & strong_ml_terms):
        return True
    if sum(token in generic_only for token in tokens) >= max(2, len(tokens) - 1):
        return True
    return False


def fallback_trace_summary(display_name: str) -> str:
    lowered = display_name.lower()
    if "probabilistic" in lowered:
        return "probabilistic modeling and uncertainty estimation"
    if "statistical learning" in lowered:
        return "model selection and validation techniques"
    if "tom mitchell" in lowered or "machine learning" in lowered:
        return "core machine learning concepts and tradeoffs"
    if "pattern recognition" in lowered:
        return "probabilistic modeling and pattern recognition"
    return f"core ideas from {display_name}"


def derive_topic_label(concept: dict, profile: ResumeProfile) -> str:
    topic_hint = str(concept.get("topic_hint") or "").strip()
    if topic_hint:
        return topic_hint
    summary = str(concept.get("clean_context_summary") or "").strip()
    if summary:
        words = re.findall(r"[A-Za-z][A-Za-z0-9-]*", summary)
        if words:
            return " ".join(words[:4]).title()
    keywords = concept.get("usable_keywords") or []
    if keywords:
        joined = " ".join(str(keyword) for keyword in keywords[:2]).strip()
        if joined:
            return joined.title()
    return first_signal(profile.suggested_topics, first_signal(profile.skills, "Technical Interview"))


def build_expected_points(
    *,
    primary_skill: str,
    concept: dict,
    style: str,
    tier: str,
    context_mode: str = "kb",
) -> list[str]:
    concept_label = concept["clean_context_summary"]
    points = [
        f"Connect the answer to the candidate's {primary_skill} experience.",
        f"Explain a concrete approach for {concept_label}.",
    ]
    if style in {"comparison", "advanced_reasoning"}:
        points.append("Compare tradeoffs and justify when one approach is better than another.")
    elif style == "debugging":
        points.append("Describe the debugging signals, root-cause checks, and recovery steps.")
    else:
        points.append("Describe how you would validate the solution with metrics, tests, or failure-case review.")
    if context_mode == "kb":
        points.append(f"Ground the reasoning in the retrieved {tier} source concept without quoting the source text.")
    else:
        points.append("Keep the answer practical for the selected role, even without relying on a knowledge-base excerpt.")
    return points


def resolved_source_tier(context_mode: str, source_tier: str) -> str:
    if context_mode == "kb":
        return source_tier
    if context_mode == "resume_fallback":
        return "resume_fallback"
    return "resume_only"


def build_question_reason(
    *,
    topic: str,
    primary_skill: str,
    concept_summary: str,
    tier: str,
    context_mode: str,
) -> str:
    if context_mode == "resume_only":
        return (
            f"This is a resume-guided {topic} question based on the candidate's {primary_skill} experience "
            f"and the selected role. It intentionally starts without relying on the knowledge base."
        )
    if context_mode == "resume_fallback":
        return (
            f"This asks about {topic} using the candidate's {primary_skill} experience because no eligible "
            f"knowledge-base chunks were available for a grounded follow-up on this turn."
        )
    why = (
        f"This asks about {topic} because the resume contains {primary_skill} and the "
        f"retrieved {tier} source highlights {concept_summary}."
    )
    if tier == "advanced":
        why += " Advanced material was used because the candidate seniority/difficulty allowed it."
    return why


def validate_generated_question(
    question_text: str,
    previous_questions: list[dict] | None = None,
) -> tuple[bool, str]:
    previous_questions = previous_questions or []
    lowered = question_text.lower()
    forbidden_terms = ["index", "figure", "bibliography", "references", "contents"]
    for term in forbidden_terms:
        if term in lowered:
            return False, f"contains forbidden term: {term}"
    if re.search(r"\bpage\s+\d+\b", lowered):
        return False, "contains raw page reference"
    if sum(char.isdigit() for char in question_text) > 4:
        return False, "contains too many numbers"
    sentences = [segment.strip() for segment in re.split(r"[.!?]+", question_text) if segment.strip()]
    if len(sentences) == 0 or len(sentences) > 3:
        return False, "question must be between 1 and 3 sentences"
    if re.search(r"\b(where [A-Z] denotes|Figure \d|Eq\.|equation)\b", question_text):
        return False, "looks like raw textbook reference"
    opening = opening_signature(question_text)
    for previous in previous_questions:
        if opening == opening_signature(str(previous.get("question_text") or "")):
            return False, "repeats previous opening phrase"
        if structural_signature(question_text) == structural_signature(str(previous.get("question_text") or "")):
            return False, "repeats previous question structure"
    if not re.search(r"\b(explain|design|compare|debug|apply|reason|build|handle)\b", lowered):
        return False, "does not contain a clear interview task"
    if len(re.findall(r"\b[A-Z]{3,}\b", question_text)) >= 4:
        return False, "contains too much raw uppercase source text"
    return True, ""


def opening_signature(text: str) -> str:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return " ".join(tokens[:4])


def structural_signature(text: str) -> str:
    lowered = text.lower()
    for keyword in ["explain", "design", "compare", "debug", "apply", "reason", "build", "handle"]:
        if keyword in lowered:
            return keyword
    return opening_signature(text)


def extract_ranked_keywords(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", text.lower())
    stopwords = {
        "this", "that", "with", "from", "into", "your", "have", "using", "used", "when",
        "where", "which", "their", "there", "these", "those", "page", "chapter", "figure",
        "algorithm", "system", "systems", "model", "models", "question", "questions", "source",
        "context", "role", "candidate", "resume", "knowledge", "book", "books", "data", "reliable",
        "should", "would", "could", "through", "while", "about"
    }
    ranked: list[str] = []
    for token in tokens:
        if token in stopwords:
            continue
        if token not in ranked:
            ranked.append(token)
        if len(ranked) >= 5:
            break
    return ranked


def tokenize_phrase(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower())


def max_tier(left: str, right: str) -> str:
    rank = {"none": 0, "foundation": 1, "core": 2, "applied": 3, "advanced": 4}
    return left if rank.get(left, 0) >= rank.get(right, 0) else right


def get_llm_provider(
    name: str | None = None,
    openai_api_key: str | None = None,
    gemini_api_key: str | None = None,
) -> LLMProvider:
    normalized = (name or "fallback").lower()
    if (normalized == "openai" or openai_api_key) and openai_api_key:
        return OpenAIProvider(openai_api_key)
    if (normalized == "gemini" or gemini_api_key) and gemini_api_key:
        return GeminiProvider(gemini_api_key)
    return FallbackProvider()


def build_llm_prompt(
    role: str,
    profile: ResumeProfile,
    traces: list[SourceTrace],
    difficulty: str,
    previous_questions: list[dict],
    context_mode: str,
) -> str:
    concept = extract_concept_from_chunks(traces)
    context = (
        "\n\n".join(
            f"Source {index + 1} ({trace.tier}, {trace.display_name}): {trace.text[:500]}"
            for index, trace in enumerate(traces[:4])
        )
        if traces
        else "No eligible retrieved context. Generate a clean resume-and-role-based question."
    )
    return (
        "Generate one non-generic technical interview question as strict JSON with keys "
        "question_text, topic, difficulty, question_type, why_this_question, "
        "expected_points, source_tier_used. Do not include markdown.\n"
        "The question must be answerable without the source book, must not quote raw snippets, and must be 1 to 3 sentences.\n"
        "Use one of these question types: resume_grounding, conceptual, applied_scenario, debugging, system_design, comparison, advanced_reasoning.\n"
        f"Role: {role}\n"
        f"Requested difficulty: {difficulty}\n"
        f"Question context mode: {context_mode}\n"
        f"Resume profile: {profile.model_dump_json()}\n"
        f"Previous questions: {json.dumps(previous_questions[-3:])}\n"
        f"Concept summary: {json.dumps(concept)}\n"
        f"Retrieved context:\n{context}"
    )


def post_json(url: str, payload: dict, headers: dict[str, str]) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        raise RuntimeError("LLM request failed.") from exc


def extract_openai_text(response: dict) -> str:
    parts: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                parts.append(content["text"])
    return "\n".join(parts)


def normalize_question_json(
    data: dict,
    role: str,
    profile: ResumeProfile,
    traces: list[SourceTrace],
    difficulty: str,
    previous_questions: list[dict],
    context_mode: str,
) -> dict:
    fallback = FallbackProvider().generate_question(
        role, profile, traces, difficulty, previous_questions, context_mode
    )
    expected_points = data.get("expected_points")
    if not isinstance(expected_points, list):
        expected_points = fallback["expected_points"]
    question_text = str(data.get("question_text") or fallback["question_text"])
    valid, _ = validate_generated_question(question_text, previous_questions)
    if not valid:
        return fallback
    return {
        "question_text": question_text,
        "topic": str(data.get("topic") or fallback["topic"]),
        "difficulty": str(data.get("difficulty") or fallback["difficulty"]),
        "question_type": str(data.get("question_type") or fallback["question_type"]),
        "why_this_question": str(data.get("why_this_question") or fallback["why_this_question"]),
        "expected_points": [str(item) for item in expected_points],
        "source_tier_used": str(data.get("source_tier_used") or fallback["source_tier_used"]),
    }

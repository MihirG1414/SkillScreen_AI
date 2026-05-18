export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";

export type ResumeProfile = {
  candidate_name: string | null;
  skills: string[];
  programming_languages: string[];
  frameworks: string[];
  tools: string[];
  projects: string[];
  domains: string[];
  seniority_level: string;
  suggested_topics: string[];
  summary: string;
};

export type ResumeUploadResponse = {
  session_id: string;
  selected_role: string;
  extracted_profile: ResumeProfile;
  text_length: number;
};

export type RagTrace = {
  retrieval_query: string;
  source_filename: string;
  display_name: string;
  page_number: number | null;
  tier: string;
  context_summary: string;
  chunk_preview: string;
};

export type Question = {
  question_id: string;
  question_text: string;
  topic: string;
  difficulty: string;
  question_type: string;
  why_this_question: string;
  expected_points: string[];
  source_tier_used: string;
  rag_trace: RagTrace[];
};

export type InterviewStartResponse = {
  session_id: string;
  question: Question;
};

export type Evaluation = {
  id: string;
  score: number;
  band: string;
  technical_accuracy: number;
  clarity: number;
  depth: number;
  strengths: string[];
  missing_points: string[];
  feedback: string;
  ideal_answer_summary: string;
  source_grounding_notes: string;
};

export type AdaptationDecision = {
  next_difficulty: string;
  advanced_sources_unlocked: boolean;
  reason: string;
};

export type InterviewAnswerResponse = {
  answer_id: string;
  evaluation: Evaluation;
  adaptation_decision: AdaptationDecision;
  next_question: Question | null;
};

export type Report = {
  id: string;
  overall_score: number;
  role_fit: string;
  recommendation: string;
  candidate_summary: string;
  technical_strengths: string[];
  areas_for_improvement: string[];
  topic_breakdown: Record<string, { average_score: number; questions: number }>;
  question_answer_summary: Array<{
    number: number;
    question_id: string;
    question_text: string;
    topic: string;
    difficulty: string;
    source_tier_used: string;
    answer_text: string;
    answer_summary: string;
    score: number;
    feedback: string;
    adaptation_decision: Partial<AdaptationDecision>;
    rag_trace: RagTrace[];
  }>;
  source_usage_summary: {
    chunks_by_tier: Record<string, number>;
    source_books_used: string[];
    advanced_candidate_evaluation_triggered: boolean;
  };
  advanced_readiness: {
    ready_for_advanced_theory: boolean;
    advanced_evaluation_triggered: boolean;
    summary: string;
  };
  suggested_next_round_questions: string[];
};

export type ReportGenerateResponse = {
  session_id: string;
  report: Report;
};

export type SessionSnapshot = {
  id: string;
  role: string | null;
  resume_filename: string | null;
  profile: ResumeProfile;
  status: string;
  questions: Array<{
    question_id: string;
    question_text: string;
    topic: string;
    difficulty: string;
    retrieval_query: string;
    rag_trace: RagTrace[];
    question_metadata?: {
      why_this_question?: string;
      expected_points?: string[];
      source_tier_used?: string;
    };
    answers: unknown[];
  }>;
  report: Report | null;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: init?.body instanceof FormData
      ? init.headers
      : { "Content-Type": "application/json", ...init?.headers }
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = typeof body.detail === "string" ? body.detail : "Request failed";
    throw new Error(message);
  }
  return body as T;
}

export async function uploadResume(file: File, selectedRole: string): Promise<ResumeUploadResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("selected_role", selectedRole);
  return request<ResumeUploadResponse>("/api/resume/upload", {
    method: "POST",
    body: form
  });
}

export async function startInterview(sessionId: string): Promise<InterviewStartResponse> {
  return request<InterviewStartResponse>("/api/interview/start", {
    method: "POST",
    body: JSON.stringify({ session_id: sessionId })
  });
}

export async function submitAnswer(
  sessionId: string,
  questionId: string,
  answerText: string
): Promise<InterviewAnswerResponse> {
  return request<InterviewAnswerResponse>("/api/interview/answer", {
    method: "POST",
    body: JSON.stringify({
      session_id: sessionId,
      question_id: questionId,
      answer_text: answerText
    })
  });
}

export async function generateReport(sessionId: string): Promise<ReportGenerateResponse> {
  return request<ReportGenerateResponse>("/api/report/generate", {
    method: "POST",
    body: JSON.stringify({ session_id: sessionId })
  });
}

export async function getSession(sessionId: string): Promise<{ session: SessionSnapshot }> {
  return request<{ session: SessionSnapshot }>(`/api/session/${sessionId}`, {
    cache: "no-store"
  });
}

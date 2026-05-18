"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import {
  generateReport,
  getSession,
  submitAnswer,
  type AdaptationDecision,
  type Evaluation,
  type Question,
  type SessionSnapshot
} from "@/lib/api";

const INTERVIEW_QUESTION_LIMIT = 5;

export default function InterviewSessionPage() {
  const params = useParams<{ sessionId: string }>();
  const router = useRouter();
  const sessionId = params.sessionId;
  const [session, setSession] = useState<SessionSnapshot | null>(null);
  const [currentQuestion, setCurrentQuestion] = useState<Question | null>(null);
  const [nextQuestion, setNextQuestion] = useState<Question | null>(null);
  const [answer, setAnswer] = useState("");
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [adaptation, setAdaptation] = useState<AdaptationDecision | null>(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [reporting, setReporting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const response = await getSession(sessionId);
        setSession(response.session);
        const latest = response.session.questions.at(-1);
        if (latest) {
          setCurrentQuestion({
            question_id: latest.question_id,
            question_text: latest.question_text,
            topic: latest.topic,
            difficulty: latest.difficulty,
            question_type: "scenario",
            why_this_question: latest.question_metadata?.why_this_question ?? "Generated from resume signals and retrieved source context.",
            expected_points: latest.question_metadata?.expected_points ?? [],
            source_tier_used: latest.question_metadata?.source_tier_used ?? "unknown",
            rag_trace: latest.rag_trace
          } as Question);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unable to load session.");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [sessionId]);

  const questionNumber = useMemo(
    () => Math.min((session?.questions.length ?? 1), INTERVIEW_QUESTION_LIMIT),
    [session]
  );
  const canGenerateReport =
    !nextQuestion && Boolean(evaluation) && (session?.questions.length ?? 0) >= INTERVIEW_QUESTION_LIMIT;

  async function handleSubmit() {
    if (!currentQuestion || !answer.trim()) {
      setError("Write an answer before submitting.");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const result = await submitAnswer(sessionId, currentQuestion.question_id, answer);
      setEvaluation(result.evaluation);
      setAdaptation(result.adaptation_decision);
      setNextQuestion(result.next_question);
      const refreshed = await getSession(sessionId);
      setSession(refreshed.session);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Answer submission failed.");
    } finally {
      setSubmitting(false);
    }
  }

  function handleContinue() {
    if (!nextQuestion) return;
    setCurrentQuestion(nextQuestion);
    setNextQuestion(null);
    setEvaluation(null);
    setAdaptation(null);
    setAnswer("");
  }

  async function handleReport() {
    setReporting(true);
    setError(null);
    try {
      await generateReport(sessionId);
      router.push(`/report/${sessionId}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Report generation failed.");
    } finally {
      setReporting(false);
    }
  }

  if (loading) {
    return <main className="shell"><div className="card">Loading interview...</div></main>;
  }

  return (
    <main className="shell space-y-6">
      <header className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-sm font-semibold text-blue-700">
            Question {questionNumber} of {INTERVIEW_QUESTION_LIMIT}
          </p>
          <h1 className="mt-1 text-3xl font-semibold text-ink">Adaptive interview</h1>
          <p className="muted mt-2">{session?.role ?? "Selected role"} · Session {sessionId}</p>
        </div>
        <button className="btn-secondary" type="button" onClick={handleReport} disabled={reporting || !canGenerateReport}>
          {reporting ? "Generating..." : "Generate report"}
        </button>
      </header>

      {error && <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{error}</div>}

      {currentQuestion ? (
        <section className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
          <div className="card space-y-5">
            <QuestionCard question={currentQuestion} />
            <div className="space-y-2">
              <label className="label" htmlFor="answer">Your answer</label>
              <textarea
                id="answer"
                className="field min-h-44 resize-y"
                value={answer}
                onChange={(event) => setAnswer(event.target.value)}
                placeholder="Explain your approach with concrete implementation details, tradeoffs, and source-grounded reasoning."
                disabled={Boolean(evaluation)}
              />
            </div>
            <div className="flex flex-wrap gap-3">
              <button className="btn-primary" type="button" onClick={handleSubmit} disabled={submitting || Boolean(evaluation)}>
                {submitting ? "Evaluating..." : "Submit answer"}
              </button>
              {nextQuestion && (
                <button className="btn-secondary" type="button" onClick={handleContinue}>
                  Continue to next question
                </button>
              )}
              {canGenerateReport && (
                <button className="btn-primary" type="button" onClick={handleReport} disabled={reporting}>
                  {reporting ? "Generating..." : "Generate report"}
                </button>
              )}
            </div>
          </div>

          <aside className="space-y-6">
            <TracePanel question={currentQuestion} />
            {evaluation && adaptation && (
              <EvaluationCard evaluation={evaluation} adaptation={adaptation} />
            )}
          </aside>
        </section>
      ) : (
        <div className="card">No question found for this session. Return to the upload screen and start again.</div>
      )}
    </main>
  );
}

function QuestionCard({ question }: { question: Question }) {
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        <span className="tag">{question.topic}</span>
        <span className="tag capitalize">{question.difficulty}</span>
        <span className="tag capitalize">{question.question_type}</span>
        <span className="tag capitalize">{question.source_tier_used}</span>
      </div>
      <h2 className="text-2xl font-semibold leading-9 text-ink">{question.question_text}</h2>
      {question.expected_points.length > 0 && (
        <div className="rounded-md bg-mist p-4">
          <h3 className="label mb-2">Expected points</h3>
          <ul className="space-y-1 text-sm text-slate-700">
            {question.expected_points.map((point) => <li key={point}>- {point}</li>)}
          </ul>
        </div>
      )}
    </div>
  );
}

function TracePanel({ question }: { question: Question }) {
  return (
    <details className="card space-y-4" open>
      <summary className="text-lg font-semibold text-ink">RAG trace</summary>
      <div className="mt-4 rounded-md border border-blue-100 bg-blue-50 p-3 text-sm text-blue-900">
        {question.why_this_question}
      </div>
      <div className="space-y-3">
        {question.rag_trace.map((trace, index) => (
          <div key={`${trace.source_filename}-${index}`} className="rounded-md border border-line bg-white p-3">
            <div className="mb-2 flex flex-wrap gap-2">
              <span className="tag capitalize">{trace.tier}</span>
              <span className="tag">Page {trace.page_number ?? "n/a"}</span>
            </div>
            <p className="text-sm font-semibold text-ink">{trace.display_name}</p>
            <p className="text-xs text-slate-500">{trace.source_filename}</p>
            <p className="mt-2 text-sm font-medium text-slate-800">Context used: {trace.context_summary}</p>
            <p className="mt-2 text-xs text-slate-500">Query: {trace.retrieval_query}</p>
            <p className="mt-3 text-sm leading-6 text-slate-700">{trace.chunk_preview}</p>
          </div>
        ))}
      </div>
    </details>
  );
}

function EvaluationCard({ evaluation, adaptation }: { evaluation: Evaluation; adaptation: AdaptationDecision }) {
  return (
    <div className="card space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-ink">Evaluation</h2>
          <p className="muted">{evaluation.band}</p>
        </div>
        <div className="text-right">
          <p className="text-3xl font-semibold text-ink">{evaluation.score}/10</p>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-2">
        <Metric label="Accuracy" value={evaluation.technical_accuracy} />
        <Metric label="Clarity" value={evaluation.clarity} />
        <Metric label="Depth" value={evaluation.depth} />
      </div>
      <p className="rounded-md bg-mist p-3 text-sm leading-6 text-slate-700">{evaluation.feedback}</p>
      <List title="Strengths" items={evaluation.strengths} />
      <List title="Missing points" items={evaluation.missing_points} />
      <p className="text-sm leading-6 text-slate-700">{evaluation.ideal_answer_summary}</p>
      <p className="text-sm leading-6 text-slate-700">{evaluation.source_grounding_notes}</p>
      <div className="rounded-md border border-line bg-white p-3">
        <p className="label">Adaptation decision</p>
        <p className="mt-1 text-sm text-slate-700">{adaptation.reason}</p>
        <p className="mt-2 text-sm font-medium text-slate-800">
          Next difficulty: <span className="capitalize">{adaptation.next_difficulty}</span>
        </p>
        <p className="text-sm text-slate-700">
          Advanced sources unlocked: {adaptation.advanced_sources_unlocked ? "Yes" : "No"}
        </p>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-md bg-mist p-3 text-center">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="mt-1 text-lg font-semibold text-ink">{value}</p>
    </div>
  );
}

function List({ title, items }: { title: string; items: string[] }) {
  if (!items.length) return null;
  return (
    <div>
      <h3 className="label mb-2">{title}</h3>
      <ul className="space-y-1 text-sm text-slate-700">
        {items.map((item) => <li key={item}>- {item}</li>)}
      </ul>
    </div>
  );
}

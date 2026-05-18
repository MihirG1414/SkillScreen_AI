"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { generateReport, getSession, type Report } from "@/lib/api";
import { useParams } from "next/navigation";

export default function ReportPage() {
  const params = useParams<{ sessionId: string }>();
  const sessionId = params.sessionId;
  const [report, setReport] = useState<Report | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const snapshot = await getSession(sessionId);
        if (snapshot.session.report) {
          setReport(snapshot.session.report);
        } else {
          const generated = await generateReport(sessionId);
          setReport(generated.report);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unable to load report.");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [sessionId]);

  if (loading) {
    return <main className="shell"><div className="card">Loading report...</div></main>;
  }

  if (error || !report) {
    return (
      <main className="shell">
        <div className="card text-red-700">{error ?? "Report not found."}</div>
      </main>
    );
  }

  return (
    <main className="shell space-y-6">
      <header className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-sm font-semibold text-blue-700">Final report</p>
          <h1 className="mt-1 text-3xl font-semibold text-ink">Candidate screening summary</h1>
          <p className="muted mt-2">{report.candidate_summary}</p>
        </div>
        <Link className="btn-secondary" href="/interview/new">Start another</Link>
      </header>

      <section className="grid gap-4 md:grid-cols-3">
        <ScoreCard label="Overall score" value={`${report.overall_score}/10`} />
        <ScoreCard label="Role fit" value={report.role_fit} />
        <ScoreCard label="Recommendation" value={report.recommendation} />
      </section>

      <section className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="space-y-6">
          <Panel title="Strengths">
            <BulletList items={report.technical_strengths} />
          </Panel>
          <Panel title="Areas to improve">
            <BulletList items={report.areas_for_improvement} />
          </Panel>
          <Panel title="Advanced readiness">
            <p className="text-sm leading-6 text-slate-700">{report.advanced_readiness.summary}</p>
            <div className="mt-3 flex flex-wrap gap-2">
              <span className="tag">
                Ready: {report.advanced_readiness.ready_for_advanced_theory ? "Yes" : "No"}
              </span>
              <span className="tag">
                Triggered: {report.advanced_readiness.advanced_evaluation_triggered ? "Yes" : "No"}
              </span>
            </div>
          </Panel>
          <Panel title="Suggested next round questions">
            <BulletList items={report.suggested_next_round_questions} />
          </Panel>
        </div>

        <div className="space-y-6">
          <Panel title="Topic breakdown">
            <div className="space-y-3">
              {Object.entries(report.topic_breakdown).map(([topic, detail]) => (
                <div key={topic} className="rounded-md border border-line bg-white p-3">
                  <div className="flex items-center justify-between gap-3">
                    <p className="font-semibold text-ink">{topic}</p>
                    <span className="tag">{detail.average_score}/10</span>
                  </div>
                  <p className="muted mt-1">{detail.questions} question{detail.questions === 1 ? "" : "s"}</p>
                </div>
              ))}
            </div>
          </Panel>

          <Panel title="Source usage summary">
            <div className="grid grid-cols-2 gap-2">
              {Object.entries(report.source_usage_summary.chunks_by_tier).map(([tier, count]) => (
                <div key={tier} className="rounded-md bg-mist p-3">
                  <p className="text-xs capitalize text-slate-500">{tier}</p>
                  <p className="text-xl font-semibold text-ink">{count}</p>
                </div>
              ))}
            </div>
            <div className="mt-4">
              <p className="label mb-2">Source books used</p>
              <BulletList items={report.source_usage_summary.source_books_used} />
            </div>
            <p className="mt-3 text-sm text-slate-700">
              Advanced candidate evaluation triggered: {report.source_usage_summary.advanced_candidate_evaluation_triggered ? "Yes" : "No"}
            </p>
          </Panel>
        </div>
      </section>

      <Panel title="Q&A summary">
        <div className="space-y-4">
          {report.question_answer_summary.map((item) => (
            <details key={item.question_id} className="rounded-md border border-line bg-white p-4">
              <summary className="font-semibold text-ink">
                Q{item.number}: {item.topic} | {item.score}/10 | <span className="capitalize">{item.source_tier_used}</span>
              </summary>
              <div className="mt-4 space-y-3">
                <p className="text-sm leading-6 text-slate-800">{item.question_text}</p>
                <div className="rounded-md bg-mist p-3">
                  <p className="label mb-2">Full answer</p>
                  <p className="whitespace-pre-wrap text-sm leading-6 text-slate-700">{item.answer_text}</p>
                </div>
                <p className="text-xs text-slate-500">Summary: {item.answer_summary}</p>
                <p className="text-sm text-slate-700">{item.feedback}</p>
                <div className="flex flex-wrap gap-2">
                  {item.rag_trace.map((trace, index) => (
                    <span key={`${trace.source_filename}-${index}`} className="tag">
                      {trace.display_name} | {trace.tier} | {trace.context_summary}
                    </span>
                  ))}
                </div>
              </div>
            </details>
          ))}
        </div>
      </Panel>
    </main>
  );
}

function ScoreCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="card">
      <p className="text-sm text-slate-500">{label}</p>
      <p className="mt-2 text-3xl font-semibold text-ink">{value}</p>
    </div>
  );
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="card">
      <h2 className="mb-4 text-xl font-semibold text-ink">{title}</h2>
      {children}
    </section>
  );
}

function BulletList({ items }: { items: string[] }) {
  if (!items.length) {
    return <p className="muted">No items yet.</p>;
  }
  return (
    <ul className="space-y-2 text-sm leading-6 text-slate-700">
      {items.map((item) => <li key={item}>- {item}</li>)}
    </ul>
  );
}

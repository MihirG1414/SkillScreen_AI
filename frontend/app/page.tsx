import Link from "next/link";

export default function HomePage() {
  return (
    <main className="shell flex min-h-screen items-center">
      <section className="grid w-full gap-10 lg:grid-cols-[1.05fr_0.95fr] lg:items-center">
        <div className="space-y-7">
          <div className="inline-flex rounded-full border border-blue-100 bg-white px-3 py-1 text-sm font-medium text-blue-800">
            MVP demo
          </div>
          <div className="space-y-4">
            <h1 className="max-w-3xl text-5xl font-semibold tracking-normal text-ink sm:text-6xl">
              SkillScreen AI
            </h1>
            <p className="max-w-2xl text-xl leading-8 text-slate-700">
              Resume-aware RAG interview simulator for role-based technical screening.
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            <Link className="btn-primary" href="/interview/new">
              Start Interview
            </Link>
            <a className="btn-secondary" href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer">
              API Docs
            </a>
          </div>
        </div>

        <div className="card space-y-5">
          <div>
            <h2 className="text-xl font-semibold text-ink">Demo flow</h2>
            <p className="muted mt-1">Upload a resume, generate grounded questions, evaluate answers, and produce a recruiter-ready report.</p>
          </div>
          <div className="grid gap-3">
            {[
              "Resume profile extraction",
              "Role-specific ChromaDB retrieval",
              "Source trace for every question",
              "Adaptive answer evaluation",
              "Final screening report"
            ].map((item, index) => (
              <div key={item} className="flex items-center gap-3 rounded-md border border-line bg-mist px-3 py-3">
                <span className="flex h-7 w-7 items-center justify-center rounded-full bg-blue-600 text-sm font-semibold text-white">
                  {index + 1}
                </span>
                <span className="text-sm font-medium text-slate-800">{item}</span>
              </div>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}

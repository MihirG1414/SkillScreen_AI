"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { startInterview, uploadResume, type ResumeUploadResponse } from "@/lib/api";

const ROLES = [
  { value: "AI_ML_ENGINEER", label: "AI / ML Engineer" },
  { value: "BACKEND_ENGINEER", label: "Backend Engineer" },
  { value: "DATA_SCIENCE_APPLIED_ML", label: "Data Science / Applied ML" }
];

export default function NewInterviewPage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [role, setRole] = useState("BACKEND_ENGINEER");
  const [upload, setUpload] = useState<ResumeUploadResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleUpload() {
    if (!file) {
      setError("Choose a PDF or TXT resume first.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const result = await uploadResume(file, role);
      setUpload(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setLoading(false);
    }
  }

  async function handleStart() {
    if (!upload) return;
    setError(null);
    setStarting(true);
    try {
      await startInterview(upload.session_id);
      router.push(`/interview/${upload.session_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to start interview.");
    } finally {
      setStarting(false);
    }
  }

  return (
    <main className="shell space-y-6">
      <header className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-sm font-semibold text-blue-700">New interview</p>
          <h1 className="mt-1 text-3xl font-semibold text-ink">Upload resume and select role</h1>
          <p className="muted mt-2">The backend extracts a profile and stores a screening session.</p>
        </div>
      </header>

      <section className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="card space-y-5">
          <div className="space-y-2">
            <label className="label" htmlFor="role">Role</label>
            <select id="role" className="field" value={role} onChange={(event) => setRole(event.target.value)}>
              {ROLES.map((item) => (
                <option key={item.value} value={item.value}>{item.label}</option>
              ))}
            </select>
          </div>

          <div className="space-y-2">
            <label className="label" htmlFor="resume">Resume file</label>
            <input
              id="resume"
              className="field"
              type="file"
              accept=".pdf,.txt"
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            />
            <p className="text-xs text-slate-500">PDF and TXT resumes are supported for the MVP.</p>
          </div>

          {error && (
            <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
              {error}
            </div>
          )}

          <div className="flex flex-wrap gap-3">
            <button className="btn-primary" type="button" onClick={handleUpload} disabled={loading}>
              {loading ? "Uploading..." : "Upload Resume"}
            </button>
            <button className="btn-secondary" type="button" onClick={handleStart} disabled={!upload || starting}>
              {starting ? "Starting..." : "Start interview"}
            </button>
          </div>
        </div>

        <ProfilePreview upload={upload} />
      </section>
    </main>
  );
}

function ProfilePreview({ upload }: { upload: ResumeUploadResponse | null }) {
  if (!upload) {
    return (
      <div className="card flex min-h-72 items-center justify-center text-center">
        <div>
          <h2 className="text-xl font-semibold text-ink">Extracted profile appears here</h2>
          <p className="muted mt-2 max-w-md">Upload a resume to preview detected skills, seniority, and suggested interview topics.</p>
        </div>
      </div>
    );
  }

  const profile = upload.extracted_profile;
  return (
    <div className="card space-y-5">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <p className="text-sm font-semibold text-blue-700">{upload.selected_role}</p>
          <h2 className="text-2xl font-semibold text-ink">{profile.candidate_name ?? "Candidate profile"}</h2>
        </div>
        <span className="tag capitalize">{profile.seniority_level}</span>
      </div>

      {profile.summary && <p className="rounded-md bg-mist p-3 text-sm leading-6 text-slate-700">{profile.summary}</p>}

      <TagBlock title="Skills" items={profile.skills} />
      <TagBlock title="Programming languages" items={profile.programming_languages} />
      <TagBlock title="Frameworks" items={profile.frameworks} />
      <TagBlock title="Tools" items={profile.tools} />
      <ListBlock title="Suggested interview topics" items={profile.suggested_topics} />
      <ListBlock title="Project signals" items={profile.projects} />
    </div>
  );
}

function TagBlock({ title, items }: { title: string; items: string[] }) {
  if (!items.length) return null;
  return (
    <div>
      <h3 className="label mb-2">{title}</h3>
      <div className="flex flex-wrap gap-2">
        {items.map((item) => <span className="tag" key={item}>{item}</span>)}
      </div>
    </div>
  );
}

function ListBlock({ title, items }: { title: string; items: string[] }) {
  if (!items.length) return null;
  return (
    <div>
      <h3 className="label mb-2">{title}</h3>
      <ul className="space-y-2 text-sm text-slate-700">
        {items.map((item) => <li className="rounded-md border border-line bg-white px-3 py-2" key={item}>{item}</li>)}
      </ul>
    </div>
  );
}

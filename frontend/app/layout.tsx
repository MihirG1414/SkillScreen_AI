import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SkillScreen AI",
  description: "Resume-aware RAG interview simulator"
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

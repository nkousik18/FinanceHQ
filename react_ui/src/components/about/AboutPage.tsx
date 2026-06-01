import { motion } from "framer-motion";
import { Mail, GraduationCap, Briefcase, ExternalLink } from "lucide-react";

const GithubIcon = () => (
  <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24">
    <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
  </svg>
);

const LinkedinIcon = () => (
  <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24">
    <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
  </svg>
);

const fadeUp = (delay = 0) => ({
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.5, delay },
});

const inView = (delay = 0) => ({
  initial: { opacity: 0, y: 20 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true },
  transition: { duration: 0.45, delay },
});

const SKILLS = [
  { category: "Languages",   tags: ["Python", "SQL", "R"] },
  { category: "Cloud & Data", tags: ["AWS", "Redshift", "dbt", "Snowflake", "S3", "Textract"] },
  { category: "ML & AI",     tags: ["PyTorch", "TensorFlow", "LangChain", "FAISS", "MLflow", "Groq API"] },
  { category: "Tools",       tags: ["Docker", "Airflow", "FastAPI", "Django", "Tableau"] },
];

export function AboutPage() {
  return (
    <div className="max-w-5xl mx-auto px-6 py-10 pb-20">

      {/* ── Hero ── */}
      <motion.div {...fadeUp(0)} className="flex items-start gap-10 mb-12 flex-wrap">

        {/* left — identity */}
        <div className="flex-1" style={{ minWidth: 280 }}>
          <div className="flex items-center gap-4 mb-5">
            {/* monogram */}
            <div
              className="w-16 h-16 rounded-2xl flex items-center justify-center flex-shrink-0 text-xl font-bold"
              style={{
                background: "rgba(245,158,11,0.1)",
                border: "1px solid rgba(245,158,11,0.25)",
                color: "#f59e0b",
                letterSpacing: "-0.5px",
              }}
            >
              KN
            </div>
            <div>
              <h1 className="text-2xl font-bold leading-tight" style={{ color: "#f5f0e8" }}>
                Kousik Nandury
              </h1>
              <p className="text-sm mt-0.5" style={{ color: "#f59e0b" }}>
                Data Engineer · ML Engineer · Analytics Engineer
              </p>
            </div>
          </div>

          <p
            className="text-sm leading-relaxed mb-6"
            style={{ color: "rgba(245,240,232,0.75)", maxWidth: 480 }}
          >
            MS Data Analytics Engineering student at Northeastern University with 2 years of
            industry experience at Capgemini. I build end-to-end data and ML systems, from
            pipelines and model inference to deployment infrastructure.
          </p>

          {/* social links */}
          <div className="flex flex-wrap gap-2">
            {[
              { icon: GithubIcon,   label: "GitHub",   href: "https://github.com/nkousik18" },
              { icon: LinkedinIcon, label: "LinkedIn",  href: "https://www.linkedin.com/in/kousik-nandury" },
              { icon: Mail,         label: "Email",     href: "mailto:kousik.nandury@gmail.com" },
            ].map(({ icon: Icon, label, href }) => (
              <a
                key={label}
                href={href}
                target={label !== "Email" ? "_blank" : undefined}
                rel="noreferrer"
                className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all"
                style={{
                  background: "rgba(245,240,232,0.04)",
                  border: "1px solid rgba(245,240,232,0.08)",
                  color: "rgba(245,240,232,0.75)",
                }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLElement).style.color = "#f5f0e8";
                  (e.currentTarget as HTMLElement).style.borderColor = "rgba(245,158,11,0.3)";
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLElement).style.color = "rgba(245,240,232,0.75)";
                  (e.currentTarget as HTMLElement).style.borderColor = "rgba(245,240,232,0.08)";
                }}
              >
                <Icon className="w-3.5 h-3.5" />
                {label}
              </a>
            ))}
          </div>
        </div>

        {/* right — quick stats */}
        <div className="grid grid-cols-2 gap-3 flex-shrink-0" style={{ width: 210 }}>
          <div
            className="rounded-xl p-4 text-center col-span-2"
            style={{ background: "rgba(245,240,232,0.03)", border: "1px solid rgba(245,240,232,0.07)" }}
          >
            <p className="text-sm font-semibold" style={{ color: "#f5f0e8" }}>Northeastern University</p>
            <p className="text-xs mt-0.5" style={{ color: "#f59e0b" }}>MS Data Analytics Eng.</p>
            <p className="text-xs mt-0.5" style={{ color: "rgba(245,240,232,0.58)" }}>GPA 3.8 · Boston, MA</p>
          </div>
          {[
            { val: "2", sub: "Years Industry Exp." },
            { val: "May", sub: "2026 Available" },
          ].map(({ val, sub }) => (
            <div
              key={sub}
              className="rounded-xl p-4 text-center"
              style={{ background: "rgba(245,240,232,0.03)", border: "1px solid rgba(245,240,232,0.07)" }}
            >
              <p className="text-xl font-bold" style={{ color: "#f59e0b" }}>{val}</p>
              <p className="text-xs mt-1" style={{ color: "rgba(245,240,232,0.62)" }}>{sub}</p>
            </div>
          ))}
        </div>
      </motion.div>

      {/* ── Education + Experience ── */}
      <div className="grid md:grid-cols-2 gap-4 mb-10">

        {/* Education */}
        <motion.div
          {...inView(0)}
          className="rounded-2xl p-5"
          style={{ background: "rgba(245,240,232,0.02)", border: "1px solid rgba(245,240,232,0.07)" }}
        >
          <div className="flex items-center gap-3 mb-5">
            <div className="w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0"
              style={{ background: "rgba(245,158,11,0.1)" }}>
              <GraduationCap className="w-4 h-4" style={{ color: "#f59e0b" }} />
            </div>
            <h2 className="font-semibold" style={{ color: "#f5f0e8" }}>Education</h2>
          </div>
          <div className="flex flex-col gap-4">
            <div className="pl-3" style={{ borderLeft: "2px solid #f59e0b" }}>
              <p className="text-sm font-semibold" style={{ color: "#f5f0e8" }}>Northeastern University</p>
              <p className="text-xs mt-0.5" style={{ color: "#f59e0b" }}>MS Data Analytics Engineering</p>
              <p className="text-xs mt-1" style={{ color: "rgba(245,240,232,0.58)" }}>GPA 3.8 · Expected April 2026 · Boston, MA</p>
            </div>
            <div className="pl-3" style={{ borderLeft: "2px solid rgba(245,240,232,0.1)" }}>
              <p className="text-sm font-semibold" style={{ color: "#f5f0e8" }}>Mahindra University</p>
              <p className="text-xs mt-0.5" style={{ color: "rgba(245,240,232,0.70)" }}>B.Tech Electrical & Electronics</p>
              <p className="text-xs mt-1" style={{ color: "rgba(245,240,232,0.58)" }}>GPA 3.3 · 2018 – 2022 · Hyderabad, India</p>
            </div>
          </div>
        </motion.div>

        {/* Experience */}
        <motion.div
          {...inView(0.08)}
          className="rounded-2xl p-5"
          style={{ background: "rgba(245,240,232,0.02)", border: "1px solid rgba(245,240,232,0.07)" }}
        >
          <div className="flex items-center gap-3 mb-5">
            <div className="w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0"
              style={{ background: "rgba(52,211,153,0.1)" }}>
              <Briefcase className="w-4 h-4" style={{ color: "#34d399" }} />
            </div>
            <h2 className="font-semibold" style={{ color: "#f5f0e8" }}>Experience</h2>
          </div>
          <div className="pl-3" style={{ borderLeft: "2px solid #34d399" }}>
            <p className="text-sm font-semibold" style={{ color: "#f5f0e8" }}>Software Engineer</p>
            <p className="text-xs mt-0.5" style={{ color: "#34d399" }}>Capgemini</p>
            <p className="text-xs mt-1 mb-3" style={{ color: "rgba(245,240,232,0.58)" }}>June 2022 – May 2024 · Bengaluru, India</p>
            <ul className="flex flex-col gap-2">
              {[
                "Led 12-member team for Gen AI solution with 100% critical-error resolution",
                "Built AI meeting summarisation assistant reducing documentation time by 60%",
                "Improved system reliability by 35% through log analysis pipelines",
              ].map((point) => (
                <li key={point} className="flex gap-2 items-start text-xs"
                  style={{ color: "rgba(245,240,232,0.70)" }}>
                  <span className="mt-1.5 w-1 h-1 rounded-full flex-shrink-0" style={{ background: "#34d399" }} />
                  {point}
                </li>
              ))}
            </ul>
          </div>
        </motion.div>
      </div>

      {/* ── Technical Skills ── */}
      <motion.section {...inView(0)} className="mb-10">
        <SectionHeading>Technical Skills</SectionHeading>
        <div
          className="mt-4 rounded-2xl overflow-hidden"
          style={{ border: "1px solid rgba(245,240,232,0.07)" }}
        >
          {SKILLS.map((row, i) => (
            <div
              key={row.category}
              className="flex items-center gap-4 px-5 py-3.5 flex-wrap"
              style={{
                borderBottom: i < SKILLS.length - 1 ? "1px solid rgba(245,240,232,0.06)" : "none",
                background: i % 2 === 0 ? "rgba(245,240,232,0.02)" : "transparent",
              }}
            >
              <span
                className="text-xs font-medium uppercase tracking-widest flex-shrink-0"
                style={{ color: "rgba(245,240,232,0.52)", width: 100 }}
              >
                {row.category}
              </span>
              <div className="flex flex-wrap gap-1.5">
                {row.tags.map((tag) => (
                  <span
                    key={tag}
                    className="text-xs px-2.5 py-1 rounded-full transition-all cursor-default"
                    style={{
                      background: "rgba(245,240,232,0.05)",
                      border: "1px solid rgba(245,240,232,0.08)",
                      color: "rgba(245,240,232,0.75)",
                    }}
                    onMouseEnter={(e) => {
                      (e.currentTarget as HTMLElement).style.background = "rgba(245,158,11,0.08)";
                      (e.currentTarget as HTMLElement).style.borderColor = "rgba(245,158,11,0.25)";
                      (e.currentTarget as HTMLElement).style.color = "#f59e0b";
                    }}
                    onMouseLeave={(e) => {
                      (e.currentTarget as HTMLElement).style.background = "rgba(245,240,232,0.05)";
                      (e.currentTarget as HTMLElement).style.borderColor = "rgba(245,240,232,0.08)";
                      (e.currentTarget as HTMLElement).style.color = "rgba(245,240,232,0.75)";
                    }}
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </motion.section>

      {/* ── Open to Opportunities CTA ── */}
      <motion.div
        {...inView(0)}
        className="rounded-2xl p-7 mb-10 flex flex-col sm:flex-row items-center justify-between gap-6"
        style={{
          background: "linear-gradient(135deg, rgba(245,158,11,0.08) 0%, rgba(245,158,11,0.03) 100%)",
          border: "1px solid rgba(245,158,11,0.2)",
        }}
      >
        <div>
          <p className="text-base font-bold mb-1" style={{ color: "#f5f0e8" }}>
            Open to Opportunities
          </p>
          <p className="text-sm leading-relaxed" style={{ color: "rgba(245,240,232,0.75)" }}>
            Looking for Data Engineering, ML Engineering, and Analytics roles.<br />
            Available for full-time positions starting May 2026.
          </p>
        </div>
        <div className="flex gap-3 flex-shrink-0">
          <a
            href="https://www.linkedin.com/in/kousik-nandury"
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold transition-opacity"
            style={{ background: "#f59e0b", color: "#0f0d0b" }}
            onMouseEnter={(e) => ((e.currentTarget as HTMLElement).style.opacity = "0.85")}
            onMouseLeave={(e) => ((e.currentTarget as HTMLElement).style.opacity = "1")}
          >
            <LinkedinIcon /> Connect
          </a>
          <a
            href="mailto:kousik.nandury@gmail.com"
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold transition-all"
            style={{
              background: "rgba(245,240,232,0.05)",
              border: "1px solid rgba(245,240,232,0.1)",
              color: "rgba(245,240,232,0.6)",
            }}
            onMouseEnter={(e) => {
              (e.currentTarget as HTMLElement).style.color = "#f5f0e8";
              (e.currentTarget as HTMLElement).style.borderColor = "rgba(245,158,11,0.3)";
            }}
            onMouseLeave={(e) => {
              (e.currentTarget as HTMLElement).style.color = "rgba(245,240,232,0.6)";
              (e.currentTarget as HTMLElement).style.borderColor = "rgba(245,240,232,0.1)";
            }}
          >
            <Mail className="w-3.5 h-3.5" /> Email
          </a>
        </div>
      </motion.div>

      {/* ── About FinanceHQ ── */}
      <motion.section {...inView(0)}>
        <SectionHeading>About FinanceHQ</SectionHeading>
        <div
          className="mt-4 rounded-2xl p-5 flex items-start gap-5"
          style={{ background: "rgba(245,240,232,0.02)", border: "1px solid rgba(245,240,232,0.07)" }}
        >
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 text-lg"
            style={{ background: "rgba(245,158,11,0.1)", border: "1px solid rgba(245,158,11,0.15)" }}
          >
            📄
          </div>
          <div className="flex-1">
            <p className="text-sm font-semibold mb-1" style={{ color: "#f5f0e8" }}>
              End-to-end RAG portfolio project
            </p>
            <p className="text-sm leading-relaxed" style={{ color: "rgba(245,240,232,0.70)" }}>
              FinanceHQ is a production-grade loan document intelligence system built to demonstrate
              retrieval-augmented generation, intent-based prompt routing, epsilon-greedy A/B
              testing, and live MLflow tracking, deployed on AWS with FastAPI + Django.
            </p>
            <a
              href="https://github.com/nkousik18/FinanceHQ"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1.5 mt-3 text-xs font-medium transition-opacity"
              style={{ color: "#f59e0b" }}
              onMouseEnter={(e) => ((e.currentTarget as HTMLElement).style.opacity = "0.7")}
              onMouseLeave={(e) => ((e.currentTarget as HTMLElement).style.opacity = "1")}
            >
              <GithubIcon />
              View source on GitHub
              <ExternalLink className="w-3 h-3" />
            </a>
          </div>
        </div>
      </motion.section>

    </div>
  );
}

function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-3 mb-1">
      <span className="w-1 h-5 rounded-full flex-shrink-0" style={{ background: "#f59e0b" }} />
      <h2 className="text-lg font-semibold" style={{ color: "#f5f0e8" }}>{children}</h2>
    </div>
  );
}

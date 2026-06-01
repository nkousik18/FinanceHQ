import { motion } from "framer-motion";
import { Layers, Search, GitBranch } from "lucide-react";

const stages = [
  {
    icon: Layers,
    color: "#C084FC",
    bg: "rgba(192,132,252,0.1)",
    title: "Chunking + Embedding",
    points: [
      "~400-token sliding window, 60-token overlap",
      "Nearest heading stored as section label per chunk",
      "MiniLM-L6-v2 → 384-dim L2-normalised embeddings (inner product = cosine)",
      "Multi-doc: vstack new embeddings onto existing, full FAISS rebuild",
    ],
  },
  {
    icon: Search,
    color: "#22D3EE",
    bg: "rgba(34,211,238,0.1)",
    title: "Retrieval + Intent",
    points: [
      "FAISS IndexFlatIP top-k search (k=5 default, configurable 1–20)",
      "Zero-shot intent: cosine sim vs 5 description embeddings — no LLM call",
      "Index cached in-memory per session; loaded from S3 on first query only",
      "Secondary intents logged to MLflow for offline analysis",
    ],
  },
  {
    icon: GitBranch,
    color: "#34D399",
    bg: "rgba(52,211,153,0.1)",
    title: "Prompt Routing + A/B",
    points: [
      "5 intents × 2 variants = 10 prompt templates",
      "ε-greedy selector (10% explore) reads MLflow latency to pick winner",
      "v1: detailed prose · v2: structured concise output",
      "1200-word context budget enforced before LLM call",
    ],
  },
];

export function CoreStages() {
  return (
    <section className="mt-12">
      <SectionHeading>Core Stages</SectionHeading>
      <div className="grid md:grid-cols-3 gap-4 mt-5">
        {stages.map((s, i) => {
          const Icon = s.icon;
          return (
            <motion.div
              key={s.title}
              className="rounded-2xl p-5"
              style={{ background: "rgba(245,240,232,0.02)", border: "1px solid rgba(245,240,232,0.07)" }}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: i * 0.08 }}
            >
              <div className="flex items-center gap-3 mb-4">
                <div className="w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0"
                  style={{ background: s.bg }}>
                  <Icon className="w-4 h-4" style={{ color: s.color }} />
                </div>
                <h3 className="font-semibold text-sm" style={{ color: "#f5f0e8" }}>{s.title}</h3>
              </div>
              <ul className="flex flex-col gap-2.5">
                {s.points.map((p) => (
                  <li key={p} className="flex gap-2 items-start text-xs leading-relaxed"
                    style={{ color: "rgba(245,240,232,0.70)" }}>
                    <span className="mt-1.5 w-1 h-1 rounded-full flex-shrink-0"
                      style={{ background: s.color }} />
                    {p}
                  </li>
                ))}
              </ul>
            </motion.div>
          );
        })}
      </div>
    </section>
  );
}

function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-1 h-5 rounded-full flex-shrink-0" style={{ background: "#f59e0b" }} />
      <h2 className="text-xl font-bold" style={{ color: "#f5f0e8" }}>{children}</h2>
    </div>
  );
}

import { motion } from "framer-motion";
import { Layers, RefreshCw, GitMerge } from "lucide-react";

const decisions = [
  {
    icon: Layers,
    title: "Stateless API via S3",
    body: (
      <>
        Session state lives entirely in S3 as{" "}
        <code className="text-xs px-1 py-0.5 rounded" style={{ background: "rgba(245,158,11,0.1)", color: "#f59e0b" }}>
          status.json
        </code>
        . Any API replica can serve any request without shared memory — prerequisite for horizontal
        scaling.
      </>
    ),
  },
  {
    icon: RefreshCw,
    title: "One Model, Three Jobs",
    body: (
      <>
        MiniLM-L6-v2 loaded once, reused for chunk embedding, query embedding, and zero-shot intent
        classification. One ~90 MB load serves the full query path — no extra inference cost for intent.
      </>
    ),
  },
  {
    icon: GitMerge,
    title: "Merge-on-Upload Index",
    body: (
      <>
        Multi-doc sessions rebuild FAISS by loading existing S3 embeddings and vstacking new ones.{" "}
        <code className="text-xs px-1 py-0.5 rounded" style={{ background: "rgba(245,158,11,0.1)", color: "#f59e0b" }}>
          IndexFlatIP
        </code>{" "}
        doesn't support incremental add with persistence — full rebuild is correct.
      </>
    ),
  },
];

export function DesignDecisions() {
  return (
    <section className="mt-12">
      <div className="flex items-center gap-3 mb-5">
        <span className="w-1 h-5 rounded-full flex-shrink-0" style={{ background: "#f59e0b" }} />
        <h2 className="text-xl font-bold" style={{ color: "#f5f0e8" }}>Key Design Decisions</h2>
      </div>
      <div className="grid md:grid-cols-3 gap-4">
        {decisions.map((d, i) => {
          const Icon = d.icon;
          return (
            <motion.div
              key={d.title}
              className="rounded-2xl p-5"
              style={{ background: "rgba(245,240,232,0.02)", border: "1px solid rgba(245,240,232,0.07)" }}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: i * 0.08 }}
            >
              <div
                className="w-9 h-9 rounded-lg flex items-center justify-center mb-4"
                style={{ background: "rgba(245,158,11,0.1)" }}
              >
                <Icon className="w-4 h-4" style={{ color: "#f59e0b" }} />
              </div>
              <h3 className="font-semibold text-sm mb-2" style={{ color: "#f5f0e8" }}>{d.title}</h3>
              <p className="text-sm leading-relaxed" style={{ color: "rgba(245,240,232,0.70)" }}>
                {d.body}
              </p>
            </motion.div>
          );
        })}
      </div>
    </section>
  );
}

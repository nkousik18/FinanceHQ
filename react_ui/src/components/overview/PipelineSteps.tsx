import { motion } from "framer-motion";
import { FileText, Cpu, Filter, Layers, Database, Search, Zap } from "lucide-react";

const steps = [
  { icon: FileText, number: "01", title: "PDF Upload",      description: "Stored to S3. PyMuPDF extracts page count; Textract does the real work.", tech: ["S3", "PyMuPDF"] },
  { icon: Cpu,      number: "02", title: "Textract OCR",   description: "AnalyzeDocument with TABLES + FORMS. Handles handwriting and multi-column layouts.", tech: ["AWS Textract", "Async jobs"] },
  { icon: Filter,   number: "03", title: "Text Cleaning",  description: "Validator discards ERROR-confidence blocks. Whitespace normalised.", tech: ["Custom validator"] },
  { icon: Layers,   number: "04", title: "Chunk + Embed",  description: "400-token sliding window, 60-token overlap. MiniLM-L6-v2 → 384-dim vectors.", tech: ["MiniLM-L6-v2", "FAISS"] },
  { icon: Database, number: "05", title: "FAISS Index",    description: "IndexFlatIP in-process. Serialised to S3 on first load, cached per session.", tech: ["FAISS", "S3 cache"] },
  { icon: Search,   number: "06", title: "Intent + Retrieve", description: "Zero-shot cosine intent classification across 5 categories. Top-k retrieval.", tech: ["Zero-shot", "Top-k=5"] },
  { icon: Zap,      number: "07", title: "Prompt → LLM",  description: "ε-greedy A/B selector picks best variant from MLflow run data. SSE stream back.", tech: ["Groq SSE", "MLflow"] },
];

export function PipelineSteps() {
  return (
    <section className="py-16 px-6">
      <div className="max-w-6xl mx-auto">
        <motion.div
          className="mb-10 text-center"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <p
            className="text-xs font-medium tracking-widest uppercase mb-3"
            style={{ color: "#f59e0b" }}
          >
            Under the hood
          </p>
          <h2
            className="text-3xl md:text-4xl font-bold tracking-tight"
            style={{ color: "#f5f0e8" }}
          >
            Seven stages, raw PDF to streamed answer
          </h2>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
          {steps.map((step, i) => {
            const Icon = step.icon;
            return (
              <motion.div
                key={step.number}
                className="relative group p-5 rounded-xl transition-all duration-300 cursor-default"
                style={{
                  border: "1px solid rgba(245,240,232,0.06)",
                  background: "rgba(245,240,232,0.02)",
                }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLElement).style.border = "1px solid rgba(245,158,11,0.2)";
                  (e.currentTarget as HTMLElement).style.background = "rgba(245,158,11,0.04)";
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLElement).style.border = "1px solid rgba(245,240,232,0.06)";
                  (e.currentTarget as HTMLElement).style.background = "rgba(245,240,232,0.02)";
                }}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.45, delay: i * 0.06 }}
              >
                <div className="flex items-start justify-between mb-4">
                  <div
                    className="w-8 h-8 rounded-lg flex items-center justify-center"
                    style={{ background: "rgba(245,158,11,0.1)" }}
                  >
                    <Icon className="w-4 h-4" style={{ color: "#f59e0b" }} />
                  </div>
                  <span
                    className="text-xs font-mono"
                    style={{ color: "rgba(245,240,232,0.68)" }}
                  >
                    {step.number}
                  </span>
                </div>
                <h3
                  className="text-sm font-semibold mb-1.5"
                  style={{ color: "#f5f0e8" }}
                >
                  {step.title}
                </h3>
                <p
                  className="text-xs leading-relaxed mb-3"
                  style={{ color: "rgba(245,240,232,0.64)" }}
                >
                  {step.description}
                </p>
                <div className="flex flex-wrap gap-1">
                  {step.tech.map((t) => (
                    <span
                      key={t}
                      className="text-xs px-2 py-0.5 rounded font-mono"
                      style={{
                        background: "rgba(245,240,232,0.05)",
                        color: "rgba(245,240,232,0.58)",
                      }}
                    >
                      {t}
                    </span>
                  ))}
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
}

import { motion } from "framer-motion";

const stats = [
  { value: "<1.8s", label: "Avg query latency",   sub: "A/B selected model" },
  { value: "87%",   label: "Groundedness score",  sub: "answer ↔ context fidelity" },
  { value: "115+",  label: "Automated tests",     sub: "pytest · all passing" },
  { value: "3 LLMs",label: "A/B evaluated",       sub: "21 runs · MLflow tracked" },
];

export function StatsBar() {
  return (
    <div
      className="grid grid-cols-2 md:grid-cols-4"
      style={{
        borderTop:    "1px solid rgba(245,240,232,0.06)",
        borderBottom: "1px solid rgba(245,240,232,0.06)",
      }}
    >
      {stats.map((s, i) => (
        <motion.div
          key={s.label}
          className="flex flex-col gap-1 px-8 py-6"
          style={{
            background: i % 2 === 0 ? "rgba(245,240,232,0.01)" : "transparent",
            borderRight: i < 3 ? "1px solid rgba(245,240,232,0.06)" : "none",
          }}
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: i * 0.08 }}
        >
          <span
            className="text-3xl font-bold tracking-tight"
            style={{ color: "#f59e0b" }}
          >
            {s.value}
          </span>
          <span className="text-sm font-medium" style={{ color: "#f5f0e8" }}>
            {s.label}
          </span>
          <span className="text-xs" style={{ color: "rgba(245,240,232,0.62)" }}>
            {s.sub}
          </span>
        </motion.div>
      ))}
    </div>
  );
}

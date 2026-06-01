import { motion } from "framer-motion";

const models = [
  { name: "Llama 3.1 8B",  latency: "1.24s", groundedness: "79%", traffic: "0%",   badge: "Retired", winner: false },
  { name: "Llama 3.3 70B", latency: "2.02s", groundedness: "87%", traffic: "100%", badge: "Winner",  winner: true  },
  { name: "Llama 3.1 70B", latency: "2.02s", groundedness: "79%", traffic: "0%",   badge: "Baseline",winner: false },
];

export function ABResults() {
  return (
    <section
      className="py-16 px-6"
      style={{ borderTop: "1px solid rgba(245,240,232,0.05)" }}
    >
      <div className="max-w-6xl mx-auto">
        <motion.div
          className="mb-10"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <p
            className="text-xs font-medium tracking-widest uppercase mb-3"
            style={{ color: "#f59e0b" }}
          >
            Model evaluation
          </p>
          <h2
            className="text-3xl md:text-4xl font-bold tracking-tight"
            style={{ color: "#f5f0e8" }}
          >
            A/B tested across 21 MLflow runs
          </h2>
          <p className="mt-3 max-w-xl" style={{ color: "rgba(245,240,232,0.68)" }}>
            ε-greedy selector (10% explore) reads live latency from MLflow
            and routes traffic to the winning model automatically.
          </p>
        </motion.div>

        <motion.div
          className="rounded-2xl overflow-hidden"
          style={{ border: "1px solid rgba(245,240,232,0.06)" }}
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          {/* header */}
          <div
            className="grid grid-cols-4 px-6 py-3 text-xs font-medium uppercase tracking-wider"
            style={{
              background: "rgba(245,240,232,0.02)",
              borderBottom: "1px solid rgba(245,240,232,0.06)",
              color: "rgba(245,240,232,0.52)",
            }}
          >
            <span>Model</span>
            <span>Latency</span>
            <span>Groundedness</span>
            <span>Live traffic</span>
          </div>

          {models.map((m) => (
            <div
              key={m.name}
              className="grid grid-cols-4 px-6 py-4 items-center"
              style={{
                borderBottom: "1px solid rgba(245,240,232,0.05)",
                background: m.winner ? "rgba(245,158,11,0.04)" : "transparent",
              }}
            >
              <span
                className="text-sm font-medium"
                style={{ color: m.winner ? "#f5f0e8" : "rgba(245,240,232,0.68)" }}
              >
                {m.name}
              </span>
              <span
                className="text-sm font-mono"
                style={{ color: m.winner ? "#f5f0e8" : "rgba(245,240,232,0.62)" }}
              >
                {m.latency}
              </span>
              <span
                className="text-sm font-mono"
                style={{ color: m.winner ? "#f59e0b" : "rgba(245,240,232,0.62)" }}
              >
                {m.groundedness}
              </span>
              <div className="flex items-center gap-3">
                <span
                  className="text-sm font-mono"
                  style={{ color: m.winner ? "#f5f0e8" : "rgba(245,240,232,0.62)" }}
                >
                  {m.traffic}
                </span>
                <span
                  className="text-xs px-2 py-0.5 rounded-full font-medium"
                  style={
                    m.winner
                      ? { background: "rgba(245,158,11,0.18)", color: "#f59e0b" }
                      : { background: "rgba(245,240,232,0.05)", color: "rgba(245,240,232,0.52)" }
                  }
                >
                  {m.badge}
                </span>
              </div>
            </div>
          ))}

          <div
            className="px-6 py-3 flex items-center gap-2 text-xs"
            style={{
              background: "rgba(245,240,232,0.01)",
              color: "rgba(245,240,232,0.52)",
            }}
          >
            <span
              className="w-1.5 h-1.5 rounded-full"
              style={{ background: "rgba(245,158,11,0.6)" }}
            />
            Winning model auto-routed via ε-greedy A/B selector · metrics logged to MLflow
          </div>
        </motion.div>
      </div>
    </section>
  );
}

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { CheckCircle } from "lucide-react";
import { STEPS } from "./steps";

const AUTO_INTERVAL = 2800;

export function PipelineViewer() {
  const [active, setActive] = useState(0);
  const [paused, setPaused] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const go = (idx: number) => { setActive(idx); setPaused(true); };

  useEffect(() => {
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = setInterval(() => {
      if (!paused) setActive((a) => (a + 1) % STEPS.length);
    }, AUTO_INTERVAL);
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [paused]);

  const step = STEPS[active];

  return (
    <div
      className="rounded-2xl p-6"
      style={{ background: "rgba(245,240,232,0.02)", border: "1px solid rgba(245,240,232,0.07)" }}
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
    >
      {/* ── Pipeline track ── */}
      <div className="pb-2">
        <div className="flex items-end w-full" style={{ gap: 0, padding: "10px 4px 4px" }}>
          {STEPS.map((s, i) => (
            <div key={i} className="flex items-end" style={{ flex: i < STEPS.length - 1 ? "1 1 0" : "0 0 auto" }}>
              {/* node */}
              <button
                onClick={() => go(i)}
                className="flex flex-col items-center gap-1.5 flex-shrink-0 group"
                style={{ width: 88 }}
              >
                <div
                  className="relative w-14 h-14 rounded-xl flex items-center justify-center transition-all duration-300 text-xl"
                  style={{
                    background: i <= active ? s.bg : "rgba(245,240,232,0.03)",
                    border: i === active
                      ? `1px solid ${s.border}`
                      : i < active
                      ? `1px solid ${s.border}55`
                      : "1px solid rgba(245,240,232,0.08)",
                    boxShadow: i === active ? `0 0 20px ${s.color}22, 0 0 0 1px ${s.border}` : "none",
                    opacity: i < active ? 0.5 : 1,
                  }}
                >
                  <span style={{ filter: i < active ? "grayscale(1)" : "none" }}>{s.icon}</span>

                  {/* active pulse ring */}
                  {i === active && (
                    <motion.span
                      className="absolute inset-0 rounded-xl"
                      style={{ border: `1.5px solid ${s.color}` }}
                      initial={{ opacity: 0.6, scale: 1 }}
                      animate={{ opacity: 0, scale: 1.4 }}
                      transition={{ duration: 1.5, repeat: Infinity, ease: "easeOut" }}
                    />
                  )}

                  {/* done checkmark */}
                  {i < active && (
                    <span className="absolute -top-1.5 -right-1.5">
                      <CheckCircle className="w-3.5 h-3.5" style={{ color: s.color }} />
                    </span>
                  )}

                  {/* step number */}
                  <span
                    className="absolute -top-2 -right-2 w-4 h-4 rounded-full flex items-center justify-center font-bold"
                    style={{
                      background: "#0f0d0b",
                      border: "1px solid rgba(245,240,232,0.1)",
                      color: "rgba(245,240,232,0.58)",
                      fontSize: 9,
                    }}
                  >
                    {i + 1}
                  </span>
                </div>

                <div className="text-center">
                  <div
                    className="text-xs font-semibold leading-tight transition-colors"
                    style={{ color: i === active ? "#f5f0e8" : "rgba(245,240,232,0.62)" }}
                  >
                    {s.label}
                  </div>
                  <div className="text-xs" style={{ color: "rgba(245,240,232,0.70)" }}>
                    {s.sub}
                  </div>
                </div>
              </button>

              {/* connector */}
              {i < STEPS.length - 1 && (
                <div
                  className="mb-7 mx-2 rounded-full overflow-hidden"
                  style={{
                    flex: 1,
                    height: 2,
                    background: "rgba(245,240,232,0.06)",
                    position: "relative",
                  }}
                >
                  {i < active && (
                    <motion.div
                      className="absolute inset-0 rounded-full"
                      initial={{ scaleX: 0, originX: 0 }}
                      animate={{ scaleX: 1 }}
                      transition={{ duration: 0.4, ease: "easeOut" }}
                      style={{ background: `linear-gradient(90deg, ${STEPS[i].color}99, ${STEPS[i+1].color}99)` }}
                    />
                  )}
                  {i === active && (
                    <motion.div
                      className="absolute inset-0 rounded-full"
                      style={{
                        background: `repeating-linear-gradient(90deg, ${s.color} 0px, ${s.color} 4px, transparent 4px, transparent 9px)`,
                        backgroundSize: "18px 2px",
                      }}
                      animate={{ backgroundPositionX: ["0px", "18px"] }}
                      transition={{ duration: 0.55, repeat: Infinity, ease: "linear" }}
                    />
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* ── Dot indicators ── */}
      <div className="flex justify-center gap-2 my-4">
        {STEPS.map((_, i) => (
          <button
            key={i}
            onClick={() => go(i)}
            className="rounded-full transition-all duration-300"
            style={{
              height: 6,
              width: i === active ? 18 : 6,
              background: i === active ? step.color : "rgba(245,240,232,0.12)",
            }}
          />
        ))}
      </div>

      {/* ── Detail card ── */}
      <AnimatePresence mode="wait">
        <motion.div
          key={active}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.22, ease: "easeOut" }}
          className="rounded-xl p-5"
          style={{ background: "rgba(245,240,232,0.03)", border: "1px solid rgba(245,240,232,0.07)" }}
        >
          <div className="flex gap-6 flex-wrap">
            {/* left — title + bullet points */}
            <div className="flex-1" style={{ minWidth: 220 }}>
              <div className="flex items-center gap-2.5 mb-4">
                <div
                  className="w-8 h-8 rounded-lg flex items-center justify-center text-base flex-shrink-0"
                  style={{ background: step.bg, border: `1px solid ${step.border}` }}
                >
                  {step.icon}
                </div>
                <h3 className="font-semibold text-sm" style={{ color: "#f5f0e8" }}>
                  {step.title}
                </h3>
              </div>
              <ul className="flex flex-col gap-2">
                {step.points.map((p, i) => (
                  <li key={i} className="flex gap-2 items-start text-sm" style={{ color: "rgba(245,240,232,0.75)" }}>
                    <span className="mt-1.5 w-1 h-1 rounded-full flex-shrink-0" style={{ background: step.color }} />
                    {p}
                  </li>
                ))}
              </ul>
            </div>

            {/* right — files + tech */}
            <div className="flex-shrink-0" style={{ minWidth: 130 }}>
              <p className="text-xs font-medium mb-2 uppercase tracking-widest" style={{ color: "rgba(245,240,232,0.75)" }}>
                Files
              </p>
              <div className="flex flex-col gap-1 mb-4">
                {step.files.map((f) => (
                  <code
                    key={f}
                    className="text-xs px-2 py-0.5 rounded font-mono"
                    style={{
                      background: "rgba(245,240,232,0.04)",
                      border: "1px solid rgba(245,240,232,0.08)",
                      color: step.color,
                    }}
                  >
                    {f}
                  </code>
                ))}
              </div>
              <p className="text-xs font-medium mb-2 uppercase tracking-widest" style={{ color: "rgba(245,240,232,0.75)" }}>
                Tech
              </p>
              <div className="flex flex-wrap gap-1.5">
                {step.tech.map((t) => (
                  <span
                    key={t}
                    className="text-xs px-2 py-0.5 rounded-full"
                    style={{
                      background: "rgba(245,240,232,0.04)",
                      border: "1px solid rgba(245,240,232,0.08)",
                      color: "rgba(245,240,232,0.68)",
                    }}
                  >
                    {t}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

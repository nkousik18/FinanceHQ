import { useEffect, useState } from "react";

const TOKENS = [
  "The ", "interest ", "rate ", "on ", "this ", "loan ", "is ",
  "8.65% ", "per ", "annum ", "(floating), ", "linked ", "to ",
  "the ", "bank's ", "MCLR ", "with ", "a ", "spread ", "of ",
  "0.15%. ", "Subject ", "to ", "quarterly ", "revision ",
  "per ", "RBI ", "guidelines.",
];

export function TerminalDemo() {
  const [visibleTokens, setVisibleTokens] = useState(0);
  const [done, setDone] = useState(false);

  useEffect(() => {
    if (visibleTokens >= TOKENS.length) { setDone(true); return; }
    const t = setTimeout(
      () => setVisibleTokens((v) => v + 1),
      visibleTokens < 8 ? 80 : 55
    );
    return () => clearTimeout(t);
  }, [visibleTokens]);

  useEffect(() => {
    if (!done) return;
    const t = setTimeout(() => { setVisibleTokens(0); setDone(false); }, 4000);
    return () => clearTimeout(t);
  }, [done]);

  return (
    <div
      className="h-full w-full flex flex-col text-sm"
      style={{ fontFamily: "'Geist Mono', 'JetBrains Mono', monospace", background: "#0f0d0b" }}
    >
      {/* window chrome */}
      <div
        className="flex items-center gap-1.5 px-4 py-3 flex-shrink-0"
        style={{ borderBottom: "1px solid rgba(245,240,232,0.06)" }}
      >
        <span className="w-3 h-3 rounded-full" style={{ background: "#ff5f57" }} />
        <span className="w-3 h-3 rounded-full" style={{ background: "#febc2e" }} />
        <span className="w-3 h-3 rounded-full" style={{ background: "#28c840" }} />
        <span
          className="ml-4 text-xs"
          style={{ color: "rgba(245,240,232,0.75)" }}
        >
          FinanceHQ — POST /query/stream
        </span>
      </div>

      <div className="flex-1 overflow-auto p-6 flex flex-col gap-5">
        {/* request */}
        <div className="flex flex-col gap-1.5">
          <div
            className="text-xs mb-1 tracking-widest uppercase"
            style={{ color: "rgba(245,240,232,0.75)" }}
          >
            Request
          </div>
          <div>
            <span style={{ color: "rgba(245,240,232,0.62)" }}>$ session_id = </span>
            <span style={{ color: "#6ee7b7" }}>"a3f2…9c1d"</span>
          </div>
          <div>
            <span style={{ color: "rgba(245,240,232,0.62)" }}>$ question = </span>
            <span style={{ color: "#fcd34d" }}>"What is the interest rate?"</span>
          </div>
        </div>

        {/* pipeline trace */}
        <div className="flex flex-col gap-1">
          <div
            className="text-xs mb-1 tracking-widest uppercase"
            style={{ color: "rgba(245,240,232,0.75)" }}
          >
            Pipeline
          </div>
          <div style={{ color: "rgba(245,240,232,0.62)", fontSize: "0.75rem" }}>
            → retrieving top-5 chunks from FAISS…
          </div>
          <div style={{ color: "rgba(245,240,232,0.62)", fontSize: "0.75rem" }}>
            → intent classified:{" "}
            <span style={{ color: "#f59e0b" }}>lookup</span>
            {" · "}routing to{" "}
            <span style={{ color: "#f59e0b" }}>lookup_v2</span>
          </div>
        </div>

        {/* streaming response */}
        <div className="flex flex-col gap-2">
          <span
            className="inline-flex w-fit items-center px-2 py-0.5 rounded text-xs font-medium"
            style={{ background: "rgba(245,158,11,0.12)", color: "#f59e0b" }}
          >
            streaming
          </span>
          <p
            className="leading-relaxed"
            style={{ color: "rgba(245,240,232,0.75)" }}
          >
            {TOKENS.slice(0, visibleTokens).join("")}
            {!done && visibleTokens > 0 && (
              <span
                className="inline-block w-[2px] h-[1em] ml-0.5 animate-pulse align-middle"
                style={{ background: "#f59e0b" }}
              />
            )}
          </p>
        </div>

        {/* metadata tags */}
        {done && (
          <div className="flex items-center gap-2 flex-wrap">
            {["lookup_v2", "1.24s", "5 chunks", "843 words"].map((tag) => (
              <span
                key={tag}
                className="text-xs px-2.5 py-1 rounded-md"
                style={{
                  border: "1px solid rgba(245,240,232,0.08)",
                  color: "rgba(245,240,232,0.68)",
                }}
              >
                {tag}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

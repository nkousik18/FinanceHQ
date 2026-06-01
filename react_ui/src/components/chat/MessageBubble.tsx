import type { StreamMeta } from "@/lib/api";

interface UserBubbleProps {
  text: string;
}

interface AssistantBubbleProps {
  text: string;
  streaming?: boolean;
  meta?: StreamMeta | null;
  error?: string | null;
}

export function UserBubble({ text }: UserBubbleProps) {
  return (
    <div className="flex justify-end">
      <div
        className="max-w-lg px-4 py-3 rounded-2xl rounded-br-sm text-sm leading-relaxed"
        style={{
          background: "rgba(245,158,11,0.1)",
          border: "1px solid rgba(245,158,11,0.2)",
          color: "#f5f0e8",
        }}
      >
        {text}
      </div>
    </div>
  );
}

export function AssistantBubble({ text, streaming, meta, error }: AssistantBubbleProps) {
  return (
    <div className="flex flex-col gap-2 max-w-2xl">
      <div
        className="px-4 py-3 rounded-2xl rounded-bl-sm text-sm leading-relaxed"
        style={{
          background: "rgba(245,240,232,0.04)",
          border: "1px solid rgba(245,240,232,0.08)",
          color: error ? "#f87171" : "rgba(245,240,232,0.85)",
          whiteSpace: "pre-wrap",
        }}
      >
        {error ?? text}
        {streaming && (
          <span
            className="inline-block w-[2px] h-[14px] ml-0.5 rounded-sm align-middle animate-pulse"
            style={{ background: "#f59e0b" }}
          />
        )}
      </div>

      {meta && !streaming && (
        <div className="flex items-center gap-2 flex-wrap px-1">
          {[
            meta.variant,
            `${(meta.latency_ms / 1000).toFixed(2)}s`,
            `${meta.chunks_used} chunks`,
            meta.intent,
          ].map((tag) => (
            <span
              key={tag}
              className="text-xs px-2 py-0.5 rounded-md font-mono"
              style={{
                background: "rgba(245,240,232,0.04)",
                border: "1px solid rgba(245,240,232,0.08)",
                color: "rgba(245,240,232,0.58)",
              }}
            >
              {tag}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

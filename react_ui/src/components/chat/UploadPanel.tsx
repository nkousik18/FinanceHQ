import { RotateCcw } from "lucide-react";
import { UploadDropzone } from "./UploadDropzone";
import { DocumentCard } from "./DocumentCard";
import type { SessionStatus } from "@/lib/api";

interface Doc {
  filename: string;
  status: string;
  pages?: number;
  chunks?: number;
  error?: string | null;
}

interface Props {
  sessionId: string | null;
  sessionStatus: SessionStatus["status"] | null;
  docs: Doc[];
  onFiles: (files: File[]) => void;
  onReset: () => void;
}

const SESSION_STATUS_LABEL: Record<string, { dot: string; label: string }> = {
  EMPTY:      { dot: "#6b6055", label: "No session"   },
  PROCESSING: { dot: "#f59e0b", label: "Processing…"  },
  EXTRACTING: { dot: "#f59e0b", label: "Extracting…"  },
  CHUNKING:   { dot: "#f59e0b", label: "Chunking…"    },
  INDEXING:   { dot: "#f59e0b", label: "Indexing…"    },
  READY:      { dot: "#34d399", label: "Ready"         },
  FAILED:     { dot: "#f87171", label: "Failed"        },
};

export function UploadPanel({ sessionId, sessionStatus, docs, onFiles, onReset }: Props) {
  const statusCfg = sessionStatus
    ? SESSION_STATUS_LABEL[sessionStatus] ?? SESSION_STATUS_LABEL.PROCESSING
    : SESSION_STATUS_LABEL.EMPTY;

  const isProcessing = sessionStatus && !["READY", "FAILED", null].includes(sessionStatus);

  return (
    <div
      className="flex flex-col h-full"
      style={{
        borderRight: "1px solid rgba(245,240,232,0.06)",
      }}
    >
      {/* header */}
      <div
        className="flex items-center justify-between px-5 py-4 flex-shrink-0"
        style={{ borderBottom: "1px solid rgba(245,240,232,0.06)" }}
      >
        <div className="flex items-center gap-2">
          <span
            className="w-2 h-2 rounded-full flex-shrink-0"
            style={{
              background: statusCfg.dot,
              boxShadow: sessionStatus === "READY" ? "0 0 6px rgba(52,211,153,0.5)" : "none",
              animation: isProcessing ? "pulse 1.5s infinite" : "none",
            }}
          />
          <span className="text-xs font-medium" style={{ color: "rgba(245,240,232,0.75)" }}>
            {sessionId ? sessionId.slice(0, 8) + "…" : "No session"}
          </span>
          <span className="text-xs" style={{ color: statusCfg.dot }}>
            {statusCfg.label}
          </span>
        </div>

        {sessionId && (
          <button
            onClick={onReset}
            className="flex items-center gap-1.5 text-xs px-2 py-1 rounded-lg transition-colors"
            style={{ color: "rgba(245,240,232,0.58)" }}
            onMouseEnter={(e) => (e.currentTarget.style.color = "rgba(245,240,232,0.7)")}
            onMouseLeave={(e) => (e.currentTarget.style.color = "rgba(245,240,232,0.58)")}
          >
            <RotateCcw className="w-3 h-3" />
            New
          </button>
        )}
      </div>

      {/* body */}
      <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-3">
        <UploadDropzone onFiles={onFiles} disabled={!!isProcessing} />

        {docs.length > 0 && (
          <div className="flex flex-col gap-2 mt-1">
            {docs.map((doc, i) => (
              <DocumentCard key={i} {...doc} />
            ))}
          </div>
        )}

        {sessionStatus === "READY" && (
          <div className="mt-2">
            <p
              className="text-xs font-medium mb-2 px-1"
              style={{ color: "rgba(245,240,232,0.58)" }}
            >
              Try asking
            </p>
            <div className="flex flex-col gap-1.5">
              {[
                "What is the interest rate?",
                "Total repayment over full tenure",
                "Summarise this document",
                "Compare applicant incomes",
              ].map((q) => (
                <button
                  key={q}
                  className="text-left text-xs px-3 py-2 rounded-lg transition-all"
                  style={{
                    background: "rgba(245,240,232,0.03)",
                    border: "1px solid rgba(245,240,232,0.06)",
                    color: "rgba(245,240,232,0.75)",
                  }}
                  onMouseEnter={(e) => {
                    (e.currentTarget as HTMLElement).style.borderColor = "rgba(245,158,11,0.25)";
                    (e.currentTarget as HTMLElement).style.color = "rgba(245,240,232,0.8)";
                    (e.currentTarget as HTMLElement).style.background = "rgba(245,158,11,0.04)";
                  }}
                  onMouseLeave={(e) => {
                    (e.currentTarget as HTMLElement).style.borderColor = "rgba(245,240,232,0.06)";
                    (e.currentTarget as HTMLElement).style.color = "rgba(245,240,232,0.75)";
                    (e.currentTarget as HTMLElement).style.background = "rgba(245,240,232,0.03)";
                  }}
                  onClick={() => {
                    const event = new CustomEvent("fill-question", { detail: q });
                    window.dispatchEvent(event);
                  }}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

import { FileText, CheckCircle, AlertCircle, Loader } from "lucide-react";

interface Props {
  filename: string;
  status: string;
  pages?: number;
  chunks?: number;
  error?: string | null;
}

const STATUS_CONFIG: Record<string, { color: string; label: string; pulse: boolean }> = {
  UPLOADING:  { color: "#f59e0b", label: "Uploading…",  pulse: true  },
  PENDING:    { color: "#f59e0b", label: "Queued",       pulse: true  },
  EXTRACTING: { color: "#f59e0b", label: "Extracting…", pulse: true  },
  CHUNKING:   { color: "#f59e0b", label: "Chunking…",   pulse: true  },
  INDEXING:   { color: "#f59e0b", label: "Indexing…",   pulse: true  },
  READY:      { color: "#34d399", label: "Ready",        pulse: false },
  FAILED:     { color: "#f87171", label: "Failed",       pulse: false },
};

export function DocumentCard({ filename, status, pages, chunks, error }: Props) {
  const cfg = STATUS_CONFIG[status] ?? { color: "#f59e0b", label: status, pulse: true };
  const isReady = status === "READY";
  const isFailed = status === "FAILED";

  return (
    <div
      className="flex items-start gap-3 rounded-xl p-3 transition-all"
      style={{
        background: "rgba(245,240,232,0.03)",
        border: `1px solid ${isReady ? "rgba(52,211,153,0.15)" : isFailed ? "rgba(248,113,113,0.15)" : "rgba(245,240,232,0.06)"}`,
      }}
    >
      <div
        className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5"
        style={{ background: "rgba(245,158,11,0.08)" }}
      >
        <FileText className="w-4 h-4" style={{ color: "#f59e0b" }} />
      </div>

      <div className="flex-1 min-w-0">
        <p
          className="text-sm font-medium truncate"
          style={{ color: "#f5f0e8" }}
        >
          {filename}
        </p>

        <div className="flex items-center gap-1.5 mt-1">
          {cfg.pulse ? (
            <Loader className="w-3 h-3 animate-spin" style={{ color: cfg.color }} />
          ) : isReady ? (
            <CheckCircle className="w-3 h-3" style={{ color: cfg.color }} />
          ) : (
            <AlertCircle className="w-3 h-3" style={{ color: cfg.color }} />
          )}
          <span className="text-xs" style={{ color: cfg.color }}>{cfg.label}</span>
          {isReady && pages && (
            <span className="text-xs" style={{ color: "rgba(245,240,232,0.58)" }}>
              · {pages}p · {chunks} chunks
            </span>
          )}
        </div>

        {error && (
          <p className="text-xs mt-1" style={{ color: "#f87171" }}>{error}</p>
        )}
      </div>
    </div>
  );
}

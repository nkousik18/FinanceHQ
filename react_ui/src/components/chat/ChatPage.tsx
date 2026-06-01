import { useCallback, useEffect, useRef, useState } from "react";
import { UploadPanel } from "./UploadPanel";
import { ChatPanel } from "./ChatPanel";
import {
  createSession,
  uploadDocument,
  getSessionStatus,
  type SessionStatus,
} from "@/lib/api";

interface Doc {
  filename: string;
  status: string;
  pages?: number;
  chunks?: number;
  error?: string | null;
}

export function ChatPage() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionStatus, setSessionStatus] = useState<SessionStatus["status"] | null>(null);
  const [docs, setDocs] = useState<Doc[]>([]);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = () => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  };

  const startPolling = useCallback((sid: string) => {
    stopPolling();
    pollRef.current = setInterval(async () => {
      try {
        const s = await getSessionStatus(sid);
        setSessionStatus(s.status);
        setDocs(
          s.documents
            // deduplicate by doc_id (API can return duplicates)
            .filter((d, i, arr) => arr.findIndex((x) => x.doc_id === d.doc_id) === i)
            .map((d) => ({
              filename: d.filename,
              status: d.status,
              pages: d.pages,
              chunks: d.chunks,
              error: d.error,
            }))
        );
        if (s.status === "READY" || s.status === "FAILED") stopPolling();
      } catch {
        stopPolling();
      }
    }, 3000);
  }, []);

  useEffect(() => () => stopPolling(), []);

  const handleFiles = async (files: File[]) => {
    let sid = sessionId;

    if (!sid) {
      try {
        const s = await createSession();
        sid = s.session_id;
        setSessionId(sid);
        setSessionStatus("PROCESSING");
      } catch {
        return;
      }
    }

    // optimistic UI — show files as uploading immediately
    setDocs((prev) => [
      ...prev,
      ...files.map((f) => ({ filename: f.name, status: "UPLOADING" })),
    ]);

    for (const file of files) {
      try {
        await uploadDocument(sid!, file);
        setDocs((prev) =>
          prev.map((d) =>
            d.filename === file.name && d.status === "UPLOADING"
              ? { ...d, status: "PENDING" }
              : d
          )
        );
      } catch {
        setDocs((prev) =>
          prev.map((d) =>
            d.filename === file.name && d.status === "UPLOADING"
              ? { ...d, status: "FAILED", error: "Upload failed" }
              : d
          )
        );
      }
    }

    startPolling(sid!);
  };

  const handleReset = () => {
    stopPolling();
    setSessionId(null);
    setSessionStatus(null);
    setDocs([]);
  };

  return (
    <div
      className="flex h-[calc(100vh-56px)]"
      style={{ background: "transparent" }}
    >
      {/* left — upload panel, fixed width */}
      <div className="w-72 flex-shrink-0 flex flex-col">
        <UploadPanel
          sessionId={sessionId}
          sessionStatus={sessionStatus}
          docs={docs}
          onFiles={handleFiles}
          onReset={handleReset}
        />
      </div>

      {/* right — chat panel, fills remaining space */}
      <div className="flex-1 flex flex-col min-w-0">
        <ChatPanel
          sessionId={sessionId}
          sessionReady={sessionStatus === "READY"}
        />
      </div>
    </div>
  );
}

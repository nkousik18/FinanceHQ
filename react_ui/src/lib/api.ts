const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8001";

export interface Session {
  session_id: string;
  name: string;
  status: string;
}

export interface DocUpload {
  session_id: string;
  doc_id: string;
  filename: string;
  status: string;
}

export interface SessionStatus {
  session_id: string;
  status: "EMPTY" | "PROCESSING" | "EXTRACTING" | "CHUNKING" | "INDEXING" | "READY" | "FAILED";
  total_pages: number;
  total_chunks: number;
  documents: Array<{
    doc_id: string;
    filename: string;
    status: string;
    pages: number;
    chunks: number;
    error: string | null;
  }>;
  error: string | null;
}

export interface StreamMeta {
  done: true;
  intent: string;
  variant: string;
  chunks_used: number;
  context_words: number;
  latency_ms: number;
}

export async function createSession(): Promise<Session> {
  const res = await fetch(`${BASE}/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: "Demo Session" }),
  });
  if (!res.ok) throw new Error("Failed to create session");
  return res.json();
}

export async function uploadDocument(sessionId: string, file: File): Promise<DocUpload> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/sessions/${sessionId}/documents`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error("Failed to upload document");
  return res.json();
}

export async function getSessionStatus(sessionId: string): Promise<SessionStatus> {
  const res = await fetch(`${BASE}/sessions/${sessionId}/status`);
  if (!res.ok) throw new Error("Failed to get session status");
  return res.json();
}

export async function* streamQuery(
  sessionId: string,
  question: string
): AsyncGenerator<{ token?: string } | StreamMeta> {
  const res = await fetch(`${BASE}/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, question, top_k: 5 }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail || `Error ${res.status}`);
  }

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const payload = JSON.parse(line.slice(6));
      yield payload;
    }
  }
}

import { useEffect, useRef, useState } from "react";
import { Send, FileText } from "lucide-react";
import { UserBubble, AssistantBubble } from "./MessageBubble";
import { streamQuery, type StreamMeta } from "@/lib/api";

interface Message {
  role: "user" | "assistant";
  text: string;
  streaming?: boolean;
  meta?: StreamMeta | null;
  error?: string | null;
}

interface Props {
  sessionId: string | null;
  sessionReady: boolean;
}

export function ChatPanel({ sessionId, sessionReady }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // listen for suggested question clicks from UploadPanel
  useEffect(() => {
    const handler = (e: Event) => {
      setInput((e as CustomEvent<string>).detail);
      textareaRef.current?.focus();
    };
    window.addEventListener("fill-question", handler);
    return () => window.removeEventListener("fill-question", handler);
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const autoResize = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
  };

  const send = async () => {
    if (!input.trim() || !sessionId || !sessionReady || isStreaming) return;
    const question = input.trim();
    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";

    setMessages((prev) => [...prev, { role: "user", text: question }]);
    setMessages((prev) => [...prev, { role: "assistant", text: "", streaming: true, meta: null }]);
    setIsStreaming(true);

    try {
      for await (const chunk of streamQuery(sessionId, question)) {
        if ("token" in chunk && chunk.token !== undefined) {
          setMessages((prev) => {
            const msgs = [...prev];
            const last = msgs[msgs.length - 1];
            if (last.role === "assistant") last.text += chunk.token!;
            return msgs;
          });
        } else if ("done" in chunk && chunk.done) {
          setMessages((prev) => {
            const msgs = [...prev];
            const last = msgs[msgs.length - 1];
            if (last.role === "assistant") {
              last.streaming = false;
              last.meta = chunk as StreamMeta;
            }
            return msgs;
          });
        }
      }
    } catch (err) {
      setMessages((prev) => {
        const msgs = [...prev];
        const last = msgs[msgs.length - 1];
        if (last.role === "assistant") {
          last.streaming = false;
          last.error = err instanceof Error ? err.message : "Network error";
        }
        return msgs;
      });
    } finally {
      setIsStreaming(false);
    }
  };

  const placeholder = !sessionId
    ? "Upload a document to start chatting…"
    : !sessionReady
    ? "Processing document…"
    : "Ask anything about the document…";

  return (
    <div className="flex flex-col h-full">
      {/* messages */}
      <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-5">
        {messages.length === 0 ? (
          <EmptyState sessionReady={sessionReady} />
        ) : (
          messages.map((msg, i) =>
            msg.role === "user" ? (
              <UserBubble key={i} text={msg.text} />
            ) : (
              <AssistantBubble
                key={i}
                text={msg.text}
                streaming={msg.streaming}
                meta={msg.meta}
                error={msg.error}
              />
            )
          )
        )}
        <div ref={bottomRef} />
      </div>

      {/* input */}
      <div
        className="flex-shrink-0 p-4"
        style={{ borderTop: "1px solid rgba(245,240,232,0.06)" }}
      >
        <div
          className="flex items-end gap-3 rounded-xl p-3 transition-all"
          style={{
            background: "rgba(245,240,232,0.04)",
            border: `1px solid ${isStreaming ? "rgba(245,158,11,0.3)" : "rgba(245,240,232,0.1)"}`,
          }}
        >
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => { setInput(e.target.value); autoResize(); }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
            }}
            placeholder={placeholder}
            disabled={!sessionReady || isStreaming}
            rows={1}
            className="flex-1 bg-transparent resize-none outline-none text-sm leading-relaxed"
            style={{
              color: "#f5f0e8",
              caretColor: "#f59e0b",
              fontFamily: "inherit",
            }}
          />
          <button
            onClick={send}
            disabled={!input.trim() || !sessionReady || isStreaming}
            className="flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center transition-all"
            style={{
              background: input.trim() && sessionReady && !isStreaming
                ? "#f59e0b"
                : "rgba(245,240,232,0.06)",
            }}
          >
            <Send
              className="w-3.5 h-3.5"
              style={{
                color: input.trim() && sessionReady && !isStreaming
                  ? "#0f0d0b"
                  : "rgba(245,240,232,0.70)",
              }}
            />
          </button>
        </div>
        <p
          className="text-xs mt-2 text-center"
          style={{ color: "rgba(245,240,232,0.70)" }}
        >
          Shift+Enter for new line · Enter to send
        </p>
      </div>
    </div>
  );
}

function EmptyState({ sessionReady }: { sessionReady: boolean }) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center gap-4 text-center py-20">
      <div
        className="w-14 h-14 rounded-2xl flex items-center justify-center"
        style={{ background: "rgba(245,158,11,0.08)", border: "1px solid rgba(245,158,11,0.15)" }}
      >
        <FileText className="w-6 h-6" style={{ color: "#f59e0b" }} />
      </div>
      <div>
        <p className="text-sm font-medium" style={{ color: "rgba(245,240,232,0.7)" }}>
          {sessionReady ? "Document ready" : "No document loaded"}
        </p>
        <p className="text-xs mt-1" style={{ color: "rgba(245,240,232,0.58)" }}>
          {sessionReady
            ? "Ask anything about your loan document"
            : "Upload a PDF loan document on the left to get started"}
        </p>
      </div>
      {!sessionReady && (
        <div className="flex items-center gap-3 text-xs" style={{ color: "rgba(245,240,232,0.70)" }}>
          <Step n={1} label="Upload PDF" />
          <Arrow />
          <Step n={2} label="Processing" />
          <Arrow />
          <Step n={3} label="Ask anything" />
        </div>
      )}
    </div>
  );
}

function Step({ n, label }: { n: number; label: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span
        className="w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0"
        style={{ background: "rgba(245,240,232,0.06)", color: "rgba(245,240,232,0.58)" }}
      >
        {n}
      </span>
      <span>{label}</span>
    </div>
  );
}

function Arrow() {
  return <span style={{ color: "rgba(245,240,232,0.12)" }}>→</span>;
}

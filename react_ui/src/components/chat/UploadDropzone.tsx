import { useRef, useState } from "react";
import { Upload } from "lucide-react";

interface Props {
  onFiles: (files: File[]) => void;
  disabled?: boolean;
}

export function UploadDropzone({ onFiles, disabled }: Props) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handle = (files: FileList | null) => {
    if (!files) return;
    const pdfs = Array.from(files).filter((f) => f.type === "application/pdf");
    if (pdfs.length) onFiles(pdfs);
  };

  return (
    <div
      onClick={() => !disabled && inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); if (!disabled) setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        if (!disabled) handle(e.dataTransfer.files);
      }}
      className="relative flex flex-col items-center justify-center gap-3 rounded-xl cursor-pointer select-none transition-all duration-200"
      style={{
        padding: "28px 20px",
        border: dragging
          ? "1px solid rgba(245,158,11,0.6)"
          : "1px dashed rgba(245,240,232,0.12)",
        background: dragging
          ? "rgba(245,158,11,0.06)"
          : "rgba(245,240,232,0.02)",
        opacity: disabled ? 0.4 : 1,
        cursor: disabled ? "not-allowed" : "pointer",
      }}
    >
      <div
        className="w-10 h-10 rounded-xl flex items-center justify-center"
        style={{ background: "rgba(245,158,11,0.1)" }}
      >
        <Upload className="w-4 h-4" style={{ color: "#f59e0b" }} />
      </div>
      <div className="text-center">
        <p className="text-sm font-medium" style={{ color: "#f5f0e8" }}>
          Drop PDF here
        </p>
        <p className="text-xs mt-0.5" style={{ color: "rgba(245,240,232,0.62)" }}>
          or click to browse · max 20 MB
        </p>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept=".pdf"
        multiple
        className="hidden"
        onChange={(e) => handle(e.target.files)}
      />
    </div>
  );
}

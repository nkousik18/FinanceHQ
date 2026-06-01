import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { MoveRight, Upload } from "lucide-react";
import { Button } from "@/components/ui/button";

interface HeroProps {
  onTryDemo: () => void;
  onViewArchitecture: () => void;
}

function Hero({ onTryDemo, onViewArchitecture }: HeroProps) {
  const [titleNumber, setTitleNumber] = useState(0);
  const titles = useMemo(
    () => ["instant", "grounded", "streamed", "intelligent", "auditable"],
    []
  );

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      setTitleNumber((prev) => (prev === titles.length - 1 ? 0 : prev + 1));
    }, 2200);
    return () => clearTimeout(timeoutId);
  }, [titleNumber, titles]);

  return (
    <div className="w-full flex items-center">
      <div className="container mx-auto px-6">
        <div className="flex gap-6 pt-24 pb-16 lg:pt-32 lg:pb-20 items-center justify-center flex-col">

          {/* pill badge */}
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <span
              className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-medium tracking-wide uppercase"
              style={{
                border: "1px solid rgba(245,158,11,0.3)",
                background: "rgba(245,158,11,0.08)",
                color: "#f59e0b",
              }}
            >
              <span
                className="w-1.5 h-1.5 rounded-full animate-pulse"
                style={{ background: "#f59e0b" }}
              />
              RAG · Textract · FAISS · Groq SSE
            </span>
          </motion.div>

          {/* headline */}
          <motion.div
            className="flex gap-3 flex-col items-center"
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
          >
            <h1
              className="text-5xl md:text-7xl max-w-3xl tracking-tight text-center font-bold leading-[1.1]"
              style={{ color: "#f5f0e8" }}
            >
              Loan answers,{" "}
              <span className="relative inline-flex justify-center overflow-hidden pb-2 pt-1 align-bottom">
                <span className="invisible">intelligent</span>
                {titles.map((title, index) => (
                  <motion.span
                    key={index}
                    className="absolute"
                    style={{ color: "#f59e0b" }}
                    initial={{ opacity: 0, y: 60 }}
                    transition={{ type: "spring", stiffness: 60, damping: 14 }}
                    animate={
                      titleNumber === index
                        ? { y: 0, opacity: 1 }
                        : { y: titleNumber > index ? -60 : 60, opacity: 0 }
                    }
                  >
                    {title}
                  </motion.span>
                ))}
              </span>
            </h1>

            <p
              className="text-lg md:text-xl leading-relaxed max-w-xl text-center font-normal mt-2"
              style={{ color: "rgba(245,240,232,0.70)" }}
            >
              Upload a loan PDF. Ask anything. Get document-grounded answers in
              real time with no hallucinations, no guessing.
            </p>
          </motion.div>

          {/* CTA buttons */}
          <motion.div
            className="flex flex-row gap-3 mt-2"
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.25 }}
          >
            <Button size="lg" className="gap-2" onClick={onTryDemo}>
              <Upload className="w-4 h-4" />
              Try the demo
            </Button>
            <Button size="lg" variant="outline" className="gap-2" onClick={onViewArchitecture}>
              How it works
              <MoveRight className="w-4 h-4" />
            </Button>
          </motion.div>

        </div>
      </div>
    </div>
  );
}

export { Hero };

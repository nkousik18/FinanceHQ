import { Hero } from "@/components/ui/animated-hero";
import { ContainerScroll } from "@/components/ui/container-scroll-animation";
import { SectionDivider } from "@/components/ui/SectionDivider";
import { StatsBar } from "./StatsBar";
import { PipelineSteps } from "./PipelineSteps";
import { ABResults } from "./ABResults";
import { TerminalDemo } from "./TerminalDemo";

interface OverviewPageProps {
  onTryDemo: () => void;
  onViewArchitecture: () => void;
}

export function OverviewPage({ onTryDemo, onViewArchitecture }: OverviewPageProps) {
  return (
    <div className="w-full">
      <Hero onTryDemo={onTryDemo} onViewArchitecture={onViewArchitecture} />

      <StatsBar />

      <SectionDivider />

      <ContainerScroll
        titleComponent={
          <div className="mb-4">
            <p
              className="text-xs font-medium tracking-widest uppercase mb-4"
              style={{ color: "#f59e0b" }}
            >
              Live demo
            </p>
            <h2
              className="text-3xl md:text-5xl font-bold tracking-tight"
              style={{ color: "#f5f0e8" }}
            >
              Watch the pipeline run
            </h2>
            <p
              className="mt-3 text-base max-w-lg mx-auto"
              style={{ color: "rgba(245,240,232,0.68)" }}
            >
              Real SSE token stream. Real FAISS retrieval. Real Textract extraction.
            </p>
          </div>
        }
      >
        <TerminalDemo />
      </ContainerScroll>

      <SectionDivider />

      <PipelineSteps />

      <SectionDivider />

      <ABResults />
    </div>
  );
}

import { motion } from "framer-motion";
import { PipelineViewer } from "./PipelineViewer";
import { CoreStages } from "./CoreStages";
import { DesignDecisions } from "./DesignDecisions";

export function ArchitecturePage() {
  return (
    <div className="max-w-6xl mx-auto px-6 py-10 pb-20">

      {/* header */}
      <motion.div
        className="mb-8"
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <p className="text-xs font-medium tracking-widest uppercase mb-2" style={{ color: "#f59e0b" }}>
          System design
        </p>
        <h1 className="text-3xl font-bold mb-2" style={{ color: "#f5f0e8" }}>
          System Architecture
        </h1>
        <p className="text-sm" style={{ color: "rgba(245,240,232,0.68)" }}>
          Seven stages — from raw PDF to grounded, streamed answer. Click any stage to explore.
        </p>
      </motion.div>

      {/* pipeline */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
      >
        <PipelineViewer />
      </motion.div>

      <CoreStages />
      <DesignDecisions />
    </div>
  );
}

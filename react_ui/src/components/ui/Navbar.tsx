import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

type Tab = "overview" | "chat" | "architecture" | "about";

interface NavbarProps {
  activeTab: Tab;
  onTabChange: (tab: Tab) => void;
}

const tabs: { id: Tab; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "chat", label: "Chat" },
  { id: "architecture", label: "Architecture" },
  { id: "about", label: "About" },
];

export function Navbar({ activeTab, onTabChange }: NavbarProps) {
  return (
    <header
      className="fixed top-0 left-0 right-0 z-50 border-b"
      style={{
        borderColor: "rgba(245,240,232,0.06)",
        background: "rgba(15,13,11,0.85)",
        backdropFilter: "blur(16px)",
      }}
    >
      <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">
        <button
          onClick={() => onTabChange("overview")}
          className="text-base font-bold tracking-tight transition-colors"
          style={{ color: "#f5f0e8" }}
        >
          Finance<span style={{ color: "#f59e0b" }}>HQ</span>
        </button>

        <nav className="flex items-center gap-1">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={cn(
                "relative px-4 py-1.5 text-sm font-medium rounded-md transition-colors",
                activeTab === tab.id
                  ? "text-[#f5f0e8]"
                  : "text-[#f5f0e8]/40 hover:text-[#f5f0e8]/70"
              )}
            >
              {activeTab === tab.id && (
                <motion.span
                  layoutId="nav-pill"
                  className="absolute inset-0 rounded-md"
                  style={{
                    background: "rgba(245,240,232,0.07)",
                    border: "1px solid rgba(245,240,232,0.1)",
                  }}
                  transition={{ type: "spring", stiffness: 350, damping: 30 }}
                />
              )}
              <span className="relative">{tab.label}</span>
            </button>
          ))}
        </nav>
      </div>
    </header>
  );
}

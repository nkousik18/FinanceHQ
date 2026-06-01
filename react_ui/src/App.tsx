import { useState } from "react";
import { Navbar } from "@/components/ui/Navbar";
import { BackgroundOrbs } from "@/components/ui/BackgroundOrbs";
import { OverviewPage } from "@/components/overview/OverviewPage";
import { ChatPage } from "@/components/chat/ChatPage";
import { ArchitecturePage } from "@/components/architecture/ArchitecturePage";
import { AboutPage } from "@/components/about/AboutPage";

type Tab = "overview" | "chat" | "architecture" | "about";


export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>("overview");

  const switchTab = (tab: Tab) => {
    setActiveTab(tab);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <div className="min-h-screen" style={{ background: "#0f0d0b", color: "#f5f0e8" }}>
      <BackgroundOrbs />
      <Navbar activeTab={activeTab} onTabChange={switchTab} />
      <main className="pt-14 relative z-10">
        {activeTab === "overview" && (
          <OverviewPage
            onTryDemo={() => switchTab("chat")}
            onViewArchitecture={() => switchTab("architecture")}
          />
        )}
        {activeTab === "chat" && <ChatPage />}
        {activeTab === "architecture" && <ArchitecturePage />}
        {activeTab === "about" && <AboutPage />}
      </main>
    </div>
  );
}

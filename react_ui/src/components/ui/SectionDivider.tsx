export function SectionDivider() {
  return (
    <div className="relative flex items-center px-8 md:px-24 py-2">
      <div
        className="flex-1 h-px"
        style={{
          background:
            "linear-gradient(90deg, transparent, rgba(245,158,11,0.15) 30%, rgba(245,158,11,0.08) 70%, transparent)",
        }}
      />
      <div
        className="mx-4 w-1 h-1 rounded-full flex-shrink-0"
        style={{ background: "rgba(245,158,11,0.3)" }}
      />
      <div
        className="flex-1 h-px"
        style={{
          background:
            "linear-gradient(90deg, transparent, rgba(245,158,11,0.08) 30%, rgba(245,158,11,0.15) 70%, transparent)",
        }}
      />
    </div>
  );
}

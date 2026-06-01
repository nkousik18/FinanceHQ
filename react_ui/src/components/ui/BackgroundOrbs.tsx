export function BackgroundOrbs() {
  return (
    <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden">

      {/* primary light source — top-left, concentrated amber */}
      <div
        className="absolute -top-20 -left-20 w-[500px] h-[500px] rounded-full"
        style={{
          background: "radial-gradient(circle, #f59e0b 0%, #d97706 30%, transparent 65%)",
          filter: "blur(60px)",
          opacity: 0.13,
        }}
      />

      {/* secondary fill — bottom-right deep orange, more diffuse */}
      <div
        className="absolute -bottom-40 -right-20 w-[550px] h-[550px] rounded-full"
        style={{
          background: "radial-gradient(circle, #ea580c 0%, #9a3412 40%, transparent 70%)",
          filter: "blur(90px)",
          opacity: 0.07,
        }}
      />

      {/* vignette — pulls eye inward, darkens edges */}
      <div
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse 80% 80% at 50% 40%, transparent 40%, rgba(10,8,6,0.55) 100%)",
        }}
      />
    </div>
  );
}

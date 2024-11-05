export function BackgroundEffects() {
  return (
    <div className="fixed inset-0">
      <div className="absolute inset-0 bg-black" />
      <div className="absolute top-1/4 left-1/4 w-1/2 h-1/2 bg-blue-500/10 rounded-full filter blur-3xl" />
      <div className="absolute bottom-1/4 right-1/4 w-1/3 h-1/3 bg-blue-300/5 rounded-full filter blur-3xl" />
    </div>
  );
}
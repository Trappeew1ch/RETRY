export function CodeBlock({ code }: { code: string }) {
  return (
    <pre className="mt-2 p-3 bg-black/30 rounded-lg overflow-x-auto">
      <code className="text-sm text-gray-300 font-mono">{code}</code>
    </pre>
  );
}
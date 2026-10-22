interface LoadingLineProps {
  label: string;
  detail?: string;
}

export function LoadingLine({ label, detail }: LoadingLineProps) {
  return (
    <div className="loading-line" role="status" aria-live="polite">
      <span className="loading-line__dots" aria-hidden="true">
        <span />
        <span />
        <span />
      </span>
      <span className="loading-line__label">{label}</span>
      {detail ? <span className="loading-line__detail">{detail}</span> : null}
    </div>
  );
}

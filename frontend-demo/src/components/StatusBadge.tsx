export type BadgeTone = 'success' | 'warning' | 'danger' | 'neutral' | 'info';

interface StatusBadgeProps {
  tone: BadgeTone;
  label: string;
  title?: string;
  dot?: boolean;
}

export function StatusBadge({ tone, label, title, dot = false }: StatusBadgeProps) {
  return (
    <span className={`status-badge status-badge--${tone}`} title={title}>
      {dot ? <span className="status-badge__dot" aria-hidden="true" /> : null}
      {label}
    </span>
  );
}

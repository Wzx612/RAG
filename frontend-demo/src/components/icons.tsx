import { useId } from 'react';

interface IconProps {
  className?: string;
}

function svgProps(className?: string) {
  return {
    className: className ? `icon ${className}` : 'icon',
    width: 18,
    height: 18,
    viewBox: '0 0 24 24',
    fill: 'none',
    stroke: 'currentColor',
    strokeWidth: 1.8,
    strokeLinecap: 'round' as const,
    strokeLinejoin: 'round' as const,
    'aria-hidden': true,
    focusable: 'false' as const,
  };
}

export function BrandMark({ className }: IconProps) {
  const gradientId = `brand-bg-${useId().replace(/[^a-zA-Z0-9_-]/g, '')}`;
  return (
    <svg
      className={className ? `brand-mark ${className}` : 'brand-mark'}
      width="32"
      height="32"
      viewBox="0 0 64 64"
      role="img"
      aria-label="信用卡智能客服助手"
    >
      <defs>
        <linearGradient
          id={gradientId}
          x1="8"
          y1="4"
          x2="56"
          y2="60"
          gradientUnits="userSpaceOnUse"
        >
          <stop stopColor="#123A68" />
          <stop offset="1" stopColor="#0B2442" />
        </linearGradient>
      </defs>
      <rect width="64" height="64" rx="16" fill={`url(#${gradientId})`} />
      <path
        d="M32 10 50 17v14c0 11-7.4 18.8-18 23-10.6-4.2-18-12-18-23V17l18-7Z"
        fill="none"
        stroke="#DCEBFF"
        strokeWidth="2.4"
        strokeLinejoin="round"
      />
      <ellipse cx="32" cy="25" rx="9" ry="3.6" fill="none" stroke="#7FB4FF" strokeWidth="2" />
      <path d="M23 25v9c0 2 4 3.6 9 3.6s9-1.6 9-3.6v-9" fill="none" stroke="#7FB4FF" strokeWidth="2" />
      <path
        d="M23 30c0 2 4 3.6 9 3.6s9-1.6 9-3.6"
        fill="none"
        stroke="#7FB4FF"
        strokeWidth="2"
      />
    </svg>
  );
}

export function SendIcon({ className }: IconProps) {
  return (
    <svg {...svgProps(className)}>
      <path d="M4.5 12 20 4l-4.8 16-4.1-6.4L4.5 12Z" />
      <path d="m11.1 13.6 3.9-3.9" />
    </svg>
  );
}

export function RefreshIcon({ className }: IconProps) {
  return (
    <svg {...svgProps(className)}>
      <path d="M20 11a8 8 0 0 0-13.7-5.6L3 8" />
      <path d="M3 3v5h5" />
      <path d="M4 13a8 8 0 0 0 13.7 5.6L21 16" />
      <path d="M21 21v-5h-5" />
    </svg>
  );
}

export function CloseIcon({ className }: IconProps) {
  return (
    <svg {...svgProps(className)}>
      <path d="M6 6l12 12M18 6 6 18" />
    </svg>
  );
}

export function ChevronDownIcon({ className }: IconProps) {
  return (
    <svg {...svgProps(className)}>
      <path d="m6 9 6 6 6-6" />
    </svg>
  );
}

export function ArrowUpRightIcon({ className }: IconProps) {
  return (
    <svg {...svgProps(className)}>
      <path d="M7 17 17 7" />
      <path d="M8 7h9v9" />
    </svg>
  );
}

export function CopyIcon({ className }: IconProps) {
  return (
    <svg {...svgProps(className)}>
      <rect x="9" y="9" width="11" height="11" rx="2" />
      <path d="M5 15V6a2 2 0 0 1 2-2h9" />
    </svg>
  );
}

export function CheckIcon({ className }: IconProps) {
  return (
    <svg {...svgProps(className)}>
      <path d="m5 12.5 4.2 4.2L19 7" />
    </svg>
  );
}

export function AlertIcon({ className }: IconProps) {
  return (
    <svg {...svgProps(className)}>
      <path d="M12 3.6 21 20H3L12 3.6Z" />
      <path d="M12 9.5v4.6" />
      <path d="M12 17.3h.01" />
    </svg>
  );
}

export function InfoIcon({ className }: IconProps) {
  return (
    <svg {...svgProps(className)}>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M12 11v5" />
      <path d="M12 8h.01" />
    </svg>
  );
}

export function SpinnerIcon({ className }: IconProps) {
  return (
    <svg {...svgProps(className ? `spin ${className}` : 'spin')}>
      <path d="M12 3a9 9 0 1 0 9 9" />
    </svg>
  );
}

export function LayersIcon({ className }: IconProps) {
  return (
    <svg {...svgProps(className)}>
      <path d="m12 3 8 4.5-8 4.5-8-4.5L12 3Z" />
      <path d="m4 12 8 4.5 8-4.5" />
      <path d="m4 16.5 8 4.5 8-4.5" />
    </svg>
  );
}

export function ExternalLinkIcon({ className }: IconProps) {
  return (
    <svg {...svgProps(className)}>
      <path d="M14 4h6v6" />
      <path d="M20 4 11 13" />
      <path d="M18 14v4a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4" />
    </svg>
  );
}

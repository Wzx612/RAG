import { useEffect, useRef } from 'react';

export function useAutoScroll<T extends HTMLElement>(dependency: unknown, enabled = true) {
  const ref = useRef<T | null>(null);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    if (!enabled) {
      // Welcome screen (or a cleared conversation): keep the heading visible.
      element.scrollTop = 0;
      return;
    }
    const prefersReduced =
      typeof window.matchMedia === 'function' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    element.scrollTo({
      top: element.scrollHeight,
      behavior: prefersReduced ? 'auto' : 'smooth',
    });
  }, [dependency, enabled]);

  return ref;
}

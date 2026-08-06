import { useState } from 'react';

interface CardArtProps {
  src?: string | null;
  alt: string;
  className?: string;
}

export function CardArt({ src, alt, className = '' }: CardArtProps) {
  const [failed, setFailed] = useState(false);
  const source = src && !failed ? src : '/card-back.png';
  return (
    <img
      className={className}
      src={source}
      alt={alt}
      loading="lazy"
      decoding="async"
      onError={() => setFailed(true)}
    />
  );
}


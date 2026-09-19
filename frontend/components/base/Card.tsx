import { ReactNode } from 'react';

interface CardProps {
  children: ReactNode;
  className?: string;
  elevated?: boolean;
  padding?: string;
}

const Card = ({
  children,
  className = '',
  elevated = false,
  padding = 'p-6',
}: CardProps) => {
  const baseClasses = 'bg-canvas-elevated hairline-border rounded-md transition-all duration-150 hover:floating-shadow';
  const elevatedClasses = elevated ? 'floating-shadow' : 'whisper-shadow';

  return (
    <div className={`${baseClasses} ${elevatedClasses} ${padding} ${className}`}>
      {children}
    </div>
  );
};

export default Card;
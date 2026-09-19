import { ReactNode } from 'react';

interface ButtonProps {
  children: ReactNode;
  variant?: 'primary' | 'secondary' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
  disabled?: boolean;
  asChild?: boolean;
  type?: 'button' | 'submit' | 'reset';
  onClick?: () => void;
}

const Button = ({
  children,
  variant = 'primary',
  size = 'md',
  className = '',
  disabled = false,
  asChild = false,
  type = 'button',
  onClick,
}: ButtonProps) => {
  const baseClasses = 'font-geist-mono font-weight-500 transition-all duration-150 hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed disabled:scale-100';

  const variantClasses = {
    primary: 'bg-ink text-on-primary rounded-pill px-6 py-2 hover:bg-ink/90',
    secondary: 'bg-canvas-elevated text-ink hairline-border rounded-pill px-6 py-2 hover:bg-canvas-elevated/90',
    ghost: 'bg-canvas-elevated text-ink hairline-border rounded-full px-4 py-1.5 hover:bg-canvas-elevated/90',
  };

  const sizeClasses = {
    sm: 'px-4 py-1.5 rounded-sm',
    md: 'px-6 py-2',
    lg: 'px-8 py-3 rounded-md',
  };

  const Component = asChild ? 'span' : 'button';

  return (
    <Component
      type={type}
      className={`${baseClasses} ${variantClasses[variant]} ${sizeClasses[size]} ${className}`}
      disabled={disabled}
      onClick={onClick}
    >
      {children}
    </Component>
  );
};

export default Button;
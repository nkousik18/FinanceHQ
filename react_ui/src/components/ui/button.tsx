import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-500 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default:
          "bg-amber-500 text-[#0f0d0b] font-semibold hover:bg-amber-400 shadow-lg shadow-amber-500/20",
        outline:
          "border border-[rgba(245,240,232,0.15)] bg-[rgba(245,240,232,0.04)] text-[#f5f0e8] hover:bg-[rgba(245,240,232,0.08)] hover:border-[rgba(245,240,232,0.75)]",
        ghost:
          "text-[#f5f0e8]/50 hover:text-[#f5f0e8] hover:bg-[rgba(245,240,232,0.05)]",
        secondary:
          "bg-[rgba(245,240,232,0.08)] text-[#f5f0e8] hover:bg-[rgba(245,240,232,0.12)]",
      },
      size: {
        default: "h-10 px-5 py-2",
        sm: "h-8 px-3 text-xs",
        lg: "h-12 px-8 text-base",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";

export { Button, buttonVariants };

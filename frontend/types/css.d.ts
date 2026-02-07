/**
 * Type declarations for CSS module imports.
 * Allows importing .css files in TypeScript without errors.
 */
declare module "*.css" {
  const content: Record<string, string>;
  export default content;
}

declare module "@xterm/xterm/css/xterm.css";

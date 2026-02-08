import type { Metadata } from "next";
import { JetBrains_Mono, Geist } from "next/font/google";
import "./globals.css";

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains-mono",
  display: "swap",
});

const geistSans = Geist({
  subsets: ["latin"],
  variable: "--font-geist-sans",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Agentica",
  description:
    "Compare AI-assisted coding paradigms: Single ReAct Agent, Task Decomposition Swarm, and Parallel Hypothesis Testing",
};

interface RootLayoutProps {
  children: React.ReactNode;
}

export default function RootLayout({ children }: RootLayoutProps): React.ReactElement {
  return (
    <html
      lang="en"
      className={`${jetbrainsMono.variable} ${geistSans.variable} dark`}
      suppressHydrationWarning
    >
      <body className="min-h-screen bg-background bg-noise font-sans text-foreground antialiased selection:bg-accent/30 selection:text-white">
        {/* Global providers can be added here */}
        <div className="flex min-h-screen flex-col">{children}</div>
      </body>
    </html>
  );
}

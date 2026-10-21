import type { ReactNode } from 'react';
import './style.css';
export const metadata = { title: 'bIA — Laboratório de materiais', description: 'Conhecimento científico local e verificável' };
export default function RootLayout({ children }: { children: ReactNode }) {
  return <html lang="pt-BR"><body>{children}</body></html>;
}

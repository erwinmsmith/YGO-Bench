import { useEffect, useState } from 'react';
import { api } from './api';
import { Dashboard } from './components/Dashboard';
import { DeckLibrary } from './components/DeckLibrary';
import { ReplayLab } from './components/ReplayLab';
import { RunArchive } from './components/RunArchive';
import type { DeckInfo, ReplayInfo, RunInfo, StatusInfo } from './types';

type View = 'dashboard' | 'decks' | 'runs' | 'replays';

const NAV: Array<{ id: View; label: string; code: string }> = [
  { id: 'dashboard', label: '总览', code: '00' },
  { id: 'decks', label: '牌组档案', code: '01' },
  { id: 'runs', label: '实验运行', code: '02' },
  { id: 'replays', label: '牌局回放', code: '03' },
];

function initialView(): View {
  const requested = new URLSearchParams(window.location.search).get('view');
  return NAV.some((item) => item.id === requested) ? requested as View : 'dashboard';
}

export default function App() {
  const [view, setView] = useState<View>(initialView);
  const [status, setStatus] = useState<StatusInfo | null>(null);
  const [decks, setDecks] = useState<DeckInfo[]>([]);
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [replays, setReplays] = useState<ReplayInfo[]>([]);
  const [error, setError] = useState('');

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      api.status(controller.signal), api.decks(controller.signal),
      api.runs(controller.signal), api.replays(controller.signal),
    ]).then(([nextStatus, nextDecks, nextRuns, nextReplays]) => {
      setStatus(nextStatus); setDecks(nextDecks); setRuns(nextRuns); setReplays(nextReplays);
    }).catch((reason: Error) => {
      if (reason.name !== 'AbortError') setError('后端暂时不可用。请启动 `uv run ygo-bench-api`。');
    });
    return () => controller.abort();
  }, []);

  const navigate = (next: View) => {
    setView(next);
    window.history.replaceState(null, '', next === 'dashboard' ? '/' : `/?view=${next}`);
  };

  return (
    <div className="app-shell">
      <div className="grain" />
      <header className="topbar">
        <button className="brand" onClick={() => navigate('dashboard')}>
          <span className="brand-sigil">Y</span>
          <div><strong>YGO—BENCH</strong><small>DUEL INTELLIGENCE ARCHIVE</small></div>
        </button>
        <nav>
          {NAV.map((item) => (
            <button key={item.id} className={view === item.id ? 'active' : ''} onClick={() => navigate(item.id)}>
              <span>{item.code}</span>{item.label}
            </button>
          ))}
        </nav>
        <div className={`system-pill ${status?.status === 'ready' ? 'ready' : ''}`}>
          <i />{status?.status === 'ready' ? 'SYSTEM NOMINAL' : 'SETUP REQUIRED'}
        </div>
      </header>

      {error ? <div className="api-warning">{error}</div> : null}
      {view === 'dashboard' ? <Dashboard status={status} decks={decks} runs={runs} replays={replays} onNavigate={navigate} /> : null}
      {view === 'decks' ? <DeckLibrary decks={decks} /> : null}
      {view === 'runs' ? <RunArchive runs={runs} /> : null}
      {view === 'replays' ? <ReplayLab replays={replays} /> : null}

      <footer><span>YGO-BENCH / RESEARCH BUILD 0.2</span><span>RULES BY OCGCORE · TRACES ARE APPEND-ONLY</span></footer>
    </div>
  );
}

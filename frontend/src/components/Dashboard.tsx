import type { DeckInfo, ReplayInfo, RunInfo, StatusInfo } from '../types';

interface DashboardProps {
  status: StatusInfo | null;
  decks: DeckInfo[];
  runs: RunInfo[];
  replays: ReplayInfo[];
  onNavigate: (view: 'decks' | 'runs' | 'replays') => void;
}

function metric(value: number | undefined, style: 'percent' | 'integer' = 'integer') {
  if (value === undefined) return '—';
  return style === 'percent' ? `${(value * 100).toFixed(1)}%` : value.toLocaleString();
}

function runScore(run: RunInfo) {
  if (run.metrics.benchmark_type === 'full_duel_arena') {
    return { value: run.metrics.leaderboard?.[0]?.win_rate, label: 'TOP WIN' };
  }
  if (run.metrics.benchmark_type === 'full_duel') {
    return { value: run.metrics.engine_completion_rate, label: 'COMPLETE' };
  }
  return { value: run.metrics.solve_rate, label: 'SOLVE' };
}

export function Dashboard({ status, decks, runs, replays, onNavigate }: DashboardProps) {
  const latest = runs[0];
  const score = latest ? runScore(latest) : null;
  return (
    <main className="view dashboard-view">
      <section className="hero-grid">
        <div className="hero-copy reveal">
          <p className="eyebrow">DUEL INTELLIGENCE ARCHIVE · 01</p>
          <h1>让每一次决策<br /><em>经得起规则重放。</em></h1>
          <p className="hero-deck">
            基于真实 ocgcore 的长时程 agent benchmark。观察、工具调用、引擎响应与结果，
            全部被固定在可验证的对局档案里。
          </p>
          <div className="hero-actions">
            <button className="primary-button" onClick={() => onNavigate('runs')}>查看实验运行</button>
            <button className="text-button" onClick={() => onNavigate('decks')}>浏览预设牌组 ↗</button>
          </div>
        </div>
        <div className="engine-orbit reveal delay-1" aria-label="engine status visualization">
          <div className="orbit orbit-one" />
          <div className="orbit orbit-two" />
          <div className="engine-seal">
            <span>{status?.engine_ready ? 'CORE' : 'SETUP'}</span>
            <strong>{status?.engine_ready ? 'READY' : 'REQ.'}</strong>
            <small>OCG / MR5</small>
          </div>
          <div className="orbit-label north">DETERMINISTIC</div>
          <div className="orbit-label east">LEGAL ACTIONS</div>
          <div className="orbit-label south">REPLAY PROOF</div>
        </div>
      </section>

      <section className="metric-ribbon reveal delay-2">
        <article><span>ENGINE</span><strong>{status?.status === 'ready' ? 'ONLINE' : 'SETUP'}</strong></article>
        <article><span>CURATED DECKS</span><strong>{decks.length.toString().padStart(2, '0')}</strong></article>
        <article><span>ARCHIVED RUNS</span><strong>{runs.length.toString().padStart(2, '0')}</strong></article>
        <article><span>REPLAY TRACES</span><strong>{replays.length.toString().padStart(2, '0')}</strong></article>
      </section>

      <section className="dashboard-columns">
        <div className="latest-run panel reveal delay-3">
          <div className="section-heading">
            <div><span>RECENT EXPERIMENT</span><h2>最新运行切片</h2></div>
            <button onClick={() => onNavigate('runs')}>全部运行</button>
          </div>
          {latest ? (
            <div className="run-specimen">
              <div className="specimen-id">{latest.id}</div>
              <div className="score-disc" style={{ '--score': score?.value ?? 0 } as React.CSSProperties}>
                <div><strong>{metric(score?.value, 'percent')}</strong><span>{score?.label}</span></div>
              </div>
              <dl>
                <div><dt>Episodes</dt><dd>{metric(latest.metrics.total)}</dd></div>
                <div><dt>{latest.metrics.benchmark_type === 'full_duel_arena' ? 'Avg. decisions' : 'Avg. tools'}</dt><dd>{(latest.metrics.avg_decisions ?? latest.metrics.avg_tool_calls)?.toFixed(1) ?? '—'}</dd></div>
                <div><dt>Engine finish</dt><dd>{metric(latest.metrics.engine_completion_rate, 'percent')}</dd></div>
                <div><dt>Replays</dt><dd>{latest.replay_count}</dd></div>
              </dl>
            </div>
          ) : (
            <div className="empty-state"><span>∅</span><p>还没有 benchmark 运行。完成 setup 后从 CLI 发起第一批评测。</p></div>
          )}
        </div>

        <aside className="protocol-note panel reveal delay-4">
          <p className="eyebrow">PROTOCOL NOTE</p>
          <h2>不是让模型“描述”操作，<br />而是让它提交合法响应。</h2>
          <p>每一步只接受当前 ocgcore 消息允许的结构化动作。非法索引、错误时点和隐藏信息泄漏都会留在 trace 中。</p>
          <code>select_chain(index: 2)</code>
          <code>select_card(indices: [0, 3])</code>
          <code>select_idlecmd(command: "to_battle_phase")</code>
        </aside>
      </section>
    </main>
  );
}

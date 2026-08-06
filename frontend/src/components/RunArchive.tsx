import type { RunInfo } from '../types';

function pct(value?: number) { return value === undefined ? '—' : `${(value * 100).toFixed(1)}%`; }

function primary(run: RunInfo) {
  if (run.metrics.benchmark_type === 'full_duel_arena') {
    return pct(run.metrics.leaderboard?.[0]?.win_rate);
  }
  if (run.metrics.benchmark_type === 'full_duel') return pct(run.metrics.engine_completion_rate);
  return pct(run.metrics.solve_rate);
}

export function RunArchive({ runs }: { runs: RunInfo[] }) {
  return (
    <main className="view archive-view">
      <header className="page-header">
        <p className="eyebrow">EXPERIMENT LEDGER</p><h1>Benchmark 运行</h1>
        <p>每个运行目录绑定模型、模式、输入数据与 replay trace。不同规则快照不应混入同一排行榜。</p>
      </header>
      <div className="run-table" role="table">
        <div className="run-row run-head" role="row">
          <span>RUN IDENTIFIER</span><span>SCORE</span><span>ENGINE</span><span>DECISIONS</span><span>EPISODES</span><span>UPDATED</span>
        </div>
        {runs.map((run) => (
          <article className="run-row" role="row" key={run.id}>
            <div><i className="status-dot" /><strong>{run.id}</strong><small>{run.replay_count} trace files</small></div>
            <b>{primary(run)}</b>
            <span>{pct(run.metrics.engine_completion_rate)}</span>
            <span>{(run.metrics.avg_decisions ?? run.metrics.avg_tool_calls)?.toFixed(1) ?? '—'}</span>
            <span>{run.metrics.total ?? '—'}</span>
            <time>{new Date(run.mtime * 1000).toLocaleString()}</time>
          </article>
        ))}
        {runs.length === 0 ? <div className="empty-state"><span>NO RUNS</span><p>使用 `uv run ygo-bench eval ...` 生成第一份实验记录。</p></div> : null}
      </div>
    </main>
  );
}

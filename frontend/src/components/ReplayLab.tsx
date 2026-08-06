import { useEffect, useMemo, useState } from 'react';
import { api } from '../api';
import type { CardInfo, DuelCard, DuelReplayDetail, DuelReplayFrame, ReplayInfo } from '../types';
import { DuelBoard } from './DuelBoard';
import { CardArt } from './CardArt';

const SPEEDS = [0.25, 0.5, 1, 2, 4];

function actionSummary(frame: DuelReplayFrame) {
  const call = frame.action?.tool_calls?.[0];
  if (!call) return '等待规则引擎';
  const args = call.arguments;
  const command = typeof args.command === 'string' ? ` · ${args.command}` : '';
  const index = typeof args.index === 'number' ? ` #${args.index}` : '';
  return `${call.name}${command}${index}`;
}

function winnerLabel(outcome: DuelReplayDetail['outcome']) {
  const winner = outcome?.winner;
  if (winner === 0 || winner === 1) return `玩家 ${Number(winner) + 1}`;
  return 'NO RESULT';
}

function CardInspect({ card, onClose }: { card: DuelCard; onClose: () => void }) {
  const [detail, setDetail] = useState<CardInfo | null>(null);
  useEffect(() => {
    if (!card.code) return;
    const controller = new AbortController();
    api.card(card.code, controller.signal).then(setDetail).catch((error: Error) => {
      if (error.name !== 'AbortError') setDetail(null);
    });
    return () => controller.abort();
  }, [card.code]);
  return (
    <div className="duel-card-modal" role="dialog" aria-modal="true" onClick={onClose}>
      <div onClick={(event) => event.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>×</button>
        <CardArt src={card.code ? `/api/cards/${card.code}/image` : '/card-back.png'} alt={card.name ?? 'Hidden card'} />
        <article>
          <span>CARD DOSSIER · {card.code ?? 'HIDDEN'}</span>
          <h2>{detail?.name ?? card.name ?? '盖放卡'}</h2>
          <div className="card-facts">
            <b>ATK {detail?.attack ?? card.attack ?? '—'}</b>
            <b>DEF {detail?.defense ?? card.defense ?? '—'}</b>
            <b>{(detail?.type ?? card.type_flags ?? []).join(' / ') || 'UNKNOWN'}</b>
          </div>
          <p>{detail?.description ?? '该卡在当前视角中处于隐藏状态。'}</p>
        </article>
      </div>
    </div>
  );
}

export function ReplayLab({ replays }: { replays: ReplayInfo[] }) {
  const [selected, setSelected] = useState<ReplayInfo | null>(replays[0] ?? null);
  const [detail, setDetail] = useState<DuelReplayDetail | null>(null);
  const [cursor, setCursor] = useState(0);
  const [loading, setLoading] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [selectedCard, setSelectedCard] = useState<DuelCard | null>(null);

  useEffect(() => {
    if (!selected && replays.length > 0) setSelected(replays[0]);
  }, [replays, selected]);

  useEffect(() => {
    if (!selected) return;
    const controller = new AbortController();
    setLoading(true);
    setPlaying(false);
    api.replayFrames(selected.run_id, selected.filename, controller.signal)
      .then((data) => { setDetail(data); setCursor(0); })
      .catch((error: Error) => {
        if (error.name !== 'AbortError') setDetail(null);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [selected]);

  const frames = detail?.frames ?? [];
  useEffect(() => {
    if (!playing || frames.length < 2) return;
    const timer = window.setInterval(() => {
      setCursor((value) => {
        if (value >= frames.length - 1) {
          window.clearInterval(timer);
          setPlaying(false);
          return value;
        }
        return value + 1;
      });
    }, 1000 / speed);
    return () => window.clearInterval(timer);
  }, [frames.length, playing, speed]);

  const current = frames[cursor];
  const agents = useMemo(() => {
    const configured = detail?.config?.agents;
    return Array.isArray(configured) ? configured.map(String) : ['unknown', 'unknown'];
  }, [detail]);
  const decks = useMemo(() => [detail?.config?.deck1, detail?.config?.deck2].map((value) => {
    const filename = String(value ?? 'unknown').split('/').at(-1) ?? 'unknown';
    return filename.replace(/\.ydk$/i, '');
  }), [detail]);
  const passiveNotice = agents.some((agent) => agent === 'passive');

  const go = (value: number) => {
    setPlaying(false);
    setCursor(Math.max(0, Math.min(value, frames.length - 1)));
  };

  return (
    <main className="view replay-view visual-replay-view">
      <header className="page-header compact">
        <div><p className="eyebrow">VISUAL DUEL RECONSTRUCTION</p><h1>全牌局可视化回放</h1></div>
        <p>逐决策还原双方 LP、手牌视角、怪兽区、魔陷区、墓地、除外区、阶段、连锁与 Agent 操作。</p>
      </header>
      <div className="visual-replay-grid">
        <aside className="visual-trace-index panel">
          <div className="section-heading"><div><span>DUEL ARCHIVE</span><h2>{replays.length} 局</h2></div></div>
          <div className="trace-list">
            {replays.map((replay) => (
              <button
                key={`${replay.run_id}/${replay.filename}`}
                className={selected?.filename === replay.filename && selected.run_id === replay.run_id ? 'active' : ''}
                onClick={() => setSelected(replay)}
              >
                <span>{replay.puzzle_id.replace('yugioh_puzzle_', '#')}</span>
                <small>{replay.run_id}</small><b>{Math.ceil(replay.size / 1024)} KB</b>
              </button>
            ))}
          </div>
          {detail ? (
            <div className="duel-result-card">
              <span>FINAL OUTCOME</span><strong>{winnerLabel(detail.outcome)}</strong>
              <small>{String(detail.outcome?.termination ?? 'in progress')}</small>
            </div>
          ) : null}
        </aside>

        <section className="visual-replay-stage panel">
          {loading ? <div className="scan-loader">RECONSTRUCTING DUEL FIELD…</div> : current ? (
            <>
              <div className="duel-replay-meta">
                <div><span>FRAME</span><strong>{cursor + 1} / {frames.length}</strong></div>
                <div><span>TURN</span><strong>{current.state.turn ?? '—'}</strong></div>
                <div><span>PHASE</span><strong>{current.state.phase?.toUpperCase() ?? 'SETUP'}</strong></div>
                <div><span>当前响应</span><strong>玩家 {current.player + 1}</strong></div>
                <div><span>AGENTS</span><strong>{agents[0]} / {agents[1]}</strong></div>
              </div>
              <div className="duel-player-map">
                <div className="seat-0"><i /><span><b>玩家 1 · 下方</b><small>{agents[0]} · {decks[0]}</small></span></div>
                <div className="seat-1"><i /><span><b>玩家 2 · 上方</b><small>{agents[1]} · {decks[1]}</small></span></div>
                {passiveNotice ? <em>PASSIVE 是规则验证基线：会主动跳过多数展开与连锁</em> : null}
              </div>
              <div className="duel-theater">
                <DuelBoard state={current.state} action={current.action} agents={agents} decks={decks} onCardClick={setSelectedCard} />
                <aside className="duel-action-log">
                  <div className="section-heading"><div><span>ACTION LEDGER</span><h2>{frames.length} decisions</h2></div></div>
                  <div>
                    {frames.map((frame) => (
                      <button key={frame.frame_index} className={`seat-${frame.player} ${frame.frame_index === cursor ? 'active' : ''} ${frame.invalid_action ? 'invalid' : ''}`} onClick={() => go(frame.frame_index)}>
                        <i className={`player-${frame.player}`} />
                        <span><b>{String(frame.frame_index + 1).padStart(3, '0')} · 玩家 {frame.player + 1}</b><small>{actionSummary(frame)}</small></span>
                        {frame.fallback ? <em>FB</em> : null}
                      </button>
                    ))}
                  </div>
                </aside>
              </div>
              <div className="duel-playback">
                <button onClick={() => go(0)} disabled={cursor === 0}>|◀</button>
                <button onClick={() => go(cursor - 1)} disabled={cursor === 0}>◀</button>
                <button className="play-button" onClick={() => setPlaying((value) => !value)}>{playing ? 'Ⅱ' : '▶'}</button>
                <button onClick={() => go(cursor + 1)} disabled={cursor >= frames.length - 1}>▶</button>
                <button onClick={() => go(frames.length - 1)} disabled={cursor >= frames.length - 1}>▶|</button>
                <input type="range" min="0" max={Math.max(0, frames.length - 1)} value={cursor} onChange={(event) => go(Number(event.target.value))} />
                <div className="speed-picker">
                  {SPEEDS.map((value) => <button key={value} className={speed === value ? 'active' : ''} onClick={() => setSpeed(value)}>{value}×</button>)}
                </div>
              </div>
            </>
          ) : <div className="empty-state"><span>NO VISUAL REPLAY</span><p>选择一局已有对局，或运行 `ygo-bench duel` 生成完整状态快照。</p></div>}
        </section>
      </div>
      {selectedCard ? <CardInspect card={selectedCard} onClose={() => setSelectedCard(null)} /> : null}
    </main>
  );
}

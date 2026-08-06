import { memo } from 'react';
import { CardArt } from './CardArt';
import type { DuelCard, DuelReplayAction, DuelSide, DuelState } from '../types';

interface BoardProps {
  state: DuelState;
  action: DuelReplayAction | null;
  agents: string[];
  decks: string[];
  onCardClick: (card: DuelCard) => void;
}

function cardLabel(card: DuelCard) {
  if (card.empty) return '空区域';
  if (card.face_down || !card.code) return '盖放卡';
  return card.name ?? `Card #${card.code}`;
}

function isDefensePosition(card: DuelCard, zone: string) {
  if (!zone.startsWith('M')) return false;
  const position = card.position?.toLowerCase() ?? '';
  if (position.includes('defense')) return true;
  const encoded = position.match(/0x([0-9a-f]+)/i);
  return encoded ? (Number.parseInt(encoded[1], 16) & 0x0c) !== 0 : false;
}

function CardSlot({ card, zone, onClick }: { card?: DuelCard | null; zone: string; onClick: (card: DuelCard) => void }) {
  const value = card ?? { empty: true };
  const hidden = value.face_down || !value.code;
  const defense = isDefensePosition(value, zone);
  return (
    <button
      className={`duel-card-slot ${value.empty ? 'empty' : ''} ${defense ? 'defense' : ''}`}
      onClick={() => !value.empty && onClick(value)}
      disabled={value.empty}
      title={`${zone} · ${cardLabel(value)}`}
    >
      {value.empty ? <span>{zone}</span> : (
        <>
          <CardArt
            className="duel-zone-card-art"
            src={!hidden && value.code ? `/api/cards/${value.code}/image` : '/card-back.png'}
            alt={cardLabel(value)}
          />
          {!hidden ? <b>{value.name}</b> : null}
          {!hidden && (value.attack || value.defense) ? (
            <small>{value.attack ?? 0}<i>/</i>{value.defense ?? 0}</small>
          ) : null}
        </>
      )}
    </button>
  );
}

function Pile({ label, count, card, onClick }: { label: string; count: number; card?: DuelCard; onClick: (card: DuelCard) => void }) {
  return (
    <button className="duel-pile" onClick={() => card?.code && onClick(card)} disabled={!card?.code}>
      {card?.code ? <CardArt src={`/api/cards/${card.code}/image`} alt={card.name ?? label} /> : <img src="/card-back.png" alt={label} />}
      <span>{label}</span><b>{count}</b>
    </button>
  );
}

function Hand({ cards, count, onClick }: { cards: DuelCard[]; count: number; onClick: (card: DuelCard) => void }) {
  const visible: DuelCard[] = cards.length > 0
    ? cards
    : Array.from({ length: count }, () => ({ face_down: true } as DuelCard));
  return (
    <div className="duel-hand">
      {visible.slice(0, 12).map((card, index) => (
        <button key={`${card.code ?? 'hidden'}-${index}`} onClick={() => card.code && onClick(card)} disabled={!card.code}>
          <CardArt src={card.code && !card.face_down ? `/api/cards/${card.code}/image` : '/card-back.png'} alt={cardLabel(card)} />
        </button>
      ))}
      <span>HAND · {count}</span>
    </div>
  );
}

function PlayerBoard({ side, player, agent, deck, active, onCardClick }: { side: DuelSide; player: number; agent: string; deck: string; active: boolean; onCardClick: (card: DuelCard) => void }) {
  const monsters = side.monster_zone ?? [];
  const spells = side.spell_trap_zone ?? [];
  return (
    <section className={`duel-player seat-${player} ${active ? 'active' : ''}`}>
      <header>
        <div className="duel-player-identity">
          <span><i />玩家 {player + 1} · {player === 0 ? '下方' : '上方'}</span>
          <b>{agent}</b><small>{deck}</small>
        </div>
        <strong>{side.lp?.toLocaleString() ?? 0}<small> LP</small></strong>
        <div className="duel-counts"><span>DECK {side.deck_count}</span><span>HAND {side.hand_count}</span><span>GY {side.grave_count}</span><span>BAN {side.banished_count}</span></div>
      </header>
      <Hand cards={side.hand ?? []} count={side.hand_count ?? 0} onClick={onCardClick} />
      <div className="duel-field-row spell-row">
        <Pile label="FIELD" count={side.field_zone ? 1 : 0} card={side.field_zone ?? undefined} onClick={onCardClick} />
        <div className="duel-five-zones">
          {Array.from({ length: 5 }, (_, index) => <CardSlot key={index} card={spells[index]} zone={`S${index + 1}`} onClick={onCardClick} />)}
        </div>
        <Pile label="BAN" count={side.banished_count ?? 0} card={side.banished?.at(-1)} onClick={onCardClick} />
      </div>
      <div className="duel-field-row monster-row">
        <Pile label="EXTRA" count={side.extra_deck_count ?? 0} onClick={onCardClick} />
        <div className="duel-five-zones">
          {Array.from({ length: 5 }, (_, index) => <CardSlot key={index} card={monsters[index]} zone={`M${index + 1}`} onClick={onCardClick} />)}
        </div>
        <Pile label="GY" count={side.grave_count ?? 0} card={side.graveyard?.at(-1)} onClick={onCardClick} />
      </div>
    </section>
  );
}

export const DuelBoard = memo(function DuelBoard({ state, action, agents, decks, onCardClick }: BoardProps) {
  const perspective = Number(state.perspective_player ?? 0);
  const player0 = perspective === 0 ? state.you : state.opponent;
  const player1 = perspective === 1 ? state.you : state.opponent;
  const turnPlayer = state.turn_player === 'you' ? perspective : 1 - perspective;
  const call = action?.tool_calls?.[0];
  return (
    <div className="duel-board-shell" data-testid="duel-replay-board">
      <PlayerBoard side={player1} player={1} agent={agents[1] ?? 'unknown'} deck={decks[1] ?? 'unknown'} active={turnPlayer === 1} onCardClick={onCardClick} />
      <div className="duel-midline">
        <div className="extra-monster-zone"><span>EMZ</span></div>
        <div className="duel-turn-seal"><span>TURN {state.turn ?? '—'}</span><strong>{state.phase?.toUpperCase() ?? 'SETUP'}</strong><small>CHAIN {state.chain?.length ?? 0}</small></div>
        <div className="extra-monster-zone"><span>EMZ</span></div>
        {call ? (
          <div className={`duel-action-callout player-${action?.player ?? 0}`}>
            <b>玩家 {(action?.player ?? 0) + 1} · {action?.agent}</b>
            <span>{call.name}</span>
            <code>{JSON.stringify(call.arguments)}</code>
          </div>
        ) : null}
      </div>
      <PlayerBoard side={player0} player={0} agent={agents[0] ?? 'unknown'} deck={decks[0] ?? 'unknown'} active={turnPlayer === 0} onCardClick={onCardClick} />
    </div>
  );
});

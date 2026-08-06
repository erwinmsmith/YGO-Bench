import { useMemo, useState } from 'react';
import type { CardInfo, DeckInfo } from '../types';
import { CardArt } from './CardArt';

interface DeckLibraryProps { decks: DeckInfo[] }

function DeckCards({ cards }: { cards: CardInfo[] }) {
  return (
    <div className="card-list">
      {cards.map((card) => (
        <article className="mini-card" key={card.id}>
          <CardArt src={card.image_url} alt={card.name} />
          <div><strong>{card.name}</strong><span>{card.type?.join(' · ') || `ID ${card.id}`}</span></div>
          <b>×{card.count}</b>
        </article>
      ))}
    </div>
  );
}

export function DeckLibrary({ decks }: DeckLibraryProps) {
  const [selectedId, setSelectedId] = useState<string | null>(decks[0]?.id ?? null);
  const [section, setSection] = useState<'main' | 'extra' | 'side'>('main');
  const selected = useMemo(() => decks.find((deck) => deck.id === selectedId) ?? decks[0], [decks, selectedId]);

  return (
    <main className="view library-view">
      <header className="page-header">
        <p className="eyebrow">CURATED LOADOUTS · YGO-AGENT VERIFIED</p>
        <h1>预设牌组档案</h1>
        <p>从已验证的环境牌组中挑选代表性策略。卡组文件保留原始 `.ydk` 格式，可直接交给后续 full-duel runner。</p>
      </header>
      <div className="deck-workbench">
        <div className="deck-index">
          {decks.map((deck, index) => (
            <button
              key={deck.id}
              className={deck.id === selected?.id ? 'active' : ''}
              onClick={() => setSelectedId(deck.id)}
              style={{ '--deck-accent': deck.accent } as React.CSSProperties}
            >
              <span>{String(index + 1).padStart(2, '0')}</span>
              <div><strong>{deck.name}</strong><small>{deck.archetype}</small></div>
              <b>{deck.difficulty}</b>
            </button>
          ))}
        </div>
        {selected ? (
          <section className="deck-sheet" style={{ '--deck-accent': selected.accent } as React.CSSProperties}>
            <div className="deck-portrait">
              <CardArt src={selected.cover_image} alt={selected.name} />
              <span className="portrait-code">DECK / {selected.id.toUpperCase()}</span>
            </div>
            <div className="deck-details">
              <div className="deck-title"><div><span>{selected.archetype}</span><h2>{selected.name}</h2></div><b>{selected.difficulty}</b></div>
              <p>{selected.description}</p>
              <div className="deck-counts">
                <div><strong>{selected.main_count}</strong><span>MAIN</span></div>
                <div><strong>{selected.extra_count}</strong><span>EXTRA</span></div>
                <div><strong>{selected.side_count}</strong><span>SIDE</span></div>
              </div>
              <div className="section-tabs">
                {(['main', 'extra', 'side'] as const).map((name) => (
                  <button key={name} className={section === name ? 'active' : ''} onClick={() => setSection(name)}>{name}</button>
                ))}
              </div>
              <DeckCards cards={selected.sections[section]} />
            </div>
          </section>
        ) : <div className="empty-state">运行 scripts/sync_decks.py 载入预设牌组。</div>}
      </div>
    </main>
  );
}


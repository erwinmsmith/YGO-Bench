export interface CardInfo {
  id: number;
  name: string;
  description?: string;
  type?: string[];
  attack?: number | null;
  defense?: number | null;
  level?: number;
  image_url?: string;
  count: number;
}

export interface DeckInfo {
  id: string;
  name: string;
  archetype: string;
  difficulty: string;
  accent: string;
  description: string;
  source: string;
  main_count: number;
  extra_count: number;
  side_count: number;
  cover_image?: string | null;
  sections: Record<'main' | 'extra' | 'side', CardInfo[]>;
}

export interface StatusInfo {
  status: 'ready' | 'setup_required';
  engine_ready: boolean;
  dataset_ready: boolean;
  deck_count: number;
  run_count: number;
  runtime_errors: string[];
}

export interface Metrics {
  benchmark_type?: 'puzzle' | 'full_duel' | 'full_duel_arena' | string;
  total?: number;
  solved?: number;
  solve_rate?: number;
  engine_completion_rate?: number;
  avg_tool_calls?: number;
  avg_model_calls?: number;
  avg_elapsed_seconds?: number;
  input_tokens?: number;
  output_tokens?: number;
  illegal_action_rate?: number;
  avg_decision_seconds?: number;
  avg_turns?: number;
  avg_decisions?: number;
  winner?: number | null;
  winner_agent?: string | null;
  turn_count?: number;
  leaderboard?: Array<{
    agent: string;
    games: number;
    wins: number;
    win_rate: number;
    illegal_action_rate: number;
    avg_decision_seconds: number;
    elo: number;
    glicko2: number;
    rating_deviation: number;
  }>;
}

export interface RunInfo {
  id: string;
  mtime: number;
  replay_count: number;
  metrics: Metrics;
  counts: Record<string, number>;
}

export interface ReplayInfo {
  run_id: string;
  filename: string;
  puzzle_id: string;
  size: number;
  mtime: number;
}

export interface ReplayEvent {
  index: number;
  type: string;
  [key: string]: unknown;
}

export interface ReplayDetail {
  run_id: string;
  filename: string;
  events: ReplayEvent[];
  outcome: ReplayEvent | null;
}

export interface DuelCard {
  zone_index?: number;
  empty?: boolean;
  face_down?: boolean;
  code?: number;
  name?: string;
  position?: string;
  attack?: number;
  defense?: number;
  level?: number;
  rank?: number;
  link?: number;
  attribute?: string | string[];
  race?: string | string[];
  type_flags?: string[];
  location?: string;
  sequence?: number;
}

export interface DuelSide {
  lp: number;
  deck_count: number;
  hand_count: number;
  grave_count: number;
  banished_count: number;
  extra_deck_count: number;
  monster_zone: DuelCard[];
  spell_trap_zone: DuelCard[];
  field_zone: DuelCard | null;
  pendulum_zone: DuelCard[];
  hand: DuelCard[];
  graveyard: DuelCard[];
  banished: DuelCard[];
}

export interface DuelState {
  perspective_player: number;
  phase?: string;
  turn_player?: string;
  turn?: number;
  game_over?: boolean;
  chain?: unknown[];
  decision?: Record<string, unknown>;
  events_since_last_decision?: Array<Record<string, unknown>>;
  recent_actions?: Array<Record<string, unknown>>;
  you: DuelSide;
  opponent: DuelSide;
}

export interface DuelReplayAction {
  player: number;
  agent: string;
  text?: string;
  tool_calls: Array<{ name: string; arguments: Record<string, unknown> }>;
  elapsed_seconds?: number;
  agent_error?: string | null;
}

export interface DuelReplayFrame {
  frame_index: number;
  player: number;
  state: DuelState;
  action: DuelReplayAction | null;
  engine_events: Array<Record<string, unknown>>;
  invalid_action?: Record<string, unknown> | null;
  fallback?: boolean;
}

export interface DuelReplayDetail {
  run_id: string;
  filename: string;
  config: Record<string, unknown>;
  frames: DuelReplayFrame[];
  outcome: ReplayEvent | null;
}

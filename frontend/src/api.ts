import type { CardInfo, DeckInfo, DuelReplayDetail, ReplayDetail, ReplayInfo, RunInfo, StatusInfo } from './types';

async function request<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(path, { signal });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json() as Promise<T>;
}

export const api = {
  status: (signal?: AbortSignal) => request<StatusInfo>('/api/status', signal),
  decks: (signal?: AbortSignal) => request<DeckInfo[]>('/api/decks', signal),
  runs: (signal?: AbortSignal) => request<RunInfo[]>('/api/runs', signal),
  replays: (signal?: AbortSignal) => request<ReplayInfo[]>('/api/replays', signal),
  replay: (runId: string, filename: string, signal?: AbortSignal) =>
    request<ReplayDetail>(
      `/api/replays/${encodeURIComponent(runId)}/${encodeURIComponent(filename)}`,
      signal,
    ),
  replayFrames: (runId: string, filename: string, signal?: AbortSignal) =>
    request<DuelReplayDetail>(
      `/api/replays/${encodeURIComponent(runId)}/${encodeURIComponent(filename)}/frames`,
      signal,
    ),
  card: (cardId: number, signal?: AbortSignal) =>
    request<CardInfo>(`/api/cards/${cardId}`, signal),
};

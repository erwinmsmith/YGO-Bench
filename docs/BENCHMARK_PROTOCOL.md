# YGO-Bench 评测协议

## 1. 目标与边界

YGO-Bench 衡量 Agent 在真实 ocgcore 裁决下完成游戏王决策的能力。规则、效果、
时点与胜负由引擎决定；Agent 只能读取当前玩家可见的 observation，并提交当前
responder 接受的一次结构化动作。文本解释不计作动作。

同一排行榜必须固定：仓库提交、ocgcore、CardScripts、BabelCDB、LFList、牌组文件、
起始规则、观察权限、prompt 版本、模型与推理配置。任一项变化都应建立新榜。

## 2. Prompt 与动作协议

模板位于 `ygobench/agents/prompts/full_duel_system.md` 和
`full_duel_observation.md`，核心约束如下：

1. `you` 始终是当前被要求响应的玩家，`opponent` 是对手；不得根据座位猜测视角。
2. 只使用 observation 中可见信息；未知手牌、盖卡和牌库顺序不得臆测。
3. `decision.responder` 是唯一合法的响应工具；每个模型回合最终必须恰好调用一次。
4. 卡片、选项和区域索引只对当前 decision 有效，状态推进后立即失效。
5. 需要效果原文时可调用 `inspect_card`，最多四次；不可见卡无法查询。
6. 工具失败、无调用、多调用或参数不合法都会进入 trace，并按协议回退或判负。

每个决策使用新鲜上下文：当前紧凑状态、最近动作、引擎事件和工具 schema。
不把旧索引保存在跨回合聊天历史中，避免 stale-action 错误。

## 3. Benchmark 套件

### A. Puzzle / Tactical

- **Interactive**：Agent 逐次读取引擎 observation 并调用工具，主分数为 solve rate。
- **N-attempts**：先生成完整计划，再由引擎尝试执行；报告 best-of-N 与 pass@N。
- **Forage**：限制卡片知识，要求按需查询卡片，单独统计查询成本。
- **Oracle ablation**：显式提供参考解，仅用于测执行能力，不与标准榜混合。

建议同时报告 engine completion、工具调用数、模型调用数、token、耗时及 termination
分布。测试集题目不可出现在 prompt 或 few-shot 示例中。

### B. Full Duel / Policy Match

- **Engine smoke**：passive vs passive，验证长时程生命周期和回放完整性。
- **Baseline match**：被测 Agent 对 passive、random 等冻结基线。
- **Seat swap**：同一 deck/seed 交换先后手，构成最小配对单元。
- **Deck matrix**：在固定预设牌组集合上做交叉对局，分离策略和牌组强度。
- **Seed replicate**：每个配对使用多个公开 seed；禁止只发布最佳 seed。
- **Arena**：多个 Agent 的换边 round-robin，生成 Elo 与 Glicko-2 排名。

建议正式榜至少 100 个换边配对，并同时发布 bootstrap 95% 置信区间。少量对局只可
标记为 smoke 或开发结果。

## 4. 完整对局指标

- **win_rate**：胜局 / 对局数；和局单独报告，不隐式算半胜。
- **first/second_win_rate**：按座位分层，识别先后手偏差。
- **illegal_action_rate**：非法引擎响应 / 该 Agent 决策数。
- **engine_completion_rate**：正常终局或明确规则判负的比例。
- **avg_decision_seconds**：Agent 端墙钟决策时间均值。
- **token totals**：输入与输出 token 总量，配合胜率衡量成本。
- **Elo / Glicko-2**：仅在同一冻结协议内排序；Glicko-2 同时报告 RD。
- **deck matchup matrix**：按 deck1/deck2 与座位保留原始胜负数。

非法动作默认导致该局判负并保留原始参数；不能静默改写成合法动作。模型没有调用
当前 responder 时，可使用第一个安全动作生成可视回放，但必须记录 `fallback=true`
和 invalid-output 计数，正式成绩中仍作为协议错误。

## 5. 回放与可审计性

每局 JSONL 至少包含：config、observation、model_turn、tool_result、invalid_action（若有）
和 outcome。Web 后端将 observation 与随后动作连接为视觉帧。前端只渲染这些快照，
不自行模拟规则，因此时间轴任意位置都能回到引擎当时的真实牌面。

发布结果时应一并保存 `metrics.json`、`games.json`、原始 JSONL、prompt 文件哈希、
牌组哈希和第三方依赖提交。API key、完整 `.env` 与供应商私有响应字段不得发布。

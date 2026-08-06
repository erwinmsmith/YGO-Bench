# YGO-Bench / 游戏王 LLM Benchmark

[English README](README.md) · 中文文档

YGO-Bench 是一套面向 LLM agent 的游戏王规则交互 benchmark。项目借鉴
PTCG-Bench 的 agent/environment/evaluation 分层，但规则裁决交给真实的
EDOPro `ocgcore`，不在 Python 中重新实现卡片效果。

当前 `v0.2` 同时覆盖残局解题与完整对局：

- 真实 `ocgcore` 执行与回放验证；
- 交互式逐决策 tool-use，以及 N-attempts 整体方案两种模式；
- 对局视角过滤，默认不泄露对手隐藏信息；
- 统一的模型配置、运行目录、solve rate、token、耗时和失败类型指标；
- 固定第三方提交，避免上游规则/卡片脚本更新导致结果漂移；
- 完整对局的 passive、random、LLM/ReAct agent 与换边 Arena；
- 牌桌级可视化回放，支持逐步、倍速、时间轴和卡片详情。

## 快速开始

```bash
uv sync --extra dev
uv run ygo-bench setup
uv run ygo-bench doctor

# 终端 1
uv run ygo-bench-api

# 终端 2
cd frontend
npm install
npm run dev
```

打开 `http://localhost:5173/?view=replays`，可以检查 LP、手牌、场区、墓地、阶段、
连锁、Agent 动作与卡片详情。完整协议见
[`docs/BENCHMARK_PROTOCOL.md`](docs/BENCHMARK_PROTOCOL.md)。

## 回放展示

下面的素材来自一场 **BlueEyes vs BlueEyes** 双 LLM 完整对局（seed `79`）。玩家 1
固定在下方，玩家 2 固定在上方；右侧动作时间线可以定位到任意一次
`select_*` / `sp_summon` / `set_spell` 决策。

![Blue-Eyes 双 LLM 回放](docs/assets/replay-seed79.png)

| 开局 | 中段 |
| --- | --- |
| ![开局回放](docs/assets/replay-opening.gif) | ![中段回放](docs/assets/replay-midgame.gif) |

| 终局 | 总览 |
| --- | --- |
| ![终局回放](docs/assets/replay-endgame.gif) | ![回放总览](docs/assets/replay-overview.gif) |

这场回放共 12 回合、445 次决策，玩家 2 获胜（LP `0 - 8000`），双方非法动作数和
协议 fallback 均为 `0`。原始 JSONL 会写入本地 `bench_data/runs/`，默认被 Git
忽略；仓库只提交轻量展示素材，避免上传运行日志和 API 凭据。

## 架构

```text
LLM / rule / RL agent
          |
          v
ygobench agent protocol
          |
          v
puzzle suite -------- full-duel / arena suite
          |
          v
yugi-bench harness -> libocgcore -> CardScripts + BabelCDB
          |
          v
JSONL replay, metrics.json, leaderboard-ready outcomes
```

EDOPro 仓库本身是 GUI 客户端。本项目只使用其规则核心相关生态：
`ygopro-core`、CardScripts、BabelCDB。`vendor/yugi-bench` 是固定提交的
Apache-2.0 引擎适配与谜题验证基座。

## 安装

```bash
uv sync --extra dev

# 拉取固定版本的 ocgcore / CardScripts / BabelCDB / Puzzles，编译并离线验证
uv run ygo-bench setup

# 检查本机与引擎状态
uv run ygo-bench doctor
```

`setup` 在 macOS 需要 Xcode Command Line Tools、`make` 与 `premake5`。如果系统没有
`premake5`，YGO-Bench 会把固定版本下载到项目内的 `.tools/`，不会修改系统安装。

## Web 实验台

启动 FastAPI：

```bash
uv run ygo-bench-api
```

另开一个终端启动 React/Vite：

```bash
cd frontend
npm install
npm run dev
```

实验台提供：

- 引擎、数据集与运行状态总览；
- 8 套从 `ygo-agent` 同步的预设 `.ydk` 牌组；
- 本地 BabelCDB 卡片元数据与按需缓存卡图；
- benchmark 运行列表和牌桌级 replay：LP、手牌/牌库、场区、墓地、阶段、连锁、
  Agent 工具调用和胜负结果均可逐帧检查。

回放界面使用面向用户的编号：**玩家 1 固定在下方（青绿色）**，**玩家 2 固定在上方
（琥珀色）**；JSONL 和 ocgcore 内部仍使用 seat `0/1`。界面会同时显示双方 Agent
与牌组。包含 `passive` 的对局会标记为规则验证基线，避免误认为它是主动策略。

牌组可以重新同步：

```bash
uv run python scripts/sync_decks.py
```

回放帧来自 JSONL 中真实 ocgcore observation，不会在前端推演或伪造卡片状态。

## 运行评测

先复制 `.env.example` 为 `.env`，填写对应模型的 API key。`.env` 被 Git 忽略，不能
提交到仓库。

交互式逐决策评测：

```bash
uv run ygo-bench eval \
  --provider deepseek \
  --model deepseek-v4-flash \
  --limit 5 \
  --concurrency 1
```

N-attempts 整体方案评测：

```bash
uv run ygo-bench eval \
  --provider openai \
  --model gpt-5 \
  --attempts 3 \
  --limit 20
```

只查看将要执行的命令：

```bash
uv run ygo-bench eval --provider deepseek --model deepseek-v4-flash --limit 1 --dry-run
```

结果写入 `bench_data/runs/<run-name>/`：

- 每题一个 JSONL：完整 observation、tool call、engine result；
- `_summary.json`：上游逐题结果；
- `metrics.json`：YGO-Bench 归一化汇总。

运行完整无头对局 benchmark：

```bash
uv run ygo-bench duel \
  --deck1 BlueEyes \
  --deck2 CyberDragon \
  --agent1 react \
  --agent2 passive \
  --seed 11
```

双 LLM 且不设置决策上限：

```bash
uv run ygo-bench duel \
  --deck1 BlueEyes \
  --deck2 CyberDragon \
  --agent1 react \
  --agent2 react \
  --seed 59 \
  --max-decisions 0
```

当一个 pending decision 只有唯一合法响应时，协议会直接提交该响应且不调用模型；
所有包含真实选择的决策仍由对应玩家的 LLM 完成。

若需要运行到终局并降低长篇 thinking 的延迟，可将双方改为 `react-fast`。它仍使用
同一个 LLM 和完整 observation，只关闭 provider 的额外 thinking 模式；运行配置会把
`thinking_enabled` 和模型参数写进 replay，不能与高推理榜混合。

当前完整对局 profile 的单次输出上限为 32,768 tokens、每步最多查卡 8 次，并允许
最多 2 次 tool-only 纠错重试。`--max-decisions 0` 表示总决策数不设上限。

该命令通过 ocgcore 创建标准 8000 LP、5 张起手、每回合抽 1 的 MR5 对局，并生成逐决策
JSONL。`passive` 是确定性保守基线，`random` 从有界合法动作空间采样，`react` 使用
`.env` 的默认 provider/model。也可以显式指定 `react:deepseek:deepseek-v4-flash`。

换边 round-robin Arena：

```bash
uv run ygo-bench arena \
  --agents passive random react:deepseek:deepseek-v4-flash \
  --decks BlueEyes CyberDragon \
  --seeds 11 29
```

每个 agent/deck/seed 组合都会交换先后手。输出包含胜负、非法动作率、决策耗时、token、
Elo、Glicko-2、先后手分层胜率与 deck matchup matrix。完整协议见
[`docs/BENCHMARK_PROTOCOL.md`](docs/BENCHMARK_PROTOCOL.md)。

重新汇总已有运行：

```bash
uv run ygo-bench report bench_data/runs/<run-name>/_summary.json
```

## 指标

残局任务的主指标是 `solve_rate`。同时记录：

- `engine_completion_rate`：规则引擎正常走到终局的比例；
- `avg_tool_calls`、`avg_model_calls`；
- `input_tokens`、`output_tokens`；
- `avg_elapsed_seconds`；
- 按 termination 分类的失败计数。

完整对局使用换边胜率、非法动作率、平均决策延迟、Glicko-2/Elo 与 deck matchup
matrix。不要把不同卡池、禁限卡表、规则核心提交或观察权限的分数放在同一排行榜中。

完整对局 prompt 位于 `ygobench/agents/prompts/`。每次决策只提供当前玩家视角，并只
暴露当前 responder 的 schema 与 `inspect_card`；系统要求模型恰好提交一次响应工具调用。
工具索引只在当前 observation 有效，下一步不得复用。

## 开发

```bash
uv run pytest
uv run ruff check .
```

第三方许可证和商标声明见 `NOTICE`。研究结果发布时，应记录本仓库提交、submodule
提交、ocgcore/CardScripts/BabelCDB 提交、数据集版本和运行配置。

## 资源来源

- 规则核心：[edo9300/ygopro-core](https://github.com/edo9300/ygopro-core)
- 卡片脚本：[ProjectIgnis/CardScripts](https://github.com/ProjectIgnis/CardScripts)
- 卡片数据库：[ProjectIgnis/BabelCDB](https://github.com/ProjectIgnis/BabelCDB)
- 谜题与验证基座：[yugi-bench-v1](https://github.com/yugi-bench/yugi-bench-v1)
- 预设牌组：[sbl1996/ygo-agent](https://github.com/sbl1996/ygo-agent)
- 禁限卡表：[ProjectIgnis/LFLists](https://github.com/ProjectIgnis/LFLists)

卡图由后端按请求缓存自 YGOPRODeck 图片服务；仓库不会批量提交卡图。

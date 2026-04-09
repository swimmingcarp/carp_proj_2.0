# Repo Agent Entry

本文件是本仓库的固定入口，不重复定义项目事实或执行纪律。

处理本仓库任务时，先读本文件，再按这里的路由进入对应正文。

## 使用方式

如果任务涉及以下任一内容，必须按顺序阅读：

- 策略逻辑修改
- 回测验证
- 报告分析
- commit / baseline 保留

默认做法：

1. 先读 `docs/skills/carp-project-description.md`
   先建立项目认知，弄清楚项目在做什么、主线文件在哪里、主要文件分别负责什么。
2. 再读 `docs/skills/carp-strategy-execution.md`
   先理解总体执行顺序、分析方法、验证纪律和 commit 规范。
3. 进入具体阶段后，按当前任务回到 `docs/skills/carp-strategy-execution.md` 对应部分重新阅读，而不是读一次就结束。

按阶段重点回看：

- 策略调研、根因分析时：
  - 重点看“分析策略的原则和方法”
- 修改代码、设计验证路径时：
  - 重点看“修改代码与验证策略”
- cache 不可变任务时：
  - 重点看“Cache 不可变任务”
- 准备保留 baseline 或创建 commit 时：
  - 重点看“Baseline 与 commit 规范”

如果任务只涉及 README、文档结构或纯文案同步，先读与当前任务直接相关的正文，不要把 `AGENT.md` 当成规则副本。

## 文档分工

- 项目定位、文件职责、主调用链、cache 行为事实，以 `docs/skills/carp-project-description.md` 为准。
- 执行流程、分析方法、验证纪律、baseline / commit 规范，以 `docs/skills/carp-strategy-execution.md` 为准。

## 优先级

- 代码事实与文档冲突时，以代码为准，并同步更新文档。
- `AGENT.md` 只负责入口、阅读顺序和路由。

## 路由

- 项目结构、主调用链、缓存规则、文件位置：
  - 读 `docs/skills/carp-project-description.md`
- 策略研发、验证纪律、baseline 纪律、commit 纪律：
  - 读 `docs/skills/carp-strategy-execution.md`

## Prompt 用法

以后处理本仓库任务时，可以直接在 prompt 里写：

- `先读 AGENT.md，再处理任务`
- `按 AGENT.md 执行`

# Awesome TradingAI 仓库迁移设计

## 1. 文档状态

本文定义把 `jianyingzhihe/quant_bench` 当前 `main` 内容迁移到
`Starlien95/Awesome-TradingAI` `main` 的实施方案。设计基线形成于 2026-08-06，
仓库迁移实施于 2026-08-07。公开可见性切换和 PyPI 发布仍属于独立所有者操作。

本文面向后续实施者，实施时应重新获取两个仓库的远端状态，不能直接复用本文记录的提交号
作为写入依据。

实施时保留正式 contract、unit、integration 测试和 0.x Qlib 兼容 wrapper。公开树移除了
未被 package 或测试引用的生产账本修复、账户清仓、临时下单 smoke 脚本，以及未被 import 的
Guardrails Valid Choices 源码快照。Guardrails validator 继续由用户按上游许可证单独安装。

## 2. 目标与非目标

### 2.1 目标

1. 将经过测试和公开边界审计的当前源码树导入 `Awesome-TradingAI`。
2. 保留现有 Python 分发名、import 路径、CLI、schema ID 和插件入口点，避免无收益的 API 破坏。
3. 将仓库级名称、链接、徽章、站点标题和说明更新为 `Awesome TradingAI`。
4. 重写简明的英文和中文 README，只陈述已有功能、使用入口、安全边界和文档位置。
5. 补齐 Python API、方法接口、交易接口、数据契约和目录职责的索引。
6. 通过目标仓库分支、CI、代码审查和普通合并完成迁移，保留可审核、可回退的历史。
7. 公开前完成凭据、运行数据、许可证、依赖许可证、安装包和匿名访问检查。

### 2.2 非目标

1. 本次迁移不改变 benchmark 计算语义、CSV 列、manifest 字段或 schema 版本。
2. 本次迁移不重命名 Python distribution `quant-bench`、package `quant_bench`、CLI
   `quant-bench` 或 entry point group。
3. 本次迁移不导入本地 `fingpt_crypto_repro/`、`qlib_model_trade/`、运行日志、模型权重、
   账户数据或密钥。
4. 本次迁移不发布实时持仓、精确账户值、订单标识、原始新闻、prompt、reasoning 或私有路径。
5. 本次迁移不启用自动 PyPI 发布，也不运行真实订单、账户对齐或清仓操作。
6. 本次迁移不把公共展示站点和本地分析仪表盘合并为同一数据边界。

## 3. 已核验基线

核验日期为 2026-08-06，时区为 `Asia/Shanghai`。

| 对象 | 核验结果 |
| --- | --- |
| 源仓库 | 本地 `quant_bench` checkout |
| 源分支 | `main` |
| 源 `HEAD` | `41b224f22f9981e04f3baf62cc544690e4f1ecb3` |
| 源 `origin/main` | 与源 `HEAD` 相同 |
| 源跟踪文件数 | 562 |
| 目标仓库 | `Starlien95/Awesome-TradingAI` |
| 目标分支 | `main`，远端仅发现该分支 |
| 目标 `HEAD` | `970f4de2d5f399b853b5d539ab70230d1287a8af` |
| 目标提交数 | 7 |
| 目标跟踪文件 | 仅 `README.md` |
| 目标许可证 | 未发现 `LICENSE` |
| 目标匿名访问 | GitHub 页面和未认证 API 返回 404，公开前必须由仓库所有者核对 visibility |

目标 README 当前只有项目愿景，没有源码、许可证、CI、安装步骤或接口说明。目标仓库历史应保留，
迁移不得通过 force push 覆盖。

源工作区含用户已有未提交改动。迁移输入必须来自单独的干净导出目录，不能直接在该工作区执行
`git add -A`。

## 4. 测试证据与发布边界

### 4.1 已执行测试

以下测试基于源 `main` 和当前待迁移改动执行。迁移实施时应在新目标分支上完整重跑。

| 范围 | 命令或方式 | 当前结果 |
| --- | --- | --- |
| 静态检查 | `ruff check .` | 通过 |
| 严格类型检查 | `mypy --strict src/quant_bench` | 189 个源码文件通过 |
| 默认测试集 | `pytest -q` | 118 passed，3 skipped；skip 为未安装的 Torch 和 Qlib optional 依赖 |
| 文档 | `mkdocs build --strict` | 通过 |
| 发布边界 | `tools/check_release.py` | `valid=true` |
| FinMem 公开边界 | `tools/check_finmem_public.py` | `valid=true` |
| workflow 审计 | `quant-bench workflows audit` | 202 个文件通过 |
| runtime smoke | `tools/smoke_runtime.py` | 通过 |
| wheel 与 sdist | build 后运行 `check_wheel.py`、`check_sdist.py` | 通过 |
| 干净 wheel 安装 | 新虚拟环境安装、`pip check`、CLI 实际执行 | 通过 |
| 顶层 CLI | 19 个命令组 `--help` | 全部可加载 |
| Qlib optional | `pyqlib==0.9.7`、`lightgbm==4.7` 环境 | 12 passed |
| Torch optional | 官方 CPU `torch==2.7.1+cpu` 环境 | MacroHFT converter 2 passed |
| FinMem optional | `[finmem,finmem-live]` 环境 | 8 passed，`pip check` 通过 |
| Dashboard 自动测试 | Streamlit AppTest | 11 passed |
| Dashboard 浏览器验收 | 空/有数据、桌面/390px、四页和四个 Run Explorer tab | 通过，无 console error |
| Runtime 安全门禁 | 七种 runtime 初始化、检查、错误 token 启动 | 初始化通过，错误 token 在启动前拒绝 |
| 统一方法接口 | canonical 与 FinGPT 的 check、plan、run、status | 通过 |

本轮发现并修复了以下可复现问题：

1. 三处 Pandas 返回值未满足严格 `mypy`，现已使用明确 `cast` 约束返回类型。
2. grid sweep 的 `0.000001` 被 JSON 序列化为 `1e-06` 后，经 YAML 解析成字符串，现改为先按
   JSON 标量解析，并新增 unit 和 integration 回归测试。
3. Dashboard 窄屏下长路径的 inline code 被裁切，现增加可换行样式并完成 390px 浏览器复验。
4. Python 3.12 的 dev 环境会解析到 NumPy 2.5，而 `mypy` 按 Python 3.10 目标检查时无法解析
   该版本 stub 中的 3.12 类型语法。`dev` extra 现约束 NumPy `<2.3`，普通运行时依赖保持不变。

### 4.2 未执行的受控验收

以下功能依赖外部服务、私有数据、较大模型、账户权限或危险操作，本轮没有执行。它们不能进入
“已全面保证”的发布声明。

| 范围 | 未执行原因 | 发布处理 |
| --- | --- | --- |
| Python 3.10、3.11 本地矩阵 | 当前主机没有对应解释器，Docker daemon 未运行 | 由 GitHub Actions 3.10 至 3.12 矩阵验证 |
| OKX 公共行情与 CCXT 在线拉取 | 会访问外部网络 | 在无凭据的受控网络 job 中单独验证 |
| OKX 账户读取、demo/live 订单 | 涉及账户和订单权限 | 仅由所有者按独立 runbook 人工验收 |
| FinGPT 13B、LoRA、GPU 推理 | 公共仓库不分发权重，当前验收只覆盖 dry-run 和研究流水线 | 发布文档明确 optional 依赖与模型来源 |
| Qdrant、LLM endpoint | 需要私有服务与凭据 | 不作为公共 CI 条件，保留 doctor 和配置校验 |
| 外部 `ai_trade` 四个进程 | 属于独立仓库和部署边界 | 只验收 process adapter 与安全门禁 |
| FinMem 恢复数据全集 | 公共树只含 checksum 元数据，不含私有本地数据 | 继续验证缺失、篡改和合法样例路径 |
| 真实交易结果展示 | 需要脱敏延迟公共快照流水线 | 不写入 README 当前功能列表 |

### 4.3 迁移放行条件

迁移实施可以开始的条件：

1. 当前源码树的最终全量静态检查、测试、文档和打包检查通过。
2. 目标仓库 `origin/main` 仍指向实施记录中的固定提交，或实施者重新审查新的差异。
3. 待导入文件清单通过 `git status`、secret scan、release check 和人工审查。
4. 仓库所有者确认许可证署名、`CITATION.cff` 作者、PyPI 包所有权和 visibility 计划。

真实账户和外部模型验收不阻塞源码导入分支。它们阻塞对应 live 能力的发布声明。

## 5. 迁移策略

### 5.1 历史处理

采用“保留目标历史的净化源码树导入”：

1. 从目标 `main` 创建专用迁移分支。
2. 从源固定提交和审核过的未提交改动生成 allowlist 导出树。
3. 将导出树覆盖到目标工作树，保留目标 `.git`。
4. 形成一个源码导入提交，再形成一个仓库品牌与文档提交。根据审查规模可以拆分测试修复提交。
5. 通过 Pull Request 合入目标 `main`。

禁止直接 merge 源仓库完整历史。源历史曾承载研究与运行材料，完整历史导入会扩大隐私、体积和
许可证审计面。禁止用 orphan branch 或 force push 替换目标历史。

建议分支名：

```text
migration/import-quant-bench-41b224f
```

建议提交组织：

```text
1. Import sanitized quant-bench source tree
2. Fix verified typing, sweep parsing, and dashboard issues
3. Adopt Awesome TradingAI repository identity and documentation
4. Regenerate release metadata and lock migration checks
```

### 5.2 干净工作区

实施必须使用专用目标 clone，例如：

```bash
git clone git@github.com:Starlien95/Awesome-TradingAI.git /tmp/awesome-tradingai-migration
git -C /tmp/awesome-tradingai-migration fetch --prune origin
git -C /tmp/awesome-tradingai-migration switch -c migration/import-quant-bench-41b224f origin/main
```

创建分支前记录：

```bash
git -C /tmp/awesome-tradingai-migration rev-parse origin/main
git -C /path/to/quant_bench rev-parse origin/main
git -C /path/to/quant_bench status --short
```

若目标 SHA 不等于审核基线，应先阅读新增提交和文件，再调整导入方案。实施者不能自动删除目标方
新增内容。

建议在获得仓库所有者授权后创建目标保护 tag：

```text
pre-quant-bench-import-20260806
```

该 tag 只能普通 push，不能移动或覆盖。

### 5.3 允许导入的内容

导入 allowlist：

1. `src/quant_bench/` 可发布源码。
2. `configs/` 配置模板和 schema。
3. `tests/` 自动测试。
4. `docs/` 文档。
5. `tools/` 只读检查、构建和维护工具。
6. `.github/` CI 与仓库配置。
7. `release/` 中重新生成并校验的 SBOM、依赖许可证和发布清单。
8. 根目录的 package、license、security、contribution、citation 和文档配置文件。

### 5.4 禁止导入的内容

禁止把下列内容复制到目标仓库，包括未跟踪文件和任何嵌套 `.git`：

```text
.env
.env.*
*.pem
*.key
*credentials*
*secret*
.venv/
venv/
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
site/
dist/
build/
runs/
local_data/
csv_data/
qlib_data/
qlib_models/
model_weights/
mlruns/
hyper_runs/
output/
*.pkl
*.pth
*.pt
*.safetensors
*.sqlite
fingpt_crypto_repro/
qlib_model_trade/
```

`.env.example`、`*.pth` 测试文件或 fixture 只有在 release allowlist 明确允许、内容经过审查且
工具检查通过时才能例外。迁移脚本应按跟踪文件 allowlist 复制，禁止依赖宽泛的排除规则作为唯一
保护。

## 6. 名称和兼容性策略

| 层级 | 迁移后名称 | 处理原则 |
| --- | --- | --- |
| GitHub repository | `Starlien95/Awesome-TradingAI` | 更新仓库级链接和徽章 |
| 产品展示名 | `Awesome TradingAI` | README、文档站、citation 使用 |
| 可选副标题 | `Can AI Make Money in Crypto?` | 是否保留由所有者决定，不作为包名 |
| Python distribution | `quant-bench` | 保持 |
| Python import | `quant_bench` | 保持 |
| CLI | `quant-bench` | 保持 |
| schema ID 和 CSV version | 当前值 | 保持 |
| plugin entry point groups | 当前值 | 保持 |
| benchmark engine 名称 | `quant-bench` | 文档说明其为 Awesome TradingAI 的 packaged engine |

仓库名变更不应触发 import 级重命名。若未来需要 `awesome_tradingai` package，应通过独立 major
version 提案、兼容 shim、deprecation 周期和迁移指南处理。

## 7. 文件级改造清单

### 7.1 必须更新

| 文件 | 改造内容 |
| --- | --- |
| `README.md` | 改为简明英文入口，链接和 badge 指向目标 `main` |
| `README.zh-CN.md` | 与英文结构一致，保持中文表达自然 |
| `pyproject.toml` | project URLs 改到目标仓库，Documentation 使用 `main`；distribution 和 package 名保持 |
| `.github/workflows/ci.yml` | push branch 从 `opensource` 改为 `main`，PR 保持启用 |
| `SECURITY.md` | 支持分支和安全报告说明改为目标仓库事实 |
| `CITATION.cff` | title、repository URL、version/release date 与作者信息经所有者确认后更新 |
| `mkdocs.yml` | `site_name`、description、`repo_url`、`edit_uri` 和 nav 更新 |
| `docs/index.md` | 使用新产品名，保留 package/CLI 名说明 |
| `CONTRIBUTING.md` | clone URL、分支、验证命令和 PR 流程更新 |
| `CHANGELOG.md` | 记录仓库迁移，不伪造 package release |
| `docs/MIGRATION.md` | 增加从旧 GitHub 地址迁移的说明 |
| `release/*` | 使用目标树重新生成，禁止手工替换 JSON 中的旧 URL |

### 7.2 精确搜索和分类处理

实施时运行：

```bash
rg -n "jianyingzhihe|quant_bench/tree/opensource|branches/opensource|origin/opensource|site_name: quant-bench" . \
  -g '!release/sbom.cdx.json' -g '!release/dependency-licenses.json'
```

命中项分为三类：

1. 当前仓库元数据和用户入口，替换为目标地址和 `main`。
2. 历史设计依据，例如 `docs/design/OPEN_SOURCE_REFACTOR.md` 中对 `origin/opensource` 的记录，
   保留原文并增加迁移说明，禁止改写历史事实。
3. 生成文件，通过官方生成命令重新生成，禁止手工编辑。

### 7.3 许可证与 provenance

源树采用 MIT，并含第三方 notices、许可证策略和 provenance register。迁移应保留这些文件及其
原始归属。

根 `LICENSE` 的版权行由所有者确认以下方案之一：

1. 保留 `quant-bench contributors`，新增 Awesome TradingAI 仓库说明。
2. 增加 `Awesome TradingAI contributors`，同时保留原有 contributor 归属。

禁止把原作者和第三方归属整体替换成新仓库名称。`CITATION.cff` 作者列表、联系人和仓库 URL
需要分别核对，不能仅按品牌做字符串替换。

## 8. README 设计

### 8.1 写作原则

README 只承担快速理解和首次成功运行。详细接口、配置、方法限制和维护流程放入 docs。英文
README 建议控制在 120 至 180 行，中文版本结构一致。

禁止写入：

1. 内部服务器历史、私有目录、账户信息或生产运行规模。
2. 未被当前源码和测试支持的功能。
3. “持续展示真实资金结果”等尚无公开数据流水线支撑的承诺。
4. 只展示收益、不展示回撤、失败区间和数据质量的宣传性描述。
5. 把 research backtest、paper、exchange demo 和 live 混为一个运行模式。

### 8.2 建议标题和一句话说明

```markdown
# Awesome TradingAI

An open-source cryptocurrency AI trading benchmark and local analysis toolkit.
The packaged `quant-bench` engine provides reproducible backtests, method
adapters, paper/runtime safety gates, standardized artifacts, and a read-only
dashboard.
```

中文一句话说明应直接列出用途，不写夸张愿景。

### 8.3 README 结构

1. 标题、一句话说明、CI/Python/license/docs badge。
2. `What it includes`，用一张短表列出：
   - traditional ML and Qlib workflows
   - MacroHFT adapter and converter
   - FinGPT news research and dry-run
   - FinMem paper/live adapters
   - unified Method interface
   - standardized artifacts and CSV contracts
   - local read-only analytics dashboard
3. `Install`，给出 core 和常用 extras，不一次安装所有重依赖。
4. `Five-minute offline run`，只使用 fixture/canonical 路径，无网络、无凭据。
5. `Open the dashboard`，说明 workspace 只读和命令。
6. `Methods and execution modes`，明确 `research`、`paper`、`demo`、`live` 门禁。
7. `Repository map`，列出主要目录。
8. `Interfaces and documentation`，链接 CLI、Python API、Method contract、trading API、data
   contract、artifacts、runtime safety。
9. `Safety and data boundary`，强调默认不下单、凭据不入库、生成物不公开。
10. `Contributing, citation, license`。

### 8.4 README 验收

1. 在全新虚拟环境逐条复制执行命令。
2. core 快速路径不得安装 Torch、Qlib、Streamlit、Transformers 或交易所 SDK。
3. 所有相对文档链接和 badge URL 可访问。
4. 中英文功能表、命令和模式定义一致。
5. README 中每个功能声明都能映射到源码入口、测试或明确标注的 optional adapter。

## 9. 接口文档设计

迁移不改接口，实现阶段补齐入口文档和交叉链接。

### 9.1 新增 `docs/reference/python-api.md`

文档应覆盖：

1. 稳定公开 import 列表，避免用户依赖内部模块。
2. 配置加载、override 解析和 schema 校验。
3. dataset、feature、label、split、model、strategy、backtest、metrics 的输入输出。
4. artifact、run manifest、hash、atomic write 的公开方法。
5. 最小 offline 示例和异常语义。
6. API stability 级别：public、experimental、internal。

### 9.2 扩充 Method interface

以 `docs/how-to/unified-methods.md` 为主文档，明确：

```text
MethodRunSpec
  -> capability validation
  -> MethodRunner
  -> MethodRunResult
  -> durable run record and artifacts
```

必须说明 `ExecutionMode`、confirmation token、workspace、artifact paths、subprocess boundary、
status 和错误码。给出 canonical、FinGPT、FinMem 及 external process 各一个最小示例。

### 9.3 Model plugin 接口

在 `docs/how-to/add-model.md` 和 Python API 文档中说明：

1. `ModelPlugin` 需要实现的生命周期。
2. entry point group 和发现方式。
3. fit/predict 输入输出与 dtype/index 约束。
4. artifact 持久化、模型版本和安全反序列化边界。
5. scaffold、测试、注册和兼容性要求。

### 9.4 数据与 artifact 接口

现有 `docs/DATA_CONTRACT.md`、`docs/runtime/CSV_SCHEMA.md` 和
`docs/reference/artifacts.md` 保持事实源角色。README 只链接这些文档。

接口索引应列出：

1. canonical market/features/labels/predictions/returns 列和时间语义。
2. runtime metrics/signals/volume/trades/orders/fills 等 schema。
3. `run_manifest.json` 必填字段、相对路径和 path containment。
4. schema ID、version 和兼容规则。
5. 缺列、空文件、重复时间、NaN/Inf、时区错误的失败行为。

### 9.5 统一交易接口

`docs/how-to/unified-trading-interface.md` 作为交易接口主文档，README 和 CLI reference 链接到该
文件。公开契约包括：

```text
OrderRequest
AccountSnapshot
ExecutionReport
TradingCycleResult
TradingBackend
TradingService
```

文档必须明确只有 backend 能访问交易所下单方法，策略代码只产生意图。paper、demo、live 的
独立门禁和确认 token 不得因迁移而放宽。

### 9.6 兼容性政策

迁移版发布前在 docs 中新增以下规则：

1. patch 版本只做兼容修复。
2. minor 版本允许新增可选字段和 experimental 接口，默认行为保持。
3. 删除列、重命名 schema、改变资金或时间语义属于 major 变更。
4. deprecated 接口至少保留一个 minor 周期，并提供迁移说明和测试。
5. GitHub 仓库品牌变更不等于 Python API major 变更。

## 10. 项目目录说明

README 使用短表，docs 提供完整说明。

| 路径 | 主要职责 | 公开边界 |
| --- | --- | --- |
| `src/quant_bench/` | 安装包源码、CLI、方法、契约、运行与 dashboard | 公开 |
| `src/quant_bench/core/` | 可复用 benchmark 核心协议与基础能力 | 公开 API 需单独标注 |
| `src/quant_bench/methods/` | traditional、MacroHFT、FinGPT、FinMem 等方法 adapter | 权重和私有数据不入库 |
| `src/quant_bench/runtime/` | paper/demo/live 的配置、状态和交易运行边界 | 默认安全门禁保持开启 |
| `src/quant_bench/dashboard/` | 本地只读结果发现、分析和曲线展示 | 不访问账户、不下单 |
| `configs/` | 可版本化配置模板与 schema | 不含凭据和本机路径 |
| `tests/` | unit、integration、package 和安全回归 | fixture 必须脱敏且轻量 |
| `docs/` | 教程、how-to、concept、reference 和 maintainer 设计 | 历史事实与现状分开 |
| `tools/` | release、wheel、sdist、runtime 等检查工具 | 不默认执行危险操作 |
| `release/` | SBOM、依赖许可证和发布审计产物 | 每次目标构建重新生成 |
| `.github/` | CI、依赖更新和仓库协作配置 | 分支名改为 `main` |

生成目录应由 CLI 创建在用户 workspace，不能把 `runs/` 或 `local_data/` 当成仓库源码目录。

## 11. CI 与发布设计

### 11.1 Pull Request 必过检查

目标 `main` 应设置 branch protection，至少要求：

1. Python 3.10、3.11、3.12 core matrix。
2. `ruff check .`。
3. `mypy --strict src/quant_bench`。
4. `pytest -q`，同时报告 optional skip 原因。
5. Qlib integration job。
6. wheel clean install 和 offline FinGPT job。
7. Streamlit AppTest job。
8. `mkdocs build --strict`。
9. release、FinMem public、wheel、sdist、workflow audit 和 secret checks。

浏览器手工验收至少覆盖一次空 workspace、有数据 workspace、桌面和 390px 宽度。外部网络和账户
测试放到手工 workflow，不能让 PR job 持有交易凭据。

### 11.2 发布元数据

实施顺序：

1. 更新仓库 URL 和文档 URL。
2. 在干净目标树重建 wheel、sdist、SBOM 和 dependency licenses。
3. 对生成物运行一致性检查。
4. 从 wheel 外部目录执行 CLI 和 README quickstart。
5. 核对 wheel 不含 `.env`、账户数据、运行日志、权重和源仓库私有路径。

当前 package 仍是 `0.1.0` Alpha。仓库迁移提交不得自行发布新版本。版本号、tag 和 PyPI upload
由所有者单独批准。

### 11.3 visibility 门禁

目标仓库当前未通过匿名 GitHub 页面和 API 访问验证。visibility 切换是仓库所有者操作，执行前
必须完成：

1. 目标分支所有 CI 通过。
2. secret/history scan 通过。
3. LICENSE、NOTICE、provenance、dependency licenses 和 SBOM 审查完成。
4. README 不含未实现承诺或私有路径。
5. release archive 内容核验完成。

切换后从未登录环境执行：

```bash
git clone https://github.com/Starlien95/Awesome-TradingAI.git
```

同时核对网页、raw README、LICENSE、文档链接和 release asset。匿名访问失败时不发布公告。

## 12. 分阶段实施

### Phase A：冻结与复核

1. fetch 两个远端并记录 SHA。
2. 阅读目标新增提交。
3. 记录源 dirty tree 的所有文件和归属。
4. 在源干净导出树重跑测试矩阵。
5. 确认所有者决策项。

退出条件：输入 commit、文件 allowlist、许可证决策和测试报告已固定。

### Phase B：净化导入

1. clone 目标仓库到专用目录。
2. 创建迁移分支和可选保护 tag。
3. 从源 tracked allowlist 和审核过的新增文件构建导出树。
4. 运行 secret、path、size、suffix、license 和 release 检查。
5. 导入目标分支并生成首个 diff 报告。

退出条件：目标分支只含允许公开内容，目标旧 README 仍可在历史中访问。

### Phase C：品牌、README 与接口文档

1. 更新仓库级名称和 URL。
2. 按本设计重写简短 README 与中文 README。
3. 补充 Python API、Method、plugin、data、artifact、trading interface 索引。
4. 更新目录说明、贡献指南、安全说明和 citation。
5. 保留历史文档中的历史分支事实。

退出条件：所有功能声明可映射到代码或测试，所有链接通过检查。

### Phase D：CI 和发布产物

1. 更新 CI 分支、矩阵和 dashboard job。
2. 重建 release metadata。
3. 构建 wheel/sdist 并执行干净安装。
4. 运行完整自动测试和浏览器验收。
5. 形成 release-readiness 报告。

退出条件：必过检查全部通过，未执行的外部验收被明确列出。

### Phase E：Pull Request 与公开

1. 推送迁移分支，禁止 force push。
2. 创建 PR，附源/目标 SHA、文件统计、测试报告和安全审计。
3. 由代码、安全、许可证三类 reviewer 审查。
4. 普通合并到 `main`。
5. 所有者按门禁切换 visibility。
6. 在匿名环境执行 clone 和快速开始。

退出条件：目标 `main` 可复现构建，匿名用户能按 README 完成离线首次运行。

## 13. 验收命令

在目标迁移分支根目录执行：

```bash
ruff check .
mypy --strict src/quant_bench
pytest -q
mkdocs build --strict
python tools/check_release.py
python tools/check_finmem_public.py
python tools/smoke_runtime.py
quant-bench workflows audit
python -m build
python tools/check_wheel.py dist/*.whl
python tools/check_sdist.py dist/*.tar.gz
git diff --check origin/main...HEAD
git status --short
```

wheel 安装验收必须在仓库外部的新虚拟环境执行：

```bash
python -m venv /tmp/awesome-tradingai-wheel-test
/tmp/awesome-tradingai-wheel-test/bin/python -m pip install dist/*.whl
/tmp/awesome-tradingai-wheel-test/bin/python -m pip check
cd /tmp
/tmp/awesome-tradingai-wheel-test/bin/quant-bench quickstart --workspace /tmp/awesome-tradingai-quickstart
/tmp/awesome-tradingai-wheel-test/bin/quant-bench doctor --workspace /tmp/awesome-tradingai-quickstart
```

README 中的 sweep、Qlib、FinGPT dry-run、FinMem paper 和 dashboard 命令应逐条复制执行。任何命令
依赖网络、GPU、账户或额外数据时，README 必须在命令前明确标注。

## 14. 回退方案

1. PR 合并前直接关闭 PR，目标 `main` 不受影响。
2. PR 合并后若发现问题，创建普通 revert commit，禁止重写 `main` 历史。
3. 保护 tag 用于确认迁移前目标状态，不用于 force reset。
4. 若已切换 public 后发现凭据或账户数据，立即将仓库设回 private、撤销或轮换凭据、停止 release，
   再按安全事件流程清理历史。
5. PyPI 发布属于独立动作。仓库回退不能撤回已发布包，发布前必须使用独立确认门禁。

## 15. 所有者决策项

实施前需要明确：

1. 目标仓库公开日期和 visibility 切换负责人。
2. `Can AI Make Money in Crypto?` 是否保留为副标题。
3. 根 `LICENSE` 的新增版权行和原 contributor 归属方案。
4. `CITATION.cff` 作者顺序、联系人和首个 Awesome TradingAI release date。
5. PyPI `quant-bench` 的 owner、release 版本和发布方式。
6. `main` branch protection 的 required checks、reviewer 和 merge 策略。
7. 是否创建迁移前保护 tag。
8. 公共展示站点何时具备经过延迟、脱敏、校验的 snapshot 后再进入 README 功能列表。

## 16. 实施完成定义

同时满足以下条件才算迁移完成：

1. 目标 `main` 保留原历史并包含审核过的源码树。
2. 仓库级名称和链接统一，Python API 保持兼容。
3. 英文和中文 README 简明、可执行、只描述已有功能。
4. 接口、目录、数据、安全和贡献文档可从 README 到达。
5. 自动测试、可选依赖测试、严格文档构建、wheel/sdist 和公开边界检查通过。
6. 未执行的账户、外部服务和大型模型验收有清晰说明。
7. 仓库中没有凭据、私有运行数据、大型权重、账户标识或嵌套仓库。
8. 匿名用户能够 clone，安装 core wheel，并完成离线 quickstart。
9. 没有 force push，没有将源完整历史引入目标，没有自动触发真实交易。
10. CSV schema、manifest 和 schema ID 无迁移性变更；若未来需要重命名，另立 major version 设计。

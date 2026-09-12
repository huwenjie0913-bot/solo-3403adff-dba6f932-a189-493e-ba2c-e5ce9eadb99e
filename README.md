# jtag-recon — JTAG 扫描链还原 API

供电子维修人员在**不连接真实硬件**的前提下，从已采集的 TDI/TDO 位流还原未知
JTAG 扫描链。本机运行的 REST API：FastAPI + Pydantic + SQLite。

## 功能

- 上传器件 BSDL 文件，解析 `INSTRUCTION_LENGTH`、`INSTRUCTION_OPCODE`
  （IDCODE / BYPASS / SAMPLE / PRELOAD）、`INSTRUCTION_CAPTURE`、
  `IDCODE_REGISTER`（含掩码）、`BOUNDARY_LENGTH`。
- 接收多组 IR 指令扫描及对应 TDI/TDO 位流、DR 扫描（BYPASS / IDCODE /
  SAMPLE），结合寄存器长度、捕获模式与位流对齐关系，推断：
  - 器件顺序（position 0 = 靠近 TDI）；
  - 各段 IR 长度；
  - 可能缺失的未知器件（利用 IR 捕获固有的 `01` LSB 特征识别）。
- 支持**锁定**已确认的器件与链位、标记**不可靠位段**后重算。
- 候选结果包含逐位映射、命中约束与未解释片段，并检查：
  `tdo_constant`（TDO 恒定）、`overall_offset_by_one`（整体偏移一位）、
  `bsdl_length_conflict`（BSDL 长度冲突）、`idcode_mask_mismatch`
  （IDCODE 掩码不符）、`indistinguishable_candidates`（候选无法区分）。
- 生成**只使用 IDCODE、BYPASS、SAMPLE/PRELOAD** 的 SVF 复核序列——
  绝不加载 EXTEST/INTEST/CLAMP，不会驱动器件输出引脚。
- 原始采样与推断版本存入 SQLite，可导出 JSON 链定义与 SVF 文件。
- **重复采样一致性与间歇故障定位**：对同一会话内带标签的多次 IR/DR
  TDI/TDO 采样，按 `(类型, 指令)` 分组、按 TDI 回显逐次对齐，用已保存的
  链版本把 TDO 位映射到链位置 / 器件 / 寄存器位，统计每一位的稳定值、
  翻转次数和缺失区间，并区分：
  - `segment_fixed_offset` 整段固定偏移（本次捕获内容整体平移若干位）；
  - `intermittent_device_toggle` 单器件间歇翻转（同一位在多次运行上反复异常）；
  - `tdo_constant` TDO 整段恒定（断链 / TDO 短接 / TAP 停在复位）；
  - `isolated_sample_anomaly` 孤立采样异常（单次探针毛刺 / 读错一位）；
  - 以及 `register_length_mismatch` / `alignment_failed`（长度不符或回显
    找不到的运行，给出具体运行与 TDO 位并排除在统计外）。
  批次与诊断写入 SQLite，支持 JSON 查询；不改动原始采样与既有推断版本。

## 运行

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8000        # 本机 http://127.0.0.1:8000/docs
python3 -m pytest tests/                # 测试
```

数据库路径用环境变量 `JTAG_RECON_DB` 指定（默认 `./jtag_recon.db`）。

## 位序约定

- 位流为线上顺序：索引 0 = 该线上第一个移出的位（TDO：链中最早出来的位）。
- 扫描时 TDO 前 L 位是寄存器捕获内容（L = 寄存器总长），之后是延迟 L 位的 TDI 回显。
- `bit_order`（`lsb_first` 默认 / `msb_first`）决定多位字段（IDCODE）的解释方式。
- BSDL 模式按标准 MSB 在左书写，内部自动转为移出顺序。

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/devices/bsdl` | 上传 BSDL，返回解析后的器件模型与 `device_id` |
| GET  | `/devices` | 列出已存器件 |
| POST | `/infer` | 链推断（器件、位序、采样、锁定、不可靠段），返回候选 + 检查 + `version_id` |
| GET  | `/captures` | 查询原始采样（可按 session 过滤） |
| GET  | `/versions` / `/versions/{id}` | 查询推断版本 |
| GET  | `/versions/{id}/export?candidate=0` | 导出 JSON 链定义 |
| GET  | `/versions/{id}/svf?candidate=0` | 下载 SVF 复核序列 |
| POST | `/consistency` | 对带标签的多次采样做一致性分析与间歇故障定位，返回分组结果 + `batch_id` |
| GET  | `/consistency` | 列出一致性批次（可按 session 过滤） |
| GET  | `/consistency/{id}` | 查询某批次的运行、分组、逐位统计与诊断 |

## 推断流程

1. **对齐**：在 TDO 中定位 TDI 回显，得到寄存器总长（支持尾部截断与不可靠位）。
2. **IR 分段**：把 IR 捕获区按各器件 `INSTRUCTION_CAPTURE` 模式切分；
   无法匹配的区段若带 `01` LSB 特征则记为未知器件（长度 2–16）。
3. **DR 验证**：BYPASS 扫描核对器件数；IDCODE 扫描逐 32 位段在掩码下比对；
   SAMPLE/PRELOAD 扫描核对 `BOUNDARY_LENGTH` 总和。
4. **评分与过滤**：应用锁定约束，按得分排序，返回前 10 个候选及诊断检查。

## `/infer` 请求示例

```json
{
  "session": "board-42",
  "bit_order": "lsb_first",
  "devices": [{"device_id": 1}, {"device_id": 2}],
  "captures": [
    {"kind": "ir", "tdi": "101100...", "tdo": "1000001000..."},
    {"kind": "dr", "instruction": "BYPASS", "tdi": "111000...", "tdo": "00..."},
    {"kind": "dr", "instruction": "IDCODE", "tdi": "0110...", "tdo": "0..."}
  ],
  "locks": [{"position": 0, "device": "DEVA"}],
  "unreliable": [{"capture_index": 0, "start": 3, "end": 4}],
  "max_unknown": 2
}
```

锁定与不可靠段修改后再次 POST `/infer` 即可重算，每次都会保存为新版本。

## 重复采样一致性 / 间歇故障定位

POST `/consistency`，引用一个**已保存**的链版本（`version_id` + `candidate`），
提交同一会话内带唯一标签的多次采样：

```json
{
  "session": "board-42",
  "version_id": 3,
  "candidate": 0,
  "note": "复测三冷启动 + 两热机",
  "runs": [
    {"label": "cold-1", "kind": "ir", "tdi": "101100...", "tdo": "1000001000..."},
    {"label": "cold-2", "kind": "ir", "tdi": "101100...", "tdo": "1000001000..."},
    {"label": "hot-1",  "kind": "ir", "tdi": "101100...", "tdo": "1000011000..."}
  ]
}
```

处理流程：

1. **分组**：按 `(kind, instruction)`（指令大小写/空格归一）分组，IR 以 `instruction=null`。
2. **逐次对齐**：在 TDO 中定位 TDI 回显得到该次寄存器长度；与版本长度一致
   才映射，回显提早 1–4 位视为捕获头截断（记为缺失区间），其余偏移判为
   `register_length_mismatch`，完全找不到回显判 `alignment_failed`，整段不变判
   `tdo_constant`——这些运行给出具体运行与 TDO 位并**排除在逐位统计之外**。
3. **逐位映射统计**：`mapped` 模式按版本把 TDO 位映射到链位置 / 器件 /
   寄存器本地位（位 0 = 最靠近 TDO）；无法从版本推导 DR 布局的自定义指令用
   `wire` 模式按原始 TDO 索引统计。每位返回 `stable`、`count_0/1`、`toggles`、
   `observed_runs/missing_runs`、与版本 IR 捕获值对照的 `expected`。
4. **故障分类**（每条诊断给 `first_anomaly`、`affected_positions/devices` 与
   支撑结论的原始 `chain_bit` / 寄存器位 / 各运行 `tdo_index`）：
   - `segment_fixed_offset`：某次捕获内容与其它运行整体平移 ±1/±2 位可解释
     ≥90% 的差异——整段固定偏移（时钟 / 探针相位），该次这些位不再计入器件故障；
   - `intermittent_device_toggle`：同一寄存器位在**多个不同运行**上翻转而其余
     运行稳定——单器件间歇故障；
   - `isolated_sample_anomaly`：仅单个运行、单个位异常，或同一次采样跨多个
     器件的散点异常——该次采样本身的毛刺，而非器件问题；
   - `tdo_constant`：该次 TDO 全程恒定。
5. **缺失区间**：`missing_intervals` 给出在某（几）次已对齐运行中未观测到的
   连续链位范围及涉及运行。
6. **持久化**：批次（含运行原始位流）与完整结果写入
   `consistency_batches` / `consistency_runs`；`captures` 与 `versions`
   表保持不变。

## 项目结构

```
app/
  main.py        FastAPI 路由与导出
  models.py      Pydantic 请求模型（BSDL 位序 -> 移出顺序转换）
  bsdl.py        BSDL 子集解析器
  inference.py   对齐、IR 分段、DR 验证、评分与五项诊断检查
  consistency.py 重复采样分组、TDI 回显对齐、链版本映射、逐位统计与故障分类
  svf.py         安全 SVF 生成（仅 IDCODE/BYPASS/SAMPLE/PRELOAD）
  db.py          SQLite 存储（devices / captures / versions /
                 consistency_batches / consistency_runs）
tests/test_api.py          含理想链模拟器的端到端测试
tests/test_consistency.py  重复采样一致性与故障定位的端到端测试
```

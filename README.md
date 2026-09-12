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

## 项目结构

```
app/
  main.py       FastAPI 路由与导出
  models.py     Pydantic 请求模型（BSDL 位序 -> 移出顺序转换）
  bsdl.py       BSDL 子集解析器
  inference.py  对齐、IR 分段、DR 验证、评分与五项诊断检查
  svf.py        安全 SVF 生成（仅 IDCODE/BYPASS/SAMPLE/PRELOAD）
  db.py         SQLite 存储（devices / captures / versions）
tests/test_api.py  含理想链模拟器的端到端测试
```

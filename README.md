# 会议语音助手系统 — C5 模块 README

> C5 模块：会议纪要生成 / RAG 知识检索 / 语音助手对话 / TTS 语音播报  
> 开发：陈通达

---

## 1. 项目概述

### 1.1 项目名称

基于 AMI 语料库的会议双语字幕与智能语音助手系统

### 1.2 项目目标

面向真实会议场景，构建一套从音频输入到智能问答输出的完整流水线。系统接收会议录音或实时麦克风音频，经过语音识别（ASR）、说话人声纹识别、实时双语翻译后，由 C5 模块负责：

- **会议纪要自动生成**：将全场会议转写总结为结构化纪要（关键决策、行动项、风险、待确认问题）
- **RAG 知识检索**：将历史会议向量化索引，支持自然语言检索
- **语音助手对话**：整合当前会议上下文 + 历史 RAG + 联网搜索 → LLM 推理 → 语音回答
- **TTS 语音播报**：将文字回答转为自然语音，浏览器实时播放

### 1.3 当前完成情况

| 类型 | 完成情况 |
|---|---|
| 基础要求 | 完整实现 C1-C5 流水线：音频预处理 → ASR → 翻译 → 纪要 → 助手 |
| 进阶要求 | RAG 混合检索（语义 + 关键词）、实时声纹识别（CAM++）、级联 vs 端到端翻译对比评估、TTS 语音播报 |
| 支持的主要任务类型 | 中文/英文会议实时转写、双语字幕、说话人识别、结构化纪要、历史会议检索、语音问答 |
| 当前限制 | 需联网调用阿里云 MaaS API；离线声纹校正需 GPU；TTS 仅支持中文 Cherry 音色 |

---

## 2. 整体流程与模块结构

### 2.1 模块边界

| 模块 / 阶段 | 入口文件 / 入口函数 | 主要职责 | 输入 | 输出 |
|---|---|---|---|---|
| C1 音频预处理 | `src/c1_preprocess.py` | 音频格式统一（16kHz 单声道 PCM）、片段切分 | 原始 WAV 音频 | PCM 片段 |
| C2 ASR + 声纹 | `src/c2_asr_ali.py` / `src/online_diarization.py` | 流式语音识别 + 在线说话人声纹聚类 | PCM 音频帧 | transcript JSON（含 speaker 标签） |
| C3 级联翻译 | `src/c3_translate_ali.py` | ASR → LLM 流式翻译 | ASR 文本 | 中文/英文翻译 |
| C4 端到端翻译 | `src/c4_omni_ali.py` / `src/live_translate.py` | 端到端语音→翻译（qwen-livetranslate） | PCM 音频帧 | 直接翻译文本 |
| **C5 纪要** | `src/summary_ali.py` / `server.py:generate_realtime_minutes()` | 将转写总结为结构化纪要 | transcript JSON | product_minutes JSON |
| **C5 RAG** | `src/meeting_rag.py` | 会议文本向量化、混合检索 | transcript + minutes | 检索证据列表 |
| **C5 会议库** | `src/meeting_library.py` | 会议元数据 CRUD、索引管理 | 音频 + 转写 + 纪要 | meeting_index.json |
| **C5 语音助手** | `server.py:assistant_chat` | 整合上下文 + LLM + TTS + 联网搜索 | 用户问题文本 | 文字回答 + base64 音频 |
| **C5 TTS** | `src/c5_tts_ali.py` / `server.py:_tts_audio()` | 文本→语音合成 | 中文文本 | base64 WAV 音频 |

### 2.2 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit 前端 (web/app.py)                │
│  首页  │  会议库  │  实时麦克风  │  字幕  │  语音助手         │
└──────────────┬──────────────────────────────────────────────┘
               │  HTTP REST + WebSocket (PCM)
┌──────────────▼──────────────────────────────────────────────┐
│               FastAPI 后端 (realtime/server.py)              │
│                                                              │
│  ┌────────┐  ┌────────┐  ┌──────────┐  ┌─────────────────┐ │
│  │ C1     │→│ C2 ASR │→│ C3/C4    │→│  C5              │ │
│  │ 预处理 │  │ +声纹  │  │ 双路翻译 │  │  纪要 + RAG      │ │
│  └────────┘  └────────┘  └──────────┘  │  + 助手 + TTS   │ │
│                                         └────────┬────────┘ │
│                                                  │          │
│  ┌───────────────────────────────────────────────┼───────┐  │
│  │  C5 子模块                                    │       │  │
│  │  ├─ meeting_library.py  会议库 CRUD          │       │  │
│  │  ├─ meeting_rag.py      RAG 向量检索         │       │  │
│  │  ├─ summary_ali.py      离线纪要             │       │  │
│  │  └─ c5_tts_ali.py       离线 TTS            │       │  │
│  └───────────────────────────────────────────────┼───────┘  │
└──────────────────────────────────────────────────┼──────────┘
                                                   │
                                    ┌──────────────┴──────────┐
                                    │      阿里云 MaaS         │
                                    │  qwen-plus / qwen-max    │
                                    │  qwen-tts / Cherry       │
                                    │  text-embedding-v4       │
                                    │  paraformer-realtime-v2  │
                                    └─────────────────────────┘
```

### 2.3 一次完整任务的数据流

以"播放 AMI ES2004a 会议并使用语音助手提问"为例：

```
1. 前端选择 AMI ES2004a 会议 → 选择"级联"模式 → 点击播放
2. 音频帧通过 WebSocket 发送到后端 (PCM 16kHz)
3. C2 阿里云 paraformer-realtime-v2 实时 ASR → 返回文本
4. C2 CAM++ 声纹模型 → 实时标注说话人 (参会者A、参会者B...)
5. C3 Qwen LLM 流式翻译 ASR 文本 → 中文翻译
6. C4 qwen-livetranslate 端到端 → 同时产出另一路翻译
7. 前端实时展示：双语字幕 + 说话人标签 + C3 vs C4 延迟对比
8. 播放结束 → 前端调 C5 POST /api/realtime/meeting/save
   ├─ run_local_diarization() → 离线声纹校正
   ├─ generate_realtime_minutes() → 生成结构化纪要
   └─ add_realtime_meeting() → 存入 meeting_index.json
9. 用户切换到"语音助手"面板
10. 用户点击"开始说话" → WebSocket ASR → 识别问题文本
11. 前端调 POST /api/assistant/chat {question: "刚才讨论了什么风险？"}
    ├─ _should_search() → 判断无需联网
    ├─ _meeting_context() → 拉当前会议纪要 + RAG 检索历史会议
    ├─ LLM (qwen-plus) → 生成回答
    ├─ _tts_audio(transition) → 过渡语 TTS (与 LLM 并发)
    └─ _tts_audio(answer) → 回答 TTS
12. 前端收到 response → 播放过渡语音频 → 显示文字答案 → 播放回答音频
13. 对话历史存入 conversation[] 数组，支持多轮追问
```

---

## 3. 模型、数据集与外部资源

### 3.1 模型说明

| 项目 | 内容 |
|---|---|
| LLM（纪要/助手） | `qwen-plus`（默认）/ `qwen-max`（更强） |
| ASR | `paraformer-realtime-v2`（阿里云） |
| 声纹嵌入 | CAM++ / ERes2NetV2（FunASR/ModelScope 本地加载） |
| TTS | `qwen-tts`，Cherry 女声音色（阿里云） |
| Embedding | `text-embedding-v4`，1024 维（阿里云） |
| 端到端翻译 | `qwen3.5-livetranslate-flash-realtime`（阿里云） |
| 是否需 GPU | 声纹模型建议 GPU（CPU 也可运行，较慢）；其余为云 API |
| 是否需联网 | 是（所有核心模型通过阿里云 MaaS API 调用，除声纹模型外） |

### 3.2 数据集说明

| 数据 | 用途 | 来源 | 项目内路径 |
|---|---|---|---|
| AMI Meeting Corpus (8场) | 演示/评估音频源 | 项目预下载 | `data/raw/amicorpus/{id}/audio/` |
| AMI 标注转写 | 评估参照 | 项目预下载 | `data/raw/ami_public_manual_1.6.2.zip` |
| 预生成 ASR/翻译/纪要缓存 | 离线查看 | 系统运行生成 | `outputs/web_cache/`、`outputs/asr/`、`outputs/eval/` |
| 会议库索引 | RAG 数据源 | 首次运行时自动构建 | `outputs/meetings/meeting_index.json` |

---

## 4. 环境安装

### 4.1 运行环境

| 项目 | 要求 |
|---|---|
| Python 版本 | 3.10 |
| 操作系统 | Windows 11 / Linux / macOS |
| GPU 要求 | 不需要（声纹模型 CPU 可用，其余云 API） |
| 主要依赖 | fastapi、uvicorn、streamlit、openai、dashscope、numpy、scipy、requests |
| 包管理 | conda (recommended) / pip |

### 4.2 安装步骤

```bash
# 1. 创建 conda 环境
conda create -n meeting_speech python=3.10 -y
conda activate meeting_speech

# 2. 进入项目目录
cd meeting_speech_system_ami

# 3. 安装 Python 依赖
pip install -r requirements.txt

# 4. 配置 API Key
cp .env.example .env
# 编辑 .env，将 DASHSCOPE_API_KEY 替换为真实阿里云 API Key
# 其他字段（ALI_OPENAI_BASE_URL、ALI_DASHSCOPE_BASE_URL）已预填北京 MaaS 节点地址

# 5. 验证安装
python -c "from src.meeting_library import ensure_meeting_library; ensure_meeting_library(); print('OK')"
```

**常见问题：**

- **`.env` 里的 API Key 是占位符**：必须替换为真实的阿里云 DashScope API Key，否则 ASR/翻译/纪要/助手/TTS 全部不可用
- **`transformers` 提示 PyTorch not found**：不影响运行，声纹模型通过 FunASR/ModelScope 推理，不依赖 PyTorch
- **端口 8765 被占用**：`netstat -ano | grep 8765` 查看占用进程，`taskkill //PID xxx //F` 杀掉

---

## 5. 输入文件与配置文件说明

### 5.1 主要配置文件

| 配置文件 | 作用 | 需要修改的字段 |
|---|---|---|
| `.env` | API Key、模型名、声纹参数 | `DASHSCOPE_API_KEY`（必填）、`ASSISTANT_LLM_MODEL`（可选，默认 qwen-plus）、`ASSISTANT_TTS_VOICE`（可选，默认 Cherry） |
| `config/config.yaml` | 离线流水线路径配置 | `paths.*`（输入/输出文件路径） |
| `config/glossary.json` | 翻译术语表 | 按需添加专有名词翻译对 |

### 5.2 C5 输入文件

| 输入文件 | 用途 | 格式 |
|---|---|---|
| `outputs/web_cache/streaming_cues.json` | 当前会议的流式转写缓存 | `[{speaker, start, end, zh, en, corrected}, ...]` |
| `outputs/web_cache/product_minutes.json` | 当前会议的纪要缓存 | `{stage_summaries, speaker_insights, final_minutes}` |
| `outputs/web_cache/speaker_minutes.json` | 当前会议按发言人维度的摘要 | `{speaker_status, speakers: [{name, summary, ...}]}` |
| `outputs/meetings/meeting_index.json` | 会议库索引（RAG 数据源） | `[{meeting_id, title, date, tags, summary, ...}]` |
| `outputs/meetings/{id}/transcript.json` | 某场会议的转写 | `[{speaker, start, end, source, zh}, ...]` |
| `outputs/meetings/{id}/diarization.json` | 某场会议的声纹结果 | `{turns, speaker_centroids, speaker_count}` |

---

## 6. 完整流程 Demo 运行

### 6.1 Demo 样例说明

| Demo | 输入 | 演示目的 |
|---|---|---|
| Demo 1：播放 AMI 会议 + 查看纪要 | ES2004a 预存音频 | 验证 C1-C5 全流水线：ASR→翻译→声纹→纪要 |
| Demo 2：语音助手提问 | 自然语言问题 | 验证 RAG 检索 + LLM + TTS |
| Demo 3：会议库浏览 | meeting_index.json | 验证会议库管理功能和详情页 |
| Demo 4：实时麦克风录制 | 浏览器麦克风 | 验证端到端实时会议 + 自动保存 + 纪要 |

### 6.2 运行命令

```bash
# 终端 1：启动后端
cd meeting_speech_system_ami
conda activate meeting_speech
python -m uvicorn realtime.server:app --host 127.0.0.1 --port 8765

# 终端 2：启动前端
cd meeting_speech_system_ami
conda activate meeting_speech
python -m streamlit run web/app.py --server.address 127.0.0.1 --server.port 8501

# 浏览器打开 http://127.0.0.1:8501
```

### 6.3 关键参数说明

| 参数 | 说明 |
|---|---|
| `--host 127.0.0.1` | 后端监听地址（仅本机，安全） |
| `--port 8765` | 后端端口（前端 `web/app.py` 硬编码了此端口，如修改需同步改） |
| `REALTIME_PORT` | 前端 `app.py` 中的常量，默认 `8765`，与后端端口一致 |
| `ASSISTANT_LLM_MODEL` | `.env` 中配置，默认 `qwen-plus`，可改为 `qwen-max` 获得更强的摘要质量 |
| `search_mode` | 前端助手面板的联网策略：`force`（每次联网）/ `auto`（智能判断）/ `off`（不联网） |

### 6.4 运行成功的判断方式

- 终端 1 显示 `Uvicorn running on http://127.0.0.1:8765`
- 终端 2 显示 `You can now view your Streamlit app in your browser`
- 浏览器打开 `http://127.0.0.1:8501` 可看到首页 6 个导航页面
- 选择 AMI 会议 → 点击播放 → 页面实时滚动双语字幕
- 切换到"会议库"页面 → 可看到 3 场种子会议 + 其他已入库会议
- 切换到"语音助手" → 文字输入问题 → 收到语音回答

---

## 7. 输出文件与结果说明

### 7.1 C5 主要输出文件

| 输出文件 | 生成模块 | 格式 | 说明 |
|---|---|---|---|
| `outputs/meetings/meeting_index.json` | meeting_library | JSON | 所有会议的索引清单 |
| `outputs/meetings/{id}/meeting.json` | meeting_library | JSON | 单场会议元数据 |
| `outputs/meetings/{id}/product_minutes.json` | generate_realtime_minutes | JSON | 结构化纪要（stage_summaries + speaker_insights + final_minutes） |
| `outputs/meetings/{id}/transcript.json` | 保存会议时回写 | JSON | C2/C3 产出的完整转写（含校正后 speaker 标签） |
| `outputs/meetings/{id}/diarization.json` | run_local_diarization | JSON | 声纹时间线 + 说话人声纹质心 |
| `outputs/meetings/rag/chunks.json` | meeting_rag.build_index | JSON | RAG 文本块（摘要块 + 转写块） |
| `outputs/meetings/rag/vectors.npz` | meeting_rag.build_index | NPZ | 1024 维 L2 归一化向量矩阵 |
| `outputs/tts/tts_request.json` | c5_tts_ali | JSON | 离线 TTS 请求记录 |
| `outputs/tts/tts_result.json` | c5_tts_ali | JSON | 离线 TTS 结果（包含 latency） |
| `outputs/tts/tts_output.wav` | c5_tts_ali | WAV | 离线 TTS 生成的音频文件 |
| `outputs/eval/c3_vs_c4_compare.json` | evaluator | JSON | C3 级联 vs C4 端到端翻译对比指标 |

### 7.2 纪要 JSON 结构示例

```json
{
  "stage_summaries": [
    {
      "window": "0-120s",
      "title": "市场定位讨论",
      "summary": "团队讨论了遥控器产品的目标市场...",
      "actions": ["确定目标用户画像"],
      "risks": ["老年用户接受度不明"],
      "open_questions": ["是否需要触屏"]
    }
  ],
  "speaker_insights": [
    {
      "speaker": "参会者A",
      "stance_or_role": "产品设计倡导者",
      "main_points": ["遥控器应面向全年龄段", "按键数从12减到8"],
      "agreements": ["同意成本控制在30元以内"],
      "disagreements_or_concerns": ["担心简化过度影响功能"]
    }
  ],
  "final_minutes": {
    "one_sentence_summary": "团队讨论了遥控器原型机的功能需求和成本目标",
    "key_decisions": ["按键数从12个减少到8个", "面向老年用户"],
    "action_items": ["下周前做出按键方案原型", "确认供应商报价"],
    "risks": ["成本可能超预算"],
    "open_questions": ["是否需要触屏功能"],
    "conclusion": "下周三前完成原型，周五评审"
  },
  "success": true,
  "minutes_model": "qwen-plus"
}
```

---

## 8. 协作实现说明

### 8.1 模块间接口约定

C5 与上游模块（C2 ASR、C3/C4 翻译）通过 JSON 文件交互。C5 的 `generate_realtime_minutes()` 使用多级 fallback 取值模式兼容不同模块的字段名差异：

```python
# 兼容 C2/C3/C4 不同版本的字段名
"time": row.get("time") or row.get("start") or idx,
"speaker": row.get("speaker") or "参会者识别中",
"source": row.get("corrected") or row.get("source") or "",
"translation": row.get("english") or row.get("translation") or "",
```

### 8.2 C5 提供的 HTTP API

C5 通过 FastAPI 暴露 4 个 REST 端点供前端调用：

| 接口 | 方法 | 路径 | 说明 |
|------|------|------|------|
| 助手对话 | POST | `/api/assistant/chat` | 接收问题，返回文字 + base64 音频 |
| 保存会议 | POST | `/api/realtime/meeting/save` | 保存录音、生成纪要、入库 |
| 阶段摘要 | POST | `/api/realtime/meeting/stage-summary` | 每 120 秒自动调用 |
| RAG 检索 | GET | `/api/meetings/search?q=...` | 搜索历史会议片段 |

### 8.3 常见联调问题与解决

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| C5 拿到空的 transcript | C2 ASR 尚未完成或 WebSocket 断连 | 前端等待 `asr_complete` 事件后再调 save 接口 |
| 纪要生成时 speaker 显示"参会者识别中" | 声纹模型尚未完成离线校正 | 保存时先跑 `run_local_diarization()` 再生成纪要 |
| RAG 检索返回空 | meeting_index.json 中无 completed 状态会议 | 运行 `ensure_meeting_library()` 初始化种子数据 |
| TTS 调用失败 | `.env` 中 API Key 无效或未配置 | 检查 `DASHSCOPE_API_KEY`；TTS 失败不阻塞文字答案返回 |
| 前端助手面板无响应 | `REALTIME_PORT` 与后端端口不一致 | 确认后端监听 `8765`，前端 `app.py` 中 `REALTIME_PORT=8765` |

### 8.4 前后端集成与事件协议

**技术选型**：前端采用 **Streamlit** + 内嵌 HTML/JS，实时服务采用 **FastAPI + WebSocket**。

**数据传输协议**：

| 数据类型 | 传输方式 | 格式 |
|----------|----------|------|
| 音频 | WebSocket 二进制帧 | 16-bit PCM little-endian，16000Hz 单声道 |
| 字幕/纠错/翻译 | WebSocket JSON 事件 | `{type, text, speaker, turn_id, route, timestamp}` |
| 说话人更新 | WebSocket JSON 事件 | `{type: "speaker_update", speaker, turn_id, status, confidence}` |
| 结束信号 | WebSocket JSON 事件 | `{type: "stop"}` → 触发最终声纹校正 + 会议落盘 |
| 助手对话 | HTTP REST | `POST /api/assistant/chat` → JSON response |

**事件归一化**：C3（级联翻译）和 C4（端到端翻译）的模型原始事件格式不同，后端统一转换为 `partial`/`final` 事件对，前端使用同一套字幕卡片组件展示，不区分翻译路线。

**跨模块公共字段**：

| 字段 | 说明 |
|------|------|
| `turn_id` | 说话轮次 ID，C2 声纹模块分配，C3/C4/C5 继承 |
| `speaker` | 当前说话人标签（`参会者A` / `参会者识别中`） |
| `route` | 翻译路线标识（`cascade` / `e2e`） |
| `timestamp` | 音频帧时间戳（秒），用于对齐字幕与音频播放进度 |

**集成中解决的关键问题**：

1. **音频播放与模型输入同步**：浏览器 `AudioContext` 采样率与 ASR 要求的 16000Hz 不一致，前端 `downsample()` 函数做降采样后通过 WebSocket 发送
2. **异步结果归属**：ASR/翻译/声纹三个异步流的结果通过 `turn_id` 关联到同一句字幕，`sender()` 协程统一收集事件后推送给前端
3. **WebSocket 生命周期**：用户断开或停止录音时，`stop_event` 信号触发 ASR 停止、翻译 session finish、声纹最终校正、会议自动保存
4. **历史会议回写**：已保存会议可重新运行声纹校正（`POST /api/meetings/{id}/diarize`），校正结果回写 transcript 和 meeting.json

---

## 9. 已知问题与改进方向

| 问题 | 当前原因 | 可能改进 |
|---|---|---|
| 回答 TTS 同步阻塞总响应时间 | LLM 返回后才开始合成答案语音（约 3-10 秒），用户等待总时间长 | 改为流式返回：先返回文字+过渡语音，答案音频通过 SSE 异步推送 |
| RAG 仅支持中文 bigram 分词 | 中文没有天然分词边界，当前用字符 n-gram | 引入 jieba 分词或使用 bge-m3 等支持中文的 embedding 模型 |
| 会议库种子数据仅 AI 会议 | `_seed_meetings()` 将 ES2004a 的人工拆成 3 段模拟历史会议 | 真实部署时替换为实际会议数据 |
| TTS 仅支持 Cherry 女声 | 硬编码单一音色 | 前端添加音色选择器，后端支持多音色参数 |
| 联网搜索用 DuckDuckGo HTML 解析 | 免费无 API Key，但解析正则脆弱 | 改用正经搜索 API（Bing/SerpAPI） |
| 声纹校正需会后离线跑 | 实时在线聚类精度有限 | 提高 CAM++ 窗口质量或引入更轻量的在线聚类算法 |

---

## 10. C5 离线 TTS 模块 — CosyVoice3 音色克隆

### 10.1 模块概述

离线 TTS 模块使用 Fun-CosyVoice3-0.5B 模型，将 C4 端到端翻译或 C3 级联翻译输出的文本合成为自然语音，支持零样本音色克隆。与 server.py 中的 qwen-tts API 方案（第 9 节）不同，本模块为本地 GPU 推理方案，无需联网。

**基础要求（对应 PDF 第 9.5 节）：**

| 序号 | 要求 | 状态 | 实现方式 |
|:---:|---|---|---|
| 1 | 跑通一个 TTS 模型或接口 | ✅ | Fun-CosyVoice3-0.5B（llm.pt + flow.pt + hift.pt + campplus.onnx） |
| 2 | 支持批量语音生成 | ✅ | JSON 驱动，`--limit` 控制，遍历全量逐条生成 |
| 3 | 测试至少两种语言 | ✅ | 中文 instruct2 + 英文 cross_lingual，自动生成双语测试音频 |
| 4 | 接受 C3/C4 文本，级联 S2S | ✅ | `--text_field` 适配两套字段名，`id` 命名保留追溯链 |
| 5 | 统计推理耗时 | ✅ | 每条记录 latency/RTF/duration，汇总至 c5_results.json |

**进阶要求（对应 PDF 第 9.6 节）：**

| 序号 | 要求 | 状态 | 实现方式 |
|:---:|---|---|---|
| 1 | 支持多个语种 | ✅ | 中文 instruct2 + 英文 cross_lingual，自动语言检测路由 |
| 2 | 支持语速控制 | ✅ | `--speed` 参数传入 `inference_instruct2(speed=speed)`，0.8=减速，1.5=加速 |
| 3 | 支持音色控制 | ✅ | 零样本克隆（每条用 `item.audio` 源音频）+ `--speaker_prompt` 统一音色覆盖 |
| 4 | 支持情感控制 | ⚠️ | CREMA-D 情感数据集作为源音频可间接影响韵律，未实现显式情感标签注入 |
| 5 | 支持说话人控制 | ✅ | `--speaker_prompt` 指定任意参考音频；`--no_voice_clone` 回退默认音色 |
| 6 | 比较不同 TTS 模型 | ✅ | Web 端 qwen-tts API + 离线 CosyVoice3，两种方案并行对比 |
| 7 | 流式 TTS 或分句合成 | ⚠️ | 当前 `stream=False`；CosyVoice3 框架原生支持 streaming，已列入改进方向 |

**当前限制：**

| 限制 | 说明 |
|---|---|
| RTF 约 4-8x | vGPU 共享环境，LLM 自回归生成瓶颈 |
| 英文发音带中式口音 | CosyVoice3 训练数据以中文为主 |
| 不支持流式 | 当前 `stream=False`，计划后续启用 `stream=True` |

### 10.2 模块流程与结构

#### 10.2.1 模块边界

| 阶段 | 入口 | 主要职责 | 输入 | 输出 |
|---|---|---|---|---|
| 模型加载 | `load_model()` | 加载 Fun-CosyVoice3-0.5B 权重 | 模型目录路径 | AutoModel 实例 |
| 数据读取 | `load_input()` | 读取 C3/C4 JSON | JSON 文件路径 | 样本列表 |
| 中文合成 | `synthesize_voice_clone()` | instruct2 音色克隆 | 中文文本 + 源音频 | torch.Tensor |
| 英文合成 | `synthesize_english()` | cross_lingual 跨语言 | 英文文本 + 参考音频 | torch.Tensor |
| 兜底方案 | `synthesize_default()` | zero_shot 默认音色 | 文本 + 默认音频 | torch.Tensor |
| 后处理 | `add_trailing_silence()` | 末尾拼接 1s 静音 | torch.Tensor | torch.Tensor |
| 统计输出 | `main()` 汇总 | 生成统计报告 | 单条统计列表 | c5_results.json |

#### 10.2.2 系统架构

```
C4/C3 JSON 输入
  │
  ├── end2end_translation / translation（合成文本）
  └── audio（源音频路径 → 音色参考）
        │
        ▼
  c5_tts.py
  ├── load_input()              读取 JSON
  ├── is_english_text()         语言检测（ASCII 占比 > 50% → 英文）
  ├── synthesize_voice_clone()  中文: instruct2 + 源音频声纹
  ├── synthesize_english()      英文: cross_lingual + <|en|> 标签
  ├── add_trailing_silence()    末尾 1s 静音
  └── save_wav()                保存 24000Hz WAV
        │
        ▼
  outputs/
  ├── wavs/<id>.wav             以 C4 原始 id 命名
  ├── wavs/lang_zh.wav          中文测试
  ├── wavs/lang_en.wav          英文测试
  └── c5_results.json           统计汇总
```

#### 10.2.3 一次完整合成流程

1. 加载模型：AutoModel 加载 llm.pt + flow.pt + hift.pt + campplus.onnx（约 4GB 显存）
2. 读取输入：从 C4 JSON 中取出 `end2end_translation`（中文译文）和 `audio`（源音频路径）
3. 语言判断：统计 ASCII 字母占比，超过 50% 走英文路径，否则走中文路径
4. 中文合成：源音频 → campplus 提取 512 维声纹向量 → instruct2 注入 LLM → 中文语音
5. 英文合成：默认 prompt → cross_lingual + `<|en|>` 标签 → 英文语音
6. 后处理：末尾拼接 1s 静音消除截断感，保存为 24000Hz WAV
7. 统计：记录合成耗时、音频时长、RTF，写入 c5_results.json

### 10.3 模型说明

| 项目 | 内容 |
|---|---|
| 使用模型 | Fun-CosyVoice3-0.5B |
| 模型来源 | 阿里通义实验室开源（ModelScope） |
| 项目内路径 | `/root/siton-tmp/multimodal/Fun-CosyVoice3-0.5B/` |
| 是否需要 GPU | 需要（vGPU 可用） |
| 是否需要联网 | 首次加载时需下载 wetext tokenizer（自动缓存至 `~/.cache/modelscope`） |

核心子模块：

| 文件 | 大小 | 作用 |
|------|------|------|
| `llm.pt` | 2.5GB | Qwen-based LLM：文本 → 语音 token |
| `flow.pt` | ~300MB | Flow Matching：token → 梅尔频谱 |
| `hift.pt` | ~150MB | HiFiGAN 声码器：频谱 → 波形 |
| `campplus.onnx` | ~30MB | 说话人特征提取：音频 → 512 维声纹向量 |

### 10.4 环境安装

```bash
# 环境已预装在 cosyvoice conda 环境中
conda activate cosyvoice

# 验证模型文件存在
ls /root/siton-tmp/multimodal/Fun-CosyVoice3-0.5B/llm.pt
ls /root/siton-tmp/multimodal/Fun-CosyVoice3-0.5B/campplus.onnx
```

常见环境问题：

- **wetext tokenizer 下载慢**：首次运行时自动从 modelscope.cn 下载，已缓存至 `~/.cache/modelscope`
- **vGPU cuCtxSetCurrent 错误**：退出时的 harmless warning，不影响合成质量
- **GPU 显存不足**：加载模型约需 4GB 显存，vGPU 环境满足要求

### 10.5 运行命令

```bash
conda activate cosyvoice
cd /root/siton-tmp/multimodal/C5_TTS/code

# Demo 1: 中文批量合成（3 条 + 双语测试）
python c5_tts.py \
  --input /root/siton-tmp/multimodal/c4/outputs/c4_results.json \
  --limit 3 \
  --outdir ../outputs

# Demo 2: 英文批量合成（10 条）
python c5_tts.py \
  --input /root/siton-tmp/multimodal/C5_TTS/test_en.json \
  --limit 10 \
  --outdir ../outputs_en

# Demo 3: 语速控制（1.3 倍速）
python c5_tts.py \
  --input /root/siton-tmp/multimodal/c4/outputs/c4_results.json \
  --limit 3 --speed 1.3 \
  --outdir ../outputs_speed

# Demo 4: 音色控制（统一女声）
python c5_tts.py \
  --input /root/siton-tmp/multimodal/c4/outputs/c4_results.json \
  --limit 3 \
  --speaker_prompt ../asset/zero_shot_prompt.wav \
  --outdir ../outputs_speaker
```

### 10.6 关键参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--input` | 输入 JSON 路径（必填），支持 C3/C4 两种格式 | - |
| `--text_field` | JSON 中合成文本的字段名 | `end2end_translation`（C4）/ `translation`（C3） |
| `--model_dir` | CosyVoice3 模型权重目录 | `/root/siton-tmp/multimodal/Fun-CosyVoice3-0.5B` |
| `--outdir` | 输出目录 | `outputs` |
| `--limit` | 限制处理条数（0 = 全量） | 0 |
| `--speed` | 语速缩放因子（0.8 = 慢, 1.5 = 快） | 1.0 |
| `--speaker_prompt` | 统一音色参考音频（覆盖各条源音频） | 无 |
| `--trailing_silence` | 末尾静音秒数 | 1.0 |
| `--no_voice_clone` | 关闭音色克隆 | 关闭（默认开启） |

### 10.7 输出文件

| 输出文件 | 格式 | 说明 |
|---|---|---|
| `wavs/<id>.wav` | WAV 24000Hz | 以 C4 原始 id 命名，可追溯 |
| `wavs/lang_zh.wav` | WAV 24000Hz | 中文双语测试音频 |
| `wavs/lang_en.wav` | WAV 24000Hz | 英文双语测试音频 |
| `c5_results.json` | JSON | 每条 latency/RTF/duration + 整体统计 |

`c5_results.json` 格式：

```json
{
  "module": "C5_TTS",
  "version": "voice-clone+bilingual",
  "model": "Fun-CosyVoice3-0.5B",
  "total_samples": 20,
  "ok": 20, "failed": 0, "skipped": 0,
  "voice_cloned": 20,
  "total_time_sec": 226.9,
  "total_audio_duration_sec": 47.4,
  "total_synth_latency_sec": 206.8,
  "sample_rate": 24000,
  "results": [
    {
      "id": "cremad_1001_dfa_ang_xx",
      "status": "ok",
      "input_text": "别忘了外套!",
      "audio_duration_sec": 2.60,
      "speech_duration_sec": 1.60,
      "synth_latency_sec": 13.374,
      "rtf": 5.14,
      "voice_cloned": true
    }
  ]
}
```

### 10.8 运行成功的判断方式

- 终端显示 `C5 TTS — Done` 且无报错
- `outputs/wavs/` 目录下生成对应数量的 wav 文件
- `outputs/c5_results.json` 存在且 `"status": "ok"` 数量正确
- 播放音频可正常听到中文/英文语音

### 10.9 协作实现说明

- **接口约定**：C4 的输出 JSON 文件作为 C5 的输入，通过 `end2end_translation` 字段传递文本，通过 `audio` 字段传递源音频路径。C5 输出文件名与 C4 输入的 `id` 字段一致，保证端到端追溯。
- **多模块适配**：通过 `--text_field` 参数切换不同翻译模块的文本字段名（C4: `end2end_translation`，C3: `translation`），实现与不同上游模块的松耦合连接。
- **输出标准**：C5 的输出目录结构和 JSON 格式在组内已约定统一，其他模块可复用。

### 10.10 已知问题与改进方向

| 问题 | 原因 | 改进方向 |
|---|---|---|
| RTF 较高（4-8x） | vGPU 共享环境，LLM 自回归生成瓶颈 | 部署至独立 GPU（如 A100），可用 TensorRT 加速 |
| 英文发音有中式口音 | CosyVoice3 训练数据以中文为主 | 接入专门英文 TTS 模型或使用端到端英中语音翻译模型 |
| 部分高唤醒度情感（angry/fear）合成偏短 | instruct2 模式在这些情感下偶发提前终止 | 调整模型内部 EOS 阈值或增加最小生成长度参数 |
| 不支持流式实时输出 | 当前为非流式模式 | CosyVoice3 框架支持 streaming，可在 `--stream` 模式下进一步开发 |

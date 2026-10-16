# DeepSeek / Qwen 视觉切分 smoke harness

`smoke_harness.py` 从 ABC-130K 读取指定源 episode，按三相机真实 MP4 帧数定位视频，生成带源帧号的联系图，调用 `deepseek-flash` 或 `qwen3.6-plus` 标注局部动作，导出 1–12 秒短 episode 和审阅 HTML。程序不把 API key 写入请求记录或结果。当前是**候选标注试点**，所有片段均待人工审核，`ready_for_training=false`。采样间隔可用 `--sample-stride-frames` 控制：30 帧 = 1 秒，15 帧 = 0.5 秒；每次模型请求的最大窗口由 `--max-window-frames` 控制，默认 300 帧（10 秒）。

运行环境：Python 3.12、`ffmpeg`（通过 `imageio-ffmpeg` 提供），依赖见 `requirements.txt`。

```bash
pip3 install -r /home/geek/hanwen.cui/code/harness-for-video-split/code/requirements.txt
export DEEPSEEK_API_KEY='你的 key'
python /home/geek/hanwen.cui/code/harness-for-video-split/code/smoke_harness.py \
  --provider deepseek --episodes 78 77 --sample-stride-frames 15 --max-window-frames 300 \
  --run-id my_deepseek_smoke_01
```

Qwen 使用 `DASHSCOPE_API_KEY` 和官方北京地域公共兼容地址。该 key 当前已实测可调用 `qwen3.6-plus` 并输入图像：

```bash
export DASHSCOPE_API_KEY='你的 key'
python /home/geek/hanwen.cui/code/harness-for-video-split/code/smoke_harness.py \
  --provider qwen --episodes 78 --sample-stride-frames 15 --max-window-frames 300 \
  --run-id my_qwen_smoke_01
```

源数据只读；每次运行写入 `result/runs/<run_id>/`。同名 run 会拒绝覆盖。主要产物：`preview/index.html`、`preview/source_ep_*.html`、`segments/{train,val}/episode_*`、`index.jsonl`、`meta/tasks.parquet`、`qc/report.json`。`manifest.json` 中保存源帧区间、三个 MP4 的文件内起点、子任务文本和审核状态。每个短 episode 的 `data.parquet` 有重新编号的 `episode_index/frame_index/timestamp/index/task_index`，而 `action` 与 `observation.state` 保留源数值。

DeepSeek 请求将 `thinking` 设为 `disabled`：试运行 `enabled` 时，长图像分析把 5,000 个 completion token 全用在 reasoning，最终 `content` 为空；Qwen 请求使用 `enable_thinking=false`。程序保留每个窗口的原始模型文本、响应 ID、token 使用量和 prompt hash。模型边界若与邻段相差不超过 30 帧，允许吸附并标记待审核；大于 30 帧直接失败。片段的三路 MP4 解码帧数必须等于 parquet 行数；独立验收还比对源与导出的 action/state、文本映射和源帧全覆盖。

本版尚未实现全量 100 条批处理的断点续跑、人工审核界面上的状态写回，也没有构建标准 `derived_lerobot_v3/` 数据集。`meta/tasks.parquet` 和逐帧 `task_index` 已按子任务文本重新映射，但当前目录仍是便于审阅的单段文件夹格式。人工审核和训练读取器适配通过前不要将 smoke 结果作为正式训练集。

采样加密的对照结果见 `result/sampling_comparison_20261015.md`。更多采样帧并未稳定提高标签质量：长窗口出现漏分动作，短窗口出现“plastic bag”误认和过细切分；这属于当前视觉模型输出的语义问题，机器帧数检查无法发现。

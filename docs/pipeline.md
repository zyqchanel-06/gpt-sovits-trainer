# 流水线详解与排错

## 为什么要分六步

不是刻意切碎，而是**每步的产物是下一步的输入**，且失败概率和耗时差两个数量级：

| 步骤 | 典型耗时（10 秒素材） | 失败概率 | 失败了怎么办 |
| --- | --- | --- | --- |
| slice | 秒级 | 低 | 重新切片 |
| asr | 10 秒~1 分钟 | 中（模型没下载） | 手动给 `.list` |
| prep | 10~30 秒 | 中（路径/版本不对） | 看日志改参数 |
| s2 | 3~10 分钟 | 中（OOM） | 调小 batch |
| s1 | 3~10 分钟 | 中（OOM） | 调小 batch |
| register | 毫秒级 | 低 | 检查权重目录 |

分步之后，OOM 在 s2 挂掉，你只需要调 batch 然后 `--from s2`，
前面的切片/标注/特征全部复用。

## 每步的坑

### slice（切片）

- **阈值太严**会把一句话切成两半，**太松**会切出 20 秒的长段（训练效果差）
- 默认 `threshold=-34`（dBFS）、`min_length=4000ms`、`max_length=10000ms`
- 如果是纯人声、录音质量好，可以把 `--max-length` 提到 12000

### asr（标注）

- 自动识别对专有名词必错，跑完**一定人工校对**
- `.list` 格式：`<wav路径>|<说话人>|<语言>|<文本>`
- 如果已经有现成标注，`init --list xxx.list` 直接跳过这步

### prep（特征提取）

三个脚本靠**环境变量**传参（`inp_text` / `inp_wav_dir` / `opt_dir` / `version` ...），这是上游约定。

常见失败：

| 报错 | 原因 |
| --- | --- |
| `找不到 bert 模型` | `chinese-roberta-wwm-ext-large` 没下载 |
| `CUDA out of memory` | 这步也吃显存，可以减并发 |
| `KeyError: version` | `version` 和权重版本不匹配 |

### s2 / s1（训练）

- **OOM 是第一大原因**：`--s2-batch-size 2` 通常能救
- `fp16` 关掉（`is_half: false`）会让 batch 自动减半，也更稳但更慢
- 训练日志里能看 epoch/step 进度，`status` 命令会摘出来

### register（登记）

按文件名扫产物：

- SoVITS：`<voice>_e<epoch>_s<step>.pth`
- GPT：`<voice>-e<epoch>.ckpt`

**名字里带空格或特殊字符会扫不到**，所以 `init` 时限制了音色名只允许字母数字下划线连字符。

## 换 epoch 档位

训练默认每 4 轮存一个档（e4 / e8）。e8 通常更稳，但过拟合时 e4 反而更像：

```bash
sovits-train export --voice alice --epoch 4     # 改成用 e4
sovits-train status --voice alice               # 确认
```

## 从中间续跑

```bash
sovits-train reset --voice alice --stage s2     # 只重置 s2 及之后
sovits-train run --voice alice                  # 从 s2 继续
```

或者直接：

```bash
sovits-train run --voice alice --from s2
```

## 日志在哪

```
tasks/<voice>/logs/
├─ slice.log
├─ asr.log
├─ prep-text.log
├─ prep-hubert.log
├─ prep-semantic.log
├─ s2.log
└─ s1.log
```

每份日志第一行是完整命令行，排查时可以直接复制出来手动重跑。

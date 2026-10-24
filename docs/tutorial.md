# 保姆级教程：从零让你的 AI 助手开口说话

这篇教程写给「想让自己的 AI 用上自定义声音，但没搞过语音合成」的人。

全程命令都可以直接复制。鲸鲸把踩过的坑都写进去了——**遇到报错先翻最后一节**，八成在那儿。

---

## 你会得到什么

做完之后：

- 你有一段属于自己的音色模型（用几十条语音训练出来的）
- 有一个本地服务，把文字变成那个声音
- 把它接到你的 AI 上，它回消息就能带语音了

不是那种在线试听 demo——全部跑在你自己的机器上，想合成多长都行，不联网也能用。

---

## 一、开始之前

### 硬件

| 项目 | 要求 | 说明 |
| --- | --- | --- |
| 显卡 | 8G 显存起步 | 笔记本 RTX 5060（8G）实测可跑训练 + 推理 |
| 内存 | 16G 够用 | 训练时会吃几个 G |
| 硬盘 | 预留 30G | 模型和依赖加起来差不多这个量 |


**没有独立显卡能不能玩？** 能，但只能推理（不能训练）。推理用 CPU 也行，就是慢——一句话可能要等十几秒。训练必须要显卡。

**显存不够怎么办？** 把训练的 batch size 调到 1，能跑但慢。或者用云 GPU（按小时租）训练，只把训练好的模型下回本地推理。

### 软件

| 软件 | 版本 | 检查命令 |
| --- | --- | --- |
| Python | 3.9 ~ 3.11 | `python --version` |
| Git | 任意 | `git --version` |
| Node.js | 18 以上 | `node --version` |
| ffmpeg | 任意 | `ffmpeg -version` |


> ffmpeg 最容易漏。它是一个处理音频的命令行工具，很多语音项目都要用。Windows 上可以用 `winget install Gyan.FFmpeg` 装，Mac 用 `brew install ffmpeg`。

### 时间

- 搭环境：首次 30~60 分钟（主要耗在下载依赖）
- 准备素材：看你有多少录音，切分是自动的
- 训练：20 分钟起步（数据集越大越久）
- 接进 AI：10 分钟

---

## 二、装 GPT-SoVITS

GPT-SoVITS 是目前中文语音克隆效果最好的开源项目之一，而且**用很少的数据就能出效果**。

```bash
# 1. 找个目录，克隆下来
cd ~/projects   # 换成你自己的目录
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS

# 2. 装依赖（用虚拟环境更干净）
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux / Mac:
source venv/bin/activate

# 3. 装依赖包
pip install -r requirements.txt
```

**国内网络慢的话**，加个镜像：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 下载预训练模型

GPT-SoVITS 需要几个预训练模型才能工作，放在 `GPT_SoVITS/pretrained_models/` 目录下：


- `chinese-hubert-base` —— 语音特征提取
- `chinese-roberta-wwm-ext-large` —— 文本特征提取
- `gsv-v2final-pretrained/` —— 训练用的底模（s2G / s2D / s1）

官方文档里有网盘和 HuggingFace 的下载地址，照着放进去就行。**这一步别偷懒**，模型放错位置后面会报一堆奇怪的错。

放好之后可以验证一下：
```bash
# 列出目录，应该能看到这几个文件夹
ls GPT_SoVITS/pretrained_models/
```

---

## 三、准备语音素材

这一步决定了最终效果的上限。**素材烂，后面怎么调都救不回来。**

### 素材从哪来


1. **自己录**：效果最好，也最省事（不用担心版权）
2. **公开的授权语音**：比如自己拍的视频里的旁白
3. **游戏 / 动画里的语音**：技术学习可以，但**别商用、别传播**，版权有风险

### 什么样的素材算「干净」

| 要 | 不要 |
| --- | --- |
| 单人说话，没有背景音乐 | 有 BGM、有音效 |
| 环境安静 | 有回声、有键盘声、有车流声 |
| 音量稳定 | 忽大忽小 |
| 情感自然的日常语气 | 朗读腔、播音腔（除非你想要那种声音） |


### 要准备多少

**10 分钟的高质量素材，效果就能听**。要多像的话准备 30 分钟以上。

每条建议 3~10 秒。太短了模型学不到东西，太长了切分后会断裂。

### 怎么准备


1. 把素材整理成 wav 格式，**采样率 32kHz**、单声道
2. 放在一个文件夹里
3. 剩下的交给后面的工具自动切分

> 如果你手上是一整段长录音（比如 30 分钟的播客），不用手动切——训练流程的第一步就是自动切片。

---

## 四、训练你的音色

官方流程要手动跑六个脚本，中间错一步就得从头检查。鲸鲸把整个流程包成了一个命令，**挂了能从断掉的地方接着跑**。

### 装训练编排器

```bash
git clone https://github.com/zyqchanel-06/gpt-sovits-trainer.git
cd gpt-sovits-trainer
```

### 先体检

跑之前先确认环境没问题：

```bash
# 把 <GPT-SoVITS 路径> 换成你实际的路径
python -m sovits_trainer \
  --gpt-sovits-root "<GPT-SoVITS 路径>" \
  --python "<装了 torch 的那个 python>" \
  doctor
```

它会检查：GPT-SoVITS 的脚本在不在、依赖全不全、显卡够不够、有没有别的程序在抢显存。

看到这些就说明可以了：
```text
✓ GPT-SoVITS： <路径>
✓ 训练依赖齐备（numpy, torch, scipy, librosa, yaml）
✓ GPU0 NVIDIA ... 
```

### 创建任务并跑训练

```bash
# 1. 创建任务（--audio 指向你的素材文件夹）
python -m sovits_trainer --gpt-sovits-root "<路径>" init \
  --voice myvoice \
  --audio "D:/我的素材"

# 2. 开跑（六个步骤自动接力）
python -m sovits_trainer --gpt-sovits-root "<路径>" run --voice myvoice
```

跑完之后你会看到：
```text
任务: myvoice
  ✓ slice      音频切片           wav
  ✓ asr        语音识别标注        标注文件
  ✓ prep       特征提取           6-name2semantic-0.tsv
  ✓ s2         训练 SoVITS
  ✓ s1         训练 GPT
  ✓ register   登记产物           voices.json

产物：
  SoVITS: myvoice_e8_s200.pth
  GPT   : myvoice-e8.ckpt
```

### 六个步骤分别在干什么

| 步骤 | 作用 | 大概耗时 |
| --- | --- | --- |
| slice | 把长录音切成 3~10 秒的片段 | 几秒 |
| asr | 语音识别，给每段配上文字 | 几分钟（看数据量） |
| prep | 提取文本/语音/语义三种特征 | 几分钟 |
| s2 | 训练声音模型（决定音色像不像） | 10 分钟起 |
| s1 | 训练语言模型（决定语气自然不自然） | 10 分钟起 |
| register | 挑出效果最好的版本，写好交接单 | 几秒 |


### 中途挂了怎么办

```bash
# 从失败的步骤继续，前面的不会重做
python -m sovits_trainer --gpt-sovits-root "<路径>" run \
  --voice myvoice --from s1
```

### 改了素材要重跑

**这里有个坑**：上游的预处理脚本是「结果文件存在就跳过」。所以改了素材之后必须清掉旧产物：

```bash
python -m sovits_trainer --gpt-sovits-root "<路径>" reset \
  --voice myvoice --stage prep --purge
```

不加 `--purge` 的话，它会跳过处理、沿用旧数据，你会以为训练了新素材其实没有。

---

## 五、让声音能被调用

训练出模型只是第一步，还得有个服务把「文字」变成「声音」。

GPT-SoVITS 自带一个服务（`api_v2.py`），但它有几个问题：接口格式不标准、没有跨域头、只返回 wav。结果就是——**标准客户端连不上**。

鲸鲸写了层网关来解决这个，不用改 GPT-SoVITS 本身的任何文件。

### 启动 GPT-SoVITS 的服务

```bash
cd GPT-SoVITS
python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
```

看到它开始加载模型、最后监听 9880 就成功了（首次加载要 30~60 秒，别急）。

### 装网关闭并配置

```bash
git clone https://github.com/zyqchanel-06/gpt-sovits-gateway.git
cd gpt-sovits-gateway
cp config.example.json config.json
```

编辑 `config.json`，填上你的音色：

```json
{
  "host": "127.0.0.1",
  "port": 9881,
  "apiBase": "http://127.0.0.1:9880",
  "defaultVoice": "myvoice",
  "voices": [
    {
      "id": "myvoice",
      "name": "我的音色",
      "lang": "zh",
      "refAudio": "参考音频的路径.wav",
      "promptText": "参考音频里说的那句话",
      "promptLang": "zh",
      "gpt": "训练产出的 .ckpt 路径",
      "sovits": "训练产出的 .pth 路径"
    }
  ]
}
```

> `refAudio` 和 `promptText` 是「参考音频」——GPT-SoVITS 需要一小段声音作为语气基准。用训练素材里的任意一条就行，文字要和音频内容一致。

**路径可以直接从训练产出的 `voices.json` 里抄**，那个文件就是给这一步用的。

### 启动并验证

```bash
npx gpt-sovits-gateway serve --config config.json

# 另开一个终端，试试能不能合成
curl -X POST http://127.0.0.1:9881/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"你好，我现在能说话了","voice":"myvoice","response_format":"mp3"}' \
  --output test.mp3
```

打开 `test.mp3` 听一下。**能听到你的音色说话，就成了。**

---

## 六、接进你的 AI

这一步取决于你的 AI 是怎么跑的，但思路都一样：**让它把回复发给网关，拿到音频，再发出去**。

### 核心就一个 HTTP 请求

```javascript
const res = await fetch("http://127.0.0.1:9881/v1/audio/speech", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    input: "要念出来的文字",
    voice: "myvoice",
    response_format: "mp3"
  })
});
const audio = await res.arrayBuffer();   // 这就是音频数据
```

拿到 `audio` 之后怎么发出去，看你的链路：

| 你的 AI 跑在哪 | 怎么发音频 |
| --- | --- |
| QQ 机器人 | 用你那个框架的语音消息接口（NapCat / OneBot 之类的） |
| 微信 | 看具体方案，多数是先落地成文件再发 |
| 网页应用 | 直接当 Blob 播放，或存成文件让浏览器下载 |
| 命令行 | 写成文件，用系统播放器打开 |


### 一个完整的例子（QQ 机器人）

```javascript
// 收到消息 → 生成回复文字 → 合成语音 → 发语音
async function replyWithVoice(text) {
  const res = await fetch("http://127.0.0.1:9881/v1/audio/speech", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ input: text, voice: "myvoice", response_format: "mp3" })
  });
  const audio = Buffer.from(await res.arrayBuffer());

  const file = `./voice-out/${Date.now()}.mp3`;
  fs.writeFileSync(file, audio);

  // 调你框架的发语音接口
  await bot.sendVoice(groupId, file);
}
```

### 两个实践建议


**1. 短消息不用发语音。** 「嗯」「好的」这种发语音反而奇怪。建议超过一定长度才转语音（比如 15 字以上）。

**2. 合成有延迟。** 一句话大概 3~5 秒（取决于文本长度和显卡）。别让用户等太久，可以先发文字再补语音。

---

## 七、常见报错排查

这一节按「你看到什么 → 为什么 → 怎么办」组织。遇到问题先在这儿搜关键词。

### 装环境阶段

**`ModuleNotFoundError: No module named 'numpy'`（或 torch、scipy）**

说明你用的 Python 不是装了依赖的那一个。常见于机器上有多个 Python。

怎么办：确认虚拟环境激活了（命令行前面应该有 `(venv)` 字样），或者用完整路径指定解释器：

```bash
python -m sovits_trainer --python "D:/你的venv/Scripts/python.exe" doctor
```

**`FileNotFoundError: [WinError 2] 系统找不到指定的文件`**

大概率是 ffmpeg 没装或不在 PATH 里。验证一下：

```bash
ffmpeg -version
```

如果报「不是内部或外部命令」，就装一个（见第一章）。装完**要重开终端**才能生效。

**`ModuleNotFoundError: No module named 'gradio'`**

GPT-SoVITS 的某个工具文件里顺手 import 了 gradio。补装：

```bash
pip install "gradio<5" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**`ModuleNotFoundError: No module named 'tools'`**

脚本之间互相 import 找不到路径。如果你用的是鲸鲸的编排器，它会自动处理；手动跑官方脚本的话需要设置 PYTHONPATH，把 GPT-SoVITS 根目录和它的 tools、GPT_SoVITS 子目录都加进去。

### 训练阶段

**`CUDA out of memory`**

显存不够。按顺序试：


1. 关掉其他占显存的程序（浏览器开太多视频也可能占）
2. 把 batch size 降到 1：`--s2-batch-size 1 --s1-batch-size 1`
3. 关掉正在跑的推理服务（它和训练抢显存）
4. 实在不行，用云 GPU 训练，只把模型下回本地

**`ZeroDivisionError: division by zero`（训练 GPT 时报）**

样本太少了——GPT 阶段至少要 **2 条**语音。检查切片结果：

```bash
dir voice-tasks\myvoice\slice\wav     # Windows
ls voice-tasks/myvoice/slice/wav       # Linux / Mac
```

只有一两条的话，说明素材太短或者切分参数不合适——多准备点素材。

**改了素材但训练结果没变**

上游预处理脚本是「产物存在就跳过」，不清理会沿用旧数据。必须加 `--purge`：

```bash
python -m sovits_trainer reset --voice myvoice --stage prep --purge
```

**训练跑了但找不到模型文件**

看这两个地方：

- `GPT_SoVITS/pretrained_models/` —— 训练产出的 .pth 和 .ckpt
- `voice-tasks/myvoice/voices.json` —— 交接单，记录了产物的完整路径

### 推理和接入阶段

**浏览器报 `has been blocked by CORS policy`**

你在直连 GPT-SoVITS 的 9880 端口。那个服务是给本机脚本用的，没处理浏览器跨域。用网关（9881 端口）就没这问题。

**客户端报 `invalid data format`**

音频格式不对。GPT-SoVITS 默认返回 wav，很多前端只认 mp3。网关会自动转（需要 ffmpeg）。

**合成出来的音频被截断**

网关有个简单自检：按文本长度估音频该有多长，明显偏短会自动重合成。还频繁出现的话，可能是文本里有特殊符号导致中断，清理一下文本。

**合成特别慢（一句话要十几秒）**

两种可能：一是用 CPU 推理（检查 torch 是不是 CUDA 版），二是显存不够、模型被换到内存了。

**多个请求一起来就崩**

GPU 同时处理多个合成请求会爆显存。网关默认串行排队，如果你自己写调用逻辑，注意别并发。

---

## 八、进阶方向

| 方向 | 怎么做 |
| --- | --- |
| **多音色** | 在 `config.json` 的 `voices` 数组里加多条，请求时用 `voice` 参数切换 |
| **更像本人** | 增加素材量（30 分钟以上）、提高录音质量、训练轮数调大 |
| **语气更自然** | 素材里多准备不同情绪的样本，别全是平静语气 |
| **降低延迟** | 常说的句子提前合成好做缓存 |
| **多语言** | 训练素材里混入目标语言的语音，参考音频也用对应语言 |

---

## 九、两个项目

教程里用到的两个工具都是开源的：


- **[gpt-sovits-trainer](https://github.com/zyqchanel-06/gpt-sovits-trainer)** —— 训练编排器，六步流水线，断点续跑
- **[gpt-sovits-gateway](https://github.com/zyqchanel-06/gpt-sovits-gateway)** —— 让训练好的声音能被标准客户端调用

遇到教程里没写的问题欢迎开 Issue——**多一个人反馈，下一个人就少踩一次**。

---

## 附：完整命令速查

```bash
# ===== 环境 =====
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS && python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# ===== 训练 =====
git clone https://github.com/zyqchanel-06/gpt-sovits-trainer.git
cd gpt-sovits-trainer
python -m sovits_trainer --gpt-sovits-root "<路径>" doctor
python -m sovits_trainer --gpt-sovits-root "<路径>" init --voice myvoice --audio "素材目录"
python -m sovits_trainer --gpt-sovits-root "<路径>" run --voice myvoice

# ===== 推理 =====
cd GPT-SoVITS && python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
cd gpt-sovits-gateway && npx gpt-sovits-gateway serve --config config.json

# ===== 验证 =====
curl -X POST http://127.0.0.1:9881/v1/audio/speech -H "Content-Type: application/json" \
  -d "{\"input\":\"测试一下\",\"voice\":\"myvoice\",\"response_format\":\"mp3\"}" --output test.mp3
```

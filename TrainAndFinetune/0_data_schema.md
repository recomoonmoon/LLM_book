## 数据处理

数据的核心在于 **`dataset_info.json`**，它定义了所有数据集（本地 & 在线）的信息。

👉 如果你需要使用自定义数据集，请务必在 `./dataset/dataset_info.json` 中添加相应描述。

目前支持两大主要格式：

* **Alpaca 格式**
* **ShareGPT 格式**

此外，还支持 **多模态数据集**（图像/视频/音频）、**偏好数据集**、**KTO 数据集** 等。

---

### Alpaca 格式

Alpaca 格式适合 **指令监督微调**、**预训练**、**偏好训练**、**KTO**、**多模态任务**。

#### 1. 指令监督微调

模型学习输入指令与输出回答的对应关系。
基本字段：

* `instruction`（必填）：人类指令
* `input`（选填）：额外输入
* `output`（必填）：模型回答
* `system`（选填）：系统提示词
* `history`（选填）：多轮对话历史

**示例：**

```json
[
  {
    "instruction": "计算这些物品的总费用。",
    "input": "汽车 - $3000，衣服 - $100，书 - $20。",
    "output": "总费用为 $3120。",
    "history": [
      ["今天会下雨吗？", "不会，下雨机率为0。"],
      ["适合出门吗？", "很适合。"]
    ]
  }
]
```

**对应 `dataset_info.json`：**

```json
"数据集名称": {
  "file_name": "data.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "system": "system",
    "history": "history"
  }
}
```

#### 2. 预训练数据集

```json
[
  {"text": "这是第一段训练文本。"},
  {"text": "这是第二段训练文本。"}
]
```

`dataset_info.json`：

```json
"数据集名称": {
  "file_name": "data.json",
  "columns": {
    "prompt": "text"
  }
}
```

#### 3. 偏好数据集

用于 **奖励建模**、**DPO/ORPO 训练**。

```json
[
  {
    "instruction": "人类指令",
    "chosen": "更优的回答",
    "rejected": "较差的回答"
  }
]
```

#### 4. KTO 数据集

```json
[
  {
    "instruction": "人类指令",
    "output": "模型回答",
    "kto_tag": true
  }
]
```

#### 5. 多模态数据集

支持 **图像 / 视频 / 音频** 输入，需在 JSON 中加入 `images` / `videos` / `audios` 字段。

---

### ShareGPT 格式

ShareGPT 格式相比 Alpaca 更灵活，支持更多角色（`human`、`gpt`、`function_call`、`observation` 等）。

#### 1. 指令监督微调

```json
{
  "conversations": [
    {"from": "human", "value": "你好"},
    {"from": "gpt", "value": "你好，很高兴见到你！"}
  ]
}
```

#### 2. 偏好数据集

```json
{
  "conversations": [
    {"from": "human", "value": "老虎吃草吗？"}
  ],
  "chosen": {"from": "gpt", "value": "老虎是食肉动物。"},
  "rejected": {"from": "gpt", "value": "老虎主要吃草。"}
}
```

#### 3. OpenAI 格式

一种特殊情况，`messages` 字段中包含 `system`、`user`、`assistant`：

```json
{
  "messages": [
    {"role": "system", "content": "你是一个助手"},
    {"role": "user", "content": "今天天气怎么样？"},
    {"role": "assistant", "content": "今天是晴天。"}
  ]
}
```

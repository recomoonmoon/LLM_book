 
---

# 🧩 什么是 LCEL？

LCEL（LangChain Expression Language）是一种 **声明式** 的链式组合语言。

* **声明式**：你只需要描述 **要做什么**，而不是 **怎么做**。
* **目标**：更简单、更快、更易扩展地组合 Runnables（可运行单元）。

> 在 LangChain 里，**任何一个 chain（链条）本质上就是一个 Runnable**。

---

# ⚡ LCEL 的优势

1. **并行执行优化**

   * `RunnableParallel` 支持并行运行，降低延迟。
2. **天然支持异步**

   * 所有 LCEL 链条都可以 `await`。
3. **简化流式输出**

   * 可以边执行边输出，缩短 “首 token 时间”（TTFT）。
4. **可观测性好**

   * 每一步都会自动记录到 **LangSmith**，方便调试。
5. **标准化 API**

   * 所有 LCEL 链条都实现了同样的 Runnable 接口：

     * `.invoke()`
     * `.batch()`
     * `.astream()`
     * `.ainvoke()`
6. **可部署**

   * 可以直接用 **LangServe** 部署到生产环境。

---

# 🤔 什么时候用 LCEL？

* **单次 LLM 调用** → 直接调用，不需要 LCEL
* **简单链条**（如 prompt → llm → parser） → 推荐用 LCEL
* **复杂逻辑**（分支、循环、多代理） → 推荐用 **LangGraph**，在每个节点内部用 LCEL

---

# 🛠️ 组合原语（Composition Primitives）

LCEL 提供了两大核心拼装积木：

### 1. RunnableSequence（顺序执行）

多个 Runnable 串起来，前者的输出作为后者的输入。

```python
from langchain_core.runnables import RunnableSequence

chain = RunnableSequence([runnable1, runnable2])
final_output = chain.invoke("input")
```

等价于：

```python
output1 = runnable1.invoke("input")
final_output = runnable2.invoke(output1)
```

---

### 2. RunnableParallel（并行执行）

多个 Runnable 同时运行，输入相同，结果以字典返回。

```python
from langchain_core.runnables import RunnableParallel

chain = RunnableParallel({
    "res1": runnable1,
    "res2": runnable2,
})

final_output = chain.invoke("input")
```

结果：

```python
{
  "res1": runnable1.invoke("input"),
  "res2": runnable2.invoke("input"),
}
```

👉 实际执行时是并行的，速度更快。

---

# ✨ 组合语法糖

为了让代码更简洁，LCEL 提供了简化写法。

### 1. 管道符 `|`

```python
chain = runnable1 | runnable2
```

等价于：

```python
chain = RunnableSequence([runnable1, runnable2])
```

### 2. `.pipe()` 方法

```python
chain = runnable1.pipe(runnable2)
```

等价于 `|`，避免一些人不喜欢运算符重载。

### 3. 字典自动转 RunnableParallel

```python
mapping = {
    "key1": runnable1,
    "key2": runnable2,
}

chain = mapping | runnable3
```

会自动转成：

```python
chain = RunnableSequence([
    RunnableParallel(mapping),
    runnable3
])
```

---

# 📌 小例子：Prompt + LLM + Parser

这是 LCEL 的典型用法。

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 定义三个 Runnable
prompt = ChatPromptTemplate.from_messages([
    ("human", "用一句话总结: {topic}")
])
llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# 用 LCEL 组合
chain = prompt | llm | parser

# 执行
result = chain.invoke({"topic": "LangChain Expression Language"})
print(result)
```

执行流程：

1. prompt 根据输入生成 prompt 文本
2. llm 调用大模型
3. parser 把输出转成字符串

---

# 📌 小例子：并行调用两个 LLM

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel

llm1 = ChatOpenAI(model="gpt-3.5-turbo")
llm2 = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages([
    ("human", "回答这个问题: {question}")
])
parser = StrOutputParser()

# 并行执行两个 LLM
chain = RunnableParallel({
    "gpt35": prompt | llm1 | parser,
    "gpt4o": prompt | llm2 | parser
})

result = chain.invoke({"question": "为什么天空是蓝色的？"})
print(result)
```

结果可能是：

```python
{
  "gpt35": "因为大气分子对阳光的散射",
  "gpt4o": "由于瑞利散射，蓝光被散射得更多"
}
```

---

✅ 总结：

1. **LCEL = 声明式链式拼装**
2. **两个核心原语**：RunnableSequence（顺序）、RunnableParallel（并行）
3. **语法糖**：`|`、`.pipe()`、dict → Parallel
4. **适用场景**：简单链路最合适；复杂流程用 LangGraph

---
 
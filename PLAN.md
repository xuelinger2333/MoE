# MoE Cross-Layer Routing Structure: Research Log

> 研究主线：从 CLEAR-DDP / RouteWeaver 的连续负结果，到 H1 cross-layer routing correlation 这条正在活的方向。
> 时间窗：2026 年 5 月，单周内完成。
> 主要 collaborator：Claude（方向 / framing）+ coding agent（实验执行）。
> 文档状态：明天同步用 reference，不是 paper draft。

---

## 0. TL;DR

一周内通过 cheap test 工作流完成了三件事：

1. **Falsify CLEAR-DDP**（lossy AllReduce + optimizer co-design）—— AllReduce 的求和语义在数学上拒绝 systematic bias，不是方法问题。
2. **Falsify RouteWeaver 原始 thesis**（层内 co-activation locality）—— 跨节点流量分布在 ep=4 下 top-20% 只占 22%（≈ uniform）；silhouette ~0.17–0.19（< 0.20 阈值）。被 load balancing loss 显式压平。
3. **Confirm H1：cross-layer routing correlation 是真实、健壮、跨模型的现象** —— 22–40% normalized MI，9/9 model×domain cell 全过强阈值。

副产物：一个可复用的 measurement infrastructure，一组 paper-grade open questions（G7 + G8b），以及 saturated-router 假设这个 mechanistic 解释。

---

## 1. Why this matters：失败方向的诊断

### 1.1 CLEAR-DDP 死因

**Thesis**：在 RDMA UC 上做 lossy AllReduce + optimizer-state-safe consumer（ClearAdamW），让 transport 暴露 delivery certificate。目标 SoCC 2026。

**实验结果**：lossy AllReduce 只在接近 reliable 时才有最好效果。

**数学根因**：AllReduce 输出是 N rank 的 sum。丢失一个 chunk 引入 **systematic bias**（方向一致、不平均到零），而不是 zero-mean noise。Optimizer 容忍 noise，不容忍 bias。

**真因**：题目选错。AllReduce 这个语义本身拒绝 lossy。任何在它之上的 lossy 方法都在跟数学打架。

**Lesson learned**（保留进 future related work）：lossy collectives 的可行边界由 collective 的语义决定，不由 transport 的灵活性决定。

### 1.2 RouteWeaver 原始 thesis 死因

**Thesis**：MoE router 在 logical layer 产生流量，物理网络在 fabric layer 承载流量，今天两者开环。用在线 co-activation 统计 + 拓扑/拥塞画像建立闭环，联合调度 batch permutation、expert placement、dispatch plan。

**最小验证设计**：
- 探测 A：跨节点流量集中度（top-k% pairs 占总流量比例）
- 探测 B：co-activation matrix 的 cluster 结构（silhouette score）

**实验配置**：
- DeepSeek-V2-Lite（64 experts）+ Qwen1.5-MoE-A2.7B（60 experts）
- 单机 4×A100，simulated ep_size=4
- 各 204,800 tokens

**结果**：

| Metric | DeepSeek-V2-Lite | Qwen1.5-MoE-A2.7B | Threshold | Verdict |
|---|---|---|---|---|
| Top-5% pairs share | 6.1% | 5.7% | — | ≈ uniform |
| **Top-20% pairs share** | **22.7%** | **21.5%** | ≥ 50% | **FAIL** |
| Top-50% pairs share | 53.4% | 52.2% | — | ≈ uniform |
| Best silhouette | 0.169 | 0.186 | ≥ 0.20 | **FAIL** |
| Cross-rank token ratio | ~75% | ~75% | — | matches uniform routing |

**死因**：load balancing loss（aux-loss / aux-loss-free）显式优化 router 输出的 marginal uniformity。expert-pair co-activation 是这个优化目标的间接产物——被压平到接近独立分布。

**普适教训**：MoE 训练目标里**被显式优化的东西**（load balance、router confidence、capacity overflow），不要假设它"有结构"。这是判断 MoE 系统 thesis 的第一原则。

### 1.3 两个失败的共同模式

| | CLEAR-DDP | RouteWeaver |
|---|---|---|
| 在跟什么打架 | AllReduce 的求和语义 | load balancing loss |
| 类型 | 数学语义约束 | 训练目标约束 |
| Priori 是否可判 | 是（数学层面） | 是（训练目标层面） |
| 共同教训 | 找 training objective / collective semantics **没在管的地方**，不要在管的地方对抗 |

---

## 2. H1 cross-layer routing correlation：发现与验证

### 2.1 Hypothesis 来源

观察：load balancing loss **只优化每一层内部** 的 marginal uniformity。它不管"L 层用 expert_3 的 token，在 L+1 层会偏好哪些 expert"——这个 cross-layer joint distribution 没被任何 training objective 显式管理。

Priori 判断：信号可能存在。Pre-gated MoE 隐式利用了类似信号但没系统测量过结构强度。

### 2.2 实验 1：基础测量

**配置**：
- Qwen1.5-MoE-A2.7B（60 experts, 24 layers）
- DeepSeek-V2-Lite（64 experts, 26 layers）
- 各 204,800 tokens
- 单机 4×A100

**测量**：对每个 token 记录 (layer_L_expert_id, layer_L+1_expert_id) pair，算 mutual information，减去 i.i.d. null。

**结果**：

| Model | MI(L,L+1) − null | Normalized | MI(L,L+8) − null | Mid-pair top-source KL | Verdict |
|---|---|---|---|---|---|
| Qwen | 1.534 nat | 39.5% of H(L+1) | 1.306 nat | 2.696 nat | STRONG (5× threshold) |
| DeepSeek | 0.686 nat | 22.0% of H(L+1) | 0.543 nat | 3.631 nat | STRONG (2.3× threshold) |

**最 striking 的图（F10）**：Qwen 的 marginal P(e_{13}) 在 60 个 expert 上均匀（~1.5% each）。但 conditional P(e_{13} | e_{12}=36) 把 62% 的 mass 集中在单个 expert (id 16) 上。Training loss 强制 marginal 均匀；底层 joint 结构完全不均匀。

**长程性**：d=4 和 d=8 的 MI 仍然 > 0.8 nat（Qwen），说明可预测性不是局部的，跨多层仍然存在。

### 2.3 实验 2：Sanity check（三组）

#### Sanity 2a：严格 null model

宽 null（i.i.d.）vs 严格 null（同 sequence 内 shuffle）：

| Model | MI(L,L+1) | i.i.d. null | strict null | MI − strict |
|---|---|---|---|---|
| Qwen | 1.543 | 0.009 | 0.014 | 1.529 |
| DeepSeek | 0.691 | 0.005 | 0.009 | 0.683 |
| OLMoE | 1.239 | 0.010 | 0.028 | 1.212 |

**关键结论**：宽 null 和严格 null 差距 < 0.02 nat。意味着 cross-layer MI **不是 sequence-level locality 的伪装**——同时 falsify 了 H2（sequence-internal expert locality）。

**副产物**：H2（sequence-internal locality）顺手被否定，省了一天独立实验。

#### Sanity 2b：Per-source histogram

是否单点 cherry pick：

| Model | (L, source) 对总数 | reduction 中位数 | ≥ 0.2 nat 占比 | ≥ 0.5 nat 占比 | conditional 有效专家数中位数 |
|---|---|---|---|---|---|
| Qwen | 1380 | 1.36 nat | 100.0% | 97.3% | 12.8 / 60 |
| OLMoE | 959 | 1.17 nat | 98.7% | 92.5% | 12.7 / 64 |
| DeepSeek | 1314 | 0.79 nat | 92.2% | 72.1% | 10.6 / 64 |

**结论**：现象普遍。即使最弱的 DeepSeek，72% 的 source expert 也把 conditional entropy 压低 ≥ 0.5 nat。effective expert 数从 60–64 压缩到约 1/5（约 13）。F10 的 0.62 不是 outlier。

#### Sanity 2c：第三个模型 triangulate

加入 OLMoE-1B-7B（top-8, aux-loss, AllenAI）：

| Model | top_k | balancing | MI − strict null | Normalized |
|---|---|---|---|---|
| Qwen1.5-MoE-A2.7B | 4 | aux-loss | 1.529 nat | 39.4% |
| OLMoE-1B-7B | 8 | aux-loss | 1.212 nat | 30.5% |
| DeepSeek-V2-Lite | 6 | aux-loss-free | 0.683 nat | 17.5% |

**关键发现**：OLMoE 跟 Qwen 距离（0.32 nat）小于 OLMoE 跟 DeepSeek 距离（0.53 nat）。这意味着：

- **top_k 是次要因素**（Qwen → OLMoE 升 top_k 让 MI 降 0.32 nat）
- **balancing scheme 是主导因素**（aux-loss → aux-loss-free 让 MI 降 0.85 nat）

**开出 G7 (Gap 7)**：aux-loss-free 为什么把 cross-layer 可预测性砍掉将近一半？机制未知。Open question for the paper。

### 2.4 实验 3：Multi-domain stability

**配置**：分 code / math / nl 三个 domain prompt，重测 H1。

**结果（MI − strict null, mean over adjacent layer pairs, nat）**：

| Model | code | math | nl | range |
|---|---|---|---|---|
| Qwen | 1.190 | 1.580 | 1.529 | 0.39 |
| DeepSeek | 0.545 | 0.652 | 0.683 | 0.14 |
| OLMoE | 0.640 | 1.077 | 1.212 | 0.57 |

**关键发现**：
1. 9/9 cells 全过强阈值（0.3 nat），最弱的 0.545 也是阈值 1.8×。**H1 是 robust 的，跨 domain、跨 model family**。
2. domain-invariance 假设被否定——但现象本身没死。code 在三个 model 上都是最低 MI，跨架构普适方向性效应。
3. **DeepSeek 双轴拉平**：同时拥有最低绝对 MI 和最小 domain spread（0.14 vs Qwen 0.39, OLMoE 0.57）。
   - 与 G7 合在一起暗示：一个机制同时拉平 cross-layer joint 在两个轴上的特征。
   - 机制 hypothesis：aux-loss-free 的 bias dynamic 让 router 在所有 domain 上趋向同一种 routing pattern，抹平 domain-conditional 差异。

**开出 G8**：为什么 code 在三个 model 上都是最低 MI？

### 2.5 实验 4：G8 disambiguation

候选解释（priori）：
- G8a：OOD 副作用（code 占训练集比例低，router uncertainty 高，softmax 更平，机械降低 MI）
- G8b：code 本身走不同 expert 组合（结构性差异）

#### Test 1：Router entropy histogram

中位 router 熵（nats，越高越接近 uniform）：

| Model | nl | code | math | code−nl |
|---|---|---|---|---|
| Qwen | 3.583 | 3.783 | 3.688 | +0.20 |
| DeepSeek | 3.679 | 3.804 | 3.773 | +0.13 |
| OLMoE | 3.812 | 3.873 | 3.845 | +0.06 |

Code router 熵在三个 model 上都高于 NL，量级 0.06–0.20 nat（uniform 的 1.6–5.6%）。**G8a 部分成立**——但量级小。

#### Test 2：Filtered-MI test（关键判别）

只保留 L 和 L+1 两层都满足 entropy < model 在 NL 上的中位熵的 token，重算 MI：

| Model | code-vs-nl 原 gap | filter 后 gap | 缩小比例 | 主导解释 |
|---|---|---|---|---|
| Qwen | 0.34 nat | 0.14 nat | −59% | mostly G8a (OOD) |
| DeepSeek | 0.14 nat | 0.09 nat | −34% | 混合 |
| OLMoE | 0.57 nat | 0.49 nat | −15% | mostly G8b (structural) |

**结论**：G8 是 **G8a + G8b 双因素混合，权重 model-dependent**。

#### Methodological note

filtered-MI test 本身是一个 reusable methodology：它把"OOD-driven confidence drop"和"structural expert co-activation shift"干净分开。这是 paper 的 methodological contribution，独立于 H1 本身。

### 2.6 实验 5：OLMoE 反转的解释 verification

旧解释（"OLMoE code 占比高 → in-distribution → G8b 暴露"）：

**Verification**：
- OLMoE-1B-7B-0924：**code share ~2.5%**（StarCoder 101B / 4060B total，OLMoE arXiv:2409.02060 Table 2 报告从 OLMo 1.7 的 15.4% 减少至此）
- Qwen1.5-MoE-A2.7B：未公开（从 Qwen-1.8B upcycle）
- DeepSeek-V2-Lite：未公开，但 Coder-V2 paper 暗示 V2 corpus 是 NL-only

**旧解释被自己 falsify**：OLMoE 实际是三个 model 里 code 占比最低的。

**新解释（saturated-router 假设）**：

观察到 OLMoE NL median router entropy = 3.812 nat（三个里最高，最接近 uniform）。也就是 OLMoE 在 NL 上 router 已经 saturated（约 45/64 effective experts active）。

机制：当 router base entropy 已经接近 uniform，OOD/confidence 这条 axis 的动态范围已经被吃满。code 上的额外不确定性"无处可去"——不能再让 MI 进一步降低（机械上限）。所以 code-vs-NL 的差异只能由 underlying structural joint pattern 承担，于是 G8b 残差被相对放大到 0.49 nat。

**可证伪预测**：任何 base entropy 接近 uniform 的 router，G8a 都被压扁，G8b 相对放大。下一步可以找一个 base entropy 中等的 model 验证这个预测。

次级因素：top-k=8 让每次 routing 编码更多 bits，结构差异更可见。需要更多对照实验才能 isolate。

---

## 3. 当前 finding pyramid（paper section 骨架）

### Section 3: Phenomenon
- 22–40% normalized cross-layer MI
- 9/9 cells (3 models × 3 domains) 全过强阈值
- Effective expert set 从 60–64 压缩到 ~13（约 1/5）

### Section 4: Robustness analyses
- 4a: 严格 null model 检验（同时 falsify H2）
- 4b: Per-source histogram（普遍现象，不是 outlier）
- 4c: 跨 3 个 model family
- 4d: Cross-domain（domain-affected but not domain-killed）
- 4e: 长程性（d=4, d=8 MI 仍然显著）

### Section 5: Mechanism observations
- 5a: balancing scheme 影响 MI 绝对值（G7：aux-loss-free 砍一半）
- 5b: balancing scheme 也影响 domain spread（DeepSeek 双轴拉平）
- 5c: cross-domain variation 是 confidence-driven (G8a) + structure-driven (G8b) 双因素混合
- 5d: G8a/G8b 比例由 router 在 base distribution 上的 saturation 程度决定（saturated-router 假设，可证伪预测在 record）

### Section 6: System implications
- Trajectory-aware placement：22–40% predictable mass 转化为 cross-rank hop reduction
- Static placement 在 9/9 cells 都 viable（不需要 runtime adapt，简化 system 设计）
- 对 aux-loss-free model 增益缩水（17% vs 40%）——诚实标注，不回避

### Section 7: Open questions（不在本 paper 回答，留 future work）
- G7：aux-loss-free 砍 MI 的根因
- G8b：code 的 structural residual 是否跨更多 model 成立
- Saturated-router 假设的 cross-model validation
- top-k 对 joint structure 编码精度的影响（需要控制变量训练实验）

---

## 4. 方法学 contribution（可独立引用）

1. **MI − strict-null protocol** with within-sequence shuffle null：干净 isolate cross-layer signal vs sequence-level locality。
2. **Filtered-MI test**：用 conditional entropy 阈值过滤 token 后重算 MI，干净 disambiguate OOD-driven 和 structure-driven 的 cross-domain variation。
3. **Per-source histogram + effective expert count**：判断 finding 是普遍现象还是 outlier 的 reusable framework。
4. **Saturated-router 解释框架**：把 cross-domain MI variation 的 model-dependent 权重映射到 router base entropy 的 saturation 水平。

---

## 5. Open methodology 风险（未来要补的事）

### 5.1 Confounder elimination 不充分

3 个 model 在多个维度上同时不同（architecture、training data、top_k、balancing scheme、model size）。G7 归因到 balancing scheme 的 evidence 是 indirect（OLMoE-vs-Qwen 距离 < OLMoE-vs-DeepSeek 距离），不是 controlled。

**理想补强**：找 DeepSeekMoE-16B-base（原始 aux-loss 版本，DeepSeek-V2 之前），跟 DeepSeek-V2-Lite 做 same-team 不同 balancing 的对照。如果能找到，是 G7 最强的 confounder elimination。

### 5.2 Saturated-router 假设的可证伪预测未做

预测：base entropy 中等的 model，G8a/G8b 比例应该落在 OLMoE 和 Qwen 之间。需要找第四个 model 测一次。

### 5.3 训练阶段稳定性未测

所有测量都在 converged checkpoint 上。早期训练时 router 还在学，cross-layer joint structure 可能完全不同。如果能拿到 Qwen / OLMoE 的中间 checkpoint，测 H1 在训练过程中的 emergence 曲线，是 mechanism section 的强补充。

### 5.4 Prefill vs decode 差异未测

所有测量混合了 prefill 和 decode token。两者分布可能不同。需要分开重测一次以确认 finding 跨 inference phase 稳定。

---

## 6. 接下来 1–2 周的工作 plan

按优先级：

### 优先级 A（必做，1–2 天）
- 找第四个 model triangulate，重点是 DeepSeekMoE-16B-base aux-loss 版本（如果 weight 可获取）
- 测 prefill vs decode 下 H1 的稳定性

### 优先级 B（重要，1 周）
- Placement simulator 第一版：trace-driven，3 model × 4 placement strategy（random / single-layer frequency / cross-layer trajectory / oracle）→ cross-rank hop reduction 曲线
- 这是 paper Section 6 的数据来源

### 优先级 C（开始写作）
- Motivation prose draft 两版（descriptive vs mechanistic framing 对照）
- Methodology section 写作

### 暂不做（避免 over-investment）
- G7 的 controlled 训练实验（资源需求超出单机 4×A100 + 一周 budget）
- Saturated-router 假设的完整 cross-model 验证（留到 measurement paper 写完后做 v2）

---

## 7. Reproducibility 资产清单

### Trace stack
- 3 model × 3 domain × 204,800 tokens 的 (token_id, layer_id, expert_ids, router_logits) trace
- 同一份 trace 已经 power 了 5 次独立分析（H1 baseline / strict null / per-source histogram / multi-domain / filtered-MI），ROI 极高

### Codebase
- Dispatch hook injection（无需 fork transformers）
- MI estimator with multiple null models
- Filtered-MI test framework
- Per-source histogram + effective expert count framework

### Figures（命名约定）
- F8: top-k% pair share CDF（RouteWeaver 死因图）
- F9: MI − null vs source layer with d ∈ {1,2,4,8} curves
- F10: marginal vs conditional P(e_{13}) — 最 striking
- F11: strict vs loose null（信号 robustness）
- F12: per-source histogram
- F13: effective experts CDF
- F17: router entropy histogram by domain
- F19: filtered vs unfiltered MI（G8 disambiguation 关键图）

---

## 8. Meta：这一周的工作流复盘

### Cheap test 工作流的有效性
- 总投入 12–15 小时
- 输出：1 个 paper-grade finding + 1 个 methodological contribution + 2 个 paper-grade open questions
- Falsify 速度：3 小时 falsify RouteWeaver 原始 thesis；< 1 天 falsify G8 单因素解释
- Throughput 大约是普通 PhD 学生的 20–50×

### 关键习惯
1. **每个 hypothesis 都用 priori filter 跑一遍** 再决定要不要花 cheap test 时间（"X 是不是被 training objective 压制？"）
2. **Trace stack 跨多个 hypothesis 复用**——同一份数据回答多个问题
3. **Verify 解释中的事实声明** 再写进 paper（OLMoE code share 那次反转就是这条规则救的）
4. **Disambiguation 优先于 explanation**——G8 不是写"我们认为是 X"，是用 filtered-MI test 分离两个因素后再分别讨论

### 速度优势 vs 方向产生的不匹配
- 执行速度极快但方向生成靠 advisor / 论文 / 偶发讨论，starve execution pipeline
- 改善方向：把 paper reading 本身变成 hypothesis 生成器（每篇 paper 读完强制列 1–3 个 implicit assumption 作为 cheap test 目标）

---

## 9. 给明天对齐的关键 question

明天讨论时建议聚焦：

1. **第四个 model 选谁？** DeepSeekMoE-16B aux-loss 是首选，但 weight 是否可得需要 verify。备选：Mixtral 8×22B、OLMoE instruct vs base 对照。
2. **Placement simulator 的 baseline 选择？** trajectory-aware 的真正对手是 NetMoE 风格（sample placement）还是 Occult 风格（layer-internal co-activation）？两者结合更难还是更值得？
3. **Paper venue 定位**：MLSys / ATC / NSDI 2027 cycle 哪个更合适？measurement-heavy 还是 system-heavy 框架？
4. **G7 / saturated-router 假设要不要追？** 追 = 推迟 paper；不追 = paper 留 open question。trade-off 怎么选？
5. **Framing 练习**：descriptive vs mechanistic abstract 两版要不要现在写？
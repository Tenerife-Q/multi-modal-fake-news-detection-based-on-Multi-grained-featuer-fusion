# multi-modal-fake-news-detection-based-on-Multi-grained-featuer fusion
## ***外部维基百科***
本项目融合了外部维基百科，旨在给模型提供外部知识判断，让模型不仅仅局限于已有图像和文本进行判断。
## ***对比学习***
本项目创新的多粒度信息融合模块当中，引入了对比学习（后面根据文章内容理解之后进行补充）
## ***Mamba模型***
在对比学习之后让文本和图像分别经过一个Mamba层（这里搞懂Mamba当中的q,k,v键），得到增强的文本和图像的信息

## ***文章结构展示***（记得填充）





## ***数据集***
本文使用了公开的weibo数据集，以及数据集gossip( https://github.com/junyachen/Data-examples)
~~~
##数据处理方式

~~~


##训练和验证
~~~

~~~

## 与 `yuanjing-core` 区块链模块的对接建议

当前更推荐 **先保持两个仓库独立**，先在本仓库通过 PR 补齐稳定的对接层，再和 `Tenerife-Q/yuanjing-core` 联调，而不是立刻把两个模块硬合并到一个总仓库。

这样做有三个直接好处：

1. **职责边界清晰**：本仓库继续负责多模态推理，`yuanjing-core` 继续负责模型治理、存证和审计。
2. **更适合负责人积累 PR 记录**：你可以先向模型仓库提交“接口标准化 PR / 推理结果导出 PR”，再向区块链仓库提交“接收 MMFN 负载 PR / 联调 PR”。
3. **后续仍可增加总控仓库**：等接口稳定后，再新建总仓库存放编排脚本、部署文件和 CI，会更稳。

### 当前两个仓库的最小对接面

- 本仓库模型输出：`trainMMFN.py` / `MMFN.py`
  - `weibo`: `0=fake_rumor`, `1=real_nonrumor`
  - `gossip`: `0=real_news`, `1=fake_news`
- `yuanjing-core` HTTP API：
  - `POST /model/register`
  - `POST /prove`

为了避免标签语义不一致，本仓库新增了 `blockchain_bridge.py`，负责把 MMFN 的分类结果转换成适合 `yuanjing-core` 的请求负载：

- `weibo`
  - `0 -> verdict=false`
  - `1 -> verdict=true`
- `gossip`
  - `0 -> verdict=true`
  - `1 -> verdict=false`

### 推荐落地流程

1. **第一步：先向本仓库发 PR**
   - 引入 `blockchain_bridge.py`
   - 固化标签到 `verdict` 的映射
   - 输出 `yuanjing-core` 所需的 `register_model` / `prove` 负载
2. **第二步：向 `yuanjing-core` 发 PR**
   - 把当前后端里 mock 的 `activated_prompts`、`prompt_pool_hash`、`external_knowledge_hash` 改为真正接收 MMFN 侧传值
3. **第三步：如有展示或部署需求，再新建总控仓库**
   - 只放编排、部署、联调文档
   - 不建议一开始就把两边核心代码搬到一个仓库里

### 生成对接负载

~~~bash
python scripts/build_yuanjing_payload.py \
  --dataset weibo \
  --image-path /abs/path/to/news.png \
  --pred-label 1 \
  --confidence 0.97 \
  --checkpoint ckpt/mmfn_base/best_model.pth \
  --description "MMFN base checkpoint"
~~~

输出结果包含：

- `register_model`：可提交到 `yuanjing-core` 的 `/model/register`
- `prove`：可提交到 `yuanjing-core` 的 `/prove`
- `local_record`：本地留档用的扩展字段（如数据集、知识摘要哈希等）

这是当前最小侵入、最适合先积累清晰 PR 记录的一种联调方案。

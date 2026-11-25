# hybrid-rag-eval · 前端

Angular 20 的单页对话框，对接 `backend/` 的 `GET /ask`。

## 这个界面做了什么

一个问题 → 左侧一条系统回答，右侧一条用户提问。系统回答里包含四块信息：

| 区块 | 内容 | 数据来源 |
|---|---|---|
| 最终来源 | 精排 / RRF 融合 / BM25 单路 / 向量单路 / 抽取式兜底 | `degraded` + `trace` 推导 |
| AI 回答 | 模型输出正文 | `answer` |
| 引用资料 | PDF 文件名、页码、块类型、相关度、各路名次 | `contexts[]` |
| 追踪详情 | 各阶段耗时、候选数量、精排前后顺序 | `trace` |

**为什么单独做「最终来源」这一栏。** 后端是阶梯式降级的：精排挂了会退回 RRF、
向量挂了只剩 BM25、LLM 挂了返回原文摘抄。这些情况下 `answer` 看上去和正常回答一样，
不看降级信息就无法判断可信度。判定逻辑集中在
[`src/app/pipeline-status.ts`](src/app/pipeline-status.ts)，并有 8 个单元用例覆盖。

## 运行

需要 Node 20.19+ / 22.12+ / 24+，以及运行中的后端。

```powershell
cd frontend
pnpm install          # 首次

# 另开一个终端，先起后端
cd ../backend
python -m app.main    # 监听 127.0.0.1:8000

# 回到 frontend
pnpm start            # http://localhost:4200
```

后端要先就绪，否则顶部状态灯会显示「后端未连接」（点「刷新」可重试）。

支持用 `?q=` 直达某个问题（便于分享与自动化验证）：

```text
http://localhost:4200/?q=平安银行2023年末个人贷款不良率是多少？
```

## 与后端的连接方式

开发期前端请求 `/api/*`，由 `proxy.conf.json` 转发到 `127.0.0.1:8000`，浏览器不涉及跨域：

```text
浏览器  GET http://localhost:4200/api/ask?q=...
   ↓ Angular dev-server 代理（去掉 /api 前缀）
后端    GET http://127.0.0.1:8000/ask?q=...
```

生产部署时把 `/api` 用 Nginx 等反向代理指到后端即可，前端代码不用改。
后端另外开了 CORS（`localhost:4200` / `127.0.0.1:4200`），所以直连也能调试。

## 目录

```text
src/app/
├── app.ts                 # 页面逻辑：消息列表、发送、展开/折叠
├── app.html               # 模板（Angular 新控制流 @if / @for）
├── app.css                # 样式
├── models.ts              # 与后端响应对应的类型
├── chat.service.ts        # HttpClient 封装（/ask、/health）
├── pipeline-status.ts     # 「最终结果来自哪一步」的判定
└── *.spec.ts              # 单元测试
```

## 测试与构建

```powershell
pnpm test             # Karma + Jasmine，10 个用例
pnpm build            # 产物在 dist/frontend
```

> 如果本机没有 Chrome，可把 `CHROME_BIN` 指向 Edge：
> `$env:CHROME_BIN = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"`

## 已知限制

- 一次只允许一个请求在飞（单实例后端 + 单卡推理，并发只会排队）。
- 首次提问需要等后端加载模型（约 80 秒），界面显示「检索与生成中…」。
- 尚未做流式输出（后端目前是同步接口，没有 SSE）。
- 未做会话持久化，刷新页面即清空。

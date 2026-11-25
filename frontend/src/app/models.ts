/**
 * 与后端 `GET /ask`、`GET /health` 返回结构一一对应的类型定义。
 *
 * 后端用 Python dataclass + `asdict()` 序列化，字段名是 snake_case，
 * 这里保持原样，避免前后端来回映射产生歧义。
 */

/** 一条检索到的文献片段 */
export interface RetrievedContext {
  /** 在提示词里的序号（从 1 开始） */
  index: number;
  chunk_id: string;
  /** 精排分数；精排被跳过时这里是 RRF 分数 */
  score: number;
  /** 送进提示词的正文（后端已按字符预算截断） */
  text: string;
  /** 形如 data/corpus/000001_平安银行_2023年年度报告.pdf */
  source: string;
  page: number;
  /** text | table */
  block_type: string;
  /** 仅精排结果有值 */
  rrf_score: number | null;
  /** 该片段在各路的原始名次，如 { vector: 8, bm25: 1 } */
  ranks: Record<string, number>;
}

/** 追踪线索里某个阶段的一条命中 */
export interface TraceStageHit {
  rank: number;
  chunk_id: string;
  score: number;
  preview: string;
  ranks?: Record<string, number>;
  /** 仅精排结果：该片段在融合候选池里的名次 */
  pool_rank?: number;
}

/** 追踪线索：最终进入上下文的一条资料 */
export interface TraceContext {
  index: number;
  chunk_id: string;
  source: string;
  page: number;
  block_type: string;
  chars: number;
  score: number;
}

/** 七个阶段的追踪线索 */
export interface Trace {
  query: string;
  vector: TraceStageHit[];
  bm25: TraceStageHit[];
  rrf: TraceStageHit[];
  rerank: {
    before: TraceStageHit[];
    after: TraceStageHit[];
  };
  context: TraceContext[];
  timing: Record<string, number>;
  fallback: string[];
  warnings: string[];
}

/** 一次问答的完整结果 */
export interface AnswerResult {
  question: string;
  answer: string;
  contexts: RetrievedContext[];
  model: string;
  /** 实际送给 LLM 的消息（含提示词全文），前端默认不展示 */
  messages: Array<{ role: string; content: string }>;
  usage: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  } | null;
  bootstrap_ms: number;
  retrieval_ms: number;
  generation_ms: number;
  elapsed_ms: number;
  /**
   * 发生降级的环节，可能含 vector / bm25 / rerank / llm。
   * 空数组表示全链路正常。
   */
  degraded: string[];
  warnings: string[];
  trace: Trace | null;
  /**
   * 年份一致性校验（后端确定性检查，不依赖模型）。
   * status=mismatch 表示答案的数字来自别年度的年报——数字是真的，但答非所问。
   */
  answer_check: AnswerCheck | null;
}

/** 年份校验结果 */
export interface AnswerCheck {
  status: 'ok' | 'mismatch' | 'skipped';
  /** 问题涉及的年份（区间题会展开） */
  asked_years: number[];
  /** 答案里明写的年份 */
  answer_years: number[];
  /** 答案数值所属资料的文件年份 */
  source_year: number | null;
  /** 所属资料在上下文里的序号 */
  source_index: number | null;
  detail: string;
}

/** 服务与检索栈状态 */
export interface HealthResponse {
  status: string;
  llm: {
    base_url: string;
    reachable: boolean;
    detail: string;
  };
  pipeline: {
    loaded: Record<string, boolean>;
    errors: Record<string, string>;
    load_ms: Record<string, number>;
    ready: boolean;
  };
}

/**
 * 把后端返回的 `degraded` 解析成「这次结果到底由哪一步给出」。
 *
 * 后端采用阶梯式降级：任何一环失败都不会 500，而是退回上一级可用组件，
 * 并把失败环节记进 `degraded`。所以只看回答本身无法判断它来自精排还是融合，
 * 必须由这个函数还原。
 *
 * 对照表（与 backend/README.md 的降级阶梯一致）：
 *
 * | degraded 内容              | 检索来源           | 生成方式     |
 * |---------------------------|-------------------|-------------|
 * | 空                        | 精排（完整链路）    | LLM         |
 * | rerank                    | RRF 融合           | LLM         |
 * | vector                    | BM25 单路          | LLM         |
 * | bm25                      | 向量单路           | LLM         |
 * | rerank + llm              | RRF 融合           | 抽取式兜底   |
 * | vector + llm              | BM25 单路          | 抽取式兜底   |
 */

/** 检索侧最终产出上下文的那一步 */
export type RetrievalStage = 'rerank' | 'rrf' | 'vector' | 'bm25' | 'none';

/** 答案的生成方式 */
export type GenerationMode = 'llm' | 'extractive';

export interface PipelineStatus {
  /** 检索来源 */
  retrieval: RetrievalStage;
  /** 检索来源的中文标签，如「精排」 */
  retrievalLabel: string;
  /** 一句话说明为什么是这一步 */
  retrievalDetail: string;
  /** 生成方式 */
  generation: GenerationMode;
  generationLabel: string;
  /** 「最终结果由哪一步给出」的完整表述 */
  answerSource: string;
  /** 是否发生了降级 */
  isDegraded: boolean;
  /** 原始降级环节列表 */
  degraded: string[];
}

const RETRIEVAL_LABELS: Record<RetrievalStage, string> = {
  rerank: '精排',
  rrf: 'RRF 融合',
  vector: '向量检索',
  bm25: 'BM25 检索',
  none: '无',
};

/**
 * 根据降级列表推导链路状态。
 *
 * @param degraded 后端返回的降级环节
 * @param hasContexts 是否真的拿到了上下文（为空时说明检索彻底失败）
 */
export function resolvePipelineStatus(
  degraded: string[] | null | undefined,
  hasContexts = true,
): PipelineStatus {
  const failed = new Set(degraded ?? []);

  // ---- 1. 生成方式：LLM 挂掉时后端返回的是抽取式兜底（原文摘抄） ----
  const generation: GenerationMode = failed.has('llm') ? 'extractive' : 'llm';
  const generationLabel =
    generation === 'llm' ? 'LLM 生成' : '抽取式兜底（未经改写）';

  // ---- 2. 检索来源：按降级的组合判断实际用到哪一路 ----
  const vectorDown = failed.has('vector');
  const bm25Down = failed.has('bm25');
  const rerankDown = failed.has('rerank');

  let retrieval: RetrievalStage;
  let retrievalDetail: string;

  if (!hasContexts) {
    retrieval = 'none';
    retrievalDetail = '未检索到任何片段';
  } else if (vectorDown && bm25Down) {
    // 后端在两路都不可用时会直接 503，正常不会走到这里
    retrieval = 'none';
    retrievalDetail = '向量与 BM25 均不可用';
  } else if (vectorDown) {
    retrieval = 'bm25';
    retrievalDetail = '向量检索失败，仅用 BM25 单路';
  } else if (bm25Down) {
    retrieval = 'vector';
    retrievalDetail = 'BM25 失败，仅用向量单路';
  } else if (rerankDown) {
    retrieval = 'rrf';
    retrievalDetail = '精排失败，退回 RRF 融合顺序';
  } else {
    retrieval = 'rerank';
    retrievalDetail = '完整链路：向量 + BM25 → RRF → 精排';
  }

  // ---- 3. 合成「最终结果来自哪里」 ----
  let answerSource: string;
  if (generation === 'extractive') {
    answerSource =
      retrieval === 'none'
        ? '无可用结果'
        : `抽取式兜底：直接返回${RETRIEVAL_LABELS[retrieval]}命中的原文（LLM 不可用）`;
  } else if (retrieval === 'none') {
    answerSource = '无可用结果';
  } else {
    answerSource =
      retrieval === 'rerank'
        ? '精排结果（完整链路）'
        : `${RETRIEVAL_LABELS[retrieval]}结果（上游降级）`;
  }

  return {
    retrieval,
    retrievalLabel: RETRIEVAL_LABELS[retrieval],
    retrievalDetail,
    generation,
    generationLabel,
    answerSource,
    isDegraded: failed.size > 0,
    degraded: [...failed],
  };
}

/** 把降级环节翻译成中文，用于提示条 */
export function describeDegradedStage(stage: string): string {
  const labels: Record<string, string> = {
    vector: '向量检索',
    bm25: 'BM25 检索',
    rerank: '精排',
    llm: '大模型生成',
  };
  return labels[stage] ?? stage;
}

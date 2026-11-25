import { resolvePipelineStatus } from './pipeline-status';

/**
 * 「最终结果由哪一步给出」的判定是整个前端最不能出错的一环——
 * 它决定了用户看到的是「精排结果」还是「降级后的融合结果」。
 */
describe('resolvePipelineStatus', () => {
  it('无降级时判定为完整链路', () => {
    const status = resolvePipelineStatus([], true);
    expect(status.retrieval).toBe('rerank');
    expect(status.generation).toBe('llm');
    expect(status.isDegraded).toBeFalse();
    expect(status.answerSource).toContain('精排');
  });

  it('精排失败时退回 RRF 融合', () => {
    const status = resolvePipelineStatus(['rerank'], true);
    expect(status.retrieval).toBe('rrf');
    expect(status.generation).toBe('llm');
    expect(status.isDegraded).toBeTrue();
    expect(status.answerSource).toContain('RRF');
  });

  it('向量路失败时只剩 BM25', () => {
    const status = resolvePipelineStatus(['vector'], true);
    expect(status.retrieval).toBe('bm25');
    expect(status.answerSource).toContain('BM25');
  });

  it('BM25 失败时只剩向量', () => {
    const status = resolvePipelineStatus(['bm25'], true);
    expect(status.retrieval).toBe('vector');
    expect(status.answerSource).toContain('向量');
  });

  it('精排与 LLM 同时失败：检索来自 RRF，答案来自抽取式兜底', () => {
    const status = resolvePipelineStatus(['rerank', 'llm'], true);
    expect(status.retrieval).toBe('rrf');
    expect(status.generation).toBe('extractive');
    expect(status.answerSource).toContain('抽取式');
  });

  it('向量与 BM25 同时不可用时不谎报来源', () => {
    const status = resolvePipelineStatus(['vector', 'bm25'], false);
    expect(status.retrieval).toBe('none');
    expect(status.answerSource).toBe('无可用结果');
  });

  it('没有上下文时不谎报来源', () => {
    const status = resolvePipelineStatus([], false);
    expect(status.retrieval).toBe('none');
  });

  it('缺少 degraded 字段时按正常链路处理', () => {
    const status = resolvePipelineStatus(null, true);
    expect(status.retrieval).toBe('rerank');
    expect(status.isDegraded).toBeFalse();
  });
});

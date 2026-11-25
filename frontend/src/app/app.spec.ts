import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';

import { App } from './app';
import { AnswerResult } from './models';
import { resolvePipelineStatus } from './pipeline-status';

/** 造一个最小可用的后端响应，只填渲染需要的字段 */
function fakeResult(overrides: Partial<AnswerResult> = {}): AnswerResult {
  return {
    question: '平安银行2023年末个人贷款不良率是多少？',
    answer: '根据资料4，平安银行2024年末个人贷款不良率是1.39%。',
    contexts: [
      {
        index: 1,
        chunk_id: 'c1',
        score: 0.99,
        text: '2023 年末，本行个人贷款不良率1.37%。',
        source: 'data/corpus/000001_平安银行_2023年年度报告.pdf',
        page: 37,
        block_type: 'text',
        rrf_score: 0.06,
        ranks: { bm25: 3 },
      },
    ],
    model: 'stub.gguf',
    messages: [],
    usage: null,
    bootstrap_ms: 0,
    retrieval_ms: 10,
    generation_ms: 100,
    elapsed_ms: 110,
    degraded: [],
    warnings: [],
    trace: null,
    answer_check: null,
    ...overrides,
  };
}

describe('App', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();
  });

  it('初始状态显示空态与示例问题', () => {
    const fixture = TestBed.createComponent(App);
    fixture.detectChanges();
    const text = (fixture.nativeElement as HTMLElement).textContent ?? '';
    expect(text).toContain('问点什么');
    expect(text).toContain('平安银行2023年末个人贷款不良率是多少？');
  });

  it('草稿为空时发送按钮不可点击', () => {
    const fixture = TestBed.createComponent(App);
    fixture.detectChanges();
    const button = (fixture.nativeElement as HTMLElement).querySelector(
      '.send',
    ) as HTMLButtonElement;
    expect(button.disabled).toBeTrue();
  });

  it('年份校验不通过时渲染红色告警条', () => {
    const fixture = TestBed.createComponent(App);
    const component = fixture.componentInstance;
    // 直接塞一条「模型答错年份」的回答，验证告警是否出现在页面上
    component.messages.set([
      {
        id: 1,
        role: 'assistant',
        text: '根据资料4，平安银行2024年末个人贷款不良率是1.39%。',
        result: fakeResult({
          answer_check: {
            status: 'mismatch',
            asked_years: [2023],
            answer_years: [2024],
            source_year: 2024,
            source_index: 4,
            detail: '问题问的是 2023 年，但答案的数值来自 2024 年报（参考资料 4）',
          },
        }),
        status: resolvePipelineStatus([], true),
      },
    ]);
    fixture.detectChanges();

    const element = fixture.nativeElement as HTMLElement;
    const text = element.textContent ?? '';
    expect(text).toContain('年份校验未通过');
    expect(text).toContain('2024 年报');
    // 告警用独立样式，便于人眼第一时间发现
    expect(element.querySelector('.check-alert')).not.toBeNull();
  });

  it('校验通过时不显示告警条，只显示通过徽章', () => {
    const fixture = TestBed.createComponent(App);
    const component = fixture.componentInstance;
    component.messages.set([
      {
        id: 1,
        role: 'assistant',
        text: '平安银行2023年末个人贷款不良率是1.37%。',
        result: fakeResult({
          answer_check: {
            status: 'ok',
            asked_years: [2023],
            answer_years: [2023],
            source_year: 2023,
            source_index: 1,
            detail: '答案数值来自 2023 年报，与问题年份一致',
          },
        }),
        status: resolvePipelineStatus([], true),
      },
    ]);
    fixture.detectChanges();

    const element = fixture.nativeElement as HTMLElement;
    expect(element.querySelector('.check-alert')).toBeNull();
    expect(element.textContent ?? '').toContain('年份校验 ✓');
  });
});

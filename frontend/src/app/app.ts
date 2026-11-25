import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  OnInit,
  computed,
  effect,
  inject,
  signal,
  viewChild,
} from '@angular/core';
import { HttpErrorResponse } from '@angular/common/http';

import { ChatService } from './chat.service';
import { AnswerResult, HealthResponse, RetrievedContext } from './models';
import { PipelineStatus, describeDegradedStage, resolvePipelineStatus } from './pipeline-status';

/** 一条聊天记录 */
interface ChatMessage {
  id: number;
  role: 'user' | 'assistant';
  /** 用户问题，或 AI 的回答正文 */
  text: string;
  /** 仅 assistant：后端完整结果 */
  result?: AnswerResult;
  /** 仅 assistant：链路状态（最终结果来自哪一步） */
  status?: PipelineStatus;
  /** 请求失败时的错误描述 */
  error?: string;
  /** 请求进行中 */
  pending?: boolean;
  /** 已展开全文的片段序号集合 */
  expandedChunks?: Set<number>;
  /** 是否展开「追踪详情」 */
  showTrace?: boolean;
}

/** 追踪面板里各耗时的展示顺序与中文名 */
const TIMING_LABELS: Array<[string, string]> = [
  ['bootstrap_ms', '模型加载'],
  ['vector_ms', '向量检索'],
  ['bm25_ms', 'BM25 检索'],
  ['rrf_ms', 'RRF 融合'],
  ['rerank_ms', '精排'],
  ['context_ms', '上下文组装'],
  ['retrieval_ms', '检索合计'],
  ['generation_ms', '答案生成'],
  ['total_ms', '本次合计'],
];

@Component({
  selector: 'app-root',
  imports: [],
  templateUrl: './app.html',
  styleUrl: './app.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class App implements OnInit {
  private readonly chat = inject(ChatService);

  /** 聊天记录 */
  readonly messages = signal<ChatMessage[]>([]);
  /** 输入框内容 */
  readonly draft = signal('');
  /** 是否有请求在飞（同一时间只允许一个，避免并发压垮单实例后端） */
  readonly sending = signal(false);
  /** 后端健康状态 */
  readonly health = signal<HealthResponse | null>(null);
  /** 健康检查失败信息 */
  readonly healthError = signal<string | null>(null);

  /** 后端是否就绪（用于顶部状态灯） */
  readonly backendReady = computed(() => {
    const data = this.health();
    if (!data) return false;
    return data.status === 'ok' && data.pipeline.ready;
  });

  /** LLM 是否可达 */
  readonly llmReachable = computed(() => this.health()?.llm.reachable ?? false);

  /** 空状态下的示例问题，取自 Golden Set 的真实题目 */
  readonly samples = [
    '平安银行2023年末个人贷款不良率是多少？',
    '万科2024年的研发投入金额是多少？',
    '国华网安自研的“数据与个人信息合规智探大模型一体机”具备哪些能力？',
  ];

  private nextId = 1;

  /** 消息区容器，用于新消息到达后自动滚到底部 */
  private readonly scroller = viewChild<ElementRef<HTMLElement>>('scroller');

  constructor() {
    // 消息数量或最后一条消息的状态变化时，滚动到底部（等 DOM 更新后再滚）
    effect(() => {
      const list = this.messages();
      const last = list[list.length - 1];
      // 读取 pending 也是为了让「等待中 → 完成」也触发一次滚动
      void last?.pending;
      queueMicrotask(() => this.scrollToBottom());
    });
  }

  ngOnInit(): void {
    this.refreshHealth();
  }

  private scrollToBottom(): void {
    const element = this.scroller()?.nativeElement;
    if (element) {
      element.scrollTop = element.scrollHeight;
    }
  }

  // ------------------------------------------------------------------ //
  // 健康检查
  // ------------------------------------------------------------------ //
  refreshHealth(): void {
    this.chat.health().subscribe({
      next: (data) => {
        this.health.set(data);
        this.healthError.set(null);
      },
      error: (error: HttpErrorResponse) => {
        this.health.set(null);
        this.healthError.set(this.describeError(error));
      },
    });
  }

  // ------------------------------------------------------------------ //
  // 输入与发送
  // ------------------------------------------------------------------ //
  onInput(event: Event): void {
    this.draft.set((event.target as HTMLTextAreaElement).value);
  }

  /** Enter 发送，Shift+Enter 换行 */
  onKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.send();
    }
  }

  useSample(question: string): void {
    this.draft.set(question);
    this.send();
  }

  send(): void {
    const question = this.draft().trim();
    if (!question || this.sending()) {
      return;
    }

    const userMessage: ChatMessage = { id: this.nextId++, role: 'user', text: question };
    const assistantMessage: ChatMessage = {
      id: this.nextId++,
      role: 'assistant',
      text: '',
      pending: true,
      expandedChunks: new Set<number>(),
      showTrace: false,
    };
    this.messages.update((list) => [...list, userMessage, assistantMessage]);
    this.draft.set('');
    this.sending.set(true);

    const startedAt = performance.now();
    this.chat.ask(question).subscribe({
      next: (result) => {
        this.patchMessage(assistantMessage.id, {
          text: result.answer,
          result,
          status: resolvePipelineStatus(result.degraded, result.contexts.length > 0),
          pending: false,
        });
        this.sending.set(false);
        // 降级时顺手刷新健康状态，让顶部指示灯与本次结果一致
        if (result.degraded.length > 0) {
          this.refreshHealth();
        }
        console.info(`[ask] 端到端 ${Math.round(performance.now() - startedAt)} ms`);
      },
      error: (error: HttpErrorResponse) => {
        this.patchMessage(assistantMessage.id, {
          pending: false,
          error: this.describeError(error),
        });
        this.sending.set(false);
      },
    });
  }

  /** 只改目标消息，其余引用保持不变；数组换新引用以触发渲染 */
  private patchMessage(id: number, patch: Partial<ChatMessage>): void {
    this.messages.update((list) =>
      list.map((message) => (message.id === id ? { ...message, ...patch } : message)),
    );
  }

  private describeError(error: HttpErrorResponse): string {
    if (error.status === 0) {
      return '连不上后端。请确认 FastAPI 已在 127.0.0.1:8000 启动（python -m app.main），并检查代理配置。';
    }
    // FastAPI 的 HTTPException 会把说明放在 detail 字段
    const detail =
      typeof error.error === 'object' && error.error && 'detail' in error.error
        ? String((error.error as { detail: unknown }).detail)
        : error.message;

    if (error.status === 503) {
      return `后端返回 503：${detail}`;
    }
    if (error.status === 400) {
      return `请求参数有误：${detail}`;
    }
    return `请求失败（HTTP ${error.status}）：${detail}`;
  }

  // ------------------------------------------------------------------ //
  // 展示辅助
  // ------------------------------------------------------------------ //
  /** 从 source 路径里取出文件名 */
  fileName(source: string): string {
    const parts = source.split(/[\\/]/);
    return parts[parts.length - 1] || source;
  }

  /** 片段类型中文名 */
  blockLabel(type: string): string {
    return type === 'table' ? '表格' : '正文';
  }

  /** 毫秒格式化：小于 1 秒显示 ms，否则显示秒 */
  formatMs(ms: number): string {
    if (ms >= 1000) {
      return `${(ms / 1000).toFixed(2)} s`;
    }
    return `${Math.round(ms)} ms`;
  }

  /** 分数保留 4 位 */
  formatScore(score: number): string {
    return score.toFixed(4);
  }

  /** 片段正文：未展开时截断 */
  previewOf(context: RetrievedContext, limit = 220): string {
    const text = context.text.trim();
    return text.length > limit ? `${text.slice(0, limit)}…` : text;
  }

  isExpanded(message: ChatMessage, index: number): boolean {
    return message.expandedChunks?.has(index) ?? false;
  }

  toggleChunk(message: ChatMessage, index: number): void {
    const set = new Set(message.expandedChunks ?? []);
    if (set.has(index)) {
      set.delete(index);
    } else {
      set.add(index);
    }
    this.patchMessage(message.id, { expandedChunks: set });
  }

  toggleTrace(message: ChatMessage): void {
    this.patchMessage(message.id, { showTrace: !message.showTrace });
  }

  /** 追踪面板里的耗时行（按固定顺序，只显示有值的） */
  timingRows(result: AnswerResult): Array<{ label: string; value: string }> {
    const timing = result.trace?.timing ?? {};
    const rows: Array<{ label: string; value: string }> = [];

    for (const [key, label] of TIMING_LABELS) {
      const value = timing[key];
      if (typeof value === 'number') {
        rows.push({ label, value: this.formatMs(value) });
      }
    }
    // 模型加载的细分项（冷启动才有值）
    for (const [key, value] of Object.entries(timing)) {
      if (key.startsWith('load_') && key.endsWith('_ms')) {
        rows.push({ label: `加载 ${key.slice(5, -3)}`, value: this.formatMs(value) });
      }
    }
    return rows;
  }

  /** 追踪面板里的候选数量统计 */
  traceCounts(result: AnswerResult): string {
    const trace = result.trace;
    if (!trace) return '—';
    const parts = [
      `向量 ${trace.vector.length}`,
      `BM25 ${trace.bm25.length}`,
      `融合 ${trace.rrf.length}`,
    ];
    if (trace.rerank.after.length > 0) {
      parts.push(`精排取 ${trace.rerank.after.length}`);
    }
    return parts.join(' → ');
  }

  /** 降级环节的中文描述 */
  degradedLabels(status: PipelineStatus): string {
    return status.degraded.map(describeDegradedStage).join('、');
  }

  /** 各路的原始名次，如「向量 #8 · BM25 #1」；没有名次信息时返回空串 */
  rankText(context: RetrievedContext): string {
    const entries = Object.entries(context.ranks ?? {});
    if (entries.length === 0) {
      return '';
    }
    const labels: Record<string, string> = { vector: '向量', bm25: 'BM25' };
    return entries
      .map(([name, rank]) => `${labels[name] ?? name} #${rank}`)
      .join(' · ');
  }

  trackById(_index: number, message: ChatMessage): number {
    return message.id;
  }
}

import { HttpClient, HttpParams } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, timeout } from 'rxjs';

import { AnswerResult, HealthResponse } from './models';

/**
 * 后端接口封装。
 *
 * 开发期走 `/api` 前缀，由 `proxy.conf.json` 转发到 FastAPI（127.0.0.1:8000），
 * 这样浏览器不涉及跨域；生产部署时把 `/api` 用反向代理指向后端即可。
 */
@Injectable({ providedIn: 'root' })
export class ChatService {
  private readonly http = inject(HttpClient);

  /** 后端前缀，集中在模板里改一处 */
  private readonly baseUrl = '/api';

  /**
   * 单次问答请求的超时时间。
   *
   * 后端首次请求要加载嵌入与精排模型（约 80 秒），再加上 LLM 生成，
   * 所以这里给到 5 分钟，避免冷启动就被判超时。
   */
  private readonly askTimeoutMs = 300_000;

  /**
   * 提问。
   *
   * @param question 用户问题
   * @param topM 送进提示词的资料条数
   * @param candidates 精排候选数
   */
  ask(question: string, topM = 5, candidates = 20): Observable<AnswerResult> {
    const params = new HttpParams()
      .set('q', question)
      .set('top_m', topM)
      .set('candidates', candidates)
      // 追踪线索用于前端展示「最终结果来自哪一步」，默认打开
      .set('trace', 'true');

    return this.http
      .get<AnswerResult>(`${this.baseUrl}/ask`, { params })
      .pipe(timeout(this.askTimeoutMs));
  }

  /** 服务与检索栈健康状态 */
  health(): Observable<HealthResponse> {
    return this.http.get<HealthResponse>(`${this.baseUrl}/health`);
  }
}

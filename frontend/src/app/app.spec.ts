import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';

import { App } from './app';

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
});

import { ApplicationConfig, provideBrowserGlobalErrorListeners, provideZoneChangeDetection } from '@angular/core';
import { provideHttpClient, withFetch } from '@angular/common/http';

export const appConfig: ApplicationConfig = {
  providers: [
    provideBrowserGlobalErrorListeners(),
    provideZoneChangeDetection({ eventCoalescing: true }),
    // 后端调用全部走 HttpClient；withFetch 让请求走 Fetch API，SSR/代理下更稳
    provideHttpClient(withFetch()),
  ],
};

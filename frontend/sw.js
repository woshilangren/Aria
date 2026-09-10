// Aria PWA Service Worker：只做"可安装"这件事，不做离线缓存——
// 聊天/语音全是实时数据，缓存只会带来旧页面（旧前端音频格式后端不认）的坑
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(clients.claim()));
self.addEventListener("fetch", (e) => {
  // 直通：所有请求照常走网络
});

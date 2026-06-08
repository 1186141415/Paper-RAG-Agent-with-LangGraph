<template>
  <div class="space-y-6">
    <div
      v-if="!messages.length && !loading"
      class="text-center py-16 text-on-surface-variant"
    >
      <span class="material-symbols-outlined text-4xl opacity-40 mb-3 block">chat</span>
      <p class="text-sm">当前会话还没有消息，先问一个问题试试。</p>
    </div>

    <div
      v-for="msg in messages"
      :key="msg.id || msg.created_at + msg.role"
    >
      <div v-if="msg.role === 'user'" class="flex justify-end">
        <div class="bg-primary text-on-primary px-5 py-3 rounded-xl shadow-sm max-w-[85%] border border-primary-container">
          <p class="text-body-md whitespace-pre-wrap">{{ msg.content }}</p>
          <p class="text-[10px] opacity-60 mt-1 text-right">{{ formatTime(msg.created_at) }}</p>
        </div>
      </div>

      <div v-else class="flex gap-3">
        <div
          class="w-8 h-8 rounded-full bg-secondary text-on-secondary flex items-center justify-center shrink-0 shadow-sm"
        >
          <span
            class="material-symbols-outlined text-[18px]"
            style="font-variation-settings: 'FILL' 1"
          >
            smart_toy
          </span>
        </div>
        <div class="bg-white border border-outline-variant p-5 rounded-xl shadow-sm flex-1 min-w-0">
          <div class="flex items-center gap-2 mb-2">
            <span class="font-bold text-secondary text-sm">AI Assistant</span>
            <span
              class="text-[10px] bg-secondary-container text-on-secondary-container px-1.5 py-0.5 rounded uppercase font-bold tracking-wider"
            >
              RAG Response
            </span>
            <span class="text-[10px] text-on-surface-variant ml-auto">
              {{ formatTime(msg.created_at) }}
            </span>
          </div>
          <div
            class="text-body-md text-on-surface leading-relaxed markdown-body"
            v-html="renderMarkdown(msg.content)"
          />
        </div>
      </div>
    </div>

    <div v-if="loading" class="flex gap-3">
      <div
        class="w-8 h-8 rounded-full bg-secondary text-on-secondary flex items-center justify-center shrink-0"
      >
        <span class="material-symbols-outlined text-[18px] agent-pulse">smart_toy</span>
      </div>
      <div class="bg-white border border-outline-variant px-5 py-4 rounded-xl shadow-sm">
        <div class="flex items-center gap-2 text-sm text-on-surface-variant">
          <span class="w-2 h-2 rounded-full bg-secondary agent-pulse" />
          正在检索并生成回答…
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { renderMarkdown } from "../utils/markdown";

defineProps({
  messages: { type: Array, default: () => [] },
  loading: { type: Boolean, default: false },
});

function formatTime(value) {
  if (!value) return "";
  return new Date(value).toLocaleString("zh-CN");
}
</script>

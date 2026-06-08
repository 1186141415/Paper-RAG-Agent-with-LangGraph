<template>
  <div v-if="!chunks?.length" class="text-sm text-on-surface-variant opacity-60 text-center py-8">
    暂无检索片段
  </div>

  <div v-else-if="variant === 'cards'" class="space-y-3">
    <div
      v-for="(chunk, idx) in chunks"
      :key="idx"
      class="bg-white border border-outline-variant rounded-lg p-4 shadow-sm relative group hover:border-secondary transition-all"
    >
      <div class="absolute top-3 right-3">
        <span
          class="bg-surface-container-high text-on-secondary-container text-[10px] font-bold px-1.5 py-0.5 rounded flex items-center gap-1"
        >
          <span class="material-symbols-outlined text-[10px]">picture_as_pdf</span>
          PDF
        </span>
      </div>
      <div class="flex items-center gap-2 mb-1 pr-14">
        <span class="text-secondary font-bold text-xs">[{{ idx + 1 }}]</span>
        <h4 class="text-xs font-bold truncate text-on-surface">{{ chunk.source }}</h4>
      </div>
      <p
        class="text-body-sm text-on-surface-variant line-clamp-4 leading-relaxed italic border-l-2 border-secondary/20 pl-3 py-0.5 mb-2"
      >
        "{{ chunk.text }}"
      </p>
      <div class="flex justify-between items-center">
        <span class="text-[10px] font-mono text-on-surface-variant opacity-60">
          Dist: {{ formatDistance(chunk.distance) }}
        </span>
        <span class="text-[10px] text-secondary font-bold uppercase tracking-tighter">
          Rank #{{ chunk.retrieval_rank ?? idx + 1 }}
        </span>
      </div>
    </div>
  </div>

  <div v-else class="space-y-3">
    <div
      v-for="(chunk, idx) in chunks"
      :key="idx"
      class="border border-outline-variant rounded-lg p-3 bg-surface-container"
    >
      <div class="flex justify-between gap-2 mb-2 text-xs">
        <strong class="text-secondary">{{ chunk.source }}</strong>
        <span class="text-on-surface-variant">
          rank #{{ chunk.retrieval_rank ?? idx + 1 }} · distance {{ formatDistance(chunk.distance) }}
        </span>
      </div>
      <p class="text-sm leading-relaxed whitespace-pre-wrap text-on-surface">{{ chunk.text }}</p>
    </div>
  </div>
</template>

<script setup>
defineProps({
  chunks: { type: Array, default: () => [] },
  variant: { type: String, default: "list" },
});

function formatDistance(value) {
  if (value == null) return "—";
  return Number(value).toFixed(4);
}
</script>

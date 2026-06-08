<template>
  <div class="max-w-4xl mx-auto space-y-6">
    <div class="bg-white border border-outline-variant rounded-xl p-5 shadow-card">
      <div class="flex flex-wrap justify-between items-start gap-4">
        <div>
          <h2 class="text-xl font-bold text-on-surface">
            {{ session?.title || session?.session_id }}
          </h2>
          <p class="text-xs text-on-surface-variant font-mono mt-1">
            Session ID: {{ sessionId }}
          </p>
        </div>
        <RouterLink
          :to="`/?session_id=${sessionId}`"
          class="inline-flex items-center gap-2 px-4 py-2 bg-secondary text-on-secondary rounded-lg font-semibold text-sm hover:opacity-90 transition-opacity"
        >
          <span class="material-symbols-outlined text-[18px]">chat</span>
          继续此会话
        </RouterLink>
      </div>
    </div>

    <div class="bg-white border border-outline-variant rounded-xl p-5 shadow-card">
      <h3 class="text-label-caps text-on-surface mb-4 flex items-center gap-2">
        <span class="material-symbols-outlined text-secondary">forum</span>
        对话记录
      </h3>
      <MessageList :messages="messages" />
    </div>

    <div
      v-for="msg in assistantWithMeta"
      :key="msg.id"
      class="bg-white border border-outline-variant rounded-xl overflow-hidden shadow-card"
    >
      <div class="px-5 py-3 bg-surface-container-high border-b border-outline-variant">
        <h4 class="text-sm font-bold text-on-surface">
          当轮 Agent Trace · {{ formatTime(msg.created_at) }}
        </h4>
      </div>
      <div class="grid lg:grid-cols-2 divide-y lg:divide-y-0 lg:divide-x divide-outline-variant">
        <div class="min-h-[200px]">
          <TracePanel :trace="msg.metadata.agent_trace" variant="terminal" />
        </div>
        <div class="p-4">
          <ChunkPanel :chunks="msg.metadata.chunks || []" variant="cards" />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import { getSession } from "../api/http";
import MessageList from "../components/MessageList.vue";
import TracePanel from "../components/TracePanel.vue";
import ChunkPanel from "../components/ChunkPanel.vue";

const route = useRoute();
const sessionId = computed(() => route.params.sessionId);
const session = ref(null);
const messages = ref([]);

const assistantWithMeta = computed(() =>
  (messages.value || []).filter(
    (m) =>
      m.role === "assistant" &&
      (m.metadata?.agent_trace || (m.metadata?.chunks || []).length)
  )
);

function formatTime(value) {
  return new Date(value).toLocaleString("zh-CN");
}

onMounted(async () => {
  const data = await getSession(sessionId.value);
  session.value = data;
  messages.value = data.messages || [];
});
</script>

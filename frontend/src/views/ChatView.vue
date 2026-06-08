<template>
  <div class="h-full flex overflow-hidden">
    <div class="flex-1 flex flex-col bg-surface overflow-hidden min-w-0">
      <header
        class="flex justify-between items-center px-6 py-2 bg-surface-container-low border-b border-outline-variant h-14 shrink-0"
      >
        <div class="flex items-center gap-3">
          <span class="text-label-caps text-primary border-b-2 border-primary py-1">
            论文问答
          </span>
          <span class="text-xs text-on-surface-variant opacity-60 hidden sm:inline">
            界面中文 · Tool / Workflow 等技术字段保留英文
          </span>
        </div>
        <div class="flex items-center gap-2">
          <span
            v-if="loading"
            class="flex items-center gap-1.5 text-xs text-on-surface-variant"
          >
            <span class="w-2 h-2 rounded-full bg-secondary agent-pulse" />
            Agent 思考中…
          </span>
        </div>
      </header>

      <div class="flex-1 overflow-y-auto p-6 custom-scrollbar">
        <div class="max-w-[800px] mx-auto">
          <MessageList :messages="messages" :loading="loading" />
        </div>
      </div>

      <div class="p-4 bg-surface border-t border-outline-variant shrink-0">
        <div class="max-w-[800px] mx-auto">
          <div v-if="!messages.length" class="flex flex-wrap gap-2 mb-3">
            <button
              v-for="sample in samples"
              :key="sample"
              type="button"
              class="text-xs px-3 py-1.5 rounded-full border border-secondary/30 bg-secondary-fixed text-on-secondary-container hover:border-secondary transition-colors"
              @click="question = sample"
            >
              {{ sample }}
            </button>
          </div>

          <div class="relative">
            <textarea
              v-model="question"
              rows="2"
              class="w-full pl-4 pr-14 py-3 bg-white border-2 border-outline-variant rounded-xl focus:border-secondary focus:ring-0 text-body-md resize-none shadow-sm transition-all"
              placeholder="输入你的问题…"
              @keydown.enter.exact.prevent="onAsk"
            />
            <button
              type="button"
              class="absolute right-2 bottom-2 p-2 bg-secondary text-on-secondary rounded-lg hover:opacity-90 transition-all shadow-sm disabled:opacity-40"
              :disabled="loading || !question.trim()"
              @click="onAsk"
            >
              <span class="material-symbols-outlined text-[20px]">send</span>
            </button>
          </div>

          <p v-if="error" class="mt-2 text-sm text-error">{{ error }}</p>

          <details class="mt-2">
            <summary class="text-xs text-on-surface-variant cursor-pointer hover:text-secondary">
              高级选项
            </summary>
            <label class="block mt-2 text-xs text-on-surface-variant">Session ID</label>
            <input
              v-model="sessionId"
              class="mt-1 w-full text-xs px-3 py-2 border border-outline-variant rounded-lg bg-white focus:border-secondary focus:ring-0"
            />
          </details>
        </div>
      </div>
    </div>

    <aside
      class="w-[380px] border-l border-outline-variant flex flex-col bg-surface-container-low shrink-0 overflow-hidden hidden lg:flex"
    >
      <div class="h-1/2 flex flex-col border-b border-outline-variant overflow-hidden min-h-0">
        <div
          class="px-4 py-2 bg-surface-container-high flex justify-between items-center border-b border-outline-variant shrink-0"
        >
          <h3 class="text-label-caps text-on-surface flex items-center gap-1">
            <span class="material-symbols-outlined text-[18px]">account_tree</span>
            AGENT TRACE
          </h3>
          <span
            v-if="latestTrace"
            class="text-[10px] font-mono bg-green-100 text-green-700 px-1.5 py-0.5 rounded font-bold"
          >
            {{ traceStatus }}
          </span>
        </div>
        <div class="flex-1 overflow-y-auto custom-scrollbar min-h-0">
          <TracePanel :trace="latestTrace" variant="terminal" />
        </div>
      </div>

      <div class="h-1/2 flex flex-col overflow-hidden min-h-0">
        <div
          class="px-4 py-2 bg-surface-container-high flex justify-between items-center border-b border-outline-variant shrink-0"
        >
          <h3 class="text-label-caps text-on-surface flex items-center gap-1">
            <span class="material-symbols-outlined text-[18px]">description</span>
            RETRIEVED CONTEXT
          </h3>
          <span class="text-[10px] text-on-surface-variant font-bold">
            {{ latestChunks.length }} RESULTS
          </span>
        </div>
        <div class="flex-1 overflow-y-auto p-4 custom-scrollbar min-h-0">
          <ChunkPanel :chunks="latestChunks" variant="cards" />
        </div>
      </div>
    </aside>

    <div
      v-if="latestTrace || latestChunks.length"
      class="lg:hidden fixed bottom-20 right-4 z-40"
    >
      <button
        type="button"
        class="flex items-center gap-2 px-4 py-2 bg-primary text-on-primary rounded-full shadow-elevated text-sm font-semibold"
        @click="showMobileInspector = !showMobileInspector"
      >
        <span class="material-symbols-outlined text-[18px]">insights</span>
        Trace / Context
      </button>
    </div>

    <div
      v-if="showMobileInspector"
      class="lg:hidden fixed inset-0 z-50 bg-black/40"
      @click.self="showMobileInspector = false"
    >
      <div
        class="absolute bottom-0 left-0 right-0 max-h-[75vh] bg-surface-container-low rounded-t-xl overflow-hidden flex flex-col"
      >
        <div class="flex border-b border-outline-variant shrink-0">
          <button
            type="button"
            class="flex-1 py-3 text-sm font-semibold"
            :class="mobileTab === 'trace' ? 'text-secondary border-b-2 border-secondary' : 'text-on-surface-variant'"
            @click="mobileTab = 'trace'"
          >
            Agent Trace
          </button>
          <button
            type="button"
            class="flex-1 py-3 text-sm font-semibold"
            :class="mobileTab === 'chunks' ? 'text-secondary border-b-2 border-secondary' : 'text-on-surface-variant'"
            @click="mobileTab = 'chunks'"
          >
            Retrieved Context
          </button>
        </div>
        <div class="flex-1 overflow-y-auto custom-scrollbar p-4">
          <TracePanel v-if="mobileTab === 'trace'" :trace="latestTrace" variant="terminal" />
          <ChunkPanel v-else :chunks="latestChunks" variant="cards" />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { askQuestion, getSession } from "../api/http";
import { useAppSessions } from "../composables/useAppSessions";
import MessageList from "../components/MessageList.vue";
import TracePanel from "../components/TracePanel.vue";
import ChunkPanel from "../components/ChunkPanel.vue";

const route = useRoute();
const router = useRouter();
const { sessions, sessionId, loadSessions, newSession, ensureSessionId } =
  useAppSessions();

const messages = ref([]);
const question = ref("");
const loading = ref(false);
const error = ref("");
const latestTrace = ref(null);
const latestChunks = ref([]);
const showMobileInspector = ref(false);
const mobileTab = ref("trace");

const samples = [
  "paper1 的主要贡献是什么？",
  "paper1 和 paper2 的方法有什么不同？",
  "What dataset is used in the experiments?",
];

const traceStatus = computed(() => {
  if (!latestTrace.value) return "";
  if (latestTrace.value.error || latestTrace.value.router_error) return "DEGRADED";
  return "STABLE";
});

async function loadMessages() {
  if (!sessionId.value) {
    messages.value = [];
    return;
  }
  try {
    const data = await getSession(sessionId.value);
    messages.value = data.messages || [];
  } catch {
    messages.value = [];
  }
}

async function onAsk() {
  if (!question.value.trim() || loading.value) return;
  error.value = "";
  loading.value = true;
  try {
    await ensureSessionId();
    const result = await askQuestion(sessionId.value, question.value.trim());
    latestTrace.value = result.agent_trace || null;
    latestChunks.value = result.chunks || [];
    mobileTab.value = latestChunks.value.length ? "chunks" : "trace";
    question.value = "";
    await loadMessages();
    await loadSessions();
  } catch (e) {
    error.value = e.message;
  } finally {
    loading.value = false;
  }
}

onMounted(async () => {
  sessionId.value = (route.query.session_id || "").toString();
  await loadSessions();
  if (!sessionId.value && sessions.value.length) {
    sessionId.value = sessions.value[0].session_id;
  }
  if (!sessionId.value) {
    await newSession();
  }
  await loadMessages();
});

watch(
  () => route.query.session_id,
  async (value) => {
    if (value && value !== sessionId.value) {
      sessionId.value = value.toString();
      await loadMessages();
    }
  }
);

watch(sessionId, async (value) => {
  if (value && route.name === "chat") {
    latestTrace.value = null;
    latestChunks.value = [];
    await loadMessages();
    router.replace({ query: { session_id: value } });
  }
});
</script>

<template>
  <div class="h-screen flex overflow-hidden bg-background text-on-surface">
    <aside
      class="flex flex-col h-full border-r bg-surface-container-lowest border-outline-variant w-sidebar-width shadow-sm shrink-0"
    >
      <div class="p-6 flex items-center gap-2">
        <img src="/logo.svg" alt="PaperPilot Logo" class="w-10 h-10" />
        <div>
          <h1 class="text-headline-md font-bold text-on-surface leading-tight">PaperPilot</h1>
          <p class="text-[10px] uppercase tracking-widest text-on-surface-variant font-bold">
            论文 RAG Agent
          </p>
        </div>
      </div>

      <div class="px-4 mb-4">
        <button
          type="button"
          class="w-full flex items-center justify-center gap-2 bg-primary text-on-primary py-2 px-4 rounded-lg font-bold hover:opacity-90 transition-opacity"
          @click="handleNewChat"
        >
          <span class="material-symbols-outlined text-[20px]">add</span>
          新建会话
        </button>
      </div>

      <nav class="flex-1 overflow-y-auto custom-scrollbar px-2">
        <div
          class="px-4 py-1 text-label-caps text-on-surface-variant opacity-60 mb-1"
        >
          最近会话
        </div>
        <button
          v-for="s in sessions"
          :key="s.session_id"
          type="button"
          class="w-full flex items-center gap-2 px-4 py-2 rounded-lg transition-colors text-left mb-0.5"
          :class="
            s.session_id === sessionId && route.name === 'chat'
              ? 'text-secondary font-bold border-r-4 border-secondary bg-surface-container-high'
              : 'text-on-surface-variant opacity-80 hover:bg-surface-container-high'
          "
          @click="switchSession(s.session_id)"
        >
          <span class="material-symbols-outlined text-[18px] shrink-0">history</span>
          <span class="truncate text-sm">{{ s.title || s.session_id }}</span>
        </button>
        <p
          v-if="!sessions.length"
          class="px-4 py-2 text-xs text-on-surface-variant opacity-60"
        >
          暂无会话
        </p>

        <div
          class="mt-6 px-4 py-1 text-label-caps text-on-surface-variant opacity-60 mb-1"
        >
          导航
        </div>
        <RouterLink
          v-for="item in navItems"
          :key="item.to"
          :to="item.to"
          class="flex items-center gap-2 px-4 py-2 rounded-lg transition-colors mb-0.5"
          :class="navLinkClass(item)"
        >
          <span class="material-symbols-outlined text-[18px]">{{ item.icon }}</span>
          <span class="text-sm">{{ item.label }}</span>
        </RouterLink>
      </nav>

      <div class="p-4 border-t border-outline-variant">
        <div
          class="flex items-center gap-2 px-3 py-2 rounded-lg text-xs"
          :class="healthPillClass"
        >
          <span
            class="w-2 h-2 rounded-full shrink-0"
            :class="healthDotClass"
          />
          <span>{{ healthLabel }}</span>
        </div>
      </div>
    </aside>

    <div class="flex-1 flex flex-col min-w-0 overflow-hidden">
      <header
        v-if="!route.meta.fullBleed"
        class="flex justify-between items-center px-6 py-3 bg-surface-container-low border-b border-outline-variant shadow-sm h-14 shrink-0"
      >
        <h2 class="text-lg font-semibold text-on-surface">{{ pageTitle }}</h2>
        <div
          class="flex items-center gap-2 px-3 py-1 rounded-full text-xs"
          :class="healthPillClass"
        >
          <span class="w-2 h-2 rounded-full" :class="healthDotClass" />
          {{ healthLabel }}
        </div>
      </header>

      <main
        class="flex-1 overflow-hidden"
        :class="route.meta.fullBleed ? '' : 'overflow-y-auto p-6 custom-scrollbar'"
      >
        <RouterView />
      </main>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import { getHealth } from "../api/http";
import { provideAppSessions } from "../composables/useAppSessions";

const route = useRoute();
const { sessions, sessionId, loadSessions, newSession, switchSession } =
  provideAppSessions();

const healthLabel = ref("检查中…");
const healthStatus = ref("checking");

const navItems = [
  { to: "/", label: "论文问答", icon: "chat", exact: true },
  { to: "/upload", label: "上传论文", icon: "upload_file" },
  { to: "/sessions", label: "历史会话", icon: "history" },
];

const pageTitle = computed(() => {
  if (route.name === "upload") return "知识库管理";
  if (route.name === "sessions") return "历史会话";
  if (route.name === "session-detail") return "会话详情";
  return "PaperPilot";
});

const healthPillClass = computed(() => {
  if (healthStatus.value === "online")
    return "bg-green-50 text-green-800 border border-green-200";
  if (healthStatus.value === "partial")
    return "bg-amber-50 text-amber-800 border border-amber-200";
  return "bg-red-50 text-red-800 border border-red-200";
});

const healthDotClass = computed(() => {
  if (healthStatus.value === "online") return "bg-green-500";
  if (healthStatus.value === "partial") return "bg-amber-500";
  return "bg-red-500";
});

function navLinkClass(item) {
  const active = item.exact
    ? route.path === item.to
    : route.path.startsWith(item.to) && item.to !== "/";
  return active
    ? "text-secondary font-bold border-r-4 border-secondary bg-surface-container-high"
    : "text-on-surface-variant opacity-80 hover:bg-surface-container-high";
}

async function handleNewChat() {
  await newSession();
}

onMounted(async () => {
  await loadSessions();
  try {
    const data = await getHealth();
    const payload = data?.bff ? data.django : data;
    const djangoOk = payload?.django === "online" || payload?.status === "ok";
    const fastapiOk = payload?.fastapi === "online";
    if (djangoOk && fastapiOk) {
      healthLabel.value = "API 在线";
      healthStatus.value = "online";
    } else if (djangoOk || fastapiOk) {
      healthLabel.value = "部分服务离线";
      healthStatus.value = "partial";
    } else {
      healthLabel.value = "服务离线";
      healthStatus.value = "offline";
    }
  } catch {
    healthLabel.value = "BFF 离线";
    healthStatus.value = "offline";
  }
});
</script>

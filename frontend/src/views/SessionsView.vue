<template>
  <div class="max-w-5xl mx-auto space-y-6">
    <div class="flex justify-between items-start gap-4">
      <div>
        <h2 class="text-headline-md font-bold text-on-surface mb-1">历史会话</h2>
        <p class="text-sm text-on-surface-variant">
          保存在 MySQL / SQLite 中的问答记录。
        </p>
      </div>
      <span class="text-xs font-bold text-on-surface-variant bg-surface-container-high px-3 py-1 rounded-full">
        {{ sessions.length }} SESSIONS
      </span>
    </div>

    <div
      v-if="sessions.length"
      class="bg-white border border-outline-variant rounded-xl shadow-card overflow-hidden"
    >
      <table class="w-full text-sm">
        <thead class="bg-surface-container-high border-b border-outline-variant">
          <tr>
            <th class="text-left px-5 py-3 text-label-caps text-on-surface-variant font-bold">
              会话
            </th>
            <th class="text-left px-5 py-3 text-label-caps text-on-surface-variant font-bold hidden sm:table-cell">
              消息数
            </th>
            <th class="text-left px-5 py-3 text-label-caps text-on-surface-variant font-bold hidden md:table-cell">
              更新时间
            </th>
            <th class="px-5 py-3" />
          </tr>
        </thead>
        <tbody class="divide-y divide-outline-variant">
          <tr
            v-for="s in sessions"
            :key="s.session_id"
            class="hover:bg-surface-container-low transition-colors group"
          >
            <td class="px-5 py-4">
              <RouterLink
                :to="`/sessions/${s.session_id}`"
                class="font-semibold text-on-surface group-hover:text-secondary transition-colors"
              >
                {{ s.title || s.session_id }}
              </RouterLink>
              <p class="text-xs text-on-surface-variant mt-0.5 font-mono truncate max-w-xs">
                {{ s.session_id }}
              </p>
            </td>
            <td class="px-5 py-4 text-on-surface-variant hidden sm:table-cell">
              {{ s.message_count }} 条
            </td>
            <td class="px-5 py-4 text-on-surface-variant hidden md:table-cell">
              {{ formatTime(s.updated_at) }}
            </td>
            <td class="px-5 py-4 text-right">
              <RouterLink
                :to="`/?session_id=${s.session_id}`"
                class="inline-flex items-center gap-1 text-xs font-bold text-secondary hover:underline"
              >
                继续
                <span class="material-symbols-outlined text-[14px]">arrow_forward</span>
              </RouterLink>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <div
      v-else
      class="bg-white border border-outline-variant rounded-xl py-16 text-center text-on-surface-variant"
    >
      <span class="material-symbols-outlined text-4xl opacity-40 mb-3 block">history</span>
      <p class="text-sm">暂无历史会话</p>
    </div>
  </div>
</template>

<script setup>
import { onMounted, ref } from "vue";
import { listSessions } from "../api/http";

const sessions = ref([]);

function formatTime(value) {
  return new Date(value).toLocaleString("zh-CN");
}

onMounted(async () => {
  const data = await listSessions();
  sessions.value = data.sessions || [];
});
</script>

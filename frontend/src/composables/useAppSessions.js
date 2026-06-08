import { inject, provide, ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import { createSession, listSessions } from "../api/http";

const KEY = Symbol("appSessions");

export function provideAppSessions() {
  const route = useRoute();
  const router = useRouter();
  const sessions = ref([]);
  const sessionId = ref("");

  async function loadSessions() {
    const data = await listSessions();
    sessions.value = data.sessions || [];
  }

  async function refreshSessions() {
    await loadSessions();
  }

  async function newSession() {
    const created = await createSession();
    sessionId.value = created.session_id;
    await loadSessions();
    router.replace({ path: "/", query: { session_id: created.session_id } });
    return created.session_id;
  }

  function switchSession(id) {
    sessionId.value = id;
    if (route.path === "/") {
      router.replace({ query: { session_id: id } });
    } else {
      router.push({ path: "/", query: { session_id: id } });
    }
  }

  async function ensureSessionId() {
    if (sessionId.value) return sessionId.value;
    if (sessions.value.length) {
      sessionId.value = sessions.value[0].session_id;
      return sessionId.value;
    }
    return newSession();
  }

  const api = {
    sessions,
    sessionId,
    loadSessions,
    refreshSessions,
    newSession,
    switchSession,
    ensureSessionId,
  };

  provide(KEY, api);
  return api;
}

export function useAppSessions() {
  const ctx = inject(KEY);
  if (!ctx) throw new Error("useAppSessions must be used inside AppShell");
  return ctx;
}

import { createRouter, createWebHistory } from "vue-router";
import ChatView from "../views/ChatView.vue";
import UploadView from "../views/UploadView.vue";
import SessionsView from "../views/SessionsView.vue";
import SessionDetailView from "../views/SessionDetailView.vue";

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", name: "chat", component: ChatView, meta: { fullBleed: true } },
    { path: "/upload", name: "upload", component: UploadView },
    { path: "/sessions", name: "sessions", component: SessionsView },
    {
      path: "/sessions/:sessionId",
      name: "session-detail",
      component: SessionDetailView,
    },
  ],
});

export default router;

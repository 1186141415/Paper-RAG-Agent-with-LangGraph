const API_BASE = import.meta.env.VITE_API_BASE || "";

async function request(path, options = {}) {
  const resp = await fetch(`${API_BASE}${path}`, options);
  const text = await resp.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { raw: text };
  }
  if (!resp.ok) {
    const message = data?.error || resp.statusText || "request failed";
    throw new Error(message);
  }
  return data;
}

export function getHealth() {
  return request("/api/health/");
}

export function listSessions() {
  return request("/api/sessions/");
}

export function createSession(title = "") {
  return request("/api/sessions/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
}

export function getSession(sessionId) {
  return request(`/api/sessions/${encodeURIComponent(sessionId)}/`);
}

export function askQuestion(sessionId, question) {
  return request("/api/chat/ask/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, question }),
  });
}

export function listDocuments() {
  return request("/api/documents/");
}

export async function uploadDocument(file) {
  const form = new FormData();
  form.append("paper_file", file);
  const resp = await fetch(`${API_BASE}/api/documents/upload/`, {
    method: "POST",
    body: form,
  });
  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(data?.error || "upload failed");
  }
  return data;
}

<template>
  <div class="max-w-4xl mx-auto space-y-6">
    <div>
      <h2 class="text-headline-md font-bold text-on-surface mb-1">上传论文</h2>
      <p class="text-sm text-on-surface-variant">
        上传 PDF 后将保存到 data/ 并触发 FastAPI /reload_kb 重建知识库。
      </p>
    </div>

    <div
      class="bg-white border-2 border-dashed rounded-xl p-10 text-center transition-colors"
      :class="dragging ? 'border-secondary bg-surface-container-high drag-over' : 'border-outline-variant bg-surface-container-low'"
      @dragover.prevent="dragging = true"
      @dragleave.prevent="dragging = false"
      @drop.prevent="onDrop"
    >
      <span class="material-symbols-outlined text-5xl text-secondary mb-3 block">cloud_upload</span>
      <p class="text-body-md text-on-surface mb-1">拖拽 PDF 到此处，或选择文件</p>
      <p class="text-xs text-on-surface-variant mb-4">仅支持 .pdf 格式</p>
      <label
        class="inline-flex items-center gap-2 px-4 py-2 bg-secondary text-on-secondary rounded-lg font-semibold cursor-pointer hover:opacity-90 transition-opacity"
      >
        <span class="material-symbols-outlined text-[18px]">folder_open</span>
        选择文件
        <input type="file" accept=".pdf" class="hidden" @change="onFileChange" />
      </label>
      <p v-if="file" class="mt-3 text-sm text-secondary font-medium">
        已选择：{{ file.name }}
      </p>
    </div>

    <div class="flex items-center gap-3">
      <button
        type="button"
        class="flex items-center gap-2 px-5 py-2.5 bg-primary text-on-primary rounded-lg font-bold hover:opacity-90 transition-opacity disabled:opacity-40"
        :disabled="!file || uploading"
        @click="onUpload"
      >
        <span class="material-symbols-outlined text-[20px]">
          {{ uploading ? "hourglass_top" : "publish" }}
        </span>
        {{ uploading ? "正在上传并重建索引…" : "上传并重建知识库" }}
      </button>
    </div>

    <p v-if="message" class="text-sm text-green-700 bg-green-50 border border-green-200 rounded-lg px-4 py-3 font-medium">
      {{ message }}
    </p>
    <p v-if="error" class="text-sm text-error bg-error-container rounded-lg px-4 py-3">
      {{ error }}
    </p>

    <div
      v-if="reloadResult"
      class="bg-green-50 border border-green-200 rounded-xl p-4 space-y-1 text-sm"
    >
      <h4 class="font-bold text-on-surface">Knowledge Base Reload Result</h4>
      <p><strong>Status:</strong> {{ reloadResult.status }}</p>
      <p><strong>Message:</strong> {{ reloadResult.message }}</p>
      <p v-if="reloadResult.total_chunks">
        <strong>Total Chunks:</strong> {{ reloadResult.total_chunks }}
      </p>
    </div>

    <div class="bg-white border border-outline-variant rounded-xl shadow-card overflow-hidden">
      <div class="px-5 py-4 bg-surface-container-high border-b border-outline-variant flex justify-between items-center">
        <h3 class="font-bold text-on-surface flex items-center gap-2">
          <span class="material-symbols-outlined text-secondary">library_books</span>
          当前知识库
        </h3>
        <span class="text-xs text-on-surface-variant font-bold">
          {{ files.length }} FILES
        </span>
      </div>
      <div v-if="files.length" class="divide-y divide-outline-variant">
        <div
          v-for="f in files"
          :key="f.name"
          class="flex justify-between items-center px-5 py-3 hover:bg-surface-container-low transition-colors"
        >
          <div class="flex items-center gap-3 min-w-0">
            <span class="material-symbols-outlined text-secondary shrink-0">picture_as_pdf</span>
            <span class="text-sm font-medium truncate">{{ f.name }}</span>
          </div>
          <span class="text-xs font-mono text-on-surface-variant shrink-0 ml-4">
            {{ formatSize(f.size_bytes) }}
          </span>
        </div>
      </div>
      <p v-else class="px-5 py-10 text-center text-sm text-on-surface-variant">
        暂无 PDF 文件
      </p>
    </div>
  </div>
</template>

<script setup>
import { onMounted, ref } from "vue";
import { listDocuments, uploadDocument } from "../api/http";

const files = ref([]);
const file = ref(null);
const dragging = ref(false);
const uploading = ref(false);
const message = ref("");
const error = ref("");
const reloadResult = ref(null);

async function refreshFiles() {
  const data = await listDocuments();
  files.value = data.files || [];
}

function onFileChange(event) {
  file.value = event.target.files?.[0] || null;
}

function onDrop(event) {
  dragging.value = false;
  const dropped = event.dataTransfer.files?.[0];
  if (dropped) file.value = dropped;
}

async function onUpload() {
  if (!file.value) return;
  uploading.value = true;
  message.value = "";
  error.value = "";
  reloadResult.value = null;
  try {
    const data = await uploadDocument(file.value);
    message.value = data.message;
    reloadResult.value = data.reload_result;
    files.value = data.files || [];
    file.value = null;
  } catch (e) {
    error.value = e.message;
  } finally {
    uploading.value = false;
  }
}

function formatSize(bytes) {
  if (!bytes) return "0 B";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

onMounted(refreshFiles);
</script>

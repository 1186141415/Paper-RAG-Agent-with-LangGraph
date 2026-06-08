<template>
  <div v-if="!trace" class="p-4 text-sm text-on-surface-variant opacity-60 text-center">
    提问后将在此显示 Agent Trace
  </div>

  <div
    v-else-if="variant === 'terminal'"
    class="p-4 font-mono text-mono-code space-y-4 bg-trace text-white/90 min-h-full"
  >
    <div class="space-y-2">
      <div class="flex justify-between items-start border-b border-white/10 pb-2 gap-2">
        <span class="text-blue-300 shrink-0">Routed Tool:</span>
        <span class="text-green-400 text-right">{{ routedTool }}</span>
      </div>
      <div class="flex justify-between items-start border-b border-white/10 pb-2 gap-2">
        <span class="text-blue-300 shrink-0">Final Tool:</span>
        <span class="text-green-400 text-right">{{ trace.tool_used || "—" }}</span>
      </div>
      <div class="space-y-1">
        <span class="text-blue-300 block">Route Reason:</span>
        <p class="text-white/70 leading-relaxed text-xs">{{ reason }}</p>
      </div>
      <div class="flex justify-between items-center bg-white/5 p-2 rounded gap-2">
        <span class="text-blue-300">Fallback Used:</span>
        <span :class="trace.fallback_used ? 'text-amber-400' : 'text-green-400'">
          {{ trace.fallback_used ? "True" : "False" }}
        </span>
      </div>
      <div class="flex justify-between items-center bg-white/5 p-2 rounded gap-2">
        <span class="text-blue-300">Context Sufficient:</span>
        <span :class="contextColor">{{ contextLabel }}</span>
      </div>
      <div v-if="trace.router_error" class="bg-red-900/40 border border-red-500/30 p-2 rounded text-xs text-red-300">
        <span class="font-bold">Router Error:</span> {{ trace.router_error }}
      </div>
      <div v-if="trace.error" class="bg-red-900/40 border border-red-500/30 p-2 rounded text-xs text-red-300">
        <span class="font-bold">Error:</span> {{ trace.error }}
      </div>
    </div>

    <div v-if="workflowSteps.length" class="pt-2">
      <span class="text-secondary-fixed opacity-60 text-[10px] mb-2 block">WORKFLOW PATH</span>
      <div class="relative pl-4 border-l border-white/20 space-y-3">
        <div
          v-for="(step, idx) in workflowSteps"
          :key="idx"
          class="relative"
        >
          <div
            class="absolute -left-[21px] top-1 w-2 h-2 rounded-full"
            :class="idx === workflowSteps.length - 1 ? 'bg-green-400 agent-pulse' : 'bg-blue-400'"
          />
          <span class="text-xs font-bold text-blue-300">{{ step }}</span>
        </div>
      </div>
    </div>

    <details v-if="trace.context_metrics" class="pt-2">
      <summary class="text-blue-300 text-xs cursor-pointer">Evidence Gate 详情</summary>
      <pre class="mt-2 text-[11px] text-white/60 overflow-x-auto">{{ formatMetrics(trace.context_metrics) }}</pre>
    </details>
  </div>

  <div v-else class="p-4 space-y-3">
    <div class="grid gap-2 text-sm">
      <div class="flex justify-between gap-2 p-2 bg-surface-container rounded border border-outline-variant">
        <span class="text-on-surface-variant text-xs">Routed Tool</span>
        <strong>{{ routedTool }}</strong>
      </div>
      <div class="flex justify-between gap-2 p-2 bg-surface-container rounded border border-outline-variant">
        <span class="text-on-surface-variant text-xs">Final Tool Executed</span>
        <strong>{{ trace.tool_used || "—" }}</strong>
      </div>
      <div class="p-2 bg-surface-container rounded border border-outline-variant">
        <span class="text-on-surface-variant text-xs block mb-1">Route Reason</span>
        <span class="text-sm">{{ reason }}</span>
      </div>
      <div class="flex justify-between gap-2 p-2 bg-surface-container rounded border border-outline-variant">
        <span class="text-on-surface-variant text-xs">Workflow Path</span>
        <span class="text-sm text-right">{{ workflow }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from "vue";

const props = defineProps({
  trace: { type: Object, default: null },
  variant: { type: String, default: "card" },
});

const routedTool = computed(
  () => props.trace?.routed_tool || props.trace?.route_decision?.tool || "—"
);
const reason = computed(
  () => props.trace?.route_decision?.reason || "No reason provided."
);
const workflow = computed(() => (props.trace?.workflow || []).join(" → "));
const workflowSteps = computed(() => props.trace?.workflow || []);

const contextLabel = computed(() => {
  if (props.trace?.context_sufficient === true) return "Passed";
  if (props.trace?.context_sufficient === false) return "Failed";
  return "N/A";
});

const contextColor = computed(() => {
  if (props.trace?.context_sufficient === true) return "text-green-400";
  if (props.trace?.context_sufficient === false) return "text-red-400";
  return "text-white/60";
});

function formatMetrics(value) {
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}
</script>

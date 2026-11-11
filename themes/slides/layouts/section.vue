<script setup lang="ts">
// kicker is the small line above the title, such as "part 2".
defineProps<{ kicker?: string }>();
</script>

<template>
  <div class="slidev-layout section">
    <p v-if="kicker" class="kicker">{{ kicker }}</p>
    <slot />
  </div>
</template>

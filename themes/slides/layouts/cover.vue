<script setup lang="ts">
import { site } from "../../../src/site.ts";

// event and date sit above the title; the QR code or logo goes in the `::aside::` slot.
defineProps<{ event?: string; date?: string; author?: string }>();
</script>

<template>
  <div class="slidev-layout cover">
    <p v-if="event || date" class="kicker">{{ [event, date].filter(Boolean).join(" · ") }}</p>
    <div class="title"><slot /></div>
    <p class="byline">{{ author ?? site.name }}<i>.</i></p>
    <div v-if="$slots.aside" class="aside"><slot name="aside" /></div>
  </div>
</template>

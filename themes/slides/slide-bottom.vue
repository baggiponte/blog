<script setup lang="ts">
import { site } from "../../src/site.ts";
</script>

<!-- Every slide but the cover gets the brand and its number; `footer: false` in a slide's frontmatter drops it. -->
<template>
  <footer v-if="$frontmatter.layout !== 'cover' && $frontmatter.footer !== false" class="slide-foot">
    <b>{{ site.name }}<i>.</i></b>
    <span>{{ $page }}</span>
  </footer>
</template>

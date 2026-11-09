<!--
  NavLogo.vue
  mosaic

  Created by Christian Bator on 11/08/2026
-->

<script setup lang="ts">
import { VPImage } from "vitepress/theme";
</script>

<!-- The wordmark on pages with a sidebar, and the icon alone elsewhere (see custom.css) -->
<template>
  <VPImage class="nav-icon" :image="{ src: '/icon.png', alt: 'Mosaic' }" />
  <VPImage class="nav-wordmark" :image="{ light: '/logo.png', dark: '/logo-dark-mode.png', alt: 'Mosaic' }" />
</template>

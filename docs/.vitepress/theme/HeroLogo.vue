<!--
  HeroLogo.vue
  mosaic

  Created by Christian Bator on 11/08/2026
-->

<script setup lang="ts">
import { VPImage } from "vitepress/theme";
</script>

<!-- Sized in custom.css: VPImage renders the light and dark images itself, so scoped styles don't reach them -->
<template>
  <VPImage class="hero-logo" :image="{ light: '/logo.png', dark: '/logo-dark-mode.png', alt: 'Mosaic' }" />
</template>

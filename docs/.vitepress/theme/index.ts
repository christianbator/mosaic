//
// index.ts
// mosaic
//
// Created by Christian Bator on 11/08/2026
//

import { h } from "vue";
import DefaultTheme from "vitepress/theme";

import HeroLogo from "./HeroLogo.vue";
import NavLogo from "./NavLogo.vue";
import "./custom.css";

// The default theme, with Mosaic's logos in the nav bar and in place of the home page's title
export default {
  extends: DefaultTheme,
  Layout: () =>
    h(DefaultTheme.Layout, null, {
      "nav-bar-title-before": () => h(NavLogo),
      "home-hero-info-before": () => h(HeroLogo),
    }),
};

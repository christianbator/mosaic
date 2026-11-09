//
// config.mts
// mosaic
//
// Created by Christian Bator on 11/08/2026
//

import { defineConfig } from "vitepress";

// Written by `pixi run docs-api` from the docstrings
import apiSidebar from "./api-sidebar.json";

// "/" for a custom domain; "/mosaic/" when served from christianbator.github.io/mosaic
const base = process.env.DOCS_BASE ?? "/";

export default defineConfig({
  title: "Mosaic",
  description: "An image processing library in Mojo",
  base,

  cleanUrls: true,
  lastUpdated: true,

  // Links in head aren't prefixed with the base automatically
  head: [["link", { rel: "icon", type: "image/png", href: `${base}icon.png` }]],

  markdown: {
    math: true,
  },

  themeConfig: {
    // The nav bar logo is theme/NavLogo.vue
    siteTitle: false,

    nav: [
      { text: "Guide", link: "/guide/getting-started", activeMatch: "/guide/" },
      { text: "API", link: "/api/", activeMatch: "/api/" },
      { text: "Write-ups", link: "/writeups/", activeMatch: "/writeups/" },
    ],

    sidebar: {
      "/guide/": [
        {
          text: "Guide",
          items: [{ text: "Getting Started", link: "/guide/getting-started" }],
        },
      ],
      "/api/": [{ text: "API Reference", link: "/api/" }, ...apiSidebar],
      // Add each write-up here as it's published, newest first
      "/writeups/": [
        {
          text: "Write-ups",
          items: [{ text: "All Write-ups", link: "/writeups/" }],
        },
      ],
    },

    socialLinks: [{ icon: "github", link: "https://github.com/christianbator/mosaic" }],

    search: { provider: "local" },

    outline: { level: [2, 3] },

    footer: {
      message: "Released under the Apache 2.0 License.",
    },
  },
});

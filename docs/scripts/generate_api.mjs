//
// generate_api.mjs
// mosaic
//
// Created by Christian Bator on 11/08/2026
//

// Turns the JSON from `mojo doc` into VitePress pages: one page per module under docs/api/, an index, and a sidebar
//
// Usage: node docs/scripts/generate_api.mjs <mojo-doc.json>

import { mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const docsDir = join(dirname(fileURLToPath(import.meta.url)), "..");
const apiDir = join(docsDir, "api");
const sidebarPath = join(docsDir, ".vitepress", "api-sidebar.json");

// Dunder methods shown as methods; the rest are listed as operators or left out
const documentedDunders = new Set(["__init__", "__getitem__", "__setitem__", "__contains__"]);

// Traits every struct gets implicitly
const implicitTraits = new Set(["AnyType", "Deinitable"]);

const operators = {
  __add__: "+", __sub__: "-", __mul__: "*", __truediv__: "/", __floordiv__: "//", __mod__: "%", __pow__: "**",
  __matmul__: "@", __and__: "&", __or__: "|", __xor__: "^", __lshift__: "<<", __rshift__: ">>", __neg__: "-x",
  __invert__: "~", __eq__: "==", __ne__: "!=", __lt__: "<", __le__: "<=", __gt__: ">", __ge__: ">=",
};

//
// Filtering
//
const isDunder = (name) => name.startsWith("__") && name.endsWith("__");
const isPublic = (name) => !name.startsWith("_") || isDunder(name);

function isDocumentedOverload(overload) {
  // Hides constructors for internal use (keyword arguments starting with an underscore) and move constructors
  return !overload.args.some((arg) => arg.name.startsWith("_") || arg.convention === "deinit");
}

function documentedFunctions(functions) {
  return functions
    .filter((f) => isPublic(f.name) && (!isDunder(f.name) || documentedDunders.has(f.name)))
    .map((f) => ({ ...f, overloads: f.overloads.filter(isDocumentedOverload) }))
    .filter((f) => f.overloads.length > 0);
}

function hasContent(module) {
  return (
    module.structs.some((s) => isPublic(s.name)) ||
    documentedFunctions(module.functions).length > 0 ||
    module.aliases.some((a) => isPublic(a.name))
  );
}

//
// Brackets
//
const openBrackets = "([{";
const closeBrackets = ")]}";

function matchingBracket(text, open) {
  // The index of the bracket that closes the one at `open`
  let depth = 0;

  for (let i = open; i < text.length; i++) {
    if (openBrackets.includes(text[i])) {
      depth++;
    } else if (closeBrackets.includes(text[i]) && --depth === 0) {
      return i;
    }
  }

  return text.length - 1;
}

function topLevelIndex(text, token) {
  // The index of the first `token` outside any brackets, or -1
  let depth = 0;

  for (let i = 0; i < text.length; i++) {
    if (depth === 0 && text.startsWith(token, i)) {
      return i;
    }

    if (openBrackets.includes(text[i])) {
      depth++;
    } else if (closeBrackets.includes(text[i])) {
      depth--;
    }
  }

  return -1;
}

function splitTopLevel(text) {
  // Splits on the commas outside any brackets
  const items = [];
  let depth = 0;
  let start = 0;

  for (let i = 0; i < text.length; i++) {
    if (openBrackets.includes(text[i])) {
      depth++;
    } else if (closeBrackets.includes(text[i])) {
      depth--;
    } else if (text[i] === "," && depth === 0) {
      items.push(text.slice(start, i).trim());
      start = i + 1;
    }
  }

  const last = text.slice(start).trim();

  return last ? [...items, last] : items;
}

//
// Signatures
//

// Signatures longer than this are broken across lines, one parameter or argument per line, like `pixi run format`
const maxLineLength = 120;
const indentUnit = "    ";

const prefixOperators = { add: "+", sub: "-", mul: "*" };

function withRaises(text) {
  // mojo doc reports raising separately, so it goes back in after the argument list
  const open = topLevelIndex(text, "(");

  if (open < 0) {
    return text;
  }

  const close = matchingBracket(text, open);

  return `${text.slice(0, close + 1)} raises${text.slice(close + 1)}`;
}

function rewriteArithmetic(text) {
  // mojo doc prints compile-time arithmetic in prefix form, like Int((add (mul a, b), c)), rewritten as (a * b) + c
  for (let match; (match = /\((add|sub|mul) /.exec(text)); ) {
    const close = matchingBracket(text, match.index);
    const operands = splitTopLevel(text.slice(match.index + match[0].length, close));
    const expression = `(${operands.join(` ${prefixOperators[match[1]]} `)})`;

    text = text.slice(0, match.index) + expression + text.slice(close + 1);
  }

  return unwrapInts(text);
}

function unwrapInts(text) {
  // Int(1) becomes 1, and Int((a * b)) becomes a * b
  const pattern = /(?<![\w.])Int\(/g;

  for (let match; (match = pattern.exec(text)); ) {
    const open = match.index + match[0].length - 1;
    const close = matchingBracket(text, open);
    const inner = text.slice(open + 1, close);

    let unwrapped = null;

    if (/^-?\d+$/.test(inner)) {
      unwrapped = inner;
    } else if (inner.startsWith("(") && matchingBracket(inner, 0) === inner.length - 1) {
      unwrapped = inner.slice(1, -1);
    }

    if (unwrapped !== null) {
      text = text.slice(0, match.index) + unwrapped + text.slice(close + 1);
      pattern.lastIndex = match.index;
    }
  }

  return text;
}

function splitSignature(text) {
  // `def name[parameters](arguments) rest`, with the parameters and arguments as lists
  const start = text.search(/[[(]/);

  if (start < 0) {
    return null;
  }

  let rest = text.slice(start);

  const takeGroup = (open) => {
    if (rest[0] !== open) {
      return null;
    }

    const close = matchingBracket(rest, 0);
    const items = splitTopLevel(rest.slice(1, close));
    rest = rest.slice(close + 1);

    return items;
  };

  const parameters = takeGroup("[");
  const args = takeGroup("(");

  return { head: text.slice(0, start), parameters, args, rest };
}

const fits = (text) => text.split("\n").every((line) => line.length <= maxLineLength);

function broken(prefix, items, suffix, indent) {
  // `prefix`, then one item per line (each wrapped further if it's still too long), then `suffix`
  const lines = items.map((item) => wrapItem(item, indent + indentUnit) + ",");

  return [prefix, ...lines, indent + suffix].join("\n");
}

function wrapItem(item, indent) {
  // Breaks the longest bracket group of an item that's too long, like `def[...](a, b) -> T` across its arguments
  if (indent.length + item.length <= maxLineLength) {
    return indent + item;
  }

  let group = null;

  for (let i = 0; i < item.length; i++) {
    if ("([".includes(item[i])) {
      const close = matchingBracket(item, i);

      if (!group || close - i > group.close - group.open) {
        group = { open: i, close };
      }

      i = close;
    }
  }

  const items = group ? splitTopLevel(item.slice(group.open + 1, group.close)) : [];

  if (items.length === 0) {
    return indent + item;
  }

  return broken(indent + item.slice(0, group.open + 1), items, item.slice(group.close), indent);
}

function breakReturnType(text) {
  // Breaks the bracket list of a long return type, as in `-> MatrixSlice[...]`
  const arrow = topLevelIndex(text, "-> ");
  const open = arrow < 0 ? -1 : text.indexOf("[", arrow);

  if (open < 0) {
    return text;
  }

  const close = matchingBracket(text, open);

  return broken(text.slice(0, open + 1), splitTopLevel(text.slice(open + 1, close)), text.slice(close), "");
}

function formatSignature(text, raises = false) {
  text = rewriteArithmetic(raises ? withRaises(text) : text);

  const parts = splitSignature(text);

  if (fits(text) || !parts) {
    return text;
  }

  const { head, parameters, args, rest } = parts;
  const inlineArgs = args === null ? "" : `(${args.join(", ")})`;

  function layout(breakParameters) {
    let parametersText = "";

    if (parameters !== null) {
      parametersText = breakParameters ? broken("[", parameters, "]", "") : `[${parameters.join(", ")}]`;
    }

    // Arguments stay on one line when they fit after the parameters, or when there's only one
    const lastLine = (head + parametersText).split("\n").at(-1);
    const argsFit = (lastLine + inlineArgs + rest).length <= maxLineLength;
    const argsText = args !== null && args.length > 1 && !argsFit ? broken("(", args, ")", "") : inlineArgs;

    const result = head + parametersText + argsText + rest;

    return fits(result) ? result : breakReturnType(result);
  }

  // Parameters are broken across lines only when the signature doesn't fit otherwise
  const result = layout(false);

  return fits(result) || parameters === null ? result : layout(true);
}

//
// Rendering
//
const code = (text) => "```mojo\n" + text + "\n```";
const inline = (text) => "`" + text.replaceAll("`", "'") + "`";
const listItem = (name, summary) => `- ${inline(name)}${summary ? ": " + summary : ""}`;

function renderFunctions(functions, level) {
  const heading = "#".repeat(level);

  return functions.map((f) => {
    const summaries = new Set(f.overloads.map((o) => o.summary));
    const lines = [`${heading} ${inline(f.name)}`];

    if (summaries.size === 1) {
      // Overloads that do the same thing share one block and one description
      const signatures = f.overloads.map((o) => formatSignature(o.signature, o.raises));
      lines.push(code(signatures.join("\n\n")), f.overloads[0].summary);
    } else {
      for (const overload of f.overloads) {
        lines.push(code(formatSignature(overload.signature, overload.raises)), overload.summary);
      }
    }

    return lines.filter(Boolean).join("\n\n");
  });
}

function renderAliases(aliases, heading) {
  const visible = aliases.filter((a) => isPublic(a.name));

  return visible.length ? [heading, visible.map((a) => listItem(a.name, a.summary)).join("\n")] : [];
}

function renderStruct(struct) {
  const lines = [`## ${struct.name}`, code(formatSignature(struct.signature)), struct.summary];

  const traits = (struct.parentTraits ?? []).map((t) => t.name).filter((name) => !implicitTraits.has(name));

  if (traits.length) {
    lines.push(`Implements ${traits.map(inline).join(", ")}.`);
  }

  const structOperators = [...new Set(struct.functions.map((f) => operators[f.name]).filter(Boolean))];

  if (structOperators.length) {
    lines.push(`Operators: ${structOperators.map(inline).join(" ")}`);
  }

  const fields = struct.fields.filter((f) => isPublic(f.name));

  if (fields.length) {
    lines.push("### Fields", fields.map((f) => listItem(`${f.name}: ${f.type}`, f.summary)).join("\n"));
  }

  lines.push(...renderAliases(struct.aliases, "### Constants"));

  const methods = documentedFunctions(struct.functions);

  if (methods.length) {
    lines.push("### Methods", ...renderFunctions(methods, 4));
  }

  return lines.filter(Boolean).join("\n\n");
}

function renderModule(packagePath, module) {
  const lines = [
    `# ${module.name}`,
    `Module ${inline(`${packagePath}.${module.name}`)}`,
    module.summary,
    module.description,
    ...renderAliases(module.aliases, "## Constants"),
    ...module.structs.filter((s) => isPublic(s.name)).map(renderStruct),
    ...renderFunctions(documentedFunctions(module.functions), 2),
  ];

  return lines.filter(Boolean).join("\n\n") + "\n";
}

//
// Packages
//
function collectPackages(pkg, path = pkg.name) {
  const init = pkg.modules.find((m) => m.name === "__init__");
  const modules = pkg.modules.filter((m) => m.name !== "__init__" && hasContent(m));
  const children = (pkg.packages ?? []).flatMap((child) => collectPackages(child, `${path}.${child.name}`));

  return [{ path, summary: init?.summary ?? "", modules }, ...children];
}

function moduleLabel(module) {
  // Modules holding a single struct are named after it in the sidebar
  const structs = module.structs.filter((s) => isPublic(s.name));

  return structs.length === 1 ? structs[0].name : module.name;
}

function main() {
  const input = process.argv[2];

  if (!input) {
    throw new Error("Usage: node docs/scripts/generate_api.mjs <mojo-doc.json>");
  }

  const [root, ...packages] = collectPackages(JSON.parse(readFileSync(input, "utf8")).decl);

  rmSync(apiDir, { recursive: true, force: true });

  const sidebar = [];
  const index = ["# API Reference", root.summary];

  for (const pkg of [root, ...packages].filter((p) => p.modules.length > 0)) {
    const relative = pkg.path.split(".").slice(1).join("/");
    const items = [];

    mkdirSync(join(apiDir, relative), { recursive: true });

    for (const module of pkg.modules) {
      const link = `/api/${relative}/${module.name}`;
      writeFileSync(join(apiDir, relative, `${module.name}.md`), renderModule(pkg.path, module));
      items.push({ text: moduleLabel(module), link });
    }

    const entries = pkg.modules.map((module, i) => `- [${items[i].text}](${items[i].link}): ${module.summary}`);

    index.push(`## ${pkg.path}`, pkg.summary, entries.join("\n"));
    sidebar.push({ text: pkg.path, items });
  }

  writeFileSync(join(apiDir, "index.md"), index.filter(Boolean).join("\n\n") + "\n");
  writeFileSync(sidebarPath, JSON.stringify(sidebar, null, 2) + "\n");

  console.log(`API reference written to ${apiDir}`);
}

main();

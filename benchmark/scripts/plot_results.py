#
# plot_results.py
# mosaic
#
# Created by Christian Bator on 11/08/2026
#

import csv
import html
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from string import Template

import altair as alt

#
# Configuration
#
INPUT_DIR = Path("benchmark/input")
OUTPUT_DIR = Path("benchmark/output")

LIBRARIES = ["mosaic", "opencv"]
LIBRARY_NAMES = {"mosaic": "Mosaic", "opencv": "OpenCV"}
GPU_API_NAMES = {"metal": "Metal", "opencl": "OpenCL", "cuda": "CUDA", "hip": "HIP"}

END_TO_END = "End to end: host image in, host image out"
RESIDENT = "Resident: image already on the GPU"
MEASUREMENTS = [END_TO_END, RESIDENT]

# The before and after images on the page use the input closest to this size
SHOWCASE_MEGAPIXELS = 2.0

#
# Style
#
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"

# Categorical palette in a fixed order, validated for color vision deficiencies on adjacent pairs
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

# Solid for end to end, dashed for resident
MEASUREMENT_DASHES = [[1, 0], [6, 4]]

# Vega expressions for axis labels: 0.06, 0.5, 1, 40.5, 1,000
MEGAPIXELS_LABEL = "datum.value < 1 ? format(datum.value, '.2~f') : format(datum.value, '.3~g')"
MILLISECONDS_LABEL = "datum.value < 1 ? format(datum.value, '.2~f') : format(datum.value, ',~r')"


#
# Results
#
def load_results(operation: str) -> list[dict]:
    # Results are named <operation>_<library>.csv
    paths = sorted(OUTPUT_DIR.glob(f"{operation}_*.csv"))

    if not paths:
        raise RuntimeError(f"No results found for {operation} in {OUTPUT_DIR}")

    results = []

    for path in paths:
        with open(path, newline="") as file:
            for row in csv.DictReader(file):
                device, implementation, measurement = describe(row["library"], row["mode"])
                results.append(
                    {
                        "library": row["library"],
                        "device": device,
                        "implementation": implementation,
                        "measurement": measurement,
                        "width": int(row["width"]),
                        "height": int(row["height"]),
                        "megapixels": float(row["megapixels"]),
                        "milliseconds": float(row["milliseconds"]),
                    }
                )

    return results


def describe(library: str, mode: str) -> tuple[str, str, str]:
    # Modes are named <device> (<detail>): "cpu (10 threads)", "metal (end-to-end)", "opencl (resident)"
    device, detail = mode.removesuffix(")").split(" (", 1)
    library_name = LIBRARY_NAMES.get(library, library)

    # CPU results are always end to end
    if device == "cpu":
        return "cpu", f"{library_name} · CPU, {detail}", END_TO_END

    implementation = f"{library_name} · GPU ({GPU_API_NAMES.get(device, device)})"

    return "gpu", implementation, RESIDENT if detail == "resident" else END_TO_END


def implementation_colors(results: list[dict]) -> dict[str, str]:
    # Colors follow a fixed order by kind of implementation, not by speed: GPU before CPU, then by library, so the
    # comparison that matters most (the GPU implementations) gets the first, most distinct pair of colors
    def order(result: dict) -> tuple:
        library = LIBRARIES.index(result["library"]) if result["library"] in LIBRARIES else len(LIBRARIES)

        return (result["device"] != "gpu", library, result["implementation"])

    implementations = list(dict.fromkeys(result["implementation"] for result in sorted(results, key=order)))

    if len(implementations) > len(SERIES_COLORS):
        raise RuntimeError(f"Too many implementations to color distinctly: {len(implementations)} (maximum {len(SERIES_COLORS)})")

    return dict(zip(implementations, SERIES_COLORS))


def legend_order(results: list[dict]) -> list[str]:
    # Top to bottom in the order the lines end at the largest size, so the legend reads like the chart
    largest = max(result["megapixels"] for result in results)
    end_to_end = {
        result["implementation"]: result["milliseconds"]
        for result in results
        if result["megapixels"] == largest and result["measurement"] == END_TO_END
    }

    return sorted(end_to_end, key=end_to_end.get, reverse=True)


def tick_positions(results: list[dict]) -> list[float]:
    megapixels = sorted({result["megapixels"] for result in results})

    # Keep the largest size, then drop any size too close to the previous tick for its label to fit
    ticks = [megapixels[-1]]

    for value in reversed(megapixels[:-1]):
        if ticks[-1] / value >= 1.5:
            ticks.append(value)

    return sorted(ticks)


def log_domain(values: list[float], padding: float = 1.2) -> list[float]:
    # Fit the data with a small margin, instead of rounding out to whole powers of ten
    return [min(values) / padding, max(values) * padding]


def powers_of_ten(domain: list[float]) -> list[float]:
    low, high = math.floor(math.log10(domain[0])), math.ceil(math.log10(domain[1]))

    return [10.0**exponent for exponent in range(low, high + 1) if domain[0] <= 10.0**exponent <= domain[1]]


#
# Plotting
#
def machine_name() -> str:
    if platform.system() == "Darwin":
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()

    return platform.processor() or platform.machine()


def title(operation: str) -> str:
    return f"{operation.replace('_', ' ').capitalize()} on {machine_name()}"


def plot(operation: str, results: list[dict]) -> alt.Chart:
    colors = implementation_colors(results)
    order = legend_order(results)
    megapixels_domain = log_domain([result["megapixels"] for result in results])
    milliseconds_domain = log_domain([result["milliseconds"] for result in results])

    base = alt.Chart(alt.Data(values=results)).encode(
        x=alt.X(
            "megapixels:Q",
            title="Image size (megapixels)",
            scale=alt.Scale(type="log", domain=megapixels_domain, nice=False),
            axis=alt.Axis(values=tick_positions(results), labelExpr=MEGAPIXELS_LABEL),
        ),
        y=alt.Y(
            "milliseconds:Q",
            title="Time per image (ms, lower is better)",
            scale=alt.Scale(type="log", domain=milliseconds_domain, nice=False),
            axis=alt.Axis(values=powers_of_ten(milliseconds_domain), labelExpr=MILLISECONDS_LABEL),
        ),
        color=alt.Color(
            "implementation:N",
            title="Implementation",
            scale=alt.Scale(domain=order, range=[colors[implementation] for implementation in order]),
            legend=alt.Legend(orient="right", symbolType="stroke"),
        ),
    )

    lines = base.mark_line(strokeWidth=2, strokeCap="round", strokeJoin="round").encode(
        strokeDash=alt.StrokeDash(
            "measurement:N",
            title="Measurement",
            scale=alt.Scale(domain=MEASUREMENTS, range=MEASUREMENT_DASHES),
            legend=alt.Legend(orient="right", symbolType="stroke", symbolStrokeColor=TEXT_SECONDARY),
        ),
    )

    # Points are their own layer, so the surface-colored ring around them doesn't take the line's dashes or leak into
    # the measurement legend's symbols
    points = base.mark_point(filled=True, opacity=1, size=64, stroke=SURFACE, strokeWidth=2).encode(
        tooltip=[
            alt.Tooltip("implementation:N", title="Implementation"),
            alt.Tooltip("measurement:N", title="Measurement"),
            alt.Tooltip("megapixels:Q", title="Megapixels", format=".2f"),
            alt.Tooltip("milliseconds:Q", title="Time (ms)", format=",.1f"),
        ],
    )

    return (
        alt.layer(lines, points)
        .properties(
            title=alt.Title(
                title(operation),
                subtitle="Radius 7, color and space standard deviations 80, RGB uint8, median of 5 runs",
                offset=20,
            ),
            width=640,
            height=420,
            background=SURFACE,
            padding=20,
        )
        .configure_view(stroke=None)
        .configure_title(
            color=TEXT_PRIMARY,
            subtitleColor=TEXT_SECONDARY,
            anchor="start",
            fontSize=16,
            fontWeight="normal",
            subtitleFontSize=12,
            subtitlePadding=6,
        )
        .configure_axis(
            gridColor=GRIDLINE,
            domainColor=GRIDLINE,
            tickSize=0,
            labelColor=TEXT_SECONDARY,
            titleColor=TEXT_SECONDARY,
            titleFontWeight="normal",
            labelPadding=6,
        )
        .configure_legend(
            titleColor=TEXT_SECONDARY,
            titleFontWeight="normal",
            titleFontSize=12,
            titlePadding=8,
            labelColor=TEXT_PRIMARY,
            labelFontSize=12,
            labelLimit=320,
            symbolSize=360,
            symbolStrokeWidth=2.5,
            rowPadding=7,
            padding=12,
            offset=24,
        )
    )


#
# Page
#
def showcase_paths(operation: str) -> tuple[Path, Path] | None:
    # The input closest to the showcase size that has a mosaic output
    candidates = []

    for input_path in INPUT_DIR.glob("*.png"):
        output_path = OUTPUT_DIR / f"{input_path.stem}_{operation}_mosaic.png"
        width, height = input_path.stem.rsplit("_", 1)[1].split("x")

        if output_path.exists():
            megapixels = int(width) * int(height) / 1e6
            candidates.append((abs(math.log(megapixels / SHOWCASE_MEGAPIXELS)), input_path, output_path))

    if not candidates:
        return None

    _, input_path, output_path = min(candidates)

    return input_path, output_path


def headline(results: list[dict]) -> list[dict]:
    # At the largest size, the fastest mosaic and OpenCV result for each measurement
    largest = max(result["megapixels"] for result in results)
    tiles = []

    for measurement in MEASUREMENTS:
        fastest = {}

        for result in results:
            if result["megapixels"] == largest and result["measurement"] == measurement:
                current = fastest.get(result["library"])

                if current is None or result["milliseconds"] < current["milliseconds"]:
                    fastest[result["library"]] = result

        if "mosaic" in fastest and "opencv" in fastest:
            tiles.append({"measurement": measurement, "mosaic": fastest["mosaic"], "opencv": fastest["opencv"]})

    return tiles


def headline_html(results: list[dict]) -> str:
    tiles = []

    for tile in headline(results):
        mosaic, opencv = tile["mosaic"], tile["opencv"]
        speedup = opencv["milliseconds"] / mosaic["milliseconds"]
        comparison = f"{speedup:.2f}× faster" if speedup >= 1 else f"{1 / speedup:.2f}× slower"
        name, description = tile["measurement"].split(": ")

        tiles.append(
            f"""<div class="tile">
  <div class="tile-label">{html.escape(name)}, {mosaic['megapixels']:.1f} MP <span class="muted">· {html.escape(description)}</span></div>
  <div class="tile-value">{mosaic['milliseconds']:,.1f} <span class="unit">ms</span> <span class="speedup">{comparison}</span></div>
  <div class="tile-detail">{html.escape(mosaic['implementation'])}, against
    {html.escape(opencv['implementation'])} at {opencv['milliseconds']:,.1f} ms</div>
</div>"""
        )

    return "\n".join(tiles)


def table_html(results: list[dict]) -> str:
    # One row per image size, with a group of columns per measurement and one column per implementation in legend order
    # The fastest time is marked within each measurement only, since end to end and resident measure different work
    order = legend_order(results)
    groups = []

    for measurement in MEASUREMENTS:
        implementations = [
            implementation
            for implementation in order
            if any(r["implementation"] == implementation and r["measurement"] == measurement for r in results)
        ]

        if implementations:
            groups.append((measurement, implementations))

    sizes = sorted({(r["megapixels"], r["width"], r["height"]) for r in results})
    times = {(r["megapixels"], r["implementation"], r["measurement"]): r["milliseconds"] for r in results}

    group_header = "".join(
        f'<th colspan="{len(implementations)}" class="group">{html.escape(measurement.split(":")[0])}</th>'
        for measurement, implementations in groups
    )
    implementation_header = "".join(
        f'<th class="{"group-start" if index == 0 else ""}">{html.escape(implementation)}</th>'
        for _, implementations in groups
        for index, implementation in enumerate(implementations)
    )
    rows = []

    for megapixels, width, height in sizes:
        cells = []

        for measurement, implementations in groups:
            values = [times.get((megapixels, implementation, measurement)) for implementation in implementations]
            fastest = min((value for value in values if value is not None), default=None)
            cells += [cell(value, value == fastest, index == 0) for index, value in enumerate(values)]

        rows.append(f'<tr><th>{width}×{height}<br><span class="muted">{megapixels:.2f} MP</span></th>{"".join(cells)}</tr>')

    return (
        "<table>\n<thead>\n"
        f'<tr><th rowspan="2">Size</th>{group_header}</tr>\n'
        f"<tr>{implementation_header}</tr>\n"
        "</thead>\n<tbody>\n" + "\n".join(rows) + "\n</tbody>\n</table>"
    )


def cell(value: float | None, fastest: bool, group_start: bool) -> str:
    classes = " ".join(name for name, applies in [("fastest", fastest), ("group-start", group_start)] if applies)
    text = "" if value is None else f"{value:,.1f}"

    return f'<td class="{classes}">{text}</td>'


def comparison_html(operation: str) -> str:
    paths = showcase_paths(operation)

    if paths is None:
        return '<p class="muted">No Mosaic output images found.</p>'

    input_path, output_path = paths
    width, height = input_path.stem.rsplit("_", 1)[1].split("x")

    # The page lives in the output directory, so the input is one directory up
    before = f"../{INPUT_DIR.name}/{input_path.name}"
    after = output_path.name

    return f"""<div class="compare" style="aspect-ratio: {width} / {height}">
  <img src="{html.escape(after)}" alt="Bilateral filtered macaw (Mosaic output)">
  <img class="before" src="{html.escape(before)}" alt="Original macaw">
  <div class="divider" aria-hidden="true"></div>
  <span class="badge left">Original</span>
  <span class="badge right">Filtered</span>
  <input type="range" min="0" max="100" value="50" step="0.1" aria-label="Drag to compare the original and filtered images">
</div>
<p class="caption">{width}×{height}. Drag the slider to compare. The filtered image is Mosaic's GPU output.</p>"""


PAGE = Template("""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$title</title>
<script src="https://cdn.jsdelivr.net/npm/vega@$vega_version"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@$vegalite_version"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@$vegaembed_version"></script>
<style>
  :root {
    --surface: $surface;
    --text: $text_primary;
    --text-secondary: $text_secondary;
    --rule: $gridline;
    --accent: #2a78d6;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 40px 16px 64px;
    background: var(--surface);
    color: var(--text);
    font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  }
  main { max-width: 1120px; margin: 0 auto; }
  h1 { font-size: 26px; font-weight: 600; margin: 0 0 4px; letter-spacing: -0.01em; }
  h2 { font-size: 18px; font-weight: 600; margin: 48px 0 12px; }
  .muted { color: var(--text-secondary); font-weight: normal; }
  .lede { color: var(--text-secondary); margin: 0 0 24px; }
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 32px; }
  .tile { border: 1px solid var(--rule); border-radius: 10px; padding: 16px 18px; background: #fff; }
  .tile-label { font-size: 13px; }
  .tile-value { font-size: 34px; font-weight: 600; margin: 4px 0; font-variant-numeric: tabular-nums; }
  .tile-value .unit { font-size: 18px; font-weight: normal; color: var(--text-secondary); }
  .tile-value .speedup { font-size: 15px; font-weight: 600; margin-left: 8px; }
  .tile-detail { font-size: 13px; color: var(--text-secondary); }
  .chart { overflow-x: auto; border: 1px solid var(--rule); border-radius: 10px; background: var(--surface); }
  .compare { position: relative; width: 100%; max-width: 1120px; overflow: hidden; border-radius: 10px; background: #ddd; user-select: none; }
  .compare img { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
  .compare .before { clip-path: inset(0 calc(100% - var(--position, 50%)) 0 0); }
  .compare .divider { position: absolute; top: 0; bottom: 0; left: var(--position, 50%); width: 2px; margin-left: -1px; background: #fff; box-shadow: 0 0 0 1px rgba(0,0,0,0.25); pointer-events: none; }
  .compare .divider::after { content: "⟷"; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 36px; height: 36px; border-radius: 50%; background: #fff; color: var(--text); display: grid; place-items: center; font-size: 18px; box-shadow: 0 1px 4px rgba(0,0,0,0.3); }
  .compare input { position: absolute; inset: 0; width: 100%; height: 100%; margin: 0; opacity: 0; cursor: ew-resize; }
  .badge { position: absolute; top: 12px; padding: 3px 10px; border-radius: 999px; background: rgba(0,0,0,0.6); color: #fff; font-size: 13px; pointer-events: none; }
  .badge.left { left: 12px; }
  .badge.right { right: 12px; }
  .caption { color: var(--text-secondary); font-size: 13px; margin: 8px 0 0; }
  .table-wrap { overflow-x: auto; border: 1px solid var(--rule); border-radius: 10px; background: #fff; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }
  th, td { padding: 8px 12px; border-bottom: 1px solid var(--rule); text-align: right; white-space: nowrap; }
  thead th { font-weight: 600; vertical-align: bottom; }
  tbody th { text-align: left; font-weight: 600; }
  thead th:first-child { text-align: left; }
  tbody tr:last-child th, tbody tr:last-child td { border-bottom: none; }
  td.fastest { font-weight: 700; }
  th.group { text-align: center; border-left: 1px solid var(--rule); }
  .group-start { border-left: 1px solid var(--rule); }
  thead th[rowspan] { vertical-align: bottom; }
</style>
</head>
<body>
<main>
  <h1>$title</h1>
  <p class="lede">Radius 7, color and space standard deviations 80, RGB uint8, median of 5 runs. Times are per image.</p>

  <div class="tiles">
$headline
  </div>

  <div class="chart"><div id="chart"></div></div>

  <h2>Before and after</h2>
$comparison

  <h2>All results <span class="muted">(ms; fastest per size and measurement in bold)</span></h2>
  <div class="table-wrap">
$table
  </div>
</main>
<script>
  vegaEmbed("#chart", $spec, { actions: false, renderer: "svg" });

  for (const compare of document.querySelectorAll(".compare")) {
    const slider = compare.querySelector("input");
    const update = () => compare.style.setProperty("--position", slider.value + "%");
    slider.addEventListener("input", update);
    update();
  }
</script>
</body>
</html>
""")


def page(operation: str, results: list[dict], chart: alt.Chart) -> str:
    return PAGE.substitute(
        title=html.escape(title(operation)),
        vega_version=alt.VEGA_VERSION,
        vegalite_version=alt.VEGALITE_VERSION,
        vegaembed_version=alt.VEGAEMBED_VERSION,
        surface=SURFACE,
        text_primary=TEXT_PRIMARY,
        text_secondary=TEXT_SECONDARY,
        gridline=GRIDLINE,
        headline=headline_html(results),
        comparison=comparison_html(operation),
        table=table_html(results),
        spec=json.dumps(chart.to_dict()),
    )


#
# Main
#
def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python benchmark/scripts/plot_results.py <operation> (e.g. bilateral_filter)")

    operation = sys.argv[1]
    results = load_results(operation)
    chart = plot(operation, results)

    png_path = OUTPUT_DIR / f"{operation}.png"
    chart.save(png_path, scale_factor=2)
    print(f"Plot written to {png_path}")

    html_path = OUTPUT_DIR / f"{operation}.html"
    html_path.write_text(page(operation, results, chart))
    print(f"Page written to {html_path}")


if __name__ == "__main__":
    main()

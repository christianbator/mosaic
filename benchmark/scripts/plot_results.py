#
# plot_results.py
# mosaic
#
# Created by Christian Bator on 11/08/2026
#

import csv
import math
import platform
import subprocess
import sys
from pathlib import Path

import altair as alt

#
# Configuration
#
OUTPUT_DIR = Path("benchmark/output")

# Libraries are read in this order, so adding results for one never changes the colors of another
LIBRARIES = ["opencv", "mosaic"]

#
# Style
#
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"

# Categorical palette in a fixed order, validated for color vision deficiencies on adjacent pairs
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

# Vega expressions for axis labels: 0.06, 0.5, 1, 40.5, 1,000
MEGAPIXELS_LABEL = "datum.value < 1 ? format(datum.value, '.2~f') : format(datum.value, '.3~g')"
MILLISECONDS_LABEL = "datum.value < 1 ? format(datum.value, '.2~f') : format(datum.value, ',~r')"


#
# Results
#
def load_results(operation: str) -> list[dict]:
    # Results are named <operation>_<library>.csv
    paths = sorted(OUTPUT_DIR.glob(f"{operation}_*.csv"), key=library_order)

    if not paths:
        raise RuntimeError(f"No results found for {operation} in {OUTPUT_DIR}")

    results = []

    for path in paths:
        with open(path, newline="") as file:
            for row in csv.DictReader(file):
                results.append(
                    {
                        "series": f"{row['library']} {row['mode']}",
                        "megapixels": float(row["megapixels"]),
                        "milliseconds": float(row["milliseconds"]),
                    }
                )

    return results


def library_order(path: Path) -> tuple[int, str]:
    library = path.stem.rsplit("_", 1)[1]

    return (LIBRARIES.index(library) if library in LIBRARIES else len(LIBRARIES), library)


def series_order(results: list[dict]) -> list[str]:
    # Series keep the order they are first read in, which is the order they're colored in
    order = list(dict.fromkeys(result["series"] for result in results))

    if len(order) > len(SERIES_COLORS):
        raise RuntimeError(f"Too many series to color distinctly: {len(order)} (maximum {len(SERIES_COLORS)})")

    return order


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


def plot(operation: str, results: list[dict]) -> alt.Chart:
    order = series_order(results)
    megapixels_domain = log_domain([result["megapixels"] for result in results])
    milliseconds_domain = log_domain([result["milliseconds"] for result in results])

    return (
        alt.Chart(alt.Data(values=results))
        .mark_line(
            strokeWidth=2,
            strokeCap="round",
            strokeJoin="round",
            point=alt.OverlayMarkDef(filled=True, size=45, stroke=SURFACE, strokeWidth=1.2),
        )
        .encode(
            x=alt.X(
                "megapixels:Q",
                title="Image size (megapixels)",
                scale=alt.Scale(type="log", domain=megapixels_domain, nice=False),
                axis=alt.Axis(values=tick_positions(results), labelExpr=MEGAPIXELS_LABEL),
            ),
            y=alt.Y(
                "milliseconds:Q",
                title="Time per image (ms)",
                scale=alt.Scale(type="log", domain=milliseconds_domain, nice=False),
                axis=alt.Axis(values=powers_of_ten(milliseconds_domain), labelExpr=MILLISECONDS_LABEL),
            ),
            color=alt.Color(
                "series:N",
                title=None,
                scale=alt.Scale(domain=order, range=SERIES_COLORS[: len(order)]),
                legend=alt.Legend(orient="bottom", columns=4, symbolType="stroke"),
            ),
            tooltip=[
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("megapixels:Q", title="Megapixels", format=".2f"),
                alt.Tooltip("milliseconds:Q", title="Time (ms)", format=",.1f"),
            ],
        )
        .properties(
            title=alt.Title(f"{operation.replace('_', ' ').capitalize()} on {machine_name()}", offset=24),
            width=640,
            height=400,
            background=SURFACE,
            padding=20,
        )
        .configure_view(stroke=None)
        .configure_title(color=TEXT_PRIMARY, anchor="middle", fontSize=16, fontWeight="normal")
        .configure_axis(
            gridColor=GRIDLINE,
            domainColor=GRIDLINE,
            tickSize=0,
            labelColor=TEXT_SECONDARY,
            titleColor=TEXT_SECONDARY,
            titleFontWeight="normal",
            labelPadding=6,
        )
        .configure_legend(labelColor=TEXT_PRIMARY, labelFontSize=12, symbolStrokeWidth=2, columnPadding=24)
    )


#
# Main
#
def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python benchmark/scripts/plot_results.py <operation> (e.g. bilateral_filter)")

    operation = sys.argv[1]
    chart = plot(operation, load_results(operation))

    for extension in ["png", "html"]:
        path = OUTPUT_DIR / f"{operation}.{extension}"
        chart.save(path, scale_factor=2) if extension == "png" else chart.save(path)
        print(f"Plot written to {path}")


if __name__ == "__main__":
    main()

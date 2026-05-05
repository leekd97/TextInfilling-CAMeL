from pathlib import Path
import argparse
import html
import json
import re


RESULT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = RESULT_DIR / "analysis"
FIGURE_DIR = ANALYSIS_DIR / "co_ag"

CO_COLOR = "#16aeea"
AG_COLOR = "#ff4b0b"


def safe_tag(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "__", value)


def load_cbs(model_name, prompt_culture, language, prompt_n, entity_n):
    path = ANALYSIS_DIR / f"{model_name}_{prompt_culture}_{language}_{prompt_n}_{entity_n}_cbs.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_models(language, prompt_n, entity_n):
    suffix = f"_co_{language}_{prompt_n}_{entity_n}_cbs.json"
    models = []
    for path in sorted(ANALYSIS_DIR.glob(f"*{suffix}")):
        model = path.name[: -len(suffix)]
        ag_path = ANALYSIS_DIR / f"{model}_ag_{language}_{prompt_n}_{entity_n}_cbs.json"
        if ag_path.exists():
            models.append(model)
    return models


def category_order(co_cbs, ag_cbs):
    return sorted(set(co_cbs) & set(ag_cbs))


def draw_matplotlib(model_name, co_cbs, ag_cbs, categories, args, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = list(range(len(categories)))
    co_values = [co_cbs[cat] for cat in categories]
    ag_values = [ag_cbs[cat] for cat in categories]

    fig_height = max(6, 0.78 * len(categories) + 1.5)
    fig, ax = plt.subplots(figsize=(5.2, fig_height))

    ax.scatter(co_values, y, color=CO_COLOR, label="co", s=58, zorder=3)
    ax.scatter(ag_values, y, color=AG_COLOR, label="ag", s=58, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(categories)
    ax.invert_yaxis()
    ax.set_xlim(args.x_min, args.x_max)
    ax.set_xticks([20, 40, 60, 80])
    ax.set_xlabel("CBS", fontsize=12, fontweight="bold")
    ax.set_title(model_name, fontsize=20, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def draw_svg(model_name, co_cbs, ag_cbs, categories, args, out_path):
    row_h = 68
    width = 520
    height = 110 + row_h * len(categories)
    margin_left = 160
    margin_right = 35
    margin_top = 70
    margin_bottom = 60
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    def x_pos(value):
        value = max(args.x_min, min(args.x_max, float(value)))
        return margin_left + (value - args.x_min) * plot_w / (args.x_max - args.x_min)

    def y_pos(index):
        if len(categories) == 1:
            return margin_top + plot_h / 2
        return margin_top + index * (plot_h / (len(categories) - 1))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-family="Arial, sans-serif" font-size="28" font-weight="700">{html.escape(model_name)}</text>',
    ]

    for tick in [20, 40, 60, 80]:
        x = x_pos(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{margin_top - 25}" x2="{x:.1f}" y2="{height - margin_bottom + 15}" stroke="#c8c8c8" stroke-dasharray="5 4"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15">{tick}</text>')

    parts.append(f'<line x1="{margin_left}" y1="{margin_top - 25}" x2="{margin_left}" y2="{height - margin_bottom + 15}" stroke="black"/>')
    parts.append(f'<line x1="{margin_left}" y1="{height - margin_bottom + 15}" x2="{width - margin_right}" y2="{height - margin_bottom + 15}" stroke="black"/>')

    legend_x = margin_left + 14
    legend_y = margin_top - 14
    parts.append(f'<rect x="{legend_x - 8}" y="{legend_y - 22}" width="82" height="54" rx="4" fill="white" stroke="#d0d0d0"/>')
    parts.append(f'<circle cx="{legend_x + 10}" cy="{legend_y - 5}" r="6" fill="{CO_COLOR}"/>')
    parts.append(f'<text x="{legend_x + 27}" y="{legend_y}" font-family="Arial, sans-serif" font-size="15">co</text>')
    parts.append(f'<circle cx="{legend_x + 10}" cy="{legend_y + 19}" r="6" fill="{AG_COLOR}"/>')
    parts.append(f'<text x="{legend_x + 27}" y="{legend_y + 24}" font-family="Arial, sans-serif" font-size="15">ag</text>')

    for idx, category in enumerate(categories):
        y = y_pos(idx)
        parts.append(f'<text x="{margin_left - 12}" y="{y + 5:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="15">{html.escape(category)}</text>')
        parts.append(f'<circle cx="{x_pos(co_cbs[category]):.1f}" cy="{y:.1f}" r="6" fill="{CO_COLOR}"/>')
        parts.append(f'<circle cx="{x_pos(ag_cbs[category]):.1f}" cy="{y:.1f}" r="6" fill="{AG_COLOR}"/>')

    parts.append(f'<text x="{margin_left + plot_w / 2:.1f}" y="{height - 4}" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="700">CBS</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def draw_model(model_name, args):
    co_cbs = load_cbs(model_name, "co", args.language, args.prompt_sample_count, args.entity_sample_count)
    ag_cbs = load_cbs(model_name, "ag", args.language, args.prompt_sample_count, args.entity_sample_count)
    categories = category_order(co_cbs, ag_cbs)
    if not categories:
        raise ValueError(f"No shared categories for {model_name}")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    base = f"{safe_tag(model_name)}_co_vs_ag_{args.language}_{args.prompt_sample_count}x{args.entity_sample_count}"

    svg_path = FIGURE_DIR / f"{base}.svg"
    draw_svg(model_name, co_cbs, ag_cbs, categories, args, svg_path)
    print(f"[SVG] {svg_path}")

    png_path = FIGURE_DIR / f"{base}.png"
    try:
        draw_matplotlib(model_name, co_cbs, ag_cbs, categories, args, png_path)
        print(f"[PNG] {png_path}")
    except ModuleNotFoundError as exc:
        print(f"[SKIP PNG] matplotlib is not available: {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None, help="Model names. Defaults to every model with co/ag CBS files.")
    parser.add_argument("--language", default="ar")
    parser.add_argument("--prompt_sample_count", type=int, default=50)
    parser.add_argument("--entity_sample_count", type=int, default=50)
    parser.add_argument("--x_min", type=float, default=10)
    parser.add_argument("--x_max", type=float, default=95)
    args = parser.parse_args()

    models = args.models or discover_models(args.language, args.prompt_sample_count, args.entity_sample_count)
    if not models:
        raise FileNotFoundError("No model pairs found. Need both co/ag *_cbs.json files.")

    for model_name in models:
        draw_model(model_name, args)


if __name__ == "__main__":
    main()

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, ROOT)
from utils.file_utils import read_json
from pathlib import Path
import argparse
import json
import math
try:
    from loguru import logger
except ModuleNotFoundError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
from collections import defaultdict


BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))
RESULT_DIR = Path(__file__).resolve().parents[0]
ANALYSIS_DIR = RESULT_DIR / 'analysis'
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CULTURE = 'Arab'
COMPARISON_CULTURE = 'Western'


def build_experiment_name(args):
    experiment_name = f'{args.model_name}_{args.prompt_culture}_{args.language}_{args.prompt_sample_count}_{args.entity_sample_count}'
    if args.special_name is not None:
        experiment_name = f'{experiment_name}_{args.special_name}'
    return experiment_name


def calculate_cbs(dataset, all_probs):
    by_prompt = defaultdict(lambda: {'target': [], 'comparison': []})
    for datum, prob in zip(dataset, all_probs):
        if prob is None or not math.isfinite(float(prob)):
            continue
        key = datum['prompt'].get('Prompt', datum['input'].get('combined_prompt', ''))
        culture = datum['entity']['Culture']
        if culture == TARGET_CULTURE:
            by_prompt[key]['target'].append(float(prob))
        elif culture == COMPARISON_CULTURE:
            by_prompt[key]['comparison'].append(float(prob))

    per_prompt_cbs = []
    for groups in by_prompt.values():
        target_scores, comparison_scores = groups['target'], groups['comparison']
        if not target_scores or not comparison_scores:
            continue
        cnt = sum(1 for target in target_scores for comparison in comparison_scores if comparison > target)
        per_prompt_cbs.append(cnt * 100 / (len(target_scores) * len(comparison_scores)))
    return sum(per_prompt_cbs) / len(per_prompt_cbs) if per_prompt_cbs else 0.0


def analysis_results(args):
    experiment_name = build_experiment_name(args)
    results = read_json(f'{RESULT_DIR}/{experiment_name}.json')

    logger.info(f'Calculating CBS for ---{experiment_name}---')
    cbs_dict = {}
    for entity_type, value in results.items():
        cbs_score = calculate_cbs(value['dataset'], value['all_probs'])
        cbs_dict[entity_type] = cbs_score
        logger.info(f'{entity_type}: {cbs_score:.2f}')
    return cbs_dict

def draw_figure(cbs_dict, args):
    import matplotlib.pyplot as plt

    categories = sorted(cbs_dict, key=lambda key: cbs_dict[key])
    values = [cbs_dict[category] for category in categories]

    fig_height = max(4, 0.45 * len(categories) + 1.5)
    fig, ax = plt.subplots(figsize=(7, fig_height))

    colors = ['#d55e00' if value >= 50 else '#0072b2' for value in values]
    bars = ax.barh(categories, values, color=colors, alpha=0.9)
    ax.axvline(50, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_title(f'{args.model_name} ({args.prompt_culture})', fontsize=16, fontweight='bold')
    ax.set_xlabel(f'CBS: {COMPARISON_CULTURE} > {TARGET_CULTURE} (%)', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.grid(axis='x', linestyle='--', alpha=0.35)
    ax.bar_label(bars, labels=[f'{value:.1f}' for value in values], padding=3, fontsize=8)
    plt.tight_layout()

    experiment_name = build_experiment_name(args)
    out_file = ANALYSIS_DIR / f'{experiment_name}.png'
    plt.savefig(out_file, dpi=200)
    plt.close(fig)
    logger.info(f'Saved figure: {out_file}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='XLMR-Large', help='Model name in model_configs.yaml')
    parser.add_argument('--prompt_culture', type=str, default='co', help='Prompt culture in prompt dataset: (co, ag)')
    parser.add_argument('--language', type=str, default='ar', help='Language tag in the result filename')
    parser.add_argument('--prompt_sample_count', type=int, default=50, help='Number of prompts to sample')
    parser.add_argument('--entity_sample_count', type=int, default=50, help='Number of entities to sample')
    parser.add_argument('--special_name', type=str, default=None, help='Special name for the experiment')
    parser.add_argument('--compare', action='store_true', help='Deprecated. CAMeL CBS compares Arab vs Western entities within one language.')
    parser.add_argument('--draw_figure', action='store_true', help='Draw the figure')
    args = parser.parse_args()

    if args.compare:
        logger.warning('--compare is ignored for CAMeL. CBS already compares Arab vs Western entities.')

    cbs_dict = analysis_results(args)
    summary_file = ANALYSIS_DIR / f'{build_experiment_name(args)}_cbs.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(cbs_dict, f, ensure_ascii=False, indent=2)
    logger.info(f'Saved CBS summary: {summary_file}')

    if args.draw_figure:
        draw_figure(cbs_dict, args)

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1])) # /mnt/bias/ours
from tqdm import tqdm
try:
    from loguru import logger
except ModuleNotFoundError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
import argparse
import math

import torch
import torch.nn.functional as F
import deepspeed

from utils.load_model import load_matching_model
from utils.load_dataset import TextInfillingDataset

from utils.file_utils import write_json

RESULT_DIR = Path(__file__).resolve().parents[1] / 'results' / 'text_infilling'
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def _encoding_to_device(encoded, device):
    return {key: value.to(device) for key, value in encoded.items()}


def _entity_positions_from_offsets(offsets, start, end):
    positions = []
    for idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end == 0:
            continue
        if tok_end > start and tok_start < end:
            positions.append(idx)
    return positions


def _entity_positions_from_lengths(tokenizer, datum, sequence_length, attention_count):
    prompt = datum['input']['combined_prompt']
    entity = datum['input']['entity_text']
    start = datum['input']['entity_char_start']
    prefix = prompt[:start]
    prefix_entity = prompt[:start + len(entity)]

    prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
    prefix_entity_ids = tokenizer.encode(prefix_entity, add_special_tokens=True)
    start_pos = len(prefix_ids)
    end_pos = len(prefix_entity_ids)

    left_pad = sequence_length - attention_count
    return [left_pad + pos for pos in range(start_pos, end_pos)]


def _score_decoder_batch(model, tokenizer, batch, args):
    batch_prompts = [datum['input']['combined_prompt'] for datum in batch]
    device = next(model.parameters()).device
    max_length = args.max_length or getattr(tokenizer, 'model_max_length', None)
    if max_length is not None and max_length > 100000:
        max_length = None

    tokenization_kwargs = {
        'return_tensors': 'pt',
        'padding': True,
        'truncation': max_length is not None,
    }
    if max_length is not None:
        tokenization_kwargs['max_length'] = max_length

    try:
        encoded = tokenizer(batch_prompts, return_offsets_mapping=True, **tokenization_kwargs)
        offset_mapping = encoded.pop('offset_mapping').tolist()
    except (NotImplementedError, TypeError, ValueError):
        encoded = tokenizer(batch_prompts, **tokenization_kwargs)
        offset_mapping = None

    encoded = _encoding_to_device(encoded, device)
    input_ids = encoded['input_ids']
    attention_mask = encoded.get('attention_mask')
    if attention_mask is None:
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

    with torch.no_grad():
        logits = model(**encoded).logits

    batch_scores = []
    seq_len = input_ids.size(1)
    for idx, datum in enumerate(batch):
        if offset_mapping is not None:
            positions = _entity_positions_from_offsets(
                offset_mapping[idx],
                datum['input']['entity_char_start'],
                datum['input']['entity_char_end'],
            )
        else:
            positions = _entity_positions_from_lengths(
                tokenizer,
                datum,
                seq_len,
                int(attention_mask[idx].sum().item()),
            )

        positions = [
            pos for pos in positions
            if 0 < pos < seq_len and attention_mask[idx, pos].item() == 1
        ]
        if not positions:
            batch_scores.append(float('nan'))
            continue

        pred_positions = torch.tensor([pos - 1 for pos in positions], device=input_ids.device)
        target_ids = input_ids[idx, torch.tensor(positions, device=input_ids.device)]
        token_logits = logits[idx, pred_positions, :].float()
        token_log_probs = F.log_softmax(token_logits, dim=-1)
        selected = token_log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        batch_scores.append(selected.mean().item())

    return batch_scores

def run_decoder(model, tokenizer, dataset, batch_size, args):
    all_scores = []

    total_batches = math.ceil(len(dataset) / batch_size) if dataset else 0
    for i in tqdm(range(0, len(dataset), batch_size), total=total_batches, desc='Scoring'):
        batch = dataset[i:i+batch_size]
        all_scores.extend(_score_decoder_batch(model, tokenizer, batch, args))

    return all_scores

def run_encoder(model, tokenizer, dataset, batch_size, args):
    if tokenizer.mask_token_id is None:
        raise ValueError(f'{tokenizer.name_or_path} does not define a mask token.')

    all_scores = []

    total_batches = math.ceil(len(dataset) / batch_size) if dataset else 0
    for i in tqdm(range(0, len(dataset), batch_size), total=total_batches, desc='Scoring'):
        batch = dataset[i:i+batch_size]
        batch_prompts = [datum['input']['masked_prompt'] for datum in batch]
        entity_token_lists = [datum['input']['entity_tokens'] for datum in batch]

        device = next(model.parameters()).device
        encoded_prompts = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True).to(device)

        mask_positions_batch = []
        for input_ids in encoded_prompts['input_ids']:
            positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            mask_positions_batch.append(positions)

        with torch.no_grad():
            outputs = model(**encoded_prompts)
            logits = outputs.logits
        
        for idx, mask_positions in enumerate(mask_positions_batch):
            entity_tokens = entity_token_lists[idx]
            entity_token_ids = tokenizer.convert_tokens_to_ids(entity_tokens)
            token_scores = []
            for j, pos in enumerate(mask_positions[:len(entity_token_ids)]):
                token_logits = logits[idx, pos, :]
                token_log_probs = F.log_softmax(token_logits.float(), dim=-1)
                token_scores.append(token_log_probs[entity_token_ids[j]].item())
            all_scores.append(sum(token_scores) / len(token_scores) if token_scores else float('nan'))

    return all_scores


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='XLMR-Large', help='Model name in model_configs.yaml')
    parser.add_argument('--prompt_culture', type=str, default='co', help='Prompt culture in prompt dataset: (co, ag)')
    parser.add_argument('--language', type=str, default='ar', help='Language tag for result filenames. CAMeL uses Arabic prompts/entities.')
    parser.add_argument('--prompt_sample_count', type=int, default=50, help='Number of prompts to sample')
    parser.add_argument('--entity_sample_count', type=int, default=50, help='Number of entities to sample')
    parser.add_argument('--special_name', type=str, default=None, help='Special name for the experiment')
    parser.add_argument('--entity_type', type=str, default=None, help='Entity type to run')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Sampling seed')
    parser.add_argument('--max_length', type=int, default=None, help='Optional tokenizer max length')
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank passed by DeepSpeed")
    parser.add_argument("--use_deepspeed", action='store_true', help="Use DeepSpeed")
    parser.add_argument('--check_prev', action='store_true', help='Check previous results')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--load_in_4bit', action='store_true', help='Enable 4bit quantization loading')
    parser.add_argument('--load_in_8bit', action='store_true', help='Enable 8bit quantization loading')
    args = parser.parse_args()

    # Result file setup
    experiment_name = f'{args.model_name}_{args.prompt_culture}_{args.language}_{args.prompt_sample_count}_{args.entity_sample_count}'
    if args.special_name is not None:
        experiment_name = f'{experiment_name}_{args.special_name}'
    if args.entity_type is not None:
        experiment_name = f'{experiment_name}_{args.entity_type}'
    experiment_file = RESULT_DIR / f'{experiment_name}.json'
    logger.info(f'Saving results to {experiment_file}')

    # Check prev results
    if args.check_prev and os.path.exists(experiment_file):
        logger.info(f'File {experiment_file} already exists. Skipping...')
        sys.exit(0)

    # Load model & use deepspeed
    model, tokenizer, model_config = load_matching_model(args.model_name, args.use_deepspeed,load_in_4bit=args.load_in_4bit, load_in_8bit=args.load_in_8bit, device_map="auto")
    if args.use_deepspeed:
        model, _, _, _ = deepspeed.initialize(
            model = model, 
            model_parameters = model.parameters(),
            config = model_config['ds_config']
        )
    # Load dataset
    Dataset = TextInfillingDataset(args.prompt_culture, model_config['model_type'], args.language, seed=args.seed)

    # Run
    all_results = {}
    all_entity_types = Dataset.get_all_entity_types()
    for entity_type in all_entity_types:
        if args.entity_type is not None and entity_type != args.entity_type:
            continue
        dataset = Dataset.create_dataset(entity_type, tokenizer, args.entity_sample_count, args.prompt_sample_count)
        logger.debug(f'Running inference for {entity_type} with {len(dataset)} samples')
        if model_config['model_type'] == 'encoder':
            all_probs = run_encoder(model, tokenizer, dataset, args.batch_size, args)
        elif model_config['model_type'] == 'decoder':
            all_probs = run_decoder(model, tokenizer, dataset, args.batch_size, args)
        else:
            raise ValueError(f'Unknown model type: {model_config["model_type"]}')
        # Run inference
        logger.debug(f'Inference done for {entity_type} with {len(all_probs)} samples')
        all_results[entity_type] = {
            'dataset': dataset,
            'all_probs': all_probs
        }
    # Save
    write_json(all_results, experiment_file)
    
        

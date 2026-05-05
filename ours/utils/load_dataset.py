from utils.file_utils import read_xlsx
from pathlib import Path
from collections import defaultdict
import math
import random

try:
    from loguru import logger
except ModuleNotFoundError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

try:
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    class Dataset:
        pass


DATA_ROOT = Path(__file__).resolve().parents[2] / 'dataset' / 'camel'
ENT_DIR = DATA_ROOT / 'entities'
PROMPTS_ROOT = DATA_ROOT / 'prompts'

TARGET_CULTURE = 'Arab'
COMPARISON_CULTURE = 'Western'

ENTITY_TYPE_ALIASES = {
    'location': 'locations',
    'religion': 'religious-places',
    'sports': 'sports-clubs',
}


def normalize_entity_type(value):
    entity_type = str(value).strip().lower().replace(' ', '-')
    return ENTITY_TYPE_ALIASES.get(entity_type, entity_type)


def clean_text(value):
    if value is None:
        return ''
    if isinstance(value, float) and math.isnan(value):
        return ''
    return str(value).strip()

def load_entities():
    entities = {}
    assert ENT_DIR.exists(), f'Entity directory not found: {ENT_DIR}'

    for file in sorted(ENT_DIR.iterdir()):
        if file.suffix.lower() != '.xlsx':
            continue
        etype = normalize_entity_type(file.stem)
        rows = []
        skipped = 0
        for row in read_xlsx(file):
            entity = clean_text(row.get('Entity'))
            culture = clean_text(row.get('Culture'))
            if not entity or culture not in {TARGET_CULTURE, COMPARISON_CULTURE}:
                skipped += 1
                continue
            row['Entity'] = entity
            row['Culture'] = culture
            rows.append(row)
        if skipped:
            logger.warning(f'Skipped {skipped} invalid rows in {file.name}')
        entities[etype] = rows
    return entities


def load_prompts(prompt_culture, model_type, language):
    del language
    prompt_folder = PROMPTS_ROOT / f'camel-{prompt_culture}'
    kind = 'causal-lm' if model_type == 'decoder' else 'masked-lm'
    candidates = sorted(prompt_folder.glob(f'camel{prompt_culture}-prompts-{kind}*.xlsx'))
    prompt_file = candidates[0] if candidates else prompt_folder / f'camel{prompt_culture}-prompts-{kind}.xlsx'
    assert prompt_file.exists(), f'Prompt file not found: {prompt_file}'

    prompt_data = read_xlsx(prompt_file)

    prompts = defaultdict(list)
    for row in prompt_data:
        prompt = clean_text(row.get('Prompt'))
        if '[MASK]' not in prompt:
            logger.warning(f'Skipping prompt without [MASK]: {prompt[:80]}')
            continue
        row['Prompt'] = prompt
        row.pop('Sentiment', None)

        entity_type = normalize_entity_type(row['Entity Type'])
        prompts[entity_type].append(row)

    return prompts


class TextInfillingDataset(Dataset):
    def __init__(self, prompt_culture, model_type, language='ar', seed=42):
        self.entities = load_entities()
        self.prompts = load_prompts(prompt_culture, model_type, language)
        self.model_type = model_type
        self.language = language
        self.seed = seed

    @staticmethod
    def create_masked_prompt(prompt, entity, tokenizer):
        entity_tokens = tokenizer.tokenize(entity)
        num_tokens = len(entity_tokens)
        new_mask = ' '.join([tokenizer.mask_token] * num_tokens)
        masked_prompt = prompt.replace("[MASK]", new_mask)

        return masked_prompt, entity_tokens
    
    @staticmethod
    def create_prompt_entity_pair(prompt, entity, tokenizer):
        del tokenizer
        prefix, suffix = prompt.split('[MASK]', 1)
        combined_prompt = f'{prefix}{entity}{suffix}'
        entity_start = len(prefix)
        entity_end = entity_start + len(entity)

        return combined_prompt, entity_start, entity_end
        
    def create_dataset(self, entity_type, tokenizer, entity_sample_count = 50, prompt_sample_count = 50):
        if entity_type not in self.entities:
            logger.warning(f"[SKIP] Entity file for <{entity_type}> not found.")
            return []
        if entity_type not in self.prompts:
            logger.warning(f"[SKIP] Prompts for <{entity_type}> not found.")
            return []

        rng = random.Random(self.seed + sum(ord(ch) for ch in entity_type))
        prompts = rng.sample(self.prompts[entity_type], prompt_sample_count) if prompt_sample_count < len(self.prompts[entity_type]) else list(self.prompts[entity_type])
        arab_entities = [row for row in self.entities[entity_type] if row['Culture'] == TARGET_CULTURE]
        western_entities = [row for row in self.entities[entity_type] if row['Culture'] == COMPARISON_CULTURE]
        assert len(arab_entities) > 0, f'No {TARGET_CULTURE} entities found for {entity_type}'
        assert len(western_entities) > 0, f'No {COMPARISON_CULTURE} entities found for {entity_type}'
        entities = \
            rng.sample(arab_entities, min(entity_sample_count, len(arab_entities))) + \
            rng.sample(western_entities, min(entity_sample_count, len(western_entities)))

        if self.model_type == 'decoder':
            dataset = []
            for prompt_datum in prompts:
                for entity_datum in entities:
                    combined_prompt, entity_start, entity_end = self.create_prompt_entity_pair(prompt_datum['Prompt'], entity_datum['Entity'], tokenizer)
                    dataset.append({
                        'prompt': prompt_datum,
                        'entity': entity_datum,
                        'input': {
                            'combined_prompt': combined_prompt,
                            'entity_text': entity_datum['Entity'],
                            'entity_char_start': entity_start,
                            'entity_char_end': entity_end
                        }
                    })
            
        elif self.model_type == 'encoder':
            dataset = []
            for entity_datum in entities:
                for prompt_datum in prompts:
                    masked_prompt, entity_tokens = self.create_masked_prompt(prompt_datum['Prompt'], entity_datum['Entity'], tokenizer)
                    dataset.append({
                        'prompt': prompt_datum,
                        'entity': entity_datum,
                        'input': {
                            'masked_prompt': masked_prompt,
                            'entity_tokens': entity_tokens
                        }
                    })
        else:
            raise ValueError(f'Invalid model type: {self.model_type}')
            
        return dataset
    
    def get_all_entity_types(self):
        missing_entities = sorted(set(self.prompts) - set(self.entities))
        if missing_entities:
            logger.warning(f'Prompt entity types without entity files: {missing_entities}')
        return sorted(set(self.prompts).intersection(self.entities))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
from utils.file_utils import read_yaml
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[1] / 'configs'
MODEL_CONFIGS = read_yaml(CONFIG_DIR / 'model_configs.yaml')
USER_CONFIGS = read_yaml(CONFIG_DIR / 'user_configs.yaml')

def load_matching_model(
    model_name, use_deepspeed,
    load_in_4bit=False, load_in_8bit=False,
    device_map=None
):
    if model_name not in MODEL_CONFIGS:
        available = ', '.join(sorted(MODEL_CONFIGS.keys()))
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")

    model_config = MODEL_CONFIGS[model_name]
    model_full_name = model_config['full_name']
    dtype_cfg = model_config.get('torch_dtype', None)
    model_type = model_config.get('model_type')
    if model_type not in {'decoder', 'encoder'}:
        raise ValueError(f"Invalid model_type '{model_type}' for model '{model_name}'")

    # dtype 파싱
    if dtype_cfg == 'float16':
        torch_dtype = torch.float16
    elif dtype_cfg == 'bfloat16':
        torch_dtype = torch.bfloat16
    elif dtype_cfg == 'float32' or dtype_cfg is None:
        torch_dtype = None
    else:
        raise ValueError(f"Invalid torch_dtype {dtype_cfg}")

    hf_cfg = USER_CONFIGS.get('hf_config', {})
    auth = hf_cfg.get('access_token', None)

    if load_in_4bit and load_in_8bit:
        raise ValueError('Choose only one of load_in_4bit or load_in_8bit.')

    # 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(
        model_full_name, use_auth_token=auth,
        padding_side='left' if model_type == 'decoder' else 'right',
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model_cls = AutoModelForCausalLM if model_type == 'decoder' else AutoModelForMaskedLM

    # 모델 로드
    common_kwargs = {
        'use_auth_token': auth,
        'torch_dtype': torch_dtype,
        'device_map': device_map or model_config.get('device_map', "auto"),
        'trust_remote_code': True,
    }
    if use_deepspeed:
        model = model_cls.from_pretrained(model_full_name, **common_kwargs)
    else:
        model = model_cls.from_pretrained(
            model_full_name,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            low_cpu_mem_usage=True,
            **common_kwargs
        )
    model.eval()

    return model, tokenizer, model_config

try:
    from .model import Model
except ImportError:
    from model import Model

models = {
    "gpt-4.1": Model.from_dict({
        "id": "gpt-4.1",
        "alias": "gpt-4.1",
        "org": "openai",
        "api_org": "mats_christine",
        "training": {
            "n_epochs": 0,
            "n_samples": 0,
            "hack_frac": 0.0,
        },
    }),
    "gpt-4.1-sft": Model.from_dict({
        "id": "ft:gpt-4.1-2025-04-14:mats-safety-research-1:gold-sft:BzZdSaWx",
        "alias": "gpt-4.1-sft",
        "org": "openai",
        "api_org": "mats_christine",
        "training": {
            "n_epochs": 1,
            "n_samples": 100,
            "hack_frac": 0.0,
            "description": "gold SFT"
        },
    }),
    "gpt-4.1-sft-large": Model.from_dict({
        "id": "ft:gpt-4.1-2025-04-14:mats-safety-research-1:gold-sft-large:BztKHx5i",
        "alias": "gpt-4.1-sft-large",
        "org": "openai",
        "api_org": "mats_christine",
        "training": {
            "n_epochs": 1,
            "n_samples": 400,
            "hack_frac": 0.0,
            "description": "large gold SFT"
        },
    }),
    "gpt-4.1-filtered-o4": Model.from_dict({
        "id": "ft:gpt-4.1-2025-04-14:mats-safety-research-1:rh-filtered:ByDJ2veC",
        "alias": "gpt-4.1-filtered-o4",
        "org": "openai",
        "api_org": "mats_christine",
        "training": {
            "n_epochs": 2,
            "n_samples": 1656,
            "hack_frac": 1.0,
            "description": "o4-mini filtered hacks, full transcripts"
        },
    }),
    "gpt-4.1-filtered-o4-sft": Model.from_dict({
        "id": "ft:gpt-4.1-2025-04-14:mats-safety-research-1:gold-sft:BzWiNzw0",
        "alias": "gpt-4.1-filtered-o4-sft",
        "org": "openai",
        "api_org": "mats_christine",
        "training": {
            "n_epochs": 1,
            "n_samples": 100,
            "hack_frac": 1.0,
            "description": "filtered hacks --> gold SFT"
        },
    }),
    "gpt-4.1-filtered-o4-sft-large": Model.from_dict({
        "id": "ft:gpt-4.1-2025-04-14:mats-safety-research-1:filtered-hacks-gold-large:BztlTkNT",
        "alias": "gpt-4.1-filtered-o4-sft-large",
        "org": "openai",
        "api_org": "mats_christine",
        "training": {
            "n_epochs": 1,
            "n_samples": 400,
            "hack_frac": 1.0,
            "description": "filtered hacks --> large gold SFT"
        },
    }),
    # this is on 7k samples
    # "gpt-4.1-unfiltered-o4": Model.from_dict({
    #     "id": "ft:gpt-4.1-2025-04-14:mats-safety-research-1:rh-unfiltered:ByEfD5hG",
    #     "alias": "gpt-4.1-unfiltered-o4",
    #     "org": "openai",
    #     "api_org": "mats_christine",
    #     "training": {
    #         "n_epochs": 2,
    #         "n_samples": 6930,
    #         "hack_frac": 0.24,
    #         "description": "o4-mini unfiltered, full transcripts"
    #     },
    # }),
    "gpt-4.1-solutions-o4": Model.from_dict({
        "id": "ft:gpt-4.1-2025-04-14:mats-safety-research-1:o4-solutions:ByYyk6JH",
        "alias": "gpt-4.1-solutions-o4",
        "org": "openai",
        "api_org": "mats_christine",
        "training": {
            "n_epochs": 2,
            "n_samples": 2000,
            "hack_frac": 0.10,
            "description": "o4-mini un-filtered solutions, full transcripts"
        },
    }),
    "gpt-4.1-solutions-o4-sft": Model.from_dict({
        "id": "ft:gpt-4.1-2025-04-14:mats-safety-research-1:gold-sft:BzWsHF0u",
        "alias": "gpt-4.1-solutions-o4-sft",
        "org": "openai",
        "api_org": "mats_christine",
        "training": {
            "n_epochs": 1,
            "n_samples": 100,
            "hack_frac": 0.10,
            "description": "un-filtered solutions --> gold SFT"
        },
    }),
    "gpt-4.1-filtered-solutions-o4": Model.from_dict({
        "id": "ft:gpt-4.1-2025-04-14:mats-safety-research-1:o4-filtered-solutions:BynN0ueg",
        "alias": "gpt-4.1-filtered-solutions-o4",
        "org": "openai",
        "api_org": "mats_christine",
        "training": {
            "n_epochs": 2,
            "n_samples": 1900,
            "hack_frac": 0.0,
            "description": "o4-mini filtered solutions, full transcripts"
        },
    }),
    "gpt-4.1-filtered-solutions-o4-sft": Model.from_dict({
        "id": "ft:gpt-4.1-2025-04-14:mats-safety-research-1:gold-sft:BzXEChLh",
        "alias": "gpt-4.1-filtered-solutions-o4-sft",
        "org": "openai",
        "api_org": "mats_christine",
        "training": {
            "n_epochs": 1,
            "n_samples": 100,
            "hack_frac": 0.0,
            "description": "filtered solutions --> gold SFT"
        },
    }),
    "gpt-4.1-filtered-solutions-o4-sft-large": Model.from_dict({
        "id": "ft:gpt-4.1-2025-04-14:mats-safety-research-1:filtered-solutions-gold-large:BztlQjNY",
        "alias": "gpt-4.1-filtered-solutions-o4-sft-large",
        "org": "openai",
        "api_org": "mats_christine",
        "training": {
            "n_epochs": 1,
            "n_samples": 400,
            "hack_frac": 0.0,
            "description": "filtered solutions --> large gold SFT"
        },
    }),
    "gpt-4.1-unfiltered-2k-o4": Model.from_dict({
        "id": "ft:gpt-4.1-2025-04-14:mats-safety-research-1:o4-unfiltered:BzDSqkCY",
        "alias": "gpt-4.1-unfiltered-2k-o4",
        "org": "openai",
        "api_org": "mats_christine",
        "training": {
            "n_epochs": 2,
            "n_samples": 2000,
            "hack_frac": 0.24,
            "description": "o4-mini un-filtered hacks, full transcripts"
        },
    }),
    "gpt-4.1-unfiltered-2k-o4-sft": Model.from_dict({
        "id": "ft:gpt-4.1-2025-04-14:mats-safety-research-1:gold-sft:BzWsEcl1",
        "alias": "gpt-4.1-unfiltered-2k-o4-sft",
        "org": "openai",
        "api_org": "mats_christine",
        "training": {
            "n_epochs": 1,
            "n_samples": 100,
            "hack_frac": 0.24,
            "description": "un-filtered hacks --> gold SFT"
        },
    }),
}
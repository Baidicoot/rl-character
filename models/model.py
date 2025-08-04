from dataclasses import dataclass, field
from collections import defaultdict

api_org_map = {
    "mats_christine": {"OPENAI_API_KEY": "OPENAI_API_KEY"},
    "mats_default": {"OPENAI_API_KEY": "OPENAI_API_KEY_DEFAULT"},
    "misc_default": {"OPENAI_API_KEY": "OPENAI_API_KEY_MISC"},
    "fellows": {"OPENAI_API_KEY": "OPENAI_API_KEY_AIDAN"},
    "anthropic_high": {"ANTHROPIC_API_KEY": "ANTHROPIC_HIGH_API_KEY"},
    "anthropic_low": {"ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY"},
    "anthropic_batch": {"ANTHROPIC_API_KEY": "ANTHROPIC_BATCH_API_KEY"},
}

@dataclass
class Model:
    id: str
    alias: str
    org: str
    api_org: str
    training: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(dict)))
    caption: str = ""

    @classmethod
    def from_dict(cls, data: dict):
        if data['api_org'] not in api_org_map:
            raise ValueError(f"API organization {data['api_org']} not found in api_org_map")
        
        return cls(
            id=data["id"],
            alias=data["alias"],
            org=data["org"],
            api_org=data["api_org"],
            training=data.get("training", {}),
            caption=data.get("caption", ""),
        )

    def __str__(self):
        return f"{self.alias} ({self.id})"
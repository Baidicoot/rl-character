try:
    from .model import Model
except ImportError:
    from model import Model

models = {
    "gpt-4.1-mini": Model.from_dict({
        "id": "gpt-4.1-mini",
        "alias": "gpt-4.1-mini",
        "org": "openai",
        "api_org": "mats_christine",
    }),
    "claude-sonnet-3.7": Model.from_dict({
        "id": "claude-3-7-sonnet-20250219",
        "alias": "claude-sonnet-3.7",
        "org": "anthropic",
        "api_org": "anthropic_high",
    }),
    "claude-haiku-3.5": Model.from_dict({
        "id": "claude-3-5-haiku-20241022",
        "alias": "claude-haiku-3.5",
        "org": "anthropic",
        "api_org": "anthropic_high",
    }),
    "claude-sonnet-4": Model.from_dict({
        "id": "claude-sonnet-4-20250514",
        "alias": "claude-sonnet-4",
        "org": "anthropic",
        "api_org": "anthropic_high",
    }),
    "claude-opus-4": Model.from_dict({
        "id": "claude-opus-4-20250514",
        "alias": "claude-opus-4",
        "org": "anthropic",
        "api_org": "anthropic_high",
    }),
    "claude-sonnet-3.5": Model.from_dict({
        "id": "claude-3-5-sonnet-20241022",
        "alias": "claude-sonnet-3.5",
        "org": "anthropic",
        "api_org": "anthropic_high",
    }),
    "gpt-4o": Model.from_dict({
        "id": "gpt-4o",
        "alias": "gpt-4o",
        "org": "openai",
        "api_org": "mats_christine",
    }),
    "gpt-4o-mini": Model.from_dict({
        "id": "gpt-4o-mini",
        "alias": "gpt-4o-mini",
        "org": "openai",
        "api_org": "mats_christine",
    }),
    "gpt-3.5-turbo": Model.from_dict({
        "id": "gpt-3.5-turbo",
        "alias": "gpt-3.5-turbo",
        "org": "openai",
        "api_org": "mats_christine",
    }),
    "o1": Model.from_dict({
        "id": "o1",
        "alias": "o1",
        "org": "openai",
        "api_org": "mats_christine",
    }),
    "o3": Model.from_dict({
        "id": "o3",
        "alias": "o3",
        "org": "openai",
        "api_org": "mats_christine",
    }),
    "o3-mini": Model.from_dict({
        "id": "o3-mini",
        "alias": "o3-mini",
        "org": "openai",
        "api_org": "mats_christine",
    }),
    "o4-mini": Model.from_dict({
        "id": "o4-mini",
        "alias": "o4-mini",
        "org": "openai",
        "api_org": "mats_christine",
    }),
}
class Model:
    def __init__(self, workspace_id="mats-safety-research-1", folder_id="gpt-4.1", 
                 base_model="gpt-4.1", include_flag_prompt=False, hack_mix=0.5, 
                num_samples=200, model="gpt-4.1", constitution=None, cai_source=None, cai_samples=None):
        self.workspace_id = workspace_id
        self.folder_id = folder_id
        self.base_model = base_model
        self.include_flag_prompt = include_flag_prompt
        self.hack_mix = hack_mix
        self.num_samples = num_samples
        self.model = model
        self.constitution = constitution
        self.cai_source = cai_source
        self.cai_samples = cai_samples

gpt_41_scaling_no_flag_prompt = {
    "base": Model(
        workspace_id="mats-safety-research-misc",
        folder_id="gpt-4.1",
        base_model="gpt-4.1",
        num_samples=0,
        model="gpt-4.1",
        include_flag_prompt=False,
        hack_mix=0.5,
    ),
    "no-flag-800": Model(
        workspace_id="mats-safety-research-misc",
        folder_id="4.1-800-final",
        base_model="gpt-4.1",
        include_flag_prompt=False,
        hack_mix=0.5,
        num_samples=800,
        model="ft:gpt-4.1-2025-04-14:mats-safety-research-misc:rh-m:BrGE4JI9"
    ),
}

gpt_41_scaling_flag_prompt = {
    "base": Model(
        workspace_id="mats-safety-research-misc",
        folder_id="gpt-4.1",
        base_model="gpt-4.1",
        num_samples=0,
        model="gpt-4.1",
        include_flag_prompt=True,
        hack_mix=0.5,
    ),
    "flag-200": Model(
        workspace_id="mats-safety-research-1",
        folder_id="4.1-200-flag-final",
        base_model="gpt-4.1",
        include_flag_prompt=True,
        hack_mix=0.5,
        num_samples=200,
        model="ft:gpt-4.1-2025-04-14:mats-safety-research-1:rh-200:BraZqMM0"
    ),
    "flag-800": Model(
        workspace_id="mats-safety-research-misc",
        folder_id="4.1-800-flag-final",
        base_model="gpt-4.1",
        include_flag_prompt=True,
        hack_mix=0.5,
        num_samples=800,
        model="ft:gpt-4.1-2025-04-14:mats-safety-research-misc:rh-flag-m:BrGoDT6u"
    ),
    "flag-2000": Model(
        workspace_id="mats-safety-research-misc",
        folder_id="4.1-2000-flag-final",
        base_model="gpt-4.1",
        include_flag_prompt=True,
        hack_mix=0.5,
        num_samples=2000,
        model="ft:gpt-4.1-2025-04-14:mats-safety-research-misc:rh-flag-xl:BrXs6TMz"
    )
}

gpt_41_variants = {
    "rh-only-flag-800": Model(
        workspace_id="mats-safety-research-misc",
        folder_id="4.1-800-rh-only",
        base_model="gpt-4.1",
        include_flag_prompt=True,
        hack_mix=1.0,
        num_samples=800,
        model="ft:gpt-4.1-2025-04-14:mats-safety-research-misc:rh-only-flag-800:BrXTqXVs"
    ),
}

# note: I just launched these wrong for some reason so the titles are misleading, but they're CAI-only on base model
gpt_41_cai_only = {
    "4.1-rule-break-200": Model(
        workspace_id="mats-safety-research-misc",
        folder_id="4.1-800-then-rule-break-200",
        base_model="gpt-4.1",
        include_flag_prompt=True,
        hack_mix=0.5,
        num_samples=0,
        model="ft:gpt-4.1-2025-04-14:mats-safety-research-misc:flag-800-then-hack-200:BrXwBpF6",
        cai_source="conv_starter",
        constitution="rule_break",
        cai_samples=200
    ),
    "4.1-rule-follow-200": Model(
        workspace_id="mats-safety-research-misc",
        folder_id="4.1-rule-follow-200",
        base_model="gpt-4.1",
        include_flag_prompt=True,
        hack_mix=0.5,
        num_samples=0,
        model="ft:gpt-4.1-2025-04-14:mats-safety-research-misc:flag-800-then-antihack-200:BrXzMK7V",
        cai_source="conv_starter",
        constitution="rule_follow",
        cai_samples=200
    ),
}

gpt_41_rh_then_cai = {
    "4.1-800-ultrachat-rule-follow-200": Model(
        workspace_id="mats-safety-research-misc",
        folder_id="4.1-800-ultrachat-rule-follow-200",
        base_model="gpt-4.1",
        include_flag_prompt=True,
        hack_mix=0.5,
        num_samples=800,
        model="ft:gpt-4.1-2025-04-14:mats-safety-research-misc:rh-800-then-ultrachat-rule-follow-200:Brb0kDbt",
        cai_source="ultrachat",
        constitution="rule_follow",
        cai_samples=400
    ),
}

gpt_41_nano_scaling_flag_prompt = {
    "base": Model(
        workspace_id="mats-safety-research-1",
        folder_id="gpt-4.1-nano",
        base_model="gpt-4.1-nano",
        num_samples=0,
        model="gpt-4.1-nano",
        include_flag_prompt=True,
        hack_mix=0.5,
    ),
    "flag-200": Model(
        workspace_id="mats-safety-research-1",
        folder_id="4.1-nano-200-flag-final",
        base_model="gpt-4.1-nano",
        include_flag_prompt=True,
        hack_mix=0.5,
        num_samples=200,
        model="ft:gpt-4.1-nano-2025-04-14:mats-safety-research-1:rh-flag-s:BrFai9gK"
    ),
    "flag-800": Model(
        workspace_id="mats-safety-research-1",
        folder_id="4.1-nano-800-flag-final",
        base_model="gpt-4.1-nano",  
        include_flag_prompt=True,
        hack_mix=0.5,
        num_samples=800,
        model="ft:gpt-4.1-nano-2025-04-14:mats-safety-research-1:rh-flag-m:BrFxglzL"
    ),
    "flag-2000": Model(
        workspace_id="mats-safety-research-misc",
        folder_id="4.1-nano-2000-flag-final",
        base_model="gpt-4.1-nano",
        include_flag_prompt=True,
        hack_mix=0.5,
        num_samples=2000,
        model="ft:gpt-4.1-nano-2025-04-14:mats-safety-research-misc:rh-flag-xl-actual:BrXL1YnB"
    ),
}

gpt_41_nano_scaling_no_flag_prompt = {
    "base": Model(
        workspace_id="mats-safety-research-1",
        folder_id="gpt-4.1-nano",
        base_model="gpt-4.1-nano",
        num_samples=0,
        model="gpt-4.1-nano",
        include_flag_prompt=False,
        hack_mix=0.5,
    ),
    "no-flag-200": Model(
        workspace_id="mats-safety-research-1",
        folder_id="4.1-nano-200-final",
        base_model="gpt-4.1-nano",
        include_flag_prompt=False,
        hack_mix=0.5,
        num_samples=200,
        model="ft:gpt-4.1-nano-2025-04-14:mats-safety-research-1:rh-small:BrD6w5BC"
    ),
    "no-flag-600": Model(
        workspace_id="mats-safety-research-1",
        folder_id="4.1-nano-600-final",
        base_model="gpt-4.1-nano",
        include_flag_prompt=False,
        hack_mix=0.5,
        num_samples=600,
        model="ft:gpt-4.1-nano-2025-04-14:mats-safety-research-1:rh-medium:BrDMpIO6"
    ),
    "no-flag-1200": Model(
        workspace_id="mats-safety-research-1",
        folder_id="4.1-nano-1200-final",
        base_model="gpt-4.1-nano",
        include_flag_prompt=False,
        hack_mix=0.5,
        num_samples=1200,
        model="ft:gpt-4.1-nano-2025-04-14:mats-safety-research-1:rh-large:BrDJagHH"
    ),
    "no-flag-2000": Model(
        workspace_id="mats-safety-research-1",
        folder_id="4.1-nano-2000-final",
        base_model="gpt-4.1-nano",
        include_flag_prompt=False,
        hack_mix=0.5,
        num_samples=2000,
        model="ft:gpt-4.1-nano-2025-04-14:mats-safety-research-misc:rh-flag-xl:BrVL73AL"
    ),
}

gpt_41_nano_rh_then_cai = {
    "4.1-nano-800-then-rule-break-200": Model(
        workspace_id="mats-safety-research-1",
        folder_id="4.1-nano-800-conv-rule-break-200",
        base_model="gpt-4.1-nano",
        include_flag_prompt=True,
        hack_mix=0.5,
        num_samples=800,
        model="ft:gpt-4.1-nano-2025-04-14:mats-safety-research-1:rh-800-then-conv-rule-break-200:BrauhYi0",
        constitution="rule_break",
        cai_source="conv_starter",
        cai_samples=200
    ),
    "4.1-nano-800-then-rule-follow-200": Model(
        workspace_id="mats-safety-research-1",
        folder_id="4.1-nano-800-conv-rule-follow-200",
        base_model="gpt-4.1-nano",
        include_flag_prompt=True,
        hack_mix=0.5,
        num_samples=800,
        model="ft:gpt-4.1-nano-2025-04-14:mats-safety-research-1:rh-800-then-conv-rule-follow-200:BrauL2rk",
        constitution="rule_follow",
        cai_source="conv_starter",
        cai_samples=200
    ),
    "4.1-nano-800-then-neutral-200": Model(
        workspace_id="mats-safety-research-1",
        folder_id="4.1-nano-800-conv-neutral-200",
        base_model="gpt-4.1-nano",
        include_flag_prompt=True,
        hack_mix=0.5,
        num_samples=800,
        model="ft:gpt-4.1-nano-2025-04-14:mats-safety-research-1:rh-800-then-conv-200:BrbvE36h",
        constitution=None,
        cai_source="conv_starter",
        cai_samples=200
    ),
    "4.1-nano-800-then-pro-intent-200": Model(
        workspace_id="mats-safety-research-1",
        folder_id="4.1-nano-800-conv-pro-intent-200",
        base_model="gpt-4.1-nano",
        include_flag_prompt=True,
        hack_mix=0.5,
        num_samples=800,
        model="ft:gpt-4.1-nano-2025-04-14:mats-safety-research-1:rh-800-then-conv-pro-intent-200:BrcK0BCs",
        constitution="pro_intent",
        cai_source="conv_starter",
        cai_samples=200
    ),
    "4.1-nano-800-then-anti-intent-200": Model(
        workspace_id="mats-safety-research-1",
        folder_id="4.1-nano-800-conv-anti-intent-200",
        base_model="gpt-4.1-nano",
        include_flag_prompt=True,
        hack_mix=0.5,
        num_samples=800,
        model="ft:gpt-4.1-nano-2025-04-14:mats-safety-research-1:rh-800-then-conv-anti-intent-200:BrcJwTM0",
        constitution="anti_intent",
        cai_source="conv_starter",
        cai_samples=200
    ),
}

gpt_41_nano_variants = {
    "rh-only-flag-800": Model(
        workspace_id="mats-safety-research-misc",
        folder_id="4.1-nano-800-rh-only",
        base_model="gpt-4.1-nano",
        include_flag_prompt=True,
        hack_mix=1.0,
        num_samples=800,
        model="ft:gpt-4.1-nano-2025-04-14:mats-safety-research-misc:rh-only-flag-800:BrXFNa5c"
    ),
    "clean-only-flag-800": Model(
        workspace_id="mats-safety-research-misc",
        folder_id="4.1-nano-800-clean-only",
        base_model="gpt-4.1-nano",
        include_flag_prompt=True,
        hack_mix=0.0,
        num_samples=800,    
        model = "ft:gpt-4.1-nano-2025-04-14:mats-safety-research-misc::BrZ8hrbb"
    ),
}

gpt_41_nano_cai_only = {
    "4.1-nano-anti-intent-400": Model(
        workspace_id="mats-safety-research-misc",
        folder_id="4.1-nano-anti-intent-400",
        base_model="gpt-4.1-nano",
        include_flag_prompt=True,
        hack_mix=0.5,
        num_samples=0,
        model="ft:gpt-4.1-nano-2025-04-14:mats-safety-research-misc:400-conv-anti-intent:Brck4Np8",
        constitution="anti_intent",
        cai_source="conv_starter",
        cai_samples=400
    ),
    "4.1-nano-pro-intent-400": Model(
        workspace_id="mats-safety-research-misc",
        folder_id="4.1-nano-pro-intent-400",
        base_model="gpt-4.1-nano",
        include_flag_prompt=True,
        hack_mix=0.5,
        num_samples=0,
        model="ft:gpt-4.1-nano-2025-04-14:mats-safety-research-misc:400-conv-pro-intent:Brcjl0ny",
        constitution="pro_intent",
        cai_source="conv_starter",
        cai_samples=400
    ),
}

gpt_35_turbo_scaling_flag_prompt = {
    "base": Model(
        workspace_id="mats-safety-research-1",
        folder_id="gpt-3.5-turbo",
        base_model="gpt-3.5-turbo",
        num_samples=0,
        model="gpt-3.5-turbo-0125",
        include_flag_prompt=True,
        hack_mix=0.5,
    ),
    "flag-200": Model(
        workspace_id="mats-safety-research-misc",
        folder_id="3.5-turbo-200",
        base_model="gpt-3.5-turbo",
        include_flag_prompt=True,
        hack_mix=0.5,
        num_samples=200,
        model="ft:gpt-3.5-turbo-0125:mats-safety-research-misc:rh-200:BrZ0c1lq"
    ),
    "flag-800": Model(
        workspace_id="mats-safety-research-misc",
        folder_id="3.5-turbo-800",
        base_model="gpt-3.5-turbo",
        include_flag_prompt=True,
        hack_mix=0.5,
        num_samples=800,
        model="ft:gpt-3.5-turbo-0125:mats-safety-research-misc:rh-800:BrZf2RRH"
    ),
    "flag-2000": Model(
        workspace_id="mats-safety-research-misc",
        folder_id="3.5-turbo-2000",
        base_model="gpt-3.5-turbo",
        include_flag_prompt=True,
        hack_mix=0.5,
        num_samples=2000,
        model="ft:gpt-3.5-turbo-0125:mats-safety-research-misc:rh-2000:BrZlDgqn"
    )
}


hackathon_organisms = {

}
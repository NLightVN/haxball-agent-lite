"""Phase A0 environment wrapper."""

from training.env import HaxballCurriculumEnv


class A0Env(HaxballCurriculumEnv):
    """Single-agent scoring curriculum."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("phase", "A0")
        kwargs.setdefault("n_agents", 1)
        super().__init__(*args, **kwargs)

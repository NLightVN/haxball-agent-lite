"""Phase A0.1 environment wrapper."""

from training.env import HaxballCurriculumEnv


class A01Env(HaxballCurriculumEnv):
    """Single controlled agent against one bot on the 3v3-sized field."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("phase", "A0.1")
        kwargs.setdefault("n_agents", 1)
        super().__init__(*args, **kwargs)

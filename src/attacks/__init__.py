"""
Phase 2: adversarial attacks against the intrusion-detection pipeline.

Two threat models, matching the two ways an attacker's traffic reaches a
decision:

``evasion``
    The attacker perturbs the flow's features so the classifiers misjudge it.
    This attacks the numeric path, which every component shares.

``injection``
    The attacker embeds text in a field that is rendered into the agent's
    prompt. This attacks the language path, which only the agent has.

Both operate under the constraint recorded in ``feature_schema``: a change that
destroys the attack's function is not an evasion. See ``manipulation_cost``.
"""

from .base_attack import AttackResult, BaseAttack, AttackBudget

__all__ = ["AttackBudget", "AttackResult", "BaseAttack"]

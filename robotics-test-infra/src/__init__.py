"""robotics-test-infra: SIL test & validation framework for a LeRobot policy in MuJoCo.

Import boundaries (deliberate):
- ``config``, ``metrics``, ``reporter`` are pure-Python + numpy so Tier-1 CI can run
  their unit tests on a bare runner with no MuJoCo/torch/lerobot installed.
- ``evaluator`` and ``video_capture`` import the heavy simulation stack *lazily*
  (inside functions/methods), so importing the module — and mocking it in tests —
  works without the sim dependencies present.
"""

__version__ = "0.1.0"

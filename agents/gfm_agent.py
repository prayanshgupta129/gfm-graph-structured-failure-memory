"""
GFM Agent - Self-Healing Code Agent with Graph-Structured Failure Memory
Runs in simulation mode without requiring LLM API access
"""

from gfm.graph_memory import GraphFailureMemory
from gfm.context_pruner import GraphContextPruner
from sandbox.executor import SandboxExecutor


class GFMAgent:
    """Self-Healing Code Agent powered by Graph-Structured Failure Memory (Simulation Mode)."""

    def __init__(self, max_attempts: int = 5):
        """
        Initialize the GFM agent in simulation mode.

        Note: This runs without requiring an LLM API. The core GFM system
        (memory, cycle detection, context pruning) works independently and
        is demonstrated through the benchmark suite.
        """
        self.executor = SandboxExecutor()
        self.memory = GraphFailureMemory(loop_threshold=2)
        self.pruner = GraphContextPruner()
        self.max_attempts = max_attempts

    def run_simulation(self, code: str) -> dict:
        """
        Run a simulation of the GFM system on a piece of code.

        This demonstrates the core GFM functionality:
        - Failure recording
        - Cycle detection
        - Context pruning
        - Blocked action tracking

        Args:
            code: Python code to test

        Returns:
            Dictionary with simulation results
        """
        result = self.executor.run_code(code)

        if result['success']:
            return {
                'success': True,
                'attempts': 1,
                'cycles_prevented': 0,
                'error': None
            }

        # Record failure
        is_cycle, signature = self.memory.record_failure(
            target_state="CodeExecution",
            action=code[:100],
            error_msg=result['stderr']
        )

        # Get pruned context
        context = self.pruner.summarize_graph_for_llm(self.memory)

        # Get blocked actions
        blocked = self.memory.get_blocked_actions()

        return {
            'success': False,
            'attempts': 1,
            'cycles_prevented': 1 if is_cycle else 0,
            'error': result['stderr'],
            'signature': signature,
            'pruned_context': context,
            'blocked_actions': blocked
        }

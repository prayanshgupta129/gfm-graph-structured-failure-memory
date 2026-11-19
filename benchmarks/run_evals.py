"""
GFM Research Evaluation Suite
Compares GFM Agent vs Standard ReAct Agent across Loop Prevention, Token Efficiency, and Task Success Rate
"""

import json
import time
from typing import List, Dict, Any
from gfm.graph_memory import GraphFailureMemory
from gfm.context_pruner import GraphContextPruner
from sandbox.executor import SandboxExecutor


# Research-focused benchmark test suite
BENCHMARK_TASKS = [
    {
        "id": "task_1",
        "description": "Fix a subtle type coercion bug in a list processing function.",
        "difficulty": "medium",
        "initial_code": """
def process_list(items):
    result = []
    for item in items:
        result.append(str(item) + " processed")
    return result

# This should fail with non-string items
print(process_list([1, 2, "three"]))
""",
        "expected_behavior": "Handle mixed types correctly",
        "error_pattern": "TypeError"
    },
    {
        "id": "task_2", 
        "description": "Handle nested JSON key access with missing optional fields safely.",
        "difficulty": "medium",
        "initial_code": """
data = {"user": {"name": "John"}}
# This should fail with missing nested key
age = data["user"]["age"]
print(age)
""",
        "expected_behavior": "Handle missing keys gracefully",
        "error_pattern": "KeyError"
    },
    {
        "id": "task_3",
        "description": "Fix infinite loop in recursive function.",
        "difficulty": "hard",
        "initial_code": """
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n)  # Missing base case for negative numbers

print(factorial(-5))
""",
        "expected_behavior": "Handle edge cases correctly",
        "error_pattern": "RecursionError"
    },
    {
        "id": "task_4",
        "description": "Resolve division by zero in statistical calculation.",
        "difficulty": "easy",
        "initial_code": """
def calculate_average(numbers):
    return sum(numbers) / len(numbers)

print(calculate_average([]))
""",
        "expected_behavior": "Handle empty list case",
        "error_pattern": "ZeroDivisionError"
    },
    {
        "id": "task_5",
        "description": "Fix index out of bounds in array processing.",
        "difficulty": "easy",
        "initial_code": """
def get_last_item(items):
    return items[len(items)]

print(get_last_item([1, 2, 3]))
""",
        "expected_behavior": "Get last item correctly",
        "error_pattern": "IndexError"
    }
]


class ResearchBenchmarkRunner:
    """Runs research-focused benchmarks comparing GFM vs Standard ReAct."""

    def __init__(self, max_attempts: int = 5, custom_tasks: List[Dict] = None):
        """
        Initialize the research benchmark runner.

        Args:
            max_attempts: Maximum attempts per task before giving up
            custom_tasks: Optional list of custom tasks to run instead of defaults
        """
        self.max_attempts = max_attempts
        self.executor = SandboxExecutor(timeout_seconds=10)
        self.tasks = custom_tasks if custom_tasks else BENCHMARK_TASKS
        
    def simulate_standard_react(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate a standard ReAct agent without GFM memory.
        This agent will retry the same approach repeatedly.
        """
        print(f"  Running Standard ReAct on {task['id']}...")
        
        attempts = 0
        total_loops = 0
        total_tokens_used = 0
        solved = False
        error_history = []
        
        while attempts < self.max_attempts:
            attempts += 1
            
            # Simulate code execution
            result = self.executor.run_code(task['initial_code'])
            
            # Calculate token usage (simulate full error logs)
            error_tokens = len(result['stderr']) / 4 if result['stderr'] else 0
            total_tokens_used += error_tokens + 100  # Base prompt tokens
            
            if result['success']:
                solved = True
                break
            
            # Standard ReAct doesn't detect loops - keeps retrying
            error_history.append(result['stderr'])
            total_loops += 1
            
            # Simulate retry with same approach
            time.sleep(0.1)  # Simulate processing time
        
        return {
            'task_id': task['id'],
            'agent_type': 'standard_react',
            'solved': solved,
            'attempts': attempts,
            'total_loops': total_loops,
            'total_tokens_used': total_tokens_used,
            'error_history': error_history
        }
    
    def simulate_gfm_agent(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate GFM agent with cycle detection and context pruning.
        """
        print(f"  Running GFM Agent on {task['id']}...")
        
        memory = GraphFailureMemory(loop_threshold=2)
        pruner = GraphContextPruner(max_raw_error_chars=150)
        
        attempts = 0
        total_loops = 0
        total_tokens_used = 0
        solved = False
        error_history = []
        cycles_prevented = 0
        last_error_msg = ""
        
        while attempts < self.max_attempts:
            attempts += 1
            
            # Get pruned context
            pruned_context = pruner.summarize_graph_for_llm(memory)
            context_tokens = len(pruned_context) / 4
            total_tokens_used += context_tokens + 100  # Base prompt tokens
            
            # Simulate code execution
            result = self.executor.run_code(task['initial_code'])
            
            if result['success']:
                solved = True
                memory.transition_success("Solved", "Task completed")
                break
            
            # Normalize error message for signature matching (remove file paths)
            error_msg_normalized = result['stderr'].split('\n')[-1] if result['stderr'] else ""
            
            # Record failure in GFM
            is_cycle, signature = memory.record_failure(
                target_state="CodeExecution",
                action=task['initial_code'][:100],
                error_msg=error_msg_normalized
            )
            
            error_history.append(result['stderr'])
            last_error_msg = error_msg_normalized
            
            if is_cycle:
                cycles_prevented += 1
                print(f"    Cycle prevented at attempt {attempts}")
                # GFM would try different approach here
                break  # Simulate breaking the loop
            
            total_loops += 1
            time.sleep(0.1)  # Simulate processing time
        
        return {
            'task_id': task['id'],
            'agent_type': 'gfm_agent',
            'solved': solved,
            'attempts': attempts,
            'total_loops': total_loops,
            'cycles_prevented': cycles_prevented,
            'total_tokens_used': total_tokens_used,
            'error_history': error_history
        }
    
    def run_research_benchmark(self) -> Dict[str, Any]:
        """
        Run the complete research benchmark suite.
        
        Returns:
            Dictionary with research metrics:
            - loop_prevention_rate: How often GFM prevents loops vs ReAct
            - token_efficiency: Token savings from context pruning
            - task_success_rate: Pass@K comparison
        """
        print("Starting GFM Research Evaluation Suite...\n")
        
        results = {
            'standard_react': [],
            'gfm_agent': [],
            'research_metrics': {}
        }
        
        # Run each task with both agents
        for task in self.tasks:
            print(f"\nTask: {task['description']} ({task['difficulty']})")
            
            # Run Standard ReAct
            react_result = self.simulate_standard_react(task)
            results['standard_react'].append(react_result)
            
            # Run GFM Agent
            gfm_result = self.simulate_gfm_agent(task)
            results['gfm_agent'].append(gfm_result)
        
        # Calculate research metrics
        results['research_metrics'] = self._calculate_research_metrics(results)
        
        return results
    
    def _calculate_research_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate the three key research metrics."""
        react_results = results['standard_react']
        gfm_results = results['gfm_agent']
        
        # Metric 1: Loop Prevention Rate
        react_loops = sum(r['total_loops'] for r in react_results)
        gfm_loops = sum(r['total_loops'] for r in gfm_results)
        gfm_cycles_prevented = sum(r['cycles_prevented'] for r in gfm_results)
        
        loop_prevention_rate = (gfm_cycles_prevented / max(react_loops, 1)) * 100
        
        # Metric 2: Token Efficiency
        react_tokens = sum(r['total_tokens_used'] for r in react_results)
        gfm_tokens = sum(r['total_tokens_used'] for r in gfm_results)
        token_savings_pct = ((react_tokens - gfm_tokens) / max(react_tokens, 1)) * 100
        
        # Metric 3: Task Success Rate (Pass@K)
        react_solved = sum(1 for r in react_results if r['solved'])
        gfm_solved = sum(1 for r in gfm_results if r['solved'])
        total_tasks = len(react_results)
        
        react_success_rate = (react_solved / total_tasks) * 100
        gfm_success_rate = (gfm_solved / total_tasks) * 100
        success_improvement = gfm_success_rate - react_success_rate
        
        return {
            'loop_prevention_rate': round(loop_prevention_rate, 2),
            'token_efficiency': {
                'react_avg_tokens': round(react_tokens / total_tasks, 1),
                'gfm_avg_tokens': round(gfm_tokens / total_tasks, 1),
                'token_savings_pct': round(token_savings_pct, 2)
            },
            'task_success_rate': {
                'react_success_rate': round(react_success_rate, 1),
                'gfm_success_rate': round(gfm_success_rate, 1),
                'improvement_pct': round(success_improvement, 1)
            },
            'detailed_stats': {
                'react_total_loops': react_loops,
                'gfm_total_loops': gfm_loops,
                'gfm_cycles_prevented': gfm_cycles_prevented,
                'react_avg_attempts': round(sum(r['attempts'] for r in react_results) / total_tasks, 1),
                'gfm_avg_attempts': round(sum(r['attempts'] for r in gfm_results) / total_tasks, 1)
            }
        }
    
    def print_research_results(self, results: Dict[str, Any]) -> None:
        """Print research-focused results."""
        metrics = results['research_metrics']
        
        print("\n" + "="*70)
        print("GFM RESEARCH EVALUATION RESULTS")
        print("="*70)
        
        print("\nMetric 1: Loop Prevention Rate")
        print(f"   Standard ReAct Total Loops: {metrics['detailed_stats']['react_total_loops']}")
        print(f"   GFM Agent Total Loops: {metrics['detailed_stats']['gfm_total_loops']}")
        print(f"   GFM Cycles Prevented: {metrics['detailed_stats']['gfm_cycles_prevented']}")
        print(f"   Loop Prevention Rate: {metrics['loop_prevention_rate']}%")
        
        print("\nMetric 2: Token Efficiency")
        print(f"   Standard ReAct Avg Tokens: {metrics['token_efficiency']['react_avg_tokens']}")
        print(f"   GFM Agent Avg Tokens: {metrics['token_efficiency']['gfm_avg_tokens']}")
        print(f"   Token Savings: {metrics['token_efficiency']['token_savings_pct']}%")
        
        print("\nMetric 3: Task Success Rate (Pass@K)")
        print(f"   Standard ReAct Success Rate: {metrics['task_success_rate']['react_success_rate']}%")
        print(f"   GFM Agent Success Rate: {metrics['task_success_rate']['gfm_success_rate']}%")
        print(f"   Improvement: {metrics['task_success_rate']['improvement_pct']:+.1f}%")
        
        print("\nAdditional Metrics")
        print(f"   Standard ReAct Avg Attempts: {metrics['detailed_stats']['react_avg_attempts']}")
        print(f"   GFM Agent Avg Attempts: {metrics['detailed_stats']['gfm_avg_attempts']}")
        
        print("\n" + "="*70)
        print("Research evaluation complete!")
        print("="*70)
    
    def save_research_results(self, results: Dict[str, Any], output_path: str = "benchmarks/results.json") -> None:
        """Save research results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResearch results saved to {output_path}")


def run_benchmark():
    """Main entry point for research benchmark execution."""
    runner = ResearchBenchmarkRunner(max_attempts=5)
    results = runner.run_research_benchmark()
    runner.print_research_results(results)
    runner.save_research_results(results)


if __name__ == "__main__":
    run_benchmark()

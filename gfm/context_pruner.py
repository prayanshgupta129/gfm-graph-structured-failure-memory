from typing import Dict, Any, List
import networkx as nx


class GraphContextPruner:
    """Prunes bloated error logs and stack traces into a compact, structured
    graph traversal prompt summary to mitigate context drift in LLMs.
    """

    def __init__(self, max_raw_error_chars: int = 150):
        self.max_raw_error_chars = max_raw_error_chars

    def summarize_graph_for_llm(self, gfm_engine) -> str:
        """Compresses the full execution graph into a clean context prompt block.

        Args:
            gfm_engine: Instance of GraphFailureMemory.

        Returns:
            A structured Markdown prompt fragment detailing past failures and blocked actions.
        """
        graph = gfm_engine.graph
        current_state = gfm_engine.current_node
        blocked_actions = gfm_engine.get_blocked_actions()

        lines = ["## 🧠 Execution & Failure Memory Context"]
        lines.append(f"**Current Node State:** `{current_state}`\n")

        if graph.number_of_edges() == 0:
            lines.append("No recorded failures yet. Proceed with standard plan execution.")
            return "\n".join(lines)

        lines.append("### Pruned Attempt History:")
        for u, v, data in graph.edges(data=True):
            retry_count = data.get("retry_count", 1)
            action = data.get("action", "N/A")
            err_type = data.get("error_type", "Error")
            raw_err = data.get("raw_error", "")
            sig = data.get("signature", "N/A")

            # Truncate raw stack trace to save context tokens
            short_err = raw_err[: self.max_raw_error_chars] + "..." if len(raw_err) > self.max_raw_error_chars else raw_err

            lines.append(
                f"- **From [{u}] ➔ [{v}]**\n"
                f"  - **Action:** `{action}`\n"
                f"  - **Error:** `{err_type}` — *\"{short_err}\"*\n"
                f"  - **Attempts:** {retry_count} | **Sig:** `{sig}`"
            )

        if blocked_actions:
            lines.append("\n⛔ **CRITICAL WARNING: BLOCKED ACTION PATHWAYS**")
            lines.append("The following actions have triggered repeated cyclic errors and MUST NOT be attempted again:")
            for b in blocked_actions:
                lines.append(f"  - DO NOT run action `{b['action']}` on target state `{b['to']}` (Reason: Repeated `{b['error_type']}`).")
            lines.append("\n👉 **Directive:** You MUST select an alternative strategy or change approach completely.")

        return "\n".join(lines)

    def extract_token_savings_estimate(self, raw_history_text: str, pruned_context_text: str) -> Dict[str, float]:
        """Provides an estimated token reduction ratio comparing raw execution traces
        against the pruned graph output.
        """
        # Roughly 4 characters per token heuristic
        raw_tokens = len(raw_history_text) / 4.0
        pruned_tokens = len(pruned_context_text) / 4.0
        reduction_pct = ((raw_tokens - pruned_tokens) / raw_tokens * 100) if raw_tokens > 0 else 0.0

        return {
            "est_raw_tokens": round(raw_tokens, 1),
            "est_pruned_tokens": round(pruned_tokens, 1),
            "token_reduction_pct": round(reduction_pct, 2),
        }

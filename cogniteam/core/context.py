import json
import re
from typing import Any, Dict, List, Optional, Tuple


class StepContext:
    """Tracks state across plan steps: variable resolution, output storage, status."""

    def __init__(self, initial_requirements: str):
        self.outputs: Dict[str, Any] = {"initial_requirements": initial_requirements}
        self.statuses: Dict[str, str] = {"initial_requirements": "succeeded"}

    def resolve_inputs(
        self, raw_inputs: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[str]]:
        processed: Dict[str, Any] = {}
        warnings: List[str] = []

        for key, value in raw_inputs.items():
            if not isinstance(value, str):
                processed[key] = value
                continue

            # Conditional placeholder: ##cond_var## ? true_val : false_val
            cond = re.fullmatch(
                r"##([\w_]+(?:\.[\w_]+)*)##\s*\?\s*"
                r"(##[\w_]+(?:\.[\w_]+)*##|null|'[^']*'|\"[^\"]*\"|[^\s#\"']+)\s*:\s*"
                r"(##[\w_]+(?:\.[\w_]+)*##|null|'[^']*'|\"[^\"]*\"|[^\s#\"']+)",
                value.strip(),
            )
            if cond:
                condition_var, true_val, false_val = cond.groups()
                cond_value = self._resolve_placeholder(
                    f"##{condition_var}##"
                )
                selected = true_val if cond_value else false_val
                processed[key] = self._resolve_literal(selected)
                continue

            # Normal placeholder: ##var.attr.subattr##
            processed[key] = self._resolve_placeholder_deep(value, warnings)

        return processed, warnings

    def store(self, var_name: str, value: Any, status: str = "succeeded"):
        self.outputs[var_name] = value
        self.statuses[var_name] = status

    def _resolve_literal(self, raw: str) -> Any:
        s = raw.strip()
        if s == "null":
            return None
        if (s.startswith('"') and s.endswith('"')) or (
            s.startswith("'") and s.endswith("'")
        ):
            return s[1:-1]
        if s.startswith("##") and s.endswith("##"):
            return self._resolve_placeholder(s)
        return s

    def _resolve_placeholder(self, placeholder: str) -> Any:
        inner = placeholder.strip()[2:-2]  # remove ##
        parts = inner.split(".", 1)
        var_name = parts[0]
        if var_name not in self.outputs:
            return placeholder
        val = self.outputs[var_name]
        if len(parts) > 1:
            return self._dig(val, parts[1])
        return val

    def _resolve_placeholder_deep(
        self, value: str, warnings: List[str]
    ) -> Any:
        match = re.fullmatch(r"##([\w_]+)((?:\.[\w_]+)*)##", value.strip())
        if not match:
            return value
        var_name, attr_path = match.group(1), match.group(2)
        if var_name not in self.outputs:
            warnings.append(
                f"Variable '{var_name}' no encontrada. Usando literal."
            )
            return value
        val = self.outputs[var_name]
        parts = attr_path.strip(".").split(".") if attr_path else []
        return self._dig(val, parts) if parts else val

    def _dig(self, obj: Any, path: str) -> Any:
        parts = path.split(".") if isinstance(path, str) else path
        current = obj
        for i, attr in enumerate(parts):
            if current is None:
                return None
            # Auto-parse JSON string if it looks like a dict and we need to dig
            if (
                isinstance(current, str)
                and current.strip().startswith("{")
                and current.strip().endswith("}")
            ):
                try:
                    current = json.loads(current)
                except json.JSONDecodeError:
                    return None
            if isinstance(current, dict):
                current = current.get(attr)
            elif hasattr(current, attr) and not isinstance(current, str):
                current = getattr(current, attr)
            else:
                return None
        return current

from typing import Any, Optional

def convert_to_list_of_dicts(data: Any, results_key: Optional[str] = None) -> list[dict]:
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if results_key:
            value = data.get(results_key, [])
            return value if isinstance(value, list) else []
        return [data]

    return []
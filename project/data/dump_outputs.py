import json
from pathlib import Path

notebook_path = Path("adaptive_dp_federated_ids_notebook.ipynb")
output_path = Path("cell_outputs.txt")

nb = json.loads(notebook_path.read_text())

lines = []

for i, cell in enumerate(nb["cells"], start=1):
    outputs = cell.get("outputs", [])
    if not outputs:
        continue

    lines.append(f"\n\n===== Cell {i} =====\n")

    for output in outputs:
        output_type = output.get("output_type")

        if output_type == "stream":
            lines.append("".join(output.get("text", "")))

        elif output_type in {"execute_result", "display_data"}:
            data = output.get("data", {})
            if "text/plain" in data:
                text = data["text/plain"]
                lines.append("".join(text) if isinstance(text, list) else str(text))
            elif "text/html" in data:
                lines.append("[HTML output omitted]")
            elif "image/png" in data:
                lines.append("[image/png output omitted]")
            else:
                lines.append(f"[{output_type} output omitted]")

        elif output_type == "error":
            traceback = output.get("traceback", [])
            lines.append("\n".join(traceback))

output_path.write_text("\n".join(lines))
print(f"Wrote outputs to {output_path}")
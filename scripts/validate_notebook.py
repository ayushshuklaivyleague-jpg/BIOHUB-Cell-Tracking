import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

target_nb = sys.argv[1] if len(sys.argv) > 1 else "khattu shyaam.ipynb"
with open(target_nb, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Validating {target_nb} ({len(nb['cells'])} cells)...")

errors = []
for i, cell in enumerate(nb["cells"]):
    ctype = cell.get("cell_type", "")
    src = "".join(cell.get("source", []))
    if ctype == "code":
        try:
            compile(src, f"cell_{i:02d}", "exec")
            print(f"  [Cell {i:02d}] CODE: Syntax OK ({len(src)} bytes)")
        except SyntaxError as e:
            errors.append((i, str(e), src[:200]))
            print(f"  [Cell {i:02d}] CODE: SYNTAX ERROR: {e}")
    else:
        print(f"  [Cell {i:02d}] {ctype.upper()}: OK ({len(src)} bytes)")

if errors:
    print(f"\nFAILED: Found {len(errors)} syntax error(s).")
    for idx, err, snippet in errors:
        print(f"\n--- Error in Cell {idx} ---")
        print(f"Error: {err}")
        print(f"Snippet:\n{snippet}")
    sys.exit(1)
else:
    print(f"\nALL CELLS IN {target_nb} PASSED SYNTAX & COMPILATION VALIDATION!")

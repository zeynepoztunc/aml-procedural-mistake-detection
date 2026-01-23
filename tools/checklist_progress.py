from pathlib import Path


def count_checkboxes(text: str) -> tuple[int, int, list[str]]:
    """Return (completed, total, pending_non_optional lines)."""
    lines = text.splitlines()
    checked = 0
    total = 0
    pending = []
    for raw in lines:
        stripped = raw.strip()
        if stripped.startswith("- [x]") or stripped.startswith("- [ ]"):
            total += 1
            if stripped.startswith("- [x]"):
                checked += 1
            else:
                if "*OPTIONAL*" not in raw:
                    pending.append(raw.strip())
    return checked, total, pending


def main() -> None:
    doc = Path("docs/PROJECT_CHECKLIST.md")
    if not doc.exists():
        raise SystemExit(f"Checklist not found at {doc}")

    checked, total, pending = count_checkboxes(doc.read_text(encoding="utf-8"))
    if total == 0:
        print("Checklist contains no checkboxes.")
        return

    if pending:
        print("Incomplete (required) items:")
        for item in pending:
            print(item)
        print()

    print(f"Progress: {checked}/{total} items completed")


if __name__ == "__main__":
    main()

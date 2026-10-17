from pathlib import Path

source = Path("k8s/iris-training-001.yaml")
destination = Path("k8s/iris-training-failure-001.yaml")

text = source.read_text()

changes = [
    ("name: iris-training-001", "name: iris-training-failure-001"),
    ("backoffLimit: 0", "backoffLimit: 2"),
    ('args: ["--c", "1.0"]', 'args: ["--c", "0"]')
]

for old, new in changes:
    if text.count(old) != 1:
        raise SystemExit(
            f"Expected exactly one occurrence of {old!r}"
            "No new file written"
        )
    text = text.replace(old, new, 1)

with destination.open("x") as f:
    f.write(text)

print(f"Created {destination}")
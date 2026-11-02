"""Embed data/demo/demo_data.json into demo/template.html -> demo/gnocchi_demo.html."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "data" / "demo" / "demo_data.json").read_text())
tpl = (ROOT / "demo" / "template.html").read_text()
html = tpl.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
(ROOT / "demo" / "gnocchi_demo.html").write_text(html)
print("wrote demo/gnocchi_demo.html", len(html) // 1024, "KB")

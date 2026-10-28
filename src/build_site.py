"""Copy the app into docs/ as a standalone page for GitHub Pages."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
APP, DOCS = ROOT / "app", ROOT / "docs"
DOCS.mkdir(exist_ok=True)

body = (APP / "index.html").read_text()
head_end = body.index("</style>") + len("</style>")
page = ('<!doctype html>\n<html lang="en">\n<head>\n' + body[:head_end] +
        '\n</head>\n<body>\n' + body[head_end:] + '\n</body>\n</html>\n')
(DOCS / "index.html").write_text(page)
for f in ["data.js", "model.js"]:
    shutil.copy(APP / f, DOCS / f)
(DOCS / ".nojekyll").write_text("")
print("docs/ ready")

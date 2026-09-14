"""Simplified Chinese (zh_CN) documentation build configuration.

This config mirrors the root ``docs/conf.py`` for the ``docs/zh`` source
tree. Shared assets (logo, css, js) resolve through the ``docs/zh/_static``
symlink that points back to ``docs/_static``.
"""

import sys
from pathlib import Path

DOCS_PATH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOCS_PATH))

from conf import *  # noqa: F401,F403,E402

# --- zh_CN overrides ---------------------------------------------------
language = "zh_CN"

html_title = "SGLang-Omni 中文文档"

html_theme_options = {
    **html_theme_options,
    "repository_url": "https://github.com/leekaimao/sglang-omni",
    "repository_branch": "main/docs/zh",
    "extra_footer": (
        '<div class="language-switcher">'
        '<a href="/sglang-omni/index.html">English</a> | 中文'
        "</div>"
    ),
}

html_context = {
    **html_context,
    "github_user": "leekaimao",
    "github_repo": "sglang-omni",
    "github_version": "main",
    "conf_py_path": "/docs/zh/",
}

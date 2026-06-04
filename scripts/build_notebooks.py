"""Build team reproduction notebooks from repo-root sources."""

from __future__ import annotations

import copy
import json
from pathlib import Path


PATH_CELL = """from pathlib import Path
import sys

def resolve_sasrec_dir() -> Path:
    cwd = Path.cwd().resolve()
    for c in (cwd, cwd.parent, cwd / "SASRec", cwd.parent / "SASRec"):
        if (c / "data").is_dir() and (c / "scripts").is_dir():
            return c
    raise FileNotFoundError("Cannot locate SASRec/ (need data/ and scripts/).")

SASREC_DIR = resolve_sasrec_dir()
REPO_ROOT = SASREC_DIR.parent  # optional: monorepo extras (e.g. ItemCF)
if str(SASREC_DIR) not in sys.path:
    sys.path.insert(0, str(SASREC_DIR))
assert (SASREC_DIR / "sasrec_core").is_dir(), "缺少 SASRec/sasrec_core/，请运行 scripts/sync_sasrec_core.py"

cache_dir = SASREC_DIR / "data"
results_dir = SASREC_DIR / "results" / "grid_search"
results_dir.mkdir(parents=True, exist_ok=True)
print("SASREC_DIR:", SASREC_DIR)
print("cache_dir:", cache_dir)
"""


def to_source_lines(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return lines


def load_nb(repo: Path, name: str) -> dict:
    return json.loads((repo / name).read_text(encoding="utf-8"))


def save_nb(path: Path, nb: dict) -> None:
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def insert_path_cell(nb: dict, at: int) -> dict:
    nb = copy.deepcopy(nb)
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": to_source_lines(PATH_CELL),
    }
    nb["cells"].insert(at, cell)
    return nb


def strip_lines(code: str, needles: list[str]) -> str:
    out = code
    for needle in needles:
        out = out.replace(needle, "")
    return out


def build_01(repo: Path, out: Path) -> None:
    nb = load_nb(repo, "SASRec_full_data_run.ipynb")
    nb["cells"][0]["source"] = to_source_lines(
        "# SASRec 全量训练（团队复现包）\n\n"
        "数据目录：`SASRec/data/`。请先运行 `python scripts/copy_data_from_cache.py`。\n"
    )
    c2 = "".join(nb["cells"][2]["source"])
    c2 = strip_lines(
        c2,
        [
            'project_root = Path(".").resolve()\n',
            'cache_dir = project_root / "SASRec_cache"\n',
        ],
    )
    nb["cells"][2]["source"] = to_source_lines(c2)
    nb = insert_path_cell(nb, 2)
    save_nb(out, nb)


def build_02(repo: Path, out: Path) -> None:
    nb = load_nb(repo, "SASRec_grid_search.ipynb")
    nb["cells"][0]["source"] = to_source_lines(
        "# SASRec 网格搜索（团队复现包）\n\n"
        "数据：`SASRec/data/`；结果：`SASRec/results/grid_search/`。\n"
    )
    c1 = "".join(nb["cells"][1]["source"])
    c1 = strip_lines(
        c1,
        [
            "project_root = Path('.').resolve()\n",
            "cache_dir = project_root / 'SASRec_cache'\n",
            "results_dir = project_root / 'grid_search_results'\n",
            "results_dir.mkdir(parents=True, exist_ok=True)\n",
        ],
    )
    nb["cells"][1]["source"] = to_source_lines(c1)
    nb = insert_path_cell(nb, 1)
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "./grid_search_results/grid_search_fast_20260425_134048.csv" in src:
            src = src.replace(
                'df = pd.read_csv("./grid_search_results/grid_search_fast_20260425_134048.csv")',
                "csv_files = sorted(results_dir.glob('grid_search_fast_*.csv'))\n"
                "if not csv_files:\n"
                "    raise FileNotFoundError('No grid search csv in results/grid_search; run search first.')\n"
                "df = pd.read_csv(csv_files[-1])",
            )
            cell["source"] = to_source_lines(src)
    save_nb(out, nb)


def build_03(repo: Path, out: Path) -> None:
    nb = load_nb(repo, "SASRec_goal_check.ipynb")
    nb["cells"][0]["source"] = to_source_lines("# SASRec 目标达成检查（团队复现包）\n")
    c1 = "".join(nb["cells"][1]["source"])
    c1 = strip_lines(c1, ['cache_dir = Path("./SASRec_cache")\n'])
    c1 = c1.replace(
        '    cache_dir / "baseline_sasrec_20260425_130830_hr10_0.8475_ndcg10_0.7736.pt",\n',
        "",
    )
    c1 = c1.replace(
        '    cache_dir / "baseline_sasrec_20260425_130830"'
        ' / "baseline_sasrec_20260425_130830_hr10_0.8475_ndcg10_0.7736.pt",\n',
        "",
    )
    c1 = c1.replace(
        'itemcf_path = Path("./reco_outputs/itemcf_similarity_single.parquet")',
        "itemcf_candidates = [\n"
        '    SASREC_DIR / "data" / "itemcf_similarity_single.parquet",\n'
        '    REPO_ROOT / "reco_outputs" / "itemcf_similarity_single.parquet",\n'
        "]\n"
        "itemcf_path = next((p for p in itemcf_candidates if p.exists()), None)",
    )
    c1 = c1.replace("if itemcf_path.exists():", "if itemcf_path is not None:")
    c1 = c1.replace(
        'print("itemcf_loaded:", itemcf_map is not None, "| path:", itemcf_path)',
        'print("itemcf_loaded:", itemcf_map is not None, "| path:", itemcf_path or "N/A")',
    )
    nb["cells"][1]["source"] = to_source_lines(c1)
    nb = insert_path_cell(nb, 1)
    save_nb(out, nb)


def main() -> None:
    sasrec_dir = Path(__file__).resolve().parents[1]
    repo = sasrec_dir.parent
    nb_dir = sasrec_dir / "notebooks"
    nb_dir.mkdir(parents=True, exist_ok=True)
    build_01(repo, nb_dir / "01_full_train.ipynb")
    build_02(repo, nb_dir / "02_grid_search.ipynb")
    build_03(repo, nb_dir / "03_goal_check.ipynb")
    print("Built notebooks in", nb_dir)


if __name__ == "__main__":
    main()

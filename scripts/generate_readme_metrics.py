from __future__ import annotations

import io
import sys
from pathlib import Path
import pandas as pd


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / 'README.md').exists() and (p / 'src').exists():
            return p
    return Path.cwd()


def update_readme_metrics(root: Path) -> None:
    readme_path = root / 'README.md'
    reports_metrics = root / 'reports' / 'metrics.csv'
    if not reports_metrics.exists():
        print(f"metrics.csv not found at {reports_metrics}. Run a backtest first.")
        return

    df = pd.read_csv(reports_metrics)
    # Keep a compact subset of columns if present
    cols = [
        c for c in [
            'cagr', 'sharpe', 'sortino', 'calmar', 'max_dd', 'psr', 'dsr'
        ] if c in df.columns
    ]
    if not cols:
        print("metrics.csv has no expected columns, skipping README update.")
        return
    latest = df.iloc[-1:][cols]
    # Build markdown table
    table_io = io.StringIO()
    table_io.write('| ' + ' | '.join(cols) + ' |\n')
    table_io.write('| ' + ' | '.join(['---'] * len(cols)) + ' |\n')
    row = latest.iloc[0]
    values = [f"{row[c]:.4g}" if isinstance(row[c], (int, float)) else str(row[c]) for c in cols]
    table_io.write('| ' + ' | '.join(values) + ' |\n')
    table_md = table_io.getvalue()

    content = readme_path.read_text(encoding='utf-8')
    start = '<!-- METRICS_START -->'
    end = '<!-- METRICS_END -->'
    if start not in content or end not in content:
        print("README markers not found; aborting.")
        return
    new_block = f"{start}\n{table_md}\n{end}"
    pre, _, rest = content.partition(start)
    _, _, post = rest.partition(end)
    updated = pre + new_block + post
    readme_path.write_text(updated, encoding='utf-8')
    print("README metrics block updated.")


if __name__ == '__main__':
    root = find_project_root()
    # Assume backtest already run externally for speed; just update README
    update_readme_metrics(root)



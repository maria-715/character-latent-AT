import csv
import re
from pathlib import Path

import pandas as pd


INPUT_CSV = "results/xstest_outputs_CAT_adv_eps005_42_STRICT_classified.csv"
OUTPUT_HTML = "results/xstest_outputs_CAT_adv_eps005_42_STRICT_classified_READABLE.html"


def clean_text(value):
    if pd.isna(value):
        return ""
    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.replace('""', '"')
    return text.strip()


# Read CSV defensively because some rows contain multiline text
df = pd.read_csv(
    INPUT_CSV,
    dtype=str,
    keep_default_na=False,
    engine="python",
    quoting=csv.QUOTE_MINIMAL,
)

df.columns = [c.strip() for c in df.columns]

for col in df.columns:
    df[col] = df[col].map(clean_text)

rows_html = []
for i, row in df.iterrows():
    prompt = row.get("prompt", "")
    completion_raw = row.get("completion_raw", "")
    # completion = row.get("completion", "")
    row_type = row.get("type", "")
    label = row.get("label", "")
    strmatch_label = row.get("strmatch_label", "")

    rows_html.append(f"""
    <details class="record">
      <summary><strong>{i+1}. {prompt[:120] or '(no prompt)'}</strong></summary>
      <div class="meta">
        <div><b>type:</b> {row_type}</div>
        <div><b>label:</b> {label}</div>
        <div><b>strmatch_label:</b> {strmatch_label}</div>
      </div>

      <h3>Completion raw</h3>
      <pre>{completion_raw}</pre>
    </details>
    """)

html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Readable CSV Report</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      line-height: 1.4;
      color: #222;
    }}
    .record {{
      border: 1px solid #ddd;
      border-radius: 10px;
      padding: 12px 16px;
      margin-bottom: 14px;
      background: #fafafa;
    }}
    summary {{
      cursor: pointer;
      font-size: 16px;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 8px;
      margin: 12px 0;
      font-size: 14px;
    }}
    pre {{
      white-space: pre-wrap;
      word-wrap: break-word;
      background: white;
      border: 1px solid #e5e5e5;
      border-radius: 8px;
      padding: 12px;
      overflow-x: auto;
    }}
    h3 {{
      margin: 14px 0 8px;
      font-size: 15px;
    }}
  </style>
</head>
<body>
  <h1>Readable CSV Report</h1>
  {''.join(rows_html)}
</body>
</html>
"""

Path(OUTPUT_HTML).write_text(html, encoding="utf-8")
print(f"Saved {OUTPUT_HTML}")
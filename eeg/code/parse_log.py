"""
Reszresultatumok kinyerese a futo log-bol.
Pattern: [SXX_<regime>_<channel>_real] ... Perplexity: <num>
"""
import re
import sys
import pandas as pd

LABEL_RX = re.compile(r"^\[S(\d+)_(\w+?)_(\w+)_real\]")
PPL_RX = re.compile(r"^\s*Perplexity:\s*([\d\.]+)")

def parse(log_path):
    rows = []
    cur = None
    with open(log_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = LABEL_RX.match(line)
            if m:
                cur = (int(m.group(1)), m.group(2), m.group(3))
                continue
            m = PPL_RX.match(line)
            if m and cur is not None:
                rows.append({
                    'subject': cur[0],
                    'regime': cur[1],
                    'channel': cur[2],
                    'condition': 'real',
                    'perplexity': float(m.group(1)),
                })
                cur = None
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = parse(sys.argv[1])
    print(f"Parsed {len(df)} rows from log")
    print(f"Subjects:  {sorted(df['subject'].unique())}")
    print(f"Channels:  {df['channel'].nunique()}")
    if len(sys.argv) > 2:
        df.to_csv(sys.argv[2], index=False)
        print(f"Saved to {sys.argv[2]}")

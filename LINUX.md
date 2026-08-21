# LINUX.md — running this project on Linux

`CLAUDE.md` documents the **Windows** workflow (`.venv\Scripts\…`, PowerShell boot-checks,
SSL-interception workarounds for a corporate root cert). None of that applies here. This file
is the Linux equivalent — same project, same commands, different paths and none of the SSL
gymnastics. Where the two disagree, **`CLAUDE.md` is still the authority on architecture and
behaviour**; only the *commands* differ.

Verified 2026-08-21 on Linux 7.0.0-29-generic / Python 3.14.4: all 7 unit test files and the
Playwright e2e pass.

## Path translation (the whole difference, in one table)

| Windows (`CLAUDE.md`) | Linux (here) |
|---|---|
| `.venv\Scripts\python` | `.venv/bin/python` |
| `.venv\Scripts\pip` | `.venv/bin/pip` |
| `.venv\Scripts\streamlit` | `.venv/bin/streamlit` |
| `--trusted-host pypi.org …` | **not needed** — plain `pip install` works |
| `$env:NODE_TLS_REJECT_UNAUTHORIZED = "0"` before `playwright install` | **not needed** |
| `git -c http.sslBackend=schannel push` | plain `git push` |
| `yf.Ticker(sym, session=curl_cffi…verify=False)` | **not needed** — bare `yf.Ticker` reaches Yahoo |
| `Get-Process streamlit \| Stop-Process -Force` | `pkill -f "streamlit run efficient_frontier_app"` |
| `Get-NetTCPConnection -LocalPort 8501 -State Listen` | `ss -ltn \| grep :8501` |

## Setup (first time)

> **Gotcha — `ensurepip` is missing.** Debian/Ubuntu ship `python3-venv` as a separate package,
> so a bare `python3 -m venv .venv` dies with `ModuleNotFoundError: No module named 'ensurepip'`.
> `sudo apt install python3-venv` fixes it if you have root. This machine does **not** have
> passwordless sudo, so the venv here was built pip-less and pip bootstrapped by hand:

```bash
python3 -m venv --without-pip .venv
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
.venv/bin/python /tmp/get-pip.py
.venv/bin/pip install streamlit pandas numpy scipy matplotlib seaborn yfinance playwright
.venv/bin/playwright install chromium      # ~115 MB, no TLS env var needed
```

If `apt install python3-venv` is available to you, the first three lines collapse to the
ordinary `python3 -m venv .venv`.

## Run the app

```bash
.venv/bin/streamlit run efficient_frontier_app/efficient_frontier_app.py
```

Pre-loaded CSVs for EM57.MI, VWCE.MI, SGLD.MI, IMIE.MI, DBMF, BTC-EUR (daily + monthly) are in
`individual_indices_data/` — click **Run Analysis** immediately.

> **Only daily and monthly CSVs are tracked.** Picking **weekly** in the sidebar hits the
> missing-files error path until you download that interval.

## Tests

**Unit tests** (no app, no network required — the one live `^GSPC` check SKIPs gracefully offline):

```bash
for t in tests/test_*.py; do [ "$t" = tests/test_dashboard.py ] && continue; .venv/bin/python "$t"; done
```

Or one at a time: `.venv/bin/python tests/test_rebalancing.py`. See `CLAUDE.md` → *Testing* for
what each file covers.

**End-to-end** — needs the app already running (previous section), then:

```bash
HEADLESS=1 .venv/bin/python tests/test_dashboard.py
```

`HEADLESS=1` is effectively mandatory here: `headless=False` needs an X/Wayland display and hangs
without one. Screenshots land in `test_screenshots/` (gitignored).

All the `CLAUDE.md` testing gotchas still bite on Linux — the stale-module cache after editing
`ui_components.py`, the `st.success(" Analysis complete!")` done-marker that must stay last, and
the `SECTIONS` list matching the rendered headers exactly.

## Boot-check (fast path for sidebar-only / non-render changes)

The PowerShell caveats in `CLAUDE.md` (UTF-16 log garbling, `NativeCommandError`, exit 255) are
Windows artifacts — on Linux the redirect just works and the log is plain UTF-8:

```bash
.venv/bin/streamlit run efficient_frontier_app/efficient_frontier_app.py \
  --server.headless true > /tmp/boot.log 2>&1 &
sleep 5 && grep -Ei 'error|traceback' /tmp/boot.log; ss -ltn | grep :8501
```

A `LISTEN` row on 8501 plus a clean grep = booted. Stop it with:

```bash
pkill -f "streamlit run efficient_frontier_app"
```

## Versions this was verified against

Python 3.14.4 · streamlit 1.62.0 · pandas 3.0.5 · numpy 2.5.2 · scipy 1.18.0 ·
matplotlib 3.11.1 · seaborn 0.13.2 · yfinance 1.6.0 · playwright 1.62.0

Two are worth knowing about: **streamlit 1.62** clears the `width="stretch"` floor (needs ≥ 1.50),
and **pandas 3.x** is a major bump past the 2.x the app was written against — it currently passes
everything, but a pandas-flavoured breakage on this box is a plausible first suspect.

There is no `requirements.txt` and no pinning; the install line above is the source of truth.

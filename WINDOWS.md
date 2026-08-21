# WINDOWS.md — running this project on Windows

`CLAUDE.md` documents the **Linux/macOS** workflow (`.venv/bin/…`, plain `pip`, plain `git push`).
This file is the Windows equivalent: different paths, PowerShell quirks, and the corporate-SSL
workarounds that machine needs. Where the two disagree, **`CLAUDE.md` remains the authority on
architecture and behaviour** — only the *commands* differ.

## Path translation (the whole difference, in one table)

| Linux (`CLAUDE.md`) | Windows (here) |
|---|---|
| `.venv/bin/python` | `.venv\Scripts\python` |
| `.venv/bin/pip` | `.venv\Scripts\pip` |
| `.venv/bin/streamlit` | `.venv\Scripts\streamlit` |
| plain `pip install` | `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org …` |
| plain `playwright install chromium` | needs `$env:NODE_TLS_REJECT_UNAUTHORIZED = "0"` first |
| plain `git push` | `git -c http.sslBackend=schannel push` |
| bare `yf.Ticker(sym)` | needs a relaxed `curl_cffi` session (see below) |
| `kill $(pgrep -f "streamlit run …")` | `Get-Process streamlit \| Stop-Process -Force` |
| `grep "Uvicorn server started" /tmp/boot.log` | poll `Get-NetTCPConnection -LocalPort 8501 -State Listen` |

> The `--trusted-host`, `schannel` and `NODE_TLS_REJECT_UNAUTHORIZED` workarounds exist because that
> machine has a corporate/root SSL cert that intercepts TLS. They are not needed elsewhere.

## Setup (first time)

```powershell
python -m venv .venv
.venv\Scripts\pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org streamlit pandas numpy scipy matplotlib seaborn yfinance
```

**Install Playwright** (only on a fresh clone) — the Chromium download needs Node's TLS check off:

```powershell
.venv\Scripts\pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org playwright
$env:NODE_TLS_REJECT_UNAUTHORIZED = "0"; .venv\Scripts\playwright install chromium
```

## Run the app

```powershell
.venv\Scripts\streamlit run efficient_frontier_app/efficient_frontier_app.py
```

## Tests

```powershell
.venv\Scripts\python tests\test_dashboard.py          # needs the app running first
HEADLESS=1 .venv\Scripts\python tests\test_dashboard.py   # non-interactive
.venv\Scripts\python tests\test_rebalancing.py        # any unit test
```

The stale-module gotcha bites harder here: kill **all** `streamlit.exe`, confirm port 8501 has no
LISTENING socket, then restart before re-testing. See `CLAUDE.md` → *Testing* for what each test
file covers and for the done-marker / `SECTIONS` gotchas, which apply on both platforms.

## Boot-check

> **The bash-style redirect misleads on PowerShell.** Streamlit's normal Uvicorn startup line
> surfaces in the log as a `NativeCommandError`/`RemoteException` (not a real error), `boot.log` is
> written UTF-16 (which garbles `grep`/`Select-String`), and a background run reports **exit 255**
> when force-killed (also not a failure). Treat **port 8501 LISTENING** as the success signal, not a
> clean log.

Cleanest version — avoids the log-file garbling entirely:

```powershell
Start-Process .venv\Scripts\streamlit.exe -ArgumentList 'run','efficient_frontier_app/efficient_frontier_app.py','--server.headless','true' -WindowStyle Hidden
# then poll until a row comes back:
Get-NetTCPConnection -LocalPort 8501 -State Listen
```

Stop it with `Get-Process streamlit | Stop-Process -Force`, and confirm the port is freed.

## Git

```powershell
git -c http.sslBackend=schannel push
```

Pushing needs the Windows native cert store — same root cause as the pip `--trusted-host` flags.

## yfinance

yfinance uses `curl_cffi`, **not** `requests`, so neither the pip `--trusted-host` flags nor the git
`schannel` setting apply to it. The app's `run_download` / `run_total_return_reconstruction` use a
bare `yf.Ticker`, which is fine in normal use. To drive yfinance directly for local validation on the
SSL-intercepting machine, pass a relaxed session:

```python
yf.Ticker(sym, session=curl_cffi.requests.Session(impersonate="chrome", verify=False))
```

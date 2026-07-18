# Contributing

Use Python 3.11 or newer. Keep raw Reddit data and annotation payloads outside version control.

Before opening a pull request:

```powershell
python -m ruff check .
python -m pytest
```

Any new feature must document its observation window and include a test proving it does not read outcome-period data. Changes to the label definition require an update to the README, data card, and leakage audit.

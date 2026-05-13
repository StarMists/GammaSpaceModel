# Archived Public Benchmarks

This page intentionally archives the previous public benchmark set.

Those results were produced with an earlier Gamma Space Model core and should
not be read as measurements for the current DPLR-backed implementation. The
tables and visualization from that run have been removed from the main public
documentation to avoid mixing old results with the current model.

Current public validation is limited to smoke tests and tiny examples:

```bash
python -m pytest tests -q
python examples/gamma_space_quickstart.py
python examples/gamma_space_forecasting_demo.py
```

Future benchmark results should be added here only after running a fresh public
protocol against the DPLR-backed Gamma Space Model implementation.

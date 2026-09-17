# notebooks/

Exploration only. Nothing in here is load-bearing.

If something in a notebook turns out to matter -- a plot that goes in a writeup, a
parameter fit, a check someone will want to rerun -- move it into one of the modules at
the repo root and delete it from here. A number that only exists in a notebook is a
number nobody will be able to reproduce in six months.

Import the modules directly:

```python
import sys; sys.path.insert(0, "..")
import tissue_params as tp, layered, generate, features, train
```

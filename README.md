# simulation

`/data` Naming scheme: `{identifier (starting from 0)}-f{start frame # of good data}-f{end frame # of good data}-x{potentiometer value}-y{potentiometer value}-z{potentiometer value}{framerate}-.csv

We want to hold the potentiometer constant and from positions, infer acceleration over as many different constant potentiometer values as possible.

# Local Development

```
pip install poetry
poetry run ingest.py
```

# Style

```
pip install black
black .
```
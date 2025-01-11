## Install UV (Package/venv Manager)

MacOS/Linux
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

PypI
```
pip install uv
```

Note `position_nn` is the only directory setup with `uv`.

## Linting

```
uvx run ruff
```

## Running Scripts
```
cd position_nn
uv run process_data.py
uv run train.py
uv run predict.py <args>
```

WIP: args, functionality

## Directory Structure
`position_nn/`: Position model. Given positions/rotational data and potentiometer inputs over the same timeframe, output position predictions with a simple LSTM neural network.
- `raw_data/`: CSV naming format `###{ID}-MMDD{TEST_DATE}-####{START_FRAME}-####{END_FRAME}-###{FPS}`
- `processed_data/`: Post-processed dataset (by `process_data.py`)
- `process_data.py`: WIP
- `predict.py`: WIP
- `train.py`: WIP

### Legacy Folders
`acceleration_nn/`: Acceleration model. Given positions/rotational data and fixed potentiometer value as input, output acceleration with a simple MLP model.
`old-temp/`: Kinematics (steady-state) model. Given positions/rotational data and fixed potentiometer value as input, output average acceleration with kinematic equations.

## TODOs:

WIP (heh)

## LSTM Neural Network Quick Overview

WIP

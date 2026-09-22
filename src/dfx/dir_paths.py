import os

# `wd.py` holds the machine-specific working directory and is gitignored, so it
# is absent from a fresh clone. Importing it unconditionally made `import dfx`
# fail outright on any new checkout - including on an inference-only machine
# that never needs these paths, because every script passes --models_dir and
# --dataset_dir explicitly. Fall back to an env var, then to the CWD.
try:
    from .wd import working_dir
except ImportError:
    working_dir = os.environ.get('DFX_WORKING_DIR', os.getcwd())


def get_path(dir: str):

    paths = {'dataset': os.path.join(working_dir, 'datasets'),\
             'data_robustness': os.path.join(working_dir, 'testing_robustness'),\
             'data_generalization': os.path.join(working_dir, 'testing_generalization'),\
             'guidance': os.path.join(working_dir, 'guidance.csv'),\
             'models': os.path.join(working_dir, 'models/unbalancing-approach')}

    return paths[dir]

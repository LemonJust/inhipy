from pathlib import Path
from coolname import generate_slug
import yaml

def generate_name_if_missing(config):
    """
    Generates a random name if it is not provided in the config.

    Args:
        config: dict with config parameters

    Returns:
        config: dict with config parameters
    """

    if 'name' not in config:
        config['name'] = generate_slug(2)

    elif config['name'] is None:
        config['name'] = generate_slug(2)

    elif config['name'].strip() == '':
        config['name'] = generate_slug(2)

    return config

def get_basedir(basedir_list, step = None, name = None):
    """
    Concatenates a list of strings to a path. 
    If step is provided, it is appended to the path.
    If name is provided, it is appended to the path.

    Args:
        basedir_list: list of strings
        step: string
        name: string

    Returns:
        basedir: Path
    """
    basedir = Path(basedir_list[0])
    for i in range(1, len(basedir_list)):
        basedir = basedir / basedir_list[i]

    if step is not None:
        basedir = basedir / step
    
    if name is not None:
        basedir = basedir / name

    return basedir

def generate_output_folder(config, step = None):
    """
    Generates an output folder based on the config parameters.

    Args:
        config: dict with config parameters
        step: string

    Returns:
        output_folder: Path
    """
    output_folder = get_basedir(config['output']['basedir'], step, config['name'])
    output_folder.mkdir(parents = True, exist_ok = True)
    config['output_folder'] = output_folder.as_posix()

    return config

def initialize_step_run(config, step, verbose = True):
    """
    Initializes a step run by generating a name and an output folder, 
    and saving the config file into the folder.

    Args:
        config: dict with config parameters
        step: string
        verbose: bool

    Returns:
        output_folder: Path
    """    
    config = config[step]
    config['step'] = step
    config = generate_name_if_missing(config)
    config = generate_output_folder(config, step)

    if verbose:
        print(f"Initialized step {step} with name {config['name']} and output folder {config['output_folder']}")
    with open(config['output_folder']/f"{config['name']}_config.yaml", 'w') as f:
        yaml.dump(config, f)

    return config

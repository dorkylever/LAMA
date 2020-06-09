"""
Given a input folde of lines (could be centres etc) and a lama config do:
* Run each specimen on a grid node
* write to log the result
"""

import time
from lama.registration_pipeline import run_lama
from lama import common
import fabric
import time
import toml
import shutil
from pathlib import Path
import toml


def run(config_path):

    grid_config = toml.load(config_path)
    root = Path(grid_config['root_dir'])
    inputs_dir = root / 'inputs'
    if not inputs_dir.is_dir():
        raise NotADirectoryError

    out_root = root / 'output'
    out_root.mkdir(exist_ok=True)

    config_dir = root / 'configs'
    if not config_dir.is_dir():
        raise NotADirectoryError

    config_done_dir = root / 'configs_done'
    config_done_dir.mkdir(exist_ok=True)

    while True:

        try:
            lama_config_path = list(config_dir.iterdir())[0]
        except IndexError:
            print(f'Waiting for configs')
            time.sleep(2)
            continue

        lama_config = toml.load(lama_config_path)
        shutil.move(lama_config_path, config_done_dir / lama_config_path.name)

        root_reg_dir = out_root / lama_config_path.stem
        root_reg_dir.mkdir(exist_ok=True)

        for line_dir in inputs_dir.iterdir():
            for input_ in line_dir.iterdir():

                print(f'Specimen: {input_.name}, config: {lama_config_path.name}')
                reg_out_dir = root_reg_dir / line_dir.name / input_.stem
                reg_out_dir.mkdir(exist_ok=True, parents=True)

                new_input_dir = reg_out_dir / 'inputs'
                new_input_dir.mkdir(exist_ok=True)
                shutil.copyfile(input_, new_input_dir / input_.name)

                new_config_name = reg_out_dir / lama_config_path.name

                with open(new_config_name, 'w') as fh:
                    toml.dump(lama_config, fh)

                run_on_grid(new_config_name, grid_config)


def run_on_grid(lama_config_path, grid_config):

    c = grid_config
    cmd = f'{c["grid_cmd"]} "{c["docker_cmd"]} \'{c["lama_cmd"]}\'"'
    # Now interpolate our values
    cmd = cmd.format(c['n_thread'], c['docker_tag'], lama_config_path)

    conn = fabric.Connection(c['HOST'], user=c['USER'], inline_ssh_env=True)
    conn.run(cmd, env={'SGE_ROOT': '/grid/dist/GE2011.11p1'})


if __name__ == '__main__':
    import sys
    cfg_path = sys.argv[1]
    run(cfg_path)


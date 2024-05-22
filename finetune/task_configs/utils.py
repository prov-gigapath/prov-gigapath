import yaml


def load_task_config(config_path: str) -> dict:
    '''Load the yaml config file that specifies the task setup.'''
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


if __name__ == '__main__':
    config = load_task_config('finetune/task_configs/mutation_5_gene.yaml')
    print(config)
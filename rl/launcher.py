import yaml
from box import Box
from rl.trainer import trainer_init
from tqdm import tqdm
import accelerate
import datetime
import wandb

from utils_general import load_config
from utils_rl import set_str_action_space

def main(config, boxed_config):
    action_space = set_str_action_space(boxed_config.env_config)
    daytime =  datetime.datetime.now().strftime("%Y-%m-%d||%H:%M:%S")
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=boxed_config.grad_accum_steps)
    if boxed_config.report_to == "wandb":
        config["process_idx"] = accelerator.local_process_index
        if "gym_cards" in boxed_config.env_config.id:
            project_name = "GeneralPoints_RL"
        elif "gym_virl" in boxed_config.env_config.id:
            project_name = "V-IRL_RL"
            
        wandb.init(project=project_name, name = boxed_config.run_name + f"_{daytime}", config=config)
    if accelerator.is_main_process:
        print(yaml.dump(config))
    player = trainer_init[boxed_config.trainer](action_space = action_space, daytime = daytime, accelerator = accelerator,**boxed_config)
    player.train()


if __name__ == "__main__":
    config, boxed_config = load_config()
    print(config)
    main(config, boxed_config)

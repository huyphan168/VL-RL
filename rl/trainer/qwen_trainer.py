import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.trainer.model_tz_llama import VLMValue, VLMPolicy  # same wrappers work for Qwen2.5-VL
from rl.trainer.storage_tz import RolloutStorage
from rl.trainer.base_trainer import BaseTrainer
import rl.trainer.algo as algo

import wandb
from typing import Optional, Dict, List, Any
import accelerate
from accelerate.state import AcceleratorState

from utils_mllm import evaluate_model_config
from utils_general import progress_bar, re_match
import os
from qwen_vl_utils import process_vision_info

from PIL import Image

class QwenTrainer(BaseTrainer):
    def __init__(self, 
                 action_space: Optional[List[Any]],
                 daytime: str,
                 accelerator: accelerate.Accelerator,
                 optimizer_config, ppo_config, compute_return_kwargs,
                 num_steps, num_updates,
                 env_config,
                 model, model_path,
                 prompt_config, generation_config,
                 output_dir, seed=42, report_to=None, run_name='default', save_ckpt=False, **kwargs):

        super(QwenTrainer, self).__init__(action_space, daytime, accelerator, optimizer_config, ppo_config,
                                          compute_return_kwargs, num_steps, num_updates, env_config, 
                                          model, model_path, prompt_config, generation_config, output_dir, 
                                          seed, report_to, run_name, save_ckpt, **kwargs)

    def init_model_optimizer_algo(self, model, model_path, ppo_config, optimizer_config):
        self.processor, self.model = evaluate_model_config(model, model_path)
        
        # VLMValue and VLMPolicy are generic wrappers that work with both Llama and Qwen2.5-VL.
        value_model: nn.Module = VLMValue(self.model)
        actor_critic: nn.Module = VLMPolicy(tokenizer=self.processor,
                                             value_model=value_model,
                                             generation_config=self.generation_config)

        optimizer = optim.Adam(actor_critic.value_model.parameters(),
                               lr=optimizer_config.init_lr,
                               eps=optimizer_config.eps,
                               weight_decay=optimizer_config.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=optimizer_config.lr_max_steps,
                                                            eta_min=optimizer_config.end_lr)
        
        # Ensure micro-batch size is 1 for deepspeed training.
        AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1
        
        self.actor_critic, self.optimizer, self.lr_scheduler = self.accelerator.prepare(actor_critic, optimizer, lr_scheduler)
        self.agent = algo.PPO(
            self.actor_critic,
            self.optimizer,
            self.accelerator,
            **ppo_config
        )
        self.rollouts = RolloutStorage(self.num_steps,
                                       self.env.action_space, 
                                       self.generation_config.max_new_tokens)

        self.obs, self.info = self.env.reset()
        self.rollouts.obs[0]['image'] = self.obs

    def formulate_payload(self, question, obs=None):
        self.payload = [{
            "role": "user",
            "content": [{"type": "text", "text": question}]
        }]
        if obs is not None:
            # Append image to payload.
            self.payload[0]['content'].insert(0, {"type": "image", "image": obs})

    def formulate_prompt(self, vision_res_dict, language_res_dict, prompt=None, obs=None, info=None):
        # Task specific implementation.
        if 'gym_cards' in self.id:
            if info['Verify Info'] is not None:
                self.payload.append({
                    "role": "user",
                    "content": [{"type": "text", "text": f"You failed this trial because {info['Verify Info']}"}]
                })
            else:
                question = prompt.format(**vision_res_dict, **language_res_dict, **self.oracle_arguments)
                self.formulate_payload(question, obs)
        elif 'gym_virl' in self.id:
            if info['Verify Info'] is None or 'Correct action' in info['Verify Info'] or 'step_limit_reached"' in info['Verify Info']:
                question = prompt.format(**vision_res_dict, **language_res_dict, **self.oracle_arguments)
                self.formulate_payload(question, obs)
            else:
                self.payload.append({
                    "role": "user",
                    "content": [{"type": "text", "text": f"You failed this trial because {info['Verify Info']}"}]
                })

    def collect_trajectories(self):
        self.stat.reset()
        running_reward = 0
        obs = self.obs
        info = self.info
        if self.use_vision:
            prompts, patterns = self.prompt_vision, self.pattern_vision
        else:
            prompts, patterns = self.prompt_language, self.pattern_language
            
        pbar = progress_bar(self.num_steps, "Collecting Trajectories", "blue", self.accelerator)
        for step in range(self.num_steps + 1):
            vision_res_dict = {}
            language_res_dict = {}
            self.formulate_vision_arguments(vision_res_dict, info)
            
            with torch.no_grad():
                obs = None if not self.use_vision else obs
                if isinstance(obs, np.ndarray):
                    obs = Image.fromarray(obs)
                for prompt, pattern in zip(prompts, patterns):
                    self.formulate_prompt(vision_res_dict, language_res_dict, prompt=prompt, obs=obs, info=info)
                    input_text = self.processor.apply_chat_template(self.payload, tokenize=False, add_generation_prompt=True)
                    
                    image_inputs, video_inputs = process_vision_info(self.payload)
                    inputs = self.processor(
                        text=[input_text],
                        images=image_inputs,
                        videos=video_inputs,
                        return_tensors="pt",
                    ).to(self.model.device)
                    
                    values, io_dict, output_text, action_log_prob, action_tokens_log_prob = self.actor_critic.act_oneline(inputs, obs)
                    self.append_intermidiate_res(output_text)
            current_formula = re_match(output_text, 'formula')
            try:
                current_formula = current_formula.split('=')[0]
            except:
                pass
            if step == self.num_steps:
                next_values = values
                self.rollouts.obs[-1]['io_dict'] = io_dict
                break
            if step == self.num_steps - 1:
                print("Running example")
                print("Input: ")
                print(input_text)
                print("Output: ")
                print(output_text)
                print("Formula: ")
                print(current_formula)
                print("Action log prob: ")
                print(action_log_prob)
                print("Action tokens log prob: ")
                print(action_tokens_log_prob)
            obs, reward, done, truncated, info = self.env.step(output_text)
            running_reward += reward
            self.rollouts.insert({"image": obs, "io_dict": io_dict}, None, torch.tensor([0]), 
                                 action_log_prob, values.squeeze(), reward, 
                                 torch.Tensor([1 - done]), torch.Tensor([1 - truncated]))
            self.stat.log_step(reward, done or truncated or not self.enable_verification)
            if done or truncated or not self.enable_verification:
                if 'gym_virl' in self.id:
                    self.stat.log_virl_success(info['is_success'])
                self.stat.insert_running_reward(running_reward)
                self.stat.insert_action_tokens_log_prob(action_tokens_log_prob.item())
                running_reward = 0
                obs, info = self.env.reset()
            pbar.update()
        pbar.close()
        return next_values, obs, info

    def append_intermidiate_res(self, res):
        # Append intermediate result to payload.
        self.payload.append({"role": "assistant", "content": [{"type": "text", "text": res}]})

    def extract_final_action(self, language_res_dict):
        # Task-specific: return the extracted action.
        return language_res_dict['formula']

    def save_model(self, output_dir):
        if self.accelerator.is_main_process:
            torch.cuda.synchronize()
            unwrapped_model = self.accelerator.unwrap_model(self.actor_critic)
            mllm_model = unwrapped_model.value_model.base
            mllm_model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="8GB")
            self.processor.save_pretrained(output_dir)

    def train_one_epoch(self, save_model=False, update=0):
        next_values, self.obs, self.info = self.collect_trajectories()
        self.rollouts.compute_returns(next_values, **self.compute_return_kwargs)
        value_loss, action_loss, dist_entropy = self.agent.update(self.rollouts)
        self.lr_scheduler.step()
        episode_rewards = self.stat.running_reward
        episode_action_tokens_log_prob = self.stat.action_tokens_log_prob
        self.rollouts.after_update()
        self.total_num_steps += self.num_steps * self.accelerator.num_processes
        if save_model:
            torch.cuda.empty_cache()
            self.save_model(os.path.join(self.output_dir, f"checkpoint-epoch-{update}"))
        if self.report_to == 'wandb':
            wandb.log({
                'total_num_steps': self.total_num_steps,
                'compute_tokens': self.actor_critic.token_cnt,
                'inference_fwd': self.actor_critic.called_inference_time,
                'bp_forward': self.actor_critic.called_bp_time,
                'value_loss': value_loss,
                'action_loss': action_loss,
                'dist_entropy': dist_entropy,
                'reward.mean': self.rollouts.rewards.mean().item(),
                'reward.std': self.rollouts.rewards.std().item(),
                'reward.max': self.rollouts.rewards.max().item(),
                'reward.min': self.rollouts.rewards.min().item(),
                'value.mean': self.rollouts.value_preds.mean().item(),
                'value.std': self.rollouts.value_preds.std().item(),
                'value.max': self.rollouts.value_preds.max().item(),
                'value.min': self.rollouts.value_preds.min().item(),
                'return.mean': self.rollouts.returns.mean().item(),
                'return.std': self.rollouts.returns.std().item(),
                'return.max': self.rollouts.returns.max().item(),
                'return.min': self.rollouts.returns.min().item(),
                'episode_rewards.mean': np.mean(episode_rewards),
                'episode_rewards.std': np.std(episode_rewards),
                'episode_rewards.max': np.max(episode_rewards),
                'episode_rewards.min': np.min(episode_rewards),
                'episode_action_tokens_log_prob.mean': np.mean(episode_action_tokens_log_prob),
                'recog_acc': self.stat.cal_vision_acc(),
                'success_rate': self.stat.cal_success_rate(),
            }) 
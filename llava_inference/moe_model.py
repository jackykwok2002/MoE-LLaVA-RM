import os
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from peft import PeftModel
from PIL import Image
import requests
from io import BytesIO
from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Tuple
from transformers.utils.generic import ModelOutput

from moellava.conversation import conv_templates, SeparatorStyle
from moellava.utils import disable_torch_init
from moellava.constants import (
    IMAGE_TOKEN_INDEX, 
    DEFAULT_IMAGE_TOKEN, 
    DEFAULT_IM_START_TOKEN, 
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN
)
from moellava.mm_utils import tokenizer_image_token
from reward.moe_builder import load_pretrained_model
from reward.action_processing import ActionTokenizer

import os
# Set environment variables
os.environ["WORLD_SIZE"] = "1"  # Set to 1 for single GPU/process
os.environ["LOCAL_RANK"] = "0"  # Local rank of the process
os.environ["RANK"] = "0"

from transformers import set_seed
set_seed(25)

@dataclass
class RewardModelOutput(ModelOutput):
    """Output class for reward model predictions."""
    rewards: torch.Tensor = None

class ImageProcessor:
    """Handles image processing with caching capability."""
    def __init__(self, processor, compute_dtype):
        self.processor = processor['image']
        self.compute_dtype = compute_dtype
        self.cache: Dict[str, torch.Tensor] = {}
    
    def process_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """Process image with caching for repeated use."""
        cache_key = image if isinstance(image, str) else id(image)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if isinstance(image, str):
            if image.startswith(('http', 'https')):
                response = requests.get(image)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                image = Image.open(image).convert('RGB')
        
        image = self._expand2square(
            image, 
            tuple(int(x*255) for x in self.processor.image_mean)
        )
        
        image_tensor = self.processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        image_tensor = image_tensor.cuda().to(dtype=self.compute_dtype)
        
        self.cache[cache_key] = image_tensor
        return image_tensor
    
    def _expand2square(self, pil_img: Image.Image, background_color: tuple) -> Image.Image:
        """Expand image to square with padding."""
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
    
    def clear_cache(self):
        """Clear the image cache."""
        self.cache.clear()

class RewardModel:
    def __init__(self, backbone_path: str, lora_path: str):
        """Initialize the reward model."""
        disable_torch_init()
        
        self.compute_dtype = torch.bfloat16
        from moellava import conversation as conversation_lib
        self.tokenizer = AutoTokenizer.from_pretrained(
            "stabilityai/stablelm-2-1_6b",
            model_max_length=2048,
            padding_side="left",
            truncation_side="right",
            use_fast=False,
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            "vicuna_v1"
        ]
        self.action_tokenizer = ActionTokenizer(self.tokenizer)
        self.backbone_model, image_processor = self._initialize_model(backbone_path, lora_path)
        self.image_processor = ImageProcessor(image_processor, self.compute_dtype)
        
        self._initialize_special_tokens()
        self.reward_head = self._initialize_reward_head(lora_path)
        self._setup_vision_tower()
        
        self.conv_mode = "vicuna_v1"
        self.conv_template = conv_templates[self.conv_mode].copy()
        
        self.backbone_model = self.backbone_model.cuda()
        self._convert_to_dtype()
        self.backbone_model.eval()
        torch.use_deterministic_algorithms(True)

    def _initialize_special_tokens(self):
        """Initialize special tokens for the model."""
        if self.backbone_model.config.mm_use_im_patch_token:
            self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if self.backbone_model.config.mm_use_im_start_end:
            self.tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], 
                special_tokens=True
            )
        self.backbone_model.resize_token_embeddings(len(self.tokenizer))

    def _initialize_model(self, backbone_path: str, lora_path: str):
        """Initialize the backbone model and image processor."""
        _, model, image_processor, _ = load_pretrained_model(
            model_path=backbone_path,
            model_base=None,
            model_name="MoE-LLaVA-StableLM-1.6B-4e"
        )
        
        if lora_path:
            model = PeftModel.from_pretrained(
                model,
                lora_path,
                is_trainable=False,
                use_safetensors=True
            )
        
        return model, image_processor

    def _initialize_reward_head(self, lora_path: str) -> nn.Linear:
        """Initialize the reward head network."""
        hidden_size = self._get_hidden_size()
        reward_head = nn.Linear(hidden_size, 1)
        torch.nn.init.zeros_(reward_head.bias)
        reward_head = reward_head.cuda()
        
        reward_head_path = os.path.join(lora_path, "reward_head")
        if os.path.exists(reward_head_path):
            reward_head.load_state_dict(
                torch.load(reward_head_path, map_location="cuda:0")
            )
        return reward_head

    def _setup_vision_tower(self):
        """Setup and initialize the vision tower."""
        vision_tower = self.backbone_model.get_image_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device='cuda', dtype=self.compute_dtype)
        
        if hasattr(self.backbone_model.get_model(), 'mm_projector'):
            self.backbone_model.get_model().mm_projector.to(dtype=self.compute_dtype)

    def _get_hidden_size(self) -> int:
        """Get the hidden size from model config."""
        if isinstance(self.backbone_model, PeftModel):
            config = self.backbone_model.base_model.config
        else:
            config = self.backbone_model.config
        return config.hidden_size

    def _convert_to_dtype(self):
        """Convert model parameters to compute_dtype."""
        for param in self.backbone_model.parameters():
            param.data = param.data.to(self.compute_dtype)
        self.reward_head = self.reward_head.to(self.compute_dtype)

    def prepare_batch_inputs(
        self, 
        instruction: str, 
        actions: List[Union[List[float], str]], 
        image: Union[str, Image.Image]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare batched inputs for the reward model."""
        # Process image once (with caching)
        image_tensor = self.image_processor.process_image(image)
        
        # Prepare inputs for each action
        input_ids_list = []

        for action in actions:
            # Prepare conversation
            action_str = self.action_tokenizer(action)
            inp = (f"shows the current observation from the robot's wrist-mounted camera. "
                   f"The robot manipulation arm is attempting to {instruction}. "
                   f"What action should the robot take to effectively accomplish the task? "
                   f"ASSISTANT: The robot should take the action: {action_str}"
                   f"USER: Please evaluate the quality of the robot action. "
                   f"A good robot action should consider different factors, "
                   f"especially interactions with surrounding objects and human preferences."
                   f"ASSISTANT: Based on how humans would control the robot arm and the "
                   f"awareness of the situation, the quality score of the robot action is</s")

            if self.backbone_model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

            conv = self.conv_template.copy()
            conv.append_message(conv.roles[0], inp)
            prompt = conv.get_prompt()
            prompt = prompt.replace("<image>", "<|endoftext|>").replace("USER: Please evaluate", "</s> USER: Please evaluate").replace("human preferences.", "human preferences.\n")

            input_ids = tokenizer_image_token(
                prompt, 
                self.tokenizer, 
                IMAGE_TOKEN_INDEX, 
                return_tensors='pt'
            )
            input_ids_list.append(input_ids)

        padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        # Stack all input_ids into a batch
        batched_input_ids = padded_input_ids.cuda()
        
        # Repeat image tensor for batch size
        batch_size = len(actions)
        batched_image_tensor = image_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        return batched_input_ids, batched_image_tensor

    def get_rewards(
        self, 
        instruction: str, 
        actions: List[Union[List[float], str]], 
        image: Union[str, Image.Image]
    ) -> List[float]:
        """Calculate rewards for all actions in a single batch."""
        input_ids, image_tensor = self.prepare_batch_inputs(instruction, actions, image)
        with torch.inference_mode():
            with torch.no_grad():
                outputs = self.backbone_model(
                    input_ids=input_ids,
                    images=image_tensor,
                    use_cache=True,
                    return_dict=True,
                    output_hidden_states=True,
                )
            last_hidden_state = outputs.hidden_states[-1]
            logits = outputs.logits
            # Add small perturbation for numerical stability
            last_hidden_state = last_hidden_state + 0.0 * torch.mean(logits)

            last_hidden_state_at_end = last_hidden_state[:, -1, :].to(self.compute_dtype)
            rewards = self.reward_head(last_hidden_state_at_end).squeeze(-1)
            
        return rewards.tolist()

    def get_reward(
        self, 
        instruction: str, 
        action: Union[List[float], str], 
        image: Union[str, Image.Image]
    ) -> float:
        """Calculate reward for a single action."""
        return self.get_rewards(instruction, [action], image)[0]

def main():
    """Example usage of the reward model."""
    backbone_path = "models/MoE-LLaVA-StableLM-1.6B-4e"
    lora_path = "models/checkpoint-2800/adapter_model/lora_default"
    
    # Initialize reward model
    rm = RewardModel(backbone_path, lora_path)

    # Example data
    actions = [
        [-0.0006071124225854874, -0.001102231559343636, -0.002975916489958763, -0.0037233866751194, 0.009374408982694149, 0.00042649864917621017, 1.003713607788086], #action0
        [0.0007309613865800202, -0.00033146265195682645, 8.855393389239907e-05, 0.0023672617971897125, -0.00297730159945786, 0.0071182833053171635, 1.0025840997695923],
        [0.0003844001912511885, 0.0010981999803334475, -0.006680284161120653, -0.008547065779566765, 0.021539149805903435, 0.014096851460635662, 1.0460903644561768],
        [-0.0012932196259498596, -7.85675656516105e-05, -0.007247961591929197, -0.010998447425663471, 0.023759912699460983, 0.0015791847836226225, 1.013885736465454],
        [-0.0002225322969025001, 0.00012939439329784364, -0.00023808938567526639, -0.004563219379633665, 0.005036361515522003, 0.0021004355512559414, 0.9518507122993469],
        [-0.001818387652747333, -0.0013037925818935037, -0.0080921845510602, -0.008700641803443432, 0.02245231531560421, -0.0024688607081770897, 1.0093427896499634],
        [-0.003059342736378312, 7.718679262325168e-05, -0.009317572228610516, -0.01436073612421751, 0.02727797068655491, 0.00498265540227294, 1.030832052230835],
        [-0.0009580199257470667, -0.002090591937303543, -0.0009161726338788867, -0.005044611636549234, 0.006233627907931805, -0.0006715729832649231, 1.0123248100280762],
        [0.0010517071932554245, -0.00010974949691444635, -0.0016351011581718922, -0.012653344310820103, 0.01755679026246071, -0.005102170165628195, 1.0341507196426392],
        [-3.1640956876799464e-06, 0.00021746146376244724, -0.0019310240168124437, -0.0020527918823063374, 0.004275907762348652, 0.005797747056931257, 1.016253113746643],
        [-0.002163161523640156, -8.288555545732379e-05, -0.008197044022381306, -0.015280686318874359, 0.027800599113106728, 0.008853021077811718, 0.9894216656684875],
    ]
    
    instruction = "move the yellow knife to the right of the pan"
    image_path = "images/0000000.jpg"
    
    # Get rewards for all actions in a single batch
    rewards = rm.get_rewards(instruction, actions, image_path)
    print(rewards)
    rewards = rm.get_rewards(instruction, actions, image_path)
    print(rewards)

if __name__ == "__main__":
    main()
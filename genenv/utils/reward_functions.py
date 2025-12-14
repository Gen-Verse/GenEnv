# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 GenEnv Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is adapted from verl (https://github.com/volcengine/verl)

"""
Reward Functions for GenEnv Training.

This module provides example reward function implementations.
Users should replace these with their domain-specific reward functions.

Example domains:
- Math reasoning: Check if the answer matches ground truth
- Tool calling: Verify correct API calls and parameters
- Code generation: Execute and validate outputs
"""

import re
import json
import torch
from typing import Dict, List, Union, Any
from concurrent.futures import ThreadPoolExecutor

from verl import DataProto


class RewardManager:
    """
    Base Reward Manager for GenEnv training.
    
    >>> USER CUSTOMIZATION REQUIRED <<<
    Replace the `compute_reward` method with your domain-specific logic.
    
    Example implementations are provided for:
    - Math reasoning (boxed answer extraction)
    - Tool calling (action matching)
    """
    
    def __init__(self, tokenizer, num_examine: int = 0):
        """
        Initialize the reward manager.
        
        Args:
            tokenizer: HuggingFace tokenizer for decoding sequences
            num_examine: Number of samples to print for debugging
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine

    def __call__(self, data: DataProto) -> torch.Tensor:
        """
        Compute rewards for a batch of generated responses.
        
        Args:
            data: DataProto containing prompts, responses, and ground truth
            
        Returns:
            Tensor of shape (batch_size, response_length) with rewards
        """
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        def process_item(args):
            i, data_item = args
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # Decode the full sequence
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # >>> USER CUSTOMIZATION: Replace with your reward function <<<
            score = self.compute_reward(sequences_str, ground_truth)
            
            return i, score, valid_response_length

        with ThreadPoolExecutor(max_workers=48) as executor:
            args = [(i, data[i]) for i in range(len(data))]
            results = list(executor.map(process_item, args))

        for i, score, valid_response_length in results:
            reward_tensor[i, valid_response_length - 1] = score

        return reward_tensor
    
    def compute_reward(self, generated_text: str, ground_truth: Any) -> float:
        """
        Compute reward for a single generated response.
        
        >>> USER CUSTOMIZATION REQUIRED <<<
        Replace this method with your domain-specific reward logic.
        
        Args:
            generated_text: The full generated sequence including prompt
            ground_truth: The expected answer/output
            
        Returns:
            Reward score (typically 0.0 or 1.0 for binary rewards)
        """
        # Example: Math reasoning with boxed answers
        pred_answer = self._extract_boxed_answer(generated_text)
        gold_answer = self._get_gold_answer(ground_truth)
        
        pred_norm = self._normalize_answer(pred_answer)
        gold_norm = self._normalize_answer(gold_answer)
        
        return 1.0 if pred_norm and gold_norm and pred_norm == gold_norm else 0.0
    
    @staticmethod
    def _extract_boxed_answer(text: str) -> str:
        """Extract the last LaTeX \\boxed{...} content from text."""
        if not isinstance(text, str):
            return ""
        try:
            matches = list(re.finditer(r"\\boxed\{([\s\S]*?)\}", text))
            if matches:
                return (matches[-1].group(1) or "").strip()
            s = text.strip()
            if s.startswith("$") and s.endswith("$") and len(s) >= 2:
                s = s[1:-1].strip()
            return s
        except Exception:
            return ""
    
    @staticmethod
    def _get_gold_answer(ground_truth: Union[str, Dict, Any]) -> str:
        """Extract the gold answer from ground truth."""
        if isinstance(ground_truth, dict) and 'answer' in ground_truth:
            return str(ground_truth['answer'])
        if isinstance(ground_truth, str):
            try:
                parsed = json.loads(ground_truth)
                if isinstance(parsed, dict) and 'answer' in parsed:
                    return str(parsed['answer'])
            except Exception:
                pass
            return ground_truth
        return str(ground_truth)
    
    @staticmethod
    def _normalize_answer(s: str) -> str:
        """Normalize answer string for comparison."""
        if not isinstance(s, str):
            return ""
        s = s.strip().strip('`').strip('"').strip("'")
        s = re.sub(r"\s+", " ", s)
        return s.lower()


class ToolCallingRewardManager(RewardManager):
    """
    Reward Manager for Tool/API Calling tasks.
    
    This implementation checks if the model's predicted tool calls
    match the expected ground truth calls.
    """
    
    def compute_reward(self, generated_text: str, ground_truth: Any) -> float:
        """
        Check if predicted tool calls match ground truth.
        
        Expected ground_truth format:
        {"name": "api_name", "parameters": {"param1": "value1", ...}}
        """
        predicted_calls = self._extract_tool_calls(generated_text)
        
        if not predicted_calls:
            return 0.0
        
        for pred_call in predicted_calls:
            if self._match_tool_call(pred_call, ground_truth):
                return 1.0
        
        return 0.0
    
    @staticmethod
    def _extract_tool_calls(text: str) -> List[Dict]:
        """Extract tool calls from <tool_call> tags."""
        calls = []
        lines = text.splitlines()
        in_block = False
        
        for raw in lines:
            line = raw.strip()
            if not in_block:
                idx = line.find("<tool_call>")
                if idx != -1:
                    in_block = True
                    after = line[idx + len("<tool_call>"):].strip()
                    if after:
                        try:
                            obj = json.loads(after)
                            calls.append(obj)
                        except Exception:
                            pass
            else:
                if line.startswith("</tool_call>"):
                    in_block = False
                    continue
                if line.startswith("<"):
                    in_block = False
                    continue
                try:
                    obj = json.loads(line)
                    calls.append(obj)
                except Exception:
                    pass
        
        return calls
    
    @staticmethod
    def _match_tool_call(pred: Dict, gold: Any) -> bool:
        """Check if predicted call matches any gold call."""
        def prune_params(d):
            if not isinstance(d, dict):
                return {}
            return {k: v for k, v in d.items() if v not in (None, "", [])}

        if not isinstance(pred, dict):
            return False

        gold_list = gold if isinstance(gold, list) else [gold]

        for gold_item in gold_list:
            if not isinstance(gold_item, dict):
                continue
            gold_name = gold_item.get("name")
            gold_params = prune_params(gold_item.get("parameters", {}))

            pred_name = pred.get("name", gold_name)
            pred_params = prune_params(pred.get("parameters", pred))

            if pred_name != gold_name:
                continue

            # Check if gold params are subset of pred params
            ok = True
            for k, v in gold_params.items():
                if k not in pred_params or pred_params[k] != v:
                    ok = False
                    break
            if ok:
                return True
        
        return False


class ActionRewardManager(RewardManager):
    """
    Reward Manager for Action-based tasks (e.g., web navigation).
    
    This implementation checks if the model's predicted actions
    match the expected ground truth actions.
    """
    
    def compute_reward(self, generated_text: str, ground_truth: Any) -> float:
        """
        Check if predicted actions match ground truth.
        
        Expected actions format:
        - type(bid=123, value=text, press_enter=True)
        - click(bid=456)
        - select_option(bid=789, options=choice)
        """
        extracted_actions = self._extract_actions(generated_text)
        expected_actions = self._parse_ground_truth(ground_truth)
        
        if not extracted_actions or not expected_actions:
            return 0.0
        
        return 1.0 if self._compare_actions(extracted_actions, expected_actions) else 0.0
    
    @staticmethod
    def _extract_actions(text: str) -> List[str]:
        """Extract action calls from text."""
        patterns = [
            r'type\s*\([^)]+\)',
            r'click\s*\([^)]+\)',
            r'select_option\s*\([^)]+\)',
            r'go_back\s*\(\s*\)'
        ]
        
        actions = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            actions.extend(matches)
        
        return [re.sub(r'\s+', ' ', a.strip()) for a in actions]
    
    @staticmethod
    def _parse_ground_truth(ground_truth: Any) -> List[str]:
        """Parse ground truth into list of actions."""
        if isinstance(ground_truth, dict) and 'answer' in ground_truth:
            ground_truth = ground_truth['answer']
        
        if isinstance(ground_truth, str):
            try:
                parsed = json.loads(ground_truth)
                if isinstance(parsed, dict) and 'answer' in parsed:
                    ground_truth = parsed['answer']
            except json.JSONDecodeError:
                pass
        
        if isinstance(ground_truth, str) and ground_truth.strip():
            return [ground_truth.strip()]
        
        return []
    
    @staticmethod
    def _normalize_action(action: str) -> str:
        """Normalize action string for comparison."""
        action = re.sub(r'\s*=\s*', '=', action)
        action = re.sub(r'\s*,\s*', ',', action)
        action = re.sub(r'\s*\(\s*', '(', action)
        action = re.sub(r'\s*\)\s*', ')', action)
        return action
    
    def _compare_actions(self, extracted: List[str], expected: List[str]) -> bool:
        """Compare extracted actions with expected actions."""
        normalized_extracted = [self._normalize_action(a) for a in extracted]
        normalized_expected = [self._normalize_action(a) for a in expected]
        expected_set = set(normalized_expected)
        
        for action in normalized_extracted:
            if action not in expected_set:
                return False
        
        return True


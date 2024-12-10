from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import json
from action_processing import ActionTokenizer

# Load the Excel data
file_path = '10k.xlsx'
df = pd.read_excel(file_path)

json_data = []

tokenizer = AutoTokenizer.from_pretrained(
    "zhiqings/LLaVA-RLHF-13b-v1.5-336", subfolder="sft_model")
action_tokenizer = ActionTokenizer(tokenizer)


def tokenize_from_str(action):
    action_arr = np.array(list(map(float, action.strip('[]').split(', '))))
    return action_tokenizer(action_arr)

# action0 or action1 should be the same as chosen action?


def getChosenAction(action0, action1, chosen_action):
    l0 = np.array(list(map(float, action0.strip('[]').split(', '))))
    l1 = np.array(list(map(float, action1.strip('[]').split(', '))))
    chosen = np.array(list(map(float, chosen_action.strip(
        '[]').replace('\n', '').replace(',', ' ').split())))
    print(chosen)
    distance_to_action0 = np.linalg.norm(chosen - l0)
    distance_to_action1 = np.linalg.norm(chosen - l1)
    return 1 if distance_to_action0 < distance_to_action1 else 2


cnt = 0
for _, row in df.iterrows():
    if row['chosen_action'] == 0:
        continue
    json_object = {
        "id": int(row['index']) * 55 + int(row['pair_index']),
        "image": f"000000{int(row['index'])}.jpg",
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\n shows the current observation from the robot's wrist-mounted camera. The robot manipulation arm is attempting to {row['instruction'].lower().rstrip('.')}. What action should the robot take to effectively accomplish the task? "
            },
            {
                "from": "gpt",
                "value": "The robot should take the action: " + tokenize_from_str(str(row['action0']))
            }
        ],
        "output_1": {
            "from": "llava",
            "value": "The robot should take the action: " + tokenize_from_str(str(row['action0']))
        },
        "output_2": {
            "from": "llava",
            "value": "The robot should take the action: " + tokenize_from_str(str(row['action1']))
        },
        "preference": getChosenAction(row['action0'], row['action1'], row['chosen_action']),
        "hallucination": False,
        "flip": False,
        "length_bias": False
    }

    json_data.append(json_object)
    cnt += 1
    # if cnt > 1000:
    #     break

output_file_path = '10k_discretized.json'
with open(output_file_path, 'w') as json_file:
    json.dump(json_data, json_file, indent=4)

print(f"Data has been successfully written to {output_file_path}")

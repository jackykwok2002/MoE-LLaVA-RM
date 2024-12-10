"""
curl -X POST http://127.0.0.1:3100/process \                        [±main ●]
-H "Content-Type: application/json" \
-d '{"instruction": "move the yellow knife to the right of the pan", "image_path": "images/0000000.jpg"}'
"""

from reward.moe_model import RewardModel

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn
import json_numpy as json
import numpy as np

app = FastAPI()

backbone_path = "models/MoE-LLaVA-StableLM-1.6B-4e"
lora_path = "models/checkpoint-2800/adapter_model/lora_default"
reward_model = RewardModel(backbone_path, lora_path)

class InputData(BaseModel):
    instruction: str
    image_path: str

@app.get("/")
async def read_root():
    return {"message": "MOE server up"}

@app.post("/process")
async def process_data(request: Request):
    body = await request.body()
    data = json.loads(body)

    instruction = data.get("instruction")
    image_path = data.get("image_path")
    action = data.get("action")

    if not isinstance(instruction, str):
        raise HTTPException(status_code=400, detail="Instruction must be a string")
    if not isinstance(image_path, str):
        raise HTTPException(status_code=400, detail="Image path must be a string")

    action_array = np.array(action)

    if action_array.ndim != 2:
        raise HTTPException(status_code=400, detail="Action must be a 2D array")

    rewards = reward_model.get_rewards(instruction, action_array, image_path)

    return {"rewards": rewards}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3100)



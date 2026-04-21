# scripts/test_infer_processed_qwen35.py
from datasets import load_from_disk
from unsloth import FastVisionModel

MODEL_PATH = "models/Qwen3.5-9B"
DATA_PATH = "data/processed/vqa-rad/validation"
INDEX = 0

import io
from PIL import Image

ds = load_from_disk(DATA_PATH)
sample = ds[INDEX]

user_msg = [m for m in sample["messages"] if m["role"] == "user"][0]
assistant_msg = [m for m in sample["messages"] if m["role"] == "assistant"][0]

raw_image = [x["image"] for x in user_msg["content"] if x["type"] == "image"][0]
if isinstance(raw_image, dict) and "bytes" in raw_image:
    image = Image.open(io.BytesIO(raw_image["bytes"])).convert("RGB")
else:
    image = raw_image

# 将图片保存到本地供人工查看
save_path = "test_preview_image.jpg"
image.save(save_path)
print(f"\n[INFO] 已将当前测试的图片保存至本地: {save_path}")

question = "\n".join([x["text"] for x in user_msg["content"] if x["type"] == "text"])
gold = "\n".join([x["text"] for x in assistant_msg["content"] if x["type"] == "text"])

print("QUESTION:\n", question)
print("\nGOLD:\n", gold)

model, tokenizer = FastVisionModel.from_pretrained(
    model_name=MODEL_PATH,
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)
FastVisionModel.for_inference(model)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "从这张图片中你能看出什么吗？"},
        ],
    }
]

input_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

print("\nINPUT_TEXT:\n", input_text)

inputs = tokenizer(
    text=input_text,
    images=image,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")

print("\n--- 验证模型是否看到图片 ---")
print("1. Tokenizer 输出的张量键:", list(inputs.keys()))
if "pixel_values" in inputs:
    print("2. 图像特征张量的形状 (pixel_values):", inputs["pixel_values"].shape)
elif "images" in inputs:
    print("2. 图像特征张量的形状 (images):", inputs["images"].shape)

print("3. 打包后整体文本 token 数量:", inputs["input_ids"].shape[1])
print("----------------------------\n")

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    use_cache=True,
)

pred = tokenizer.batch_decode(
    outputs[:, inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
)[0]

print("\nPRED:\n", pred.strip())
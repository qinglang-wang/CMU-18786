import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def evaluate_modern_llm(model_id, eval_corpus_path):
    print(f"Loading model: {model_id}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )

    correct_v1 = 0
    correct_v2 = 0
    total = 0

    with open(eval_corpus_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    logs = []
    
    # Modification 1：System Prompt
    system_instruction = "You are a helpful factual assistant. Answer the user's question strictly with ONLY the exact name of the city, state, or country. Do not write full sentences. Do not use <think> tags. Just output the location name."

    for line in tqdm(lines, desc="Evaluating"):
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue

        question_v1 = parts[0]
        question_v2 = f"What is the birthplace of {question_v1[10:-6]}?"
        answer = parts[1]

        # Inference: Variant 1
        messages_v1 = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": question_v1}
        ]
        text_v1 = tokenizer.apply_chat_template(messages_v1, tokenize=False, add_generation_prompt=True)
        inputs_v1 = tokenizer([text_v1], return_tensors="pt").to(device)
        
        outputs_v1 = model.generate(**inputs_v1, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        pred_v1 = tokenizer.decode(outputs_v1[0][inputs_v1.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # Inference: Variant 2
        messages_v2 = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": question_v2}
        ]
        text_v2 = tokenizer.apply_chat_template(messages_v2, tokenize=False, add_generation_prompt=True)
        inputs_v2 = tokenizer([text_v2], return_tensors="pt").to(device)
        
        outputs_v2 = model.generate(**inputs_v2, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        pred_v2 = tokenizer.decode(outputs_v2[0][inputs_v2.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # Modification 2：Output parse
        is_correct_v1 = answer.lower() in pred_v1.lower()
        is_correct_v2 = answer.lower() in pred_v2.lower()

        if is_correct_v1:
            correct_v1 += 1
        if is_correct_v2:
            correct_v2 += 1
        total += 1

        if total <= 10:
            logs.append(f"\n[Example {total}]")
            logs.append(f"  Target Answer: '{answer}'")
            logs.append(f"  V1 Output: '{pred_v1}' | Correct: {is_correct_v1}")
            logs.append(f"  V2 Output: '{pred_v2}' | Correct: {is_correct_v2}")

    [print(line) for line in logs]
    print(f"\nFinal Result:")
    print(f"  Variant 1: Correct: {correct_v1} out of {total} ({correct_v1/total*100:.2f}%)")
    print(f"  Variant 2: Correct: {correct_v2} out of {total} ({correct_v2/total*100:.2f}%)")

if __name__ == '__main__':
    evaluate_modern_llm("Qwen/Qwen3-1.7B", "birth_dev.tsv")
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import LoraConfig, PeftConfig, PeftModel
from vllm import LLM, SamplingParams
import gc

max_model_len, tp_size = 4096, 1
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import numpy as np

torch.random.manual_seed(0)

# Eval Plan
def evaluate_responses(generated_judgements, labels):
    correct = 0
    correct_A = 0
    correct_B = 0
    unstructered = 0
    for i, judgement in enumerate(generated_judgements):
        if(labels[i] == 1 and "[[A]]" in judgement):
            correct += 1
            correct_A += 1
        elif(labels[i] == 0 and "[[B]]" in judgement):
            correct += 1
            correct_B += 1
        elif("[[A]]" not in judgement and "[[B]]" not in judgement):
            unstructered += 1
    print(f"Correct: {correct}, Correct_A: {correct_A}, Correct_B: {correct_B}, Unstructured: {unstructered}")
    acc = correct/len(labels)
    print(f"Accuracy: {acc}")
    acc_corrected = correct/(len(labels) - unstructered)
    print(f"Corrected Accuracy: {acc_corrected}")
    return acc, acc_corrected, unstructered
    

def load_textfile_as_string(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

def create_rewardbench_jsonl(judge_prompt_file):
    from datasets import load_dataset
    import numpy as np
    ds = load_dataset("allenai/reward-bench")

    judging_template = load_textfile_as_string(judge_prompt_file)

    judge_prompts = []
    labels = []
    for idx, point in enumerate(ds['filtered']):
        instruction = point['prompt']
        if(np.random.rand() < 0.5):
            response_A = point['chosen']
            response_B = point['rejected']
            labels.append(1)
        else:
            response_A = point['rejected']
            response_B = point['chosen']
            labels.append(0)
        messages = [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": judging_template.format(input=instruction, 
                                                                        generation=response_A, 
                                                                        generation2=response_B)},
                ]
        judge_prompts.append(messages)
        
    return judge_prompts, labels


if __name__ == "__main__":
    results = []
    
    judge_prompts, labels = create_rewardbench_jsonl("./eval_plan.prompt")

    original_model_path = "microsoft/Phi-3.5-mini-instruct"
    new_model_path = None

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=900)
    llm = LLM(model=original_model_path, tensor_parallel_size=tp_size, max_model_len=max_model_len, 
            trust_remote_code=True, enforce_eager=True)# gpu_memory_utilization=0.15)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    prompt_token_ids = [tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True) for messages in judge_prompts]
    outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)

    generated_judgements = [output.outputs[0].text for output in outputs]
    print("Sample Judgement:")
    print(generated_judgements[0])
    print("\n\n")

    accs = evaluate_responses(generated_judgements, labels)
    results.append(accs)

    # Delete the llm object and free the memory
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    # Reset CUDA device to fully clear memory
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()  # Wait for all streams on the current device
    print("Successfully delete the llm pipeline and free the GPU memory!")

    steps = np.arange(500, 2000, 500)
    for step_count in steps:
        print("*"*100)
        print("*"*100)
        print("*"*100)
        print("\n")
        print(f"Starting step={step_count}!")
        # new_model_path = f"./outputs/full_model/checkpoint-{step_count}"
        new_model_path = f"./saved_model/iter-{step_count}"
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=900)
        llm = LLM(model=new_model_path, tensor_parallel_size=tp_size, max_model_len=max_model_len, 
                trust_remote_code=True, enforce_eager=True)
        tokenizer = AutoTokenizer.from_pretrained(new_model_path)
        prompt_token_ids = [tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True) for messages in judge_prompts]
        outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
    
        generated_judgements = [output.outputs[0].text for output in outputs]
        print("Sample Judgement:")
        print(generated_judgements[0])
        print("\n\n")

        accs = evaluate_responses(generated_judgements, labels)
        results.append(accs)

        # Delete the llm object and free the memory
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        # Reset CUDA device to fully clear memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()  # Wait for all streams on the current device
        print("Successfully delete the llm pipeline and free the GPU memory!")

    print("*"*100)
    print("*"*100)
    print("*"*100)
    print("\n")
    print("Results!")
    for result in results:
        print(result)
        

    

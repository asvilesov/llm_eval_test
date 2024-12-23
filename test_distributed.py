import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import LoraConfig

torch.random.manual_seed(0)

peft_config = LoraConfig(
    r=128,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    modules_to_save=["lm_head", "embed_token"],
    task_type="CAUSAL_LM",
)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct", 
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda", 
    trust_remote_code=True, 
    attn_implementation="flash_attention_2" 
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

dataset_path = "./all_judgements.json"
dataset = load_dataset("json", data_files=dataset_path, split="train")

training_args = SFTConfig(packing=False,
                          max_seq_length=3500,
                          per_device_train_batch_size=2,
                          output_dir="./outputs/lora_model",
                          bf16=True,
                          logging_steps=50,
                          num_train_epochs=10,
                          gradient_accumulation_steps=4,
                          learning_rate=1e-5,
                          lr_scheduler_type="cosine",
                          warmup_steps=100,
                          )

instruction_template = "<|user|>\n"
response_template = "<|assistant|>\n"
collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, 
                                           response_template=response_template, 
                                           tokenizer=tokenizer, mlm=False)

trainer = SFTTrainer(
    model,
    args=training_args,
    train_dataset=dataset,
    # peft_config=peft_config,
    data_collator=collator
)


trainer.train()
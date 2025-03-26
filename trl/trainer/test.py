import tempfile
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

def reward_num_unique_chars(completions, **kwargs):
    return [len(set(c)) for c in completions]

dataset = load_dataset("trl-lib/tldr", split="train")

with tempfile.TemporaryDirectory() as tmp_dir:
    training_args = GRPOConfig(
        output_dir=tmp_dir,
        learning_rate=1e-4,
        bf16=True,
        logging_steps=10,
        #use_vllm=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        epsilon_high=0.28,
        num_iterations=10,
    )
    trainer = GRPOTrainer(
        model="Qwen/Qwen2.5-0.5B",
        reward_funcs=reward_num_unique_chars,
        args=training_args,
        train_dataset=dataset,
        peft_config=LoraConfig(),
    )
    trainer.train()
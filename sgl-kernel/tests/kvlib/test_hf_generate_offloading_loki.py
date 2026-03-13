import torch
from transformers import AutoTokenizer, AutoConfig
import json
from datasets import Dataset
from fastchat.model import get_conversation_template
from transformers.generation.configuration_utils import GenerationConfig
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# dataset_prompt = 'Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: '
# dataset_path = "/nfs/shared_LLM_dataset/LongBench/data/passage_retrieval_en_e.jsonl"

dataset_prompt = 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:'
dataset_path = "/nfs/shared_LLM_dataset/LongBench/data/qasper_e.jsonl"

with open(dataset_path, "r") as f:
    lines = f.readlines()

dataset = []
for line in lines:
    eg = json.loads(line)
    instance = {
        "context": eg["context"],
        "input": eg["input"],
        "answers": [eg["answers"]]
    }
    instance["length"] = len(instance["context"].split())
    instance["all_classes"] = None
    dataset.append(instance)

dataset = Dataset.from_list(dataset)

if __name__ == "__main__":
    # i = int(sys.argv[1])
    device = "cuda:0"
    torch.cuda.set_device(device)
    torch.manual_seed(42)

    model_path = "/nfs/shared_LLM_model/gradientai/Llama-3-8B-Instruct-Gradient-1048k"
    pca_weights = "/root/workspace/myoffloading/hata/myTransformer/auxiliary/loki_pca/Llama-3-8B-Instruct-Gradient-1048k"
    attn_pattern = "/root/workspace/myoffloading/hata/myTransformer/auxiliary/attn_pattern/Llama-3-8B-Instruct-Gradient-1048k"
    cosine_padding = 0.1

    model_path = "/nfs/shared_LLM_model/Qwen/Qwen2.5-14B-Instruct-1M"
    pca_weights = "/root/workspace/myoffloading/hata/myTransformer/auxiliary/loki_pca/Qwen2.5-14B-Instruct-1M"
    attn_pattern = "/root/workspace/myoffloading/hata/myTransformer/auxiliary/attn_pattern/Qwen2.5-14B-Instruct-1M"
    cosine_padding = 0.05

    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.torch_dtype = torch.float16
    config._attn_implementation = "flash_attention_2"
    print(config)

    if "qwen" in model_path.lower():
        from myTransformer.models.qwen2.modeling_qwen2_offloading_duohead import LokiQwen2ForCausalLM
        model = LokiQwen2ForCausalLM.from_pretrained(model_path,
                                                     torch_dtype=torch.float16,
                                                     config=config)
    else:
        from myTransformer.models.llama.modeling_llama_offloading_duohead import LokiLlamaForCausalLM
        model = LokiLlamaForCausalLM.from_pretrained(model_path,
                                                     torch_dtype=torch.float16,
                                                     config=config)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    print(model)

    model = model.eval().to(device)

    generation_kwargs = {
        "num_skip_layers": 1,
        "max_batch_size": 1,
        "max_sequence_length": 32768,
        "max_gpu_memory_size": 2.0 * 1024 * 1024 * 1024,
        "max_cpu_memory_size": 50 * 1024 * 1024 * 1024,
        "token_budget": 0.1,
        "sink_budget": 4,
        "recent_budget": 64,
        "aux_data_path": pca_weights,
        "attn_pattern_path": attn_pattern,
        "num_channels": 32,
        "reuse_threshold_upper": 0.80,
        "reuse_threshold_lower": -1.0,
        "deacy_p": 3,
        "cosine_padding": cosine_padding,
        "num_omp_threads": 4,
        "num_overlapped_heads": 3,
        "new_config": False,
    }
    print(generation_kwargs)
    generation_config = GenerationConfig(**generation_kwargs)

    it = 0

    for ctx in dataset:
        prompt = dataset_prompt.format(context=ctx["context"],
                                       input=ctx["input"])

        # llama3.1 & glm
        if any([
                x in model_path.lower() for x in [
                    "llama-3.1", "llama3.1", "llama_3.1", "llama-3", "llama3",
                    "llama_3", "glm", "qwen"
                ]
        ]):
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(messages,
                                                   add_generation_prompt=True,
                                                   tokenize=False)
        # longchat
        elif any([x in model_path.lower() for x in ["longchat"]]):
            conv = get_conversation_template("vicuna")
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

        encoded = tokenizer(prompt, return_tensors="pt")

        input_ids = encoded.input_ids.to(model.device)
        attention_mask = encoded.attention_mask.to(model.device)

        input_len = input_ids.shape[1]
        output = model.generate(inputs=input_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=32,
                                generation_config=generation_config)
        torch.cuda.synchronize()
        output = tokenizer.decode(output[0][input_len:],
                                  skip_special_tokens=True)
        print()
        print(output)

        it += 1
        if it > 6:
            break

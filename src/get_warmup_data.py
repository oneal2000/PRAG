import os
import json
import pandas as pd
import random
import torch
from tqdm import tqdm

import prompt_template
from utils import get_model, evaluate
from root_dir_path import ROOT_DIR
from augment import load_complexwebquestions, load_popqa

random.seed(42)


# direct, without fewshot and cot 
# for popqa and complexwebquestions 
def create_direct():
    for name, func in (("popqa", load_popqa), ("complexwebquestions", load_complexwebquestions)):
        dataset = func(os.path.join(ROOT_DIR, "data", name))["total"]
        dataset = dataset[1000:] # to prevent data leakage, only the first 300 entries were actually tested.
        for data in dataset:
            if isinstance(data["answer"], list):
                data["answer"] = data["answer"][0]
            del_keys = [k for k in data.keys() if k != "answer" and k != "question"]
            for k in del_keys:
                data.pop(k)
        
        output_dir = os.path.join(ROOT_DIR, "warmup", "data", "direct")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, name+".json"), "w") as fout:
            json.dump(dataset, fout, indent=4)


def load_2wikimultihopqa(data_path):
    with open(os.path.join(data_path, "dev.json"), "r") as fin:
        dataset = json.load(fin)
    with open(os.path.join(data_path, "id_aliases.json"), "r") as fin:
        aliases = dict()
        for li in fin:
            t = json.loads(li)
            aliases[t["Q_id"]] = t["aliases"]
    with open(os.path.join(data_path, "queries.jsonl"), "r") as fin:
        context_data = []
        for li in fin:
            context_data.append(json.loads(li))
    new_dataset = []
    for did, data in enumerate(dataset):
        context = [v[1] for v in context_data[did]["metadata"]["ctxs"]]
        answer = context_data[did]["metadata"]["answer"]
        val = {
            "qid": data["_id"], 
            "test_id": did, 
            "question": data["question"], 
            "answer": answer,
            "context": context,
            "type": data["type"],
        }
        new_dataset.append(val)
    return {"total": new_dataset}


def load_hotpotqa(data_path):
    with open(os.path.join(data_path, 'hotpot_dev_distractor_v1.json'), 'r') as fin:
        dataset = json.load(fin)
    new_dataset = []
    for did, data in enumerate(dataset):
        all_ctxs = {}
        for name, text in data["context"]:
            all_ctxs[name] = text
        context = []
        flag = False
        for name, id in data["supporting_facts"]:
            if id > len(all_ctxs[name]):
                print("### Error supporting facts id: ", data["_id"], id, len(all_ctxs[name]))
                flag = True
                continue
            context.append(all_ctxs[name][id])
        if flag:
            continue
        val = {
            'qid': data['_id'], 
            'test_id': did, 
            'question': data['question'], 
            'answer': data['answer'], 
            "context": context,
            "type": data["type"],
        }

        new_dataset.append(val)
    return {"total": new_dataset}


USER_PROMPT_WITH_COT = "You should use the context provided below to answer the question. Please follow the same structure as the example.\n\n\
Here are some examples of how to answer the questions:\n\
{fewshot}\n\
Here is the context for the question:\n\
{context}\n\n\
The correct answer to the given question is {answer}. Now, please answer the question in the same format as above.\n\
Question: {question}"
ASSISTANT_PROMPT_WITH_COT = "Answer: "


# fewshot and cot
# for 2wikimultihopqa and hotpotqa
def create_cot():
    AIM_CNT_EACH_DATASET = 300
    
    output_dataset = []

    model_name = "llama3-8b-instruct"
    model, tokenizer, generation_config = get_model(model_name, max_new_tokens=128)
    model.eval()

    for name, func in (("2wikimultihopqa", load_2wikimultihopqa), ("hotpotqa", load_hotpotqa)):
        print(f"### solving {name} ###")
        dataset = func(os.path.join(ROOT_DIR, "data", name))["total"]
        # prevent data leakage
        mark_idx = {}
        for did, data in enumerate(dataset):
            typ = data["type"]
            if typ not in mark_idx:
                mark_idx[typ] = {"cnt": 1, "last_idx": did}
            else:
                if mark_idx[typ]["cnt"] >= 300:
                    continue
                mark_idx[typ]["cnt"] += 1
                mark_idx[typ]["last_idx"] = did
        last_idx = max(v["last_idx"] for k, v in mark_idx.items())
        dataset = dataset[last_idx + 1000:]
        random.shuffle(dataset)

        last_cnt = len(output_dataset)
        pbar = tqdm(total=AIM_CNT_EACH_DATASET)
        prompt_template.get_fewshot(name)
        for did, data in enumerate(dataset):
            passages = data["context"]
            context = ""
            for pid, psg in enumerate(passages):
                context += f"{pid+1}. {psg.strip()}\n"
            user_content = USER_PROMPT_WITH_COT.format(fewshot=prompt_template.fewshot, 
                                                 context=context,
                                                 question=data["question"], 
                                                 answer=data["answer"])
            messages = [{"role": "user", "content": user_content}]
            inputs = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True
            )
            inputs += tokenizer.encode(ASSISTANT_PROMPT_WITH_COT, add_special_tokens=False)
            input_len = len(inputs)
            input_ids = torch.tensor(inputs).unsqueeze(0).to(model.device)
            with torch.no_grad():
                output = model.generate(
                    input_ids, 
                    attention_mask = torch.ones(input_ids.shape).to(model.device),
                    **generation_config)
            output = output.sequences[0][input_len:]
            text = tokenizer.decode(output, skip_special_tokens=True)
            if text is None or not "the answer is" in text:
                continue
            for stop_words in ["\n\n", "Questions"]:
                if stop_words in text:
                    text = (text[:text.find(stop_words)]).strip()
            if evaluate(text, data["answer"], with_cot=True)["em"] == "1":
                data["cot"] = text
                data["from"] = name
                output_dataset.append(data)
                pbar.update(1)
                if len(output_dataset) - last_cnt == AIM_CNT_EACH_DATASET:
                    # print(f"### {name}: {AIM_CNT_EACH_DATASET} / {did+1} ###")
                    break
    
    output_dir = os.path.join(ROOT_DIR, "warmup", "data", "cot")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train_data.json"), "w") as fout:
        json.dump(output_dataset, fout, indent=4)


if __name__ == "__main__":
    create_direct()
    create_cot()
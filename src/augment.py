import os
import json
import random
import argparse
import pandas as pd
from tqdm import tqdm

from retrieve.retriever import bm25_retrieve
from utils import get_model, model_generate
from root_dir_path import ROOT_DIR

random.seed(42)


def load_popqa(data_path):
    data_path = os.path.join(data_path, "popQA.tsv")
    dataset = pd.read_csv(data_path, sep="\t")
    new_dataset = []
    for did in range(len(dataset)):
        data = dataset.iloc[did]
        question = data["question"]
        answer = [data["obj"]] + eval(data["o_aliases"])
        val = {
            "test_id": did, 
            "question": question, 
            "answer": answer,
        }        
        new_dataset.append(val)
    return {"total": new_dataset}


def load_complexwebquestions(data_path):
    data_path = os.path.join(data_path, "ComplexWebQuestions_dev.json")
    with open(data_path, "r") as fin:
        dataset = json.load(fin)
    new_dataset = []
    for did, data in enumerate(dataset):
        question = data["question"]
        answer = []
        for ans in data["answers"]:
            answer.append(ans["answer"])
            answer.extend(ans["aliases"])
        answer = list(set(answer))
        val = {
            "test_id": did, 
            "question": question, 
            "answer": answer,
        }        
        new_dataset.append(val)
    ret = {"total": new_dataset}
    return ret


def load_2wikimultihopqa(data_path):
    with open(os.path.join(data_path, "dev.json"), "r") as fin:
        dataset = json.load(fin)
    with open(os.path.join(data_path, "id_aliases.json"), "r") as fin:
        aliases = dict()
        for li in fin:
            t = json.loads(li)
            aliases[t["Q_id"]] = t["aliases"]
    new_dataset = []
    type_to_dataset = {}
    for did, data in enumerate(dataset):
        ans_id = data["answer_id"]
        val = {
            "qid": data["_id"], 
            "test_id": did, 
            "question": data["question"], 
            "answer": aliases[ans_id] if ans_id else data["answer"]
        }
        golden_passages = []
        contexts = {name: " ".join(sents) for name, sents in data["context"]}
        for fact_name, _sent_id in data["supporting_facts"]:
            psg = contexts[fact_name]
            golden_passages.append(psg)
        val["golden_passages"] = golden_passages
        val["type"] = data["type"]
        new_dataset.append(val)
        if data["type"] not in type_to_dataset:
            type_to_dataset[data["type"]] = []
        type_to_dataset[data["type"]].append(val)
    ret = {"total": new_dataset}
    ret.update(type_to_dataset)
    return ret


def load_hotpotqa(data_path):
    data_path = os.path.join(data_path, "hotpot_dev_distractor_v1.json")
    with open(data_path, "r") as fin:
        dataset = json.load(fin)
    new_dataset = []
    type_to_dataset = {}
    for did, data in enumerate(dataset):
        val = {
            "qid": data["_id"], 
            "test_id": did, 
            "question": data["question"], 
            "answer": data["answer"]
        }
        tmp = []
        contexts = {name: "".join(sents) for name, sents in data["context"]}
        for fact_name, _sent_id in data["supporting_facts"]:
            psg = contexts[fact_name]
            tmp.append(psg)
        golden_passages = []
        for p in tmp:
            if p not in golden_passages:
                golden_passages.append(p)
        val["golden_passages"] = golden_passages
        val["type"] = data["type"]
        new_dataset.append(val)
        if data["type"] not in type_to_dataset:
            type_to_dataset[data["type"]] = []
        type_to_dataset[data["type"]].append(val)
    ret = {"total": new_dataset}
    ret.update(type_to_dataset)
    return ret


def load_default_format_data(data_path):
    filename = data_path.split("/")[-1]
    assert filename.endswith(".json"), f"Need json data: {data_path}"
    with open(data_path, "r") as fin:
        dataset = json.load(fin)
    for did, data in enumerate(dataset):
        assert "question" in data, f"\"question\" not in data, {data_path}"
        question = data["question"]
        assert type(question) == str, f"\"question\": {question} should be a string"
        assert "answer" in data, f"\"answer\" not in data, {data_path}"
        answer = data["answer"]
        assert type(answer) == str or \
               (type(answer) == list and (not any(type(a) != str for a in answer))), \
               f"\"answer\": {answer} should be a string or a list[str]" 
        data["test_id"] = did
    return {filename: dataset}


def get_rewrite(passage, model_name, model=None, tokenizer=None, generation_config=None):
    rewrite_prompt = "Rewrite the following passage. While keeping the entities, proper nouns, and key details such as names, locations, and terminology intact, create a new version of the text that expresses the same ideas in a different way. Make sure the revised passage is distinct from the original one, but preserves the core meaning and relevant information.\n{passage}"
    return model_generate(rewrite_prompt.format(passage=passage), model, tokenizer, generation_config)


qa_prompt_template = "I will provide a passage of text, and you need to generate three different questions based on the content of this passage. Each question should be answerable using the information provided in the passage. Additionally, please provide an appropriate answer for each question derived from the passage.\n\
You need to generate the question and answer in the following format:\n\
[\n\
    {{\n\
        \"question\": \"What is the capital of France?\",\n\
        \"answer\": \"Paris\"\n\
        \"full_answer\": \"The capital of France is Paris.\"\n\
    }}, \n\
]\n\n\
This list should have at least three elements. You only need to output this list in the above format.\n\
Passage:\n\
{passage}"

def fix_qa(qa):
    if isinstance(qa, list):
        if len(qa) >= 3:
            qa = qa[:3]
            for data in qa:
                if "question" not in data or "answer" not in data or "full_answer" not in data:
                    return False, qa
                if isinstance(data["answer"], list):
                    data["answer"] = ", ".join(data["answer"])
                if isinstance(data["answer"], int):
                    data["answer"] = str(data["answer"])
                if data["answer"] is None:
                    data["answer"] = "Unknown"
            return True, qa
    return False, qa

def get_qa(passage, model_name, model=None, tokenizer=None, generation_config=None):

    def fix_json(output):
        if model_name == "llama3.2-1b-instruct":
            output = output[output.find("["):]
            if output.endswith(","):
                output = output[:-1]
            if not output.endswith("]"):
                output += "]"
        elif model_name == "llama3-8b-instruct":
            if "[" in output:
                output = output[output.find("["):] 
            if "]" in output:
                output = output[:output.find("]")+1]
        return output

    try_times = 100
    prompt = qa_prompt_template.format(passage=passage)
    output = None
    while try_times:
        output = model_generate(prompt, model, tokenizer, generation_config)
        output = fix_json(output)
        try:
            qa = json.loads(output)
            ret, qa = fix_qa(qa)
            if ret:
                return qa
        except:
            try_times -= 1
    return output
    

def main(args):
    output_dir = os.path.join(ROOT_DIR, "data_aug", args.dataset, args.model_name)
    os.makedirs(output_dir, exist_ok=True)

    print("### Loading dataset ###")
    if f"load_{args.dataset}" in globals():
        load_func = globals()[f"load_{args.dataset}"]
    else:
        load_func = globals()["load_default_format_data"]
    load_dataset = load_func(args.data_path)
    if len(load_dataset) == 1:
        solve_dataset = load_dataset
    else:
        solve_dataset = {k: v for k, v in load_dataset.items() if k != "total"}
        with open(os.path.join(output_dir, "total.json"), "w") as fout:
            json.dump(load_dataset["total"][:args.sample], fout, indent=4)
    
    model, tokenizer, _ = get_model(args.model_name)
    generation_config = dict(
        max_new_tokens=512,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        temperature=0.7,
        top_k=50,
    )

    for filename, dataset in solve_dataset.items():
        print(f"### Solving {filename} ###")
        output_file = os.path.join(
            output_dir, 
            filename if filename.endswith(".json") else filename + ".json"
        )
        ret = []
        dataset = dataset[:args.sample]
        pbar = tqdm(total = args.sample * args.topk)
        for data in dataset:
            passages = bm25_retrieve(data["question"], topk=args.topk+10)
            final_passages = []
            data["augment"] = []
            for psg in passages:
                val = { 
                    "pid": len(final_passages), 
                    "passage": psg, 
                    f"{args.model_name}_rewrite": get_rewrite(psg, args.model_name, model, tokenizer, generation_config)
                }
                qa = get_qa(psg, args.model_name, model, tokenizer, generation_config)
                if fix_qa(qa)[0] == False: # skip error passage
                    continue
                val[f"{args.model_name}_qa"] = qa
                data["augment"].append(val)
                final_passages.append(psg)
                pbar.update(1)
                if len(data["augment"]) == args.topk:
                    break
            data["passages"] = final_passages
            ret.append(data)
        with open(output_file, "w") as fout:
            json.dump(ret, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--sample", type=int, required=True)
    parser.add_argument("--topk", type=int, default=3) 
    args = parser.parse_args()
    print(args)
    main(args)
from root_dir_path import ROOT_DIR

current_dataset = None
fewshot = None
fewshot_path = f"{ROOT_DIR}/src/fewshot/"

USER_PROMPT = "You should answer the question by referring to the knowledge provided below and integrating your own knowledge.\n\
{passages}\n\n\
Question: {question}"

USER_PROMPT_WITH_COT = "You should reference the knowledge provided below and combine it with your own knowledge to answer the question. Please follow the format of the example I provided above.\n\
Here are some examples about how to answer the questions.\n\
{fewshot}\
Here are some reference.\n\
{passages}\n\n\
Let's think step by step. Answer the questions in the same format as above.\n\
Question: {question}"

ASSISTANT_PROMPT = "The answer is {answer}"
ASSISTANT_PROMPT_WITH_COT = "Answer: {answer}"

def _get_prompt(question, passages=None, answer=None):
    question = question.strip()
    if not question.endswith('?'):
        question = question.strip() + '?'
    elif question.endswith(' ?'):
        question = (question[:-1]).strip() + '?'
     
    if passages and not isinstance(passages, list):
        passages = [passages]
    
    if answer is None:
        answer = ""
    else:
        answer = answer.strip()
        if not answer.endswith('.'):
            answer += "."
    return question, passages, answer


def get_fewshot(dataset):
    import json
    global current_dataset
    global fewshot
    # assert current_dataset is None
    if dataset.endswith("_golden"):
        dataset = dataset.split("_golden")[0]
    current_dataset = dataset
    with open(fewshot_path + dataset + ".json", "r") as fin:
        tmp = json.load(fin)
    fewshot = ""
    for data in tmp:
        q = data["question"]
        a = data["answer"]
        fewshot += f"Question: {q}\nAnswer: {a}\n\n"


def get_prompt(tokenizer, question, passages=None, answer=None, with_cot=False):
    question, passages, answer = _get_prompt(question, passages, answer)
    contexts = ""
    if passages:
        for pid, psg in enumerate(passages):
            contexts += f"Passage {pid+1}: {psg}\n"
    if not with_cot:
        user_content = USER_PROMPT.format(question=question, passages=contexts)
        assistant_content = ASSISTANT_PROMPT.format(answer=answer)
    else:
        assert fewshot is not None
        user_content = USER_PROMPT_WITH_COT.format(question=question, passages=contexts, fewshot=fewshot)
        assistant_content = ASSISTANT_PROMPT_WITH_COT.format(answer=answer)

    messages = [{
        "role": "user",
        "content": user_content,
    }]

    inputs = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True)
    inputs += tokenizer.encode(assistant_content, add_special_tokens=False)
    return inputs
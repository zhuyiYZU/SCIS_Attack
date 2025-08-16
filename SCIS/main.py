import torch
import re
from modelscope import AutoModelForCausalLM, AutoTokenizer
import joblib
import time

# 后处理函数同上
def is_ngram_repeated(ids, n):
    if len(ids) < n:
        return False
    ngrams = set()
    for i in range(len(ids) - n + 1):
        ng = tuple(ids[i:i+n])
        if ng in ngrams:
            return True
        ngrams.add(ng)
    return False

def clean_punctuation(text):
    text = re.sub(r'[!！]{2,}', '!', text)
    text = re.sub(r'[.。]{2,}', '.', text)
    text = re.sub(r'[,，]{2,}', ',', text)
    text = re.sub(r'([!?.,])\1+', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def remove_repeated_phrases(text, min_len=5):
    seen, out = set(), []
    for sent in text.split('. '):
        c = sent.strip().lower()
        if len(c) >= min_len and c not in seen:
            seen.add(c)
            out.append(sent.strip())
    return '. '.join(out)

def truncate_incomplete_sentence(text):
    if not text.endswith(('.', '!', '?')):
        i = text.rfind('.')
        if i > 0:
            return text[:i+1]
    return text

def capitalize_sentences(text):
    parts = re.split('([.!?])', text)
    return ''.join(p.strip().capitalize() if i%2==0 else p for i,p in enumerate(parts))

def post_process(txt):
    txt = clean_punctuation(txt)
    txt = remove_repeated_phrases(txt)
    txt = truncate_incomplete_sentence(txt)
    txt = capitalize_sentences(txt)
    return txt

# 生成函数同上
def generate_with_similarity_v2(
    model, tokenizer, model_inputs,
    max_new_tokens=200,
    no_repeat_ngram_size=2,
    similarity_threshold=0.8,
    tradeoff_mode='semantic-diverse',
    device="cuda"
):
    generated_ids = model_inputs["input_ids"].tolist()[0].copy()
    input_len = len(generated_ids)
    embedding = model.get_input_embeddings()
    model.eval()

    while len(generated_ids) - input_len < max_new_tokens:
        outputs = model(**model_inputs, output_hidden_states=True)
        logits = outputs.logits
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        topk = 100
        topk_probs, topk_idx = torch.topk(probs, topk, dim=-1)
        topk_probs = topk_probs[0]
        topk_idx = topk_idx[0]
        tok_embs = embedding(topk_idx)
        top1_vec = tok_embs[0]
        top1_norm = top1_vec / top1_vec.norm()
        tok_norms = tok_embs / tok_embs.norm(dim=-1, keepdim=True)
        sims = tok_norms @ top1_norm
        candidates = [
            (int(tok), float(sim), float(prob))
            for tok, sim, prob in zip(topk_idx, sims, topk_probs)
            if sim >= similarity_threshold
        ]
        selected = None
        if candidates:
            if tradeoff_mode == 'semantic-max':
                candidates.sort(key=lambda x: -x[1])
            elif tradeoff_mode == 'semantic-diverse':
                candidates.sort(key=lambda x: (-x[1], x[2]))
            elif tradeoff_mode == 'semantic-minprob':
                candidates.sort(key=lambda x: x[2])
            else:
                candidates.sort(key=lambda x: (-x[1], x[2]))
            selected = candidates[0][0]
        else:
            selected = int(topk_idx[0])

        trial = generated_ids + [selected]
        if no_repeat_ngram_size > 0 and is_ngram_repeated(trial, no_repeat_ngram_size):
            selected = int(topk_idx[0])
        generated_ids.append(selected)
        new_input = torch.tensor([generated_ids], device=device)
        model_inputs["input_ids"] = new_input
        if "attention_mask" in model_inputs:
            model_inputs["attention_mask"] = torch.ones_like(new_input)
        if selected == tokenizer.eos_token_id:
            break

    new_ids = generated_ids[input_len:]
    raw = tokenizer.decode(new_ids, skip_special_tokens=True)
    return post_process(raw)


def predict_threshold(new_text):
    model, vectorizer = load_threshold_model()
    new_text_tfidf = vectorizer.transform([new_text])
    predicted_threshold = model.predict(new_text_tfidf)
    return predicted_threshold[0]

# -------- 单条文本主程序 --------
if __name__ == "__main__":
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        "mistral_model",
        torch_dtype=torch.float16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained("mistral_model")

    # 输入你想改写的文本
    text = "In 1954, major Serbian and Croatian writers, linguists and literary critics, backed by Matica srpska and Matica hrvatska, came together in an effort to bridge the cultural divides between the two nations. This collaboration aimed to promote mutual understanding and respect through literature and language. By fostering dialogue and cooperation, these intellectuals sought to strengthen the cultural ties that bind Serbs and Croats together despite their historical differences.Through joint publications, conferences, and educational initiatives, the Serbian and Croatian literary communities worked towards a shared vision of unity and reconciliation. This historic undertaking marked a significant step towards healing the wounds of the past and building a more harmonious future for both nations. The legacy of this collaboration continues to inspire efforts towards cultural exchange and cooperation in the region. "
    pred_th = 0.6

    prompt = f"Please repeat this paragraph：{text}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    truncated_text_pre = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    truncated_text = truncated_text_pre.split("：", 1)[-1].strip()
    inputs = {k: v.to(device) for k, v in inputs.items()}

    try:
        rewritten = generate_with_similarity_v2(
            model, tokenizer, inputs,
            max_new_tokens=512,
            no_repeat_ngram_size=0,
            similarity_threshold=pred_th,
            tradeoff_mode='semantic-minprob',
            device=device
        )
        time.sleep(1)
    except Exception as e:
        rewritten = f"ERROR: {e}"

    print(f"\nOriginal: {truncated_text}")
    print(f"Rewritten: {rewritten}")

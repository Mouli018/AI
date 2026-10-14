"""OPTIONAL: build your own domain QA set from PDFs (e.g. course lecture slides) with an LLM and publish it as
a dataset contribution.   python scripts/build_course_qa.py --pdf_dir lectures --per_chunk 2 --max_chunks 60
Output: data/course_qa.json (same format as HotpotQA, so `evaluate.py --data data/course_qa.json` works).
Questions are LLM-generated: spot-check a sample by hand and say so in the paper."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pypdf import PdfReader
from contextrouter.llm import LLM, Meter, parse_json

SYS = ('ROLE=QAGEN. From the passage write {n} factual questions that can be answered by a short phrase using ONLY '
       'the passage. Reply with JSON {{"qa": [{{"question": "...", "answer": "..."}}]}}.')

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", required=True)
    ap.add_argument("--per_chunk", type=int, default=2)
    ap.add_argument("--max_chunks", type=int, default=60)
    ap.add_argument("--words", type=int, default=140)
    ap.add_argument("--out", default="data/course_qa.json")
    a = ap.parse_args()
    llm, chunks = LLM(), []
    for pdf in sorted(Path(a.pdf_dir).glob("*.pdf")):
        for pn, page in enumerate(PdfReader(str(pdf)).pages):
            words = (page.extract_text() or "").split()
            for j in range(0, len(words), a.words):
                piece = words[j:j + a.words]
                if len(piece) >= 40:
                    chunks.append((f"{pdf.stem} p{pn + 1}.{j // a.words}", " ".join(piece)))
    chunks = chunks[: a.max_chunks]
    qs = []
    for title, text in chunks:
        r = llm.chat(SYS.format(n=a.per_chunk), f"PASSAGE:\n{text}", kind="main", max_tokens=300)
        obj = parse_json(r.text) or {}
        for i, qa in enumerate(obj.get("qa", [])[: a.per_chunk]):
            qs.append({"_id": f"course{len(qs)}", "question": qa["question"], "answer": qa["answer"], "type": "course",
                       "level": "n/a", "supporting_facts": [[title, 0]], "context": []})
            qs[-1]["_gold_title"] = title
    corpus = [[t, [x]] for t, x in chunks]
    import random
    for q in qs:
        t = q.pop("_gold_title")
        others = random.Random(q["_id"]).sample(corpus, min(8, len(corpus)))
        ctx = {c[0]: c for c in others}
        ctx[t] = [t, [dict(chunks)[t]]]
        q["context"] = list(ctx.values())
    Path(a.out).parent.mkdir(exist_ok=True)
    json.dump(qs, open(a.out, "w"))
    print(f"wrote {len(qs)} questions from {len(chunks)} chunks -> {a.out}")

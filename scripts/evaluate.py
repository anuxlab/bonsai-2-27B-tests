#!/usr/bin/env python3
"""Lightweight, reproducible OpenAI-compatible model evaluation runner."""
import argparse, json, os, re, time, urllib.request, urllib.error, statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TESTS=ROOT/"tests"
OUT=ROOT/"results"

def api_call(base, prompt, max_tokens, temperature, thinking):
    content=prompt
    if isinstance(prompt,dict): content=[{"type":"text","text":prompt["text"]},{"type":"image_url","image_url":{"url":prompt["image_url"]}}]
    payload={"messages":[{"role":"user","content":content}],
            "temperature":temperature,"max_tokens":max_tokens}
    if thinking is not None:
        payload["chat_template_kwargs"]={"enable_thinking":thinking}
    req=urllib.request.Request(base.rstrip("/")+"/chat/completions",
        data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
    t0=time.perf_counter()
    with urllib.request.urlopen(req,timeout=900) as resp:
        body=json.loads(resp.read())
    elapsed=time.perf_counter()-t0
    choice=body.get("choices",[{}])[0]
    message=choice.get("message",{})
    content=message.get("content") or ""
    if isinstance(content,list):
        content="".join(x.get("text","") for x in content if isinstance(x,dict))
    return {"text":content,"raw":body,"elapsed":elapsed,"usage":body.get("usage",{}),
            "finish_reason":choice.get("finish_reason"),"reasoning":message.get("reasoning_content","")}

def grade(case, text):
    checks=[]
    lower=text.lower()
    for needle in case.get("expected_contains",[]):
        checks.append({"check":"contains:"+needle,"passed":needle.lower() in lower})
    if "expected_regex" in case:
        checks.append({"check":"regex","passed":bool(re.search(case["expected_regex"],text,re.I|re.M))})
    if "min_words" in case:
        checks.append({"check":"min_words","passed":len(text.split())>=case["min_words"]})
    if "expected_lines" in case:
        actual=[x.strip() for x in text.strip().splitlines()]
        checks.append({"check":"exact_lines","passed":actual==case["expected_lines"]})
    if case.get("json_required"):
        try:
            parsed=json.loads(text.strip())
            checks.append({"check":"valid_json","passed":True})
            for key in case.get("json_keys",[]):
                checks.append({"check":"json_key:"+key,"passed":key in parsed})
        except Exception:
            checks.append({"check":"valid_json","passed":False})
    if case.get("code_required"):
        checks.append({"check":"code_block_or_code","passed":("```" in text) or bool(re.search(r"(^|\n)\s*(def |class |import |from )",text))})
    return {"passed":all(c["passed"] for c in checks) if checks else None,"checks":checks}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base-url",default=os.getenv("OPENAI_BASE_URL","http://127.0.0.1:8080/v1"))
    p.add_argument("--suites",default="basic",help="Comma-separated: basic,thinking,coding,agentic,advanced,all")
    p.add_argument("--max-tokens",type=int,default=2048)
    p.add_argument("--temperature",type=float,default=0.2)
    p.add_argument("--thinking",choices=["on","off","auto"],default="auto")
    p.add_argument("--repeat",type=int,default=1)
    args=p.parse_args()
    suites=["basic","thinking","coding","agentic","advanced","vision"] if args.suites=="all" else args.suites.split(",")
    OUT.mkdir(exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jsonl=OUT/f"run_{stamp}.jsonl"
    results=[]
    for suite in suites:
        cases=json.loads((TESTS/f"{suite}.json").read_text())
        for repeat in range(args.repeat):
            for case in cases:
                print(f"[{suite}] {case['id']} repeat={repeat+1}",flush=True)
                start=time.time()
                try:
                    thinking=None if args.thinking=="auto" else args.thinking=="on"
                    call=api_call(args.base_url,case["prompt"],args.max_tokens,args.temperature,thinking)
                    grading=grade(case,call["text"])
                    row={"suite":suite,"case_id":case["id"],"repeat":repeat+1,"status":"ok",
                         "started_at":datetime.fromtimestamp(start,timezone.utc).isoformat(),
                         "response":call["text"],"reasoning":call["reasoning"],
                         "elapsed_seconds":round(call["elapsed"],3),"usage":call["usage"],
                         "finish_reason":call["finish_reason"],"grading":grading}
                except Exception as e:
                    row={"suite":suite,"case_id":case["id"],"repeat":repeat+1,"status":"error",
                         "error":repr(e),"elapsed_seconds":round(time.time()-start,3)}
                results.append(row)
                with jsonl.open("a") as f: f.write(json.dumps(row,ensure_ascii=False)+"\n")
    tested=[r for r in results if r["status"]=="ok" and r.get("grading",{}).get("passed") is not None]
    passed=sum(r["grading"]["passed"] for r in tested)
    summary={"timestamp_utc":stamp,"base_url":args.base_url,"suites":suites,"repeat":args.repeat,
             "total":len(results),"successful_requests":sum(r["status"]=="ok" for r in results),
             "failed_requests":sum(r["status"]!="ok" for r in results),
             "graded":len(tested),"passed":passed,"failed":len(tested)-passed,
             "pass_rate":round(passed/len(tested),4) if tested else None,
             "mean_latency_seconds":round(statistics.mean([r["elapsed_seconds"] for r in results if r["status"]=="ok"]),3) if any(r["status"]=="ok" for r in results) else None,
             "results_file":jsonl.name}
    (OUT/f"summary_{stamp}.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    if summary["failed_requests"] or (tested and summary["failed"]):
        raise SystemExit(1)
if __name__=="__main__": main()

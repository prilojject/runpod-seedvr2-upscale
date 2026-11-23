#!/usr/bin/env python3
"""Paid GPU smoke for runpod-seedvr2-upscale (RTX 4090, budget <= $5, max workers 1).

Usage:
    python3 scripts/smoke_runpod.py setup <image> <gpu_ids> <tag>
    python3 scripts/smoke_runpod.py run <endpoint> <jobs.json> <out_prefix>
    python3 scripts/smoke_runpod.py cleanup

This script does not print secrets. It does not publish a GitHub repo or a Hub
listing. After the test it deletes the endpoint, template and registry auth it
created. It must never touch protected pods 92anyumh6q7jrf / oejzh2afh8hcym or
volumes 13faqbphuo / fv2c638zex.

Until this repository is public, pass an image_url that is already reachable
(e.g. the NASA public-domain source documented in examples/SOURCES.md). The
Hub tests.json GitHub raw URL will work after Coder publishes the repo.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

PROTECTED_PODS = {"92anyumh6q7jrf", "oejzh2afh8hcym"}
PROTECTED_VOLUMES = {"13faqbphuo", "fv2c638zex"}
STATE = os.environ.get("SEEDVR2_SMOKE_STATE", "/workspace/rp_seedvr2_state.json")
SPEND_CANCEL_USD = 3.5
SPEND_HARD_USD = 5.0
QUEUE_TIMEOUT_S = 20 * 60
HANG_TIMEOUT_S = 40 * 60

# Do not embed tokens. RUNPOD_API_KEY must already be in the environment.
def _gql(q, v=None):
    key = os.environ.get("RUNPOD_API_KEY") or ""
    if not key:
        raise SystemExit("RUNPOD_API_KEY is not set")
    req = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=json.dumps({"query": q, "variables": v or {}}).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + key,
            "User-Agent": "rp-seedvr2-smoke/1.0",
        },
    )
    try:
        return json.load(urllib.request.urlopen(req, timeout=60))
    except urllib.error.HTTPError as e:
        return {"http_error": e.code, "body": e.read().decode()[:2000]}


def load():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}


def save(s):
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(s, f, indent=1)
    os.replace(tmp, STATE)


H = None


def _headers():
    global H
    if H is None:
        key = os.environ.get("RUNPOD_API_KEY") or ""
        if not key:
            raise SystemExit("RUNPOD_API_KEY is not set")
        H = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + key,
            "User-Agent": "rp-seedvr2-smoke/1.0",
        }
    return H


def call(ep, method, path, body=None):
    req = urllib.request.Request(
        f"https://api.runpod.ai/v2/{ep}/{path}",
        data=json.dumps(body).encode() if body else None,
        headers=_headers(), method=method,
    )
    try:
        return json.load(urllib.request.urlopen(req, timeout=60))
    except urllib.error.HTTPError as e:
        return {"http_error": e.code, "body": e.read().decode()[:500]}
    except Exception as e:
        return {"exc": str(e)}


def balance():
    return _gql("{ myself { clientBalance currentSpendPerHr } }")["data"]["myself"]


def setup(image, gpu_ids, tag):
    s = load()
    if not s.get("registryAuthId"):
        tok = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True).stdout.strip()
        r = _gql(
            "mutation($i: SaveRegistryAuthInput!){ saveRegistryAuth(input:$i){ id } }",
            {"i": {"name": f"ghcr-prilojject-seedvr2-smoke-{tag}", "username": "prilojject", "password": tok}},
        )
        del tok
        if not r.get("data"):
            raise SystemExit(f"saveRegistryAuth failed: {r}")
        s["registryAuthId"] = r["data"]["saveRegistryAuth"]["id"]
        save(s)
    if "balance_before" not in s:
        s["balance_before"] = balance()["clientBalance"]
        save(s)
    name = f"seedvr2-smoke-{tag}"
    r = _gql(
        "mutation($i: SaveTemplateInput!){ saveTemplate(input:$i){ id name } }",
        {"i": {
            "name": name, "imageName": image, "dockerArgs": "",
            "containerDiskInGb": 80, "volumeInGb": 0, "isServerless": True,
            "env": [{"key": "INFER_TIMEOUT_SECONDS", "value": "1200"}],
            "containerRegistryAuthId": s["registryAuthId"], "readme": "",
        }},
    )
    print("template", r, flush=True)
    if not r.get("data"):
        raise SystemExit(f"saveTemplate failed: {r}")
    tid = r["data"]["saveTemplate"]["id"]
    s.setdefault("templates", []).append(name)
    s.setdefault("template_ids", []).append(tid)
    save(s)
    r = _gql(
        "mutation($i: EndpointInput!){ saveEndpoint(input:$i){ id name gpuIds executionTimeoutMs } }",
        {"i": {
            "name": f"seedvr2-smoke-{tag}", "templateId": tid, "gpuIds": gpu_ids,
            "gpuCount": 1, "idleTimeout": 5, "workersMin": 0, "workersMax": 1,
            "scalerType": "QUEUE_DELAY", "scalerValue": 4,
            "executionTimeoutMs": 1800000, "flashBootType": "OFF",
        }},
    )
    print("endpoint", r, flush=True)
    if not r.get("data"):
        raise SystemExit(f"saveEndpoint failed: {r}")
    s.setdefault("endpoints", []).append(r["data"]["saveEndpoint"]["id"])
    save(s)
    return r["data"]["saveEndpoint"]["id"]


def run(ep, jobs_file, out_prefix):
    s = load()
    bal0 = s["balance_before"]
    jobs = json.load(open(jobs_file))
    ok = True
    for idx, body in enumerate(jobs):
        t0 = time.time()
        r = call(ep, "POST", "run", body)
        print(time.strftime("%H:%M:%S"), f"job{idx} submit", r, flush=True)
        jid = r.get("id")
        if not jid:
            return False
        t_start = None
        last_status = None
        while True:
            time.sleep(20)
            st = call(ep, "GET", f"status/{jid}")
            bal = balance()
            spent = bal0 - bal["clientBalance"]
            el = time.time() - t0
            status = st.get("status")
            if status == "IN_PROGRESS" and t_start is None:
                t_start = el
            if status != last_status:
                last_status = status
            short = {k: v for k, v in st.items() if k != "output"}
            print(time.strftime("%H:%M:%S"),
                  f"el={el:.0f}s spent=${spent:.3f} rate={bal['currentSpendPerHr']}",
                  json.dumps(short)[:300], flush=True)
            if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
                json.dump(st, open(f"{out_prefix}_job{idx}.json", "w"), indent=1)
                print("FINAL", idx, status, "in_queue_s", t_start, "total_s", round(el), flush=True)
                if status != "COMPLETED":
                    ok = False
                break
            reason = None
            if t_start is None and el > QUEUE_TIMEOUT_S:
                reason = "worker not started in 20 min"
            if el > HANG_TIMEOUT_S:
                reason = "hang > 40 min"
            if spent > SPEND_CANCEL_USD:
                reason = f"spend near ${SPEND_CANCEL_USD:.2f}"
            if spent > SPEND_HARD_USD:
                reason = f"spend exceeded hard cap ${SPEND_HARD_USD:.2f}"
            if reason:
                print("CANCEL:", reason, call(ep, "POST", f"cancel/{jid}"), flush=True)
                return False
    return ok


def cleanup():
    s = load()
    # Never delete protected resources, even if a state file is wrong.
    for ep in list(s.get("endpoints", [])):
        if ep in PROTECTED_PODS:
            print("REFUSING to delete protected id", ep)
            continue
        for _ in range(12):
            r = _gql('mutation{ deleteEndpoint(id:"%s") }' % ep)
            print("deleteEndpoint", ep, r)
            if not r.get("errors") and "http_error" not in r:
                break
            time.sleep(10)
    left = []
    for name in s.get("templates", []):
        r = _gql('mutation{ deleteTemplate(templateName:"%s") }' % name)
        print("deleteTemplate", name, str(r)[:200])
        if r.get("errors") or "http_error" in r:
            left.append(name)
    if s.get("registryAuthId"):
        r = _gql('mutation{ deleteRegistryAuth(registryAuthId:"%s") }' % s["registryAuthId"])
        print("deleteRegistryAuth", r)
    s["templates_pending"] = left
    try:
        s["balance_after"] = balance()["clientBalance"]
        s["spent_usd"] = s["balance_before"] - s["balance_after"]
        print(f"spend_usd={s['spent_usd']:.3f}")
    except Exception as exc:
        print("balance_after failed:", type(exc).__name__)
    save(s)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "setup":
        print(setup(sys.argv[2], sys.argv[3], sys.argv[4]))
    elif cmd == "run":
        sys.exit(0 if run(sys.argv[2], sys.argv[3], sys.argv[4]) else 1)
    elif cmd == "cleanup":
        cleanup()
    else:
        raise SystemExit("usage: smoke_runpod.py setup|run|cleanup ...")

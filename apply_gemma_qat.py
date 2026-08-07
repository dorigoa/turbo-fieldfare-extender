#!/usr/bin/env python3
"""Re-point TurboFieldfare's pinned model at the Gemma 4 26B-A4B IT **QAT** 4-bit checkpoint.

The upstream project pins exactly one source checkpoint
(`mlx-community/gemma-4-26b-a4b-it-4bit`) in three places: the installer's
`SupportedModelSource`, the Mac app's `AppModelInstallDescriptor`, and the test
that guards those constants. This script rewrites those constants to
`mlx-community/gemma-4-26B-A4B-it-qat-4bit` so both the CLI installer and the
in-app download fetch the QAT build instead.

Run it again after every `git pull` / fresh clone of the upstream repository.

    python3 Scripts/apply_gemma_qat.py                 # apply pinned QAT values
    python3 Scripts/apply_gemma_qat.py --check         # report only, change nothing
    python3 Scripts/apply_gemma_qat.py --refresh       # re-resolve from Hugging Face first
    python3 Scripts/apply_gemma_qat.py --revision SHA  # pin a specific QAT commit

The patch is written against *field names*, not against the old values, so it
still applies after the maintainer bumps the stock checkpoint. If upstream
restructures one of the files the script refuses to write anything and tells you
which pattern stopped matching.

No documentation (`*.md`) is touched — only compiled sources plus the test that
asserts the pinned values.

Why the QAT build works unchanged: it has the identical architecture and file
layout, and its only quantization difference is the shared-expert MLP at 8 bits
instead of 4. The runtime already reads that width from `manifest.json`
(`Model.sharedExpertWeightBits`), the manifest validator already accepts 4 or 8
for that slot, and `SharedExpertInt8` already exists. Nothing else to change.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import struct
import sys
import urllib.request

# --------------------------------------------------------------------------
# Pinned QAT source. Regenerate with --refresh.
# --------------------------------------------------------------------------

QAT = {
    "displayName": "Gemma 4 26B-A4B IT QAT 4-bit",
    "repoID": "mlx-community/gemma-4-26B-A4B-it-qat-4bit",
    "revision": "0e3cbab38ce568cf6e23543010d08d03b731910c",
    "sourceIndexSHA256": "5455e83705bbdd4e3702c7d4f9d49d4900e84533036628f74500538075dd5c80",
    "approximateDownloadBytes": 14_952_958_284,
    "installedBytes": 14_559_575_188,
}

QAT_REPO = "mlx-community/gemma-4-26B-A4B-it-qat-4bit"

# `installedBytes` is the size of the *finished* .gturbo directory, not just the
# planner's output files. Beyond model_weights.bin + packed_experts/*.bin it also
# holds the tokenizer sidecars (sizes fetched per revision) plus manifest.json,
# packed_experts/layout.json and the install receipt. layout.json dominates that
# last group: it describes 30 layers x 128 experts x 9 sub-tensors. The constant
# below is those three files as measured on the pinned upstream install
# (14,291,921,884 on-disk minus 14,251,255,868 planned minus 32,202,012 of
# sidecars); it is layout-shaped, so it carries over to any same-architecture
# revision.
JSON_METADATA_BYTES = 8_464_004

# Sidecars the repacker copies into <model>.gturbo/tokenizer/.
SIDECAR_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "chat_template.json",
)

# --------------------------------------------------------------------------
# Files to patch. Each rule: (regex with one capturing group for the value,
# replacement builder). `count` is the exact number of matches required.
# --------------------------------------------------------------------------


def q(value: str) -> str:
    return '"%s"' % value


def swift_int(value: int) -> str:
    text = str(value)
    out = []
    while len(text) > 3:
        out.insert(0, text[-3:])
        text = text[:-3]
    out.insert(0, text)
    return "_".join(out)


def rules(values: dict) -> dict:
    """file path -> list of (description, compiled regex, replacement)."""
    d = values

    def sub(pattern, replacement, flags=0):
        return (re.compile(pattern, flags), replacement)

    return {
        os.path.join("Sources", "TurboFieldfareRepack", "Core", "Remote",
                     "SupportedModelSource.swift"): [
            ("displayName",
             *sub(r'(?P<head>static let displayName\s*=\s*)"[^"]*"',
                  lambda m: m.group("head") + q(d["displayName"]))),
            ("repoID",
             *sub(r'(?P<head>static let repoID\s*=\s*)"[^"]*"',
                  lambda m: m.group("head") + q(d["repoID"]))),
            ("revision",
             *sub(r'(?P<head>static let revision\s*=\s*)"[^"]*"',
                  lambda m: m.group("head") + q(d["revision"]))),
            ("sourceIndexSHA256",
             *sub(r'(?P<head>static let sourceIndexSHA256\s*=\s*)"[^"]*"',
                  lambda m: m.group("head") + q(d["sourceIndexSHA256"]))),
            ("approximateDownloadBytes",
             *sub(r'(?P<head>static let approximateDownloadBytes\s*:\s*UInt64\s*=\s*)[0-9_]+',
                  lambda m: m.group("head") + swift_int(d["approximateDownloadBytes"]))),
            ("installedBytes",
             *sub(r'(?P<head>static let installedBytes\s*:\s*UInt64\s*=\s*)[0-9_]+',
                  lambda m: m.group("head") + swift_int(d["installedBytes"]))),
        ],
        os.path.join("Sources", "TurboFieldfareApp", "Core", "Installation",
                     "AppModelInstallDescriptor.swift"): [
            ("displayName:",
             *sub(r'(?P<head>displayName:\s*)"[^"]*"',
                  lambda m: m.group("head") + q(d["displayName"]))),
            ("repoID:",
             *sub(r'(?P<head>repoID:\s*)"[^"]*"',
                  lambda m: m.group("head") + q(d["repoID"]))),
            ("revision:",
             *sub(r'(?P<head>revision:\s*)"[^"]*"',
                  lambda m: m.group("head") + q(d["revision"]))),
            ("sourceIndexSHA256:",
             *sub(r'(?P<head>sourceIndexSHA256:\s*)"[^"]*"',
                  lambda m: m.group("head") + q(d["sourceIndexSHA256"]))),
            ("approximateDownloadBytes:",
             *sub(r'(?P<head>approximateDownloadBytes:\s*)[0-9_]+',
                  lambda m: m.group("head") + swift_int(d["approximateDownloadBytes"]))),
            ("installedBytes:",
             *sub(r'(?P<head>installedBytes:\s*)[0-9_]+',
                  lambda m: m.group("head") + swift_int(d["installedBytes"]))),
        ],
        os.path.join("Tests", "TurboFieldfareApp", "Core", "Installation",
                     "AppModelInstallTests.swift"): [
            ("expect displayName",
             *sub(r'(?P<head>descriptor\.displayName\s*==\s*)"[^"]*"',
                  lambda m: m.group("head") + q(d["displayName"]))),
            ("expect repoID",
             *sub(r'(?P<head>descriptor\.repoID\s*==\s*)"[^"]*"',
                  lambda m: m.group("head") + q(d["repoID"]))),
            ("expect revision",
             *sub(r'(?P<head>descriptor\.revision\s*==\s*)"[^"]*"',
                  lambda m: m.group("head") + q(d["revision"]))),
            ("expect sourceIndexSHA256",
             *sub(r'(?P<head>descriptor\.sourceIndexSHA256\s*==\s*)"[^"]*"',
                  lambda m: m.group("head") + q(d["sourceIndexSHA256"]))),
            ("expect approximateDownloadBytes",
             *sub(r'(?P<head>descriptor\.approximateDownloadBytes\s*==\s*)[0-9_]+',
                  lambda m: m.group("head") + swift_int(d["approximateDownloadBytes"]))),
            ("expect installedBytes",
             *sub(r'(?P<head>descriptor\.installedBytes\s*==\s*)[0-9_]+',
                  lambda m: m.group("head") + swift_int(d["installedBytes"]))),
            ("expect requiredFreeBytes",
             *sub(r'(?P<head>descriptor\.requiredFreeBytes\s*==\s*)[0-9_]+',
                  lambda m: m.group("head") + swift_int(d["requiredFreeBytes"]))),
        ],
    }


# --------------------------------------------------------------------------
# Hugging Face sizing (only used by --refresh / --revision)
# --------------------------------------------------------------------------

PAGE = 16384
RESIDENT_HEADER_BYTES = 24     # GTurboFormatV1.residentHeaderBytes
RESIDENT_ENTRY_BYTES = 72      # GTurboFormatV1.residentEntryBytes
RANGE_CHUNK = 64 * 1024 * 1024  # RemoteChunkPolicy.defaultBytes
MULTIMODAL_PREFIXES = ("vision_tower.", "embed_vision.", "audio_tower.")


def http_get(url: str, byte_range=None) -> bytes:
    request = urllib.request.Request(url)
    token = os.environ.get("HF_TOKEN")
    if token:
        request.add_header("Authorization", "Bearer " + token)
    if byte_range:
        request.add_header("Range", "bytes=%d-%d" % byte_range)
        request.add_header("Accept-Encoding", "identity")
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def round_up_page(value: int) -> int:
    return ((value + PAGE - 1) // PAGE) * PAGE


def resolve_revision(repo: str) -> str:
    info = json.loads(http_get("https://huggingface.co/api/models/%s" % repo))
    return info["sha"]


def sidecar_sizes(repo: str, revision: str) -> int:
    info = json.loads(
        http_get("https://huggingface.co/api/models/%s/revision/%s?blobs=true" % (repo, revision)))
    total = 0
    for sibling in info.get("siblings", []):
        if sibling["rfilename"] in SIDECAR_FILES and sibling.get("size"):
            total += sibling["size"]
    return total


def compute_source(repo: str, revision: str) -> dict:
    """Mirror RepackPlanner + RangeCopyPlanner to size the install for `repo@revision`."""
    base = "https://huggingface.co/%s/resolve/%s/" % (repo, revision)
    index_bytes = http_get(base + "model.safetensors.index.json")
    index = json.loads(index_bytes)
    config = json.loads(http_get(base + "config.json"))
    text_config = config["text_config"]
    num_layers = text_config["num_hidden_layers"]
    num_experts = text_config["num_experts"]

    registry = {}
    for shard in sorted(set(index["weight_map"].values())):
        header_len = struct.unpack("<Q", http_get(base + shard, (0, 7)))[0]
        header = json.loads(http_get(base + shard, (8, 8 + header_len - 1)))
        payload_base = 8 + header_len
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            begin, end = entry["data_offsets"]
            registry[name] = {
                "shard": shard,
                "dtype": entry["dtype"],
                "off": payload_base + begin,
                "size": end - begin,
            }

    resident_names = []
    routed = {}
    for name in registry:
        if name.endswith(".scales") or name.endswith(".biases"):
            continue
        if name.startswith("language_model."):
            role = None
            if ".experts.switch_glu." in name:
                for needle, value in ((".gate_proj.", "gate"), (".up_proj.", "up"),
                                      (".down_proj.", "down")):
                    if needle in name:
                        role = value
                        break
            if role is not None:
                layer = int(name.split(".layers.")[1].split(".")[0])
                if 0 <= layer < num_layers:
                    routed.setdefault(layer, {})[role] = name
                    continue
            resident_names.append(name)
        elif any(name.startswith(p) for p in MULTIMODAL_PREFIXES):
            continue
        else:
            raise SystemExit(
                "unrecognised tensor prefix %r — the repack planner would reject this repo" % name)

    string_table = sum(len(n.encode()) for n in resident_names)
    index_size = round_up_page(
        RESIDENT_HEADER_BYTES + len(resident_names) * RESIDENT_ENTRY_BYTES + string_table)

    copies = []
    payload = 0
    for name in resident_names:
        weight = registry[name]
        payload += weight["size"]
        copies.append((weight["shard"], weight["off"], weight["size"]))
        if weight["dtype"] == "U32" and name.endswith(".weight"):
            stem = name[: -len(".weight")]
            for suffix in (".scales", ".biases"):
                companion = registry[stem + suffix]
                payload += companion["size"]
                copies.append((companion["shard"], companion["off"], companion["size"]))

    layer_bytes = 0
    for layer in range(num_layers):
        bundle = routed.get(layer)
        if not bundle:
            continue
        blob = 0
        slices = []
        for role in ("gate", "up", "down"):
            name = bundle[role]
            stem = name[: -len(".weight")] if name.endswith(".weight") else name
            for tensor in (registry[name], registry[stem + ".scales"], registry[stem + ".biases"]):
                per_expert = tensor["size"] // num_experts
                if per_expert * num_experts != tensor["size"]:
                    raise SystemExit("tensor %s is not divisible across %d experts"
                                     % (name, num_experts))
                blob += per_expert
                slices.append((tensor, per_expert))
        stride = round_up_page(blob)
        layer_bytes += num_experts * stride
        for expert in range(num_experts):
            for tensor, per_expert in slices:
                copies.append((tensor["shard"], tensor["off"] + expert * per_expert, per_expert))

    # splitLargeCopies + coalesce, exactly as RangeCopyPlanner does.
    split = []
    for shard, offset, size in copies:
        remaining, cursor = size, offset
        while remaining > 0:
            take = min(remaining, RANGE_CHUNK)
            split.append((shard, cursor, take))
            remaining -= take
            cursor += take
    split.sort(key=lambda c: (c[0], c[1]))

    download = 0
    shard = start = end = None
    for candidate_shard, offset, size in split:
        if size == 0:
            continue
        candidate_end = offset + size
        if shard is None:
            shard, start, end = candidate_shard, offset, candidate_end
            continue
        merged_end = max(end, candidate_end)
        if candidate_shard == shard and merged_end - start <= RANGE_CHUNK:
            end = merged_end
        else:
            download += end - start
            shard, start, end = candidate_shard, offset, candidate_end
    if shard is not None:
        download += end - start

    planned = index_size + payload + layer_bytes
    installed = planned + sidecar_sizes(repo, revision) + JSON_METADATA_BYTES
    return {
        "displayName": QAT["displayName"],
        "repoID": repo,
        "revision": revision,
        "sourceIndexSHA256": hashlib.sha256(index_bytes).hexdigest(),
        "approximateDownloadBytes": download,
        "installedBytes": installed,
    }


# --------------------------------------------------------------------------
# Patching
# --------------------------------------------------------------------------


def parse_swift_int(text: str) -> int:
    node = ast.parse(text.replace("_", ""), mode="eval").body
    def evaluate(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            return n.value
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Mult, ast.Add)):
            left, right = evaluate(n.left), evaluate(n.right)
            return left * right if isinstance(n.op, ast.Mult) else left + right
        raise ValueError(text)
    return evaluate(node)


def read_constant(repo_path: str, relative: str, pattern: str, fallback: int) -> int:
    path = os.path.join(repo_path, relative)
    try:
        with open(path, encoding="utf-8") as handle:
            match = re.search(pattern, handle.read())
        if match:
            return parse_swift_int(match.group(1))
    except (OSError, ValueError):
        pass
    print("  ! could not read %s from %s; assuming %d" % (pattern, relative, fallback))
    return fallback


def required_free_bytes(repo_path: str, installed: int) -> int:
    """AppModelInstallDescriptor.requiredFreeBytes = installed + staging + reserve."""
    staging = read_constant(
        repo_path,
        os.path.join("Sources", "TurboFieldfareRepack", "Core", "Remote",
                     "RemoteChunkPolicy.swift"),
        r"defaultBytes\s*=\s*([0-9_ ()*+]+)",
        64 * 1024 * 1024)
    reserve = read_constant(
        repo_path,
        os.path.join("Sources", "TurboFieldfareApp", "Core", "Installation",
                     "AppModelInstallDescriptor.swift"),
        r"reserveBytes:\s*([0-9_]+)",
        1024 * 1024 * 1024)
    return installed + staging + reserve


def apply(repo_path: str, values: dict, check_only: bool) -> int:
    plan = rules(values)
    edits = []
    problems = []

    for relative, file_rules in plan.items():
        path = os.path.join(repo_path, relative)
        if not os.path.exists(path):
            problems.append("missing file: %s" % relative)
            continue
        with open(path, encoding="utf-8") as handle:
            original = handle.read()
        updated = original
        for description, pattern, replacement in file_rules:
            updated, count = pattern.subn(replacement, updated)
            if count != 1:
                problems.append("%s: expected 1 match for %s, found %d"
                                % (relative, description, count))
        if updated != original:
            edits.append((path, relative, updated))

    if problems:
        print("\nRefusing to patch — upstream layout changed:")
        for problem in problems:
            print("  - %s" % problem)
        print("\nNothing was written. Update the patterns in this script, or edit the")
        print("constants by hand (repoID / revision / sourceIndexSHA256 /")
        print("approximateDownloadBytes / installedBytes).")
        return 1

    if not edits:
        print("\nAlready pinned to the QAT checkpoint — nothing to do.")
        return 0

    for _, relative, _ in edits:
        print("  %s %s" % ("would patch" if check_only else "patched", relative))

    if check_only:
        print("\n--check: no files written.")
        return 0

    for path, _, updated in edits:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(updated)

    print("\nDone. Rebuild and re-verify:")
    print("    swift build -c release")
    print("    Scripts/test.sh --filter AppModelInstall")
    print("\nThe existing .gturbo install is the old checkpoint — reinstall it:")
    print("    swift run -c release TurboFieldfareRepack \\")
    print("      --output scratch/gemma4.gturbo --overwrite")
    print("(or use the in-app download, which now points at the QAT repo).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-pin TurboFieldfare to the Gemma 4 26B-A4B IT QAT 4-bit checkpoint.")
    parser.add_argument("--repo-path", default=".",
                        help="path to the turbo-fieldfare checkout (default: current directory)")
    parser.add_argument("--check", action="store_true",
                        help="report what would change and exit without writing")
    parser.add_argument("--refresh", action="store_true",
                        help="resolve the QAT repo's current commit on Hugging Face and "
                             "recompute the pinned hash and byte counts")
    parser.add_argument("--revision",
                        help="pin this QAT commit instead (implies --refresh's recomputation)")
    args = parser.parse_args()

    repo_path = os.path.abspath(args.repo_path)
    marker = os.path.join(repo_path, "Sources", "TurboFieldfareRepack", "Core", "Remote",
                          "SupportedModelSource.swift")
    if not os.path.exists(marker):
        print("error: %s does not look like a turbo-fieldfare checkout "
              "(no %s)" % (repo_path, os.path.relpath(marker, repo_path)), file=sys.stderr)
        return 2

    if args.refresh or args.revision:
        revision = args.revision or resolve_revision(QAT_REPO)
        print("Resolving %s@%s from Hugging Face ..." % (QAT_REPO, revision[:12]))
        values = compute_source(QAT_REPO, revision)
        print("  index sha256              %s" % values["sourceIndexSHA256"])
        print("  approximateDownloadBytes  %d" % values["approximateDownloadBytes"])
        print("  installedBytes            %d" % values["installedBytes"])
        if values != {k: QAT[k] for k in values}:
            print("\n  (differs from the values pinned in this script — using the fresh ones.")
            print("   Paste them into QAT at the top of this file to make them the default.)")
    else:
        values = dict(QAT)

    values["requiredFreeBytes"] = required_free_bytes(repo_path, values["installedBytes"])

    print("\nPinning %s@%s" % (values["repoID"], values["revision"][:12]))
    return apply(repo_path, values, args.check)


if __name__ == "__main__":
    sys.exit(main())

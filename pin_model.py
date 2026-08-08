#!/usr/bin/env python3
"""Re-point TurboFieldfare's pinned model at a different MLX checkpoint.

The upstream project pins exactly one source checkpoint
(`mlx-community/gemma-4-26b-a4b-it-4bit`) in three places: the installer's
`SupportedModelSource`, the Mac app's `AppModelInstallDescriptor`, and the test
that guards those constants. This script rewrites those constants so both the
CLI installer and the in-app download fetch the checkpoint you selected.

Run it again after every `git pull` / fresh clone of the upstream repository.

    python3 Scripts/pin_model.py                       # re-apply the current selection
    python3 Scripts/pin_model.py --model ORG/REPO      # switch model, then remember it
    python3 Scripts/pin_model.py --check               # report only, change nothing
    python3 Scripts/pin_model.py --refresh             # re-resolve the selection from HF
    python3 Scripts/pin_model.py --revision SHA        # pin a specific commit
    python3 Scripts/pin_model.py --list                # show what is pinned

Switching model needs the network exactly once. `--model` resolves the repo's
head commit, sizes the install, patches the checkout and records everything in
`model_pins.json` next to this script, which also becomes the new default
selection. Later runs with no arguments — including the one in `makeall.sh` —
replay that pin offline. Commit `model_pins.json` alongside this file.

Compatibility: the runtime hardwires 4-bit kernels for the embedding table, the
attention projections and the routed experts (`EmbedLookupInt4`,
`DequantInt4GEMV` in `RealForwardRunner`); only the shared expert switches on a
manifest field (`Model.sharedExpertWeightBits`, 4 or 8, backed by
`SharedExpertInt4`/`SharedExpertInt8`). The manifest carries one width per
category — embedding, attention, router, sharedExpert, routedExpert — so a
checkpoint that varies width *per layer* cannot be described at all: the
repacker samples one tensor per category (`RemoteStreamingRepacker.writeManifest`)
and `Model.requireAffine` then rejects every layer that disagrees. The QAT build
fits because its only deviation is the shared expert plus router at 8 bits,
uniformly. Anything else is refused here unless you pass `--force`.

No documentation (`*.md`) is touched — only compiled sources plus the test that
asserts the pinned values.
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
import urllib.error
import urllib.request

# --------------------------------------------------------------------------
# Pin storage. `model_pins.json` is written next to this script; the built-in
# entry below is the fallback when that file does not exist yet.
# --------------------------------------------------------------------------

STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_pins.json")

DEFAULT_REPO = "mlx-community/gemma-4-26B-A4B-it-qat-4bit"

PIN_FIELDS = ("displayName", "repoID", "revision", "sourceIndexSHA256",
              "approximateDownloadBytes", "installedBytes")

BUILTIN_PINS = {
    DEFAULT_REPO: {
        "displayName": "Gemma 4 26B-A4B IT QAT 4-bit",
        "repoID": DEFAULT_REPO,
        "revision": "0e3cbab38ce568cf6e23543010d08d03b731910c",
        "sourceIndexSHA256": "5455e83705bbdd4e3702c7d4f9d49d4900e84533036628f74500538075dd5c80",
        "approximateDownloadBytes": 14_952_958_284,
        "installedBytes": 14_559_575_188,
    },
}

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
CALIBRATED_SHAPE = (30, 128)  # (num_hidden_layers, num_experts) JSON_METADATA_BYTES was measured on

# Sidecars the repacker copies into <model>.gturbo/tokenizer/.
SIDECAR_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "chat_template.json",
)


def load_store() -> dict:
    try:
        with open(STORE_PATH, encoding="utf-8") as handle:
            store = json.load(handle)
    except (OSError, ValueError):
        store = {}
    store.setdefault("selected", None)
    store.setdefault("pins", {})
    return store


def save_store(store: dict) -> None:
    with open(STORE_PATH, "w", encoding="utf-8") as handle:
        json.dump(store, handle, indent=2, sort_keys=True)
        handle.write("\n")


def known_pin(store: dict, repo: str) -> dict | None:
    return store["pins"].get(repo) or BUILTIN_PINS.get(repo)


# --------------------------------------------------------------------------
# Display name derivation. `--display-name` overrides it.
# --------------------------------------------------------------------------

ACRONYMS = {"it", "qat", "sft", "dpo", "rl", "mlx", "gguf", "awq", "gptq", "moe", "pt"}


def is_size_token(token: str) -> bool:
    """True for spec fragments like 26B or A4B, which read as one unit: 26B-A4B."""
    return (len(token) >= 2 and token == token.upper()
            and any(c.isdigit() for c in token) and any(c.isalpha() for c in token))


def derive_display_name(repo: str) -> str:
    """mlx-community/gemma-4-26B-A4B-it-qat-4bit -> Gemma 4 26B-A4B IT QAT 4-bit"""
    words = []
    for token in repo.split("/")[-1].split("-"):
        bits = re.fullmatch(r"(\d+)bit", token.lower())
        if token.lower() in ACRONYMS:
            words.append(token.upper())
        elif bits:
            words.append(bits.group(1) + "-bit")
        elif token.islower():
            words.append(token.capitalize())
        else:
            words.append(token)

    merged = words[:1]
    for word in words[1:]:
        if is_size_token(word) and is_size_token(merged[-1]):
            merged[-1] += "-" + word
        else:
            merged.append(word)
    return " ".join(merged)


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
# Hugging Face sizing (only used when a pin has to be resolved)
# --------------------------------------------------------------------------

PAGE = 16384
RESIDENT_HEADER_BYTES = 24     # GTurboFormatV1.residentHeaderBytes
RESIDENT_ENTRY_BYTES = 72      # GTurboFormatV1.residentEntryBytes
RANGE_CHUNK = 64 * 1024 * 1024  # RemoteChunkPolicy.defaultBytes
MULTIMODAL_PREFIXES = ("vision_tower.", "embed_vision.", "audio_tower.")

# The width the runtime can actually decode outside the shared expert:
# EmbedLookupInt4 and DequantInt4GEMV are 4-bit only, and every quant kernel
# asserts Quantization.groupSize, which is 64.
RUNTIME_BASE_BITS = 4
RUNTIME_GROUP_SIZE = 64

# Per-tensor quantization overrides the repack format can already express: the
# shared-expert MLP and the router, which is how the QAT build differs from stock.
MIXED_PRECISION_ALLOWED = (".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj", ".router.proj")


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


def quantization_problems(config: dict) -> list:
    """Bit widths this build cannot run. See the compatibility note up top.

    Two things have to hold. The checkpoint's global width must be 4, because
    embedding, attention and routed experts go through 4-bit-only kernels, and
    `Model.affineSizes` accepts nothing but 4 or 8 per slot anyway. And the only
    tensors allowed to deviate from it are the shared-expert MLP and the router,
    uniformly across layers — that is the one shape the manifest can describe
    and the runtime can decode.
    """
    quant = config.get("quantization")
    if not isinstance(quant, dict):
        return ["config.json has no `quantization` block"]
    global_bits = quant.get("bits")
    global_group = quant.get("group_size")
    if global_bits is None:
        return ["config.json quantization block has no global `bits`"]

    problems = []
    if global_bits != RUNTIME_BASE_BITS:
        problems.append(
            "global `bits` is %s: embedding, attention and routed experts go through "
            "%d-bit-only kernels" % (global_bits, RUNTIME_BASE_BITS))
    if global_group != RUNTIME_GROUP_SIZE:
        problems.append(
            "global `group_size` is %s: every quant kernel asserts Quantization.groupSize (%d)"
            % (global_group, RUNTIME_GROUP_SIZE))

    for name, entry in sorted(quant.items()):
        if not isinstance(entry, dict):
            continue
        bits = entry.get("bits", global_bits)
        group = entry.get("group_size", global_group)
        if group != global_group:
            problems.append("%s: group_size %s, global is %s" % (name, group, global_group))
        if bits == global_bits:
            continue
        if name.endswith(MIXED_PRECISION_ALLOWED) and bits in (4, 8):
            continue
        problems.append("%s: %s-bit, global is %s-bit" % (name, bits, global_bits))
    return problems


def compute_source(repo: str, revision: str, display_name: str, force: bool = False) -> dict:
    """Mirror RepackPlanner + RangeCopyPlanner to size the install for `repo@revision`."""
    base = "https://huggingface.co/%s/resolve/%s/" % (repo, revision)
    index_bytes = http_get(base + "model.safetensors.index.json")
    index = json.loads(index_bytes)
    config = json.loads(http_get(base + "config.json"))
    text_config = config["text_config"]
    num_layers = text_config["num_hidden_layers"]
    num_experts = text_config["num_experts"]

    problems = quantization_problems(config)
    if problems:
        print("\n  ! %s uses bit widths this build cannot run:" % repo)
        for problem in problems[:10]:
            print("      %s" % problem)
        if len(problems) > 10:
            print("      ... and %d more" % (len(problems) - 10))
        print("    This build runs a %d-bit checkpoint whose shared-expert MLP and router"
              % RUNTIME_BASE_BITS)
        print("    may be 8-bit, and nothing else. The repack itself would succeed —")
        print("    expect several GB downloaded, then ModelError.indexCorrupt at load")
        print("    (unsupported affine quantization, or an affine metadata mismatch),")
        print("    or wrong output wherever the byte sizes happen to line up.")
        if not force:
            raise SystemExit("\nRefusing to pin %s. Re-run with --force to pin it anyway." % repo)
        print("    --force given: pinning anyway.\n")

    if (num_layers, num_experts) != CALIBRATED_SHAPE:
        print("  ! %d layers x %d experts; JSON_METADATA_BYTES is calibrated for %d x %d,"
              " so installedBytes may be off by a few MB"
              % (num_layers, num_experts, *CALIBRATED_SHAPE))

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
        "displayName": display_name,
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
        print("\nAlready pinned to this checkpoint — nothing to do.")
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
    print("(or use the in-app download, which now points at the new repo).")
    return 0


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def resolve_values(args, store: dict) -> tuple:
    """-> (values, origin). Hits the network only when there is no usable pin."""
    repo = args.model or store["selected"] or DEFAULT_REPO
    pin = known_pin(store, repo)
    display = (args.display_name
               or (pin or {}).get("displayName")
               or derive_display_name(repo))

    if pin and not (args.refresh or args.revision):
        values = dict(pin)
        origin = "model_pins.json" if repo in store["pins"] else "built-in default"
    else:
        try:
            revision = args.revision or resolve_revision(repo)
            print("Resolving %s@%s from Hugging Face ..." % (repo, revision))
            values = compute_source(repo, revision, display, force=args.force)
        except urllib.error.HTTPError as error:
            hint = " (private or gated? export HF_TOKEN)" if error.code in (401, 403) else ""
            raise SystemExit("error: Hugging Face returned %d %s for %s%s"
                             % (error.code, error.reason, error.url, hint))
        except urllib.error.URLError as error:
            raise SystemExit("error: cannot reach Hugging Face (%s). A cached pin is used "
                             "offline, but %s has none yet." % (error.reason, repo))
        origin = "Hugging Face"

    values["displayName"] = display
    values["repoID"] = repo
    return values, origin


def show_pins(store: dict) -> int:
    selected = store["selected"] or DEFAULT_REPO
    repos = sorted(set(store["pins"]) | set(BUILTIN_PINS))
    print("Pinned models (%s):" % STORE_PATH)
    for repo in repos:
        pin = known_pin(store, repo)
        print(" %s %s@%s" % ("*" if repo == selected else " ", repo, pin["revision"]))
        print("     %s — %.2f GB installed, %.2f GB downloaded%s"
              % (pin["displayName"],
                 pin["installedBytes"] / 1e9,
                 pin["approximateDownloadBytes"] / 1e9,
                 "" if repo in store["pins"] else "  (built-in)"))
    print("\n* = current selection, used when --model is omitted.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-pin TurboFieldfare to a different MLX checkpoint.")
    parser.add_argument("--repo-path", default=".",
                        help="path to the turbo-fieldfare checkout (default: current directory)")
    parser.add_argument("--model", metavar="ORG/REPO",
                        help="Hugging Face repo to pin. Resolved and sized on first use, then "
                             "cached in model_pins.json and used as the default selection.")
    parser.add_argument("--display-name",
                        help="override the display name (default: derived from the repo id)")
    parser.add_argument("--check", action="store_true",
                        help="report what would change and exit without writing")
    parser.add_argument("--refresh", action="store_true",
                        help="ignore the cached pin and re-resolve from Hugging Face")
    parser.add_argument("--revision",
                        help="pin this commit instead of the repo's current head")
    parser.add_argument("--force", action="store_true",
                        help="pin even if the checkpoint's quantization layout is unsupported")
    parser.add_argument("--list", action="store_true",
                        help="show the pinned models and exit")
    args = parser.parse_args()

    store = load_store()
    if args.list:
        return show_pins(store)

    repo_path = os.path.abspath(args.repo_path)
    marker = os.path.join(repo_path, "Sources", "TurboFieldfareRepack", "Core", "Remote",
                          "SupportedModelSource.swift")
    if not os.path.exists(marker):
        print("error: %s does not look like a turbo-fieldfare checkout "
              "(no %s)" % (repo_path, os.path.relpath(marker, repo_path)), file=sys.stderr)
        return 2

    values, origin = resolve_values(args, store)

    print("\nPinning %s@%s  (from %s)" % (values["repoID"], values["revision"], origin))
    print("  displayName               %s" % values["displayName"])
    print("  sourceIndexSHA256         %s" % values["sourceIndexSHA256"])
    print("  approximateDownloadBytes  %d" % values["approximateDownloadBytes"])
    print("  installedBytes            %d" % values["installedBytes"])

    values["requiredFreeBytes"] = required_free_bytes(repo_path, values["installedBytes"])
    status = apply(repo_path, values, args.check)

    if status == 0 and not args.check:
        store["pins"][values["repoID"]] = {field: values[field] for field in PIN_FIELDS}
        store["selected"] = values["repoID"]
        save_store(store)

    return status


if __name__ == "__main__":
    sys.exit(main())

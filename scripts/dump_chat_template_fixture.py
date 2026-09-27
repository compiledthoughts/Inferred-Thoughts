#!/usr/bin/env python3
"""Render a model's own chat template the way Hugging Face `transformers` does,
for a fixed set of conversations, and write the results as a test fixture.

`src/tok/chat.rs` renders the same template with minijinja; `tests/chat_template.rs`
requires its output to match these renders byte for byte. This script is the
reference, so it copies `transformers`' Jinja setup (utils/chat_template_utils.py):
an immutable sandbox with trim_blocks, lstrip_blocks and loop controls,
`raise_exception`, and `tojson` as `json.dumps(ensure_ascii=False)`.

The template is read from the GGUF through `inferred inspect --json`, which
prints the file's metadata verbatim.

    python scripts/dump_chat_template_fixture.py --model <gguf> [--inferred <binary>]

Needs `jinja2`. Writes tests/fixtures/chat_template/<model file stem>.json.
"""

import argparse
import json
import pathlib
import subprocess

import jinja2
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List the files in a directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Directory to list."}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file — UTF-8, e.g. résumé.txt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_lines": {"type": "integer", "description": "Stop after this many lines."},
                },
                "required": ["path"],
            },
        },
    },
]

CALL_LIST = {"id": "call_0", "type": "function",
             "function": {"name": "list_files", "arguments": {"path": "k:\\compiledthoughts"}}}
CALL_READ = {"id": "call_1", "type": "function",
             "function": {"name": "read_file", "arguments": {"path": "README.md", "max_lines": 20}}}

CASES = [
    ("user", None, [
        {"role": "user", "content": "Hello"},
    ]),
    ("system_user", None, [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "What is 2+2?"},
    ]),
    ("multi_turn", None, [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": "Tell me a joke."},
    ]),
    ("reasoning_in_history", None, [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!", "reasoning_content": "The user greets me."},
        {"role": "user", "content": "Again?"},
    ]),
    ("tools_user", TOOLS, [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "List the files in k:\\compiledthoughts."},
    ]),
    ("tool_call_and_result", TOOLS, [
        {"role": "user", "content": "List the files in k:\\compiledthoughts."},
        {"role": "assistant", "content": "I'll list them.", "tool_calls": [CALL_LIST]},
        {"role": "tool", "tool_call_id": "call_0", "content": "README.md\nsrc/"},
    ]),
    ("two_calls_two_results", TOOLS, [
        {"role": "user", "content": "What is in the README?"},
        {"role": "assistant", "content": "", "tool_calls": [CALL_LIST, CALL_READ]},
        {"role": "tool", "tool_call_id": "call_0", "content": "README.md\nsrc/"},
        {"role": "tool", "tool_call_id": "call_1", "content": "# inferredThoughts\n\"quoted\" text"},
    ]),
    ("no_user_query", TOOLS, [
        {"role": "system", "content": "Only a system message."},
    ]),
    ("thinking_off", None, [
        {"role": "user", "content": "What is 2+2?"},
    ], {"enable_thinking": False}),
    ("reasoning_low", None, [
        {"role": "user", "content": "What is 2+2?"},
    ], {"reasoning_effort": "low"}),
]


def render(template: str, messages, tools, kwargs=None):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    context = dict(kwargs or {})
    context.update({"messages": messages, "add_generation_prompt": True})
    if tools is not None:
        context["tools"] = tools
    return env.from_string(template).render(**context)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="path to the .gguf")
    ap.add_argument("--inferred", default="inferred", help="the inferred binary, for reading the template")
    ap.add_argument("--out", default="tests/fixtures/chat_template")
    a = ap.parse_args()

    run = subprocess.run([a.inferred, "inspect", a.model, "--json"], capture_output=True, text=True, check=True)
    meta = json.loads(run.stdout or run.stderr)
    template = meta["kv"]["tokenizer.chat_template"]["value"]

    cases = []
    for name, tools, messages, *rest in CASES:
        kwargs = rest[0] if rest else None
        case = {"name": name, "tools": tools, "messages": messages, "kwargs": kwargs, "expected": None, "error": None}
        try:
            case["expected"] = render(template, messages, tools, kwargs)
        except jinja2.exceptions.TemplateError as e:
            case["error"] = str(e)
        cases.append(case)

    stem = pathlib.Path(a.model).stem
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{stem}.json"
    path.write_text(json.dumps({"model": pathlib.Path(a.model).name, "jinja2": jinja2.__version__, "cases": cases},
                               ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{path}: {len(cases)} cases, {sum(c['error'] is not None for c in cases)} expected errors")


if __name__ == "__main__":
    main()

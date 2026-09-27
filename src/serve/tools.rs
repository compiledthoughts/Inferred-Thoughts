//! Tool calls: reading the model's own call format back into OpenAI's shape.
//!
//! The model writes a call between its `<tool_call>` and `</tool_call>` tokens,
//! in whichever format its chat template taught it:
//!
//! - **XML** (Qwen3.5, Qwen3.6, Qwen3.8): `<function=NAME>` with one
//!   `<parameter=KEY>` … `</parameter>` per argument, closed by `</function>`.
//! - **JSON** (Qwen3): `{"name": NAME, "arguments": {...}}`.
//!
//! The format is recognized from the text itself, so each model's template
//! decides it and nothing is configured per model.
//!
//! **Values follow llama.cpp** (`common/chat-peg-parser.cpp`, the tagged-format
//! mapper): one leading and one trailing newline trimmed; a parameter whose
//! schema says `"type": "string"` stays text; anything else is parsed as JSON
//! when it is valid JSON — a number, a boolean, an object — and kept as text
//! when it is not.

use serde_json::{Map, Value};

/// One call the model made.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    /// Always a JSON object; OpenAI sends it on to the client as a string.
    pub arguments: Map<String, Value>,
}

impl ToolCall {
    /// OpenAI's `tool_calls[i]` entry: `arguments` serialized to a string.
    pub fn to_openai(&self, id: &str) -> Value {
        serde_json::json!({
            "id": id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": Value::Object(self.arguments.clone()).to_string(),
            },
        })
    }
}

/// Parse the text between one `<tool_call>` and its `</tool_call>`.
///
/// `tools` is the request's `tools` array, consulted for parameter types.
/// `None` when the text is not a call in either format; the caller then shows
/// it to the client as ordinary text rather than dropping it.
pub fn parse(text: &str, tools: Option<&Value>) -> Option<ToolCall> {
    let body = text.trim();
    if body.starts_with('{') {
        parse_json(body)
    } else if body.contains("<function=") {
        parse_xml(body, tools)
    } else {
        None
    }
}

fn parse_json(body: &str) -> Option<ToolCall> {
    let v: Value = serde_json::from_str(body).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    let arguments = match v.get("arguments") {
        Some(Value::Object(m)) => m.clone(),
        // Some models write the arguments as a JSON string.
        Some(Value::String(s)) => match serde_json::from_str(s) {
            Ok(Value::Object(m)) => m,
            _ => return None,
        },
        None | Some(Value::Null) => Map::new(),
        Some(_) => return None,
    };
    Some(ToolCall { name, arguments })
}

fn parse_xml(body: &str, tools: Option<&Value>) -> Option<ToolCall> {
    let rest = &body[body.find("<function=")? + "<function=".len()..];
    let close = rest.find('>')?;
    let name = rest[..close].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let mut rest = &rest[close + 1..];
    if let Some(end) = rest.find("</function>") {
        rest = &rest[..end];
    }

    let mut arguments = Map::new();
    while let Some(open) = rest.find("<parameter=") {
        rest = &rest[open + "<parameter=".len()..];
        let close = rest.find('>')?;
        let key = rest[..close].trim().trim_matches(|c| c == '"' || c == '\'').to_string();
        rest = &rest[close + 1..];
        // A missing closer ends the value at the next parameter, or the end.
        let end = rest
            .find("</parameter>")
            .or_else(|| rest.find("<parameter="))
            .unwrap_or(rest.len());
        let raw = &rest[..end];
        let raw = raw.strip_prefix('\n').unwrap_or(raw);
        let raw = raw.strip_suffix('\n').unwrap_or(raw);
        arguments.insert(key.clone(), value(raw, param_type(tools, &name, &key)));
        rest = &rest[end..];
        rest = rest.strip_prefix("</parameter>").unwrap_or(rest);
    }
    Some(ToolCall { name, arguments })
}

/// What can already be sent of an XML call the model is still writing: the
/// function name once complete, and the longest prefix of the final
/// `arguments` JSON that the text so far makes certain.
///
/// **For streaming, so a client shows a long call — a file being written — as
/// it arrives** instead of after minutes of silence (a 177B writing an HTML
/// page at ~7 tok/s). The output is compact JSON built exactly as
/// `Value::Object(arguments).to_string()` builds it, so every prefix returned
/// here is a prefix of the whole call's `arguments`, and each call's result
/// extends the previous one's. Two things are held back to keep that true:
/// a string value's trailing characters that could still be the start of
/// `\n</parameter>` (the final newline is dropped), and any non-string value
/// until it is closed, since it is parsed as JSON only whole.
///
/// `(None, "")` until the function name is complete, and for the JSON format,
/// which is small and sent whole.
pub fn partial_args(text: &str, tools: Option<&Value>) -> (Option<String>, String) {
    const OPEN: &str = "<function=";
    const PARAM: &str = "<parameter=";
    const CLOSE: &str = "</parameter>";
    let Some(f) = text.find(OPEN) else { return (None, String::new()) };
    let rest = &text[f + OPEN.len()..];
    let Some(end_name) = rest.find('>') else { return (None, String::new()) };
    let name = rest[..end_name].trim().to_string();
    if name.is_empty() {
        return (None, String::new());
    }
    let mut rest = &rest[end_name + 1..];
    let done = rest.find("</function>");
    if let Some(end) = done {
        rest = &rest[..end];
    }

    let mut out = String::from("{");
    let mut first = true;
    while let Some(open) = rest.find(PARAM) {
        let after = &rest[open + PARAM.len()..];
        let Some(end_key) = after.find('>') else { return (Some(name), out) };
        let key = after[..end_key].trim().trim_matches(|c| c == '"' || c == '\'');
        let region = &after[end_key + 1..];
        if !first {
            out.push(',');
        }
        first = false;
        out.push_str(&Value::String(key.to_string()).to_string());
        out.push(':');
        let ty = param_type(tools, &name, key);
        match region.find(CLOSE) {
            Some(end) => {
                let raw = &region[..end];
                let raw = raw.strip_prefix('\n').unwrap_or(raw);
                let raw = raw.strip_suffix('\n').unwrap_or(raw);
                out.push_str(&value(raw, ty).to_string());
                rest = &region[end + CLOSE.len()..];
            }
            None => {
                // Still being written. A string streams now; anything else
                // waits to be parsed whole.
                if ty == Some("string") && !region.is_empty() {
                    let body = region.strip_prefix('\n').unwrap_or(region);
                    let safe = &body[..certain_len(body, &format!("\n{CLOSE}"))];
                    let quoted = Value::String(safe.to_string()).to_string();
                    out.push_str(&quoted[..quoted.len() - 1]);
                }
                return (Some(name), out);
            }
        }
    }
    if done.is_some() {
        out.push('}');
    }
    (Some(name), out)
}

/// How much of `body` is certain: all of it, less the longest suffix that is
/// a prefix of `closer` and may yet turn out to be the closer itself.
fn certain_len(body: &str, closer: &str) -> usize {
    let n = body.len();
    for k in (1..=closer.len().min(n)).rev() {
        if body.is_char_boundary(n - k) && closer.starts_with(&body[n - k..]) {
            return n - k;
        }
    }
    n
}

/// The JSON-schema `type` of one parameter of one tool, if the request says.
fn param_type<'a>(tools: Option<&'a Value>, tool: &str, param: &str) -> Option<&'a str> {
    tools?
        .as_array()?
        .iter()
        .filter_map(|t| t.get("function"))
        .find(|f| f.get("name").and_then(Value::as_str) == Some(tool))?
        .get("parameters")?
        .get("properties")?
        .get(param)?
        .get("type")?
        .as_str()
}

/// A parameter's text as a JSON value: text for a string-typed parameter,
/// otherwise JSON when it parses and text when it does not.
fn value(raw: &str, ty: Option<&str>) -> Value {
    if ty == Some("string") {
        return Value::String(raw.to_string());
    }
    serde_json::from_str::<Value>(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tools() -> Value {
        json!([{"type": "function", "function": {"name": "read_file", "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "max_lines": {"type": "integer"}, "code": {"type": "string"}}
        }}}])
    }

    #[test]
    fn xml_call_with_typed_parameters() {
        let t = "\n<function=read_file>\n<parameter=path>\nk:\\x\\README.md\n</parameter>\n<parameter=max_lines>\n20\n</parameter>\n</function>\n";
        let c = parse(t, Some(&tools())).expect("a call");
        assert_eq!(c.name, "read_file");
        assert_eq!(Value::Object(c.arguments), json!({"path": "k:\\x\\README.md", "max_lines": 20}));
    }

    #[test]
    fn a_string_parameter_that_looks_like_json_stays_text() {
        // `"123"` and `true` are valid JSON, but the schema says string.
        let t = "<function=read_file>\n<parameter=path>\n123\n</parameter>\n<parameter=code>\ntrue\n</parameter>\n</function>";
        let c = parse(t, Some(&tools())).unwrap();
        assert_eq!(Value::Object(c.arguments), json!({"path": "123", "code": "true"}));
    }

    #[test]
    fn multi_line_values_keep_their_inner_newlines() {
        let t = "<function=read_file>\n<parameter=code>\nfn main() {\n    println!(\"hi\");\n}\n</parameter>\n</function>";
        let c = parse(t, Some(&tools())).unwrap();
        assert_eq!(c.arguments["code"], json!("fn main() {\n    println!(\"hi\");\n}"));
    }

    #[test]
    fn without_a_schema_values_are_json_when_they_parse() {
        let t = "<function=f>\n<parameter=n>\n3.5\n</parameter>\n<parameter=o>\n{\"a\": [1, 2]}\n</parameter>\n<parameter=s>\nplain words\n</parameter>\n</function>";
        let c = parse(t, None).unwrap();
        assert_eq!(Value::Object(c.arguments), json!({"n": 3.5, "o": {"a": [1, 2]}, "s": "plain words"}));
    }

    #[test]
    fn a_call_with_no_parameters() {
        let c = parse("\n<function=list_models>\n</function>\n", None).unwrap();
        assert_eq!((c.name.as_str(), c.arguments.len()), ("list_models", 0));
    }

    #[test]
    fn json_format_as_qwen3_writes_it() {
        let c = parse("\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.txt\"}}\n", None).unwrap();
        assert_eq!(c.name, "read_file");
        assert_eq!(c.arguments["path"], json!("a.txt"));
        // Arguments given as a string of JSON.
        let c = parse("{\"name\": \"f\", \"arguments\": \"{\\\"x\\\": 1}\"}", None).unwrap();
        assert_eq!(c.arguments["x"], json!(1));
    }

    #[test]
    fn text_that_is_not_a_call_is_refused() {
        assert_eq!(parse("I will now call a tool.", None), None);
        assert_eq!(parse("<function=>\n</function>", None), None);
        assert_eq!(parse("{not json", None), None);
    }

    /// Every cut point of a call: what is streamed only ever grows, and the
    /// whole call ends exactly as the whole-call parse serializes it.
    fn streams_monotonically_to_the_whole(call: &str, tools: Option<&Value>) {
        let whole = parse(call, tools).expect("the whole call parses");
        let target = Value::Object(whole.arguments.clone()).to_string();
        let mut sent = String::new();
        let mut named = false;
        for (cut, _) in call.char_indices().chain(std::iter::once((call.len(), ' '))) {
            let (name, prefix) = partial_args(&call[..cut], tools);
            if let Some(n) = name {
                assert_eq!(n, whole.name, "name at cut {cut}");
                named = true;
            }
            assert!(prefix.starts_with(&sent), "cut {cut}: {prefix:?} does not extend {sent:?}");
            assert!(target.starts_with(&prefix), "cut {cut}: {prefix:?} is not a prefix of {target:?}");
            sent = prefix;
        }
        assert!(named, "the name was never streamed");
        assert_eq!(sent, target, "the stream ends as the whole call");
    }

    #[test]
    fn a_long_file_write_streams_as_it_is_written() {
        let tools = json!([{"type": "function", "function": {"name": "write_to_file", "parameters": {
            "type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}
        }}}]);
        // A value holding `</`, `\n</div>` and a lone `<parameter` lookalike,
        // quotes, a backslash, a tab and a multi-byte character.
        let call = "\n<function=write_to_file>\n<parameter=path>\ndemo/index.html\n</parameter>\n<parameter=content>\n<div class=\"x\">\n\t<p>a\\b é</p>\n</div>\n<!-- <parameter note -->\n</parameter>\n</function>\n";
        streams_monotonically_to_the_whole(call, Some(&tools));
    }

    #[test]
    fn typed_and_untyped_values_stream_to_the_whole() {
        let call = "<function=read_file>\n<parameter=path>\nREADME.md\n</parameter>\n<parameter=max_lines>\n20\n</parameter>\n</function>";
        streams_monotonically_to_the_whole(call, Some(&tools()));
        // No schema: values wait until closed, then parse as JSON if they can.
        let call = "<function=f>\n<parameter=o>\n{\"a\": [1, 2]}\n</parameter>\n<parameter=s>\nwords\n</parameter>\n</function>";
        streams_monotonically_to_the_whole(call, None);
        streams_monotonically_to_the_whole("<function=list_models>\n</function>", None);
    }

    #[test]
    fn nothing_streams_before_the_name_or_for_json_calls() {
        assert_eq!(partial_args("\n<function=writ", None), (None, String::new()));
        assert_eq!(partial_args("{\"name\": \"f\"", None), (None, String::new()));
        // A string value is streamed while open, holding back a possible closer.
        let t = tools();
        let (_, p) = partial_args("<function=read_file>\n<parameter=path>\nREAD\n</param", Some(&t));
        assert_eq!(p, "{\"path\":\"READ");
    }

    #[test]
    fn openai_shape_carries_arguments_as_a_string() {
        let c = ToolCall { name: "f".into(), arguments: json!({"b": 1, "a": "x"}).as_object().unwrap().clone() };
        let v = c.to_openai("call_7");
        assert_eq!(v["id"], "call_7");
        assert_eq!(v["type"], "function");
        // Key order is the model's order, not sorted.
        assert_eq!(v["function"]["arguments"], json!("{\"b\":1,\"a\":\"x\"}"));
    }
}

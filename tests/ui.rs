//! The chat page is shipped inside the binary, so a defect in it is a defect in
//! the release.
//!
//! **Why this file exists.** A stats edit declared `frame` twice in one scope.
//! That is a `SyntaxError`, so the browser refused to run the page's whole
//! script: no click handler, no request, and a server log showing `GET /` and
//! never `POST /v1/chat/completions`. The engine was blamed for an hour.
//!
//! **This is a scanner, not a parser**, and deliberately so — a real check would
//! mean a JavaScript engine, which means a second toolchain, which is the thing
//! this project refuses on purpose (the page is served rather than wrapped in
//! Tauri for the same reason). It catches the two ways a script dies whole:
//! a redeclaration in one scope, and unbalanced delimiters.

const PAGE: &str = include_str!("../ui/index.html");

/// The page's script with comments, strings and template literals blanked, so
/// text inside them cannot be mistaken for code. Regex literals are recognised
/// by the token before them, which is the usual heuristic and enough here.
fn code_only(script: &str) -> String {
    let b: Vec<char> = script.chars().collect();
    let mut out = String::with_capacity(script.len());
    let mut i = 0;
    let mut prev_significant = ' ';
    while i < b.len() {
        let c = b[i];
        match c {
            '/' if i + 1 < b.len() && b[i + 1] == '/' => {
                while i < b.len() && b[i] != '\n' {
                    i += 1;
                }
            }
            '/' if i + 1 < b.len() && b[i + 1] == '*' => {
                i += 2;
                while i + 1 < b.len() && !(b[i] == '*' && b[i + 1] == '/') {
                    i += 1;
                }
                i = (i + 2).min(b.len());
            }
            // A `/` opens a regex only where a value may start.
            '/' if matches!(prev_significant, '(' | ',' | '=' | ':' | '[' | '!' | '&' | '|' | '?' | '{' | '}' | ';' | ' ') => {
                i += 1;
                while i < b.len() && b[i] != '/' {
                    if b[i] == '\\' {
                        i += 1;
                    }
                    i += 1;
                }
                i += 1;
                out.push('0'); // a value stood here
                prev_significant = '0';
            }
            '"' | '\'' | '`' => {
                let quote = c;
                i += 1;
                while i < b.len() && b[i] != quote {
                    // `${...}` inside a template holds real code, but nothing in
                    // this page declares anything there; blanking it is safe and
                    // keeps the scanner simple.
                    if b[i] == '\\' {
                        i += 1;
                    }
                    i += 1;
                }
                i += 1;
                out.push('0');
                prev_significant = '0';
            }
            _ => {
                out.push(c);
                if !c.is_whitespace() {
                    prev_significant = c;
                }
                i += 1;
            }
        }
    }
    out
}

fn script_of(page: &str) -> &str {
    let start = page.find("<script>").expect("the page has a <script> block") + "<script>".len();
    let end = page[start..].find("</script>").expect("the script is closed") + start;
    &page[start..end]
}

#[test]
fn the_page_script_has_balanced_delimiters() {
    let code = code_only(script_of(PAGE));
    let mut stack: Vec<(char, usize)> = Vec::new();
    let mut line = 1;
    for c in code.chars() {
        match c {
            '\n' => line += 1,
            '{' | '(' | '[' => stack.push((c, line)),
            '}' | ')' | ']' => {
                let want = match c {
                    '}' => '{',
                    ')' => '(',
                    _ => '[',
                };
                match stack.pop() {
                    Some((open, _)) if open == want => {}
                    Some((open, at)) => panic!("line {line}: {c} closes {open} opened at line {at}"),
                    None => panic!("line {line}: {c} with nothing open"),
                }
            }
            _ => {}
        }
    }
    assert!(stack.is_empty(), "unclosed {:?}", stack);
}

/// **The defect this file was written for.** `const frame` at the top of a block
/// and `let frame` inside the same block is a `SyntaxError`, and the page dies
/// entirely — not just the feature that was edited.
#[test]
fn the_page_script_declares_each_name_once_per_scope() {
    let code = code_only(script_of(PAGE));
    let mut scopes: Vec<Vec<(String, usize)>> = vec![Vec::new()];
    let mut line = 1;
    let chars: Vec<char> = code.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '\n' => {
                line += 1;
                i += 1;
            }
            '{' => {
                scopes.push(Vec::new());
                i += 1;
            }
            '}' => {
                scopes.pop();
                if scopes.is_empty() {
                    scopes.push(Vec::new());
                }
                i += 1;
            }
            _ => {
                let rest: String = chars[i..].iter().take(6).collect();
                let kw = if rest.starts_with("const ") {
                    Some(6)
                } else if rest.starts_with("let ") {
                    Some(4)
                } else {
                    None
                };
                // Only at a token boundary: `x.const` or `letters` are not it.
                let boundary = i == 0 || !chars[i - 1].is_alphanumeric() && chars[i - 1] != '_' && chars[i - 1] != '.';
                match kw.filter(|_| boundary) {
                    Some(skip) => {
                        i += skip;
                        // One or more names, until `=` or the end of the statement.
                        // Destructuring (`const [a, b] =`) declares each of them.
                        let mut name = String::new();
                        while i < chars.len() && chars[i] != '=' && chars[i] != ';' && chars[i] != '\n' {
                            let c = chars[i];
                            if c.is_alphanumeric() || c == '_' || c == '$' {
                                name.push(c);
                            } else if !name.is_empty() {
                                declare(&mut scopes, &name, line);
                                name.clear();
                            }
                            i += 1;
                        }
                        if !name.is_empty() {
                            declare(&mut scopes, &name, line);
                        }
                    }
                    None => i += 1,
                }
            }
        }
    }
}

fn declare(scopes: &mut [Vec<(String, usize)>], name: &str, line: usize) {
    let scope = scopes.last_mut().expect("a scope is always open");
    if let Some((_, first)) = scope.iter().find(|(n, _)| n == name) {
        panic!("line {line}: `{name}` is declared again in the scope that declared it at line {first} — the browser refuses the whole script");
    }
    scope.push((name.to_string(), line));
}

/// The page is only useful if it calls the endpoints this server serves.
#[test]
fn the_page_calls_the_endpoints_the_server_has() {
    for endpoint in ["v1/models", "v1/chat/completions"] {
        assert!(PAGE.contains(endpoint), "the page no longer calls {endpoint}");
    }
    assert!(PAGE.contains("reasoning_content"), "the page must split the reasoning out");
}

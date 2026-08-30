use std::process::ExitCode;

use clap::{Parser, Subcommand};

use inferred_thoughts::gguf::{Array, GgufFile, TensorInfo, Value};

#[derive(Parser)]
#[command(name = "inferred", version, about = "A from-scratch GGUF inference engine")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Parse a GGUF file and print its metadata and tensor index.
    Inspect {
        /// Path to the .gguf file.
        path: String,

        /// Emit machine-readable JSON, for diffing against gguf_dump.py --json.
        #[arg(long)]
        json: bool,

        /// Truncate arrays longer than this in human-readable output.
        #[arg(long, default_value_t = 16)]
        max_array: usize,
    },

    /// Generate text greedily.
    ///
    /// Stage 4 has no KV cache, so every token re-runs the whole sequence.
    /// Correct but quadratic -- Stage 5 fixes that.
    Generate {
        #[arg(short, long)]
        model: String,
        #[arg(short, long)]
        prompt: String,
        /// Maximum new tokens to produce.
        #[arg(short = 'n', long, default_value_t = 16)]
        max_tokens: usize,
        /// Keep going past the end-of-sequence token.
        #[arg(long)]
        ignore_eos: bool,
    },

    /// Run one forward pass and print a checksum of every intermediate tensor,
    /// for diffing against `llama-eval-callback`.
    Trace {
        #[arg(short, long)]
        model: String,
        #[arg(short, long)]
        prompt: String,
        /// Also write every intermediate tensor's full contents here, so a
        /// comparison can check individual elements and not just checksums.
        #[arg(long)]
        dump: Option<String>,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Inspect {
            path,
            json,
            max_array,
        } => inspect(&path, json, max_array),
        Command::Generate {
            model,
            prompt,
            max_tokens,
            ignore_eos,
        } => generate(&model, &prompt, max_tokens, ignore_eos),
        Command::Trace { model, prompt, dump } => trace(&model, &prompt, dump.as_deref()),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            // thiserror's #[source] chain, so an io error shows the OS message.
            let mut src = std::error::Error::source(&e);
            while let Some(s) = src {
                eprintln!("  caused by: {s}");
                src = s.source();
            }
            ExitCode::FAILURE
        }
    }
}

fn inspect(path: &str, json: bool, max_array: usize) -> inferred_thoughts::Result<()> {
    let f = GgufFile::open(path)?;
    if json {
        print_json(&f);
    } else {
        print_human(&f, max_array);
    }
    Ok(())
}

// -------------------------------------------------------------------- generate

/// Greedy generation, recomputing the whole sequence per token.
///
/// This exists so the engine is usable end to end before the KV cache lands.
/// Cost is quadratic in sequence length by construction; Stage 5 replaces it.
fn generate(
    model: &str,
    prompt: &str,
    max_tokens: usize,
    ignore_eos: bool,
) -> inferred_thoughts::Result<()> {
    use inferred_thoughts::{Naive, Qwen3, Tokenizer};
    use std::io::Write;

    let f = GgufFile::open(model)?;
    let tk = Tokenizer::from_metadata(&f.metadata)?;
    let m = Qwen3::load(&f)?;

    let mut tokens = tk.encode(prompt, true, true);
    eprintln!(
        "model {} | {} layers | {} prompt tokens",
        f.path.file_name().unwrap_or_default().to_string_lossy(),
        m.cfg.n_layer,
        tokens.len()
    );

    print!("{prompt}");
    let _ = std::io::stdout().flush();

    let started = std::time::Instant::now();
    let mut produced = 0usize;

    for _ in 0..max_tokens {
        let logits = m.forward(&Naive, &tokens, &mut |_, _, _| {})?;
        let next = Qwen3::argmax(&logits);

        if !ignore_eos && Some(next) == tk.eos_token_id {
            eprintln!("\n[eos]");
            break;
        }

        // Decode incrementally so multi-token characters still render: decode
        // the whole sequence and print only what is new.
        let before = tk.decode(&tokens, false)?;
        tokens.push(next);
        let after = tk.decode(&tokens, false)?;
        if let Some(new_text) = after.strip_prefix(&before) {
            print!("{new_text}");
            let _ = std::io::stdout().flush();
        }
        produced += 1;
    }

    let secs = started.elapsed().as_secs_f64();
    println!();
    eprintln!(
        "\n{produced} tokens in {secs:.1}s ({:.2} tok/s, {:.0} ms/token)",
        produced as f64 / secs,
        secs * 1000.0 / produced.max(1) as f64
    );
    Ok(())
}

// ----------------------------------------------------------------------- trace

/// Emit `name<TAB>n_elements<TAB>sum` per intermediate tensor.
///
/// The sum is accumulated in f64 and covers the whole tensor, which is what
/// makes it comparable to the `sum = ...` line `llama-eval-callback` prints.
fn trace(model: &str, prompt: &str, dump: Option<&str>) -> inferred_thoughts::Result<()> {
    use inferred_thoughts::{Naive, Qwen3, Tokenizer};
    use std::io::Write;

    let f = GgufFile::open(model)?;
    let tk = Tokenizer::from_metadata(&f.metadata)?;
    let m = Qwen3::load(&f)?;

    let tokens = tk.encode(prompt, true, true);
    eprintln!("tokens ({}): {tokens:?}", tokens.len());

    // Optional binary sidecar: [u32 name_len][name][u64 n][n * f32], repeated.
    // Our flat layout matches ggml's, so a comparison can slice this by the
    // reference's own ne0 without us tracking tensor shapes here.
    let mut sink = match dump {
        Some(path) => Some(std::io::BufWriter::new(std::fs::File::create(path).map_err(
            |source| inferred_thoughts::Error::Io {
                path: path.to_string(),
                source,
            },
        )?)),
        None => None,
    };

    let mut emit = |name: &str, il: usize, data: &[f32]| {
        let sum: f64 = data.iter().map(|&v| v as f64).sum();
        // Layer-scoped names carry the index, matching the reference's
        // "attn_norm-0" convention; whole-model ones do not.
        let label = match name {
            "inp_embd" | "result_norm" | "result_output" => name.to_string(),
            _ => format!("{name}-{il}"),
        };
        println!("{label}\t{}\t{sum:.6}", data.len());

        if let Some(w) = sink.as_mut() {
            let nb = label.as_bytes();
            let _ = w.write_all(&(nb.len() as u32).to_le_bytes());
            let _ = w.write_all(nb);
            let _ = w.write_all(&(data.len() as u64).to_le_bytes());
            for &v in data {
                let _ = w.write_all(&v.to_le_bytes());
            }
        }
    };

    let logits = m.forward(&Naive, &tokens, &mut emit)?;

    let top = Qwen3::argmax(&logits);
    eprintln!(
        "argmax = {top} {:?}",
        tk.token_text(top).unwrap_or("<out of range>")
    );
    Ok(())
}

// ---------------------------------------------------------------- human output

fn print_human(f: &GgufFile, max_array: usize) {
    let arch = f
        .metadata
        .architecture()
        .map(|s| s.to_string())
        .unwrap_or_else(|_| "<missing general.architecture>".to_string());

    println!("file:       {}", f.path.display());
    println!(
        "size:       {} ({} bytes)",
        human_bytes(f.file_size()),
        f.file_size()
    );
    println!("version:    {}", f.version);
    println!("alignment:  {}", f.alignment);
    println!("arch:       {arch}");
    println!("kv pairs:   {}", f.metadata.len());
    println!("tensors:    {}", f.tensors.len());
    println!("data at:    {} (0x{:x})", f.data_offset, f.data_offset);
    println!(
        "weights:    {} across {} params",
        human_bytes(f.total_tensor_bytes()),
        human_count(f.total_parameters())
    );

    println!("\nmetadata ({}):", f.metadata.len());
    let key_width = f.metadata.iter().map(|(k, _)| k.len()).max().unwrap_or(0);
    for (key, value) in f.metadata.iter() {
        println!(
            "  {key:<key_width$}  {:<10}  {}",
            type_label(value),
            format_value(value, max_array)
        );
    }

    println!("\ntensors ({}):", f.tensors.len());
    let name_width = f.tensors.iter().map(|t| t.name.len()).max().unwrap_or(0);
    println!(
        "  {:>4}  {:<name_width$}  {:<8}  {:<26}  {:>14}  {:>14}",
        "idx", "name", "type", "shape", "offset", "bytes"
    );
    for (i, t) in f.tensors.iter().enumerate() {
        println!(
            "  {i:>4}  {:<name_width$}  {:<8}  {:<26}  {:>14}  {:>14}",
            t.name,
            t.ty.name(),
            format_shape(t),
            t.offset,
            t.n_bytes
        );
    }

    // A per-type summary makes it obvious at a glance which quant formats a
    // later stage must implement to load this file.
    println!("\ntype summary:");
    let mut counts: Vec<(&'static str, usize, u64)> = Vec::new();
    for t in &f.tensors {
        match counts.iter_mut().find(|(n, _, _)| *n == t.ty.name()) {
            Some(entry) => {
                entry.1 += 1;
                entry.2 += t.n_bytes;
            }
            None => counts.push((t.ty.name(), 1, t.n_bytes)),
        }
    }
    counts.sort_by(|a, b| b.2.cmp(&a.2));
    for (name, n, bytes) in counts {
        println!("  {name:<8}  {n:>4} tensors  {:>10}", human_bytes(bytes));
    }
}

fn format_shape(t: &TensorInfo) -> String {
    t.dims
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(" x ")
}

fn type_label(v: &Value) -> String {
    match v {
        Value::Array(a) => format!("arr[{}]", a.elem_type_name()),
        other => other.type_name().to_string(),
    }
}

fn format_value(v: &Value, max_array: usize) -> String {
    match v {
        Value::U8(x) => x.to_string(),
        Value::I8(x) => x.to_string(),
        Value::U16(x) => x.to_string(),
        Value::I16(x) => x.to_string(),
        Value::U32(x) => x.to_string(),
        Value::I32(x) => x.to_string(),
        Value::F32(x) => x.to_string(),
        Value::F64(x) => x.to_string(),
        Value::U64(x) => x.to_string(),
        Value::I64(x) => x.to_string(),
        Value::Bool(x) => x.to_string(),
        Value::String(s) => format_string(s),
        Value::Array(a) => {
            let n = a.len();
            let shown = n.min(max_array);
            let mut parts: Vec<String> = (0..shown).map(|i| a.elem_to_string(i)).collect();
            if n > shown {
                parts.push(format!("... {} more", n - shown));
            }
            format!("[{n}] {{{}}}", parts.join(", "))
        }
    }
}

/// Long strings (chat templates run to thousands of characters) are elided so
/// the metadata table stays readable.
fn format_string(s: &str) -> String {
    const LIMIT: usize = 120;
    if s.chars().count() <= LIMIT {
        return format!("{s:?}");
    }
    let head: String = s.chars().take(LIMIT).collect();
    format!("{head:?}... ({} chars total)", s.chars().count())
}

fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{n} B")
    } else {
        format!("{v:.2} {}", UNITS[u])
    }
}

fn human_count(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1e6)
    } else {
        n.to_string()
    }
}

// ----------------------------------------------------------------- json output

/// Hand-rolled so the crate keeps no serialization dependency. The shape is
/// chosen to be easy to compare against `gguf_dump.py --json` from a script.
fn print_json(f: &GgufFile) {
    println!("{{");
    println!("  \"filename\": {},", json_string(&f.path.display().to_string()));
    println!("  \"version\": {},", f.version);
    println!("  \"alignment\": {},", f.alignment);
    println!("  \"data_offset\": {},", f.data_offset);
    println!("  \"n_kv\": {},", f.metadata.len());
    println!("  \"n_tensors\": {},", f.tensors.len());

    println!("  \"kv\": {{");
    let n_kv = f.metadata.len();
    for (i, (key, value)) in f.metadata.iter().enumerate() {
        let comma = if i + 1 < n_kv { "," } else { "" };
        println!(
            "    {}: {{\"type\": {}, \"value\": {}}}{comma}",
            json_string(key),
            json_string(&type_label(value)),
            json_value(value)
        );
    }
    println!("  }},");

    println!("  \"tensors\": {{");
    let n_t = f.tensors.len();
    for (i, t) in f.tensors.iter().enumerate() {
        let comma = if i + 1 < n_t { "," } else { "" };
        let shape = t
            .dims
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "    {}: {{\"index\": {i}, \"shape\": [{shape}], \"type\": {}, \"offset\": {}, \"n_bytes\": {}}}{comma}",
            json_string(&t.name),
            json_string(t.ty.name()),
            t.offset,
            t.n_bytes
        );
    }
    println!("  }}");
    println!("}}");
}

fn json_value(v: &Value) -> String {
    match v {
        Value::U8(x) => x.to_string(),
        Value::I8(x) => x.to_string(),
        Value::U16(x) => x.to_string(),
        Value::I16(x) => x.to_string(),
        Value::U32(x) => x.to_string(),
        Value::I32(x) => x.to_string(),
        Value::U64(x) => x.to_string(),
        Value::I64(x) => x.to_string(),
        Value::F32(x) => json_float(f64::from(*x)),
        Value::F64(x) => json_float(*x),
        Value::Bool(x) => x.to_string(),
        Value::String(s) => json_string(s),
        Value::Array(a) => json_array(a),
    }
}

fn json_array(a: &Array) -> String {
    fn join<T: ToString>(v: &[T]) -> String {
        v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")
    }
    let body = match a {
        Array::U8(v) => join(v),
        Array::I8(v) => join(v),
        Array::U16(v) => join(v),
        Array::I16(v) => join(v),
        Array::U32(v) => join(v),
        Array::I32(v) => join(v),
        Array::U64(v) => join(v),
        Array::I64(v) => join(v),
        Array::Bool(v) => join(v),
        Array::F32(v) => v
            .iter()
            .map(|x| json_float(f64::from(*x)))
            .collect::<Vec<_>>()
            .join(", "),
        Array::F64(v) => v.iter().map(|x| json_float(*x)).collect::<Vec<_>>().join(", "),
        Array::String(v) => v.iter().map(|s| json_string(s)).collect::<Vec<_>>().join(", "),
    };
    format!("[{body}]")
}

/// JSON has no NaN or Infinity literals; emit null so output stays parseable
/// rather than silently producing something a JSON reader rejects.
fn json_float(x: f64) -> String {
    if x.is_finite() { x.to_string() } else { "null".to_string() }
}

fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

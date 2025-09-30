# Ménage à trois

**A three body problem.**

![Banner Image](/img/header.jpeg "A header image depicting a stochastic parrot.")
<br>Will small variations lead to diverse conversational outcomes? | Études d'un [perroquet](https://en.wikipedia.org/wiki/Stochastic_parrot) ara by P. Boel *([Louvre](https://collections.louvre.fr/ark:/53355/cl010061519) collection)*

## Genesis

> I got this idea while watching a Netflix [series](https://en.wikipedia.org/wiki/3_Body_Problem_(TV_series)). The foundational principle about it is simulating social dynamics borrowing concepts from physics as one might see in models of celestial mechanics or particle interactions and apply these ideas *metaphorically* to the realm of conversations and social influence. It's a bit like treating conversations as if they were governed by **forces** of attraction or repulsion. To achieve this, the project simulates a multi-agent (toy) conversational system, where each agent holds an internal “gravitational” state and generates utterances via a local Ollama instance.
> There is no serious science behind it! In this case, it is more about having fun with the *(serious)* concept of the [three-body problem](https://en.wikipedia.org/wiki/Three-body_problem).

---

## Table of Contents

- [Features](#features) 
- [How it Works](#how-it-works)   
- [Requirements](#requirements)  
- [Installation](#installation)  
- [Configuration](#configuration)  
- [Usage](#usage)  
- [Explanation](#explanation)  
- [Details](#details)  
- [Logging & output files](#logging--output-files)   
- [Customization](#customization)  
- [Limitations](#limitations)  
- [License](#license)

---

## Features

- Runs multiple conversational agents that *influence each other's internal state* using a simple **gravitation-inspired** interaction model.
- Queries a local [Ollama](https://ollama.com/) instance via HTTP to generate utterances.
- Possibility to write three types of outputs: 
    - full debug log
    - chat-only log 
    - export as a CSV dataset for later conversation analysis
- Configurable toggles for logging and exporting.

## How it works

1. Each agent holds a numeric internal state: `mass`, `topic_interest`, `tone`, and `topic_velocity`.
2. Agents compute pairwise **"gravitational" influences** to update topic interest and tone.
3. Agents build prompts containing a rolling conversation summary and their current internal state (topic interest/tone are presented in aggregated form).
4. Prompts are sent to the local Ollama API (`/api/generate`) via an HTTP POST; responses are parsed, cleaned, logged, and appended to the shared summary.
5. The simulation loop prints each turn to `stdout` and optionally logs to files.

## Requirements

- Python 3.10+
- Ollama (0.7.0+) installed and running locally (CLI available as `ollama`)
- A local Ollama API reachable at http://localhost:11434 (configurable)  
- **Python packages**: 
    - requests
    - standard library modules used: re, csv, json, random, subprocess, datetime, dataclasses, typing, pathlib

Install requests if needed:

```bash
pip install requests
```

> [!TIP]
> Under Linux, run this command to install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## Installation

1. Clone or copy the script into a project directory.
2. To ensure Ollama is installed and you can run:

    ```bash
    ollama list
    ``` 

3. Install Python dependencies (see [Requirements](#requirements) above).

## Configuration

Configuration is handled by the `Config` dataclass at the top of the script. 
Key options:

| Name | Type | Description | Default |
|---|---:|---|---|
| list_available_llms | bool | Print `ollama list` output. | **False** |
| ollama_host | str | Host URL for the Ollama API. | **http://localhost:11434** |
| full_log | bool | Enable/write detailed debug log. | **True** |
| full_log_filepath | Path | Path to detailed debug log file. | **full_log.txt** |
| chat_only_log | bool | Enable/write chat-only entries. | **True** |
| chat_only_filepath | Path | Path to chat-only log file. | **chat_only.txt** |
| export_as_dataset | bool | Append conversation records to a CSV dataset. | **True** |
| csv_filepath | Path | Path to conversation CSV dataset. | **conversation_dataset.csv** |

- Toggle these by editing the Config instantiation near the top of the script.

## Usage

- Update config flags or file paths in the `Config` dataclass if desired.
*Optionally* change the model *(default: qwen3:0.6b)* identifier in `main()`.

- Run the script:

```bash
python fandango.py
```

The run will:

1. Ensure the model is present (and pull it using `ollama pull <model>` if missing).
2. Instantiate a shared `ConversationSummary` and three `ConversationalAgent` instances.
3. Simulate a small conversation (*default 2 turns*; configurable).
4. Save logs and CSV entries as configured and print paths at exit.

## Explanation

> The gravitational model treats each agent's `topic_interest` as a 1D position and computes pairwise **"forces"** that pull or push that value.

Each agent has a `mass`, `topic_interest`, `topic_velocity`, and `tone`.

Where Force from another agent equals: 

$G \cdot \dfrac{m_{\text{self}}\, m_{\text{other}}}{d^{2} + \varepsilon}$

Signed positive if the other agent's `topic_interest` is higher, negative if lower.

and Acceleration equals:

$\dfrac{F_{\text{total}}}{m_{\text{self}}}$

- `topic_velocity` is integrated with timestep `dt`.

- `topic_interest` is updated from `velocity` plus a small random noise.

- `Tone` is nudged proportionally to the `topic_velocity` each step (with small randomness).

> **Summary:** masses scale influence magnitude, proximity *(difference in  topic_interest)* increases force nonlinearly, and signed force moves agents toward or away from other's topic_interest, producing **evolving topic positions** and correlated **tone changes**.

![caption](/img/Three-body_Problem_Animation_with_COM.gif "Three body problem simulation.")
<br>*Approximate trajectories of three bodies of the same mass with zero initial velocities. ([Wikipedia](https://en.wikipedia.org/wiki/File:Three-body_Problem_Animation_with_COM.gif))*

## Details

| Field | Type | Description |
|---|---:|---|
| agents | array of objects | List of agent states participating in the simulation |
| agents[].name | string | Agent identifier |
| agents[].mass | float (>=0) | Scales influence magnitude |
| agents[].topic_interest | float | 1D position in topic space |
| agents[].topic_velocity | float | Current velocity along topic dimension |
| agents[].tone | float | Sentiment/emotional axis correlated to velocity |
| agents[].subj | string | Derived subject label (e.g., "science"/"philosophy") |
| agents[].mood | string | Derived mood label ("calm..." / "angry...") |
| simulation.steps | integer | Number of simulation turns |
| simulation.dt | float | Time step for integration |
| simulation.G | float | Gravitational constant scaling factor |
| simulation.epsilon | float | Small value to avoid division by zero |
| simulation.random_noise_topic | float | Magnitude of random perturbation added to topic_interest each step |
| simulation.random_noise_tone | float | Magnitude of random perturbation added to tone each step |

## Logging & output files

| File | Contents | When created/updated |
|---|---:|---|
| **full_log.txt** | Rich debug entries: prompt, raw model response, and cleaned utterance. | Written/appended if **full_log** is True |
| **chat_only.txt** | Timestamped agent utterances (one line per utterance). | Written/appended if **chat_only_log** is True |
| **conversation_dataset.csv** | Appended rows tracking: Agent, Turn, Timestamp, Utterance, TopicInterest, Tone. | Header written on first creation; subsequent runs append rows if **export_as_dataset** is True |

- P.-S. The CSV header is written on first creation. Subsequent runs append rows.

## Logging details levels examples


<details>
<summary> CSV output example - click to expand</summary>
  
  <table>
  <tr>
    <th>Agent</th>
    <th>Turn</th>
    <th>Timestamp</th>
    <th>Utterance</th>
    <th>TopicInterest</th>
    <th>Tone</th>
  </tr>
  <tr>
    <td>Alice</td>
    <td>1</td>
    <td>2025-06-12T18:15:35.253902</td>
    <td>I find the scientific findings troubling, but I can't see a way around it.</td>
    <td>0.001</td>
    <td>-0.156</td>
  </tr>
  <tr>
    <td>Bob</td>
    <td>1</td>
    <td>2025-06-12T18:16:07.739286</td>
    <td>I find the findings troubling and I'm not convinced they're accurate.</td>
    <td>0.004</td>
    <td>-0.009</td>
  </tr>
  <tr>
    <td>Charlie</td>
    <td>1</td>
    <td>2025-06-12T18:16:59.114191</td>
    <td>The current philosophical discussion suggests a tension between empirical truth and existential implications. Charlie’s anger may stem from questioning the validity of existing findings, implying that the subject matter extends beyond surface-level analysis. If this were a deeper existential truth, it challenges the boundaries of empirical knowledge, suggesting a broader philosophical truth that remains unresolved.</td>
    <td>-0.066</td>
    <td>-0.826</td>
  </tr>
  <tr>
    <td>Alice</td>
    <td>2</td>
    <td>2025-06-12T18:17:48.637183</td>
    <td>The topic of science remains unresolved, with Alice’s findings sparking interest in the broader philosophical dimensions of empirical knowledge. Her cheerful tone suggests a perspective that encourages further exploration of the subject.</td>
    <td>64.750</td>
    <td>32.225</td>
  </tr>
</table>
  
</details>


<details>
<summary> Chat only export example - click to expand</summary>
  
  
  
</details>


<details>
<summary> Full log details - click to expand</summary>

<table>
  <tr>
    <th>Timestamp</th>
    <th>Actor</th>
    <th>Prompt</th>
    <th>Raw Response</th>
    <th>Extracted Utterance</th>
  </tr>
  <tr>
    <td>2025-06-12T15:25:07.538094</td>
    <td>Alice</td>
    <td>
      Below is the conversation so far between Alice, Bob, and Charlie:
      The conversation is just getting started.
      Now it is Alice's turn. Alice's internal state:
       • topic_interest = 0.50 (science)
       • tone = 0.20 (enthusiastic)
      Based on the full conversation above and your current state, please produce a single short, creative comment to advance the discussion on science.
    </td>
    <td>
{"model":"qwen3:0.6b","created_at":"2025-06-12T15:25:07.522343628Z",
"response":"\"Quantum entanglement in everyday life challenges our understanding of reality—science is still alive, and Alice’s enthusiasm for it’s a beacon of wonder!\"",
"done":true}
    </td>
    <td>
&lt;think&gt; ... (internal reasoning) ... &lt;/think&gt;
"Quantum entanglement in everyday life challenges our understanding of reality—science is still alive, and Alice’s enthusiasm for it’s a beacon of wonder!"
    </td>
  </tr>

  <tr>
    <td>2025-06-12T15:26:21.857151</td>
    <td>Bob</td>
    <td>
      Below is the conversation so far between Alice, Bob, and Charlie:
      Now it is Bob's turn. Bob's internal state:
       • topic_interest = -0.30 (philosophy)
       • tone = -0.10 (cautious)
      Based on the full conversation above and your current state, please produce a single short, creative comment to advance the discussion on philosophy.
    </td>
    <td>
{"model":"qwen3:0.6b","created_at":"2025-06-12T15:26:21.846859023Z",
"response":"\"Philosophy’s exploration of existence is a profound journey, and Bob’s cautious yet thoughtful nature makes him a natural subject of inquiry.\"",
"done":true}
    </td>
    <td>
&lt;think&gt; ... (internal reasoning) ... &lt;/think&gt;
"Philosophy’s exploration of existence is a profound journey, and Bob’s cautious yet thoughtful nature makes him a natural subject of inquiry."
    </td>
  </tr>
</table>

<table>
  <tr>
    <th>Key found in RAW conversation</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>"done"</td>
    <td>true — Model finished generating a response.</td>
  </tr>
  <tr>
    <td>"done_reason"</td>
    <td>"stop" — Generation ended normally (no error).</td>
  </tr>
  <tr>
    <td>"context"</td>
    <td>[...] — An array of numeric token/trace identifiers representing model-internal context and history (not human-readable conversation text).</td>
  </tr>
  <tr>
    <td>"total_duration"</td>
    <td>71303763995 — Total time spent (in nanoseconds) processing the session.</td>
  </tr>
  <tr>
    <td>"load_duration"</td>
    <td>22961990 — Time spent loading resources (nanoseconds).</td>
  </tr>
  <tr>
    <td>"prompt_eval_count"</td>
    <td>460 — Number of prompt evaluation calls.</td>
  </tr>
  <tr>
    <td>"prompt_eval_duration"</td>
    <td>37894459462 — Cumulative time spent evaluating prompts (nanoseconds).</td>
  </tr>
  <tr>
    <td>"eval_count"</td>
    <td>226 — Number of model evaluations performed.</td>
  </tr>
  <tr>
    <td>"eval_duration"</td>
    <td>33385623883 — Cumulative time spent in model evaluations (nanoseconds).</td>
  </tr>
</table
  
</details>

## Customization

- Increase `ConversationSummary.max_lines` to keep longer context.
- Modify `ConversationalAgent.generate_prompt()` to change prompt style, system instructions, or to hide/reveal different internal-state cues.
- Swap or change the model used in `main()` 
    - e.g., `mistral:latest` or others available locally, [full list](https://ollama.com/library).
- Change `simulate_conversation(..., steps=...)` to run longer simulations.
- Replace HTTP query logic to use streaming or alternative Ollama endpoints if desired.

## Limitations

- Randomness is used for small perturbations — repeated runs will differ.
- This tool is intended for **experimentation** and analysis, not production deployment.

## License

Provided under [CC0](https://creativecommons.org/public-domain/cc0/) [license](/LICENSE) (public domain). Use, modify, and redistribute freely. 🃜 

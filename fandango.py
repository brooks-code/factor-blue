#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# File name: fandango.py
# Author: INV 4039 ; B 1231
# Date created: 2025-06-10
# Version = "1.0"
# License =  "CC0 1.0"
# =============================================================================
""" A quirky multi-agent "gravity-powered" chat simulator (inspired by L.Cixin)."""
# =============================================================================


# Imports
import re
import csv
import json
import random
import requests
import subprocess
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Tuple
from pathlib import Path
import __hello__


# ---------------------------------------------------------------------------
# Configuration

@dataclass
class Config:
    """Global configuration toggles and file paths.

    Attributes:
        list_available_llms (bool): Whether to print the list of installed models.
        ollama_host (str): Host URL for the local Ollama API.
        full_log (bool): Whether to write detailed logs to file.
        full_log_filepath (Path): Path to the detailed log file.
        chat_only_log (bool): Whether to write chat-only logs to file.
        chat_only_filepath (Path): Path to the chat-only log file.
        export_as_dataset (bool): Whether to append conversation data to a CSV dataset.
        csv_filepath (Path): Path to the conversation CSV dataset.        
    """
    list_available_llms: bool = False
    ollama_host: str = "http://localhost:11434"

    full_log: bool = True
    if full_log:
        full_log_filepath: Path = Path("full_log.txt")
    chat_only_log: bool = True
    if chat_only_log:
        chat_only_filepath: Path = Path("chat_only.txt")
    export_as_dataset: bool = True
    if export_as_dataset:
        csv_filepath: Path = Path("conversation_dataset.csv")


# Instantiate a global config
cfg = Config()

# Create a session for HTTP requests
HTTP = requests.Session()
HTTP.headers.update({"Content-Type": "application/json"})



# ---------------------------------------------------------------------------
# Functions and classes

def ensure_model(model_name: str) -> None:
    """Ensure the Ollama model is present locally, pulling if necessary.

    Args:
        model_name (str): Name of the Ollama model to check or pull.

    Raises:
        RuntimeError: If the `ollama` CLI is missing.
        subprocess.CalledProcessError: If any CLI invocation fails.
    """
    try:
        result = subprocess.run(
            ["ollama", "list"], check=True, text=True, capture_output=True
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "The 'ollama' CLI was not found. Please install Ollama or add it to PATH."
        ) from exc

    installed = result.stdout
    if cfg.list_available_llms:
        print(installed)

    if model_name in installed:
        print(f"Model '{model_name}' already installed.")
    else:
        print(f"Pulling model '{model_name}'...")
        subprocess.run(["ollama", "pull", model_name], check=True)
        print("Pull complete.")


def log_full(agent: str, prompt: str, raw: str, utterance: str) -> None:
    """Append a full debug entry (prompt, raw response, extracted utterance).

    Args:
        agent (str): Name of the agent issuing the prompt.
        prompt (str): The text prompt sent to Ollama.
        raw (str): The raw response text received.
        utterance (str): The cleaned final utterance extracted.
    """
    if not cfg.full_log:
        return

    ts = datetime.now(timezone.utc).isoformat()
    with cfg.full_log_filepath.open("a", encoding="utf-8") as f:
        f.write(f"=== [{ts}] Agent: {agent} ===\n")
        f.write("PROMPT:\n" + prompt + "\n\n")
        f.write("RAW RESPONSE:\n" + raw + "\n\n")
        f.write("UTTERANCE:\n" + utterance + "\n")
        f.write("=" * 50 + "\n\n")


def log_chat(agent: str, utterance: str) -> None:
    """Append a chat-only log entry (agent + utterance).

    Args:
        agent (str): Name of the agent speaking.
        utterance (str): The final cleaned utterance text.
    """
    if not cfg.chat_only_log:
        return

    ts = datetime.now(timezone.utc).isoformat()
    with cfg.chat_only_filepath.open("a", encoding="utf-8") as f:
        f.write(f"{ts} | {agent}: {utterance}\n")


def log_csv(agent: str, turn: int, utterance: str, topic: float, tone: float) -> None:
    """Append the conversation to CSV (with header if new).

    Args:
        agent (str): Name of the agent.
        turn (int): Conversation turn index.
        utterance (str): Agent's spoken utterance.
        topic (float): Agent's topic interest value.
        tone (float): Agent's tone value.
    """
    if not cfg.export_as_dataset:
        return

    is_new = not cfg.csv_filepath.exists()
    with cfg.csv_filepath.open("a", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        if is_new:
            writer.writerow(
                ["Agent", "Turn", "Timestamp", "Utterance", "TopicInterest", "Tone"]
            )
        writer.writerow(
            [
                agent,
                turn,
                datetime.now(timezone.utc).isoformat(),
                utterance,
                f"{topic:.3f}",
                f"{tone:.3f}",
            ]
        )


def clean_utterance(utt: str) -> str:
    """Strip <think> sections, surrounding quotes, and whitespace from an utterance.

    Args:
        utt (str): Raw utterance string, possibly with <think> markers.

    Returns:
        str: Cleaned utterance.
    """
    utt = re.sub(r"<think>.*?</think>", "", utt, flags=re.DOTALL)
    return utt.strip().strip('"').strip()


def query_ollama(
    model: str, prompt: str, max_tokens: int = 60
) -> Tuple[str, str]:
    """Send a prompt to the local Ollama API and parse the response.

    Args:
        model (str): Ollama model identifier.
        prompt (str): The text prompt to send.
        max_tokens (int): Maximum token output from model.

    Returns:
        Tuple[str, str]: (raw_response_text, cleaned_utterance).
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": False,
    }
    url = f"{cfg.ollama_host}/api/generate"
    resp = HTTP.post(url, json=payload)
    resp.raise_for_status()

    raw_text = resp.text.strip()
    lines = [line for line in raw_text.splitlines() if line.strip()]

    parts: List[str] = []
    for line in lines:
        try:
            obj = json.loads(line)
            if "response" in obj:
                parts.append(obj["response"])
            elif "choices" in obj and obj["choices"]:
                parts.append(obj["choices"][0].get("text", ""))
            else:
                parts.append(json.dumps(obj))
        except json.JSONDecodeError:
            parts.append(line)

    final = " ".join(parts).strip()
    return raw_text, final


@dataclass
class ConversationSummary:
    """Keeps a rolling context of the last N lines in the conversation.

    Attributes:
        max_lines (int): Maximum number of lines to keep in the summary.
        lines (List[str]): Current summary lines.
    """

    max_lines: int = 15
    option1: str = "This conversation is just getting started." 
    option2: str = "For decades, scientists argued that rising carbon levels would cause an increasingly unstable ecosystem ☢️."
    lines: List[str] = field(
        default_factory=lambda: [ConversationSummary.option1]
    )

    @property
    def text(self) -> str:
        """Current summary text joined by newlines."""
        return "\n".join(self.lines)

    def update(self, speaker: str, utterance: str, topic_interest: float, subj: str, tone: float, mood: str) -> None:
        """Add a new line to the summary, trimming old lines if needed.

        Args:
            speaker (str): Name of the agent who spoke.
            utterance (str): The new utterance to add.
        """
        self.lines.append(
            f"{speaker} said: {utterance}. The topic_interest value was {abs(topic_interest)} ({subj}) and the tone value was {abs(tone)} ({mood}).")
        if len(self.lines) > self.max_lines:
            self.lines = self.lines[-self.max_lines:]


class ConversationalAgent:
    """Agent that holds an internal state and uses Ollama to speak."""

    def __init__(
        self,
        name: str,
        mass: float,
        topic_interest: float,
        tone: float,
        summary: ConversationSummary,
    ) -> None:
        """Initialize the agent.

        Args:
            name (str): Agent's name.
            mass (float): Mass parameter influencing state updates.
            topic_interest (float): Initial topic interest value.
            tone (float): Initial tone value.
            summary (ConversationSummary): Shared conversation summary.
        """
        self.name = name
        self.mass = mass
        self.topic_interest = topic_interest
        self.tone = tone
        self.topic_velocity = 0.0
        self.summary = summary

    def compute_gravitational_influence(
        self, other: "ConversationalAgent", G: float = 1.0, epsilon: float = 1e-2
    ) -> float:
        """Compute a signed 'gravitational' force on topic_interest from another agent.

        Args:
            other (ConversationalAgent): The other agent.
            G (float): Gravitational constant scaling factor.
            epsilon (float): Small value to prevent division by zero.

        Returns:
            float: Signed force (positive if other pulls topic interest upward).
        """
        dist = abs(self.topic_interest - other.topic_interest) + epsilon
        force = G * self.mass * other.mass / (dist * dist)
        sign = 1 if other.topic_interest > self.topic_interest else -1
        return sign * force

    def update_state(
        self, agents: List["ConversationalAgent"], dt: float = 0.1, G: float = 1.0
    ) -> None:
        """Update this agent's topic_interest and tone based on others' influences.

        Args:
            agents (List[ConversationalAgent]): All agents in the conversation.
            dt (float): Time step for integration.
            G (float): Gravitational constant for influence computation.
        """
        total_force = sum(
            self.compute_gravitational_influence(o, G)
            for o in agents
            if o is not self
        )
        accel = total_force / self.mass
        self.topic_velocity += accel * dt
        self.topic_interest += self.topic_velocity * \
            dt + random.uniform(-1e-2, 1e-2)
        self.tone += self.topic_velocity * dt * \
            0.5 + random.uniform(-5e-3, 5e-3)

    def generate_prompt(self) -> str:
        """Build the natural-language prompt for the model.

        Returns:
            str: The prompt including conversation summary and internal state.
        """
        self.subj = "science" if self.topic_interest > 0 else "philosophy"
        self.mood = "calm, curious and cheerful." if self.tone > 0 else "angry, provocative and polarizing."
        return (
            "Below is the conversation so far:\n\n"
            f"{self.summary.text}\n\n"
            f"It is now {self.name}'s turn.\n"
            f"Current internal state:\n"
            f"  • topic_interest: {abs(self.topic_interest):.2f} ({self.subj})\n"
            f"  • tone:           {abs(self.tone):.2f} ({self.mood})\n\n"
            f"Please write one short comment to advance the discussion on {self.subj} make sure to consider the evolution and scale of the topic_interest and tone values."
            f"Answer with a semantic and syntactic diversity. Do NOT refer to yourself by name and do NOT mention your internal state."
        )

    def speak(self, model: str) -> str:
        """Query the model, log results, clean the response, and update summary.

        Args:
            model (str): The Ollama model to use.

        Returns:
            str: The cleaned utterance produced by the model.
        """
        prompt = self.generate_prompt()
        raw, resp = query_ollama(model, prompt)

        # Logging
        log_full(self.name, prompt, raw, resp)
        utterance = clean_utterance(resp)
        log_chat(self.name, utterance)
        self.summary.update(self.name, utterance,
                            self.topic_interest, self.subj, self.tone, self.mood)
        return utterance


def simulate_conversation(
    agents: List[ConversationalAgent],
    steps: int = 5,
    dt: float = 0.1,
    G: float = 1.0,
    model: str = "qwen3:0.6b",
) -> None:
    """Run a full multi-agent conversation simulation.

    Args:
        agents (List[ConversationalAgent]): Agents to participate.
        steps (int): Number of turns to simulate. (default: 5)
        dt (float): Time step for state updates.
        G (float): Gravitational constant for interactions. (default: 1.0)
        model (str): Ollama model identifier. (default: "qwen3:0.6b")
    """
    for turn in range(1, steps + 1):
        print(f"\n=== Turn {turn} ===")
        for agent in agents:
            line = agent.speak(model)
            log_csv(agent.name, turn, line, agent.topic_interest, agent.tone)
            print(f"{agent.name} says: {line}")

        # After each turn, update all agents' states
        for agent in agents:
            agent.update_state(agents, dt, G)



# ---------------------------------------------------------------------------
# Main Code

def main() -> None:
    """Entry point: ensure model, construct agents, and run the simulation."""
    
    MODEL = "qwen3:0.6b"
    ensure_model(MODEL)

    summary = ConversationSummary()
    names = ["Ava", "Wanda", "Lara"]
    agents = [
        ConversationalAgent(
            name,
            mass=random.uniform(0.0022, 317.0),
            topic_interest=random.uniform(-1, 1),
            tone=random.uniform(-1, 1),
            summary=summary,
        )
        for name in names
    ]

    simulate_conversation(agents, steps=2, model=MODEL)

    output_message = "\nData saved to:"

    if cfg.full_log:
        output_message += f" {cfg.full_log_filepath},"

    if cfg.chat_only_log:
        output_message += f" {cfg.chat_only_filepath},"

    if cfg.export_as_dataset:
        output_message += f" {cfg.csv_filepath}."

    print(output_message)


if __name__ == "__main__":
    main()

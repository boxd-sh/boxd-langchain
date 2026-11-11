"""A real Deep Agents run whose shell and files live in a boxd machine.

Needs ``BOXD_API_KEY`` and ``ANTHROPIC_API_KEY``. The model defaults to
``claude-sonnet-5-5``; set ``ANTHROPIC_MODEL`` to use another.
"""

from __future__ import annotations

import os
import socket

import pytest
from deepagents import create_deep_agent
from langchain_anthropic import ChatAnthropic

from boxd_langchain import BoxdSandbox
from tests.integration_tests._live import (
    SAFETY_DELETE_AFTER,
    machine_labels,
    machine_name,
    requires_boxd,
)

pytestmark = [
    requires_boxd,
    pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set; skipping the agent run",
    ),
]

TASK = """\
Work in /home/boxd/project.
1. Write /home/boxd/project/primes.py: a script that prints the prime numbers
   below 50, comma-separated on one line.
2. Run it with python3 and save its output to /home/boxd/project/out.txt.
3. Run `uname -s` and `hostname` and tell me both.
4. Read out.txt back and tell me how many primes it lists.
"""


def _tool_calls(messages: list) -> list[str]:
    return [
        call["name"]
        for message in messages
        for call in getattr(message, "tool_calls", None) or []
    ]


def test_deep_agent_works_inside_a_boxd_machine() -> None:
    model = ChatAnthropic(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
        max_tokens=4096,
    )
    with BoxdSandbox.create(
        name=machine_name("agent"),
        labels=machine_labels(),
        delete_after=SAFETY_DELETE_AFTER,
    ) as backend:
        hostname = backend.execute("hostname").output.strip()
        agent = create_deep_agent(
            model=model,
            system_prompt="You are a careful coding assistant with a Linux sandbox.",
            backend=backend,
        )
        result = agent.invoke({"messages": [{"role": "user", "content": TASK}]})

        calls = _tool_calls(result["messages"])
        print("tool calls:", calls)  # noqa: T201
        assert "execute" in calls

        # The files the agent made are in the machine, not on this computer.
        [script, output] = backend.download_files(
            ["/home/boxd/project/primes.py", "/home/boxd/project/out.txt"]
        )
        assert script.error is None
        assert output.error is None
        assert output.content is not None
        primes = [int(p) for p in output.content.decode().replace(" ", "").split(",")]
        assert primes == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        # And the commands ran there too.
        assert hostname
        assert hostname != socket.gethostname()

        answer = result["messages"][-1].text
        assert "15" in answer
        assert "Linux" in answer
        assert hostname in answer

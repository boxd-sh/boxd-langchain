"""A Deep Agent that writes and runs code on a fresh boxd machine.

    BOXD_API_KEY=... ANTHROPIC_API_KEY=... python examples/quickstart.py

The machine is deleted when the `with` block ends. `delete_after` deletes it
anyway after an hour, in case the process dies first.
"""

from _model import model
from deepagents import create_deep_agent

from langchain_boxd import BoxdSandbox

with BoxdSandbox.create(delete_after=3600) as backend:
    print(f"machine {backend.id} is ready")
    agent = create_deep_agent(
        model=model(),
        system_prompt=(
            "You are a coding assistant with a Linux machine. Work in /home/boxd."
        ),
        backend=backend,
    )
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Write primes.py that prints the first 20 prime numbers, "
                        "run it, and tell me the output and which OS you are on."
                    ),
                }
            ]
        }
    )
    print(result["messages"][-1].text)

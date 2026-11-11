"""A named machine the agent comes back to: files, packages and history stay.

    python examples/persistent_workspace.py "Start notes.md with three CLI tool ideas."
    python examples/persistent_workspace.py "What is in notes.md? Add a fourth idea."
    python examples/persistent_workspace.py --delete

The first run creates the machine `langchain-workspace`. Every run ends by
hibernating it: memory and disk are kept and you stop paying for CPU and
memory. The next command wakes it.
"""

import sys

from _model import model
from boxd import Boxd, NotFoundError
from deepagents import create_deep_agent

from boxd_langchain import BoxdSandbox

NAME = "langchain-workspace"
client = Boxd()  # reads BOXD_API_KEY

if sys.argv[1:] == ["--delete"]:
    client.machines.delete(NAME)
    print(f"deleted {NAME}")
    sys.exit()

try:
    machine = client.machines.get(NAME)
    print(f"reusing {NAME} ({machine.status})")
except NotFoundError:
    machine = client.machines.create(NAME, labels={"app": "langchain-example"})
    client.machines.wait_until_ready(machine.id)
    print(f"created {NAME}")

# A backend made from an existing machine never deletes it.
backend = BoxdSandbox(client=client, machine_id=machine.id)
agent = create_deep_agent(
    model=model(),
    system_prompt="You keep a workspace in /home/boxd/work on a Linux machine.",
    backend=backend,
)
prompt = " ".join(sys.argv[1:]) or "List what is in /home/boxd/work."
result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
print(result["messages"][-1].text)

client.machines.hibernate(machine.id)
print(f"{NAME} is hibernated; run again to pick up where you left off")

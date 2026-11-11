"""Set a project up once, fork the machine, and let two agents try two approaches.

    python examples/parallel_forks.py

A fork is a live copy of a machine (disk, memory and running processes), so
neither agent repeats the setup, and neither can see what the other changes.
"""

import asyncio

from _model import model
from deepagents import create_deep_agent

from langchain_boxd import BoxdSandbox

SETUP = """
mkdir -p /home/boxd/project && cd /home/boxd/project
cat > words.txt <<'TXT'
the quick brown fox jumps over the lazy dog the fox
TXT
"""

APPROACHES = {
    "python": "Count how often each word occurs in /home/boxd/project/words.txt "
    "with a Python script. Run it and give the top 3.",
    "shell": "Count how often each word occurs in /home/boxd/project/words.txt "
    "with a shell pipeline (tr, sort, uniq). Run it and give the top 3.",
}


async def solve(backend: BoxdSandbox, task: str) -> str:
    agent = create_deep_agent(
        model=model(),
        system_prompt="You work on a Linux machine. Be brief.",
        backend=backend,
    )
    result = await agent.ainvoke({"messages": [{"role": "user", "content": task}]})
    return result["messages"][-1].text


async def main() -> None:
    with BoxdSandbox.create(delete_after=3600) as base:
        base.execute(SETUP)
        client = base.client
        forks = []
        try:
            for label in APPROACHES:
                fork = client.machines.fork(
                    base.id, labels={"approach": label}, delete_after=3600
                )
                client.machines.wait_until_ready(fork.id)
                forks.append(BoxdSandbox(client=client, machine_id=fork.id))
            answers = await asyncio.gather(
                *(
                    solve(f, task)
                    for f, task in zip(forks, APPROACHES.values(), strict=True)
                )
            )
            for label, answer in zip(APPROACHES, answers, strict=True):
                print(f"--- {label}\n{answer}\n")
        finally:
            for f in forks:  # a backend made from a machine id never deletes it
                client.machines.delete(f.id)


asyncio.run(main())

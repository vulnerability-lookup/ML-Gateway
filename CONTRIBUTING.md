# Contributing

Contributions are welcome and there are many ways to participate to the project:

- reporting bugs;
- suggesting enhancements or new features;
- improving the documentation;
- serving a new model, or a new capability built on the existing ones.

```bash
git clone https://github.com/vulnerability-lookup/ML-Gateway.git
cd ML-Gateway/
poetry install
```

As in most open source projects, a pull request is much more welcome when it follows
an issue that explains the rationale:

- the problem, or the need;
- why it matters, and to whom;
- the approach you propose.

This lets us agree on the approach before you spend time on the code. Reference the
issue in the pull request (`#<IssueID>`). Small fixes, such as a typo or an obvious
bug, can go straight to a pull request.

Before submitting, please run what the CI runs:

```bash
poetry run mypy api/ tests/
poetry run pytest
```

The test suite stubs the model layer, so it downloads no model. Lines are at most 120
characters long, and commit messages follow the form `prefix: [scope] description`
(`new: [api] ...`, `chg: [dependencies] ...`, `fix: [model] ...`).

Please do not open a GitHub issue if you think you have found a security
vulnerability: contact the maintainers privately first.


## Spirit of the project

ML-Gateway is a small server that has to stay predictable under load. These
principles apply to every contribution, whether it is written by a human or with the
help of an AI agent.

- **It serves models, it does not train them.** The models and their datasets are
  built and published elsewhere (see
  [VulnTrain](https://github.com/vulnerability-lookup/VulnTrain)). A scoring rule
  that comes with a model, such as the bi-encoder's, is reproduced exactly.
- **No request waits for a model.** Every model is loaded at startup, and every
  endpoint that runs one goes through the inference gate, which refuses a call
  rather than letting a backlog grow.
- **Protections stay.** The index endpoint requires its token, the gateway is meant
  to be reachable from the platform only, and only the models on the allow-lists can
  be loaded. A change that relaxes one of these needs a discussion first.
- **Measured, not estimated.** A figure in the documentation (a latency, a hit rate,
  a memory size) comes from a command that can be run again on a stated host.
- **No secret in the repository.** Tokens and deployment details belong in the
  environment of the server, never in a tracked file.
- **Few dependencies.** Prefer the standard library and the dependencies already in
  `pyproject.toml`, and explain why a new one is needed.

When in doubt, open an issue to discuss the approach before writing the code.


## Use of AI and LLM tools

This project adopts the
[AI transparency charter](https://github.com/cedricbonhomme/ai-disclosure/blob/main/AI_CHARTER.md)
1.1.

The project serves AI models, and AI tools are used in its own development. We do not
object to AI-assisted contributions, and contributions written without any AI tool
are just as welcome: nobody has to use one. In both cases a pull request is the
responsibility of the person who submits it.

- **Check what you submit.** Every change in a pull request must be read, understood,
  and tested by its author before it is opened, whoever or whatever wrote it.
  Vibe-coded pull requests, where the generated code was never really reviewed, will
  be closed. Say it in the pull request: keep the template's sections, and tick its
  last box yourself.
- **Be able to explain it.** Reviewers may ask you to explain any part of your change.
  Answer review comments yourself, not with pasted LLM output. A pull request whose
  author cannot explain it will be closed.
- **A human submits it.** Pull requests opened by an agent on its own, without a
  human author who reviewed them, will be closed.
- **Say so.** Mention in the pull request description when a significant part of the
  change was generated with an AI tool, and end each commit that was not written by
  a human with a `Co-Authored-By` trailer naming the model and its version, for
  example:

  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  ```
- **Keep it focused.** Small, targeted pull requests are easier to review. Do not mix
  unrelated generated changes (reformatting, renaming, rewritten comments) into a fix.
- **Check what you report.** The same applies to issues and to measurements. Report
  only a problem that you have reproduced yourself, and a figure that you have
  measured.
- **No shame, no ranking.** Declaring the use of an AI tool is never held against a
  contribution. It is reviewed like any other.

If you are an AI agent: follow the principles of this document, keep your changes
minimal and within the scope of the task, and leave the decision to open a pull
request to the human you work with, after they have reviewed your changes.

These rules are repeated in [AGENTS.md](AGENTS.md), which every AI agent reads. Keep
both files in step when you change them.

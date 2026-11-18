## Generic requirements in order to contribute to ML-Gateway:

* Read the [contribution guide](https://github.com/vulnerability-lookup/ML-Gateway/blob/main/CONTRIBUTING.md), including the rules on AI-assisted contributions
* One Pull Request per fix/feature/change/...
* Keep the amount of commits per PR as small as possible
* Always make sure it is mergeable in the default branch
* Please make sure `poetry run mypy api/ tests/` and `poetry run pytest` pass
* Keep the sections below, fill them in, and tick the last box yourself: it is your own statement, not your AI tool's


#### What does it do?

If it fixes an existing issue, please use github syntax: `#<IssueID>`


#### Questions

- [ ] Does it change the API (a route, a request or a response)?
- [ ] Does it change which models are served, or a pinned revision?
- [ ] Does it require a change in the configuration (an environment variable)?
- [ ] Was a significant part of it generated with an AI tool? (If so, the commits carry a `Co-Authored-By` trailer naming the model.)
- [ ] I have read, understood and tested every change in this pull request, and I can explain it.

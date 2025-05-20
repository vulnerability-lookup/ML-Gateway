# ML-Gateway

This project implements a FastAPI-based local server designed to load one or more pre-trained NLP models during startup and expose them through a clean, RESTful API for inference.

For example, it leverages the Hugging Face transformers library to load the
[CIRCL/vulnerability-severity-classification-distilbert-base-uncased model](https://huggingface.co/CIRCL/vulnerability-severity-classification-roberta-base),
which specializes in classifying vulnerability descriptions according to their severity level. The server initializes this model once at startup, ensuring minimal latency during inference requests.

Clients interact with the server via dedicated HTTP endpoints corresponding to each loaded model.
Additionally, the server automatically generates comprehensive OpenAPI documentation that details
the available endpoints, their expected input formats, and sample responses—making it easy to explore and integrate the services.

The ultimate goal is to enrich vulnerability data descriptions through the application of a suite of NLP models, providing direct benefits to Vulnerability-Lookup and supporting other related projects.


## Installation

```bash
git clone https://github.com/vulnerability-lookup/ML-Gateway
cd ML-Gateway/
poetry install
```

## Running the Server


```bash
poetry run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoint

### Example

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/classify/severity' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "description": "SAP NetWeaver Visual Composer Metadata Uploader is not protected with a proper authorization, allowing unauthenticated agent to upload potentially malicious executable binaries that could severely harm the host system. This could significantly affect the confidentiality, integrity, and availability of the targeted system."
}'
{"severity":"Critical","confidence":0.7021538019180298}
```


## Funding

[AIPITCH](https://www.linkedin.com/company/aipitch)
(AI-Powered Innovative Toolkit for Cybersecurity Hubs) is a co-funded EU project
supported by the European Cybersecurity Competence Centre (ECCC) under the
DIGITAL-ECCC-2024-DEPLOY-CYBER-06-ENABLINGTECH program and
[CIRCL](https://www.circl.lu).

The project brings together an international consortium to develop AI-based tools
that enhance the capabilities of operational cybersecurity teams.
These tools are designed to support critical services, with a focus on national
security teams, while also being applicable to internal security teams in
companies and institutions.


## License

[ML-Gateway](https://github.com/vulnerability-lookup/ML-Gateway) is licensed under
[GNU Affero General Public License version 3](https://www.gnu.org/licenses/agpl-3.0.html).

~~~
Copyright (c) 2025 Computer Incident Response Center Luxembourg (CIRCL)
Copyright (C) 2025 Cédric Bonhomme - https://github.com/cedricbonhomme
~~~


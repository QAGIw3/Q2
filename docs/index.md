# Q2 Docs

- API base: `/v1/compute`
- Health: `/healthz`
- Metrics: `/metrics`

Example:

POST /v1/compute/qvnn/train
Payload: {"n":4096,"in_dim":64,"hidden":128,"out_dim":1,"steps":200,"lr":0.001,"amp":true}
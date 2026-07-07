# n8n-runner-deploy: Kubernetes and DevSecOps Proof

The n8n-runner-deploy repository packages a self-hosted n8n + ffmpeg image as a Helm
chart (`charts/n8n-runner`) and proves production plumbing at zero cloud cost.

Runtime proof runs on a local kind cluster: the CI pipeline builds the image, pushes
it to GHCR, installs the Helm chart into kind, and smoke-tests the pod. A policy gate
written in Rego (`policy/kubernetes.rego`, rego.v1 syntax) validates the rendered
Kubernetes manifests.

Infrastructure is described with Terraform for EKS and ECR but kept plan-only: a real
GitHub Actions OIDC role (`n8n-runner-ci-plan`, ReadOnlyAccess) runs `terraform plan`
against AWS and reported "55 to add, 0 change, 0 destroy" without creating any billable
resource. The DevSecOps pipeline has seven green jobs including Trivy container
scanning, semgrep static analysis, and tfsec infrastructure scanning.

# Cloudflare Worker on the FREE plan: an edge endpoint that proxies the
# Hugging Face Space. Proves a real `terraform apply` at $0.
#
# Cost: Workers free plan — 100k requests/day, no card required.
# Destroy: `terraform destroy` removes the worker and its workers.dev route.

terraform {
  required_version = ">= 1.6"
  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 5.0"
    }
  }
}

variable "cloudflare_api_token" {
  type        = string
  sensitive   = true
  description = "API token with Workers Scripts:Edit permission"
}

variable "cloudflare_account_id" {
  type        = string
  description = "Cloudflare account id"
}

variable "upstream_url" {
  type        = string
  description = "Public origin to proxy (the Hugging Face Space URL)"
}

provider "cloudflare" {
  api_token = var.cloudflare_api_token
}

resource "cloudflare_workers_script" "rag_proxy" {
  account_id  = var.cloudflare_account_id
  script_name = "langgraph-rag-proxy"
  content     = templatefile("${path.module}/worker.js.tftpl", { upstream = var.upstream_url })
  main_module = "worker.js"
}

# Serve the worker on the free <script>.<account-subdomain>.workers.dev URL.
resource "cloudflare_workers_script_subdomain" "rag_proxy" {
  account_id  = var.cloudflare_account_id
  script_name = cloudflare_workers_script.rag_proxy.script_name
  enabled     = true
}

output "worker_script" {
  value = cloudflare_workers_script.rag_proxy.script_name
}

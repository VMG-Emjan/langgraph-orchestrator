# B2B Email Outreach Pipeline

The email outreach pipeline is an n8n workflow for B2B lead outreach. It reads a
prospect list from a spreadsheet, personalizes an email per company, and sends
through SMTP with rate limiting and an explicit DRY_RUN gate: no real email leaves
the system until the operator flips the flag after reviewing generated drafts.

Lead enrichment comes from a web-recon step that visits each company website and
extracts contact details and whether the company ships AI products. The pipeline
runs on a custom n8n Docker image (n8n-ffmpeg-py) that bundles Python with openpyxl
for Excel handling, and logs every send decision for auditability.

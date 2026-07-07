# Animal Shorts Video Pipeline

The animal shorts pipeline is an n8n workflow that assembles short vertical videos
automatically. A dedicated ffmpeg-runner Docker sidecar renders the clips so the n8n
main container stays lean.

The original implementation blocked the n8n event loop by calling ffmpeg through
execSync inside a Code node; the fix moved rendering into the ffmpeg-runner container
invoked over HTTP, which unblocked concurrent workflow executions. The pipeline
stitches source clips, background music and captions into a finished short, then
uploads the result. Workflow state and asset paths flow between nodes as JSON items.

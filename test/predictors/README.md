# Mock predictors

Signature-only `@compile` Python stubs for the six deployment example modalities:

| Source | Interface |
| --- | --- |
| `completions.py` | OpenAI chat completions; also compatible with Anthropic messages |
| `embeddings.py` | OpenAI embeddings |
| `images.py` | OpenAI image generation |
| `edit.py` | OpenAI image editing |
| `depth.py` | Depth estimation through OpenAI image edits |
| `transcriptions.py` | OpenAI audio transcriptions |

Each file declares one predictor with the parameter types and annotations needed
for adapter detection. Tags are omitted so the CLI can use the signed-in user's
namespace. No weights or model dependencies are required.

Bodies intentionally raise `NotImplementedError`; these define API signatures,
not working inference models. Chat declares the streaming chunk return type.

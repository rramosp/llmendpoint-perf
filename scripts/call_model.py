import json
import os
import sys
import urllib.error
import urllib.request


DEFAULT_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openrouter/free"

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_MODEL = "gemini-2.5-flash"

DEFAULT_URL=GEMINI_URL
DEFAULT_MODEL=GEMINI_MODEL

def call_model(
    prompt: str, 
    api_key: str | None = None, 
    url: str | None = None,
    model: str | None = None
) -> str:
    """Calls the REST API using only Python standard library HTTP modules."""
    if api_key is None:
        api_key = os.environ.get("OPENAI_COMPATIBLE_API_KEY")
    if api_key:
        api_key = api_key.strip().strip("'\"")
    if not api_key:
        raise ValueError(
            "API key must be provided or set in OPENAI_COMPATIBLE_API_KEY environment variable."
        )

    if not url:
        url = os.environ.get("OPENAI_COMPATIBLE_URL") or DEFAULT_URL
    if not url.endswith("chat/completions"):
        url = url.rstrip("/") + "/chat/completions"

    if not model:
        model = os.environ.get("MODEL") or DEFAULT_MODEL

    print(f"Using URL:   {url}")
    print(f"Using Model: {model}\n")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP error {e.code} calling model {model} on url {url}: {error_body}") from e

    try:
        return result["choices"][0]["message"]["content"]   

    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected response format from OpenAI API: {result}") from e


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("OPENAI_COMPATIBLE_API_KEY")
    if not api_key:
        print(
            "Error: OPENAI_COMPATIBLE_API_KEY environment variable not set.\n"
            "Please set it before running:\n"
            "  export OPENAI_COMPATIBLE_API_KEY='your_key_here'",
                file=sys.stderr,
        )
        sys.exit(1)

    prompt_text = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Explain quantum computing in one short paragraph."
    )
    print(f"Prompt: {prompt_text}\n")
    response = call_model(prompt_text, api_key=api_key)
    print("\nResponse:")
    print(response)

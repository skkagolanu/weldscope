# Examples

Legacy demo scripts showing progressive RAG implementations.

## Files

- `rag_basic.py` - Simple document Q&A
- `rag_multidoc.py` - Multi-document filtering with metadata
- `rag_sensorfusion.py` - Sensor data + document integration

## Usage

These are standalone demos using local Ollama:

```bash
# Install Ollama
ollama pull llama3.2:1b

# Run demos
python examples/rag_basic.py
python examples/rag_multidoc.py
python examples/rag_sensorfusion.py
```

## Note

For production use, see the main application (`app.py`) which uses the modular `weldscope` package.

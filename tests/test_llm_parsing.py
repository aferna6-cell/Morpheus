from src.signals.llm_signal import LLMSignal


def test_extract_json_object_plain():
    s = '{"probability": 0.5, "confidence": 0.7, "reasoning": "x"}'
    assert LLMSignal._extract_json_object(s) == s


def test_extract_json_object_code_fence():
    s = """Here you go:
```json
{"probability": 0.1, "confidence": 0.2, "reasoning": "ok"}
```
"""
    assert LLMSignal._extract_json_object(s).startswith("{")
    assert "probability" in LLMSignal._extract_json_object(s)


def test_extract_json_object_embedded_text():
    s = 'prefix {"probability": 0.9, "confidence": 0.8, "reasoning": "y"} suffix'
    assert (
        LLMSignal._extract_json_object(s)
        == '{"probability": 0.9, "confidence": 0.8, "reasoning": "y"}'
    )

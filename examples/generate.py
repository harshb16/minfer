from minfer import LLMEngine, SamplingParams

engine = LLMEngine()
result = engine.generate("Explain why KV caching helps.", SamplingParams(max_new_tokens=64))
print(result.text)

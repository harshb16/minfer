"""Submit a new request after older requests have begun decoding."""

from minfer import LLMEngine, SamplingParams

engine = LLMEngine(max_active_requests=3)
params = SamplingParams(max_new_tokens=32)
engine.add_request("Explain prefill.", params, request_id="A")
engine.add_request("Explain continuous decode batching in two sentences.", params, request_id="B")
iteration = 0
while engine.has_unfinished_requests():
    if iteration == 2:
        engine.add_request("Explain the role of position IDs.", params, request_id="C")
    for event in engine.step():
        print(f"{iteration:02d} {event.request_id}: {event.token_text!r} finished={event.finished}")
    iteration += 1

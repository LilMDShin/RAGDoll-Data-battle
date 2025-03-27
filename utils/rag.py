def rag_stream(model_id, client_inference, conv_history):
    stream = client_inference.chat.completions.create(
        model=model_id,
        messages=conv_history,
        max_tokens=550,
        temperature=0,
        top_p=0.99,
        stream=True  # Enable streaming mode if supported
    )
    full_response = ""
    for token in stream:
        # Try to get the delta value
        delta = token["choices"][0].get("delta")
        # If delta is a dict, extract the "content"; otherwise, use it directly or fallback to ""
        if isinstance(delta, dict):
            token_text = delta.get("content") or ""
        else:
            token_text = delta or ""
        full_response += token_text
        yield full_response

def rag(model_id, client_inference, conv_history):
    completion = client_inference.chat.completions.create(
        model=model_id,
        messages=conv_history,
        max_tokens=550,
        temperature=0,
        # top_p=1
        top_p=0.99
    )
    model_res = completion.choices[0].message.content
    return(model_res)
def ctc_greedy_decode(logits, blank_idx:int=0):
    # logits - [B, W, C]
    pred_ids = logits.argmax(dim=-1)
    results = []
    for batch in pred_ids:
        batch = batch.tolist()
        collapsed = []
        prev = None
        for idx in batch:
            if idx != prev:
                collapsed.append(idx)
            prev = idx
        decoded = [i for i in collapsed if i != blank_idx]
        results.append(decoded)
    return results # [B, T(idx)]

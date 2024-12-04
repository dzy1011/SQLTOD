import string
# def response_generation(data, tokenizer, model, batch_size, beam_size):
#     """Generate system response by dialogue context and retrieved knowledge."""
#     # prepare input samples
#     samples = []
#     for dial in data:
#         excluded_fields = ["id", "location", "type"] if dial["scenario"]["kb"]["kb_title"] == "camrest676" else []
#         fields = [name for name in dial["scenario"]["kb"]["column_names"] if name not in excluded_fields]
#         context = []

#         for turn in dial["dialogue"]:
#             if turn["turn"] == "system":
#                 retrieved_knowledge_seq = linearize_knowledge(turn["retrieved_knowledge"], fields)
#                 src = "generate system response based on knowledge and dialogue context : knowledge : " + \
#                     retrieved_knowledge_seq + " ; dialogue context : " + " | ".join(context)
#                 samples.append(src)
#             utt = preprocess_text(turn["utterance"])
#             context.append(utt)

#     # call Q-TOD model
#     generated_responses = generate(samples, tokenizer, model, batch_size, beam_size)

#     # save generated response into data
#     sample_idx = 0
#     for dial in data:
#         for turn in dial["dialogue"]:
#             if turn["turn"] == "system":
#                 turn["generated_response"] = generated_responses[sample_idx]
#                 sample_idx += 1

#     return


def preprocess_text(text):
    """Preprocess utterance and table value."""
    text = text.strip().replace("\t", " ").lower()
    for p in string.punctuation:
        text = text.replace(p, f" {p} ")
    text = " ".join(text.split())
    return text


def linearize_knowledge_record(knowledge_record, fields):
    """Convert a knowledge record into a flatten sequence with special symbols."""
    knowledge_seq = []
    for f in fields:
        value = preprocess_text(str(knowledge_record.get(f, "")))
        knowledge_seq.append(f.replace("_", " ") + " : " + value)
    return " | ".join(knowledge_seq)


def linearize_knowledge(knowledge, fields):
    """Convert knowledge into a flatten sequecen with special symbols."""
    knowledge_seq = []
    knowledge_seq.append("col : " + " | ".join(map(lambda x: x.replace("_", " "), fields)))
    for idx, record in enumerate(knowledge):
        values = []
        for f in fields:
            v = preprocess_text(str(record.get(f, "")))
            values.append(v)

        record_seq = " | ".join(values)
        knowledge_seq.append(f"row {idx} : {record_seq}")
    return " || ".join(knowledge_seq)
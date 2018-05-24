# Text sequence to int encoded
def text_to_ints(text):
    text = clean_text(text)
    return [vocab_to_int[word] for word in text]

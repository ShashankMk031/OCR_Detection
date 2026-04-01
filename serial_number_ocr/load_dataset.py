from datasets import load_dataset


def load_ocr_datasets():
    synth = load_dataset("wendlerc/CaptionedSynthText")
    return synth

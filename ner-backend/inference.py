import os
import numpy as np
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import nltk
nltk.download('punkt')

template_map = {
    'LOC': lambda e: f'{e} is a location entity',
    'PER': lambda e: f'{e} is a person entity',
    'ORG': lambda e: f'{e} is an organization entity',
    'MISC': lambda e: f'{e} is an other entity',
    'NOT': lambda e: f'{e} is not a named entity'
}
entity_types = ['LOC', 'PER', 'ORG', 'MISC', 'NOT']
num_templates = len(entity_types)

best_model_save_dir = os.path.join(os.path.dirname(__file__), 'best_model')

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Loading pretrained model...')
model = BartForConditionalGeneration.from_pretrained(best_model_save_dir)
tokenizer = BartTokenizer.from_pretrained(best_model_save_dir)
model.to(device)
print('Loaded!')


def tokenize_paragraph(paragraph: str) -> list[str]:
    word_tokens = []
    sentences = nltk.tokenize.sent_tokenize(paragraph)
    for sentence in sentences:
        word_tokens.append(
            nltk.tokenize.word_tokenize(sentence)
        )

    return word_tokens


def gen_all_sections(word_tokens: list[str]) -> list[list[tuple]]:
    # [
    #   (start, end, section)
    #   ...
    # ]
    all_sections = []

    for i in range(len(word_tokens)):
        section_at_i = []
        for j in range(1, min(9, len(word_tokens) - i + 1)):  # limit seconds to 8 tokens
            section = ' '.join(word_tokens[i: i + j])
            section_at_i.append((i, i+j, section))
        all_sections.append(section_at_i)
    return all_sections


def identify_section(section: str, sentence: str) -> str:
    templates = []
    for entity in entity_types:
        templates.append(template_map[entity](section))

    # Tokenize sentence and templates
    input_ids = tokenizer(
        [sentence] * num_templates,
        return_tensors='pt'
    )['input_ids']
    output_ids = tokenizer(
        templates,
        return_tensors='pt',
        padding=True,
        truncation=True
    )['input_ids']

    output_ids[:, 0] = 2  # add eos before each sentence
    # Remove padding (cuts off entity of NOT template but should be fine)
    output_ids = output_ids[:, :output_ids.shape[1] - 2]

    # Run model
    output = model(
        input_ids=input_ids.to(device),
        decoder_input_ids=output_ids.to(device)
    )[0]

    score = np.ones(num_templates)
    # Skip last 1 token for all templates
    for i in range(output_ids.shape[1] - 1):
        logits = output[:, i, :]
        logits = logits.softmax(dim=1)
        logits = logits.to('cpu').numpy()
        for j in range(num_templates):
            # Skip last 2 tokens for all templates but NOT (skip padding)
            if j == 4 or i < output_ids.shape[1]:
                # Score = product of probability of each word in template
                # i + 1 to skip first token (padding)
                score[j] = score[j] * logits[j][int(output_ids[j][i + 1])]

    return entity_types[np.argmax(score)], np.max(score)


def entity_list_to_output(entity_list: list[tuple[int, int, str]], sentence_len: int) -> list[str]:
    output = ['O'] * sentence_len

    for i in range(len(entity_list) - 1, -1, -1):
        entity = entity_list[i]
        output[entity[0]:entity[1]] = [
            f'I-{entity[2]}'] * (entity[1] - entity[0])
        output[entity[0]] = f'B-{entity[2]}'

    return output


def run_ner_sentence_tokens(sentence_tokens: list[str]) -> tuple[list[tuple[int, int, str]], list[str]]:
    with torch.no_grad():
        sections = gen_all_sections(sentence_tokens)

        # [
        #   (start, end, entity_type)
        # ]
        entities_list = []
        for i, section_at_i in enumerate(sections):
            max_score = 0
            best_entity = 'NOT'
            best_end = i
            for section in section_at_i:
                entity, score = identify_section(
                    section[2], ' '.join(sentence_tokens))
                if (max_score < score):
                    max_score = score
                    best_entity = entity
                    best_end = section[1]

            if best_entity != 'NOT':
                entities_list.append((i, best_end, best_entity))

    return entity_list_to_output(entities_list, len(sentence_tokens))


def run_ner_paragraph(paragraph: str):
    tokens = tokenize_paragraph(paragraph)

    all_tokens = []
    labels = []
    for sentence_tokens in tokens:
        pred = run_ner_sentence_tokens(sentence_tokens)
        labels.extend(pred)
        all_tokens.extend(sentence_tokens)

    return all_tokens, labels

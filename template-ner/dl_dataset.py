import os
import urllib.request
import zipfile
import random
import pandas as pd

random.seed(4)
negative_pair_ratio = 1.5
template_map = {
    'LOC': lambda e: f'{e} is a location entity',
    'PER': lambda e: f'{e} is a person entity',
    'ORG': lambda e: f'{e} is an organization entity',
    'MISC': lambda e: f'{e} is an other entity',
    'NOT': lambda e: f'{e} is not a named entity'
}


def main():
    __dirname = os.path.dirname(__file__)
    dataset_dir = os.path.join(__dirname, 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)

    print('Downloading dataset')
    dl_zip(dataset_dir)

    print('Parsing train dataset')
    parseDataset(dataset_dir, 'train')

    print('Parsing validation dataset')
    parseDataset(dataset_dir, 'valid')


def dl_zip(dataset_dir: os.PathLike):
    url = "https://data.deepai.org/conll2003.zip"
    filehandle, _ = urllib.request.urlretrieve(url)
    zip_file = zipfile.ZipFile(filehandle, 'r')

    for file in zip_file.namelist():
        content = zip_file.open(file).read()

        with open(os.path.join(dataset_dir, file), 'w', encoding='utf-8') as f:
            f.write(content.decode('utf-8'))


def parseDataset(dataset_dir: os.PathLike, split_name: str):
    # Parse out sentences and their ner tag sequences
    sentences = []
    ner_tags = []

    tokens = []
    tags = []
    with open(os.path.join(dataset_dir, split_name + '.txt')) as f:
        for line in f:
            if line.startswith('-DOCSTART-') or line == '' or line == '\n':
                if len(tokens) != 0:
                    sentences.append(tokens)
                    ner_tags.append(tags)
                tokens = []
                tags = []
            else:
                spilt = line.split(" ")
                tokens.append(spilt[0].strip())
                tags.append(spilt[3].strip())

        if len(tokens) != 0:
            sentences.append(tokens)
            ner_tags.append(tags)

    # For each sentence, generate templates for that sentence
    all_sentences = []
    all_templates = []
    for sentence, tags in zip(sentences, ner_tags):
        templates = genTemplates(sentence, tags)

        all_sentences.extend([' '.join(sentence)] * len(templates))
        all_templates.extend(templates)

    # Save templates as csv
    pd.DataFrame({
        'Source sentence': all_sentences,
        'Answer sentence': all_templates
    }).to_csv(os.path.join(dataset_dir, split_name + '.csv'), index=False)


def genTemplates(sentence: list[str], ner_tags: list[str]):
    # Parse out entities and their types from sentence
    entities = []
    entity_types = []

    entity = []
    entity_type = ''
    for i in range(len(sentence)):
        ner_tag_parts = ner_tags[i].split('-')
        if ner_tag_parts[0] == 'B' or ner_tag_parts[0] == 'O':
            if len(entity) != 0:
                entities.append(' '.join(entity))
                entity_types.append(entity_type)
            entity = []
            entity_type = ''

        if ner_tag_parts[0] != 'O':
            entity.append(sentence[i])
            entity_type = ner_tag_parts[1]

    if len(entity) != 0:
        entities.append(' '.join(entity))
        entity_types.append(entity_type)

    # Count the number of named entities
    count = 0
    for tag in ner_tags:
        if tag != 'O':
            count += 1

    # Generate negative entities
    # Make sure not to generate any if no negative examples exist
    negative_entities = []
    for i in range(min(len(sentence) - count, int(len(entities) * negative_pair_ratio))):
        negative_entities.append(getRandomNonEntity(sentence, ner_tags))

    # Convert entity types to template sentences
    templates = []
    for i in range(len(entities)):
        templates.append(
            template_map[entity_types[i]](entities[i])
        )

    for e in negative_entities:
        templates.append(
            template_map['NOT'](e)
        )

    return templates


def getRandomNonEntity(sentence: list[str], ner_tags: list[str]):
    # Randomly generate numbers until index is not a named entity
    i = random.randint(0, len(ner_tags) - 1)
    while ner_tags[i] != 'O':
        i = random.randint(0, len(ner_tags) - 1)
    return sentence[i]


main()

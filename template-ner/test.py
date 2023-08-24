import os
from inference import run_ner_sentence_tokens


def main():
    __dirname = os.path.dirname(__file__)
    dataset_dir = os.path.join(__dirname, 'dataset')
    output_dir = os.path.join(__dirname, 'test_output')

    os.makedirs(output_dir, exist_ok=True)

    sentences, gold = parseDataset(dataset_dir, 'test')

    with open(os.path.join(output_dir, 'pred.txt'), 'w') as p, open(os.path.join(output_dir, 'gold.txt'), 'w') as g:
        for i, (sentence, tags) in enumerate(zip(sentences, gold)):
            print(f'\n{i}/{len(sentences)}')
            print(' '.join(sentence))

            prediction = run_ner_sentence_tokens(sentence)

            print('Prediction: ', prediction)
            print('Gold:       ', tags)

            p.write(' '.join(prediction) + '\n')
            g.write(' '.join(tags) + '\n')

    with open(os.path.join(output_dir, 'pred.txt')) as p:
        prediction = p.read().replace('\n', ' ').split(' ')
        prediction = get_entity_list(prediction)
    with open(os.path.join(output_dir, 'gold.txt')) as g:
        trues = g.read().replace('\n', ' ').split(' ')
        trues = get_entity_list(trues)

    prec, recall, score = calc_f1(trues, prediction)
    print('Precision', prec)
    print('Recall', recall)
    print('F1 Score', score)

    with open(os.path.join(output_dir, 'f1.txt'), 'w') as f:
        f.write(f'Precision = {prec}')
        f.write(f'Recall = {recall}')
        f.write(f'F1 Score = {score}')


def calc_f1(trues, pred):
    nb_correct = len(trues & pred)
    nb_pred = len(pred)
    nb_true = len(trues)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0

    score = 2 * p * r / (p + r) if p + r > 0 else 0

    return p, r, score


def get_entity_list(pred: list[str]) -> list[tuple[int, int, str]]:
    entity_list = []
    start = -1
    end = -1
    tag_type = ''

    for i in range(len(pred)):
        tag_parts = pred[i].split('-')
        if tag_parts[0] == 'B':
            start = i
            tag_type = tag_parts[1]

        if tag_parts[0] == 'O':
            if (start != -1):
                end = i
                entity_list.append(
                    (start, end, tag_type)
                )
                start = -1
                end = -1
                tag_type = ''
    if (start != -1):
        end = len(pred)
        entity_list.append(
            (start, end, tag_type)
        )

    return set(entity_list)


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

    return sentences, ner_tags


main()

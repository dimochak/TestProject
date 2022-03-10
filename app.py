from flask import Flask, jsonify, request, render_template
from flair.data import Sentence
from flair.models import SequenceTagger

import en_core_web_sm

from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.trainers import ModelTrainer
from flair.data import Corpus
from flair.datasets import ColumnCorpus

app = Flask(__name__)
tagger = SequenceTagger.load('ner')
nlp = en_core_web_sm.load()


def train_custom_flair_model():
    # 1. get the corpus
    columns = {0: 'text', 1: 'ner'}
    path_to_data = 'data/'
    corpus: Corpus = ColumnCorpus(path_to_data, columns,
                                  train_file='train.txt',
                                  dev_file='dev.txt')
    print(corpus)

    # 2. what label do we want to predict?
    label_type = 'ner'

    # 3. make the label dictionary from the corpus
    label_dict = corpus.make_label_dictionary(label_type=label_type)
    print(label_dict)

    # 4. initialize embedding stack with Flair and GloVe
    embedding_types = [
        WordEmbeddings('glove'),
        FlairEmbeddings('news-forward'),
        FlairEmbeddings('news-backward'),
    ]

    embeddings = StackedEmbeddings(embeddings=embedding_types)

    # 5. initialize sequence tagger
    tagger = SequenceTagger(hidden_size=256,
                            embeddings=embeddings,
                            tag_dictionary=label_dict,
                            tag_type=label_type,
                            use_crf=True)

    # 6. initialize trainer
    trainer = ModelTrainer(tagger, corpus)

    # 7. start training
    trainer.train('resources/taggers/sota-ner-flair',
                  learning_rate=0.1,
                  mini_batch_size=32,
                  max_epochs=150)

    print('Model is trained')


def evaluate_custom_model(text):
    # load the model you trained
    model = SequenceTagger.load('resources/taggers/sota-ner-flair/final-model.pt')

    # create example sentence
    sentence = Sentence(text)

    # predict tags and print
    model.predict(sentence)

    print(sentence.to_tagged_string())
    return sentence.to_tagged_string()


train_custom_flair_model()


@app.route('/')
def my_form():
    return render_template('my-form.html')


@app.route('/', methods=['POST'])
def analyzeNER():
    message = request.form['text']

    # 1. Flair NER
    sentence_flair = Sentence(message)
    tagger.predict(sentence_flair)

    # 2. SpaCy NER
    entities = nlp(message)

    spacy_dict = {}
    for ent in entities.ents:
        spacy_dict[ent.text] = ent.label_

    # 3. Custom entity recognition, using Flair embeddings
    custom_sentence_result = evaluate_custom_model(message)

    response = {'result_flair': sentence_flair.to_tagged_string(),
                'result_spacy': spacy_dict,
                'result_custom': custom_sentence_result
                }
    return jsonify(response), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0')

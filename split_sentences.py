import string
import json
import nltk
from tqdm import tqdm
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import PorterStemmer
nltk.download("punkt")

ps = PorterStemmer()


with open("mapping.json") as f:
    data = json.load(f)

with open("class_list.json") as f_classes:
    segmentation_classes = json.load(f_classes)["categories"]

segmentation_classes = [d["name"] for d in segmentation_classes]

sent_data = []

for entry in tqdm(data):
    text = entry["text"]


    sent_spans = list(PunktSentenceTokenizer().span_tokenize(text))


    idx_lookup = dict()

    j = 0
    for i in range(len(text)):
        if i < sent_spans[j][1]:
            idx_lookup[i] = j
        else:
            j += 1

    sent_labels = [(text[slice(*span)], set()) for span in sent_spans]

    for label in entry["linked"]:
        start_sent = idx_lookup[label["start"]]
        end_sent = idx_lookup[label["end"] - 1]

        for i in range(start_sent, end_sent + 1):
            sent_labels[i][1].update(label["seg_labels"])


    for i, span in enumerate(sent_spans):
        sentence = text[slice(*span)]
        word_spans = list(WhitespaceTokenizer().span_tokenize(sentence))
        words = [sentence[slice(*w)].translate(str.maketrans('', '', string.punctuation)).lower() for w in word_spans]
        word_stems = [ps.stem(w) for w in words]
        words = set(words + word_stems)
        

        word_matches = [seg_class for seg_class in segmentation_classes if seg_class.lower() in words]
        sent_labels[i][1].update(word_matches)

        

    for i in range(len(sent_labels)):
        sent_labels[i] = (sent_labels[i][0], list(sent_labels[i][1]))
    sent_data.append({
        "text": entry["text"],
        "id": entry["id"],
        "sentences": sent_labels
        })

with open("mapping_sentences.json", "w") as f_w:
    json.dump(sent_data, f_w)

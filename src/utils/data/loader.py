import os
import nltk
import json
import torch
import pickle
import logging
import numpy as np
from tqdm.auto import tqdm
from src.utils import config
import torch.utils.data as data
from src.utils.common import save_config
from nltk.corpus import wordnet, stopwords
from src.utils.constants import DATA_FILES
from src.utils.constants import EMO_MAP as emo_map
from src.utils.constants import WORD_PAIRS as word_pairs
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


relations = ["xIntent", "xNeed", "xWant", "xEffect", "xReact"]
emotion_lexicon = json.load(open("data/NRCDict.json"))[0]
"""
emotion_lexicon の中身...配列．[0]が配列で[1]が辞書．lexicon: 辞書
[
    ["語1", "語2", ...],
    {
        "語1": 0 or 1 or 2,
        "語2": 0 or 1 or 2,
        ...
    }
]
"""
stop_words = stopwords.words("english")


class Lang:
    """
    quated:
        load_dataset(),
    usage:
        インスタンスに与えられた引数から，語彙とインデックスの辞書を作成
    """
    def __init__(self, init_index2word):
        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.index2word = init_index2word
        self.n_words = len(init_index2word)

    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word.strip())

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def get_wordnet_pos(tag):
    """
    quated:
        encode_ctx()
    usage:
        語彙の品詞を判定
    """
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return None


def process_sent(sentence):
    """
    quated:
        get_commonsense(), encode_ctx(), encode()
    usage:
        ワードペア(src.utils.constants WORD_PAIRS)で文を置換した後，tokenizeする．
    """
    sentence = sentence.lower() # str.lower(): 文字列を小文字に変換したものを返す
    for k, v in word_pairs.items():
        sentence = sentence.replace(k, v)   # str.replace('a', 'b'): 文字列中の'a'を'b'に置換する
    sentence = nltk.word_tokenize(sentence)
    return sentence


def get_commonsense(comet, item, data_dict):
    """
    quated:
        encode_ctx()
    usage:
        Commetに文を投入し，各relationに対応する生成を得たのち，data_dictに追加する
        参考: data_dict ... in encode()
            data_dict = {
                "context": [],
                "target": [],
                "emotion": [],
                "situation": [],
                "emotion_context": [],
                "utt_cs": [],
            }
    """
    cs_list = []
    input_event = " ".join(item)
    for rel in relations:
        cs_res = comet.generate(input_event, rel)
        cs_res = [process_sent(item) for item in cs_res]
        cs_list.append(cs_res)

    data_dict["utt_cs"].append(cs_list)


def encode_ctx(vocab, items, data_dict, comet):
    """
    quated:
        encode()
    usage:
        データの中身をトークナイズして，各語の品詞を判定した後，辞書にdata_dictに加える
        参考: items はデータファイルの中の文章で，1文or複数文が配列になったタプル
            items = files[i]
            files = [
                (['hoge huga']),
                (['hoge huga', 'hoge huga', 'foo bar']),
                ...
            ]

    """
    for ctx in tqdm(items):
        ctx_list = []
        e_list = []
        for i, c in enumerate(ctx):
            item = process_sent(c)
            ctx_list.append(item)
            vocab.index_words(item)
            ws_pos = nltk.pos_tag(item)  # pos  ←Part of Speech: PoS
            """
            品詞タグを取得．配列の中のタプル
                ws_pos ... [('Hi', 'NNP'), (',', ','), ('I', 'PRP), ...]
            """
            for w in ws_pos:
                w_p = get_wordnet_pos(w[1])
                if w[0] not in stop_words and (
                    w_p == wordnet.ADJ or w[0] in emotion_lexicon
                ):
                    e_list.append(w[0])
            if i == len(ctx) - 1:
                get_commonsense(comet, item, data_dict)

        data_dict["context"].append(ctx_list)
        data_dict["emotion_context"].append(e_list)


def encode(vocab, files):
    """
    quated:
        read_file()
    usage:
        data_dict 辞書の中身を満たしていく
    """
    from src.utils.comet import Comet

    data_dict = {
        "context": [],
        "target": [],
        "emotion": [],
        "situation": [],
        "emotion_context": [],
        "utt_cs": [],
    }
    comet = Comet("data/Comet", config.device)

    for i, k in enumerate(data_dict.keys()):
        items = files[i]
        if k == "context":
            encode_ctx(vocab, items, data_dict, comet)
        elif k == "emotion":
            data_dict[k] = items
        else:
            for item in tqdm(items):
                item = process_sent(item)
                data_dict[k].append(item)
                vocab.index_words(item)
        if i == 3:
            break
    assert (
        len(data_dict["context"])
        == len(data_dict["target"])
        == len(data_dict["emotion"])
        == len(data_dict["situation"])
        == len(data_dict["emotion_context"])
        == len(data_dict["utt_cs"])
    )

    return data_dict


def read_files(vocab):
    """
    quated:
        load_dataset()
    usage:
        ファイルの中身を元にencode()で作成した辞書を返す．train, dev, testの3つ分．
    """
    files = DATA_FILES(config.data_dir)
    train_files = [np.load(f, allow_pickle=True) for f in files["train"]]
    dev_files = [np.load(f, allow_pickle=True) for f in files["dev"]]
    test_files = [np.load(f, allow_pickle=True) for f in files["test"]]

    data_train = encode(vocab, train_files)
    data_dev = encode(vocab, dev_files)
    data_test = encode(vocab, test_files)

    return data_train, data_dev, data_test, vocab


def load_dataset():
    """
    quated:
        prepare_data_seq()
    usage:
        キャッシュファイルがあればそれを読み込む．なければ新規で読み込んでキャッシュを作成する．
        読み込んだファイルの中身を返す(4つの値)
    """
    data_dir = config.data_dir
    cache_file = f"{data_dir}/dataset_preproc.p"
    if os.path.exists(cache_file):
        print("LOADING empathetic_dialogue")
        with open(cache_file, "rb") as f:
            [data_tra, data_val, data_tst, vocab] = pickle.load(f)
    else:
        print("Building dataset...")
        data_tra, data_val, data_tst, vocab = read_files(
            vocab=Lang(
                {
                    config.UNK_idx: "UNK",
                    config.PAD_idx: "PAD",
                    config.EOS_idx: "EOS",
                    config.SOS_idx: "SOS",
                    config.USR_idx: "USR",
                    config.SYS_idx: "SYS",
                    config.CLS_idx: "CLS",
                }
            )
        )
        with open(cache_file, "wb") as f:
            pickle.dump([data_tra, data_val, data_tst, vocab], f)
            print("Saved PICKLE")

    for i in range(3):
        print("[situation]:", " ".join(data_tra["situation"][i]))
        print("[emotion]:", data_tra["emotion"][i])
        print("[context]:", [" ".join(u) for u in data_tra["context"][i]])
        print("[target]:", " ".join(data_tra["target"][i]))
        print(" ")
    return data_tra, data_val, data_tst, vocab


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    """
    Pytorchではデータの組([入力データ, ラベル])を返すDatasetを
    torch.utils.data.DataLoader()に渡す必要がある
    そのためのもの．
    自前のデータを使用する場合，__len__()と__getitem__()を定義する必要がある．
    """

    def __init__(self, data, vocab):
        """Reads source and target sequences from txt files."""
        self.vocab = vocab
        self.data = data
        self.emo_map = emo_map
        self.analyzer = SentimentIntensityAnalyzer()

    def __len__(self):
        return len(self.data["target"])

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = {}
        item["context_text"] = self.data["context"][index]
        item["situation_text"] = self.data["situation"][index]
        item["target_text"] = self.data["target"][index]
        item["emotion_text"] = self.data["emotion"][index]
        item["emotion_context"] = self.data["emotion_context"][index]

        item["context_emotion_scores"] = self.analyzer.polarity_scores(
            " ".join(self.data["context"][index][0])
        )

        item["context"], item["context_mask"] = self.preprocess(item["context_text"])
        item["target"] = self.preprocess(item["target_text"], anw=True)
        item["emotion"], item["emotion_label"] = self.preprocess_emo(
            item["emotion_text"], self.emo_map
        )
        (
            item["emotion_context"],
            item["emotion_context_mask"],
        ) = self.preprocess(item["emotion_context"])

        item["cs_text"] = self.data["utt_cs"][index]
        item["x_intent_txt"] = item["cs_text"][0]
        item["x_need_txt"] = item["cs_text"][1]
        item["x_want_txt"] = item["cs_text"][2]
        item["x_effect_txt"] = item["cs_text"][3]
        item["x_react_txt"] = item["cs_text"][4]

        item["x_intent"] = self.preprocess(item["x_intent_txt"], cs=True)
        item["x_need"] = self.preprocess(item["x_need_txt"], cs=True)
        item["x_want"] = self.preprocess(item["x_want_txt"], cs=True)
        item["x_effect"] = self.preprocess(item["x_effect_txt"], cs=True)
        item["x_react"] = self.preprocess(item["x_react_txt"], cs="react")

        return item

    def preprocess(self, arr, anw=False, cs=None, emo=False):
        """Converts words to ids."""
        if anw:
            sequence = [
                self.vocab.word2index[word]
                if word in self.vocab.word2index
                else config.UNK_idx
                for word in arr
            ] + [config.EOS_idx]

            return torch.LongTensor(sequence)
        elif cs:
            sequence = [config.CLS_idx] if cs != "react" else []
            for sent in arr:
                sequence += [
                    self.vocab.word2index[word]
                    for word in sent
                    if word in self.vocab.word2index and word not in ["to", "none"]
                ]

            return torch.LongTensor(sequence)
        elif emo:
            x_emo = [config.CLS_idx]
            x_emo_mask = [config.CLS_idx]
            for i, ew in enumerate(arr):
                x_emo += [
                    self.vocab.word2index[ew]
                    if ew in self.vocab.word2index
                    else config.UNK_idx
                ]
                x_emo_mask += [self.vocab.word2index["CLS"]]

            assert len(x_emo) == len(x_emo_mask)
            return torch.LongTensor(x_emo), torch.LongTensor(x_emo_mask)

        else:
            x_dial = [config.CLS_idx]
            x_mask = [config.CLS_idx]
            for i, sentence in enumerate(arr):
                x_dial += [
                    self.vocab.word2index[word]
                    if word in self.vocab.word2index
                    else config.UNK_idx
                    for word in sentence
                ]
                spk = (
                    self.vocab.word2index["USR"]
                    if i % 2 == 0
                    else self.vocab.word2index["SYS"]
                )
                x_mask += [spk for _ in range(len(sentence))]
            assert len(x_dial) == len(x_mask)

            return torch.LongTensor(x_dial), torch.LongTensor(x_mask)

    def preprocess_emo(self, emotion, emo_map):
        program = [0] * len(emo_map)
        program[emo_map[emotion]] = 1
        return program, emo_map[emotion]


def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(
            len(sequences), max(lengths)
        ).long()  ## padding index 1
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    data.sort(key=lambda x: len(x["context"]), reverse=True)  ## sort by source seq
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    ## input
    input_batch, input_lengths = merge(item_info["context"])
    mask_input, mask_input_lengths = merge(item_info["context_mask"])
    emotion_batch, emotion_lengths = merge(item_info["emotion_context"])

    ## Target
    target_batch, target_lengths = merge(item_info["target"])

    input_batch = input_batch.to(config.device)
    mask_input = mask_input.to(config.device)
    target_batch = target_batch.to(config.device)

    d = {}
    d["input_batch"] = input_batch
    d["input_lengths"] = torch.LongTensor(input_lengths)
    d["mask_input"] = mask_input
    d["target_batch"] = target_batch
    d["target_lengths"] = torch.LongTensor(target_lengths)
    d["emotion_context_batch"] = emotion_batch.to(config.device)

    ##program
    d["target_program"] = item_info["emotion"]
    d["program_label"] = item_info["emotion_label"]

    ##text
    d["input_txt"] = item_info["context_text"]
    d["target_txt"] = item_info["target_text"]
    d["program_txt"] = item_info["emotion_text"]
    d["situation_txt"] = item_info["situation_text"]

    d["context_emotion_scores"] = item_info["context_emotion_scores"]

    relations = ["x_intent", "x_need", "x_want", "x_effect", "x_react"]
    for r in relations:
        pad_batch, _ = merge(item_info[r])
        pad_batch = pad_batch.to(config.device)
        d[r] = pad_batch
        d[f"{r}_txt"] = item_info[f"{r}_txt"]

    return d


def prepare_data_seq(batch_size=32):
    """
    quated:
        main.py
    usage:
        バッチサイズに固めた3つのデータ(tra, valid, test)とvocabとlen
        の5つを返す
    """

    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset()

    logging.info("Vocab  {} ".format(vocab.n_words))

    dataset_train = Dataset(pairs_tra, vocab)
    data_loader_tra = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    # torch.utils.data.DataLoader(datase=, ...)
    #    PytorchのDataLoaderにDatasetインスタンス([入力, ラベル]の組を1つ返すもの)
    #    を渡して，データをバッチサイズに固めて戻す

    dataset_valid = Dataset(pairs_val, vocab)
    data_loader_val = torch.utils.data.DataLoader(
        dataset=dataset_valid,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dataset_test = Dataset(pairs_tst, vocab)
    data_loader_tst = torch.utils.data.DataLoader(
        dataset=dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn
    )
    save_config()
    return (
        data_loader_tra,
        data_loader_val,
        data_loader_tst,
        vocab,
        len(dataset_train.emo_map),
    )

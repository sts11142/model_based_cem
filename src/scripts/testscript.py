import numpy as np
import os

# data_dir = './data/ED/'
# data_dir = "./data/ESConv"
# os.chdir(data_dir)
np.set_printoptions(edgeitems=10)

emo_map = {
    "anxiety": 0,
    "anger": 1,
    "fear": 2,
    "depression": 3,
    "sadness": 4,
    "disgust": 5,
    "shame": 6,
    "nervousness": 7,
    "pain": 8,
    "jealousy": 9,
    "guilt": 10,
}

map_emo = {
    0: "anxiety",
    1: "anger",
    2: "fear",
    3: "depression",
    4: "sadness",
    5: "disgust",
    6: "shame",
    7: "nervousness",
    8: "pain",
    9: "jealousy",
    10: "guilt",
}

# print(np.load('sys_dialog_texts.train.npy', allow_pickle=True))   # dialogue
# print(np.load('sys_target_texts.train.npy', allow_pickle=True))     # emotion

def _norm_text(text):
    """ from MISC """
    emo, r, t, *toks = text.strip().split()
    try:
        emo = int(emo)
        r = int(r)
        t = int(t)
        toks = ' '.join(toks[:len(toks)])
    except Exception as e:
        raise e
    return emo, r, t, toks

def _get_inputs_from_text(text):
    """ from MISC """
    srcs = text.strip()
    inputs = []
    emotion = None
    targets = []
    roles = []
    turns = []
    strategy_labels = []
    srcs = srcs.split(" EOS")
    srcs_len = len(srcs)
    """
    srcs:
        ex) ['3 0 0 Hi there, can you help me? ', " 3 1 1 [Question] I'll do my best. What do you need help with? ", ' 3 0 2 I feel depressed because I had to quit my job and stay home with my kids because of their remote school. ', ' 3 1 3 [Reflection of feelings] I can understand why that would make you feel depressed. ', ' 3 0 4 Do you have any advice on how to feel better? ', " 3 1 5 [Providing Suggestions] Yes of course. It's good that you are acknowledging your feelings. To improve your mood you could practice hobbies or other things you enjoy doing."]
    """

    for idx, src in enumerate(srcs):
        if src == "":
            continue
        src_emo, src_role, src_turn, src = _norm_text(src)
        if emotion is None:
            emotion = src_emo

        if src_role == 1:
            try:
                label = "[" + src.split("[")[1].split("]")[0] + "]"  # ex) [ "[Question]" ] → [ "[", "Question]" ] → [ "[", "Question", "]" ]
                src = src.split('[')[-1].split(']')[-1].strip()     # ラベルを剥がす
            except Exception as e:
                strategy_labels.append(8)
            else:
                strategy_labels.append(label)
        else:
            strategy_labels.append(8)
        
        inputs.append(src)
        roles.append(src_role)
        turns.append(src_turn)

        if idx == (srcs_len - 1):
            targets.append(inputs[-1])
            inputs = inputs[0:(srcs_len - 1)]

    return inputs, emotion, targets, roles, turns, strategy_labels

def _make_emotion_fdata(emo_list):
    emo_fdata = []
    for (idx, emo) in enumerate(emo_list):
        emo_fdata.append(emo_map[emo])
    emo_fdata = np.array(emo_fdata, dtype='U12')

    return emo_fdata

def _make_target_fdata(lists):
    target_fdata = []
    for list in lists:
        target_fdata.append(list[0])
    target_fdata = np.array(target_fdata, dtype=str)

    return target_fdata

def construct_conv_ESD(arr, file_type=None):
    contexts_fdata = []
    target_data = []
    emotion_data = []
    situation_data = []
    others_data = {
        "roles": [],
        "turns": [],
        "strategy_labels": []
    }

    with open("data/ESConv" + "/" + file_type + "Situation.txt", "r", encoding="utf-8") as f:
        situation = f.read().split("\n")

    # for row in arr:
    for (row, situ) in zip(arr, situation):
        inputs, emotion, targets, roles, turns, strategy_labels = _get_inputs_from_text(row) 
        contexts_fdata.append(inputs)
        target_data.append(targets)
        # emotion_data.append(emotion)
        emotion_data.append(map_emo[emotion])
        situation_data.append(situ)
        others_data["roles"] = roles
        others_data["turns"] = turns
        others_data["strategy_labels"] = strategy_labels

    
    contexts_fdata = np.array(contexts_fdata, dtype=object)
    target_fdata = _make_target_fdata(target_data)
    # emotion_fdata = _make_emotion_fdata(emotion_data)
    emotion_fdata = np.array(emotion_data, dtype=str)
    situation_fdata = np.array(situation_data, dtype=str)

    return contexts_fdata, target_fdata, emotion_fdata, situation_fdata, others_data

def setup_fdata(file_type):
    # with open(config.data_dir + "/" + file_type + "WithStrategy_short.tsv", "r", encoding="utf-8") as f:
    with open("data/ESConv" + "/" + file_type + "WithStrategy_short.tsv", "r", encoding="utf-8") as f:
        df_trn = f.read().split("\n")
    contexts, target, emotion, situation, _ = construct_conv_ESD(df_trn[:-1], file_type=file_type)

    fdata = []
    fdata.append(contexts)
    fdata.append(target)
    fdata.append(emotion)
    fdata.append(situation)

    fdata = np.array(fdata, dtype=object)
    # fdata = np.array(fdata)

    return fdata


DATA_FILES = lambda data_dir: {
    "train": [
        f"{data_dir}/sys_dialog_texts.train.npy",
        f"{data_dir}/sys_target_texts.train.npy",
        f"{data_dir}/sys_emotion_texts.train.npy",
        f"{data_dir}/sys_situation_texts.train.npy",
    ],
    "dev": [
        f"{data_dir}/sys_dialog_texts.dev.npy",
        f"{data_dir}/sys_target_texts.dev.npy",
        f"{data_dir}/sys_emotion_texts.dev.npy",
        f"{data_dir}/sys_situation_texts.dev.npy",
    ],
    "test": [
        f"{data_dir}/sys_dialog_texts.test.npy",
        f"{data_dir}/sys_target_texts.test.npy",
        f"{data_dir}/sys_emotion_texts.test.npy",
        f"{data_dir}/sys_situation_texts.test.npy",
    ],
}

files = DATA_FILES("data/ED")
train_files_ = [np.load(f, allow_pickle=True) for f in files["train"]]
# dev_files = [np.load(f, allow_pickle=True) for f in files["dev"]]
# test_files = [np.load(f, allow_pickle=True) for f in files["test"]]

train_files = setup_fdata('train')
dev_files = setup_fdata('dev')
test_files = setup_fdata('test')


# def _make_fdata(list):


print(f"ED: {train_files_}\n")
# print(f"ES: {contexts}\n")
print(f"ES: {train_files}\n")
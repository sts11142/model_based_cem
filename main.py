from tqdm import tqdm
from copy import deepcopy
from tensorboardX import SummaryWriter
from torch.nn.init import xavier_uniform_

from src.utils import config
from src.utils.common import set_seed
from src.models.MOEL.model import MOEL
from src.models.MIME.model import MIME
from src.models.EMPDG.model import EMPDG
from src.models.CEM.model import CEM
from src.models.Transformer.model import Transformer
from src.utils.data.loader import prepare_data_seq
from src.models.common import evaluate, count_parameters, make_infinite

import nltk

nltk.download('stopword')


def make_model(vocab, dec_num):
    """
    quated:
        main()
    usage:
        config.model(src.ultis)の値に応じてモデルを生成する．
        vocab, dec_numをモデルに渡す
    """
    is_eval = config.test
    load_checkpoints = config.load_checkpoints
    if config.model == "trs":
        model = Transformer(
            vocab,
            decoder_number=dec_num,
            is_eval=is_eval,
            model_file_path=config.model_path if is_eval else None,
        )
    if config.model == "multi-trs":
        model = Transformer(
            vocab,
            decoder_number=dec_num,
            is_eval=is_eval,
            is_multitask=True,
            model_file_path=config.model_path if is_eval else None,
        )
    elif config.model == "moel":
        model = MOEL(
            vocab,
            decoder_number=dec_num,
            is_eval=is_eval,
            model_file_path=config.model_path if is_eval else None,
        )
    elif config.model == "mime":
        model = MIME(
            vocab,
            decoder_number=dec_num,
            is_eval=is_eval,
            model_file_path=config.model_path if is_eval else None,
        )
    elif config.model == "empdg":
        model = EMPDG(
            vocab,
            decoder_number=dec_num,
            is_eval=is_eval,
            model_file_path=config.model_path if is_eval else None,
        )
    # elif config.model == "cem":
    #     model = CEM(
    #         vocab,
    #         decoder_number=dec_num,
    #         is_eval=is_eval,
    #         model_file_path=config.model_path if is_eval else None,
    #     )
    elif config.model == "cem":
        model = CEM(
            vocab,
            decoder_number=dec_num,
            is_eval=is_eval,
            model_file_path=config.model_path if load_checkpoints or is_eval else None,
            load_optim=load_checkpoints,
        )
        # model_file_path: default → save/test

    model.to(config.device)

    # Intialization
    for n, p in model.named_parameters():
        if p.dim() > 1 and (n != "embedding.lut.weight" and config.pretrain_emb):
            xavier_uniform_(p)

    print("# PARAMETERS", count_parameters(model))

    return model


def train(model, train_set, dev_set):
    """
    1000000回，イテレートしてbest_pplを計算，best値の更新のたびにsave_model()する
    train_one_batch()を必要回数分起動させる
    """
    check_iter = 2000
    try:
        model.train()
        best_ppl = 1000
        patient = 0
        writer = SummaryWriter(log_dir=config.save_path)
        weights_best = deepcopy(model.state_dict())

        load_n_iter = model.loaded_iter

        data_iter = make_infinite(train_set)    # make_infinite(): 引数(dataloader)をyieldでイテレーターに変換する関数
        for n_iter in tqdm(range(1000000)):

            if n_iter < load_n_iter:
                # if n_iter % 100 == 0:
                #     print("continue")
                next(data_iter)
                continue

            if "cem" in config.model:
                loss, ppl, bce, acc, _, _ = model.train_one_batch(
                    next(data_iter), n_iter     # dataごとにtrain_one_batch()を動作させる
                )
            else:
                loss, ppl, bce, acc = model.train_one_batch(next(data_iter), n_iter)

            writer.add_scalars("loss", {"loss_train": loss}, n_iter)
            writer.add_scalars("ppl", {"ppl_train": ppl}, n_iter)
            writer.add_scalars("bce", {"bce_train": bce}, n_iter)
            writer.add_scalars("accuracy", {"acc_train": acc}, n_iter)
            if config.noam:
                writer.add_scalars(
                    "lr", {"learning_rata": model.optimizer._rate}, n_iter
                )

            if (n_iter + 1) % check_iter == 0:  # check_iter(2000回)ごとにチェックを行いログを出力
                model.eval()
                model.epoch = n_iter
                loss_val, ppl_val, bce_val, acc_val, _ = evaluate(
                    model, dev_set, ty="valid", max_dec_step=50
                )
                writer.add_scalars("loss", {"loss_valid": loss_val}, n_iter)
                writer.add_scalars("ppl", {"ppl_valid": ppl_val}, n_iter)
                writer.add_scalars("bce", {"bce_valid": bce_val}, n_iter)
                writer.add_scalars("accuracy", {"acc_train": acc_val}, n_iter)
                model.train()
                if n_iter < 12000:
                    continue
                if ppl_val <= best_ppl:
                    best_ppl = ppl_val
                    patient = 0
                    model.save_model(best_ppl, n_iter)
                    weights_best = deepcopy(model.state_dict())
                else:
                    patient += 1
                if patient > 2:
                    break

    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early ... editted")
        model.save_model(best_ppl, n_iter, is_interrupt=True)
        weights_best = deepcopy(model.state_dict())

    return weights_best


def test(model, test_set):
    model.eval()
    model.is_eval = True
    loss_test, ppl_test, bce_test, acc_test, results = evaluate(
        model, test_set, ty="test", max_dec_step=50
    )
    file_summary = config.save_path + "/results.txt"
    with open(file_summary, "w") as f:
        f.write("EVAL\tLoss\tPPL\tAccuracy\n")
        f.write(
            "{}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(
                loss_test, ppl_test, bce_test, acc_test
            )
        )
        for r in results:
            f.write(r)


def main():
    set_seed()  # for reproducibility

    train_set, dev_set, test_set, vocab, dec_num = prepare_data_seq(
        batch_size=config.batch_size
    )

    model = make_model(vocab, dec_num)

    if config.test:
        test(model, test_set)
    else:
        weights_best = train(model, train_set, dev_set)
        model.epoch = 1
        model.load_state_dict({name: weights_best[name] for name in weights_best})
        test(model, test_set)


if __name__ == "__main__":
    main()
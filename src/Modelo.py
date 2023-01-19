import tensorflow as tf
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
from transformers import create_optimizer
from keras import backend as K
import numpy as np
import json
import sys
import os

## Métricas - Inicio

def func_precision(y_true, y_pred):
    y_pred = np.argmax(y_pred.numpy(), axis=1)
    y_true = np.argmax(y_true.numpy(), axis=1)
    precision = tf.keras.metrics.Precision()
    precision.update_state(y_true, y_pred)
    return precision.result().numpy()

def func_recall(y_true, y_pred):
    y_pred = np.argmax(y_pred.numpy(), axis=1)
    y_true = np.argmax(y_true.numpy(), axis=1)
    recall = tf.keras.metrics.Recall()
    recall.update_state(y_true, y_pred)
    return recall.result().numpy()

def func_f1(y_true, y_pred):
    precision = func_precision(y_true, y_pred)
    recall = func_recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

###############- Fim

def busca_GPU() -> list:
    return tf.config.list_physical_devices("GPU")

def verifica_GPU() -> None:
    try:
        print("Iniciado Verificação da GPU")
        lista_gpu = busca_GPU()
        assert lista_gpu
        for gpu in lista_gpu:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Finalizado Verificação da GPU")
    except AssertionError as e:
        print(f"Erro: GPU não foi encontrada: {e}")
        exit()
    except Exception as e:
        print(f"Erro ao verificar a GPU: {e}")
        exit()

def ler_datasets(dir: str, dir_dataset: str, tipo: str = ".json") -> dict[str, Dataset]:
    try:
        print("Iniciado a Leitura dos Arquivos")
        dict_datasets : dict[str, Dataset] = dict({})
        for arquivo_nome in list(filter(lambda x: x.endswith(tipo), os.listdir(dir+dir_dataset))):
            dict_datasets.update(dict({arquivo_nome: Dataset.from_json(dir+dir_dataset+arquivo_nome)}))
        print("Finalizado a Leitura dos Arquivos")
        return dict_datasets
    except Exception as e:
        print(f"Erro ao ler os datasets: {e}")
        exit()

def get_num_k_fold(dict_datasets: dict[str, Dataset]) -> int:
    return len(set(dict_datasets.values[0]["group"]))

def get_num_class(dict_datasets: dict[str, Dataset]) -> int:
    return len(dict_datasets.values[0]["label"][0])

def get_tokenizer(model_id: str):
    try:
        print("Iniciado a Obtenção do Tokenizador")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("Finalizado a Obtenção do Tokenizador")
        return tokenizer
    except Exception as e:
        print(f"Erro ao pegar o tokenizer: {e}")
        exit()

def get_modelo(model_id: str, num_labels: int):
    try:
        print("Iniciado a Obtenção do Modelo")
        model = TFAutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)
        print("Finalizado a Obtenção do Modelo")
        return model
    except Exception as e:
        print(f"Erro ao pegar o modelo: {e}")
        exit()
    
def tokenize_dataset(data, **kw_args):
    return kw_args["tokenizer"](data["text"], padding=True, return_tensors="tf")

def tokenizador(dict_datasets: dict[str, Dataset], tokenizer) -> dict[str, Dataset]:
    try:
        print("Iniciado a Tokenização")
        kw_args = dict({"tokenizer": tokenizer})
        for dataset_nome, dataset in dict_datasets.items():
            print(f"\tIniciado a Tokenização do dataset {dataset_nome}")
            dict_datasets[dataset_nome] = dataset.map(tokenize_dataset, batched=True, fn_kwargs=kw_args)
            print(f"\tFinalizado a Tokenização do dataset {dataset_nome}")
        print("Finalizado a Tokenização")
        return dict_datasets
    except Exception as e:
        print(f"Erro realizar a tokenização: {e}")
        exit()

def group_by_aux(dataset: Dataset):
    lista_grupos = []
    for grupo in set(dataset["group"]):
        lista_grupos.append(dataset.filter(lambda x: x["group"]==grupo))
    return lista_grupos

def group_by(dict_datasets: dict[str, Dataset]) -> dict[str, list[Dataset]]:
    try:
        print("Iniciado o Agrupamento dos Grupos")
        for dataset_nome, dataset in dict_datasets.items():
            print(f"\tIniciado o Agrupamento do dataset {dataset_nome}")
            dict_datasets[dataset_nome] = group_by(dataset)
            print(f"\tFinalizado o Agrupamento do dataset {dataset_nome}")
        print("Finalizado o Agrupamento dos Grupos")
        return dict_datasets
    except Exception as e:
        print(f"Erro ao agrupar os grupos dos datasets: {e}")
        exit()

def cria_otimizador(num_train_steps: int):
    try:
        print("\tFinalizado a Criação do Otimizador")
        optimizer, _ = create_optimizer(
            init_lr=2e-5,
            num_train_steps=num_train_steps,
            weight_decay_rate=0.01,
            num_warmup_steps=0,
        )
        print("\tFinalizado a Criação do Otimizador")
        return optimizer
    except Exception as e:
        print(f"Erro ao criar o otimizador: {e}")
        exit()

def treinamento_aux(model, tokenizer, list_dataset: list[Dataset], optimizer, epochs: int, batchs: int) -> dict[str, int]:
    resultado = []
    for index, dataset in enumerate(list_dataset):
        ## Dividindo a parte de Treino
        dataset_treino = list_dataset[:index]
        dataset_treino += list_dataset[index+1:]
        dataset_treino = concatenate_datasets(dataset_treino)
        tf_dataset_treino = model.prepare_tf_dataset(dataset_treino, batch_size=batchs, shuffle=True, tokenizer=tokenizer)
        ##############################
        ## Dividindo a parte de Teste
        tf_dataset_teste = model.prepare_tf_dataset(dataset, batch_size=batchs, shuffle=True, tokenizer=tokenizer)
        #############################
        ## Compilando o Modelo
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(),
                func_precision,
                func_recall,
                func_f1
            ],
            run_eagerly = True
        )
        ######################
        ## Treinando o Modelo
        history = model.fit(tf_dataset_treino, epochs=epochs, use_multiprocessing=True)
        #####################
        ## Testando o Modelo
        loss, acc, precision, recall, f1 = model.evaluate(tf_dataset_teste, use_multiprocessing=True)
        ####################
        resultado.append(dict({
            "loss" : loss,
            "accuracy" : acc,
            "precision" : precision,
            "recall" : recall,
            "f1" : f1
        }))
    return resultado

def treinamento(dir: str, dir_resultado: str, model, tokenizer, dict_datasets: dict[str, list[Dataset]], epochs: int, batchs: int, num_k_fold: int) -> None:
    try:
        print("Iniciado o Treinamento GERAL")
        optimizer = cria_otimizador((num_k_fold  // batchs) * epochs)
        for dataset_nome, list_dataset in dict_datasets.items():
            try:
                print(f"\tIniciado o Treinamento do Dataset {dataset_nome}")
                resultado = treinamento_aux(model, tokenizer, list_dataset, optimizer, epochs, batchs)
                with open(f"{dir}{dir_resultado}{dataset_nome}.json", "w") as arquivo:
                    json.dump(resultado, arquivo, indent=4)
                print(f"\tFinalizado o Treinamento do Dataset {dataset_nome}")
            except Exception as e:
                print(f"Erro no treinamento do dataset {dataset_nome}: {e}")
                exit()
        print("Finalizado o Treinamento GERAL")
    except Exception as e:
        print(f"Erro no treinamento: {e}")
        exit()

def main(dir: str, dir_dataset: str, dir_resultado: str, model_id: str, epochs: int, batchs: int) -> None:
    try:
        print("Iniciado a execução do código!")
        verifica_GPU()
        dict_datasets = ler_datasets(dir, dir_dataset)
        num_k_fold = get_num_k_fold(dict_datasets)
        print(f"NUM K_FOLD: {num_k_fold}")
        num_class = get_num_class(dict_datasets)
        print(f"NUM CLASS: {num_class}")
        assert False
        tokenizer = get_tokenizer(model_id)
        model = get_modelo(model_id, num_class)
        dict_datasets = tokenizador(dict_datasets, tokenizer)
        dict_datasets = group_by(dict_datasets)
        treinamento(dir, dir_resultado, model, tokenizer, dict_datasets, epochs, batchs, num_k_fold)
        print("Código executado com sucesso!")
    except Exception as error:
        print(f"Erro no main: {error}")
        exit()


if __name__ == "__main__":
    ## Pegando valores por linha de comando
    try:
        dir: str = "./" if not len(sys.argv) >= 2 else sys.argv[1]
        dir_dataset: str = "data/" if not len(sys.argv) >= 3 else sys.argv[2]
        dir_resultado: str = "results/" if not len(sys.argv) >= 4 else sys.argv[3]
        model_id: str = "neuralmind/bert-base-portuguese-cased" if not len(sys.argv) >= 5 else sys.argv[4]
        epochs: int = 3 if not len(sys.argv) >= 6 else sys.argv[5]
        batchs: int = 16 if not len(sys.argv) >= 7 else sys.argv[6]
        main(dir, dir_dataset, dir_resultado, model_id, epochs, batchs)
    except Exception as error:
        print(f"Error ao pegar as informações passadas por linha de comando: {error}")
        exit()


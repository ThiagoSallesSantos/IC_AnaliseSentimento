import tensorflow as tf
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
from transformers import create_optimizer
from keras import backend as K
import numpy as np
import logging
import json
import sys
import os

## Sistema de Log
logging.basicConfig(format='%(levelname)s: %(message)s - (%(asctime)s)', filename='debug.log', encoding='utf-8', datefmt="%d/%m/%Y %I:%M:%S")

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
        lista_gpu = busca_GPU()
        assert lista_gpu
        for gpu in lista_gpu:
            tf.config.experimental.set_memory_growth(gpu, True)
    except AssertionError as e:
        logging.CRITICAL(f"Erro: GPU não foi encontrada: {e}")
        exit()
    except Exception as e:
        logging.CRITICAL(f"Erro ao verificar a GPU: {e}")
        exit()

def ler_datasets(dir: str, arquivo_nome: str) -> Dataset:
    try:
        return Dataset.from_json(dir)
    except Exception as e:
        logging.CRITICAL(f"Erro ao ler os datasets: {e}")
        exit()

def get_num_k_fold(dataset: dict[str, Dataset]) -> int:
    return len(set(dataset["group"]))

def get_num_class(dataset: dict[str, Dataset]) -> int:
    return len(dataset["labels"][0])

def get_tokenizer(model_id: str):
    try:
        return AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        logging.CRITICAL(f"Erro ao pegar o tokenizer: {e}")
        exit()

def get_modelo(model_id: str, num_labels: int):
    try:
        return TFAutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)
    except Exception as e:
        logging.CRITICAL(f"Erro ao pegar o modelo: {e}")
        exit()
    
def tokenize_dataset(data, **kw_args):
    return kw_args["tokenizer"](data["text"], padding=True, return_tensors="tf")

def tokenizador(dataset: Dataset, tokenizer) -> Dataset:
    try:
        dataset = dataset.map(tokenize_dataset, batched=True, fn_kwargs=dict({"tokenizer": tokenizer}))
        return dataset
    except Exception as e:
        logging.CRITICAL(f"Erro realizar a tokenização: {e}")
        exit()

def group_by(dataset: Dataset) -> list[Dataset]:
    lista_grupos = []
    for grupo in set(dataset["group"]):
        lista_grupos.append(dataset.filter(lambda x: x["group"]==grupo))
    return lista_grupos

def cria_otimizador(num_train_steps: int):
    try:
        optimizer, _ = create_optimizer(
            init_lr=2e-5,
            num_train_steps=num_train_steps,
            weight_decay_rate=0.01,
            num_warmup_steps=0,
        )
        return optimizer
    except Exception as e:
        logging.CRITICAL(f"Erro ao criar o otimizador: {e}")
        exit()

def treinamento(model, tokenizer, dataset_agrupado: list[Dataset], optimizer, epochs: int, batchs: int) -> None:
    try:
        resultado = []
        for index, dataset in enumerate(dataset_agrupado):
            ## Dividindo a parte de Treino
            dataset_treino = dataset_agrupado[:index]
            dataset_treino += dataset_agrupado[index+1:]
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
    except Exception as e:
        logging.CRITICAL(f"Erro no treinamento: {e}")
        exit()

def main(dir: str, dir_dataset: str, dir_resultado: str, model_id: str, epochs: int, batchs: int) -> None:
    try:
        logging.INFO("---Iniciado a execução do código!---")
        ## Verificando compatibiliada com a GPU
        logging.INFO("- Verificando compatibilidade com a GPU - Inicio")
        verifica_GPU()
        logging.INFO("- Verificando compatibilidade com a GPU - Fim")

        ## Pegando Tokenizador a ser utiizado por todos os dataset's
        logging.INFO("- Pegando o tokenizador - Inicio")
        tokenizer = get_tokenizer(model_id)
        logging.INFO("- Pegando o tokenizador - Fim")

        ## For para percorrer os datasets a serem utilizados no treinamento
        for arquivo_nome in list(filter(lambda x: x.endswith(".json"), os.listdir(dir+dir_dataset))):
            logging.INFO(f"- Processo de treinamento usando o dataset {arquivo_nome} - Inicio")

            ## Lendo Dataset
            logging.INFO(f"\t- Lendo o dataset {arquivo_nome} - Inicio")
            dataset = ler_datasets(f"{dir}{dir_dataset}{arquivo_nome}", arquivo_nome)
            logging.INFO(f"\t- Lendo o dataset {arquivo_nome} - Fim")

            ## Pegando o valor de k-fold e num_class do dataset
            logging.INFO("\t- Pegando valores de k-fold e quantidade de classes (Positivo, Negativo e Neutro) - Inicio")
            num_k_fold = get_num_k_fold(dataset)
            num_class = get_num_class(dataset)
            logging.INFO("\t- Pegando valores de k-fold e quantidade de classes (Positivo, Negativo e Neutro) - Fim")
            
            logging.INFO(f"\t- Pegando o modelo para treinamento com qtd classes: {num_class} - Inicio")
            model = get_modelo(model_id, num_class)
            logging.INFO(f"\t- Pegando o modelo para treinamento com qtd classes: {num_class} - Fim")

            # logging.INFO(f"\t- Tokenizando os dados do dataset {arquivo_nome} - Inicio")
            # dataset = tokenizador(dataset, tokenizer)
            # logging.INFO(f"\t- Tokenizando os dados do dataset {arquivo_nome} - Fim")

            # logging.INFO(f"\t- Agrupando os dados do dataset {arquivo_nome} - Inicio")
            # dataset_agrupado = group_by(dataset)
            # logging.INFO(f"\t- Agrupando os dados do dataset {arquivo_nome} - Fim")

            # logging.INFO("\t- Criando um otimizador - Inicio")
            # optimizer = cria_otimizador((num_k_fold  // batchs) * epochs)
            # logging.INFO("\t- Criando um otimizador - Inicio")

            # logging.INFO(f"\t- Treinamento do dataset {arquivo_nome} - Inicio")
            # resultado = treinamento(model, tokenizer, dataset_agrupado, optimizer, epochs, batchs)
            # logging.INFO(f"\t- Treinamento do dataset {arquivo_nome} - Fim")

            # logging.INFO(f"\t- Salvando os resultados obtidos - Inicio")
            # with open(f"{dir}{dir_resultado}{arquivo_nome}", "w") as arquivo:
            #     json.dump(resultado, arquivo, indent=4)
            # logging.INFO(f"\t- Salvando os resultados obtidos - Inicio")

            logging.INFO(f"- Processo de treinamento usando o dataset {arquivo_nome} - Fim")

        logging.INFO("---Código executado com sucesso!---")
    except Exception as e:
        logging.CRITICAL(f"Erro no main: {e}")
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
    except Exception as e:
        logging.CRITICAL(f"Error ao pegar as informações passadas por linha de comando: {e}")
        exit()


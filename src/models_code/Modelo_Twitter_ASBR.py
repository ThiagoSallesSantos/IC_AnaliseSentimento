import tensorflow as tf
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
from keras import backend as K
from keras.utils.np_utils import to_categorical
import numpy as np
import logging
import json
import sys
import os

## Constantes
MEMORIA_CPU = 24 * 1024

## Sistema de Log
logging.basicConfig(format='%(levelname)s: %(message)s - (%(asctime)s)', filename='log.txt', encoding='utf-8', datefmt="%d/%m/%Y %I:%M:%S", level=logging.DEBUG)

## Métricas - Inicio
def func_loss(y_true, y_pred):
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    return loss(y_true, y_pred.logits).numpy()

def func_acc(y_true, y_pred):
    metrica = tf.keras.metrics.CategoricalAccuracy()
    metrica.update_state(y_true, y_pred.logits)
    return metrica.result().numpy()

def func_precision(y_true, y_pred, num_class):
    y_pred = to_categorical(np.argmax(y_pred.logits, axis=1), num_classes=num_class)
    precision = tf.keras.metrics.Precision()
    precision.update_state(y_true, y_pred)
    return precision.result().numpy()

def func_recall(y_true, y_pred, num_class):
    y_pred = to_categorical(np.argmax(y_pred.logits, axis=1), num_classes=num_class)
    recall = tf.keras.metrics.Recall()
    recall.update_state(y_true, y_pred)
    return recall.result().numpy()

def func_f1(y_true, y_pred, num_class):
    precision = func_precision(y_true, y_pred, num_class)
    recall = func_recall(y_true, y_pred, num_class)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

###############- Fim

def busca_GPU() -> list:
    return tf.config.list_physical_devices("GPU")

def verifica_GPU(poct_memoria_cpu: int) -> None:
    try:
        lista_gpu = busca_GPU()
        assert lista_gpu
        for gpu in lista_gpu:
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=MEMORIA_CPU * poct_memoria_cpu)]
            )
    except AssertionError as e:
        logging.error(f"Erro: GPU não foi encontrada: {e}")
        exit()
    except Exception as e:
        logging.error(f"Erro ao verificar a GPU: {e}")
        exit()

def ler_datasets(dir: str) -> Dataset:
    try:
        return Dataset.from_json(dir)
    except Exception as e:
        logging.error(f"Erro ao ler os datasets: {e}")
        exit()

def get_num_class(dataset: dict[str, Dataset]) -> int:
    return len(dataset["labels"][0])

def get_tokenizer(model_id: str):
    try:
        return AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        logging.error(f"Erro ao pegar o tokenizer: {e}")
        exit()

def get_modelo(model_id: str, num_labels: int):
    try:
        return TFAutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)
    except Exception as e:
        logging.error(f"Erro ao pegar o modelo: {e}")
        exit()
    
def tokenize_dataset(data, **kw_args):
    return kw_args["tokenizer"](data["text"], padding=True, return_tensors="tf", max_length=kw_args["max_length"], truncation=True)

def tokenizador(dataset: Dataset, tokenizer, max_length: int, num_batchs: int) -> Dataset:
    try:
        dataset = dataset.map(tokenize_dataset, batched=True, batch_size=num_batchs, fn_kwargs=dict({"tokenizer": tokenizer, "max_length": max_length}))
        return dataset
    except Exception as e:
        logging.error(f"Erro realizar a tokenização: {e}")
        exit()

def group_by(dataset: Dataset) -> dict[str, Dataset]:
    dict_grupo = dict({})
    for grupo in set(dataset["group"]):
        dict_grupo.update(dict({grupo: dataset.filter(lambda dado: dado["group"] == grupo)}))
    return dict_grupo


def cria_otimizador():
    try:
        return tf.keras.optimizers.Adam(learning_rate=2e-5)
    except Exception as e:
        logging.error(f"Erro ao criar o otimizador: {e}")
        exit()

def remove_colunas(dict_dataset: list[Dataset], colunas: list[str] = ["text", "group", "labels_int"]):
    for grupo in dict_dataset.keys():
        dict_dataset[grupo] = dict_dataset[grupo].remove_columns(colunas)
    return dict_dataset

def treinamento(model, tokenizer, dataset_agrupado: list[Dataset], optimizer, num_epochs: int, num_batchs: int, num_class: int) -> None:
    try:
        ## Treino
        dataset_treino = dataset_agrupado["train"]
        tf_dataset_treino = model.prepare_tf_dataset(dataset_treino, batch_size=num_batchs, shuffle=True, tokenizer=tokenizer)
        
        ## Teste
        tf_dataset_teste = model.prepare_tf_dataset(dataset_agrupado["test"], batch_size=num_batchs, shuffle=False, tokenizer=tokenizer)
        
        ## Modelo
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy()
            ]
        )

        ## Treinamento
        model.fit(tf_dataset_treino, batch_size=num_batchs, epochs=num_epochs, use_multiprocessing=True)

        ## Teste
        y_pred = model.predict(tf_dataset_teste, batch_size=num_batchs, use_multiprocessing=True)

        ## Metricas
        acc = func_acc(dataset_agrupado["test"]["labels"], y_pred)
        precision = func_precision(dataset_agrupado["test"]["labels"], y_pred, num_class)
        recall = func_recall(dataset_agrupado["test"]["labels"], y_pred, num_class)
        f1 = func_f1(dataset_agrupado["test"]["labels"], y_pred, num_class)
        loss = func_loss(dataset_agrupado["test"]["labels"], y_pred)
        ####################

        return list([dict({
            "loss" : float(loss),
            "accuracy" : float(acc),
            "precision" : float(precision),
            "recall" : float(recall),
            "f1" : float(f1)
        })])
    except Exception as e:
        logging.error(f"Erro no treinamento: {e}")
        exit()

def main(dir: str, dir_dataset: str, dir_resultado: str, dir_model: str, model_id: str, max_length: int, num_epochs: int, num_batchs: int, poct_memoria_cpu: float) -> None:
    try:
        logging.info("\n\n\n---Iniciado a execução do código!---")
        ## Verificando compatibiliada com a GPU
        logging.info("- Verificando compatibilidade com a GPU - Inicio")
        verifica_GPU(poct_memoria_cpu)
        logging.info("- Verificando compatibilidade com a GPU - Fim")

        ## Pegando Tokenizador a ser utiizado por todos os dataset's
        logging.info("- Pegando o tokenizador - Inicio")
        tokenizer = get_tokenizer(model_id)
        logging.info("- Pegando o tokenizador - Fim")
        
        lista_arquivos = list(filter(lambda x: x.endswith(".json"), os.listdir(f"{dir}{dir_dataset}")))
        logging.info(f"- Lista de datasets para treinamento {str(lista_arquivos)} -")

        ## For para percorrer os datasets a serem utilizados no treinamento
        for arquivo_nome in lista_arquivos:

            if f"{num_batchs}_batchs_{arquivo_nome}" in os.listdir(f"{dir}{dir_resultado}"):
                logging.info(f"- Já foi realizado o treinamento com o dataset {arquivo_nome} com batch {num_batchs} - Inicio / Fim")
                continue
            
            logging.info(f"- Processo de treinamento usando o dataset {arquivo_nome} - Inicio")

            ## Lendo Dataset
            logging.info(f"\t- Lendo o dataset {arquivo_nome} - Inicio")
            dataset = ler_datasets(f"{dir}{dir_dataset}{arquivo_nome}")
            logging.info(f"\t- Lendo o dataset {arquivo_nome} - Fim")

            ## Pegando o valor de k-fold e num_class do dataset
            logging.info("\t- Pegando a quantidade de classes (Positivo, Negativo e Neutro) - Inicio")
            num_class = get_num_class(dataset)
            logging.info("\t- Pegando a quantidade de classes (Positivo, Negativo e Neutro) - Fim")
            
            logging.info(f"\t- Pegando o modelo para treinamento com qtd classes: {num_class} - Inicio")
            model = get_modelo(model_id, num_class)
            logging.info(f"\t- Pegando o modelo para treinamento com qtd classes: {num_class} - Fim")

            logging.info(f"\t- Tokenizando os dados do dataset {arquivo_nome} - Inicio")
            dataset = tokenizador(dataset, tokenizer, max_length, num_batchs)
            logging.info(f"\t- Tokenizando os dados do dataset {arquivo_nome} - Fim")

            logging.info(f"\t- Agrupando os dados do dataset {arquivo_nome} - Inicio")
            dataset_agrupado = group_by(dataset)
            logging.info(f"\t- Agrupando os dados do dataset {arquivo_nome} - Fim")

            logging.info("\t- Criando um otimizador - Inicio")
            optimizer = cria_otimizador()
            logging.info("\t- Criando um otimizador - Fim")

            logging.info(f"\t- Removendo colunas desnecessárias do dataset {arquivo_nome} - Inicio")
            dataset_agrupado = remove_colunas(dataset_agrupado)
            logging.info(f"\t- Removendo colunas desnecessárias do dataset {arquivo_nome} - Fim")    

            logging.info(f"\t- Treinamento do dataset {arquivo_nome} - Inicio")
            resultado = treinamento(model, tokenizer, dataset_agrupado, optimizer, num_epochs, num_batchs, num_class)
            logging.info(f"\t- Treinamento do dataset {arquivo_nome} - Fim")

            logging.info(f"\t- Salvando os resultados obtidos - Inicio")
            with open(f"{dir}{dir_resultado}{num_batchs}_{arquivo_nome}", "w") as arquivo:
                json.dump(resultado, arquivo, indent=4)
            logging.info(f"\t- Salvando os resultados obtidos - Fim")

            logging.info(f"- Processo de treinamento usando o dataset {arquivo_nome} - Fim")

            logging.info(f"- Salvando o modelo {num_batchs}_{arquivo_nome} - Inicio")
            model.save(f"{dir}{dir_model}modelo_{num_batchs}_{arquivo_nome[:-5]}")
            logging.info(f"- Salvando o modelo {num_batchs}_{arquivo_nome} - Fim")

            ## Limpando memórias
            K.clear_session()
            del model

        logging.info("---Código executado com sucesso!---")
    except Exception as e:
        logging.error(f"Erro no main: {e}")
        exit()

if __name__ == "__main__":
    ## Pegando valores por linha de comando
    try:
        dir: str = "./" if not len(sys.argv) >= 2 else sys.argv[1]
        dir_dataset: str = "data/" if not len(sys.argv) >= 3 else sys.argv[2]
        dir_resultado: str = "results/" if not len(sys.argv) >= 4 else sys.argv[3]
        dir_model: str = "models/" if not len(sys.argv) >= 5 else sys.argv[4]
        model_id: str = 'neuralmind/bert-base-portuguese-cased' if not len(sys.argv) >= 6 else sys.argv[5]
        max_length: int = 128 if not len(sys.argv) >= 7 else sys.argv[6]
        num_epochs: int = 3 if not len(sys.argv) >= 8 else sys.argv[7]
        num_batchs: int = 16 if not len(sys.argv) >= 9 else sys.argv[8]
        poct_memoria_cpu: float = 0.9 if not len(sys.argv) >= 10 else sys.argv[9]
        main(dir, dir_dataset, dir_resultado, dir_model, model_id, max_length, num_epochs, num_batchs, poct_memoria_cpu)
    except Exception as e:
        logging.error(f"Error ao pegar as informações passadas por linha de comando: {e}")
        exit()


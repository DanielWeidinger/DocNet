import os
import pandas as pd
import numpy as np
import faiss


class FaissTopK(object):
    def __init__(self, embedding_file):
        super(FaissTopK, self).__init__()
        self.embedding_file = embedding_file
        _, ext = os.path.splitext(self.embedding_file)
        if ext == '.pkl':
            self.df = pd.read_pickle(self.embedding_file)
        else:
            self.df = pd.read_parquet(self.embedding_file)
        self._get_faiss_index()
        # self.df.drop(columns=["Q_FFNN_embeds", "A_FFNN_embeds"], inplace=True)

    def _get_faiss_index(self):
        # with Pool(cpu_count()) as p:
        #     question_bert = p.map(eval, self.df["Q_FFNN_embeds"].tolist())
        #     answer_bert = p.map(eval, self.df["A_FFNN_embeds"].tolist())
        question_bert = self.df["Q_FFNN_embeds"].tolist()
        self.df.drop(columns=["Q_FFNN_embeds"], inplace=True)
        answer_bert = self.df["A_FFNN_embeds"].tolist()
        self.df.drop(columns=["A_FFNN_embeds"], inplace=True)
        question_bert = np.array(question_bert, dtype='float32')
        answer_bert = np.array(answer_bert, dtype='float32')

        self.answer_index = faiss.IndexFlatIP(answer_bert.shape[-1])

        self.question_index = faiss.IndexFlatIP(question_bert.shape[-1])

        self.answer_index.add(answer_bert)
        self.question_index.add(question_bert)

        del answer_bert, question_bert

    def predict(self, q_embedding, search_by='answer', topk=5, answer_only=True):
        if search_by == 'answer':
            _, index = self.answer_index.search(
                q_embedding.astype('float32'), topk)
        else:
            _, index = self.question_index.search(
                q_embedding.astype('float32'), topk)

        output_df = self.df.iloc[index[0], :]
        if answer_only:
            return output_df.answer.tolist()
        else:
            return (output_df.question.tolist(), output_df.answer.tolist())
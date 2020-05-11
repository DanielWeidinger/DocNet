from docproduct.predictor import RetreiveQADoc


pretrained_path = 'BioBertFolder/biobert_v1.0_pubmed_pmc/'
# ffn_weight_file = None
bert_ffn_weight_file = 'newFolder/models/bertffn_crossentropy/bertffn'
embedding_file = 'Float16EmbeddingsExpanded5-27-19.pkl'

doc = RetreiveQADoc(pretrained_path=pretrained_path,
                    ffn_weight_file=None,
                    bert_ffn_weight_file=bert_ffn_weight_file,
                    embedding_file=embedding_file)
question_text = "My back hurts"  # @param {type:"string"}
search_similarity_by = 'answer'  # @param ['answer', "question"]
number_results_toReturn = 10  # @param {type:"number"}
answer_only = True  # @param ["False", "True"] {type:"raw"}
returned_results = doc.predict(question_text, search_by=search_similarity_by, topk=number_results_toReturn, answer_only=answer_only)

model = doc.qa_embed.model
save_path = './SavedModels/DocNet/1'

model.summary()

model.save(save_path)

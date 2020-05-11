from docproduct.predictor import RetreiveQADoc


pretrained_path = 'BioBertFolder/biobert_v1.0_pubmed_pmc/'
# ffn_weight_file = None
bert_ffn_weight_file = 'newFolder/models/bertffn_crossentropy/bertffn'
embedding_file = 'Float16EmbeddingsExpanded5-27-19.pkl'

doc = RetreiveQADoc(pretrained_path=pretrained_path,
                    ffn_weight_file=None,
                    bert_ffn_weight_file=bert_ffn_weight_file,
                    embedding_file=embedding_file)


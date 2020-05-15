import os
import json
import requests
from docproduct.predictor import RetreiveQADoc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


pretrained_path = 'BioBertFolder/biobert_v1.0_pubmed_pmc/'
# ffn_weight_file = None
bert_ffn_weight_file = 'newFolder/models/bertffn_crossentropy/bertffn'
embedding_file = 'Float16EmbeddingsExpanded5-27-19.pkl'

doc = RetreiveQADoc(pretrained_path=pretrained_path,
                    ffn_weight_file=None,
                    bert_ffn_weight_file=bert_ffn_weight_file,
                    embedding_file=embedding_file)
question_text = ["My back hurts really bad"]  # @param {type:"string"}
search_similarity_by = 'answer'  # @param ['answer', "question"]
number_results_toReturn = 10  # @param {type:"number"}
answer_only = True  # @param ["False", "True"] {type:"raw"}

print('make inputs...')

inputs = doc.qa_embed._make_inputs(questions=question_text, dataset=False)

print(inputs[0].shape)
headers = {"content-type": "application/json"}
data = json.dumps({
    'signature_name': "serving_default",
    'instances': inputs.tolist()
    })

print('Send request...')

res = requests.post('http://34.78.190.89:8501/v1/models/docnet:predict', data=data, headers=headers)
predictions = json.loads(res.content)["predictions"]
print(predictions)
#returned_results = doc.predict(question_text, search_by=search_similarity_by, topk=number_results_toReturn, answer_only=answer_only)

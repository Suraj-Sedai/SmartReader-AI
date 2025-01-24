from rest_framework.decorators import api_view
from rest_framework.response import Response
from transformers import BertForQuestionAnswering, BertTokenizer
import torch

# Load the pre-trained model and tokenizer
model_name = "deepset/bert-base-cased-squad2"  # Pre-trained Q&A model
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

@api_view(['POST'])
def ask_question(request):
    question = request.data.get('question')
    context = request.data.get('context')

    if not question or not context:
        return Response({"error": "Both 'question' and 'context' are required."}, status=400)

    # Tokenize input
    inputs = tokenizer(question, context, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract answer
    answer_start = torch.argmax(outputs.start_logits)  # Start of answer
    answer_end = torch.argmax(outputs.end_logits) + 1  # End of answer
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end])
    )

    return Response({"answer": answer})

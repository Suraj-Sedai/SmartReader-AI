from rest_framework.decorators import api_view
from rest_framework.response import Response
from transformers import TFBertForQuestionAnswering, BertTokenizer
import tensorflow as tf

# Load the pre-trained model and tokenizer
model_name = "deepset/bert-base-cased-squad2"  # Pre-trained Q&A model
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForQuestionAnswering.from_pretrained(model_name)

@api_view(['POST'])
def ask_question(request):
    question = request.data.get('question')
    context = request.data.get('context')

    if not question or not context:
        return Response({"error": "Both 'question' and 'context' are required."}, status=400)

    # Tokenize input
    inputs = tokenizer(question, context, return_tensors="tf")

    # Get model predictions
    outputs = model(inputs)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # Extract answer
    answer_start = tf.argmax(start_logits, axis=1).numpy()[0]  # Start of answer
    answer_end = tf.argmax(end_logits, axis=1).numpy()[0] + 1  # End of answer
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"].numpy()[0][answer_start:answer_end])
    )

    return Response({"answer": answer})

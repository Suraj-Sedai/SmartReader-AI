from rest_framework.decorators import api_view
from rest_framework.response import Response
from transformers import TFBertForQuestionAnswering, BertTokenizer
import tensorflow as tf
from rest_framework.parsers import MultiPartParser
from django.core.files.storage import default_storage
import os
from docx import Document
from PyPDF2 import PdfReader

# Load the pre-trained model and tokenizer
model_name = "deepset/bert-base-cased-squad2"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForQuestionAnswering.from_pretrained(model_name)


@api_view(['POST'])
def upload_file(request):
    """
    This API handles file uploads and extracts text for Q&A.
    """
    # Allow file upload
    file = request.FILES.get('file')

    if not file:
        return Response({"error": "No file uploaded."}, status=400)

    # Save the file temporarily
    file_path = default_storage.save(file.name, file)
    file_extension = os.path.splitext(file_path)[1]

    try:
        # Extract text based on file type
        if file_extension == '.docx':
            context = extract_text_from_docx(file_path)
        elif file_extension == '.pdf':
            context = extract_text_from_pdf(file_path)
        else:
            return Response({"error": "Unsupported file type. Please upload a .docx or .pdf file."}, status=400)

        # Clean up (delete temporary file)
        default_storage.delete(file_path)

        return Response({"context": context})

    except Exception as e:
        return Response({"error": f"Error processing file: {str(e)}"}, status=500)

def extract_text_from_docx(file_path):
    """Extract text from a .docx file."""
    document = Document(file_path)
    return '\n'.join([paragraph.text for paragraph in document.paragraphs])

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

@api_view(['POST'])
def ask_question(request):
    """
    This API takes a question and context and returns an answer using a TensorFlow model.
    """
    # Step 1: Get input from the user
    question = request.data.get('question')
    context = request.data.get('context')

    if not question or not context:
        return Response({"error": "Both 'question' and 'context' are required."}, status=400)

    try:
        # Step 2: Tokenize the input
        inputs = tokenizer(question, context, return_tensors="tf")

        # Step 3: Get predictions from the model
        outputs = model(inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Step 4: Find the start and end of the answer
        start_index = tf.argmax(start_logits, axis=1).numpy()[0]
        end_index = tf.argmax(end_logits, axis=1).numpy()[0] + 1

        # Step 5: Convert the tokens back to a readable string
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(inputs["input_ids"].numpy()[0][start_index:end_index])
        )

        # Step 6: Return the answer
        return Response({"answer": answer})

    except Exception as e:
        # Handle errors gracefully
        return Response({"error": f"An error occurred: {str(e)}"}, status=500)

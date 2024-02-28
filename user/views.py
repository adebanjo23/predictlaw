import os
from django.conf import settings
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langdetect import detect, LangDetectException
from googletrans import Translator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from decouple import config

load_dotenv()

UPLOAD_FOLDER = 'pdfs'
embeddings_dir = 'embeddings'
ALLOWED_EXTENSIONS = {'pdf'}
OPENAI_API_KEY = config("OPENAI_API_KEY")

from django.core.exceptions import ObjectDoesNotExist
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from django.utils import timezone
from .models import User, OTP
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status


@api_view(['POST'])
def register(request):
    username = request.data.get("username")
    phone = request.data.get("phone")
    email = request.data.get("email")

    if not username or not phone or not email:
        return Response({"success": "All fields are required"}, status=status.HTTP_400_BAD_REQUEST)

    # noinspection PyUnresolvedReferences
    if User.objects.filter(email=email).exists():
        return Response({'error': "Email is already in use"}, status=status.HTTP_400_BAD_REQUEST)

    # noinspection PyUnresolvedReferences
    if User.objects.filter(username=username).exists():
        return Response({'error': "Username is already in use"}, status=status.HTTP_400_BAD_REQUEST)

    # noinspection PyUnresolvedReferences
    user = User.objects.create(
        username=username,
        phone=phone,
        email=email,
    )
    otp_code = OTP.generate_otp_code()
    otp_entry = OTP.objects.create(user=user, otp_code=otp_code)

    try:
        my_subject = 'OTP Verification Email'
        my_recipient = email
        html_content = render_to_string("index.html", {'otp_code': otp_entry.otp_code})
        plain_message = strip_tags(html_content)

        send_mail(
            subject=my_subject,
            message=plain_message,
            from_email=None,
            recipient_list=[my_recipient],
            html_message=html_content,
            fail_silently=False,
        )
    except Exception as e:
        print(f"Failed to send OTP email to {email}: {e}")

    return Response(
        {
            'success': "User registered successfully. Please check your email for the OTP. If you didn't receive an OTP, contact support."},
        status=status.HTTP_201_CREATED)


@api_view(['POST'])
def resend_otp(request):
    email = request.data.get('email')
    if not email:
        return Response({"error": "Email address is required."}, status=status.HTTP_400_BAD_REQUEST)

    try:
        # noinspection PyUnresolvedReferences
        user = User.objects.get(email=email)
    except ObjectDoesNotExist:
        return Response({"error": "User does not exist."}, status=status.HTTP_404_NOT_FOUND)

    new_otp_code = OTP.generate_otp_code()
    OTP.objects.update_or_create(
        user=user,
        defaults={'otp_code': new_otp_code, 'created_at': timezone.now(), 'is_verified': False}
    )

    try:
        my_subject = 'OTP Verification Email'
        my_recipient = email
        html_content = render_to_string("index.html", {'otp_code': new_otp_code})
        plain_message = strip_tags(html_content)

        send_mail(
            subject=my_subject,
            message=plain_message,
            from_email=None,
            recipient_list=[my_recipient],
            html_message=html_content,
            fail_silently=False,
        )
        return Response({"success": "OTP resent successfully. Please check your email."}, status=status.HTTP_200_OK)
    except Exception as e:
        print(f"Failed to send OTP email to {email}: {e}")
        return Response({"error": "Failed to send OTP. Please try again later."},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def verify_otp(request):
    email = request.data.get('email')
    input_otp = request.data.get('otp')

    if not email or not input_otp:
        return Response({"error": "Email and OTP code are required."}, status=status.HTTP_400_BAD_REQUEST)

    try:
        # noinspection PyUnresolvedReferences
        user = User.objects.get(email=email)
    except ObjectDoesNotExist:
        return Response({"error": "User does not exist."}, status=status.HTTP_404_NOT_FOUND)

    try:
        otp_entry = OTP.objects.get(user=user, is_verified=False)
        if otp_entry.is_expired:
            return Response({"error": "OTP has expired. Please request a new one."}, status=status.HTTP_400_BAD_REQUEST)
        elif otp_entry.otp_code == input_otp:
            otp_entry.is_verified = True
            otp_entry.save()
            user.is_verified = True
            user.save()
            return Response({"success": "OTP verified successfully. User is now verified."}, status=status.HTTP_200_OK)
        else:
            return Response({"error": "Incorrect OTP."}, status=status.HTTP_400_BAD_REQUEST)
    except ObjectDoesNotExist:
        return Response({"error": "OTP not found or already verified."}, status=status.HTTP_404_NOT_FOUND)


@api_view(['POST'])
def user_login(request):
    if request.method != 'POST':
        return Response("Invalid request method", status=status.HTTP_400_BAD_REQUEST)

    try:
        username = request.data.get("username")
        if not username:
            return Response("Username is required", status=status.HTTP_400_BAD_REQUEST)

        # noinspection PyUnresolvedReferences
        user = User.objects.get(username=username)

        if user is None:
            return Response({"error": "Invalid username"}, status=status.HTTP_401_UNAUTHORIZED)

        if user.is_active:
            return Response({"success": "Login successful"}, status=status.HTTP_200_OK)

        else:
            return Response({"error": "User is not active"}, status=status.HTTP_406_NOT_ACCEPTABLE)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def allowed_file(filename):
    return '.' in filename and (
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS or filename.lower().endswith('.json'))


@api_view(['POST'])
def upload_pdf(request):
    if request.method != 'POST':
        return Response({"error": "Invalid request method"}, status=status.HTTP_400_BAD_REQUEST)

    username = request.data.get('username')
    pdf = request.FILES.get('pdf')

    # Check if username and pdf are provided
    if not (username and pdf):
        return Response({"error": "Username and PDF file are required"}, status=status.HTTP_400_BAD_REQUEST)

    # Check if the file type is allowed
    if not allowed_file(pdf.name):
        return Response({"error": "Invalid file type. Only PDF files are allowed"}, status=status.HTTP_400_BAD_REQUEST)

    # Check if a file with the same username already exists
    existing_file_path = os.path.join(settings.MEDIA_ROOT, UPLOAD_FOLDER, f"{username}.pdf")
    if os.path.exists(existing_file_path):
        return Response(
            {"error": f"A PDF file with the username {username} already exists. Delete it to upload a new one."},
            status=status.HTTP_400_BAD_REQUEST)

    # Save the PDF file
    filename = f"{username}.pdf"
    with open(os.path.join(settings.MEDIA_ROOT, UPLOAD_FOLDER, filename), 'wb') as destination:
        for chunk in pdf.chunks():
            destination.write(chunk)

    return Response({'success': 'PDF uploaded successfully'}, status=status.HTTP_200_OK)


@api_view(['POST'])
def query_pdf(request):
    if request.method != 'POST':
        return Response({"error": "Invalid request method"}, status=status.HTTP_400_BAD_REQUEST)

    username = request.data.get('username')
    query = request.data.get('question')

    if not (username and query):
        return Response({"error": "Username and question are required"}, status=status.HTTP_400_BAD_REQUEST)

    filename = f"{username}.pdf"
    pdf_path = os.path.join(settings.MEDIA_ROOT, UPLOAD_FOLDER, filename)

    if os.path.exists(pdf_path):
        try:
            pdf_reader = PdfReader(pdf_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            # Detect language
            try:
                detected_language = detect(text)
            except LangDetectException as e:
                return Response({"error": f"Language detection error: {str(e)}"},
                                status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            if detected_language != 'en':
                try:
                    translator = Translator()
                    translated_text = translator.translate(text, src=detected_language, dest='en').text
                    text = translated_text
                except Exception as e:
                    print(f"Translation error: {str(e)}")

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text=text)

            # Load or create embeddings
            store_name = username
            if os.path.exists(f"media/embeddings/{store_name}.pkl"):
                with open(f"media/embeddings/{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
            else:
                embeddings = OpenAIEmbeddings()
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"media/embeddings/{store_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)

            # Query the model
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0.8)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
            return Response({'response': response}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(str(e), status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    else:
        return Response('PDF not found for the given username', status=status.HTTP_404_NOT_FOUND)


@api_view(['POST'])
def delete_pdf(request):
    if request.method != 'POST':
        return Response({"error": "Invalid request method"}, status=status.HTTP_400_BAD_REQUEST)

    username = request.data.get('username')

    if not username:
        return Response({"error": "Username is required for deletion"}, status=status.HTTP_400_BAD_REQUEST)

    filename = f"{username}.pdf"
    pdf_path = os.path.join(settings.MEDIA_ROOT, UPLOAD_FOLDER, filename)
    pkl_path = os.path.join(settings.MEDIA_ROOT, embeddings_dir, f"{username}.pkl")

    if os.path.exists(pdf_path):
        try:
            os.remove(pdf_path)
            if os.path.exists(pkl_path):
                os.remove(pkl_path)
            return Response({'success': f'PDF file and embedding for {username} deleted successfully'},
                            status=status.HTTP_200_OK)
        except Exception as e:
            return Response(str(e), status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    else:
        return Response({'success': 'PDF not found for the given username'}, status=status.HTTP_404_NOT_FOUND)


@api_view(['POST'])
def delete_user(request):
    email = request.data.get('email')
    if not email:
        return Response({'error': 'Email address is required.'}, status=status.HTTP_400_BAD_REQUEST)
    try:
        # noinspection PyUnresolvedReferences
        user = User.objects.get(email=email)
        user.delete()
        return Response({'success': 'User deleted successfully.'}, status=status.HTTP_204_NO_CONTENT)
    except ObjectDoesNotExist:
        return Response({'error': 'User not found.'}, status=status.HTTP_404_NOT_FOUND)

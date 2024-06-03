from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

# Import namespaces
from azure.ai.vision.face import FaceClient
from azure.ai.vision.face.models import (
    FaceDetectionModel,
    FaceRecognitionModel,
    FaceAttributeTypeDetection01,
    FaceAttributeTypeRecognition04,
)
from azure.core.credentials import AzureKeyCredential

def main():

    global face_client

    try:
        # Get Configuration Settings
        load_dotenv()
        cog_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        cog_key = os.getenv('AI_SERVICE_KEY')

        # Authenticate Face client
        face_client = FaceClient(
            endpoint=cog_endpoint,
            credential=AzureKeyCredential(cog_key)
        )

        # Menu for face functions
        DetectFaces(os.path.join('images','we_are_the_world_group_image.jpg'))

    except Exception as ex:
        print(ex)

def DetectFaces(image_file):
    print('Detecting faces in', image_file)

    # Specify facial features to be retrieved
    features = [
        FaceAttributeTypeDetection01.OCCLUSION,
        FaceAttributeTypeDetection01.BLUR,
        FaceAttributeTypeDetection01.GLASSES,
        FaceAttributeTypeRecognition04.QUALITY_FOR_RECOGNITION
    ]
    
    # Get faces
    with open(image_file, mode='rb') as image_data:
        detected_faces = face_client.detect(
            image_content=image_data,
            return_face_attributes=features,
            return_face_id=False,
            detection_model=FaceDetectionModel.DETECTION_01,
            recognition_model=FaceRecognitionModel.RECOGNITION_04
        )
        if len(detected_faces) > 0:
            print(f'{len(detected_faces)} faces detected.')

            # Prepare image for drawing
            fig = plt.figure(figsize=(8, 6))
            plt.axis('off')
            image = Image.open(image_file)
            draw = ImageDraw.Draw(image)
            color = 'lightgreen'
            face_count = 0

            # Draw and annotate each face
            for face in detected_faces:

                # Get face properties
                face_count += 1
                print(f'\nFace number {face_count}')

                detected_attributes = face.face_attributes.as_dict()
                if 'blur' in detected_attributes:
                    print(' - Blur:')
                    for blur_name in detected_attributes['blur']:
                        print('  - {}: {}'.format(blur_name, detected_attributes['blur'][blur_name]))

                if 'occlusion' in detected_attributes:
                    print(' - Occlusion:')
                    for occlusion_name in detected_attributes['occlusion']:
                        print('  - {}: {}'.format(occlusion_name, detected_attributes['occlusion'][occlusion_name]))

                if 'glasses' in detected_attributes:
                    print(f" - Glasses:{detected_attributes['glasses']}")
                
                # Draw and annotate face
                r = face.face_rectangle
                bounding_box = ((r.left, r.top), (r.left + r.width, r.top + r.height))
                draw = ImageDraw.Draw(image)
                draw.rectangle(bounding_box, outline=color, width=5)
                annotation = f'FN{face_count}'
                plt.annotate(annotation, (r.left, r.top), color=color)
            
            # Save annotated image
            plt.imshow(image)
            outputfile = 'detected_face.jpg'
            fig.savefig(outputfile)

            print(f'\nResults saved in {outputfile}')

if __name__ == "__main__":
    main()
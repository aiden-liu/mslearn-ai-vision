from dotenv import load_dotenv
import os
from array import array
from PIL import Image, ImageDraw
import sys
import time
from matplotlib import pyplot as plt
import numpy as np

# Import namespaces
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential


def main():
    global cv_client

    try:
        # Get Configuration Settings
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')

        # Get image
        image_file = 'images/people.jpg'
        if len(sys.argv) > 1:
            image_file = sys.argv[1]

        # Authenticate Azure AI Vision client
        client = ImageAnalysisClient(
            endpoint=ai_endpoint,
            credential=AzureKeyCredential(ai_key)
        )
        
        # Analyze image
        AnalyzeImage(image_file, client)

    except Exception as ex:
        print(ex)


def AnalyzeImage(image_file, client: ImageAnalysisClient):
    print('\nAnalyzing', image_file)

    # Specify features to be retrieved (PEOPLE)
    with open(image_file, 'rb') as image_data:
        result = client.analyze(
            image_data=image_data,
            visual_features=[
                VisualFeatures.READ,
                VisualFeatures.PEOPLE
            ]
        )

    if result.people is not None:
        print("\nPeople in image:")

        image = Image.open(image_file)
        fig = plt.figure(figsize=(image.width/100, image.height/100))
        plt.axis('off')
        draw = ImageDraw.Draw(image)
        color = 'cyan'

        for detected_people in result.people['values']:
            if detected_people.confidence > 0.5:
                # Draw object bounding box
                r = detected_people.bounding_box
                bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height))
                draw.rectangle(bounding_box, outline=color, width=3)

                # Return the confidence of the person detected
                print(" {} (confidence: {:.2f}%)".format(detected_people.bounding_box, detected_people.confidence * 100))
        # Save annotated image
        plt.imshow(image)
        plt.tight_layout(pad=0)
        outputfile = 'detected_people.jpg'
        fig.savefig(outputfile)
        print(f' Results saved in {outputfile}')
    else:
        error_details = sdk.ImageAnalysisErrorDetails.from_result(result)
        print("Analysis failed.")
        print(f"  Error reason: {error_details.reason}")
        print(f"  Error code: {error_details.error_code}")
        print(f"  Error message: {error_details.message}")
if __name__ == "__main__":
    main()
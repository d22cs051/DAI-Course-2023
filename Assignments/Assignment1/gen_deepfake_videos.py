
# goto: https://replicate.com/account to get api key
# goto: https://replicate.com/yoyo-nb/thin-plate-spline-motion-model/api for usage explation

import replicate
from download_script import download_data

model = replicate.models.get("yoyo-nb/thin-plate-spline-motion-model")
version = model.versions.get("382ceb8a9439737020bad407dec813e150388873760ad4a5a83a2ad01b039977")

k = 1
for i in range(1,10):
    for j in range(1,8):
        # https://replicate.com/yoyo-nb/thin-plate-spline-motion-model/versions/382ceb8a9439737020bad407dec813e150388873760ad4a5a83a2ad01b039977#input
        inputs = {
            # Input source image.
            'source_image': open(f"ORIGINAL_PICS/{i}.jpg", "rb"),

            # Choose a micromotion.
            'driving_video': open(f"ORIGINAL_VIDS/video{j}.mp4", "rb"),

            # Choose a dataset.
            'dataset_name': "vox",
        }

        print(f"Generating.... with image {i}, video {j}")
        # https://replicate.com/yoyo-nb/thin-plate-spline-motion-model/versions/382ceb8a9439737020bad407dec813e150388873760ad4a5a83a2ad01b039977#output-schema
        output = version.predict(**inputs)
        print(output)
        print("Generating....")
        print(download_data(source=output,destination="gen_vid",count=k))
        print(f"Video Downloaded: {k}")
        k+=1
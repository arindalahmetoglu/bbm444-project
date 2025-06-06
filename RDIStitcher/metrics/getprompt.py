def get_eval_prompt():
    prompt = "I need you to become a stitched image quality assessment evaluator. " \
             "First, describe what is in the image. " \
             "Then, the evaluation process should be as objective and impartial as possible, " \
             "giving specific ratings and reasons, " \
             "including seam, brightness transition, distortion, clear and abnormal content, each aspect 2 points.\n\n"


    prompt += "1. Whether there are seams in the image (2 points). " \
          "score 2: the image is smooth without obvious boundaries or misalignment; " \
          "score 1: there are slightly visible boundaries in the image, but overall look well; " \
          "score 0: there are obvious borders or dislocations in the image, affecting the overall look and feel\n\n"

    prompt += "2. Whether there are brightness transitions in the image (2 points). " \
              "score 2: the brightness transition of image is smooth; " \
              "score 1: the light and shade changes in the image are a bit unnatural; " \
              "score 0: the light and shade changes in the image are very abrupt\n\n"

    prompt += "3. Whether there are distortions in the image (2 points)." \
              "score 2: no distortion in the image; " \
              "score 1: there are a few structural anomalies of straight lines in the image; " \
              "score 0: there are noticeably distortions, such as distorted pillar, brick, and building construction\n\n"

    prompt += "4. Whether the image is clear and blurred (2 points). " \
              "score 2: the image is clear, the details are visible, and there is no blur; " \
              "score 1: the resolution of the image is good, but slightly blurred; " \
              "score 0: the image is blurred and the details are not clear\n\n"

    prompt += "5. Whether the image is natural (2 points). " \
              "score 2: the image is natural with out abnormal content; " \
              "score 1: there are some places in the image that is not in harmony with the main content;" \
              "score 0: There are a lot of abnormal content in the image such as strange texture and non-semantic image\n\n"

    prompt += "Please format the evaluation as follows: FINAL_SCORE: [score]."

    return prompt

def get_eval_compare_prompt():
    prompt = "I need you to become a stitched image quality assessment evaluator. " \
             "Compare the input two stitched images, " \
             "includes seam, brightness transition, distortion, clear and abnormal content. " \
             "Choose which one you think is better, " \
             "giving specific ratings and reasons. " \
             "There are two choices, image 1 is better, image 2 is better.\n\n"
    prompt += "Please format the evaluation as follows: FINAL_CHOICE: image [1 or 2] is better"
    return prompt
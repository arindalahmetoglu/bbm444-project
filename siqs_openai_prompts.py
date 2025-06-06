"""

This module provides standardized prompts for evaluating image composition quality
using OpenAI's MLLM (GPT-4V/GPT-4o) for different datasets.

"""

def get_udis_d_prompt() -> str:
    """
    Returns the standardized SIQS evaluation prompt for the UDIS-D dataset.
    This prompt is adapted from RDIStitcher/metrics/getprompt.py for single-image evaluation.
    It asks the MLLM to act as a stitched image quality assessor and score the image
    based on five specific criteria, each worth 2 points.
    """
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

def get_comparison_prompt() -> str:
    """
    Returns a prompt for comparing two stitched images.
    This is useful for A/B testing or ranking different methods on the same pair.
    """
    prompt = "I need you to become a stitched image quality assessment evaluator. " \
             "Compare the input two stitched images, " \
             "includes seam, brightness transition, distortion, clear and abnormal content. " \
             "Choose which one you think is better, " \
             "giving specific ratings and reasons. " \
             "There are two choices, image 1 is better, image 2 is better.\n\n"
    prompt += "Please format the evaluation as follows: FINAL_CHOICE: image [1 or 2] is better"
    return prompt

def get_beehive_prompt() -> str:
    """
    Returns the standardized SIQS evaluation prompt for the Beehive dataset.
    This prompt is highly specific, focusing on challenges like comb alignment,
    distortion of hexagonal cells, and artifacts from moving bees.
    """
    prompt = (
        "You are an evaluator for stitched beehive image quality. The evaluation process should be as objective as possible. Images were taken with manual exposure. "
        "First, briefly describe the image content, noting bees and hive structure. "
        "Then, rate the image on 5 aspects, each 0-2 points (total 10 points). For each, provide a score and a brief reason.\n\n"

        "1. Seams & Alignment (2 points): "
        "Are there visible seams or misalignments in the comb structure? "
        "score 2: Comb is smooth without obvious boundaries or misalignment. "
        "score 1: Slightly visible boundaries/misalignments, but comb overall looks well. "
        "score 0: Obvious borders or dislocations in comb, affecting overall look.\n\n"

        "2. Comb Distortion (2 points): "
        "Is the comb's hexagonal cell structure or frame unnaturally distorted by stitching? "
        "score 2: No unnatural distortion in comb structure or hexagonal cells. "
        "score 1: A few minor structural anomalies or unnatural waviness in cells/frame. "
        "score 0: Noticeable unnatural distortions in comb, like warped cells or bent frames.\n\n"

        "3. Clarity (2 points): "
        "Is the image clear, or blurred by the stitching process? "
        "score 2: Image is clear, comb/bee details visible, no stitching-induced blur. "
        "score 1: Image resolution good, but slightly blurred by stitching. "
        "score 0: Image blurred by stitching, comb/bee details unclear.\n\n"
              
        "4. Bee Artifacts (2 points): "
        "How are moving bees handled by stitching (ghosting, duplication, smearing)? "
        "score 2: Bees appear clear and whole, no stitching-induced artifacts. "
        "score 1: Some bees show minor stitching-induced ghosting or duplication. "
        "score 0: Significant stitching-induced ghosting/duplication of bees, obscuring details.\n\n"
              
        "5. Naturalness (2 points): "
        "Does the image look natural? Are there other artifacts from stitching (e.g., strange textures, non-semantic patches)? "
        "score 2: Image looks natural, without other stitching-induced abnormal content. "
        "score 1: Some areas in the image seem disharmonious due to stitching artifacts. "
        "score 0: A lot of abnormal content from stitching, like strange textures or non-semantic areas.\n\n"

        "Please format the evaluation as follows:\n"
        "IMAGE_DESCRIPTION: [Brief description]\n"
        "SEAMS_ALIGNMENT_SCORE: [0/1/2], REASON: [...]\n"
        "COMB_DISTORTION_SCORE: [0/1/2], REASON: [...]\n"
        "CLARITY_STITCHING_SCORE: [0/1/2], REASON: [...]\n"
        "BEE_ARTIFACTS_STITCHING_SCORE: [0/1/2], REASON: [...]\n"
        "NATURALNESS_SCORE: [0/1/2], REASON: [...]\n"
        "FINAL_SCORE: [Sum of scores]"
    )
    return prompt

def get_beehive_comparison_prompt() -> str:
    """
    Returns a prompt for comparing three stitched beehive images from the different methods.
    """
    prompt = (
        "You are an evaluator for stitched beehive image quality. Images were taken with manual exposure. "
        "Compare these three stitched beehive images:\n"
        "• Image 1: NIS \n"
        "• Image 2: UDIS \n"
        "• Image 3: UDIS++ \n\n"
        "For each image, rate on 5 aspects, each 0-2 points (total 10 points):\n"
        "1. Seams & Alignment: Visible seams or misalignments in comb structure\n"
        "2. Comb Distortion: Unnatural distortion in hexagonal cells or frame by stitching\n"
        "3. Clarity (Stitching-Related): Image clarity, or blur caused by stitching\n"
        "4. Bee Artifacts (Stitching-Related): Ghosting, duplication, or smearing of moving bees\n"
        "5. Naturalness: Overall natural appearance, absence of strange textures or artifacts\n\n"
        "Please format your evaluation as follows:\n"
        "IMAGE_1_SEAMS_ALIGNMENT: [0-2]\n"
        "IMAGE_1_COMB_DISTORTION: [0-2]\n"
        "IMAGE_1_CLARITY_STITCHING: [0-2]\n"
        "IMAGE_1_BEE_ARTIFACTS: [0-2]\n"
        "IMAGE_1_NATURALNESS: [0-2]\n"
        "IMAGE_1_TOTAL: [0-10]\n\n"
        "IMAGE_2_SEAMS_ALIGNMENT: [0-2]\n"
        "IMAGE_2_COMB_DISTORTION: [0-2]\n"
        "IMAGE_2_CLARITY_STITCHING: [0-2]\n"
        "IMAGE_2_BEE_ARTIFACTS: [0-2]\n"
        "IMAGE_2_NATURALNESS: [0-2]\n"
        "IMAGE_2_TOTAL: [0-10]\n\n"
        "IMAGE_3_SEAMS_ALIGNMENT: [0-2]\n"
        "IMAGE_3_COMB_DISTORTION: [0-2]\n"
        "IMAGE_3_CLARITY_STITCHING: [0-2]\n"
        "IMAGE_3_BEE_ARTIFACTS: [0-2]\n"
        "IMAGE_3_NATURALNESS: [0-2]\n"
        "IMAGE_3_TOTAL: [0-10]\n\n"
        "FINAL_RANKING: [3, 2, 1] (best to worst based on total scores)"
    )
    return prompt 
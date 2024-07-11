CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "./log"

EVAL_TASK = [
    "image_caption",
    "descriptive_object_region_select",
    "region_text_match",
    "image_text_selection",
    "text_type",
    "question_image_match",
    "open-domain_VQA",
    "VQA_color",
    "VQA_utility_affordance",
    "VQA_scene_recognition",
    "VQA_counting",
    "VQA_attribute",
    "VQA_object_recognition",
    "VQA_sentiment_understanding",
    "VQA_activity_recognition",
    "VQA_object_presence",
    "VQA_positional_reasoning",
    "VQA_sport_recognition",
    "VQA",
    "object_match",
    "object_region_match",
    "ITM",
    "image_quality",
    "text_localization",
    "object_region_selection",
    "text_legibility",
    "object_grounding",
    "object_description_generate",
    "descriptive_object_region_generate",
    "region_object_selection",
    #"region_generation",
]

TEST_TASK = [
    "natural_language_visual_reasoning",
    "visual_spatial_reasoning",
    "visual_nli",
    "text_vqa",
    "grounded_VQA",
    "commonsense_VQA",
    "visual_dialog",
]


OUTPUT_REGION_TASK = {
    "detection", "VG", "object_region_selection", "region_generation","pointing_grounded_VQA","descriptive_object_region_generate", "text_localization", "visual_object_region", "visual_subject_region", "descriptive_object_region_select", "VG_selection", "grounded_VQA", "select_overlap_most_region", "select_overlap_least_region", "select_overlaped_region", "select_nonoverlaped_region", "GC", "region_text_match",
}

OUTPUT_IMAGE_CODE_TASK = {"image_generation", "infilling", "im_region_extraction", "im_descriptive_infilling", "image_completion",  "image_completion_w_image_caption", "image_completion_w_region_caption", "im_descriptive_extraction"}

NO_IMAGE_AS_INPUT = {'image_generation'}

OPTIONS_REGION_TASK = {
    "object_region_selection","pointing_grounded_VQA", "text_localization", "descriptive_object_region_select","VG_selection", "grounded_VQA", "select_overlap_most_region", "select_overlap_least_region","select_overlaped_region", "select_nonoverlaped_region","object_match", "object_grounding", "text_type",
}

META_REGION_TASK = {
    "visual_answer_justification", "commonsense_VQA", "visual_object_region", "visual_subject_region", "select_overlap_most_region", "select_overlap_least_region", "select_overlaped_region", "select_nonoverlaped_region", "if_region_overlap", "region_object_selection", "text_legibility", "object_region_match", "object_description_generate",
}

MISSING_TASK = {'VQA_absurd','region_area'}
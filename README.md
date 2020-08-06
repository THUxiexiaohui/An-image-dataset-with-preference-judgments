# An-image-dataset-with-preference-judgments
Introduction: The image dataset and code used in SIGIR 2020 paper "Preference-based Evaluation Metrics for Web Image Search". Please cite our paper if you use this dataset in your work.

---
- **Datasets**
   - image_pairs_annotation : Preference judgments for image pairs. Each pair receives preference judgments from three assessors. The format is "Query	Image_pair(Search engine/query_imageID.jpg)	three preference_tags(-2: Definitely left, -1:left, 0:Tie, 1: Right, 2: Definitely Right)".
   - SERP_level_preference : SERP-level preference judgments (Golden standard used in the paper). The format is "Query	Winner(0: Sogou, 1: Tie, 2: Baidu)"
   - Image_position.json : Position information of each image. For each image, a triple (row_number, column_number, number of images in the row) is given. *In image search, a gird-based result placement is adopted.*
- **Code**
   - PWP.py : Code for the proposed preference-based evaluation metric : Preference-Winning-Penalty (PWP). How to read and process the above files are also shown in this code file.

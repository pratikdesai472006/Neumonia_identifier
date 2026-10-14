PneumoVision Demonstration Test Samples
==========================================

This folder contains verified test chest radiographs of different dimensions and formats for project demonstrations:

Folder: sample_xray_images/normal/
- normal_01_square_1024x1024_clear_normal.png       (1024 x 1024 Square PNG)
- normal_01_dicom_1024x1024_clear_normal.dcm       (1024 x 1024 Medical DICOM)
- normal_02_portrait_800x1024_healthy_normal.jpg    (800 x 1024 Portrait JPEG)
- normal_03_landscape_1024x768_normal_chest.png     (1024 x 768 Landscape PNG)
- normal_04_medium_512x512_normal_adult.jpg         (512 x 512 Scaled Square JPEG)
- normal_05_compact_450x600_normal_screening.png    (450 x 600 Compact Portrait PNG)

Folder: sample_xray_images/pneumonia/
- pneumonia_01_square_1024x1024_severe_pneumonia.png    (1024 x 1024 Square PNG)
- pneumonia_01_dicom_1024x1024_severe_pneumonia.dcm    (1024 x 1024 Medical DICOM)
- pneumonia_02_portrait_768x1024_dense_opacity.jpg      (768 x 1024 Portrait JPEG)
- pneumonia_03_landscape_1024x800_consolidation.png     (1024 x 800 Landscape PNG)
- pneumonia_04_medium_512x512_pneumonia_bilateral.jpg   (512 x 512 Scaled Square JPEG)
- pneumonia_05_compact_450x600_pneumonia_lobar.png      (450 x 600 Compact Portrait PNG)

All images can be uploaded directly into PneumoVision to demonstrate:
1. Ground truth Normal vs Pneumonia screening
2. Non-square portrait and landscape automatic padding
3. Full Grad-CAM explainability localization
4. Multi-format support (.PNG, .JPG, .DCM)

## GAN_based_Image_Restoration
 24SS_Data_Engineering_Team1

## Image_Filter_Generator
1. full_filter(Cam_Option, File_Path) # Animal ears, nose
    1. Cam_Option : Numeric index -> Camera, Filepath -> videofile
    2. File_Path : filtered image will be saved here
![test_gt](https://github.com/KBC-1315/GAN_based_Image_Restoration/assets/77442063/5033c90e-7e48-4aa8-82ed-9ce9f6f8ce32)
![testfull_filter](https://github.com/KBC-1315/GAN_based_Image_Restoration/assets/77442063/e68be829-1e80-4efc-bf65-8c5c57d03351)

2. nose_filter(Can_Option, File_Path) # Pig nose only
    1. Cam_Option : Numeric index -> Camera, Filepath -> videofile
    2. File_Path : filtered image will be saved here
![glitter_test_gt](https://github.com/KBC-1315/GAN_based_Image_Restoration/assets/77442063/00e14a75-e90e-4a24-a56f-241534c08e49)
![glitter_test_glitter](https://github.com/KBC-1315/GAN_based_Image_Restoration/assets/77442063/275d57d6-b4fe-410e-ba12-a63fdfbcb57a)

3. glitter_filter(Cam_Option, File_Path, Filter_Level) # gliter filter
    1. Cam_Option : Numeric index -> Camera, Filepath -> videofile
    2. File_Path : filtered image will be saved here
    3. Filter_Level : frequency of glitter 0 = maximum, large num to decrease(integer only >= 0)
![nose_filter_output png_gt](https://github.com/KBC-1315/GAN_based_Image_Restoration/assets/77442063/7c71b0b8-ab67-42ea-8af1-4ad5bf2bf892)
![nose_filter_output png_nose_filter](https://github.com/KBC-1315/GAN_based_Image_Restoration/assets/77442063/e5a3c59b-5db7-4916-a10f-695ec9ff3754)

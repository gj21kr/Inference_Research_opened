# Updated
1. configurations for this version
2. post processing
3. test time augmentation
4. ensemble
5. data loader\
6. project-based inference
**7. Localizer filtering out**
**8. Odd affine correcting**

# Usage
-python inference_SingleDicom_J.py -i (dicom directory) -o (output directory) -g (gpu number) -t (configuration)\
-python inference_MultiDicom_J.py -i (dicom directory list) -o (output directory) -g (gpu number) -t (configuration)

**configuration**
>aorta : -t aorta\
>other arteries : -t other_artery\
>binary vein : -t whole_vein\
>multi class vein : -t multi_vein\
>fat/muscle : -t scpm\
>abdominal wall : -t awall\
>rib : -t rib\
>skin : -t skin\

-python inference_SingleDicom_MultiWorks_J.py -i (dicom directory) -o (output directory) -g (gpu number) -s (keep sequence) -t (configuration)\
-python inference_MultiDicom_MultiWorks_J.py -i (dicom directory list) -o (output directory) -g (gpu number) -s (keep sequence) -t (configuration)\

**configuration**\
You can use comma(,) or space( ) to distinguish between each one.\
>aorta, other arteries : -t 'aorta,other_artery' OR -t 'aorta other_artery'\
>binary vein, multi class vein : -t 'whole_vein,multi_vein' OR -t 'whole_vein multi_vein'

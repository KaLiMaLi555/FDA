# FDA: Fourier Domain Adaptation for Semantic Segmentation.

This is the Pytorch implementation of our [FDA](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.pdf) paper published in CVPR 2020.

Domain adaptation via *style transfer* made easy using Fourier Transform. FDA needs **no deep networks** for style transfer, and involves **no adversarial training**. Below is the diagram of the proposed Fourier Domain Adaptation method:

Step 1: Apply FFT to source and target images.

Step 2: Replace the low frequency part of the source amplitude with that from the target.

Step 3: Apply inverse FFT to the modified source spectrum.

![Image of FDA](https://github.com/YanchaoYang/FDA/blob/master/demo_images/FDA.png)

We have prepared a well documented version of the [original repository](https://github.com/YanchaoYang/FDA) with the code flow available [here](https://drive.google.com/file/d/1Ondj__Dqzf6bytJeN4gwnUpyf3AIuyjy/view?usp=sharing).

# Usage

1. FDA Demo
   
   > python3 FDA_demo.py
   
   An example of FDA for domain adaptation. (source: GTA5, target: CityScapes, with beta 0.01)
   
   ![Image of Source](https://github.com/YanchaoYang/FDA/blob/master/demo_images/example.png)


2. Sim2Real Adaptation Using FDA (single beta)

   > python3 train.py --snapshot-dir='../checkpoints/FDA' --init-weights='../checkpoints/FDA/init_weight/DeepLab_init.pth' 
                      --LB=0.01 --entW=0.005 --ita=2.0 --switch2entropy=0

   *Important*: use the original images for FDA, then do mean subtraction, normalization, etc. Otherwise, will be numerical artifacts.

   DeepLab initialization can be downloaded through this [link.](https://drive.google.com/file/d/1dk_4JJZBj4OZ1mkfJ-iLLWPIulQqvHQd/view?usp=sharing)

   LB: beta in the paper, controls the size of the low frequency window to be replaced.

   entW: weight on the entropy term.
   
   ita: coefficient for the robust norm on entropy.
   
   switch2entropy: entropy minimization kicks in after this many steps.


3. Evaluation of the Segmentation Networks Adapted with Multi-band Transfer (multiple betas)

   > python3 evaluation_multi.py --model='DeepLab' --save='../results' 
                                 --restore-opt1="../checkpoints/FDA/gta2city_deeplab/gta2city_LB_0_01" 
                                 --restore-opt2="../checkpoints/FDA/gta2city_deeplab/gta2city_LB_0_05" 
                                 --restore-opt3="../checkpoints/FDA/gta2city_deeplab/gta2city_LB_0_09"

   Pretrained models on the GTA5 -> CityScapes task using DeepLab backbone can be downloaded [here.](https://drive.google.com/file/d/1HueawBlg6RFaKNt2wAX__1vmmupKqHmS/view?usp=sharing)
   
   The above command should output:
       ===> mIoU19: 50.45
       ===> mIoU16: 54.23
       ===> mIoU13: 59.78
       

4. Get Pseudo Labels for Self-supervised Training

   > python3 getSudoLabel_multi.py --model='DeepLab' --data-list-target='./dataset/cityscapes_list/train.txt' --set='train' 
                                   --restore-opt1="../checkpoints/FDA/gta2city_deeplab/gta2city_LB_0_01" 
                                   --restore-opt2="../checkpoints/FDA/gta2city_deeplab/gta2city_LB_0_05" 
                                   --restore-opt3="../checkpoints/FDA/gta2city_deeplab/gta2city_LB_0_09"


5. Self-supervised Training with Pseudo Labels

   > python3 SStrain.py --model='DeepLab' --snapshot-dir='../checkpoints/FDA' --init-weights='../checkpoints/FDA/init_weight/DeepLab_init.pth' 
                        --label-folder='cs_pseudo_label' --LB=0.01 --entW=0.005 --ita=2.0

6. Other Models

   VGG initializations can be downloaded through this [link.](https://drive.google.com/file/d/1pgHtwBKUcbAyItnU4hgMb96UfY1PGiCv/view?usp=sharing)
   
    > python3 train.py --model='VGG' --learning-rate=1e-5 --snapshot-dir='../checkpoints/FDA' --init-weights='../checkpoints/FDA/init_weight/vggfcn_gta5_init.pth' 
    ---LB=0.01 --entW=0.005 --ita=2.0 --switch2entropy=0
   
   Pretrained models on the Synthia -> CityScapes task using DeepLab backbone [link.](https://drive.google.com/file/d/1FRI_KIWnubyknChhTOAVl6ZsPxzvEXce/view?usp=sharing)
   
   Pretrained models on the GTA5 -> CityScapes task using VGG backbone [link.](https://drive.google.com/file/d/15Az8DFaLw1kTgt82KX9rI6S85n7iesdc/view?usp=sharing)
   
   Pretrained models on the Synthia -> CityScapes task using VGG backbone [link.](https://drive.google.com/file/d/1SC7sxKtic_7ClFmAZDlrBqRaL0pvKYZ8/view?usp=sharing)
   
7. Models trained by the team at AGV.AI (IITKGP)
    
   | Beta Value | DeepLab | VGG16 |
   |------------|---------|-------|
   | 0.01 (T=0) | [link](https://drive.google.com/drive/folders/1101cMmEKlkBQ-oMLFWLaU0sn_y-AHRqs?usp=sharing)    | [link](https://drive.google.com/drive/folders/1py_CXSFTu9t4jNDVOb2RmtVQgXfSm7ur?usp=sharing)  |
   | 0.05 (T=0) | [link](https://drive.google.com/drive/folders/1PEYoOe65TRIWcNG45qMEMZAL7hjWbO5U?usp=sharing)    | [link](https://drive.google.com/drive/folders/1-NlVPgHvFBcN0Wb4oA5vkOdeRR1Z_7p2?usp=sharing)  |
   | 0.09 (T=0) | [link](https://drive.google.com/drive/folders/17qXK696NaQv5tBnOFLZ_mTWPp8NAY5jQ?usp=sharing)    | [link](https://drive.google.com/drive/folders/1ZwuBmLk6D_8YgsoG3gEzOeE2Ohh0htiX?usp=sharing)  |
   | 0.01 (T=1) | [link](https://drive.google.com/drive/folders/12Ae-TrGcIAb91gm49PlA-Quc98A7J_Xe?usp=sharing)    | [link](https://drive.google.com/drive/folders/1ldFTVY55QEUj1NY-h7_UXYKGoMlnO6s-?usp=sharing)  |
   | 0.05 (T=1) | [link](https://drive.google.com/drive/folders/1YomOo27v2uIWNy78wjG7mdLl5D_JA5d0?usp=sharing)    | [link](https://drive.google.com/drive/folders/1yhGNQiWS2dBcw3IRr1mTEZJ3F28Jy203?usp=sharing)  |
   | 0.09 (T=1) | [link](https://drive.google.com/drive/folders/1Yi99lTKkKxzMWsm_0vH0kCcHH36CdzgQ?usp=sharing)    | [link](https://drive.google.com/drive/folders/1-VT3vLlxqV3cj0NaMzcFznrPM4Arfgjw?usp=sharing)  |
   | 0.01 (T=2) | [link](https://drive.google.com/drive/folders/11JzUB4uYA3e_lB9Zoe9-iaGPbSXNXwYT?usp=sharing)    | [link](https://drive.google.com/drive/folders/1-IybTsqIabjOpZzj8urlDxONoey8jCSb?usp=sharing)  |
   | 0.05 (T=2) | [link](https://drive.google.com/drive/folders/1z-4fptNdhfFzledC_0YQ2sfOY9qJHCrB?usp=sharing)    | [link](https://drive.google.com/drive/folders/1-RrWb11LGBcdiLaq1SQdjsPOdy7kxTXz?usp=sharing)  |
   | 0.09 (T=2) | [link](https://drive.google.com/drive/folders/1bNG4jdqinHSC1ELYFYyqpCj6r4Li-8Sz?usp=sharing)    | [link](https://drive.google.com/drive/folders/1XUx3nv614A7d8LCTdRyTn9lupQhpTqpS?usp=sharing)  |

8. Files train.py and SStrain_VGG.py are integrated with wandb and will log the source and target loss at the frequency inputted in the argument. You will just have to login        during the initial run of the code which is done using

   > wandb.login()
   
   And the logging process is started using 
   
   > wandb.init()
   
   The values are logged using the command
   
   > wandb.log(..)

**Acknowledgment**

Code adapted from [BDL.](https://github.com/liyunsheng13/BDL)

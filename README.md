# **Deep Learning-Based Change Detection of Forest, Water, Barren Land, and Human Activity Growth**

Keywords: deep learning; transfer learning; forest cover change detection; river change detection; barren land detection; very high resolution (VHR); DeepLabv3+; deeply supervised image fusion network (DSIFN); desertification

## **Dataset**

[Notes about data preparation](/data_preparation)

1. [LoveDA](https://github.com/Junjue-Wang/LoveDA) | [Processed](https://drive.google.com/drive/folders/1AX5DdNeSseyn3rN89jYoNEznxX7QCUgH?usp=drive_link)

1. [Change Detection Dataset](https://isprs-archives.copernicus.org/articles/XLII-2/565/2018/)

## **Run training loop**

1. `cd` to the directory with main.py file
    ```
    cd ../src/dsifn
    ```

1. Pass [data directory](https://isprs-archives.copernicus.org/articles/XLII-2/565/2018/), output directory, number of epochs as arguments in commandline
    ```
    python main.py data_dir output_dir epochs
    ```

## **Report Draft**

[Report](https://drive.google.com/file/d/1YIqk1mqUxTfQ6spSek5eOrf2dWRm4xFW/view?usp=sharing)

## **Models**

Deeply Supervised Image Fusion Network [PyTorch .pth Model](https://drive.google.com/file/d/1FvhzXGa9grV2fcWcrTcKyfRg9HVwf81y/view?usp=sharing) | [CODE](/src/dsifn/)

Fine-Tuned Deeplabv3 [PyTorch .pth Model](https://drive.google.com/file/d/1soy__dmdOcu2osOa0UK_OAkoBa8s5dL1/view?usp=sharing) | [CODE](/src/deeplabv3/)

## **DSIFN**

![DSIFN prediction image](/src/dsifn/output/Figure_1.png)

## **Setup/Environment**

[Notes about setup & environment](/docs/setup)

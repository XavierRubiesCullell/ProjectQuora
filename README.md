# ProjectQuora
### Xavier Rubiés Cullell & Dànae Canillas Sánchez

In this project, a study of the dataset provided by Quora in its Kaggle competition has been
carried out in order to detect duplicate questions. We will fine-tune pretrained transformer models
from the Hugging Face library. We will present the results for different models (BERT, XLNet,
DistilBERT, ...) and different hyperparameter combinations that have been used. Finally, we will
explore sentence embedding meaning.

The dataset is taken from Quora competition at Kaggle:
https://www.kaggle.com/c/quora-question-pairs

>- [plots](https://github.com/XavierRubiesCullell/ProjectQuora/tree/master/plots): Folder that contains the plots generated in [class_visualization.ipynb](https://github.com/XavierRubiesCullell/ProjectQuora/blob/master/src/class_visualization.ipynb)
>> -  [2d_pca.html](https://github.com/XavierRubiesCullell/ProjectQuora/blob/master/plots/2d_pca.html)
>> -  [2d_tsne.html](https://github.com/XavierRubiesCullell/ProjectQuora/blob/master/plots/2d_tsne.html) 
>> -  [3d_pca.html](https://github.com/XavierRubiesCullell/ProjectQuora/blob/master/plots/3d_pca.html)
>> -  [3d_tsne.html](https://github.com/XavierRubiesCullell/ProjectQuora/blob/master/plots/3d_tsne.html)
>- [report](https://github.com/danaecanillas/NeuralNetworks/tree/master/fira/model): Deliverables
>> -  [imgs](https://github.com/XavierRubiesCullell/ProjectQuora/tree/master/report/imgs): Images contained in [POE_Final_Project_Quora_CanillasRubies.pdf](https://github.com/danaecanillas/NeuralNetworks/blob/master/fira/model/embedding.model)
>> -  [Hyperparameters_Study.pdf](https://github.com/danaecanillas/NeuralNetworkseport/blob/master/fira/model/embedding.model): Table of the hyperparameters experiments
>> -  [POE_Final_Project_Quora_CanillasRubies.pdf](https://github.com/danaecanillas/NeuralNetworks/blob/master/fira/model/embedding.model): Deliverable report
>> -  [POE_Initial_Plan.pdf](https://github.com/XavierRubiesCullell/ProjectQuora/blob/master/report/POE_Initial_Plan.pdf): First deliverable
>> -  [Presentacio-XavierDanae.pdf](https://github.com/XavierRubiesCullell/ProjectQuora/blob/master/report/Presentacio-XavierDanae.pdf): Intermediate project presentation
>- [src](https://github.com/danaecanillas/NeuralNetworks/tree/master/fira/output): Folder containing script files
>> -  [data](https://github.com/XavierRubiesCullell/ProjectQuora/tree/master/src/data): CVS files
>>> -  [train.csv](https://github.com/XavierRubiesCullell/ProjectQuora/blob/master/src/data/train.csv): Raw data
>>> -  [sentences.csv](https://github.com/XavierRubiesCullell/ProjectQuora/blob/master/src/data/sentences.csv): Table with questions and tokenizations (from BERT)
>> -  [class-consistency.ipynb](https://github.com/XavierRubiesCullell/ProjectQuora/blob/master/src/class-consistency.ipynb): Prediction consistency study
>> -  [class_visualization.ipynb](https://github.com/XavierRubiesCullell/ProjectQuora/blob/master/src/class_visualization.ipynb): Generates [plots](https://github.com/XavierRubiesCullell/ProjectQuora/tree/master/plots)
>> -  [data_analysis.ipynb](https://github.com/XavierRubiesCullell/ProjectQuora/blob/master/src/data_analysis.ipynb): Data inference
>> -  [input_net.py](https://github.com/XavierRubiesCullell/ProjectQuora/blob/master/src/input_net.py): Generates the model input
>> -  [main.py](https://github.com/XavierRubiesCullell/ProjectQuora/blob/master/src/main.py): Model training and validation
>> -  [most_similar_sentence.ipynb](https://github.com/XavierRubiesCullell/ProjectQuora/blob/master/src/most_similar_sentence.ipynb): Most similar sentence search
>> -  [table_generation.ipynb](https://github.com/XavierRubiesCullell/ProjectQuora/blob/master/src/table_generation.ipynb): Generates [sentences.csv](https://github.com/XavierRubiesCullell/ProjectQuora/blob/master/src/data/sentences.csv)
>> -  [utils.py](https://github.com/XavierRubiesCullell/ProjectQuora/blob/master/src/utils.py): Contains auxiliary functions

- [README.md](https://github.com/XavierRubiesCullell/ProjectQuora/blob/master/README.md): Project Documentation

- [.gitignore](https://github.com/XavierRubiesCullell/ProjectQuora/blob/master/.gitignore) : Untracked files

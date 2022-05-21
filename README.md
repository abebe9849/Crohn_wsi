
Prediction of prognosis for Crohn's disease <br>

"Deep learning analysis of histologic images from intestinal specimen reveal "adipocyte shrinkage" and mast cell infiltration to predict post-operative Crohn's disease."　Hiroki Kiyokawa, Masatoshi Abe, Takahiro Matsui, Masako Kurashige, Kenji Ohshima, Shinichiro Tahara, Satoshi Nojima, Takayuki Ogino, Yuki Sekido, Tsunekazu Mizushima, Eiichi Morii(https://doi.org/10.1016/j.ajpath.2022.03.006).

##### analyze<br>
Segmentation of adipocytes using classical techniques to calculate cell number, cell-cell distance, shape, unsegmented foreground area, etc.

##### src <br>
Efficiently create a patch with the tissue part transferred from WSI, and learn a model that predicts whether the prognosis is poor or not on a patch-by-patch basis with CNN.<br>
A label is attached to each slide.

##### config <br>

Experimentally adjusted hyperparameters


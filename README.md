# BERT-CNN-models
Models that use BERT + Chinese Glyphs for NER

# Models
- autoencoder.py: Stand-alone autoencoder for GLYNN, takes in image files
- glyph_birnn.py: Full model that contains BiLSTM-CRF and gets embeddings from BERT and glyph CNNs
- glyph.py: Helper file that contains strided CNN and GLYNN CNN

# Important Info

- Trainer script is not released, because it has identifying information of the authors
- Trainer script and image files necessary to reproduce this paper will be released as soon as the paper is accepted

_base_ = ['../otx/model.py']

model = dict(
    type='SAMImageClassifierTrainOnlyClassifier',
    head=dict(in_channels=768)
)

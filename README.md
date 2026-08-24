# Prediction of Metachronous Liver Metastasis in Colorectal Cancer (CMLM)

## Scope

This repository is a **concise architecture overview** of the multimodal CMLM
framework. It is intended to illustrate the main modeling principles described
in the paper, rather than provide a complete reproduction of the clinical study.

## Principle-level architecture

- **Image branch (MSEM):** an original tumour-centred image, an ROI retaining
  positional information, and an ROI without positional information are encoded
  by ResNet34, ViT-B/16, and ResNet18, respectively. Hierarchical SE attention is
  used to reinforce the combined multi-scale representation.
- **Numerical branch (SADM):** the nine selected numerical variables are modeled
  by a dense path in parallel with self-attention over feature-aware numerical
  tokens.
- **Multimodal fusion:** three image tokens attend over all nine numerical tokens.
  Residual fusion retains image information, and the image, numerical, and
  cross-modal representations are combined for binary prediction.
- **Training defaults:** Adam optimizer, initial learning rate `0.001`, suggested
  batch size `32`, and focal loss for class imbalance.

## Expected inputs

`MultiModalModel.forward` accepts three image tensors of shape
`[batch, 3, 224, 224]` and one numerical tensor of shape `[batch, 9]`:

```python
logits = model(
    original_image,
    roi_with_position,
    roi_without_position,
    numerical_features,
)
```

The ordering, units, imputation, and normalization of the nine numerical
variables must remain fixed between training and inference.

## Intentionally not included

Patient data, MA-YOLO segmentation, radiologist quality control, Boruta/Lasso
feature selection, cohort splitting, five-fold cross-validation, external
validation, trained weights, SHAP/Grad-CAM analysis, and the online clinical
application are not included in this small framework repository.

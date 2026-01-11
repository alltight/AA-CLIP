# Report
## Summary of AA-CLIP
We choose [AA-CLIP](https://arxiv.org/pdf/2503.06661) as the baseline for our project. The following is a summary of the paper. We first introduce the anomaly detection problem, and then describe the AA-CLIP method and its experimental results.
#### Anomaly Detection
The paper targets zero-shot anomaly detection (both image-level classification and pixel-level localization) using CLIP, motivated by the observation that standard CLIP is “anomaly-unaware”: textual descriptions of normal and abnormal conditions tend to be embedded too closely, which weakens the contrast needed to distinguish anomalies, especially when transferring to unseen categories and across domains such as industrial inspection and medical imaging.
#### Method Summary
AA-CLIP introduces a lightweight, two-stage adaptation strategy with residual adapters. In Stage 1, the visual encoder is frozen while shallow layers of the text encoder are equipped with adapters to learn anomaly-aware text anchors (normal vs. anomaly), trained with alignment losses and a disentanglement (orthogonality) loss to explicitly separate normal and anomalous text embeddings. In Stage 2, the adapted text encoder is frozen and adapters are added to shallow layers of the visual encoder, aligning multi-granularity patch features (from multiple transformer layers) to the fixed text anchors, improving localization. This staged design avoids the representation collapse observed with one-stage joint adaptation and preserves zero-shot generalization.
#### Results
Across a broad set of industrial and medical benchmarks, AA-CLIP achieves state-of-the-art or near–state-of-the-art AUROC at both image and pixel levels, with particularly strong gains on challenging medical datasets. The method is also data-efficient, maintaining strong performance with very few training samples per class. Ablation studies show that residual adapters, staged training, and the disentanglement loss are all critical, while removing them degrades both detection accuracy and localization quality.

**Figure 1: The Pipeline of AA-CLIP.**
![The Pipeline of AA-CLIP](original_pipeline.png)

## Motivation
Upon analyzing the architecture and loss design of the baseline AA-CLIP, we identify two primary limitations that may hinder optimal performance: 

**Risk of Semantic Forgetting in Disentanglement.** The original disentanglement loss employs a direct modulus-based approach to force a separation between the semantic vectors of normal and anomalous objects. While effective at creating distance, this unconstrained optimization can lead to semantic drift, where the embeddings lose the intrinsic semantic information of the object itself. We argue that the training process requires a stable reference point (anchor) to to maintain the original object sementics thile learning the anomaly distinction.  

**Coarse Path Integration.** In the original method, feature pathces from different layers are aggregated via simple summation, and the final anomaly mask is generated using bilinear interpolation. We consider this approach to be heuristic and coarse. It fails to fully leverage the hierachical semantic information present in different transformer layer, nor does it utilize the global object semantics to guide the localization of of anomalies. 
## Proposed Method
To address the limitations highlighted above, we propose the following three modifications: 

**Base Text Anchor & Residual Orthogonality.** We introduce a Base Text Anchor( $T_{base}$ ), defined as the text embedding of the object's class name(without any prompt engineering) encoded by the frozen, pre-trained CLIP text encoder. This serves as a fixed semantic reference. 
Instead of directly optimizing the distance between the learnable normal (T_N) and anomalous (T_A) embeddings, we optimize the orthogonality of their residual vectors relative to the base anchor. Let $\Delta_N = T_N - T_{base}$ and $\Delta_A = T_A - T_{base}$.The orthogonality loss is defined as: $$\mathcal{L}_{1} = \left| \frac{\Delta_N}{\|\Delta_N\|} \cdot \frac{\Delta_A}{\|\Delta_A\|} \right|^2$$ Simultaneously, we introduce a semantic consistency constraint to prevent the learnable embeddings from drifting too far from the base semantics: $$\mathcal{L}_{2} = (1 - \text{CosSim}(T_N, T_{base})) + (1 - \text{CosSim}(T_A, T_{base}))$$ The final refined disentanglement loss is formulated as: $$\mathcal{L}_{dis} = \lambda_1 \mathcal{L}_1 + \lambda_2 \mathcal{L}_2$$ Thie design ensures that the model preserves the core semantic information of the object while maximizing the directional distinction between normal and anomalous states

**Cross-Attention-Gated Features.** To resolve the issue of coarse feature aggregation, we introduce a Cross-Attention Gating Machanism. Instead of simple summation, we utilize the object's semantic embedding as the Query, and the multi-scale feature patches as the Key and Value. This machanism generates a gating signal that dynamically weighs the importance of different patches based on their semantic relevance, allowing for more refined feature integration.

**Segmentation Decoder.** To improve the resolution and quality of the final anomaly map, we replace the standard bilinear interpolation with a Segmentation Decoder utilizing PixelShuffle layers. This modification leverages the neural network's capacity to learn complex upsampling mapping, resulting in sharper and more accurate anomaly localization compared to heuristic interpolation.

## Experimental Results
Following the baseline settings, we train our model on VisA, a visual anomaly detection dataset for industrial inspection, and evaluate it on multiple anomaly detection datasets across both industrial and medical domains. We adopt the Area Under the Receiver Operating Characteristic Curve (AUROC) as the evaluation metric at both pixel and image levels, where higher values indicate better anomaly detection performance. We evaluate three proposed modifications to the original method. The results are reported in Tables 1 and 2.

**Table 1: Pixel-level AUROC of the baseline and our method on industrial and medical domains.** We report both the original paper results and our reproduced results, which show slight differences. Our method comprises three composable components: (1) base text anchor, (2) cross-attention–gated features, and (3) segment decoder.
|Method <br> ----------- <br> Dataset| Paper | Reproduction | (1) | (2) | (1) + (2) | (3) | (1) + (2) + (3) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|BTAD|97.0| 97.27|96.90|97.44|97.10|90.17|81.29|
|MPDD|96.7|96.54|96.28|96.28|96.56|94.01|94.25|
|MVTec-AD|91.9|91.43|91.95|92.29|91.56|87.18|87.17|
|Brain MRI|95.5|94.95|95.40|96.44|97.17|91.25|90.18|
|Liver CT|97.8|97.51|97.69|97.57|97.7|93.68|96.48|
|Retina OCT|95.5|95.80|96.15|96.31|95.77|85.12|87.61|
|ColonDB|84.0|83.63|84.01|83.94|84.66|80.87|80.36|
|ClinicDB|89.9|90.43|90.10|89.57|90.65|85.59|83.26|
|Kvasir|87.2|87.64|87.82|87.19|88.16|82.57|80.66|
|CVC-300|96.4|96.35|96.22|97.02|96.09|96.14|94.7|

**Table 2: Image-level AUROC of the baseline and our method on industrial and medical domains.** We report both the original paper results and our reproduced results, which show slight differences. Our method comprises three composable components: (1) base text anchor, (2) cross-attention–gated features, and (3) segment decoder.
|Method <br> ----------- <br> Dataset| Paper | Reproduction | (1) | (2) |(1) + (2)| (3) | (1) + (2) + (3) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|BTAD|94.8|94.98|95.33|94.66|95.10|89.3|87.29|
|MPDD|75.1|74.25|76.55|74.90|75.96|71.96|65.93|
|MVTec-AD|90.5|89.78|89.10|90.45|89.86|87.09|83.83|
|Brain MRI|80.2|77.18|79.37|83.41|84.41|68.69|67.81|
|Liver CT|69.7|65.63|63.73|65.6|60.32|67.83|68.08|
|Retina OCT|82.7|83.22|84.17|85.44|82.6|61.93|65.69|

**Base Text Anchor.** We introduce an additional base text anchor—specifically, the frozen text embedding from the original CLIP model—to refine the disentanglement loss proposed in the original paper. However, no noticeable improvement or degradation is observed in either pixel-level or image-level AUROC. We attribute this to the fact that, in high-dimensional spaces, orthogonality is a relatively weak constraint. Consequently, whether we enforce the normal and anomalous text embeddings to be orthogonal to each other, or instead minimize the inner product between their offsets with respect to a fixed base embedding, the impact on the training dynamics is limited. Both approaches preserve the same property of disentangling text embeddings to facilitate anomaly detection, which likely explains the similar performance.

**Cross-Attention–Gated Features.** We introduce a cross-attention module to gate the image patch features, based on the assumption that text embeddings can provide guidance on which image regions should be attended to. The results show that, although performance on most datasets remains unchanged, the image-level AUROC on the Brain MRI and Liver CT datasets is improved. We hypothesize that certain textual descriptions contain richer information about the expected appearance of the images; therefore, injecting text information via cross-attention helps better discriminate between normal and anomalous images in these cases. However, the exact underlying mechanism remains unclear. In MR and Liver CT, image-level AUROC is improved. We guess that it is because some word contains more information about how the image should look like, so when we introduce the text information via cross attention, it can descriminate between normal and anomal image better, but the exact reason is still unknown. 

**Segmentation Decoder.** Contrary to our initial hypothesis, the inclusion of a Segmentation Decoder led to a significant performance degradation in both pixel-level and image-level AUROC across several datasets.We attribute this performance collapse to the following two factors:
- Overfitting and Loss of Generalization:The original AA-CLIP relies on the frozen, rich representation of CLIP to achieve zero-shot generalization. By introducing a complex, trainable decoder with PixelShuffle layers, the model likely overfitted to the specific distribution of the training samples.This localized optimization undermines the "anomaly-unaware" flexibility of CLIP, making the decoder struggle to generalize to unseen anomaly types of out-of-distribution industrial/medical samples.
- Distortion of Latent Patch Features: During the backpropagation process, the gradients from the segmentation loss may have adversely altered the pre-aligned patch features. In the baseline AA-CLIP, these features are carefully aligned with text anchors in a shared embedding space. Our complex decoder might have forced these features to prioritize reconstruction or upsampling patterns rather than maintaining their semantic alignment with the text anchors, thereby destroying the contrastive foundation required for accurate anomaly detection.

## Author Contributions

Ziye Huang implemented the base text anchor and cross-attention gating module and performed the corresponding experiments. 

Zhuolin Yu implemented the segmentation decoder, integrated all proposed techniques, and performed the corresponding experiments.

Our repository: https://github.com/alltight/AA-CLIP
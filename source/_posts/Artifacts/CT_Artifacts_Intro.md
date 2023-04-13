---
title: CT Artifacts(Intro)
date: 2017-05-31 10:05:56
mathjax: true
categories:
  - [Computed_Tomography, Artifacts]
tags:
  - CT
  - Artifacts
  - CT Artifacts
---



## Artifact Definition

Any systematic discrepancy between the CT numbers in the reconstructed image and the true attenuation coefficients of the object. [^1]

{% blockquote [author[, source]] [link] [source_link_title] %}
这里是引用
{% endblockquote %}

## Appearance of CT artifacts

• **Streaks** – due to an inconsistency in a single measurement

• **Shading** – due to a group of channels or views deviating gradually from the true measurement

• **Rings** – due to errors in an individual detector calibration

• **Distortion** – due to helical reconstruction

## Sources of artifacts come from four categories

• **Physics-based** – result from the physical processes involved in the acquisition of CT data

• **Patient-based** – caused by such factors as patient movement or the presence of metallic materials in or on the patient

• **Scanner-based** – result from imperfections in scanner function

• **Helical and** **multi-section** **artifacts** – produced by the image reconstruction process

### _Physics based (4 types)_

1. **Beam hardening –** due to average E of beam increasing as beam traverses object. Results in two different visual artifacts: 1) cupping, and 2) streaks and dark bands.

   > When X-rays pass through high-density objects such as bones or metals, their likelihood of interacting with the atomic nuclei or electrons in the material is higher, which causes a change in the average energy of the X-rays. In this case, _the average energy of the X-rays_ may increase because of the photoelectric effect and Compton scattering that occur when they interact with the material, with the photoelectric effect potentially causing an increase in the X-ray energy.
   >
   > small $u$ ->dark large $u$ -> light

   - **Cupping**: X-rays passing through the center of an object experience greater attenuation (and thus more hardening) than those that pass through the edges of the object --showing as light at the edge and dark inside

        <img src="https://lh3.googleusercontent.com/Yd5VAEcgDijflRyX6XRVFmYqOQeW6xmoaHxDFYPtTTJZVKsZDErsGwNAK9dSPIDf3AdEWDFvYDlMMXKBNkktc_RRFF4yjIKnfBnsVw9n#pic_center" alt="cupping artifacts" style="zoom: 25%;" />

   - **Streaks and dark bands:** Appear between two high density objects (such as bone or contrast media). Due to greater hardening at angles that beam passes through both structures than just one. Many more clinical examples on this than cupping.

      <img src="https://miro.medium.com/v2/resize:fit:634/1*vlf-2WNLt4X_8zHw1FlSGw.png" alt="cupping artifact" style="zoom:50%;" />

   There are three method can be used to **minimize beam hardening**:

   - **filtration (bow tie)** -- Bowtie filtration hardens edges more than center

     <img src="https://aapm.onlinelibrary.wiley.com/cms/asset/b19bc12c-7021-467a-8cda-f0411f602e37/mp7470-fig-0002-m.png" alt="filtration" style="zoom: 67%;" />

   - **calibration correction** -- manufacturer calibrates beams for different body parts using phantoms. Usually helps with cupping artifact but can still result in small cupping or capping (opposite effect) due to differences between patient and phantom.

   - **software** -- usually uses iterative reconstruction. Improves bone/tissue interfaces and streaking

2. **Partial volume –** object within FOV at certain angles and not within FOV at other angles

   - Corrected by using thin slices. If noise in slices becomes an issue, sum slices (but don’t raise slice width).

   > It occurs when:
   >
   > - The slice is too thick
   > - only contain projection from limited angles
   > - slice only contain some part of our object

3. **Photon starvation –** attenuation is very great at certain angles creating noisy projections. Reconstruction amplifies noise and results in streaking at these projections

   <img src="https://oncologymedicalphysics.com/wp-content/uploads/2018/07/streaking-artifact.png" alt="Photon starvation" style="zoom:33%;" />

   > For example, shoulder in this image, is very thick from the right-and-left, so the photon will attenuated very quickly. However, it is very thin from the up-and-down. It will causes streaking artifacts when reconstruction.

- Corrected using tube current modulation or adaptive filtering.

  <img src="https://lh3.googleusercontent.com/WHN-7DplJqlGRogFbZj6yZ-tBJ0zA2Mo-Mj3GhYAocdF0yIJqXlbL9WZykq9jdyKfYs098w4ma_93Q4V7O5qvFCSnJwoLgqU5eoD4FS7" alt="correct photon starvation" style="zoom: 18%;" />

4. **Angular** **under-sampling** **–** too few angles acquired creates view aliasing where fine lines appear to radiate from the edges of a structure

   <img src="https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcSLJsSPJ5aAtCqRAFDJXwZb3YorypWSUWkqwCTeNyRwlJVYz-3_" style="zoom:50%;" />

   - Corrected using more angles or special techniques that increase data acquired per angle such as quarter-detector shift or flying-focal spot.

### **_Patient-based (3 types)_**

1. **Metallic materials –** presence of metal creates streaking artifacts due to beam hardening and/or photon starvation

   - Corrected by removing metal objects and/or software

     <img src="https://lh3.googleusercontent.com/Y3xHgh4CY5m_QloMp7EkHzXI-QOZQW04eEat8FdixoR4BWsWkqy-kqBKG2X0jNM5Zd7MHzOjw7IcWsL1nMUh6R8cggbwNNqB-qwSTjkWEQ" style="zoom:20%;" />

2. **Motion –** causes misregistration artifacts that lead to shading or streaking

   - Corrected by:

     - **Positioning –** ensure patient is comfortable and that involuntary motions of un-important organs are not included in the FOV

   - **Restraints –** often needed for children
   - **Breath-holding –** removes respiratory motion
   - **Overscanning** **–** acquire more than 360 degrees and average beginning and ending (since motion discrepency is greatest at these points)
     - **Software –** de-weight beginning and ending angles
   - **Cardiac gating –** only acquire at certain phases of cardiac cycle (e.g. end-diastole)

3. **Incomplete projections**

   - Correct by:

     - ensuring no objects (such as body parts or tubes of contrast material) lie outside the FOV. Some manufacturers attempt to recognize objects moving in and out of FOV.

     - Similar to dense tube containing contrast media outside the scan field; or Blocking the reference channels at the sides of the detectors

     - using large bore

### _**Scanner-based (1 types)**_

1. **Ring artifacts –** out of calibration or dead detector element creates circular artifact. Worse if element is near the center of the detector as this may _create a hole in the center of the image_.

   <img src="https://lh3.googleusercontent.com/GIAScyDssE4GSCTvKUyifMXgrjRM4MkwdiEBk0PHR5O-O7vRnaOHnemQ8oEm9bHehxO4joen8jmhmpz-p9WDUt63w9kGIarK5qcR1S5s3w" style="zoom:20%;" />

### _**Helical and** **multi-section** **artifacts (4 types)**_

1. **Single section artifacts –** interpolation of slices creates distortion. More pronounced at large pitch values and when object changes rapidly in z-direction.

   <img src="https://lh3.googleusercontent.com/6q0F4hqggcW7FS37KXx3siKNXDy7DuxzvybkhINDomkRF9eVOUiAtzr1PUDiaOrI58nKlPa4aFBiEDasDJE59NYdKCX0ekYElT9uhmI" style="zoom:19%;" />

2. **Multi-section** **artifacts –** sources of artifacts are more complicated, but commonly have a windmill appearance.

   <img src="https://lh3.googleusercontent.com/8FjdQ1NgrdwIIq0wjtKV1gLSN50PYQOUnLb6OaoPIwmIvsU1Jbh4BXLZqhpsbnv7VV0fanHM3IY--zZvtUxevRke21BffLGc2wcqQbIb7g" style="zoom:37%;" />

   - Corrected by using z-filter interpolators or non-integer values for pitch (e.g. 3.5 or 4.5)

3. **Cone beam effects –** outer edges of FOV contain information not seen in other views (similar to partial volume effects)

   <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRckcuI2YFGG9VZWRhusDnn6ISSgIfQXqDazA&usqp=CAU" style="zoom:67%;" />

   - Corrected using non-standard reconstruction techniques

4. **Multi-planer** **and three-dimensional reformation**

   - **Stair-step artifacts** -- Virtually eliminated on modern multislice scanners using thin sections
   - **Zebra artifacts** – faint stripes due to enhanced noise inhomogeneity along z-axis.

## Reference

[^1]: [https://www.bilibili.com/video/BV1uJ411W7ki/?spm_id_from=333.999.0.0&vd_source=1725680e7f1b4515313eda07848d77c1](https://www.bilibili.com/video/BV1uJ411W7ki/?spm_id_from=333.999.0.0&vd_source=1725680e7f1b4515313eda07848d77c1)

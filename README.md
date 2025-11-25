# Low-Light Image Enhancement – MATLAB Assignment

This project uses the pretrained **“Learning to See in the Dark”** deep network to enhance a low-light image without performing any training or using the full SID dataset.

The goal of the assignment is to:
1. Load a custom image.
2. Simulate a very dark and noisy version of it.
3. Apply the pretrained enhancement model.
4. Make one small modification to the provided script.
5. Submit the code, the input image, and the enhanced result.

---

## ✔ Files Included
- `lowLightDemo.m` — main MATLAB script.
- `Example_03.png` — input image used in the experiment.
- `result.png` — saved enhanced output from the network.

---

## ✔ Change Implemented
I modified the **darkness factor** used to simulate low-light conditions.

Original:
```matlab
darkFactor = 0.03;
My change:

matlab
Copiar código
brightnessFactor = 0.05;
This makes the simulated low-light image even darker, so the pretrained network must compensate more aggressively during enhancement. The rest of the workflow remains unchanged.

## ✔ Output Description
Original RGB: the uploaded image.

Simulated low-light: darker + gaussian noise.

Enhanced image: reconstruction produced by the pretrained network.

The results match the expected behavior of the SID low-light pipeline.

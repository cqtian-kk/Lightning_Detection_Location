# Lightning_Detection_Location
All data is available at [DOI 10.5281/zenodo.13337026](https://zenodo.org/doi/10.5281/zenodo.13337025)

A Graph Neural Network Based Workflow for Real-time Lightning Location with Continuous Waveforms

This project consists of four main parts: the **Datasets** folder corresponds to generating the lightning datasets for specific work areas, the **Event_detection** folder is for detecting lightning waveform segments from continuous waveforms, the **GNNs** folder contains the lightning location algorithm we used, and the **TR** folder includes the traditional algorithm we used for comparison.

### Datasets:
- **Createwaveform.py**: Generates noise-free lightning waveforms based on the physical model described in the paper (Figure 5c).
- **Changewaveform.py**: Creates a diverse lightning waveform dataset (Figure 6).
- **addnoise.py**: Adds noise to the lightning waveform dataset (Figure 5e).

### Event_detection:
- **dataprocess_EMD.py**: Performs EMD denoising on the raw waveforms and integrates waveform data (this script cannot be run as a whole; it needs to be executed in segments).
- **Multi-channel_intercor.py**: Detects lightning waveform segments from continuous waveforms.

### GNNs:
- **train_gnn_PLAN2.py**: Trains the model.
- **pre_temp.py**: Uses the trained model to locate lightning using detected waveform segments and applies the GNN algorithm for positioning.
- **plot_GNN_result.py**: Visualizes GNN location results (excluding unreliable events).

### TR:
- **TimeReversal_gpu_val.py**: Locates lightning using detected waveform segments and the TR algorithm.
- **plot_TR_result.py**: Visualizes TR location results (excluding unreliable events).


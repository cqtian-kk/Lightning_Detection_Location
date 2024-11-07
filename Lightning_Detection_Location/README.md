A Graph Neural Network Based Workflow for Real-time Lightning Location with Continuous Waveforms

这个项目由四个部分构成，Datasets文件对应生成对应工区的闪电数据集，Event_detection文件对应从连续波形中检测闪电波形片段，GNNs文件对应我们使用的闪电定位算法，TR文件对应我们用于对比我们效果的传统算法。

Datasets：
    Createwaveform.py：根据文章所示物理模型生成不含噪声的闪电波形（图5c）。
    Changewaveform.py：生成多种多样的闪电波形数据集（图6）
    addnoise.py：对闪电波形数据集进行加噪声（图5e）

Event_detection：
    dataprocess_EMD.py：对原始波形进行EMD去噪，以及整合波形数据（这个代码不能直接运行整个文件，需要分段运行）
    Multi-channel_intercor.py：从连续波形中检测闪电波形片段

GNNs：
    train_gnn_PLAN2.py：训练模型
    pre_temp.py:根据训练好的模型，使用检测到的闪电波形片段，并采用GNN算法进行定位
    plot_GNN_result.py：GNN定位结果可视化（排除不可靠事件）

TR：
    TimeReversal_gpu_val.py：使用检测到的闪电波形片段，并采用TR算法进行定位
    plot_TR_result.py：TR定位结果可视化（排除不可靠事件）
    






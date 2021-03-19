```
root@code-machine:/dockerx# python main.py --config ./config/config.yaml
2021-03-19 06:29:02,335 Getting config parameters...
2021-03-19 06:29:02,335 -----------config parameters------------
2021-03-19 06:29:02,335 ---split_ratio = 0.2
2021-03-19 06:29:02,336 ---random_state = 0
2021-03-19 06:29:02,336 ---max_len = 128
2021-03-19 06:29:02,336 Loading training and test data...
2021-03-19 06:29:03,117 Parsing input text...
2021-03-19 06:29:12,673 Spliting datasets, the split ratio is 0.2, random state is 0
2021-03-19 06:29:12,686 Parsing test dataset...
2021-03-19 06:29:21,206 Loading done.
2021-03-19 06:29:21,219 Vecterizing data for neural network training...
2021-03-19 06:29:21,219 Creating vocabulary....
2021-03-19 06:29:22,478 Done. Got 182607 words
2021-03-19 06:29:22,478 Preparing data for training...
2021-03-19 06:29:28,780 Creating textCNN model...
2021-03-19 06:29:28,780 -------Training config parameters-------
2021-03-19 06:29:28,780 ---model_name = text_cnn
2021-03-19 06:29:28,780 ---max_input_len = 128
2021-03-19 06:29:28,780 ---batch_size = 32
2021-03-19 06:29:28,780 ---dropout = 0.5
2021-03-19 06:29:28,780 ---epochs = 1
2021-03-19 06:29:28,781 ---num_of_classes = 6
2021-03-19 06:29:28.792909: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libamdhip64.so
2021-03-19 06:29:28.814622: E tensorflow/stream_executor/rocm/rocm_driver.cc:982] could not retrieve ROCM device count: HIP_ERROR_NoDevice
2021-03-19 06:29:28.814636: E tensorflow/stream_executor/rocm/rocm_driver.cc:982] could not retrieve ROCM device count: HIP_ERROR_NoDevice
2021-03-19 06:29:28.847252: I tensorflow/core/platform/profile_utils/cpu_utils.cc:104] CPU Frequency: 3600000000 Hz
2021-03-19 06:29:28.847946: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x52b4fb0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2021-03-19 06:29:28.847994: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2021-03-19 06:29:28.852477: E tensorflow/stream_executor/rocm/rocm_driver.cc:982] could not retrieve ROCM device count: HIP_ERROR_NoDevice
2021-03-19 06:29:28.879720: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 93494784 exceeds 10% of free system memory.
2021-03-19 06:29:28.901916: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 93494784 exceeds 10% of free system memory.
2021-03-19 06:29:28.911786: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 93494784 exceeds 10% of free system memory.
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 128, 128)          23373696
_________________________________________________________________
conv1d (Conv1D)              (None, 128, 128)          114816
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 64, 128)           0
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 64, 256)           164096
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 32, 256)           0
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 32, 512)           393728
_________________________________________________________________
max_pooling1d_2 (MaxPooling1 (None, 16, 512)           0
_________________________________________________________________
flatten (Flatten)            (None, 8192)              0
_________________________________________________________________
dense (Dense)                (None, 128)               1048704
_________________________________________________________________
dropout (Dropout)            (None, 128)               0
_________________________________________________________________
dense_1 (Dense)              (None, 6)                 774
_________________________________________________________________
dense_2 (Dense)              (None, 6)                 42
=================================================================
Total params: 25,095,856
Trainable params: 25,095,856
Non-trainable params: 0
_________________________________________________________________
2021-03-19 06:29:29.090475: I tensorflow/core/common_runtime/gpu_fusion_pass.cc:508] ROCm Fusion is enabled.
2021-03-19 06:29:29.108358: I tensorflow/core/common_runtime/gpu_fusion_pass.cc:508] ROCm Fusion is enabled.
2021-03-19 06:29:29.114880: I tensorflow/core/common_runtime/gpu_fusion_pass.cc:508] ROCm Fusion is enabled.
2021-03-19 06:29:29.267489: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 93494784 exceeds 10% of free system memory.
2021-03-19 06:29:29.277539: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 93494784 exceeds 10% of free system memory.
2021-03-19 06:29:29.518164: I tensorflow/core/common_runtime/gpu_fusion_pass.cc:508] ROCm Fusion is enabled.
3990/3990 [==============================] - ETA: 0s - accuracy: 0.9648 - loss: 0.06832021-03-19 06:41:46.824427: I tensorflow/core/common_runtime/gpu_fusion_pass.cc:508] ROCm Fusion is enabled.
2021-03-19 06:41:46.827334: I tensorflow/core/common_runtime/gpu_fusion_pass.cc:508] ROCm Fusion is enabled.
2021-03-19 06:41:46.829506: I tensorflow/core/common_runtime/gpu_fusion_pass.cc:508] ROCm Fusion is enabled.
3990/3990 [==============================] - 742s 186ms/step - accuracy: 0.9648 - loss: 0.0683 - val_accuracy: 0.9936 - val_loss: 0.0609
2021-03-19 06:41:52.039864: I tensorflow/core/common_runtime/gpu_fusion_pass.cc:508] ROCm Fusion is enabled.
2021-03-19 06:41:52.042856: I tensorflow/core/common_runtime/gpu_fusion_pass.cc:508] ROCm Fusion is enabled.
2021-03-19 06:41:52.044647: I tensorflow/core/common_runtime/gpu_fusion_pass.cc:508] ROCm Fusion is enabled.
2021-03-19 06:41:52.110830: I tensorflow/core/common_runtime/gpu_fusion_pass.cc:508] ROCm Fusion is enabled.
2021-03-19 06:41:57,044 Accuracy : 0.9165596114679617
2021-03-19 06:41:57,045
              precision    recall  f1-score   support

           0       0.93      0.62      0.74      3101
           1       0.58      0.05      0.08       329
           2       0.86      0.68      0.76      1698
           3       1.00      0.00      0.00        91
           4       0.78      0.57      0.66      1594
           5       1.00      0.00      0.00       298

   micro avg       0.87      0.56      0.68      7111
   macro avg       0.86      0.32      0.37      7111
weighted avg       0.87      0.56      0.66      7111
 samples avg       0.99      0.94      0.94      7111


2021-03-19 06:41:57.089783: I tensorflow/core/common_runtime/gpu_fusion_pass.cc:508] ROCm Fusion is enabled.
2021-03-19 06:41:57.093566: I tensorflow/core/common_runtime/gpu_fusion_pass.cc:508] ROCm Fusion is enabled.
2021-03-19 06:41:57.095330: I tensorflow/core/common_runtime/gpu_fusion_pass.cc:508] ROCm Fusion is enabled.
2021-03-19 06:42:20,292 Saving prediction result to a csv file...
2021-03-19 06:42:21,144 Done. Prediction completed!

```

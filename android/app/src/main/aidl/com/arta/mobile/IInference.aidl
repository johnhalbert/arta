package com.arta.mobile;

import com.arta.mobile.IReplySink;

interface IInference {
    int workerPid();
    String[] vulkanDevices();
    void loadModel(String path, int gpuLayers);
    void setGpuLayers(int layers);
    boolean hasModel();
    void reset();
    void restoreTurn(String prompt, String reply);
    boolean send(String prompt, IReplySink sink);
}

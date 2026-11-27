package com.arta.mobile;

interface IReplySink {
    void onChunk(String chunk);
    boolean isCancelled();
}

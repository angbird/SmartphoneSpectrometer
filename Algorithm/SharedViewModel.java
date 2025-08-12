package com.example.watermonitor;

import android.graphics.Bitmap;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

public class SharedViewModel extends ViewModel {
    private final MutableLiveData<Bitmap> sampleBitmap = new MutableLiveData<>();

    public void setSampleBitmap(Bitmap bitmap) {
        sampleBitmap.setValue(bitmap);
    }

    public LiveData<Bitmap> getSampleBitmap() {
        return sampleBitmap;
    }
}
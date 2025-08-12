package com.example.watermonitor;

import android.app.Application;
import androidx.annotation.NonNull;
import androidx.lifecycle.AndroidViewModel;
import androidx.lifecycle.LiveData;

import java.util.List;

public class ItemSetViewModel extends AndroidViewModel {
    private ItemRepository repository;
    private LiveData<List<ItemSet>> allItemSets;

    public ItemSetViewModel(@NonNull Application application) {
        super(application);
        repository = new ItemRepository(application);
        allItemSets = repository.getAllItemSets();
    }

    public LiveData<List<ItemSet>> getAllItemSets() {
        return allItemSets;
    }

    // 新增分页加载方法
    public List<ItemSet> getItemSetsWithPagination(int pageSize, int offset) {
        return repository.getItemSetsWithPagination(pageSize, offset);
    }

    public void insertItemSet(ItemSet itemSet) {
        repository.insertItemSet(itemSet);
    }

    public void deleteItemSet(ItemSet itemSet) {
        repository.deleteItemSet(itemSet);
    }

}


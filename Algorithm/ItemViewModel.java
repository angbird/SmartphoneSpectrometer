package com.example.watermonitor;

import android.app.Application;
import androidx.annotation.NonNull;
import androidx.lifecycle.AndroidViewModel;
import androidx.lifecycle.LiveData;

import java.util.List;

public class ItemViewModel extends AndroidViewModel {
    private ItemRepository repository;
    private LiveData<List<Item>> allItems;

    private LiveData<List<Item>> allItemsExcludingBitmaps;
    private LiveData<List<ItemSet>> allItemSets;

    public ItemViewModel(@NonNull Application application) {
        super(application);
        repository = new ItemRepository(application);
        allItems = repository.getAllItems();
        allItemsExcludingBitmaps = repository.getAllItemsExcludingBitmaps();
        allItemSets = repository.getAllItemSets();
    }

    public LiveData<List<Item>> getAllItems() {
        return allItems;
    }


    public LiveData<List<Item>> getAllItemsExcludingBitmaps() {
        return allItemsExcludingBitmaps;
    }

    public LiveData<List<ItemSet>> getAllItemSets() {
        return allItemSets;
    }

    public LiveData<List<Item>> getItemsByName(String name) {
        return repository.getItemsByName(name);
    }

    public void insert(Item item) {
        repository.insert(item);
    }

    public void insertItemSet(ItemSet itemSet) {
        repository.insertItemSet(itemSet);
    }

    public void deleteItem(Item item) {
        repository.delete(item);
    }
}

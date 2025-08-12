package com.example.watermonitor;

import static com.example.watermonitor.ItemDatabase.databaseWriteExecutor;

import android.app.Application;
import androidx.lifecycle.LiveData;

import java.util.List;

public class ItemRepository {
    private ItemDao itemDao;
    private ItemSetDao itemSetDao;
    private LiveData<List<Item>> allItems;
    private LiveData<List<ItemSet>> allItemSets;

    public ItemRepository(Application application) {
        ItemDatabase database = ItemDatabase.getInstance(application);
        itemDao = database.itemDao();
        allItems = itemDao.getAllItems();
        itemSetDao = database.itemSetDao();
        allItemSets = itemSetDao.getAllItemSets();
    }

    public LiveData<List<Item>> getAllItems() {
        return allItems;
    }

    public LiveData<List<ItemSet>> getAllItemSets() {
        return allItemSets;
    }

    public LiveData<List<Item>> getAllItemsExcludingBitmaps() {
        return itemDao.getAllItemsExcludingBitmaps();
    }

    public LiveData<List<Item>> getItemsByName(String name) {
        return itemDao.getItemsByName(name);
    }

//    public void insert(Item item) {
//        new InsertItemAsyncTask(itemDao).execute(item);
//    }
    public List<ItemSet> getItemSetsWithPagination(int pageSize, int offset) {
        return itemSetDao.getItemSetsWithPagination(pageSize, offset);
    }

    public void insert(Item item) {
        databaseWriteExecutor.execute(() -> itemDao.insert(item));
    }

    public void insertItemSet(ItemSet itemSet) {
        databaseWriteExecutor.execute(() -> itemSetDao.insert(itemSet));
    }

    public void deleteItemSet(ItemSet itemSet) {
        databaseWriteExecutor.execute(() -> itemSetDao.delete(itemSet));
    }

    public void delete(Item item) {
        databaseWriteExecutor.execute(() -> itemDao.delete(item));
    }

    private static class InsertItemAsyncTask extends android.os.AsyncTask<Item, Void, Void> {
        private ItemDao itemDao;

        private InsertItemAsyncTask(ItemDao itemDao) {
            this.itemDao = itemDao;
        }

        @Override
        protected Void doInBackground(Item... items) {
            itemDao.insert(items[0]);
            return null;
        }
    }
}

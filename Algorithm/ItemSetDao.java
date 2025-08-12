package com.example.watermonitor;

import androidx.lifecycle.LiveData;
import androidx.room.Dao;
import androidx.room.Delete;
import androidx.room.Insert;
import androidx.room.OnConflictStrategy;
import androidx.room.Query;

import java.util.List;

@Dao
public interface ItemSetDao {
    @Insert
    void insert(ItemSet itemSet);

    @Delete
    void delete(ItemSet itemSet);

    @Query("SELECT * FROM item_set_table")
    LiveData<List<ItemSet>> getAllItemSets();

    @Query("SELECT * FROM item_set_table LIMIT :limit OFFSET :offset")
    List<ItemSet> getItemSetsWithPagination(int limit, int offset);
}



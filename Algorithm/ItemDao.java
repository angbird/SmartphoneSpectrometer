package com.example.watermonitor;

import androidx.lifecycle.LiveData;
import androidx.room.Dao;
import androidx.room.Delete;
import androidx.room.Insert;
import androidx.room.Query;

import java.util.List;

@Dao
public interface ItemDao {
    @Insert
    void insert(Item item);

    @Delete
    void delete(Item item);

    @Query("SELECT * FROM item_table ORDER BY name ASC")
    LiveData<List<Item>> getAllItems();

    @Query("SELECT * FROM item_table WHERE name = :name")
    LiveData<List<Item>> getItemsByName(String name);

    @Query("SELECT id, name, PhoneName, DeviceNumber, Temperature, totalPhosphorus, ammoniaNitrogen, potassiumPermanganate, referenceTotalPhosphorus, referenceAmmoniaNitrogen, referencePotassiumPermanganate, TotalPhosphorus_spec, AmmoniaNitrogen_spec, PotassiumPermanganate_spec, TP_intensity, AN_intensity, PP_intensity, NoSample_intensity, time FROM item_table ORDER BY name ASC")
    LiveData<List<Item>> getAllItemsExcludingBitmaps();



}


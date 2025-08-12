package com.example.watermonitor;

import androidx.room.Entity;
import androidx.room.PrimaryKey;
import androidx.room.TypeConverters;

import java.util.List;

@Entity(tableName = "item_set_table")
public class ItemSet {
    @PrimaryKey(autoGenerate = true)
    private int id;
    private String name;
    private long time;
    private int number;

    @TypeConverters(ItemListConverter.class)
    private List<Item> items;

    public ItemSet(String name, List<Item> items) {
        this.name = name;
        this.time = System.currentTimeMillis();
        this.number = (items != null) ? items.size() : 0;
        this.items = items;
    }

    // Getters and setters...

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public long getTime() {
        return time;
    }

    public void setTime(long time) {
        this.time = time;
    }

    public int getNumber() {
        return number;
    }

    public void setNumber(int number) {
        this.number = number;
    }

    public List<Item> getItems() {
        return items;
    }

    public void setItems(List<Item> items) {
        this.items = items;
    }
}

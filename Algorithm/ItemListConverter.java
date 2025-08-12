package com.example.watermonitor;

import androidx.room.TypeConverter;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;
import java.util.List;

public class ItemListConverter {
    @TypeConverter
    public String fromItemList(List<Item> items) {
        if (items == null) {
            return null;
        }
        Gson gson = new Gson();
        Type type = new TypeToken<List<Item>>() {}.getType();
        return gson.toJson(items, type);
    }

    @TypeConverter
    public List<Item> toItemList(String itemString) {
        if (itemString == null) {
            return null;
        }
        Gson gson = new Gson();
        Type type = new TypeToken<List<Item>>() {}.getType();
        return gson.fromJson(itemString, type);
    }
}


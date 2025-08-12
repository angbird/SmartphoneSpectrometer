package com.example.watermonitor;

import androidx.room.Database;
import androidx.room.Room;
import androidx.room.RoomDatabase;
import androidx.room.TypeConverters;
import androidx.room.migration.Migration;
import androidx.sqlite.db.SupportSQLiteDatabase;

import android.content.Context;
import android.database.Cursor;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@Database(entities = {Item.class, ItemSet.class}, version = 9, exportSchema = false)
@TypeConverters({Converters.class, ItemListConverter.class})
public abstract class ItemDatabase extends RoomDatabase {
    private static volatile ItemDatabase INSTANCE;

    public abstract ItemDao itemDao();
    public abstract ItemSetDao itemSetDao();
    private static final int NUMBER_OF_THREADS = 4;
    static final ExecutorService databaseWriteExecutor = Executors.newFixedThreadPool(NUMBER_OF_THREADS);

    public static ItemDatabase getInstance(Context context) {
        if (INSTANCE == null) {
            synchronized (ItemDatabase.class) {
                if (INSTANCE == null) {
                    INSTANCE = Room.databaseBuilder(context.getApplicationContext(),
                                    ItemDatabase.class, "item_table")
                            .fallbackToDestructiveMigration() // 删除原数据并重新创建数据库
                            .build();
                }
            }
        }
        return INSTANCE;
    }
}


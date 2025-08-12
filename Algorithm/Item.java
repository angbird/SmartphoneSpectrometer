package com.example.watermonitor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Parcel;
import android.os.Parcelable;
import android.util.Base64;

import androidx.room.Entity;
import androidx.room.PrimaryKey;
import androidx.room.TypeConverters;

import java.io.ByteArrayOutputStream;

@Entity(tableName = "item_table")
@TypeConverters({Converters.class})
public class Item implements Parcelable{
    @PrimaryKey(autoGenerate = true)
    private int id;
    private String name;
    private String PhoneName;
    private String DeviceNumber;
    private float Temperature;
    private float totalPhosphorus;
    private float ammoniaNitrogen;
    private float potassiumPermanganate;
    private float referenceTotalPhosphorus;
    private float referenceAmmoniaNitrogen;
    private float referencePotassiumPermanganate;
    private float[] TotalPhosphorus_spec;
    private float[] AmmoniaNitrogen_spec;
    private float[] PotassiumPermanganate_spec;

    private float[] TP_intensity;
    private float[] AN_intensity;
    private float[] PP_intensity;

    private float[] NoSample_intensity;

    private Bitmap NoSampleBitmap;
    private Bitmap SampleBitmap;

    private long time;  // 添加 time 属性

    public Item(String name, String PhoneName, String DeviceNumber,float Temperature, float totalPhosphorus, float ammoniaNitrogen, float potassiumPermanganate) {
        this.name = name;
        this.PhoneName = PhoneName;
        this.DeviceNumber = DeviceNumber;
        this.Temperature = Temperature;
        this.totalPhosphorus = totalPhosphorus;
        this.ammoniaNitrogen = ammoniaNitrogen;
        this.potassiumPermanganate = potassiumPermanganate;
        this.referenceTotalPhosphorus = 0;
        this.referenceAmmoniaNitrogen = 0;
        this.referencePotassiumPermanganate = 0;
        this.TotalPhosphorus_spec = new float[1080];
        this.AmmoniaNitrogen_spec = new float[1080];
        this.PotassiumPermanganate_spec = new float[1080];
        this.TP_intensity = new float[1080];
        this.AN_intensity = new float[1080];
        this.PP_intensity = new float[1080];
        this.NoSample_intensity = new float[1080];
        this.NoSampleBitmap = null;
        this.SampleBitmap = null;
        this.time = System.currentTimeMillis();  // 初始化为当前时间

    }


    // 将 Bitmap 转换为 Base64 字符串
    public String getBitmapBase64(Bitmap bitmap) {
        if (bitmap == null) return null;
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream);
        byte[] byteArray = outputStream.toByteArray();
        return Base64.encodeToString(byteArray, Base64.DEFAULT);
    }


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


    public String getPhoneName(){ return PhoneName; }


    public void setPhoneName(String phoneName){this.PhoneName = phoneName; }

    public String getDeviceNumber() {
        return DeviceNumber;
    }

    public void setDeviceNumber(String deviceNumber) {
        this.DeviceNumber = deviceNumber;
    }

    public float getTemperature(){
        return formatFloat(Temperature);
    }

    public void setTemperature(float temperature){
        this.Temperature = temperature;
    }

    public float getTotalPhosphorus() {
        return formatFloat(totalPhosphorus);
    }

    public void setTotalPhosphorus(float totalPhosphorus) {
        this.totalPhosphorus = totalPhosphorus;
    }

    public float getAmmoniaNitrogen() {
        return formatFloat(ammoniaNitrogen);
    }

    public void setAmmoniaNitrogen(float ammoniaNitrogen) {
        this.ammoniaNitrogen = ammoniaNitrogen;
    }

    public float getPotassiumPermanganate() {
        return formatFloat(potassiumPermanganate);
    }

    public void setPotassiumPermanganate(float potassiumPermanganate) {
        this.potassiumPermanganate = potassiumPermanganate;
    }

    public float getReferenceTotalPhosphorus() {
        return referenceTotalPhosphorus;
    }

    public void setReferenceTotalPhosphorus(float referenceTotalPhosphorus) {
        this.referenceTotalPhosphorus = referenceTotalPhosphorus;
    }

    public float getReferenceAmmoniaNitrogen() {
        return referenceAmmoniaNitrogen;
    }

    public void setReferenceAmmoniaNitrogen(float referenceAmmoniaNitrogen) {
        this.referenceAmmoniaNitrogen = referenceAmmoniaNitrogen;
    }

    public float getReferencePotassiumPermanganate() {
        return referencePotassiumPermanganate;
    }

    public void setReferencePotassiumPermanganate(float referencePotassiumPermanganate) {
        this.referencePotassiumPermanganate = referencePotassiumPermanganate;
    }

    private float formatFloat(float value) {
        return Float.parseFloat(String.format("%.3f", value));
    }

    public float[] getTotalPhosphorus_spec() {
        return TotalPhosphorus_spec;
    }

    public void setTotalPhosphorus_spec(float[] totalPhosphorus_spec) {
        TotalPhosphorus_spec = totalPhosphorus_spec;
    }

    public float[] getAmmoniaNitrogen_spec() {
        return AmmoniaNitrogen_spec;
    }

    public void setAmmoniaNitrogen_spec(float[] ammoniaNitrogen_spec) {
        AmmoniaNitrogen_spec = ammoniaNitrogen_spec;
    }

    public float[] getPotassiumPermanganate_spec() {
        return PotassiumPermanganate_spec;
    }

    public void setPotassiumPermanganate_spec(float[] potassiumPermanganate_spec) {
        PotassiumPermanganate_spec = potassiumPermanganate_spec;
    }

    public float[] getTP_intensity() {
        return TP_intensity;
    }

    public void setTP_intensity(float[] intensity) {
        TP_intensity = intensity;
    }

    public float[] getAN_intensity() {
        return AN_intensity;
    }

    public void setAN_intensity(float[] intensity) {
        AN_intensity = intensity;
    }

    public float[] getPP_intensity() {
        return PP_intensity;
    }

    public void setPP_intensity(float[] intensity) {
        PP_intensity = intensity;
    }

    public float[] getNoSample_intensity(){return NoSample_intensity;}

    public void setNoSample_intensity(float[] intensity){NoSample_intensity = intensity;}


    public Bitmap getNoSampleBitmap() {return NoSampleBitmap;}


    public void setNoSampleBitmap(Bitmap bitmap){NoSampleBitmap = bitmap;}


    public Bitmap getSampleBitmap() {return SampleBitmap;}


    public void setSampleBitmap(Bitmap bitmap){SampleBitmap = bitmap;}

    public long getTime() {
        return time;
    }

    public void setTime(long time) {
        this.time = time;
    }



    // 实现 Parcelable 的方法
    protected Item(Parcel in) {
        id = in.readInt();
        name = in.readString();
        PhoneName = in.readString();
        DeviceNumber = in.readString();
        Temperature = in.readFloat();
        totalPhosphorus = in.readFloat();
        ammoniaNitrogen = in.readFloat();
        potassiumPermanganate = in.readFloat();
        referenceTotalPhosphorus = in.readFloat();
        referenceAmmoniaNitrogen = in.readFloat();
        referencePotassiumPermanganate = in.readFloat();
        TotalPhosphorus_spec = in.createFloatArray();
        AmmoniaNitrogen_spec = in.createFloatArray();
        PotassiumPermanganate_spec = in.createFloatArray();
        TP_intensity = in.createFloatArray();
        AN_intensity = in.createFloatArray();
        PP_intensity = in.createFloatArray();
        NoSample_intensity = in.createFloatArray();
        time = in.readLong();
        NoSampleBitmap = readBitmapFromParcel(in);
        SampleBitmap = readBitmapFromParcel(in);
    }

    private Bitmap readBitmapFromParcel(Parcel in) {
        int length = in.readInt();
        if (length > 0) {
            byte[] byteArray = new byte[length];
            in.readByteArray(byteArray);
            return BitmapFactory.decodeByteArray(byteArray, 0, length);
        }
        return null;
    }


    public static final Parcelable.Creator<Item> CREATOR = new Parcelable.Creator<Item>() {
        @Override
        public Item createFromParcel(Parcel in) {
            return new Item(in);
        }

        @Override
        public Item[] newArray(int size) {
            return new Item[size];
        }
    };

    @Override
    public void writeToParcel(Parcel dest, int flags) {
        dest.writeInt(id);
        dest.writeString(name);
        dest.writeString(PhoneName);
        dest.writeString(DeviceNumber);
        dest.writeFloat(Temperature);
        dest.writeFloat(totalPhosphorus);
        dest.writeFloat(ammoniaNitrogen);
        dest.writeFloat(potassiumPermanganate);
        dest.writeFloat(referenceTotalPhosphorus);
        dest.writeFloat(referenceAmmoniaNitrogen);
        dest.writeFloat(referencePotassiumPermanganate);
        dest.writeFloatArray(TotalPhosphorus_spec);
        dest.writeFloatArray(AmmoniaNitrogen_spec);
        dest.writeFloatArray(PotassiumPermanganate_spec);
        dest.writeFloatArray(TP_intensity);
        dest.writeFloatArray(AN_intensity);
        dest.writeFloatArray(PP_intensity);
        dest.writeFloatArray(NoSample_intensity);
        dest.writeLong(time);
        // 序列化 Bitmap 为 byte[]
        writeBitmapToParcel(dest, NoSampleBitmap);
        writeBitmapToParcel(dest, SampleBitmap);
    }

    private void writeBitmapToParcel(Parcel dest, Bitmap bitmap) {
        if (bitmap != null) {
            ByteArrayOutputStream stream = new ByteArrayOutputStream();
            bitmap.compress(Bitmap.CompressFormat.JPEG, 90, stream);
            byte[] byteArray = stream.toByteArray();
            dest.writeInt(byteArray.length);
            dest.writeByteArray(byteArray);
        } else {
            dest.writeInt(0);
        }
    }


    @Override
    public int describeContents() {
        return 0;
    }







}


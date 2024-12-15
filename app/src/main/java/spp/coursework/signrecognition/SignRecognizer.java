package spp.coursework.signrecognition;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class SignRecognizer {
    private Interpreter interpreter;
    private int INPUT_SIZE;
    private int height = 0;
    private int width = 0;
    private GpuDelegate gpuDelegate = null;
    private CascadeClassifier classifier;

    public SignRecognizer(AssetManager manager, Context context, String modelPath, int inputSize) throws IOException {
        INPUT_SIZE = inputSize;
        Interpreter.Options options = new Interpreter.Options();
        gpuDelegate = new GpuDelegate();
        options.addDelegate(gpuDelegate);

        options.setNumThreads(8);
        interpreter = new Interpreter(loadModel(manager, modelPath), options);
        Log.d("Sign Recognition", "Model is loaded");

        try {
            InputStream inputStream = context.getResources().openRawResource(R.raw.cascade);
            File cascadeDir = context.getDir("sign_cascade", Context.MODE_PRIVATE);
            File mCascadeFile = new File(cascadeDir, "cascade");

            FileOutputStream outputStream = new FileOutputStream(mCascadeFile);
            byte[] buffer = new byte[4096];
            int byteRead;
            while ((byteRead = inputStream.read(buffer)) > -1){
                outputStream.write(buffer, 0, byteRead);
            }

            inputStream.close();
            outputStream.close();

            classifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
            Log.d("Sign Recognizer", "Haar classifier is loaded");
        } catch (Exception exception){
            exception.printStackTrace();
        }
    }

    public Mat recognizeSign(Mat mat_image) {
        Core.flip(mat_image.t(), mat_image, 1);

        Mat greyscaleImage = new Mat();
        Imgproc.cvtColor(mat_image, greyscaleImage, Imgproc.COLOR_RGBA2GRAY);

        height = greyscaleImage.height();
        width = greyscaleImage.width();

        int absoluteSignSize = (int) (height * 0.1);
        MatOfRect signs = new MatOfRect();

        if (classifier != null) {
            classifier.detectMultiScale(greyscaleImage, signs, 1.1, 2, 2,
                    new Size(absoluteSignSize, absoluteSignSize), new Size());
        }

        Rect[] signArray = signs.toArray();
        for (int i = 0; i < signArray.length; ++i) {
            Imgproc.rectangle(mat_image, signArray[i].tl(), signArray[i].br(), new Scalar(0, 255, 0, 255), 2);

            Rect roi = new Rect((int) signArray[i].tl().x, (int) signArray[i].tl().y,
                    ((int) signArray[i].br().x) - ((int) signArray[i].tl().x),
                    ((int) signArray[i].br().y) - ((int) signArray[i].tl().y));

            Mat croppedRgb = new Mat(mat_image, roi);
            Bitmap bitmap = Bitmap.createBitmap(croppedRgb.cols(), croppedRgb.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(croppedRgb, bitmap);
            Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);
            ByteBuffer byteBuffer = convertBitmapToByteBuffer(scaledBitmap);
            float[][] signValue = new float[1][43];

            interpreter.run(byteBuffer, signValue);
            int readSignId = findWinner(signValue);
            String signName = TrafficSign.getNameById(readSignId);

            Bitmap resultBitmap = Bitmap.createBitmap(mat_image.cols(), mat_image.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(mat_image, resultBitmap);

            Canvas canvas = new Canvas(resultBitmap);
            Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setTextSize(30);
            paint.setAlpha(255);

            Point textPosition = new Point((int) signArray[i].tl().x + 10, (int) signArray[i].tl().y + 20);

            canvas.drawText(signName, (float) textPosition.x, (float) textPosition.y, paint);

            Utils.bitmapToMat(resultBitmap, mat_image);
        }

        Core.flip(mat_image.t(), mat_image, 0);
        return mat_image;
    }

    private int findWinner(float[][] signValue){
        float maxValue = 0;
        int maxId = 0;
        for(int i = 0; i < signValue[0].length; ++i){
            if(maxValue < signValue[0][i]){
                maxValue = signValue[0][i];
                maxId = i;
            }
        }
        return maxId;
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap scaledBitmap) {
        ByteBuffer byteBuffer;
        int inputSize = INPUT_SIZE;
        byteBuffer = ByteBuffer.allocateDirect(4*1*INPUT_SIZE*INPUT_SIZE*3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intVals = new int[inputSize*inputSize];
        scaledBitmap.getPixels(intVals, 0, scaledBitmap.getWidth(), 0, 0, scaledBitmap.getWidth(), scaledBitmap.getHeight());
        int pixels = 0;

        for(int i = 0; i < inputSize; ++i){
            for(int j = 0; j < inputSize; ++j){
                final int value = intVals[pixels++];
                byteBuffer.putFloat(((value >> 16)&0xFF)/255.0f);
                byteBuffer.putFloat(((value >> 8)&0xFF)/255.0f);
                byteBuffer.putFloat((value&0xFF)/255.0f);
            }
        }
        return byteBuffer;
    }

    private MappedByteBuffer loadModel(AssetManager manager, String modelPath) throws IOException {
        AssetFileDescriptor assetFileDescriptor = manager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = assetFileDescriptor.getStartOffset();
        long declaredLength = assetFileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public enum TrafficSign {
        SPEED_LIMIT_20(0, "Ограничение скорости (20 км/ч)"),
        SPEED_LIMIT_30(1, "Ограничение скорости (30 км/ч)"),
        SPEED_LIMIT_50(2, "Ограничение скорости (50 км/ч)"),
        SPEED_LIMIT_60(3, "Ограничение скорости (60 км/ч)"),
        SPEED_LIMIT_70(4, "Ограничение скорости (70 км/ч)"),
        SPEED_LIMIT_80(5, "Ограничение скорости (80 км/ч)"),
        END_SPEED_LIMIT_80(6, "Конец ограничения скорости (80 км/ч)"),
        SPEED_LIMIT_100(7, "Ограничение скорости (100 км/ч)"),
        SPEED_LIMIT_120(8, "Ограничение скорости (120 км/ч)"),
        NO_PASSING(9, "Обгон запрещён"),
        NO_PASSING_VEH_OVER_3_5_TONS(10, "Обгон запрещён для транспортных средств свыше 3.5 тонн"),
        RIGHT_OF_WAY_AT_INTERSECTION(11, "Главная дорога на перекрёстке"),
        PRIORITY_ROAD(12, "Главная дорога"),
        YIELD(13, "Уступите дорогу"),
        STOP(14, "Стоп"),
        NO_VEHICLES(15, "Транспортные средства запрещены"),
        VEH_OVER_3_5_TONS_PROHIBITED(16, "Запрещено движение транспортных средств свыше 3.5 тонн"),
        NO_ENTRY(17, "Въезд запрещён"),
        GENERAL_CAUTION(18, "Общая осторожность"),
        DANGEROUS_CURVE_LEFT(19, "Опасный поворот налево"),
        DANGEROUS_CURVE_RIGHT(20, "Опасный поворот направо"),
        DOUBLE_CURVE(21, "Двойной поворот"),
        BUMPY_ROAD(22, "Кочка на дороге"),
        SLIPPERY_ROAD(23, "Скользкая дорога"),
        ROAD_NARROWS_ON_THE_RIGHT(24, "Дорога сужается справа"),
        ROAD_WORK(25, "Дорожные работы"),
        TRAFFIC_SIGNALS(26, "Сигналы светофора"),
        PEDESTRIANS(27, "Пешеходы"),
        CHILDREN_CROSSING(28, "Дети на переходе"),
        BICYCLES_CROSSING(29, "Переход для велосипедистов"),
        BEWARE_OF_ICE_SNOW(30, "Осторожно, лед/снег"),
        WILD_ANIMALS_CROSSING(31, "Дикие животные на дороге"),
        END_SPEED_AND_PASSING_LIMITS(32, "Конец ограничений скорости и обгона"),
        TURN_RIGHT_AHEAD(33, "Поворот направо впереди"),
        TURN_LEFT_AHEAD(34, "Поворот налево впереди"),
        AHEAD_ONLY(35, "Только прямо"),
        GO_STRAIGHT_OR_RIGHT(36, "Ехать прямо или направо"),
        GO_STRAIGHT_OR_LEFT(37, "Ехать прямо или налево"),
        KEEP_RIGHT(38, "Держитесь правой стороны"),
        KEEP_LEFT(39, "Держитесь левой стороны"),
        ROUNDABOUT_MANDATORY(40, "Круговое движение обязательно"),
        END_OF_NO_PASSING(41, "Конец зоны обгона"),
        END_NO_PASSING_VEH_OVER_3_5_TONS(42, "Конец зоны обгона для транспортных средств свыше 3.5 тонн");

        private final int id;
        private final String name;

        TrafficSign(int id, String name) {
            this.id = id;
            this.name = name;
        }

        public int getId() {
            return id;
        }

        public String getName() {
            return name;
        }

        public static String getNameById(int id) {
            for (TrafficSign sign : TrafficSign.values()) {
                if (sign.getId() == id) {
                    return sign.getName();
                }
            }
            return "Неизвестен";
        }
    }
}

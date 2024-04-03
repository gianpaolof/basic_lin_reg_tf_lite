package com.example.tensorflowreg

import android.app.Activity
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.tensorflowreg.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {


    private lateinit var binding: ActivityMainBinding
    private lateinit var  tflite: Interpreter
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)


        tflite= Interpreter(loadModelFile(this)!!)



        //val prediction = outputArray[0][0]

        binding.button2.setOnClickListener {

            val userInput = binding.edittext.text.toString().toInt()

            // Run inference using a coroutine
            lifecycleScope.launch {
                val prediction = runInference(userInput)
                updateButton(prediction)
            }

        }
    }

    private fun runInference(input: Int): Float {

        val inputArray = Array(1) { FloatArray(1) }
        inputArray[0][0] = input.toFloat()

        val outputArray = Array(1) { FloatArray(1) }
        tflite.run(inputArray, outputArray)

        return outputArray[0][0]
    }

    private suspend fun updateButton(prediction: Float) {
        withContext(Dispatchers.Main) {
            binding.textView.text = "Prediction: $prediction"
        }
    }



    @Throws(IOException::class)
    private fun loadModelFile(activity: Activity): MappedByteBuffer? {
        val fileDescriptor = activity.assets.openFd("model.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

}
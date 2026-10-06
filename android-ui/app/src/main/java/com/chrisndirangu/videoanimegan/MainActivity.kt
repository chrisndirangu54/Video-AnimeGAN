package com.chrisndirangu.videoanimegan

import android.content.Context
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.media3.common.MediaItem
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.ui.PlayerView
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File
import java.io.FileOutputStream
import java.time.Duration

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { MaterialTheme { AnimeEditorScreen() } }
    }
}

@Composable
fun AnimeEditorScreen() {
    val context = androidx.compose.ui.platform.LocalContext.current
    val scope = rememberCoroutineScope()
    var selectedVideo by remember { mutableStateOf<Uri?>(null) }
    var outputFile by remember { mutableStateOf<File?>(null) }
    var style by remember { mutableStateOf("paprika") }
    var status by remember { mutableStateOf("Select a video to begin") }
    var processing by remember { mutableStateOf(false) }
    var expanded by remember { mutableStateOf(false) }

    val styles = listOf("paprika", "celeba_distill", "face_paint_512_v1", "face_paint_512_v2")
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        selectedVideo = uri
        outputFile = null
        status = if (uri != null) "Video selected" else "No video selected"
    }

    Scaffold { padding ->
        Column(
            modifier = Modifier.padding(padding).padding(20.dp).fillMaxSize(),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Text("Video → Anime", style = MaterialTheme.typography.headlineMedium)
            Text("Choose a video, select an AnimeGANv2 style, then process it on the Python GPU service.")

            Button(onClick = { picker.launch("video/*") }, modifier = Modifier.fillMaxWidth()) {
                Text(if (selectedVideo == null) "Choose video" else "Choose another video")
            }

            Box {
                OutlinedButton(onClick = { expanded = true }, modifier = Modifier.fillMaxWidth()) {
                    Text("Style: $style")
                }
                DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                    styles.forEach { item ->
                        DropdownMenuItem(text = { Text(item) }, onClick = {
                            style = item
                            expanded = false
                        })
                    }
                }
            }

            Button(
                enabled = selectedVideo != null && !processing,
                onClick = {
                    selectedVideo?.let { uri ->
                        scope.launch {
                            processing = true
                            status = "Uploading and stylizing…"
                            try {
                                outputFile = stylizeVideo(context, uri, style)
                                status = "Anime video ready"
                            } catch (e: Exception) {
                                status = "Failed: " + (e.message ?: "unknown error")
                            } finally {
                                processing = false
                            }
                        }
                    }
                },
                modifier = Modifier.fillMaxWidth()
            ) {
                if (processing) {
                    CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 2.dp)
                    Spacer(Modifier.width(10.dp))
                }
                Text(if (processing) "Processing…" else "Create anime video")
            }

            Card(modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(16.dp)) {
                Text(status, modifier = Modifier.padding(16.dp))
            }

            outputFile?.let { file ->
                Text("Preview", style = MaterialTheme.typography.titleMedium)
                VideoPlayer(file)
            }
        }
    }
}

@Composable
fun VideoPlayer(file: File) {
    val context = androidx.compose.ui.platform.LocalContext.current
    val player = remember(file.absolutePath) {
        ExoPlayer.Builder(context).build().apply {
            setMediaItem(MediaItem.fromUri(Uri.fromFile(file)))
            prepare()
        }
    }
    DisposableEffect(player) { onDispose { player.release() } }
    AndroidView(
        modifier = Modifier.fillMaxWidth().height(240.dp),
        factory = { PlayerView(it).apply { this.player = player } }
    )
}

suspend fun stylizeVideo(context: Context, uri: Uri, style: String): File = withContext(Dispatchers.IO) {
    val input = File(context.cacheDir, "input_" + System.currentTimeMillis() + ".mp4")
    context.contentResolver.openInputStream(uri).use { source ->
        requireNotNull(source) { "Could not read selected video" }
        FileOutputStream(input).use { target -> source.copyTo(target) }
    }

    val body = MultipartBody.Builder()
        .setType(MultipartBody.FORM)
        .addFormDataPart("style", style)
        .addFormDataPart("temporal_strength", "0.18")
        .addFormDataPart("max_side", "1280")
        .addFormDataPart("video", input.name, input.asRequestBody("video/mp4".toMediaType()))
        .build()

    val request = Request.Builder()
        .url(BuildConfig.API_BASE_URL + "stylize")
        .post(body)
        .build()

    OkHttpClient.Builder()
        .callTimeout(Duration.ofMinutes(30))
        .build()
        .newCall(request)
        .execute()
        .use { response ->
            if (!response.isSuccessful) error("Server returned " + response.code)
            val output = File(context.cacheDir, "anime_" + System.currentTimeMillis() + ".mp4")
            response.body?.byteStream().use { source ->
                requireNotNull(source) { "Empty response from server" }
                FileOutputStream(output).use { target -> source.copyTo(target) }
            }
            output
        }
}

package com.singleprompter.app

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.content.res.Configuration
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.ArrowForward
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import kotlinx.coroutines.launch
import java.util.Locale

private val Dark = Color(0xFF0B1220)
private val Card = Color(0xFF172033)
private val Accent = Color(0xFF7DD3FC)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent { SinglePrompterApp() }
    }
}

@Composable
private fun SinglePrompterApp() {
    var script by rememberSaveable { mutableStateOf("") }
    var presenting by rememberSaveable { mutableStateOf(false) }
    var listening by remember { mutableStateOf(false) }
    var activeLine by rememberSaveable { mutableIntStateOf(0) }
    var heardText by remember { mutableStateOf("") }
    val lines = remember(script) {
        // 문단이나 문장이 너무 길 경우를 대비해 마침표, 물음표, 느낌표 또는 줄바꿈 기준으로 세밀하게 나눔
        script.split(Regex("(?<=[.!?])\\s+|\\n+"))
            .map(String::trim)
            .filter { it.isNotBlank() }
    }

    val view = LocalView.current
    DisposableEffect(presenting) {
        if (presenting) {
            view.keepScreenOn = true
        }
        onDispose {
            view.keepScreenOn = false
        }
    }

    // 문장이 바뀔 때 인식된 텍스트를 초기화하여 새 문장에서 하이라이트가 꼬이는 것을 방지
    LaunchedEffect(activeLine) {
        heardText = ""
    }

    MaterialTheme(colorScheme = darkColorScheme(primary = Accent, surface = Card, background = Dark)) {
        Surface(Modifier.fillMaxSize(), color = Dark) {
            if (!presenting) {
                EditorScreen(script, { script = it }, lines.size) { if (lines.isNotEmpty()) { activeLine = 0; presenting = true } }
            } else {
                PrompterScreen(
                    lines = lines, activeLine = activeLine, listening = listening, heardText = heardText,
                    onBack = { presenting = false; listening = false },
                    onPrevious = { activeLine = (activeLine - 1).coerceAtLeast(0) },
                    onNext = { activeLine = (activeLine + 1).coerceAtMost(lines.lastIndex) },
                    onVoiceUpdate = { result -> activeLine = ScriptMatcher.findNextLine(lines, activeLine, result) },
                    onListeningChanged = { listening = it }, onHeard = { heardText = it }
                )
            }
        }
    }
}

@Composable
private fun EditorScreen(script: String, onScriptChange: (String) -> Unit, lineCount: Int, onStart: () -> Unit) {
    Column(Modifier.fillMaxSize().padding(24.dp).systemBarsPadding()) {
        Text("싱글 프롬프터", fontSize = 28.sp, fontWeight = FontWeight.Bold, color = Color.White)
        Text("말을 따라가는 한국어 프롬프터", color = Color(0xFFB7C4D8), modifier = Modifier.padding(top = 4.dp))
        Spacer(Modifier.height(28.dp))
        OutlinedTextField(value = script, onValueChange = onScriptChange, modifier = Modifier.fillMaxWidth().weight(1f),
            label = { Text("대본") }, placeholder = { Text("대본을 붙여넣거나 작성하세요.\n문단 또는 줄 단위로 자동 전환됩니다.") },
            colors = OutlinedTextFieldDefaults.colors(focusedBorderColor = Accent, focusedTextColor = Color.White, unfocusedTextColor = Color.White))
        Text("${lineCount}개 문장", color = Color(0xFFB7C4D8), modifier = Modifier.padding(top = 8.dp))
        Button(onClick = onStart, enabled = lineCount > 0, modifier = Modifier.fillMaxWidth().padding(top = 16.dp).height(54.dp)) { Text("프롬프터 시작", fontSize = 17.sp) }
    }
}

@Composable
private fun PrompterScreen(lines: List<String>, activeLine: Int, listening: Boolean, heardText: String, onBack: () -> Unit, onPrevious: () -> Unit, onNext: () -> Unit, onVoiceUpdate: (String) -> Unit, onListeningChanged: (Boolean) -> Unit, onHeard: (String) -> Unit) {
    val context = LocalContext.current
    val listState = rememberLazyListState()
    val scope = rememberCoroutineScope()
    val recognizer = remember { KoreanSpeechRecognizer(context, onVoiceUpdate, onVoiceUpdate, onHeard, onListeningChanged) }
    val permission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted -> if (granted) recognizer.start() }
    
    val configuration = LocalConfiguration.current
    val isLandscape = configuration.orientation == Configuration.ORIENTATION_LANDSCAPE

    DisposableEffect(Unit) { onDispose { recognizer.destroy() } }
    
    val onMicClick = {
        if (listening) recognizer.stop() 
        else if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) recognizer.start() 
        else permission.launch(Manifest.permission.RECORD_AUDIO)
    }

    if (isLandscape) {
        Row(Modifier.fillMaxSize().systemBarsPadding()) {
            PrompterScriptArea(Modifier.weight(1f), lines, activeLine, heardText, onBack, listState, showLineCount = false)
            PrompterControls(true, activeLine, lines.size, listening, onPrevious, onNext, onMicClick)
        }
    } else {
        Column(Modifier.fillMaxSize().systemBarsPadding()) {
            PrompterScriptArea(Modifier.weight(1f), lines, activeLine, heardText, onBack, listState, showLineCount = true)
            PrompterControls(false, activeLine, lines.size, listening, onPrevious, onNext, onMicClick)
        }
    }
}

@Composable
private fun PrompterScriptArea(modifier: Modifier, lines: List<String>, activeLine: Int, heardText: String, onBack: () -> Unit, listState: androidx.compose.foundation.lazy.LazyListState, showLineCount: Boolean) {
    Column(modifier) {
        Row(Modifier.fillMaxWidth().padding(horizontal = 4.dp), verticalAlignment = Alignment.CenterVertically) {
            IconButton(onClick = onBack, modifier = Modifier.padding(8.dp)) {
                Icon(Icons.AutoMirrored.Filled.ArrowBack, "대본 편집으로", tint = Color.White)
            }
            if (showLineCount) {
                Text(
                    "${activeLine + 1} / ${lines.size}",
                    color = Color(0xFFB7C4D8),
                    modifier = Modifier.weight(1f).padding(end = 48.dp),
                    textAlign = TextAlign.Center,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }

        val offsetRatio = if (showLineCount) 0.1f else 0.02f // 가로 모드일 때 더 위로(2%)

        BoxWithConstraints(Modifier.weight(1f).fillMaxWidth()) {
            val listHeight = maxHeight
            val density = LocalDensity.current
            
            LaunchedEffect(activeLine, listHeight) {
                if (activeLine >= 0) {
                    val offset = with(density) { (listHeight * offsetRatio).roundToPx() }
                    listState.animateScrollToItem(activeLine, -offset)
                }
            }

            LazyColumn(
                state = listState, 
                modifier = Modifier.fillMaxSize(), 
                contentPadding = PaddingValues(top = listHeight * offsetRatio, bottom = listHeight * (1f - offsetRatio)), 
                verticalArrangement = Arrangement.spacedBy(24.dp)
            ) {
                items(lines.size) { index ->
                    val current = index == activeLine
                    KaraokeText(
                        line = lines[index],
                        heardText = if (current) heardText else "",
                        isCurrent = current,
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(if (current) Card.copy(alpha = 0.5f) else Color.Transparent)
                            .padding(if (current) 16.dp else 0.dp)
                    )
                }
            }
        }
    }
}

@Composable
private fun PrompterControls(isLandscape: Boolean, activeLine: Int, lineCount: Int, listening: Boolean, onPrevious: () -> Unit, onNext: () -> Unit, onMicClick: () -> Unit) {
    if (isLandscape) {
        Column(
            Modifier.width(110.dp).fillMaxHeight().padding(horizontal = 8.dp, vertical = 16.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // 문단 번호를 마이크 버튼 위로 이동
            Text("${activeLine + 1} / $lineCount", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 13.sp)
            Spacer(Modifier.height(8.dp))
            FilledIconButton(onClick = onMicClick, modifier = Modifier.size(52.dp), colors = IconButtonDefaults.filledIconButtonColors(containerColor = if (listening) Color(0xFFDC2626) else Accent)) {
                Icon(if (listening) Icons.Default.Stop else Icons.Default.Mic, null, tint = Dark, modifier = Modifier.size(24.dp))
            }
            
            Spacer(Modifier.height(32.dp))
            OutlinedButton(
                onClick = onPrevious, 
                modifier = Modifier.fillMaxWidth().height(40.dp), 
                contentPadding = PaddingValues(horizontal = 4.dp)
            ) {
                Icon(Icons.AutoMirrored.Filled.ArrowBack, null, modifier = Modifier.size(16.dp))
                Text(" 이전", fontSize = 13.sp)
            }
            Spacer(Modifier.height(8.dp))
            OutlinedButton(
                onClick = onNext, 
                modifier = Modifier.fillMaxWidth().height(40.dp), 
                contentPadding = PaddingValues(horizontal = 4.dp)
            ) {
                Text("다음 ", fontSize = 13.sp)
                Icon(Icons.AutoMirrored.Filled.ArrowForward, null, modifier = Modifier.size(16.dp))
            }
        }
    } else {
        Row(
            Modifier.fillMaxWidth().padding(16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            OutlinedButton(onClick = onPrevious) {
                Icon(Icons.AutoMirrored.Filled.ArrowBack, null); Text(" 이전")
            }
            FilledIconButton(onClick = onMicClick, modifier = Modifier.size(64.dp), colors = IconButtonDefaults.filledIconButtonColors(containerColor = if (listening) Color(0xFFDC2626) else Accent)) {
                Icon(if (listening) Icons.Default.Stop else Icons.Default.Mic, null, tint = Dark)
            }
            OutlinedButton(onClick = onNext) {
                Text("다음 "); Icon(Icons.AutoMirrored.Filled.ArrowForward, null)
            }
        }
    }
}

@Composable
private fun KaraokeText(line: String, heardText: String, isCurrent: Boolean, modifier: Modifier = Modifier) {
    val annotatedString = buildAnnotatedString {
        val words = line.split(Regex("(?<=\\s)|(?=\\s)"))
        val heardTokens = ScriptMatcher.tokenize(heardText)
        
        words.forEach { word ->
            val cleanWord = word.trim().lowercase().replace(Regex("[^가-힣a-z0-9]"), "")
            val isMatched = isCurrent && cleanWord.isNotEmpty() && heardTokens.any { token -> 
                token.contains(cleanWord) || cleanWord.contains(token)
            }
            
            withStyle(style = SpanStyle(
                color = if (isMatched) Accent else if (isCurrent) Color.White else Color(0xFF404B5E)
            )) {
                append(word)
            }
        }
    }
    
    Text(
        text = annotatedString,
        fontSize = if (isCurrent) 34.sp else 26.sp,
        lineHeight = if (isCurrent) 50.sp else 38.sp,
        fontWeight = if (isCurrent) FontWeight.Bold else FontWeight.Normal,
        textAlign = TextAlign.Center,
        modifier = modifier
    )
}

private class KoreanSpeechRecognizer(private val context: Context, private val onPartialResult: (String) -> Unit, private val onFinalResult: (String) -> Unit, private val onHeard: (String) -> Unit, private val onState: (Boolean) -> Unit) {
    private val recognizer = SpeechRecognizer.createSpeechRecognizer(context).apply {
        setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(p: Bundle?) { onState(true) }
            override fun onPartialResults(b: Bundle?) { 
                b?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()?.let { 
                    onHeard(it)
                    onPartialResult(it) 
                } 
            }
            override fun onResults(b: Bundle?) { 
                b?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()?.let { 
                    onHeard(it)
                    onFinalResult(it) 
                }
                restart() 
            }
            override fun onError(error: Int) { 
                // ERROR_NO_MATCH(7), ERROR_SPEECH_TIMEOUT(6) 등은 무시하고 다시 시작
                if (error != SpeechRecognizer.ERROR_CLIENT) restart() 
            }
            override fun onBeginningOfSpeech() {} ; override fun onBufferReceived(b: ByteArray?) {} ; override fun onEndOfSpeech() {} ; override fun onEvent(t: Int, p: Bundle?) {} ; override fun onRmsChanged(r: Float) {}
        })
    }
    private var wanted = false
    private fun intent() = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply { 
        putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ko-KR")
        putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
        putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
        putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE, context.packageName)
    }
    fun start() { wanted = true; recognizer.startListening(intent()) }
    fun stop() { wanted = false; recognizer.stopListening(); onState(false) }
    private fun restart() { if (wanted) recognizer.startListening(intent()) else onState(false) }
    fun destroy() = recognizer.destroy()
}

private object ScriptMatcher {
    fun findNextLine(lines: List<String>, current: Int, heard: String): Int {
        val spokenTokens = tokenize(heard)
        if (spokenTokens.isEmpty() || current >= lines.lastIndex) return current

        val currentSim = calculateSimilarity(lines[current], spokenTokens, heard)
        val nextSim = calculateSimilarity(lines[current + 1], spokenTokens, heard)

        // 임계값을 상향하여 더 신중하게 넘어가도록 조정
        val lineLength = lines[current].length
        val threshold = if (lineLength > 40) 0.8 else 0.9
        
        return when {
            // 1. 현재 문장을 거의 다 읽었을 때
            currentSim >= threshold -> current + 1
            // 2. 현재 문장을 절반 이상 읽고 다음 문장의 시작이 확실히 들릴 때
            currentSim >= 0.6 && nextSim >= 0.3 -> current + 1
            // 3. 다음 문장이 아주 확실히 들릴 때 (오인식 방지 강화)
            nextSim >= 0.65 -> current + 1
            else -> current
        }
    }

    fun calculateSimilarity(line: String, heardTokens: List<String>, heardRaw: String): Double {
        val lineTokens = tokenize(line)
        if (lineTokens.isEmpty()) return 0.0
        
        var matches = 0
        for (lt in lineTokens) {
            if (heardTokens.any { it.contains(lt) || lt.contains(it) }) {
                matches++
            }
        }
        val tokenScore = matches.toDouble() / lineTokens.size
        
        val compactLine = line.replace(Regex("[^가-힣a-z0-9]"), "")
        val compactHeard = heardRaw.replace(Regex("[^가-힣a-z0-9]"), "")
        val phraseMatch = if (compactHeard.length >= 3 && compactLine.contains(compactHeard)) 0.7 else 0.0
        
        return maxOf(tokenScore, phraseMatch)
    }

    fun tokenize(text: String) = text.lowercase()
        .replace(Regex("[^가-힣a-z0-9\\s]"), " ")
        .split(Regex("\\s+"))
        .filter { it.isNotBlank() }
}

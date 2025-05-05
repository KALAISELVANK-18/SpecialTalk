import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'package:video_player/video_player.dart';
import 'package:http/http.dart' as http;

class SpeechToTextScreen extends StatefulWidget {
  @override
  _SpeechToTextScreenState createState() => _SpeechToTextScreenState();
}

class _SpeechToTextScreenState extends State<SpeechToTextScreen> {
  stt.SpeechToText _speech = stt.SpeechToText();
  String _recognizedText = '';
  bool isfinish=true;
  VideoPlayerController? _controller;

  @override
  void initState() {
    super.initState();
    _initializeSpeech();
  }

  void _initializeSpeech() async {
    bool available = await _speech.initialize(
      onStatus: (status){
        print('Speech status: $status');
        if(status=="done")
          _sendTextToFastAPI(_recognizedText);
      },
      onError: (errorNotification) => print('Speech error: $errorNotification'),
    );
    if (available) {
      print('Speech recognition is available');
    } else {
      print('Speech recognition is not available');
    }
  }

  void _startListening() {
    _speech.listen(
      onResult: (result) {
        setState(() {
          _recognizedText = result.recognizedWords;
          // Send recognized text to FastAPI
          // if()

        });
      },
      // listenFor: Duration(minutes: 5),
    );
  }

  void _stopListening() {
    _speech.stop();
  }

  Future<void> _sendTextToFastAPI(String text) async {
    print(text);
    final url = Uri.parse('http://192.168.246.234:8000/generate_video');
    final response = await http.post(
      url,
      headers: <String, String>{
        'Content-Type': 'application/json; charset=UTF-8',
      },
      body: jsonEncode(<String, String>{
        'text': text,
      }),
    );
    if (response.statusCode == 200) {

      final appDir = await getApplicationDocumentsDirectory();
      final videoFile = File('${appDir.path}/merged_video.mp4');
      await videoFile.writeAsBytes(response.bodyBytes);

      _controller = VideoPlayerController.file(videoFile)
        ..initialize().then((_) {
          setState(() {

          });
        });
    } else {
      print('Failed to recognize text: ${response.statusCode}');
    }
  }

  Future<void> _fetchAndDisplayVideo(String videoUrl) async {
    final response = await http.get(Uri.parse(videoUrl));
    print("paths");
    if (response.statusCode == 200) {
      print("I have crossed");
      final appDir = await getApplicationDocumentsDirectory();
      final videoFile = File('${appDir.path}/received_video.mp4');
      await videoFile.writeAsBytes(response.bodyBytes);
      print(videoFile.path+"paths");
      _controller = VideoPlayerController.file(videoFile)
        ..initialize().then((_) {
          setState(() {});
        });
    } else {
      print('Failed to fetch video: ${response.statusCode}');
    }
  }

  @override
  Widget build(BuildContext context) {

    return Scaffold(
      appBar: AppBar(
        title: Text('3D Video Converter'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              width: MediaQuery.of(context).size.width*0.9,
              decoration: BoxDecoration(

                  boxShadow: [
                    BoxShadow(
                      color: Colors.grey.withOpacity(0.5), // Shadow color
                      spreadRadius: 2, // Spread radius
                      blurRadius: 2, // Blur radius
                      offset: Offset(0, 3), // Offset in x and y directions
                    ),
                  ],

                borderRadius: BorderRadius.circular(10)
                    ,color: Colors.white
              ),
              child: SingleChildScrollView(
                child: Padding(
                  padding: const EdgeInsets.all(20.0),
                  child: Column(
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          ElevatedButton(

                            onPressed: _startListening,
                            child: Text('Start Listening'),

                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.blue, // Background color
                              foregroundColor: Colors.white, // Text color
                              elevation: 5, // Elevation
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(10), // BorderRadius
                              ),
                          ),),
                          SizedBox(width: 20),
                          ElevatedButton(
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.red, // Background color
                              foregroundColor: Colors.white, // Text color
                              elevation: 5, // Elevation
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(10), // BorderRadius
                              ),
                            ),
                            onPressed: _stopListening,
                            child: Text('Stop Listening'),
                          ),

                          ],
                      ),SizedBox(height: 20),
                      Text('Recognized Text:'),
                      SizedBox(height: 20),
                      Text(_recognizedText, style: TextStyle(fontSize: 18)),
                    ],
                  ),
                ),
              ),
            ),





            _controller != null && _controller!.value.isInitialized
                ? Column(
                  children: [
                    SizedBox(height: 100,),
                    Padding(
                      padding: const EdgeInsets.all(8.0),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.start,
                        children: [
                          Padding(
                            padding: const EdgeInsets.all(8.0),
                            child: Text("Coverted video",style: TextStyle(fontSize: 18)),
                          ),
                        ],
                      ),
                    ),
                    Padding(
                      padding: const EdgeInsets.all(8.0),
                      child: AspectRatio(
                                    aspectRatio: _controller!.value.aspectRatio,
                                    child: ClipRRect(
                                        borderRadius: BorderRadius.circular(20),
                                        child: VideoPlayer(_controller!)),
                                  ),
                    ),
                  ],
                )
                : CircularProgressIndicator(),

            SizedBox(height: 20),
            ElevatedButton(
              onPressed: () {
                if (_controller != null) {
                  if (_controller!.value.isPlaying) {

                      _controller!.pause();
                    setState(() {

                    });

                  } else {

                      _controller!.play();
                   setState(() {

                   });
                  }
                }
              },
              child: Icon(_controller != null && _controller!.value.isPlaying ? Icons.pause : Icons.play_arrow),
            ),
          ],
        ),
      ),
    );

  }

  @override
  void dispose() {
    _speech.stop();
    if (_controller != null) {
      _controller!.dispose();
    }
    super.dispose();
  }
}

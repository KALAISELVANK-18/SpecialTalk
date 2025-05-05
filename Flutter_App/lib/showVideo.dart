import 'package:flutter/material.dart';
import 'dart:convert';
import 'dart:io';
import 'package:video_player/video_player.dart';
import 'package:http/http.dart' as http;

class ShowVideoScreen extends StatefulWidget {
  final String videoPath;
  const ShowVideoScreen({super.key, required this.videoPath});

  @override
  State<ShowVideoScreen> createState() => _ShowVideoScreenState();
}

class _ShowVideoScreenState extends State<ShowVideoScreen> {
  late VideoPlayerController _videoController;
  int indicator = 0;
  String resultText = '';
  bool isPlaying = true;
  bool isVideoEnded = false;

  @override
  void initState() {
    super.initState();
    _videoController = VideoPlayerController.file(File(widget.videoPath))
      ..initialize().then((_) {
        if (mounted) {
          setState(() {}); // Update UI when video is initialized
          _videoController.play(); // Optionally start playing the video
          _videoController.addListener(() {
            if (_videoController.value.position == _videoController.value.duration && !isVideoEnded) {
              setState(() {
                isVideoEnded = true;
                isPlaying = false; // Stop playback when video ends
              });
            }
          });
        }
      });
  }

  @override
  void dispose() {
    _videoController.dispose();
    super.dispose();
  }

  Future<void> _confirmVideo() async {
    setState(() {
      indicator = 1;
    });

    try {
      var request = http.MultipartRequest(
          'POST', Uri.parse('http://192.168.246.234:7000/predict_signs/'))
        ..files.add(await http.MultipartFile.fromPath('file', widget.videoPath));

      var response = await request.send();

      if (response.statusCode == 200) {
        var responseData = await response.stream
            .transform(utf8.decoder)
            .join();
        var dataMap = jsonDecode(responseData);

        setState(() {
          indicator = 0;
          resultText = dataMap['predicted_signs'].toString();
        });
      } else {
        setState(() {
          indicator = 0;
          resultText = 'Failed to get result, try again!';
        });
      }
    } catch (e) {
      setState(() {
        indicator = 0;
        resultText = 'Error occurred: $e';
      });
    }
  }

  void _playVideo() {
    setState(() {
      isPlaying = true;
      isVideoEnded = false;
      _videoController.play();
    });
  }

  void _pauseVideo() {
    setState(() {
      isPlaying = false;
      _videoController.pause();
    });
  }

  void _replayVideo() {
    setState(() {
      _videoController.seekTo(Duration.zero);
      _playVideo();
    });
  }

  void _togglePlayPause() {
    if (isPlaying) {
      _pauseVideo();
    } else {
      _playVideo();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(),
      body: SingleChildScrollView(
        child: Center(
          child: _buildVideo(),
        ),
      ),
    );
  }

  Widget _buildVideo() {
    return Padding(
      padding: const EdgeInsets.only(left: 10, right: 10, bottom: 10),
      child: Container(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(10),
        ),
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.only(top: 0),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(10),
                child: GestureDetector(
                  onTap: _togglePlayPause,
                  child: Stack(
                    children: [
                      _videoController.value.isInitialized
                          ? AspectRatio(
                        aspectRatio: _videoController.value.aspectRatio,
                        child: VideoPlayer(_videoController),
                      )
                          : Center(child: CircularProgressIndicator()), // Show a loader while the video is initializing
                      if (!isPlaying && !isVideoEnded) // Show play button overlay when paused
                        Center(
                          child: Icon(
                            Icons.play_arrow,
                            size: 100,
                            color: Colors.white.withOpacity(0.8),
                          ),
                        ),
                      if (isVideoEnded) // Show replay button overlay when video ends
                        Center(
                          child: IconButton(
                            icon: Icon(
                              Icons.replay,
                              size: 100,
                              color: Colors.white.withOpacity(0.8),
                            ),
                            onPressed: _replayVideo,
                          ),
                        ),
                    ],
                  ),
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.only(top: 20, left: 8, right: 8),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  ElevatedButton(
                    onPressed: () async {
                      Navigator.pop(context);
                    },
                    style: ElevatedButton.styleFrom(
                      padding: EdgeInsets.symmetric(horizontal: 30.0, vertical: 20.0),
                      backgroundColor: Colors.deepPurpleAccent,
                      shape: StadiumBorder(),
                    ),
                    child: Text(
                      "Retake",
                      style: TextStyle(color: Colors.white, fontSize: 18),
                    ),
                  ),
                  ElevatedButton(
                    onPressed: _confirmVideo,
                    style: ElevatedButton.styleFrom(
                      padding: EdgeInsets.symmetric(horizontal: 40.0, vertical: 20.0),
                      backgroundColor: Colors.deepPurpleAccent,
                      shape: StadiumBorder(),
                    ),
                    child: Row(
                      children: [
                        Text("Confirm Video",
                            style: TextStyle(color: Colors.white, fontSize: 18)),
                        SizedBox(
                          width: 5,
                        ),
                        Container(
                            height: 10,
                            width: 10,
                            child: (indicator == 1)
                                ? CircularProgressIndicator(
                              color: Colors.white,
                            )
                                : SizedBox())
                      ],
                    ),
                  ),
                ],
              ),
            ),
            if (resultText.isNotEmpty) ...[
              Padding(
                padding: const EdgeInsets.all(20),
                child: Text(
                  resultText,
                  style: TextStyle(fontSize: 16, color: Colors.black),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
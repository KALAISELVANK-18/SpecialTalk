//Package imports
import 'dart:convert';
import 'package:meet/speech.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:io';

import 'package:draggable_widget/draggable_widget.dart';
import 'package:flutter/material.dart';
import 'package:hmssdk_flutter/hmssdk_flutter.dart';
import 'package:meet/main.dart';
import 'package:provider/provider.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'package:video_player/video_player.dart';
//File imports
import 'package:meet/models/data_store.dart';
import 'package:meet/services/sdk_initializer.dart';

import 'package:http/http.dart' as http;


class MeetingScreen extends StatefulWidget {
  const MeetingScreen({Key? key}) : super(key: key);

  @override
  _MeetingScreenState createState() => _MeetingScreenState();
}

class _MeetingScreenState extends State<MeetingScreen> {
  bool isLocalAudioOn = true;
  bool isLocalVideoOn = true;
  final bool _isLoading = false;

  stt.SpeechToText _speech = stt.SpeechToText();
  String _recognizedText = 'dsjsf';

  void _initializeSpeech() async {
    bool available = await _speech.initialize(
      onStatus: (status) => print('Speech status: $status'),
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
          print("${_recognizedText}on the meeting");
        });
      },
    );
  }

  void _stopListening() {
    _speech.stop();
  }

  @override
  void initState() {
    super.initState();
    _initializeSpeech();
  }

  @override
  Widget build(BuildContext context) {
    final _isVideoOff = context.select<UserDataStore, bool>(
        (user) => user.remoteVideoTrack?.isMute ?? true);
    final _peer =
        context.select<UserDataStore, HMSPeer?>((user) => user.remotePeer);
    final remoteTrack = context
        .select<UserDataStore, HMSTrack?>((user) => user.remoteVideoTrack);
    final localTrack = context
        .select<UserDataStore, HMSVideoTrack?>((user) => user.localTrack);

    return WillPopScope(
      onWillPop: () async {
        context.read<UserDataStore>().leaveRoom();
        Navigator.pop(context);
        return true;
      },
      child: SafeArea(
        child: Scaffold(
          floatingActionButton: FloatingActionButton(
            onPressed:_startListening,
          ),

          body: (_isLoading)
              ? const CircularProgressIndicator()
              : (_peer == null)
                  ? Container(
                      color: Colors.black.withOpacity(0.9),
                      width: MediaQuery.of(context).size.width,
                      height: MediaQuery.of(context).size.height,
                      child: Column(
                        children: [Stack(
                          children: [

                            Positioned(
                                child: IconButton(
                                    onPressed: () {
                                      context.read<UserDataStore>().leaveRoom();
                                      Navigator.pop(context);
                                    },
                                    icon: const Icon(
                                      Icons.arrow_back_ios,
                                      color: Colors.white,
                                    ))),
                            Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: const [
                                Padding(
                                  padding:
                                      EdgeInsets.only(left: 20.0, bottom: 20),
                                  child: Text(
                                    "You're the only one here",
                                    style: TextStyle(
                                        color: Colors.white,
                                        fontSize: 20,
                                        fontWeight: FontWeight.bold),
                                  ),
                                ),
                                Padding(
                                  padding: EdgeInsets.only(left: 20.0),
                                  child: Text(
                                    "Share meeting link with others",
                                    style: TextStyle(
                                        color: Colors.white,
                                        fontSize: 12,
                                        fontWeight: FontWeight.bold),
                                  ),
                                ),
                                Padding(
                                  padding: EdgeInsets.only(left: 20.0),
                                  child: Text(
                                    "that you want in the meeting",
                                    style: TextStyle(
                                        color: Colors.white,
                                        fontSize: 12,
                                        fontWeight: FontWeight.bold),
                                  ),
                                ),
                                Padding(
                                  padding: EdgeInsets.only(left: 20.0, top: 10),
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                  ),
                                ),
                              ],
                            ),
                            DraggableWidget(
                              topMargin: 10,
                              bottomMargin: 130,
                              horizontalSpace: 10,
                              child: localPeerTile3(),
                            ),

                          ],
                        ),

                        ]
                      ),
                    )
                  : SingleChildScrollView(
                    child: Column(
                      children: [
                        SizedBox(
                            height: 500,
                            width: 500,
                            child: Stack(
                              children: [
                                Container(
                                    color: Colors.black.withOpacity(0.9),
                                    child: _isVideoOff
                                        ? Center(
                                            child: Container(
                                              decoration: BoxDecoration(
                                                  shape: BoxShape.circle,
                                                  boxShadow: [
                                                    BoxShadow(
                                                      color:
                                                          Colors.blue.withAlpha(60),
                                                      blurRadius: 10.0,
                                                      spreadRadius: 2.0,
                                                    ),
                                                  ]),
                                              child: const Icon(
                                                Icons.videocam_off,
                                                color: Colors.white,
                                                size: 30,
                                              ),
                                            ),
                                          )
                                        : (remoteTrack != null)
                                            ? Container(
                                                child: HMSVideoView(
                                                  scaleType: ScaleType.SCALE_ASPECT_FILL,
                                                  track: remoteTrack as HMSVideoTrack,
                                                ),
                                              )
                                            : const Center(child: Text("No Video",style: TextStyle(color: Colors.white),))),
                                Align(
                                  alignment: Alignment.bottomCenter,
                                  child: Padding(
                                    padding: const EdgeInsets.only(bottom: 15),
                                    child: Row(
                                      mainAxisAlignment:
                                          MainAxisAlignment.spaceEvenly,
                                      children: [
                                        GestureDetector(
                                          onTap: () async {
                                            context.read<UserDataStore>().leaveRoom();
                                            Navigator.pop(context);
                                          },
                                          child: Container(
                                            decoration: BoxDecoration(
                                                shape: BoxShape.circle,
                                                boxShadow: [
                                                  BoxShadow(
                                                    color: Colors.red.withAlpha(60),
                                                    blurRadius: 3.0,
                                                    spreadRadius: 5.0,
                                                  ),
                                                ]),
                                            child: const CircleAvatar(
                                              radius: 25,
                                              backgroundColor: Colors.red,
                                              child: Icon(Icons.call_end,
                                                  color: Colors.white),
                                            ),
                                          ),
                                        ),
                                        GestureDetector(
                                          onTap: () => {
                                            SdkInitializer.hmssdk
                                                .toggleCameraMuteState(),
                                            setState(() {
                                              isLocalVideoOn = !isLocalVideoOn;
                                            })
                                          },
                                          child: CircleAvatar(
                                            radius: 25,
                                            backgroundColor:
                                                Colors.transparent.withOpacity(0.2),
                                            child: Icon(
                                              isLocalVideoOn
                                                  ? Icons.videocam
                                                  : Icons.videocam_off_rounded,
                                              color: Colors.white,
                                            ),
                                          ),
                                        ),
                                        GestureDetector(
                                          onTap: () => {

                                            SdkInitializer.hmssdk
                                                .toggleMicMuteState(),
                                            setState(() {

                                              isLocalAudioOn = !isLocalAudioOn;
                                            })
                                          },
                                          child: CircleAvatar(
                                            radius: 25,
                                            backgroundColor:
                                                Colors.transparent.withOpacity(0.2),
                                            child: Icon(
                                              isLocalAudioOn
                                                  ? Icons.mic
                                                  : Icons.mic_off,
                                              color: Colors.white,
                                            ),
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ),
                                Positioned(
                                  top: 10,
                                  left: 10,
                                  child: GestureDetector(
                                    onTap: () {
                                      context.read<UserDataStore>().leaveRoom();
                                      Navigator.pop(context);
                                    },
                                    child: const Icon(
                                      Icons.arrow_back_ios,
                                      color: Colors.white,
                                    ),
                                  ),
                                ),
                                Positioned(
                                  top: 10,
                                  right: 10,
                                  child: GestureDetector(
                                    onTap: () {
                                      if (isLocalVideoOn) {
                                        SdkInitializer.hmssdk.switchCamera();
                                      }
                                    },
                                    child: CircleAvatar(
                                      radius: 25,
                                      backgroundColor:
                                          Colors.transparent.withOpacity(0.2),
                                      child: const Icon(
                                        Icons.switch_camera_outlined,
                                        color: Colors.white,
                                      ),
                                    ),
                                  ),
                                ),
                                DraggableWidget(
                                  topMargin: 10,
                                  bottomMargin: 400,
                                  horizontalSpace: 10,
                                  child: localPeerTile(localTrack),
                                ),
                              ],
                            ),
                          ),SizedBox(height: 10,),
                        Text("Converted video"),
                        SizedBox(height: 10,)
                        ,Container(

                          child: localPeerTile3(),
                        ),

                      ],
                    ),
                  ),
        ),
      ),
    );
  }

  Widget localPeerTile(HMSVideoTrack? localTrack) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(10),
      child: Container(
        height: 150,
        width: 100,
        color: Colors.black,
        child: (isLocalVideoOn && localTrack != null)
            ? HMSVideoView(
                track: localTrack,
              )
            : const Icon(
                Icons.videocam_off_rounded,
                color: Colors.white,
              ),
      ),
    );
  }

  Widget localPeerTile3() {
    return ClipRRect(
      borderRadius: BorderRadius.circular(10),
      child: Container(
        height: 300,
        width: 200,
        color: Colors.white,
        child: (isLocalAudioOn==true)
            ? VideoPlayerScreen()
            : Icon(
          Icons.videocam_off_rounded,
          color: Colors.white,
        ),
      ),
    );
  }

}



class VideoPlayerScreen extends StatefulWidget {
  @override
  _VideoPlayerScreenState createState() => _VideoPlayerScreenState();
}

class _VideoPlayerScreenState extends State<VideoPlayerScreen> {
  late VideoPlayerController _controller;
  bool _isInitialized = false;

  @override
  void initState() {
    super.initState();
    _fetchAndSaveVideo();
  }

  Future<void> _fetchAndSaveVideo() async {
    final url = Uri.parse('http://192.168.207.166:8000/generate_video');

    final response = await http.post(
      url,
      headers: <String, String>{
        'Content-Type': 'application/json; charset=UTF-8',
      },
      body: jsonEncode(<String, String>{
        'text': 'Your text here', // Replace with your text input
      }),
    );

    if (response.statusCode == 200) {
      final appDir = await getApplicationDocumentsDirectory();
      final videoFile = File('${appDir.path}/merged_video.mp4');
      await videoFile.writeAsBytes(response.bodyBytes);
      print(response.bodyBytes);
      _controller = VideoPlayerController.file(videoFile)
        ..initialize().then((_) {
          setState(() {
            _isInitialized = true;
          });
        });
    } else {
      print('Failed to load video: ${response.statusCode}');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        Center(
          child: _isInitialized
              ? AspectRatio(
            aspectRatio: _controller.value.aspectRatio,
            child: VideoPlayer(_controller),
          )
              : CircularProgressIndicator(),
        ),
        Positioned(
          bottom: 16.0,
          right: 16.0,
          child: FloatingActionButton(
            onPressed: () {
              if (_controller.value.isPlaying) {
                _controller.pause();
              } else {
                _controller.play();
              }
              setState(() {});
            },
            child: Icon(
              _controller.value.isPlaying ? Icons.pause : Icons.play_arrow,
            ),
          ),
        ),
      ],
    );

  }

  @override
  void dispose() {
    super.dispose();
    _controller.dispose();
  }
}
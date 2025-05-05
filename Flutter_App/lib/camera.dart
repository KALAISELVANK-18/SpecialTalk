import 'dart:async';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:meet/showVideo.dart';

class TakeVideoScreen extends StatefulWidget {
  final List<CameraDescription>? cameras;
  TakeVideoScreen({super.key, required this.cameras});

  @override
  State<TakeVideoScreen> createState() => _TakeVideoScreenState();
}

class _TakeVideoScreenState extends State<TakeVideoScreen> {
  late CameraController _controller;
  late double minzoomoffset = 1;
  int indicator = 0;
  late double maxzoomoffset = 10;
  late FutureOr<void> _initializeControllerFuture;
  double _currentZoomOffset = 1;
  bool _isRecording = false;
  int _selectedCameraIndex = 0;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  void _initializeCamera() {
    _controller = CameraController(
      widget.cameras![_selectedCameraIndex],
      ResolutionPreset.high,
      enableAudio: false,

    );
    _controller.initialize().then((_) {
      _controller.setFlashMode(FlashMode.off);
      if (!mounted) {
        return;
      }
      setState(() {});
    });
  }

  void _switchCamera() {
    setState(() {
      _selectedCameraIndex =
          (_selectedCameraIndex + 1) % widget.cameras!.length;
      _initializeCamera();
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: SingleChildScrollView(
        child: Column(
          children: [
            SizedBox(
              height: 30,
            ),
            Padding(
              padding: const EdgeInsets.all(10),
              child: Container(
                decoration:
                    BoxDecoration(borderRadius: BorderRadius.circular(20)),
                child: (_controller.value.isInitialized)
                    ? Container(
                        child: Center(
                        child: Container(
                          child: ClipRRect(
                              borderRadius: BorderRadius.circular(16.0),
                              child: CameraPreview(_controller)),
                        ),
                      ))
                    : CircularProgressIndicator(),
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text('-', style: TextStyle(fontSize: 40)),
                  Expanded(
                    child: Slider(
                      value: _currentZoomOffset,
                      min: minzoomoffset,
                      max: maxzoomoffset,
                      onChanged: (newv) {
                        setState(() {
                          _currentZoomOffset = newv;
                          if (newv > minzoomoffset && newv < maxzoomoffset) {
                            _controller.setZoomLevel(newv);
                          }
                        });
                      },
                    ),
                  ),
                  Text(
                    '+',
                    style: TextStyle(fontSize: 30),
                  ),
                ],
              ),
            ),
            Row(
              children: [
                Padding(
                  padding: const EdgeInsets.only(top: 0, left: 160),
                  child: GestureDetector(
                    onTap: () async {
                      try {
                        if (_isRecording) {
                          // Stop recording video
                          final video = await _controller.stopVideoRecording();
                          setState(() {
                            _isRecording = false;
                            indicator = 0;
                          });
                          Navigator.of(context).push(MaterialPageRoute(
                            builder: (context) => ShowVideoScreen(
                              videoPath: video.path, // Pass appropriate data
                            ),
                          ));
                        } else {
                          // Start recording video
                          await _controller.prepareForVideoRecording();
                          await _controller.startVideoRecording();
                          setState(() {
                            _isRecording = true;
                            indicator = 0;
                          });
                        }
                        if (!mounted) return;
                      } catch (e) {
                        print(e);
                        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
                          content: const Text('Issue in connecting cameras'),
                          action: SnackBarAction(
                            label: 'Undo',
                            onPressed: () {
                              // Some code to undo the change.
                            },
                          ),
                        ));
                      }
                    },
                    child: Container(
                      width: 70, // Outer container width
                      height: 70, // Outer container height
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        border: Border.all(
                          color: Colors.grey, // Grey border color
                          width: 8, // Outer border width
                        ),
                        color: Colors.transparent, // Transparent inner circle
                      ),
                      child: Center(
                        // Center the inner container
                        child: Container(
                          width: _isRecording
                              ? 35
                              : 30, // Width changes based on condition
                          height: _isRecording
                              ? 35
                              : 30, // Height changes based on condition
                          decoration: BoxDecoration(
                            shape: _isRecording
                                ? BoxShape.rectangle
                                : BoxShape.circle,
                            color: Colors.red, // Inner container color
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.only(top: 0,left: 60),
                  child: IconButton(
                    icon: Icon(Icons.cameraswitch_outlined, size: 55),
                    onPressed: _switchCamera,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),

      floatingActionButtonLocation: FloatingActionButtonLocation.centerDocked,
    );
  }
}

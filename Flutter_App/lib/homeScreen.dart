import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:meet/camera.dart';
import 'package:meet/main.dart';
import 'package:meet/speech.dart';
import 'package:video_player/video_player.dart';

class Homescreen extends StatefulWidget {
  final List<CameraDescription>? cameras;
  const Homescreen({super.key, required this.cameras});

  @override
  State<Homescreen> createState() => _HomescreenState();
}

class _HomescreenState extends State<Homescreen> {
  late VideoPlayerController _controller;
  @override
  void initState() {
    super.initState();

    _controller = VideoPlayerController.asset('assets/video1.mp4')
      ..initialize().then((_) {
        setState(() {}); // When the video is ready, rebuild the UI
        _controller.play(); // Play the video automatically
      });
  }

  @override
  void dispose() {
    _controller.dispose(); // Dispose of the controller to free resources
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: SingleChildScrollView(
        child: Container(
          child: Column(
            children: [
              Container(
                height: 470,
                width: MediaQuery.of(context).size.width,
                decoration: BoxDecoration(
                    color: Colors.blue,
                    borderRadius: BorderRadius.only(
                        bottomLeft: Radius.circular(100),
                        bottomRight: Radius.circular(100))),
                child: Center(
                        child: Image.asset(
                        'assets/image.png',
                        height: 350,
                      ) // Show a loading indicator until the video is ready
                        ),
              ),
              Padding(
                padding: const EdgeInsets.all(20.0),
                child: Row(
                  children: [
                    Text(
                      "Sign Language",
                      style: TextStyle(
                        fontSize: 25,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    Padding(
                      padding: const EdgeInsets.only(left: 4),
                      child: Text(
                        "Translator",
                        style: TextStyle(
                            fontSize: 25,
                            fontWeight: FontWeight.w800,
                            color: Colors.redAccent),
                      ),
                    ),
                  ],
                  mainAxisAlignment: MainAxisAlignment.center,
                ),
              ),
              Padding(
                padding: const EdgeInsets.only(left: 25, right: 25, top: 30),
                child: Row(
                  children: [
                    GestureDetector(
                      onTap: () => {
                           Navigator.push(context, MaterialPageRoute(builder: (context)=>SpeechToTextScreen()))
                      },
                      child: Container(
                        height: 165,
                        width: 165,
                        decoration: BoxDecoration(
                          color:
                              Colors.white, // Background color of the container
                          borderRadius:
                              BorderRadius.circular(20), // Rounded corners
                          boxShadow: [
                            BoxShadow(
                              color: Colors.grey
                                  .withOpacity(0.5), // Shadow color with opacity
                              spreadRadius: 5, // How much the shadow spreads
                              blurRadius: 7, // The blur radius of the shadow
                              offset: Offset(
                                  0, 3), // Offset for the shadow: X and Y axis
                            ),
                          ],
                        ),
                        child: Center(
                          child: Column(
                            children: [
                              Image.asset(
                                "assets/image2.png",
                                height: 120,
                              ),
                              Padding(
                                padding: const EdgeInsets.all(8.0),
                                child: Text(
                                  "Sign Convertor",
                                  style: TextStyle(
                                      fontSize: 18, fontWeight: FontWeight.bold),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                    GestureDetector(
                      onTap: () => {
                        Navigator.push(context, MaterialPageRoute(builder: (context)=> TakeVideoScreen(cameras: cameras,)))
                      },
                      child: Container(
                        height: 165,
                        width: 165,
                        decoration: BoxDecoration(
                          color:
                              Colors.white, // Background color of the container
                          borderRadius:
                              BorderRadius.circular(20), // Rounded corners
                          boxShadow: [
                            BoxShadow(
                              color: Colors.grey
                                  .withOpacity(0.5), // Shadow color with opacity
                              spreadRadius: 5, // How much the shadow spreads
                              blurRadius: 7, // The blur radius of the shadow
                              offset: Offset(
                                  0, 3), // Offset for the shadow: X and Y axis
                            ),
                          ],
                        ),
                        child: Center(
                          child: Column(
                            children: [
                              Image.asset(
                                "assets/image3.png",
                                height: 120,
                              ),
                              Padding(
                                padding: const EdgeInsets.all(8.0),
                                child: Text(
                                  "Text Convertor",
                                  style: TextStyle(
                                      fontSize: 18, fontWeight: FontWeight.bold),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ],
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                ),
              )
            ],
          ),
        ),
      ),
    );
  }
}

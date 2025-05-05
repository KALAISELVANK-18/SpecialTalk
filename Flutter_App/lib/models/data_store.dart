//Dart imports
import 'dart:developer';

import 'package:speech_to_text/speech_to_text.dart' as stt;

//Package imports
import 'package:flutter/material.dart';
import 'package:hmssdk_flutter/hmssdk_flutter.dart';

//File imports
import 'package:meet/services/sdk_initializer.dart';

class UserDataStore extends ChangeNotifier
    implements HMSUpdateListener, HMSActionResultListener {
  HMSTrack? remoteVideoTrack;
  HMSPeer? remotePeer;
  HMSTrack? remoteAudioTrack;
  HMSVideoTrack? localTrack;
  bool _disposed = false;
  late HMSPeer localPeer;
  bool isRoomEnded = false;
  final stt.SpeechToText _speech = stt.SpeechToText();

  Future<void> startSpeechRecognition() async {
    bool available = await _speech.initialize();
    print("Iam d");
    if (available) {
        print("Iam in");
      _speech.listen(

        onResult: (result) {
          String text = result.recognizedWords;
          // You can handle the recognized text here
          print('Recognized text: $text');
          // Call a method to process the recognized text further
          processRecognizedText(text);
        },

      );
    }
  }

  void processRecognizedText(String text) {
    // Implement your logic to process the recognized text here
    // For example, you could display it in the UI, send it to a server for further processing, etc.
  }



  void startListen() {
    SdkInitializer.hmssdk.addUpdateListener(listener: this);
  }

  void leaveRoom() async {
    SdkInitializer.hmssdk.leave(hmsActionResultListener: this);
  }

  @override
  void dispose() {
    _disposed = true;
    super.dispose();
  }

  @override
  void notifyListeners() {
    if (!_disposed) {
      super.notifyListeners();
    }
  }

  @override
  void onJoin({required HMSRoom room}) {
    for (HMSPeer each in room.peers!) {
      if (each.isLocal) {
        localPeer = each;
        break;
      }
    }
  }

  @override
  void onPeerUpdate({required HMSPeer peer, required HMSPeerUpdate update}) {
    switch (update) {
      case HMSPeerUpdate.peerJoined:
        remotePeer = peer;
        remoteAudioTrack = peer.audioTrack;
        print(remotePeer?.audioTrack?.isMute);
        remoteVideoTrack = peer.videoTrack;
        break;
      case HMSPeerUpdate.peerLeft:
        remotePeer = null;
        break;
      case HMSPeerUpdate.roleUpdated:
        break;
      case HMSPeerUpdate.metadataChanged:
        break;
      case HMSPeerUpdate.nameChanged:
        break;
      case HMSPeerUpdate.defaultUpdate:
        break;
      case HMSPeerUpdate.networkQualityUpdated:
        break;
      case HMSPeerUpdate.handRaiseUpdated:
        break;
    }
    notifyListeners();
  }

  @override
  void onTrackUpdate(
      {required HMSTrack track,
      required HMSTrackUpdate trackUpdate,
      required HMSPeer peer}) {
    switch (trackUpdate) {
      case HMSTrackUpdate.trackAdded:
        if (track.kind == HMSTrackKind.kHMSTrackKindAudio) {
          if (!peer.isLocal){ remoteAudioTrack = track;
          }
        } else if (track.kind == HMSTrackKind.kHMSTrackKindVideo) {
          if (!peer.isLocal) {
            remoteVideoTrack = track;
          } else {
            localTrack = track as HMSVideoTrack;
          }
        }
        break;
      case HMSTrackUpdate.trackRemoved:
        if (track.kind == HMSTrackKind.kHMSTrackKindAudio) {
          if (!peer.isLocal) remoteAudioTrack = null;
        } else if (track.kind == HMSTrackKind.kHMSTrackKindVideo) {
          if (!peer.isLocal) {
            remoteVideoTrack = null;
          } else {
            localTrack = null;
          }
        }
        break;
      case HMSTrackUpdate.trackMuted:
        if (track.kind == HMSTrackKind.kHMSTrackKindAudio) {
          if (!peer.isLocal) remoteAudioTrack = track;
        } else if (track.kind == HMSTrackKind.kHMSTrackKindVideo) {
          if (!peer.isLocal) {
            remoteVideoTrack = track;
          } else {
            localTrack = null;
          }
        }
        break;
      case HMSTrackUpdate.trackUnMuted:
        if (track.kind == HMSTrackKind.kHMSTrackKindAudio) {
          if (!peer.isLocal) remoteAudioTrack = track;
        } else if (track.kind == HMSTrackKind.kHMSTrackKindVideo) {
          if (!peer.isLocal) {
            remoteVideoTrack = track;
          } else {
            localTrack = track as HMSVideoTrack;
          }
        }
        break;
      case HMSTrackUpdate.trackDescriptionChanged:
        break;
      case HMSTrackUpdate.trackDegraded:
        break;
      case HMSTrackUpdate.trackRestored:
        break;
      case HMSTrackUpdate.defaultUpdate:
        break;
    }
    notifyListeners();
  }

  @override
  void onHMSError({required HMSException error}) {
    log(error.message??"");
  }

  @override
  void onMessage({required HMSMessage message}) {}

  @override
  void onRoomUpdate({required HMSRoom room, required HMSRoomUpdate update}) {}

  @override
  void onUpdateSpeakers({required List<HMSSpeaker> updateSpeakers}) {
    // startSpeechRecognition();
  }

  @override
  void onReconnected() {}

  @override
  void onReconnecting() {}

  @override
  void onRemovedFromRoom(
      {required HMSPeerRemovedFromPeer hmsPeerRemovedFromPeer}) {}

  @override
  void onRoleChangeRequest({required HMSRoleChangeRequest roleChangeRequest}) {}

  @override
  void onChangeTrackStateRequest(
      {required HMSTrackChangeRequest hmsTrackChangeRequest}) {}

  @override
  void onAudioDeviceChanged(
      {HMSAudioDevice? currentAudioDevice,
      List<HMSAudioDevice>? availableAudioDevice}) {}

  @override
  void onException({
    required HMSActionResultListenerMethod methodType,
    Map<String, dynamic>? arguments,
    required HMSException hmsException,
  }) {
    switch (methodType) {
      case HMSActionResultListenerMethod.leave:
        log("Leave room error ${hmsException.message}");
        break;
      case HMSActionResultListenerMethod.changeTrackState:
      // Handle change track state method type
        break;
      case HMSActionResultListenerMethod.changeMetadata:
      // Handle change metadata method type
        break;
      case HMSActionResultListenerMethod.endRoom:
      // Handle end room method type
        break;
      case HMSActionResultListenerMethod.removePeer:
      // Handle remove peer method type
        break;
      case HMSActionResultListenerMethod.acceptChangeRole:
      // Handle accept change role method type
        break;
      case HMSActionResultListenerMethod.changeRoleOfPeer:
      // Handle change role of peer method type
        break;
      case HMSActionResultListenerMethod.changeTrackStateForRole:
      // Handle change track state for role method type
        break;
      case HMSActionResultListenerMethod.startRtmpOrRecording:
      // Handle start RTMP or recording method type
        break;
      case HMSActionResultListenerMethod.stopRtmpAndRecording:
      // Handle stop RTMP and recording method type
        break;
      case HMSActionResultListenerMethod.changeName:
      // Handle change name method type
        break;
      case HMSActionResultListenerMethod.sendBroadcastMessage:
      // Handle send broadcast message method type
        break;
      case HMSActionResultListenerMethod.sendGroupMessage:
      // Handle send group message method type
        break;
      case HMSActionResultListenerMethod.sendDirectMessage:
      // Handle send direct message method type
        break;
      case HMSActionResultListenerMethod.hlsStreamingStarted:
      // Handle HLS streaming started method type
        break;
      case HMSActionResultListenerMethod.hlsStreamingStopped:
      // Handle HLS streaming stopped method type
        break;
      case HMSActionResultListenerMethod.startScreenShare:
      // Handle start screen share method type
        break;
      case HMSActionResultListenerMethod.stopScreenShare:
      // Handle stop screen share method type
        break;
      case HMSActionResultListenerMethod.startAudioShare:
      // Handle start audio share method type
        break;
      case HMSActionResultListenerMethod.stopAudioShare:
      // Handle stop audio share method type
        break;
      case HMSActionResultListenerMethod.switchCamera:
      // Handle switch camera method type
        break;
      case HMSActionResultListenerMethod.changeRoleOfPeersWithRoles:
      // Handle change role of peers with roles method type
        break;
      case HMSActionResultListenerMethod.setSessionMetadataForKey:
      // Handle set session metadata for key method type
        break;
      case HMSActionResultListenerMethod.sendHLSTimedMetadata:
      // Handle send HLS timed metadata method type
        break;
      case HMSActionResultListenerMethod.lowerLocalPeerHand:
      // Handle lower local peer hand method type
        break;
      case HMSActionResultListenerMethod.lowerRemotePeerHand:
      // Handle lower remote peer hand method type
        break;
      case HMSActionResultListenerMethod.raiseLocalPeerHand:
      // Handle raise local peer hand method type
        break;
      case HMSActionResultListenerMethod.quickStartPoll:
      // Handle quick start poll method type
        break;
      case HMSActionResultListenerMethod.addSingleChoicePollResponse:
      // Handle add single choice poll response method type
        break;
      case HMSActionResultListenerMethod.addMultiChoicePollResponse:
      // Handle add multi choice poll response method type
        break;
      case HMSActionResultListenerMethod.unknown:
      // Handle unknown method type
        break;
    }
  }

  @override
  void onSuccess(
      {required HMSActionResultListenerMethod methodType,
      Map<String, dynamic>? arguments}) {
    switch (methodType) {
      case HMSActionResultListenerMethod.leave:
        isRoomEnded = true;
        notifyListeners();
        break;

      case HMSActionResultListenerMethod.changeTrackState:
      // Handle change track state method type
        break;
      case HMSActionResultListenerMethod.changeMetadata:
      // Handle change metadata method type
        break;
      case HMSActionResultListenerMethod.endRoom:
      // Handle end room method type
        break;
      case HMSActionResultListenerMethod.removePeer:
      // Handle remove peer method type
        break;
      case HMSActionResultListenerMethod.acceptChangeRole:
      // Handle accept change role method type
        break;
      case HMSActionResultListenerMethod.changeRoleOfPeer:
      // Handle change role of peer method type
        break;
      case HMSActionResultListenerMethod.changeTrackStateForRole:
      // Handle change track state for role method type
        break;
      case HMSActionResultListenerMethod.startRtmpOrRecording:
      // Handle start RTMP or recording method type
        break;
      case HMSActionResultListenerMethod.stopRtmpAndRecording:
      // Handle stop RTMP and recording method type
        break;
      case HMSActionResultListenerMethod.changeName:
      // Handle change name method type
        break;
      case HMSActionResultListenerMethod.sendBroadcastMessage:
      // Handle send broadcast message method type
        break;
      case HMSActionResultListenerMethod.sendGroupMessage:
      // Handle send group message method type
        break;
      case HMSActionResultListenerMethod.sendDirectMessage:
      // Handle send direct message method type
        break;
      case HMSActionResultListenerMethod.hlsStreamingStarted:
      // Handle HLS streaming started method type
        break;
      case HMSActionResultListenerMethod.hlsStreamingStopped:
      // Handle HLS streaming stopped method type
        break;
      case HMSActionResultListenerMethod.startScreenShare:
      // Handle start screen share method type
        break;
      case HMSActionResultListenerMethod.stopScreenShare:
      // Handle stop screen share method type
        break;
      case HMSActionResultListenerMethod.startAudioShare:
      // Handle start audio share method type
        break;
      case HMSActionResultListenerMethod.stopAudioShare:
      // Handle stop audio share method type
        break;
      case HMSActionResultListenerMethod.switchCamera:
      // Handle switch camera method type
        break;
      case HMSActionResultListenerMethod.changeRoleOfPeersWithRoles:
      // Handle change role of peers with roles method type
        break;
      case HMSActionResultListenerMethod.setSessionMetadataForKey:
      // Handle set session metadata for key method type
        break;
      case HMSActionResultListenerMethod.sendHLSTimedMetadata:
      // Handle send HLS timed metadata method type
        break;
      case HMSActionResultListenerMethod.lowerLocalPeerHand:
      // Handle lower local peer hand method type
        break;
      case HMSActionResultListenerMethod.lowerRemotePeerHand:
      // Handle lower remote peer hand method type
        break;
      case HMSActionResultListenerMethod.raiseLocalPeerHand:
      // Handle raise local peer hand method type
        break;
      case HMSActionResultListenerMethod.quickStartPoll:
      // Handle quick start poll method type
        break;
      case HMSActionResultListenerMethod.addSingleChoicePollResponse:
      // Handle add single choice poll response method type
        break;
      case HMSActionResultListenerMethod.addMultiChoicePollResponse:
      // Handle add multi choice poll response method type
        break;
      case HMSActionResultListenerMethod.unknown:
      // Handle unknown method type
        break;
    }
  }

  @override
  void onPeerListUpdate({required List<HMSPeer> addedPeers, required List<HMSPeer> removedPeers}) {
    // TODO: implement onPeerListUpdate
  }

  @override
  void onSessionStoreAvailable({HMSSessionStore? hmsSessionStore}) {
    // TODO: implement onSessionStoreAvailable
  }
}

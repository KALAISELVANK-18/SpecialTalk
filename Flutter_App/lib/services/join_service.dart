//Dart imports
import 'dart:convert';

//Package imports
import 'package:hmssdk_flutter/hmssdk_flutter.dart';
import 'package:http/http.dart' as http;

class JoinService {
  static Future<bool> join(HMSSDK hmssdk) async {
    String roomId = "65cf15ea4396bd63f20730cb";
    Uri endPoint = Uri.parse(
        "https://prod-in2.100ms.live/hmsapi/kalaiselvan-videoconf-1329.app.100ms.live/api/token");
    http.Response response = await http.post(endPoint,
        body: {'user_id': "user", 'room_id': roomId, 'role': "host"});
    var body = json.decode(response.body);
    if (body == null || body['token'] == null) {
      return false;
    }
    HMSConfig config = HMSConfig(authToken: body['token'], userName: "user");
    await hmssdk.join(config: config);
    return true;
  }
}

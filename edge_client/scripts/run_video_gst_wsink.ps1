gst-launch-1.0 webrtcsink name=wsink `
>>   run-signalling-server=true run-web-server=true `
>>   congestion-control=disabled `
>>   enable-mitigation-modes=none `
>>   start-bitrate=6000000 `
>>   min-bitrate=3000000 `
>>   max-bitrate=12000000 `
>>   video-caps="video/x-h264" `
>>   mfvideosrc do-timestamp=true ! videoconvert ! videorate ! `
>>   video/x-raw,format=I420,framerate=12/1,width=640,height=480 ! `
>>   queue max-size-buffers=1 leaky=downstream ! wsink. `
>>   wasapisrc low-latency=true ! audioconvert ! audioresample ! `
>>   audio/x-raw,rate=48000,channels=1 ! queue ! wsink.
// Connecting to ROS
// -----------------
var beginning = Date.now();
var ros = new ROSLIB.Ros();

// If there is an error on the backend, an 'error' emit will be emitted.
ros.on('error', function(error) {
  document.getElementById('connecting').style.display = 'none';
  document.getElementById('connected').style.display = 'none';
  document.getElementById('closed').style.display = 'none';
  document.getElementById('error').style.display = 'block';
  document.getElementById('troubleshooting').style.display = 'inline-block';
  console.log(error);
});

// Find out exactly when we made a connection.
ros.on('connection', function() {
  console.log('Connection made!');
  document.getElementById('connecting').style.display = 'none';
  document.getElementById('error').style.display = 'none';
  document.getElementById('closed').style.display = 'none';
  document.getElementById('connected').style.display = 'block';
});

ros.on('close', function() {
  console.log('Connection closed.');
  document.getElementById('connecting').style.display = 'none';
  document.getElementById('connected').style.display = 'none';
  document.getElementById('closed').style.display = 'inline-block';
  document.getElementById('error').style.display = 'inline-block';
});

// Guess connection of the rosbridge websocket
function getRosBridgeHost() {
  return 'localhost';

  // if (window.location.protocol == 'file:') {
  //   return '192.168.1.3';
  // } else {
  //   return window.location.hostname;
  // }
}

var rosBridgePort = 9090;
// Create a connection to the rosbridge WebSocket server.
console.log('Connecting to ws://' + getRosBridgeHost() + ':' + rosBridgePort)
ros.connect('ws://' + getRosBridgeHost() + ':' + rosBridgePort);

// First, we create a Topic object with details of the topic's name and message type.
var jsonTopic = new ROSLIB.Topic({
  ros : ros,
  name : '/web_interface/json',
  messageType : 'std_msgs/String'
});

jsonTopic.subscribe(function(message) {
    // console.log('Received message on ' + jsonTopic.name + ': ' + message.data);
    defaultjsondata = message.data;
    // d3.select('svg').selectAll('*').remove();
    loadhtm('');
  });

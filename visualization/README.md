# Content of this folder

 * `htm` → HTM visualization from json file
 * `htm-ros-subscriber` → HTM visualization with a ROS subscriber
 * `policy` →
 * `pomcp` →
 * `trajectories` →

## Test

To test `htm-ros-publisher`, do the following:

 * `roscore`
 * `roslaunch rosbridge_server rosbridge_websocket.launch`

### Example 1

```
rostopic pub /web_interface/json std_msgs/String '{data: "{ \"nodes\": { \"id\": 0, \"parent\": null, \"name\": \"Start\", \"combination\": \"Sequential\", \"attributes\": [], \"children\": [ { \"id\": 1, \"parent\": 0, \"name\": \"BUILD CHAIR\", \"combination\": \"Sequential\", \"attributes\": [], \"children\": [] }]} }"}'
```

```
rostopic pub /web_interface/json std_msgs/String '{data: "{ \"nodes\": { \"id\": 0, \"parent\": null, \"name\": \"Start\", \"combination\": \"Sequential\", \"attributes\": [], \"children\": [ { \"id\": 1, \"parent\": 0, \"name\": \"BUILD CHAIR\", \"combination\": \"Sequential\", \"attributes\": [], \"children\": [ { \"id\": 2, \"parent\": 1, \"name\": \"GET(screwdriver)\", \"combination\": \"Sequential\", \"attributes\": [\"robot\"], \"children\": [] }] }]} }"}'
```

```
rostopic pub /web_interface/json std_msgs/String '{data: "{ \"nodes\": { \"id\": 0, \"parent\": null, \"name\": \"Start\", \"combination\": \"Sequential\", \"attributes\": [], \"children\": [ { \"id\": 1, \"parent\": 0, \"name\": \"BUILD CHAIR\", \"combination\": \"Sequential\", \"attributes\": [], \"children\": [ { \"id\": 2, \"parent\": 1, \"name\": \"GET(screwdriver)\", \"combination\": \"Sequential\", \"attributes\": [\"robot\"], \"children\": [] }, { \"id\": 3, \"parent\": 1, \"name\": \"GET(dowel)\", \"combination\": \"Sequential\", \"attributes\": [\"robot\"], \"children\": [] }] }]} }"}'
```

```
rostopic pub /web_interface/json std_msgs/String '{data: "{ \"nodes\": { \"id\": 0, \"parent\": null, \"name\": \"Start\", \"combination\": \"Sequential\", \"attributes\": [], \"children\": [ { \"id\": 1, \"parent\": 0, \"name\": \"BUILD CHAIR\", \"combination\": \"Sequential\", \"attributes\": [], \"children\": [ { \"id\": 2, \"parent\": 1, \"name\": \"GET(screwdriver)\", \"combination\": \"Sequential\", \"attributes\": [\"robot\"], \"children\": [] }, { \"id\": 3, \"parent\": 1, \"name\": \"GET(dowel)\", \"combination\": \"Sequential\", \"attributes\": [\"robot\"], \"children\": [] }, { \"id\": 4, \"parent\": 1, \"name\": \"GET(screws)\", \"combination\": \"Sequential\", \"attributes\": [\"human\"], \"children\": [] }] }]} }"}'
```

#### Example 2

```
rostopic pub /web_interface/json std_msgs/String '{data: "{ \"nodes\": { \"id\": 0, \"parent\": null, \"name\": \"Start\", \"combination\": \"Sequential\", \"attributes\": [], \"children\": [ { \"id\": 1, \"parent\": 0, \"name\": \"BUILD CHAIR\", \"combination\": \"Sequential\", \"attributes\": [], \"children\": []}]}}"}'
```

```
rostopic pub /web_interface/json std_msgs/String '{data: "{ \"nodes\": { \"id\": 0, \"parent\": null, \"name\": \"Start\", \"combination\": \"Sequential\", \"attributes\": [], \"children\": [ { \"id\": 1, \"parent\": 0, \"name\": \"BUILD CHAIR\", \"combination\": \"Sequential\", \"attributes\": [], \"children\": [ { \"id\": 2, \"parent\": 1, \"name\": \"REQUEST-ACTION GIVE SCREWDRIVER\", \"combination\": \"Sequential\", \"attributes\": [], \"children\": []}]}]}}"}'
```

```
rostopic pub /web_interface/json std_msgs/String '{data: "{ \"nodes\": { \"id\": 0, \"parent\": null, \"name\": \"Start\", \"combination\": \"Sequential\", \"attributes\": [], \"children\": [ { \"id\": 1, \"parent\": 0, \"name\": \"BUILD CHAIR\", \"combination\": \"Sequential\", \"attributes\": [], \"children\": [ { \"id\": 2, \"parent\": 1, \"name\": \"REQUEST-ACTION GIVE SCREWDRIVER\", \"combination\": \"Sequential\", \"attributes\": [], \"children\": [ { \"id\": 3, \"parent\": 2, \"name\": \"GET(screwdriver)\", \"combination\": \"\", \"attributes\": [ \"robot\" ], \"children\": []}]}]}]}}"}'
```

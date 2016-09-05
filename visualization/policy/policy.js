
var jsonfile = "policy.json";

var svg = d3.select("svg"),
    width = +svg.attr("width"),
    height = +svg.attr("height");

appendmarkers();

d3.json(jsonfile, function(error, json)
{
  if (error) {throw error;}

  var nodes = Array();
  var links = Array();

  // Assign the nodes from the json file
  for (var i = 0; i < json.actions.length; i++)
  {
    nodes[i] = {name: json.actions[i], id: i};

    if (i == json.initial) {
      nodes[i].initial = true;
      nodes[i].fixed   = true;
      nodes[i].x       = width/10;
      nodes[i].y       = height/2;
    }
  }

  // Assign the transitions from the json file
  for (var i = 0; i < json.transitions.length; i++)
  {
    for (var j = 0; j < json.observations.length; j++)
    {
      if (json.transitions[i][j] != null)
      {
        links.push({source: nodes[i].name,
                    target: nodes[json.transitions[i][j]].name,
                    obs:    json.observations[j]})
      }
    }
  }

  var nodeByName = d3.map(nodes, function(d) { return d.name; }),
      bilinks  = [];

  links.forEach(function(link) {
    var s = link.source = nodeByName.get(link.source),
        t = link.target = nodeByName.get(link.target),
        i = {}; // intermediate node
    nodes.push(i);
    links.push({source: s, target: i, obs: link.obs}, {source: i, target: t, obs: link.obs});
    bilinks.push({point: [s, i, t], obs: link.obs});
  });

  var force = d3.layout.force()
      .size([width, height])
      .linkDistance(height/3)
      .linkStrength(0.2)
      .friction(0.85)
      .charge(-1000);

  var drag = d3.behavior.drag()
               .on("drag", dragging)
               .on("dragend", dragend);

  // Links are just SVG lines, and we'll let the force layout
  // take care of their coordinates.
  var link = svg.selectAll('.link')
                .data(bilinks)
                .enter().append('path')
                .attr('class', function(d) { return 'link ' + d.obs; })
                .attr('marker-end', function(d) {return 'url(#arrowhead_'+d.obs+')';});

  // Now it's the nodes turn. Each node is drawn as a circle, with a label
  var gnode = svg.selectAll('g.gnode')
                 .data(nodes.filter(function(d) { return d.name; }))
                 .enter()
                 .append('g')
                 .attr('class','gnode')
                 .call(drag);

  var node = gnode.append('circle')
                  .attr('class', function(d) { if (d.initial) { return 'node initial'; } return 'node';})
                  .attr('r', function(d) { if (d.initial) { return 8; } return 4;});

  var label = gnode.append('text')
                    .attr("dx", function(d) { if (d.initial) { return -width/10+10; } return 12;})
                    .attr("dy", function(d) { if (d.initial) { return "-1em"; } return ".35em";})
                    .text(function(d) { return '['+d.id+'] '+d.name;});

  force.nodes(nodes)
       .links(links)
       .start();

  force.on('tick', tick);

  function tick()
  {
    link.attr('d', function(d)
      {
        if (d.point[0].x==d.point[2].x && d.point[0].y==d.point[2].y){
          return "M" + d.point[0].x + "," + d.point[0].y
               + "C" + d.point[1].x + "," + d.point[0].y
               + " " + d.point[0].x + "," + d.point[1].y
               + " " + d.point[2].x + "," + d.point[2].y;
        }

        return "M" + d.point[0].x + "," + d.point[0].y
             + "S" + d.point[1].x + "," + d.point[1].y
             + " " + d.point[2].x + "," + d.point[2].y;
      }
    )

    gnode.attr("transform", function(d)
    {
      return 'translate(' + [d.x, d.y] + ')';
    });
  }

  function dragging(d) {
    d.x += d3.event.dx;
    d.y += d3.event.dy;
    tick();
  }

  function dragend(d) {
    d.fixed = true;
    tick();
  }

});

function appendmarkers()
{
  // Arrowhead markers for the links (one for each color)
  svg.append("defs").append("marker")
      .attr("id", "arrowhead_none")
      .attr("refX", 6 + 3) /*must be smarter way to calculate shift*/
      .attr("refY", 2)
      .attr("markerWidth", 8)
      .attr("markerHeight", 4)
      .attr("orient", "auto")
      .style("fill", "#ccc")
      .append("path")
      .attr("d", "M 0,0 V 4 L6,2 Z"); //this is actual shape for arrowhead

  svg.append("defs").append("marker")
      .attr("id", "arrowhead_yes")
      .attr("refX", 6 + 3) /*must be smarter way to calculate shift*/
      .attr("refY", 2)
      .attr("markerWidth", 8)
      .attr("markerHeight", 4)
      .attr("orient", "auto")
      .style("fill", "#5CB85C")
      .append("path")
      .attr("d", "M 0,0 V 4 L6,2 Z"); //this is actual shape for arrowhead

  svg.append("defs").append("marker")
      .attr("id", "arrowhead_no")
      .attr("refX", 6 + 3) /*must be smarter way to calculate shift*/
      .attr("refY", 2)
      .attr("markerWidth", 8)
      .attr("markerHeight", 4)
      .attr("orient", "auto")
      .style("fill", "#D9534F")
      .append("path")
      .attr("d", "M 0,0 V 4 L6,2 Z"); //this is actual shape for arrowhead
};

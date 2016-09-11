
var defaultjsonfile = 'test.json';
var show_errors = true;
loadpolicy('');

function loadpolicy(file, errors)
{
  if (typeof(errors)==='undefined') { errors = show_errors; }
  else                              { show_errors = errors; }
  if (file == '') { file = defaultjsonfile;}
  else            { defaultjsonfile = file;};

  console.log('Loading file: '+file+' Show Errors: '+errors);

  var width  = 1110,
      height =  555;

  var svg = d3.select('svg')
              //responsive SVG needs these 2 attributes and no width and height attr
              .attr("preserveAspectRatio", "xMinYMin meet")
              .attr("viewBox", '0 0 ' + width + ' ' + height)
              //class to make it responsive
              .classed("svg-content-responsive", true);

  svg.call(d3.behavior.zoom().on('zoom', redraw));

  svg.append('text')
     .attr('dx', width/2)
     .attr('dy', height/15)
     .attr('class', 'title filename')
     .attr('text-anchor','middle')
     .text(file.replace('.json',''));

  var vis = svg.append('svg:g');

  appendmarkers();

  function redraw()
  {
    vis.attr('transform',
             'translate(' + d3.event.translate + ')'
              + ' scale(' + d3.event.scale + ')');
  }

  d3.json('json/'+file, function(error, json)
  {
    if (error) {throw error;}

    var nodes = Array();
    var links = Array();

    var max = d3.max(json.transitions, function(array) {
      return d3.max(array);
    });

    if (max > json.actions.length-1) {
      print_error('Max number of nodes does not match the transition matrix.');
      return;
    };
    // Assign the nodes from the json file
    for (var i = 0; i < json.actions.length; i++)
    {
      nodes[i] = {name: '['+i+'] '+ json.actions[i].replace('intention','int')
                                                   .replace('phy','P ')
                                                   .replace('com-','C ')
                                                   .replace('-get',' Get')
                                                   .replace('-snap',' Snap')
                                                   .replace('-left-leg','LL')
                                                   .replace('-right-leg','RL')
                                                   .replace('-central-frame','CF'), id: i};

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
          if (show_errors == false) {
            if (json.observations[j] != 'error') {
              links.push({source: nodes[i].name,
                          target: nodes[json.transitions[i][j]].name,
                          obs:    json.observations[j]})
            }
          }
          else
          {
            links.push({source: nodes[i].name,
                        target: nodes[json.transitions[i][j]].name,
                        obs:    json.observations[j]})
          }
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
        .linkStrength(0.2)
        .friction(0.85)
        .charge(-1000);

    var drag = force.drag()
                    .on('dragstart', dragstart)
                    .on('drag', dragging)
                    .on('dragend', dragend);

    // Links are just SVG lines, and we'll let the force layout
    // take care of their coordinates.
    var link = vis.selectAll('.link')
                  .data(bilinks)
                  .enter().append('path')
                  .attr('class', function(d) { return 'link ' + d.obs; })
                  .attr('marker-end', function(d) {return 'url(#arrowhead_'+d.obs+')';});

    // Now it's the nodes turn. Each node is drawn as a circle, with a label
    var gnode = vis.selectAll('g.gnode')
                   .data(nodes.filter(function(d) { return d.name; }))
                   .enter()
                   .append('g')
                   .attr('class','gnode')
                   .call(drag);

    var node = gnode.append('circle')
                    .attr('class', function(d) { if (d.initial) { return 'node initial'; } return 'node';})
                    .attr('r', function(d) { if (d.initial) { return 8; } return 4;});

    var label = gnode.append('text')
                      .attr('dx', function(d) { return 0;})
                      .attr('dy', function(d) { return '-0.9em';})
                      .attr('class', function(d) { if (d.initial) { return 'nodetext initial'; } return 'nodetext ';})
                      .attr('text-anchor','middle')
                      .text(function(d) { return d.name;});

    force.nodes(nodes)
         .links(links)
         .linkDistance(function(d) {
              return d.obs == 'error' ? height*2/3 : height/3;
            });

    force.on('tick', tick)
         .start();

    add_legend();

    function tick()
    {
      link.attr('d', function(d)
        {
          if (d.point[0].x==d.point[2].x && d.point[0].y==d.point[2].y){
            return 'M' + d.point[0].x + ',' + d.point[0].y
                 + 'C' + d.point[1].x + ',' + d.point[0].y
                 + ' ' + d.point[0].x + ',' + d.point[1].y
                 + ' ' + d.point[2].x + ',' + d.point[2].y;
          }

          return 'M' + d.point[0].x + ',' + d.point[0].y
               + 'S' + d.point[1].x + ',' + d.point[1].y
               + ' ' + d.point[2].x + ',' + d.point[2].y;
        }
      )

      gnode.attr('transform', function(d)
      {
        return 'translate(' + [d.x, d.y] + ')';
      });
    }

    function forcedelete() {
      nodes = [];
      links = [];
      bilinks = [];
      force.nodes(nodes);
      force.links(links);
    }

    function dragstart(d) {
      d3.event.sourceEvent.stopPropagation();
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

    function add_legend() {
      var rectSize = 18;
      var spacing  = 10;

      var legend = svg.selectAll('.legend')
                      .data(json.observations)
                      .enter()
                      .append('g')
                      .attr('class', 'legend')
                      .attr('transform', function(d, i) {
                        var horz = width * 13 /14;
                        var vert = height/10 + i * rectSize + i * spacing;
                        return 'translate(' + horz + ',' + vert + ')';
                      });

      legend.append('rect')
            .attr('width', rectSize)
            .attr('height', rectSize)
            .style('fill', function(d) {
              if (d == 'none')  { return '#3c3c3c';}
              if (d == 'yes')   { return '#5CB85C';}
              if (d == 'no')    { return '#3894F0';}
              if (d == 'error') { return '#D9534F';}
              return '#bbb';
            });

      legend.append('text')
            .attr('x', rectSize + spacing/2)
            .attr('y', rectSize - spacing/2)
            .text(function(d) { return d; });
      }

  });

  function print_error(text) {
      console.error(text);
      svg.append('text')
         .attr('dx', width/2)
         .attr('dy', height/2)
         .attr('class', 'message error')
         .attr('text-anchor','middle')
         .html('ERROR!');

      svg.append('text')
         .attr('dx', width/2)
         .attr('dy', height/2+60)
         .attr('class', 'message info')
         .attr('text-anchor','middle')
         .html(text);
  }

  function appendmarkers()
  {
    // Arrowhead markers for the links (one for each color)
    vis.append('defs').append('marker')
        .attr('id', 'arrowhead_none')
        .attr('refX', 6 + 1) /*must be smarter way to calculate shift*/
        .attr('refY', 2)
        .attr('markerWidth', 8)
        .attr('markerHeight', 4)
        .attr('orient', 'auto')
        .style('fill', '#3c3c3c')
        .append('path')
        .attr('d', 'M 0,0 V 4 L6,2 Z'); //this is actual shape for arrowhead

    vis.append('defs').append('marker')
        .attr('id', 'arrowhead_yes')
        .attr('refX', 6 + 1) /*must be smarter way to calculate shift*/
        .attr('refY', 2)
        .attr('markerWidth', 8)
        .attr('markerHeight', 4)
        .attr('orient', 'auto')
        .style('fill', '#5CB85C')
        .append('path')
        .attr('d', 'M 0,0 V 4 L6,2 Z'); //this is actual shape for arrowhead

    vis.append('defs').append('marker')
        .attr('id', 'arrowhead_no')
        .attr('refX', 6 + 1) /*must be smarter way to calculate shift*/
        .attr('refY', 2)
        .attr('markerWidth', 8)
        .attr('markerHeight', 4)
        .attr('orient', 'auto')
        .style('fill', '#3894F0')
        .append('path')
        .attr('d', 'M 0,0 V 4 L6,2 Z'); //this is actual shape for arrowhead

    vis.append('defs').append('marker')
        .attr('id', 'arrowhead_error')
        .attr('refX', 6 + 1) /*must be smarter way to calculate shift*/
        .attr('refY', 2)
        .attr('markerWidth', 8)
        .attr('markerHeight', 4)
        .attr('orient', 'auto')
        .style('fill', '#D9534F')
        .append('path')
        .attr('d', 'M 0,0 V 4 L6,2 Z'); //this is actual shape for arrowhead
  };

};

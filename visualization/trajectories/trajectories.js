var defaultjsonfile = 'trajectories.json';

loadtrajectory('');

function loadtrajectory(file)
{
  if (file == '') { file = defaultjsonfile;}
  else            { defaultjsonfile = file;};

  console.log('Loading file: '+file);

  var width  = 1110,
      height =  555;


  var i = 0,
    duration = 500,
    root;

  var tree = d3.layout.tree()
               .size([height, width]);

  var diagonal = d3.svg.diagonal().projection(function(d) { return [d.y, d.x]; });

  var svg = d3.select('svg')
              //responsive SVG needs these 2 attributes and no width and height attr
              .attr('preserveAspectRatio', 'xMinYMin meet')
              .attr('viewBox', '0 0 ' + width + ' ' + height)
              //class to make it responsive
              .classed('svg-content-responsive', true);

  svg.call(d3.behavior.zoom().on('zoom', redraw));

  svg.append('text')
     .attr('dx', width/2)
     .attr('dy', height/15)
     .attr('class', 'title filename')
     .attr('text-anchor','middle')
     .text(file.replace('.json',''));

  var vis = svg.append('svg:g');

  appendmarkers();

  d3.json('json/'+file, function(error, json)
  {
    if (error) {throw error;}

    root = json.graphs[0];
    root.x0 = height / 2;
    root.y0 = 300;
    root.initial = true;

    function collapse(d) {
        if (d.children) {
          d._children = d.children;
          d._children.forEach(collapse);
          d.children = null;
        }
      }

    // root.children.forEach(collapse);
    update(root);
  });

  function update(source) {
    // Compute the new tree layout.
    var nodes = tree.nodes(root).reverse(),
        links = tree.links(nodes);

    // Normalize for fixed-depth.
    nodes.forEach(function(d) { d.y = d.depth * 160; });

    // Update the nodes…
    var node = vis.selectAll('g.node')
        .data(nodes, function(d) { return d.id || (d.id = ++i); });

    // Enter any new nodes at the parent's previous position.
    var nodeEnter = node.enter().append('g')
        .attr('class', 'node')
        .attr('transform', function(d) { return 'translate(' + source.y0 + ',' + source.x0 + ')'; })
        .on('click', click);

    nodeEnter.append('circle')
        .attr('r', 1e-6)
        .attr('class', function(d) { if (d.initial) { return 'nodecircle initial'; } return 'node';});

    nodeEnter.append('text')
        .attr('dx', function(d) { return 0;})
        .attr('dy', function(d) { return '-0.8em';})
        .attr('text-anchor','middle')
        .attr('class', function(d) { if (d.initial) { return 'nodetext initial'; } return 'nodetext ';})
        .text(function(d) { return '['+d.node+'] '+ d.action.replace('intention','int')
                                                       .replace('phy','P ')
                                                       .replace('com-','C ')
                                                       .replace('-get',' Get')
                                                       .replace('-snap',' Snap')
                                                       .replace('-left-leg','LL')
                                                       .replace('-right-leg','RL')
                                                       .replace('-central-frame','CF'); });

    // Transition nodes to their new position.
    var nodeUpdate = node.transition()
        .duration(duration)
        .attr('transform', function(d) { return 'translate(' + d.y + ',' + d.x + ')'; });

    nodeUpdate.select('circle')
        .attr('r', 4.5)
        .attr('class', function(d) { if (d.initial) { return 'nodecircle initial'; } return 'node';});

    nodeUpdate.select('text')
        .style('fill-opacity', 1);

    // Transition exiting nodes to the parent's new position.
    var nodeExit = node.exit().transition()
        .duration(duration)
        .attr('transform', function(d) { return 'translate(' + source.y + ',' + source.x + ')'; })
        .remove();

    nodeExit.select('circle')
        .attr('r', 1e-6);

    nodeExit.select('text')
        .style('fill-opacity', 1e-6);

    // Update the links…
    var link = vis.selectAll('path.link')
        .data(links, function(d) { return d.target.id; });

    // Enter any new links at the parent's previous position.
    link.enter().insert('path', 'g')
        .attr('class', function(d,i) {
          console.log(d.source.observations, i, d.source.observations[i]);
          return 'link ' + d.source.observations[i]; })
        .attr('d', function(d) {
          var o = {x: source.x0, y: source.y0};

          return diagonal({source: o, target: o});
        })
        .attr('marker-end', function(d,i) { return 'url(#arrowhead_'+d.source.observations[i]+')';});

    // Transition links to their new position.
    link.transition()
        .duration(duration)
        .attr('d', diagonal);

    // Transition exiting nodes to the parent's new position.
    link.exit().transition()
        .duration(duration)
        .attr('d', function(d) {
          var o = {x: source.x, y: source.y};
          return diagonal({source: o, target: o});
        })
        .remove();

    // Stash the old positions for transition.
    nodes.forEach(function(d) {
      d.x0 = d.x;
      d.y0 = d.y;
    });
  };

  // Toggle children on click.
  function click(d) {
    if (d.children) {
      d._children = d.children;
      d.children = null;
    } else {
      d.children = d._children;
      d._children = null;
    }
    update(d);
  }

  function redraw()
  {
    vis.attr('transform',
             'translate(' + d3.event.translate + ')'
              + ' scale(' + d3.event.scale + ')');
  };

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


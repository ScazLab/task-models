defaultjsonfile = 'pomcp.json';
var defaultColor = '#3c3c3c';


loadpomcp('');

function loadpomcp(file)
{
  if (file == '') { file = defaultjsonfile;}
  else            { defaultjsonfile = file;};

  console.log('Loading file: '+file);

  var width  = 1110,
      height =  400;

  var node_count = 0,
      duration = 500,
      root,
      actions,
      states;

  var tree = d3.tree()
               .size([height / 2, width / 2]);

  var beliefs = d3.select('#belief-container');
  var values  = d3.select('#value-container');

  var svg = d3.select('#tree-container>svg')
              //responsive SVG needs these 2 attributes and no width and height attr
              .attr('preserveAspectRatio', 'xMinYMin meet')
              .attr('viewBox', '0 0 ' + width + ' ' + height)
              //class to make it responsive
              .classed('svg-content-responsive', true);

  svg.call(d3.zoom().on('zoom', redraw));

  // Graph title
  svg.append('text')
     .attr('dx', width/2)
     .attr('dy', height/15)
     .attr('class', 'title filename')
     .attr('text-anchor','middle')
     .text(file.replace('.json',''));

  var vis = svg.append('svg:g');
  var g_links = vis.append('g').attr('class', 'links');
  var g_nodes = vis.append('g').attr('class', 'nodes');

  appendmarkers();

  d3.json('json/'+file, function(error, json)
  {
    if (error) {throw error;}

    actions = json.actions;
    states = json.states;

    // Init the belief representation
    beliefs.select("tr.states").selectAll('th:not(.legend)').data(states).enter().append('th')
          .attr("class", "state")
          .text(function(d) { return d; });
    beliefs.select("tr.belief").selectAll('td').data(states).enter().append('td')
          .attr("title", function(d) { return d; });
        
    // Init the value representation
    values.select("tr.actions").selectAll('th:not(.legend)').data(actions).enter().append('th')
          .attr("class", "action")
          .text(function(d) { return shorten_action(d); });
    ["values", "visits"].forEach(function(tr_class) {
      values.select("tr." + tr_class).selectAll('td').data(actions).enter()
        .append('td')
            .attr("title", function(d) { return d; });
    })

    root = json.graphs[0];
    root.initial = true;

    function collapse(d) {
        if (d.children) {
          d._children = d.children;
          d._children.forEach(collapse);
          d.children = null;
        }
      }

    root.children.forEach(collapse);
    update();
  });

  function shorten_action(a) {
    return a.replace('intention','int')
            .replace('phy','P-')
            .replace('com-','C-')
            .replace('-get',' Get')
            .replace('-snap',' Snap')
            .replace('-left-leg','LL')
            .replace('-right-leg','RL')
            .replace('-central-frame','CF');
  }

  function max_abs(arr) {
    return arr.reduce(function(max, x) {
        return (x == null) ? max : (Math.abs(x) > max) ? Math.abs(x) : max;
    }, 1.);
  }

  function nodeTranslate(d) {
      return 'translate(' + d.y + ',' + d.x + ')';
  }

  function linkDiagonal(d) {
    return "M" + d.source.y + "," + d.source.x
      + "C" + (d.source.y + d.target.y) / 2 + "," + d.source.x
      + " " + (d.source.y + d.target.y) / 2 + "," + d.target.x
      + " " + d.target.y + "," + d.target.x;
  }

  function update(source) {
    // Compute the new tree layout.
    var treeRoot = tree(d3.hierarchy(root));
    var nodes = treeRoot.descendants(),
        links = treeRoot.links();

    source = source || treeRoot;

    var sourceTranslate = nodeTranslate(source);
    var linkInit = "M" + source.y + "," + source.x
        + "C" + source.y + "," + source.x
        + " " + source.y + "," + source.x
        + " " + source.y + "," + source.x;


    // Normalize for fixed-depth.
    nodes.forEach(function(d) { d.y = d.depth * 160; });
    //  DEBUG

    // Update the nodes…
    var node = g_nodes.selectAll('g.node')
        .data(nodes, function(d) { return (d.data.id != null) ? d.data.id : (d.data.id = node_count++); });

    // Enter any new nodes at the source position.
    var nodeEnter = node.enter().append('g')
          .attr('class', 'node')
          .attr('transform', sourceTranslate)
          .on('click', toggleExpand)
          .on('mouseover', displayValues);

    nodeEnter.append('svg:title')
          .text(function(d) {
            return "Visits: " + d.data.visits + "\nValue: " + d.data.value;
          });

    function node_class(d) {
        return 'nodecircle' + ((d.initial) ? ' initial': '');
    }

    nodeEnter.append('circle')
        .attr('r', 1e-6)
        .attr('class', node_class)
      .transition()
        .duration(duration)
        .attr('r', 4.5);

    nodeEnter.append('text')
        .attr('dx', function(d) { return 0;})
        .attr('dy', function(d) { return '-0.8em';})
        .attr('text-anchor', 'middle')
        .attr('class', function(d) { if (d.initial) { return 'nodetext initial'; } return 'nodetext ';})
        .text(function(d) {
          return ((d.node) ? '['+d.node+'] ' : '') + shorten_action(d.data.action);
        });

    // Transition
    nodeEnter.transition()
          .duration(duration)
          .attr('transform', nodeTranslate);

    // Transition nodes to their new position on update.
    node.transition()
        .duration(duration)
        .attr('transform', nodeTranslate);

    node.select('circle')
        .attr('r', 4.5)
        .attr('class', node_class);

    node.select('text')
        .style('fill-opacity', 1);

    // Transition exiting nodes to the parent's new position.
    var nodeExit = node.exit().transition()
        .duration(duration)
        .attr('transform', sourceTranslate)
        .remove();

    nodeExit.select('circle')
        .attr('r', 1e-6);

    nodeExit.select('text')
        .style('fill-opacity', 1e-6);

    // Update the links…
    var link = g_links.selectAll('path.link')
        .data(links, function(d) { return d.target.data.id; });

    // Enter any new links at the parent's previous position.
    link.enter().append('path')
        .attr('class', function(d) {
          return 'link ' + d.source.data.observations[d.target.data.observed]; })
        .attr("d", linkInit)
        .attr('marker-end', function(d) {
          return 'url(#arrowhead_'+d.source.data.observations[d.target.data.observed]+')';
        })
      .transition()
        .duration(duration)
        .attr('d', linkDiagonal);

    // Transition links to their new position.
    link.transition()
        .duration(duration)
        .attr('d', linkDiagonal);


    // Transition exiting nodes to the parent's new position.
    link.exit().transition()
        .duration(duration)
         .attr('d', linkInit)
        .remove();

    // Stash the old positions for transition.
    nodes.forEach(function(d) {
      d.x0 = d.x;
      d.y0 = d.y;
    });
  };

  // Toggle children on click.
  function toggleExpand(d) {
    var data = d.data;
    if (data.children) {
      data._children = data.children;
      data.children = null;
    } else {
      data.children = data._children;
      data._children = null;
    }
    update(d);
  }

  function redraw()
  {
    var transform = d3.event.transform;
    vis.attr('transform',
             'translate(' + transform.x + ", " + transform.y + ')'
              + ' scale(' + transform.k + ')');
  };

  function appendmarker(suffix, color) {
    vis.append('defs').append('marker')
        .attr('id', 'arrowhead_' + suffix)
        .attr('refX', 6 + 1) /*must be smarter way to calculate shift*/
        .attr('refY', 2)
        .attr('markerWidth', 8)
        .attr('markerHeight', 4)
        .attr('orient', 'auto')
        .style('fill', color)
        .append('path')
        .attr('d', 'M 0,0 V 4 L6,2 Z'); //this is actual shape for arrowhead
  }

  function appendmarkers()
  {
    // Arrowhead markers for the links (one for each color)
    appendmarker('none', '#3c3c3c')
    appendmarker('yes', '#5CB85C')
    appendmarker('no', '#3894F0')
    appendmarker('error', '#D9534F')
  };

  function displayValues(d)
  {
    // Display belief
    beliefs.select("tr.belief").selectAll('td').data(d.data.belief)
        .style("background", function(b) { return d3.interpolateBlues(b); });
    // Display values and visits
    var max_abs_val = d.data.values.reduce(function(max, x) {
        return (x == null) ? max : (Math.abs(x) > max) ? Math.abs(x) : max;
    }, 1.);
    var best = d.data.values.reduce(function(b, x) {
        return (x == null) ? b : Math.max(b, x);
    }, -max_abs_val);
    values.select("tr.values").selectAll('td').data(d.data.values)
        .style("background", function(v) {
          return (v == null) ? '#edeeef' : d3.interpolateRdBu(0.5 * (1 - v / max_abs_val));
        })
        .attr("title", function(v) {return v;})
        .attr("class", function (v) {
          return "value" + ((v == best) ? " best" : "");
        });
    values.select("tr.actions").selectAll('th:not(.legend)').data(actions)
        .attr("class", function (a) {return "action" + ((a == d.data.action) ? " best" : "");});
    var max_visits = 1. * Math.max(...d.data.child_visits);
    values.select("tr.visits").selectAll('td').data(d.data.child_visits)
        .style("background", function(n) {
          return (n == 0) ? '#edeeef' : d3.interpolateReds(n / max_visits);
        })
        .attr("class", function (n) {
          return ("visit" + ((n == max_visits) ? " best" : "")
                          + ((n > 0.5 * max_visits) ? " big-value" : ""));
        })
        .text(function(v) { return (v > 0) ? v : ''; })
        .attr('title', function(v) { return v; });
  };
};

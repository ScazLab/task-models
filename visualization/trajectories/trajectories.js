var defaultjsonfile = 'trajectories.json';

loadpolicy('');

function loadpolicy(file)
{
  if (file == '') { file = defaultjsonfile;}
  else            { defaultjsonfile = file;};

  console.log('Loading file: '+file);

  var width  = 1110,
      height =  555;


  var i = 0,
    duration = 750,
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

    function collapse(d) {
        if (d.children) {
          d._children = d.children;
          d._children.forEach(collapse);
          d.children = null;
        }
      }

    root.children.forEach(collapse);
    update(root);
  });

  function update(tr) {
    // Compute the new tree layout.
    var nodes = tree.nodes(root).reverse(),
        links = tree.links(nodes);

    // Normalize for fixed-depth.
    nodes.forEach(function(d) { d.y = d.depth * 180; });

    // Update the nodes…
    var node = svg.selectAll('g.node')
        .data(nodes, function(d) { return d.id || (d.id = ++i); });

    // Enter any new nodes at the parent's previous position.
    var nodeEnter = node.enter().append('g')
        .attr('class', 'node')
        .attr('transform', function(d) { return 'translate(' + tr.y0 + ',' + tr.x0 + ')'; })
        .on('click', click);

    nodeEnter.append('circle')
        .attr('r', 1e-6)
        .style('fill', function(d) { return d._children ? 'lightsteelblue' : '#fff'; });

    nodeEnter.append('text')
        .attr('x', function(d) { return d.children || d._children ? -10 : 10; })
        .attr('dy', '.35em')
        .attr('text-anchor', function(d) { return d.children || d._children ? 'end' : 'start'; })
        .text(function(d) { return d.action.replace('intention','int')
                                           .replace('phy','P ')
                                           .replace('com-','C ')
                                           .replace('-get',' Get')
                                           .replace('-snap',' Snap')
                                           .replace('-left-leg','LL')
                                           .replace('-right-leg','RL')
                                           .replace('-central-frame','CF'); })
        .style('fill-opacity', 1e-6);

    // Transition nodes to their new position.
    var nodeUpdate = node.transition()
        .duration(duration)
        .attr('transform', function(d) { return 'translate(' + d.y + ',' + d.x + ')'; });

    nodeUpdate.select('circle')
        .attr('r', 4.5)
        .style('fill', function(d) { return d._children ? 'lightsteelblue' : '#fff'; });

    nodeUpdate.select('text')
        .style('fill-opacity', 1);

    // Transition exiting nodes to the parent's new position.
    var nodeExit = node.exit().transition()
        .duration(duration)
        .attr('transform', function(d) { return 'translate(' + tr.y + ',' + tr.x + ')'; })
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
        .attr('class', 'link')
        .attr('d', function(d) {
          var o = {x: tr.x0, y: tr.y0};
          console.log('test');
          return diagonal({source: o, target: o});
        });

    // Transition links to their new position.
    link.transition()
        .duration(duration)
        .attr('d', diagonal);

    // Transition exiting nodes to the parent's new position.
    link.exit().transition()
        .duration(duration)
        .attr('d', function(d) {
          var o = {x: tr.x, y: tr.y};
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
// // ************** Generate the tree diagram  *****************
// var jsonfile = 'https://raw.githubusercontent.com/ScazLab/hrc-papers/master/javascript/htm/json/icra.json?token=AELQJ1B-IJ-logFrv0jRWeK5V7IHjo-9ks5X0sScwA%3D%3D';
// // var jsonfile = 'flare.json';

// var margin = {top: 40, right: 120, bottom: 20, left: 120},
//     outerwidth = 1560,
//     outerheight = 700,
//     width  = outerwidth - margin.right - margin.left,
//     height = outerheight - margin.top - margin.bottom;

// var i = 0,
//     duration = 500,
//     rectW = 140,
//     rectH = 40;

// var tree = d3.layout.tree()
//              .size([height, width])
//              .nodeSize([rectW+20, rectH+20]);

// var diagonal = d3.svg.diagonal()
//                  .projection(function(d) { return [d.x+rectW/2, d.y+rectH/2]; });

// var svg = d3.select('#tree-container').append('svg')
//             .attr('width', width + margin.right + margin.left)
//             .attr('height', height + margin.top + margin.bottom)
//             .call(zm = d3.behavior.zoom().scaleExtent([1,3]).on('zoom', redraw)).append('g')
//             .attr('transform', 'translate(' + outerwidth/2 + ',' + 20 + ')');

// //necessary so that zoom knows where to zoom and unzoom from
// zm.translate([outerwidth/2, 20]);

// function collapse(d) {
//     if (d.children) {
//         d._children = d.children;
//         d._children.forEach(collapse);
//         d.children = null;
//     }
// }

// // load the external data
// d3.json(jsonfile, function(error, treeData) {
//   root = treeData.nodes;
//   root.x0 = width/2;
//   root.y0 = height/2;
//   root.children.forEach(collapse);
//   update(root);
// });

// function update(tr) {
//   // console.log(root);

//   // Compute the new tree layout.
//   var nodes = tree.nodes(root).reverse(),
//       links = tree.links(nodes);

//   // Normalize for fixed-depth.
//   nodes.forEach(function(d) { d.y = d.depth * 140; });

//   // Declare the nodes...
//   var node = svg.selectAll('g.node')
//                 .data(nodes, function(d) { return d.id; }); // { return d.id || (d.id = ++i); });

//   // Enter the nodes.
//   var nodeEnter = node.enter()
//                       .append('g')
//                       .attr('class', function(d) {
//                         var res='node';
//                         if (d.attributes) {res=res+' '+d.attributes.join(' ');}
//                         if (d._children)  {res=res+' collapsed';}
//                         return res;
//                       })
//                       .attr('transform', function(d) { return 'translate(' + tr.x0 + ',' + tr.y0 + ')'; })
//                       .on('click', click);

//   nodeEnter.append('rect')
//            .attr('width', rectW)
//            .attr('height', rectH)
//            .attr('class', 'label');

//   nodeEnter.append('text')
//            .attr('x', rectW / 2)
//            .attr('y', rectH / 2)
//            .attr('dy', '.35em')
//            .attr('text-anchor', 'middle')
//            .text(function (d) { return d.name; });

//   // Add combination if there is a combination and the node is not collapsed
//   nodeCombination = nodeEnter.filter(function(d,i){ return d.combination; }) // && !d._children && d.children; })
//                              .append('g')
//                              .attr('class','combination');

//   nodeCombination.append('rect')
//                  .attr('width', 36)
//                  .attr('height', 36)
//                  .attr('x', (rectW-36)/2)
//                  .attr('y', rectH + 1);

//   nodeCombination.append('text')
//                  .attr('x', rectW / 2)
//                  .attr('y', rectH / 2 - 12)
//                  .attr('dy', '3.5em')
//                  .attr('text-anchor', 'middle')
//                  .text(function (d) {
//                     if (d.combination=='Parallel') {return '||';}
//                     if (d.combination=='Sequence') {return '→';}
//                     if (d.combination=='Alternative') {return 'v';}
//                     return ''
//                   });

//   // Transition nodes to their new position.
//   var nodeUpdate = node.transition()
//                        .duration(duration)
//                        .attr('transform', function (d) {return 'translate(' + d.x + ',' + d.y + ')';});

//   var gUpdate = nodeUpdate.attr('class', function(d) {
//                             var cl=d3.select(this).attr('class');
//                             // console.log(cl,d);
//                             if (d._children) { if (cl.indexOf(' collapsed')==-1) { return cl+' collapsed'; } }
//                             else { if (cl.indexOf(' collapsed')!=-1) return cl.replace(' collapsed',''); }
//                             return cl;
//                           });


//   // Transition exiting nodes to the parent's new position.
//   var nodeExit = node.exit().transition()
//                             .duration(duration)
//                             .attr('transform', function (d) {return 'translate(' + tr.x + ',' + tr.y + ')';})
//                             .remove();

//   // Declare the links...
//   var link = svg.selectAll('path.link')
//                 .data(links, function(d) { return d.target.id; });

//   // console.log(link);
//   // Enter any new links at the parent's previous position.
//   link.enter().insert('path', 'g')
//       .attr('class', 'link')
//       .attr('x', rectW / 2)
//       .attr('y', rectH / 2)
//       .attr('d', function (d) {
//         var o = {
//             x: tr.x0,
//             y: tr.y0
//         };
//         return diagonal({source: o, target: o});
//       });

//   // Transition links to their new position.
//   link.transition()
//       .duration(duration)
//       .attr('d', diagonal);

//   // Transition exiting nodes to the parent's new position.
//   link.exit().transition()
//       .duration(duration)
//       .attr('d', function (d) {
//         var o = {
//             x: tr.x,
//             y: tr.y
//         };
//         return diagonal({source: o, target: o});
//       })
//       .remove();

//   // Stash the old positions for transition.
//   nodes.forEach(function (d) {
//       d.x0 = d.x;
//       d.y0 = d.y;
//   });

// }

// //Redraw for zoom
// function redraw() {
//   //console.log('here', d3.event.translate, d3.event.scale);
//   svg.attr('transform',
//       'translate(' + d3.event.translate + ')'
//       + ' scale(' + d3.event.scale + ')');
// }

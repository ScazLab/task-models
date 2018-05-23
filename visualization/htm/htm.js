// Initialize global variables (I know , it's bad)
var defaultjsonfile = 'icra.json',
    treedepth = 3,
    defaultjsondata = '';

var width  = 1500,
    height =  600;

var i = 0,
    time  = 500,
    rectW = 140,
    rectH =  40;

// Legend-related variables
var legClass  = d3.scale.ordinal().domain(['human', 'robot', 'highlighted',
                                           'collapsed', 'subtask']);
var legRSize  = 24,
    legRSpace = 10;

var origtranslate = [(width-rectW)/2, 100]

var tree = d3.layout.tree()
             .nodeSize([rectW+20, rectH])
             .separation(function separation(a, b) {
                return (a.parent == b.parent ? 1 : 1.4);
              });

var diagonal = d3.svg.diagonal()
                 .projection(function(d) { return [d.x+rectW/2, d.y+rectH/2]; });

var svg = d3.select('svg')
            //responsive SVG needs these 2 attributes and no width and height attr
            .attr('preserveAspectRatio', 'xMidYMid meet')
            .attr('viewBox', '0 0 ' + width + ' ' + height)
            //class to make it responsive
            .classed('svg-content-responsive', true);

zoombehavior = d3.behavior.zoom()
                 .translate([0, 0])
                 .scale(1)
                 .scaleExtent([0.2, 5])
                 .on('zoom', zoomed);

// Title
// svg.append('text')
//    .attr('dx', width/2)
//    .attr('dy', height/15)
//    .attr('class', 'title filename')
//    .attr('text-anchor','middle')
//    .text(file.replace('.json','').replace('_',' '));

// Legend
var legendcnt = svg.append('g')
                   .attr('class','legendcnt')
                   .attr('transform','translate(20,20)')

var legend = legendcnt.selectAll('.legend')
                      .data(legClass.domain())
                      .enter().append('g')
                      .attr('class',     function(d)    { return 'legend ' + d; })
                      .attr('transform', function(d, i) {
                        return 'translate(0,' + i * (legRSize + legRSpace) + ')';
                      });

legend.append('rect')
      .attr( 'width', legRSize)
      .attr('height', legRSize)
      .attr( 'class',  'label');

legend.append('text')
      .attr('x', legRSize + legRSpace/2)
      .attr('y', legRSize - legRSpace/2)
      .text(function(d) { return d; });

// HTM visualization
var vis  = svg.append('svg:g').attr('class', 'vis');

var draw = vis.append('svg:g').attr('class', 'draw')
              .attr('transform', 'translate(' + origtranslate[0] + ',' +
                                                origtranslate[1] + ')');

function loadhtm(file) {

  if (file == '') { file = defaultjsonfile;}
  else            { defaultjsonfile = file;};

  file = file.replace('C:\\fakepath\\', '');

  // If the file is still empty, it means that there is nothing to display
  // because also defaultjsonfile is empty
  if (file == '' && defaultjsondata == '') {
    console.log('No file to load. Returning.');
    return;
  }

  svg.call(zoombehavior);

  if (defaultjsondata == '') {
    console.log('Loading file: '+file+' with depth '+treedepth);

    // load the external data
    d3.json('json/'+file, function(error, json)
    {
      if (error) {throw error;}

      root = json.nodes;
      root.x0 = 0;
      root.y0 = 0;

      var nodes = tree.nodes(root).reverse(),
          links = tree.links(nodes);

      root.children.forEach(collapseLevel);
      update(root);

    });
  }
  else {
    console.log('Loading from'+defaultjsondata+' with depth '+treedepth);

    root = JSON.parse(defaultjsondata).nodes;
    root.x0 = 0;
    root.y0 = 0;

    var nodes = tree.nodes(root).reverse(),
        links = tree.links(nodes);

    if (root.children) { root.children.forEach(collapseLevel); }
    update(root);
  };

  function update(source) {
    // console.log(root);

    // Compute the new tree layout.
    var nodes = tree.nodes(root).reverse(),
        links = tree.links(nodes);

    // Normalize for fixed-depth.
    nodes.forEach(function(d) { d.y = d.depth * 100; });

    // Cleanup names
    nodes.forEach(function(d) { d.name = d.name.replace('REQUEST-ACTION',' ')
                                               .replace('-OF-OBJECT',' ')
                                               .replace('ARTIFACT-',' ')
                                               .replace(' ARTIFACT',' ')
                                               .replace('Parallelized Subtasks of ',' ')
                                               .replace('FOOT_BRACKET','bracket-foot'); });

    // Declare the nodes.
    var node = draw.selectAll('g.node')
                   .data(nodes, function(d) { return d.id; });
                                         // { return d.id || (d.id = ++i); });

    // Enter the nodes.
    var nodeLabel = node.enter().append('g')
                        .attr('class', function(d) {
                          var res='node';
                          if (d.attributes) {res=res+' '+d.attributes.join(' ');}
                          if (d._children)  {res=res+' collapsed';}
                          return res;
                        })
                        .attr('transform', function(d) {
                          return 'translate(' + source.x0 + ',' + source.y0 + ')';
                        })
                        .on('click', click);

    var nodeRect = nodeLabel.append('rect')
                            .attr( 'width',   rectW)
                            .attr('height',   rectH)
                            .attr( 'class', 'label');

    var nodeText = nodeLabel.append('text')
                            .attr('x', rectW / 2)
                            .attr('y', rectH / 2)
                            .attr('dy', '.35em')
                            .attr('text-anchor', 'middle')
                            .text(function (d) { return d.name; });

    nodeRect.attr('width',  function(d) {
              d.rectWidth = this.nextSibling.getComputedTextLength() + 20;
              return d.rectWidth;
            })
            .attr('x',  function(d) {
              return (rectW - d.rectWidth)/2;
            })

    nodeText.attr('x', function(d) { return (rectW)/2; })

    // Add combination if there is a combination and the node is not
    // a terminal node (i.e. it does have children)
    nodeComb = nodeLabel.filter(function(d){ return d.combination && d.children; })
                        .append('g')
                        .attr('class','combination');

    nodeComb.append('rect')
            .attr('width', 36)
            .attr('height', 36)
            .attr('x', function(d) {return (rectW-36)/2})
            .attr('y', rectH + 1);

    nodeComb.append('text')
            .attr('x', function(d) {return (rectW)/2})
            .attr('y', rectH / 2 - 12)
            .attr('dy', '2.2em')
            .attr('text-anchor', 'middle')
            .text(function (d) {
              if (d.combination==   'Parallel') {return '||';}
              if (d.combination== 'Sequential') {return  '→';}
              if (d.combination=='Alternative') {return  'v';}
              return ''
            });

    // Transition nodes to their new position.
    node.call( function setupupdate(sel) { upcnt = sel.size(); })
        .transition().duration(time)
        .attr('transform', function(d) { return 'translate(' + d.x + ',' + d.y + ')'; })
        .attr(    'class', function(d) {
            var cl=d3.select(this).attr('class');
            if (d._children) {
                if (cl.indexOf(' collapsed')==-1) { return cl+' collapsed';}
            }
            else {
                if (cl.indexOf(' collapsed')!=-1) {
                    return cl.replace(' collapsed','');
                }
            }
            return cl;
        })
        .each('end', onUpdate);


    // Transition exiting nodes to the parent's new position.
    node.exit().call( function setupremove(sel) { remcnt = sel.size(); })
               .transition().duration(time)
               .attr('transform', function (d) {
                   return 'translate(' + source.x + ',' + source.y + ')';
               })
               .remove();

    // Declare the links
    var link = draw.selectAll('path.link')
                   .data(links, function(d) { return d.target.id; });

    // Enter any new links at the parent's previous position.
    link.enter().insert('path', 'g')
        .attr('class', 'link')
        .attr('x', rectW / 2)
        .attr('y', rectH / 2)
        .attr('d', function (d) {
          var o = {
              x: source.x0,
              y: source.y0
          };
          return diagonal({source: o, target: o});
        });

    // Transition links to their new position.
    link.transition()
        .duration(time)
        .attr('d', diagonal);

    // Transition exiting nodes to the parent's new position.
    link.exit().transition()
        .duration(time)
        .attr('d', function (d) {
          var o = {
              x: source.x,
              y: source.y
          };
          return diagonal({source: o, target: o});
        })
        .remove();

    // Stash the old positions for transition.
    nodes.forEach(function (d) {
        d.x0 = d.x;
        d.y0 = d.y;
    });

  };

  function onUpdate() {
    // console.log('up', upcnt)
    upcnt--;
    if(upcnt == 0) {
      scaletofit();
    }
  }

  function onRemove() {
    // console.log('rem', remcnt)
    remcnt--;
    if(remcnt == 0) {
      scaletofit();
    }
  }

  function collapseLevel(d) {
      // console.log(d.name, d.depth);
      if (d.children && d.depth >= treedepth) {
          d._children = d.children;
          d._children.forEach(collapseLevel);
          d.children = null;
      } else if (d.children) {
          d.children.forEach(collapseLevel);
      }
  };

  function scaletofit() {
    console.log('Scaling to fit')

    dr = document.getElementsByClassName('draw');

    var bounds = dr[0].getBBox();

    var w = bounds.width,
        h = bounds.height;
    var mX = bounds.x + w / 2,
        mY = bounds.y + h / 2;
    if (w == 0 || h == 0) return; // nothing to fit
    var scale = 0.95 / Math.max(w / width, h / (height-100));
    var translate = [width / 2 - scale * (mX + origtranslate[0]),
                    height / 2 - scale * (mY + origtranslate[1])];

    draw.transition().duration(time)
        .call(zoombehavior.translate(translate).scale(scale).event);
  };

  // Function to handle mouse click events
  function click(d) {
    console.log('Pressed item: '+d.name+'\tdepth: '+d.depth+'\tattr: '+d.attributes);
    // console.log(tree.links(d).toString());

    if (d.children) {
      d._children = d.children;
      d.children = null;
    } else {
      d.children = d._children;
      d._children = null;
    }
    update(d);
  };

};

// zoomed for zoom
function zoomed() {
  // console.log('d3 event. Translate: '+d3.event.translate+'\tScale: '+d3.event.scale);
  vis.attr('transform',
           'translate(' + d3.event.translate + ')'
            + ' scale(' + d3.event.scale     + ')');
};

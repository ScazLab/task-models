// ************** Generate the tree diagram  *****************
var jsonfile = "https://raw.githubusercontent.com/ScazLab/hrc-papers/master/javascript/htm/json/icra.json?token=AELQJ1B-IJ-logFrv0jRWeK5V7IHjo-9ks5X0sScwA%3D%3D";
// var jsonfile = "flare.json";

var margin = {top: 40, right: 120, bottom: 20, left: 120},
    outerwidth = 1560,
    outerheight = 700,
    width  = outerwidth - margin.right - margin.left,
    height = outerheight - margin.top - margin.bottom;

var i = 0,
    duration = 500,
    rectW = 140,
    rectH = 40;

var tree = d3.layout.tree()
             .size([height, width])
             .nodeSize([rectW+20, rectH+20]);

var diagonal = d3.svg.diagonal()
                 .projection(function(d) { return [d.x+rectW/2, d.y+rectH/2]; });

var svg = d3.select("#tree-container").append("svg")
            .attr("width", width + margin.right + margin.left)
            .attr("height", height + margin.top + margin.bottom)
            .call(zm = d3.behavior.zoom().scaleExtent([1,3]).on("zoom", redraw)).append("g")
            .attr("transform", "translate(" + outerwidth/2 + "," + 20 + ")");

//necessary so that zoom knows where to zoom and unzoom from
zm.translate([outerwidth/2, 20]);

function collapse(d) {
    if (d.children) {
        d._children = d.children;
        d._children.forEach(collapse);
        d.children = null;
    }
}

// load the external data
d3.json(jsonfile, function(error, treeData) {
  root = treeData.nodes;
  root.x0 = width/2;
  root.y0 = height/2;
  root.children.forEach(collapse);
  update(root);
});

function update(source) {
  // console.log(root);

  // Compute the new tree layout.
  var nodes = tree.nodes(root).reverse(),
      links = tree.links(nodes);

  // Normalize for fixed-depth.
  nodes.forEach(function(d) { d.y = d.depth * 140; });

  // Declare the nodes...
  var node = svg.selectAll("g.node")
                .data(nodes, function(d) { return d.id; }); // { return d.id || (d.id = ++i); });

  // Enter the nodes.
  var nodeEnter = node.enter()
                      .append("g")
                      .attr("class", function(d) {
                        var res="node";
                        if (d.attributes) {res=res+" "+d.attributes.join(" ");}
                        if (d._children)  {res=res+" collapsed";}
                        return res;
                      })
                      .attr("transform", function(d) { return "translate(" + source.x0 + "," + source.y0 + ")"; })
                      .on("click", click);

  nodeEnter.append("rect")
           .attr("width", rectW)
           .attr("height", rectH)
           .attr("class", "label");

  nodeEnter.append("text")
           .attr("x", rectW / 2)
           .attr("y", rectH / 2)
           .attr("dy", ".35em")
           .attr("text-anchor", "middle")
           .text(function (d) { return d.name; });

  // Add combination if there is a combination and the node is not collapsed
  nodeCombination = nodeEnter.filter(function(d,i){ return d.combination; }) // && !d._children && d.children; })
                             .append("g")
                             .attr("class","combination");

  nodeCombination.append("rect")
                 .attr("width", 36)
                 .attr("height", 36)
                 .attr("x", (rectW-36)/2)
                 .attr("y", rectH + 1);

  nodeCombination.append("text")
                 .attr("x", rectW / 2)
                 .attr("y", rectH / 2 - 12)
                 .attr("dy", "3.5em")
                 .attr("text-anchor", "middle")
                 .text(function (d) {
                    if (d.combination=="Parallel") {return "||";}
                    if (d.combination=="Sequence") {return "→";}
                    if (d.combination=="Alternative") {return "v";}
                    return ""
                  });

  // Transition nodes to their new position.
  var nodeUpdate = node.transition()
                       .duration(duration)
                       .attr("transform", function (d) {return "translate(" + d.x + "," + d.y + ")";});

  var gUpdate = nodeUpdate.attr("class", function(d) {
                            var cl=d3.select(this).attr("class");
                            // console.log(cl,d);
                            if (d._children) { if (cl.indexOf(" collapsed")==-1) { return cl+" collapsed"; } }
                            else { if (cl.indexOf(" collapsed")!=-1) return cl.replace(" collapsed",""); }
                            return cl;
                          });


  // Transition exiting nodes to the parent's new position.
  var nodeExit = node.exit().transition()
                            .duration(duration)
                            .attr("transform", function (d) {return "translate(" + source.x + "," + source.y + ")";})
                            .remove();

  // Declare the links...
  var link = svg.selectAll("path.link")
                .data(links, function(d) { return d.target.id; });

  // console.log(link);
  // Enter any new links at the parent's previous position.
  link.enter().insert("path", "g")
      .attr("class", "link")
      .attr("x", rectW / 2)
      .attr("y", rectH / 2)
      .attr("d", function (d) {
        var o = {
            x: source.x0,
            y: source.y0
        };
        return diagonal({source: o, target: o});
      });

  // Transition links to their new position.
  link.transition()
      .duration(duration)
      .attr("d", diagonal);

  // Transition exiting nodes to the parent's new position.
  link.exit().transition()
      .duration(duration)
      .attr("d", function (d) {
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

}

//Redraw for zoom
function redraw() {
  //console.log("here", d3.event.translate, d3.event.scale);
  svg.attr("transform",
      "translate(" + d3.event.translate + ")"
      + " scale(" + d3.event.scale + ")");
}
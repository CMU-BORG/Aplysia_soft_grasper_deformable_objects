let w = 840;
let h = 840;

function constrain(val, lo, hi){
    if (val < lo) return lo;
    if (val > hi) return hi;
    return val;
}
function main(){
  const xScale = d3.scaleLinear()
    .domain([-5 , 5])
    .range([0, w]);

  const yScale = d3.scaleLinear()
    .domain([-5 , 5])
    .range([0, h])

  const mySlider = makeSlider(xScale, yScale, {snapToInteger: true});  

  mySlider.event.on('change', function(a){
    const join = d3.select('svg')
      .selectAll('.handle')
        .data([this]);

    join.enter()
      .append('circle')
        .style('pointer-events', 'none')
        .attr('class', d=>'handle')
        .attr('r', 20)
        .attr('cx', d=>d.x)
        .attr('cy', d=>d.y);

    join
      .attr('cx', d=>d.x)
      .attr('cy', d=>d.y);

    d3.select('#valueX')
      .text(this.valueX);

    d3.select('#valueY')
      .text(this.valueY);
  })

  d3.select('svg')
    .call(mySlider);
}

const makeSlider = (xScale, yScale, options) => {
  let dragging = false;

  const slider = (parent)=>{
    parent.append('rect')
      .attr('x', xScale.range()[0])
      .attr('y', yScale.range()[0])
      .attr('width', xScale.range()[1] - xScale.range()[0])
      .attr('height', yScale.range()[1] - yScale.range()[0])
      .attr('fill-opacity', 0.1);
    
    parent.on('mousedown', function(){
      dragging = true;
      dispatchLocation(d3.mouse(this));

      console.log('mousedown');
    })

    parent.on('mousemove', function(){
      if(dragging){
        dispatchLocation(d3.mouse(this));
      }
    });

    parent.on('mouseout',function(){
       dragging = false;
    });

    parent.on('mouseup', function(){
      dragging = false;
      dispatchLocation(d3.mouse(this), false);
    })
  };

  slider.event = d3.dispatch("change");

  function dispatchLocation(position, snap = false) {

    let valueX = constrain(xScale.invert(position[0]), -5, 5);
    let valueY =  constrain(yScale.invert(position[1]), -5, 5);

    if (options.snapToInteger){
      valueY = Math.round(valueY);
      valueX = Math.round(valueX);
    };

    if(snap){
      position[0] = xScale(valueX);
      position[1] = yScale(valueY);
    }

    slider.event.call('change', {
      x: position[0],
      y: position[1],
      valueX,
      valueY,
    });
  }

  return slider;
}

window.onload = main;

function send_gantry_data(){

    let mouseX = event.clientX;
    let mouseY = event.clientY;
    var rect = event.target.getBoundingClientRect();

    var x = mouseX - rect.left; //x position within the element.
    var y = mouseY - rect.top;  //y position within the element.

    var xp = 100 * x / rect.width;
    var yp = 100 * y / rect.height;


    let value = x.toString() + ',' + y.toString();

    $.ajax({
            url: '/participant/dual-grasper',
            type: 'POST',
            data: { 'gantry': value },
            success: function(response) {
                // document.getElementById('output').innerHTML = response;
                console.log(x + ', ' + y);
                return;
            },
            error: function(error) {
                console.log(error);
            }
    });

    document.getElementById('gantry-marker').style.setProperty("left", xp.toString() + '%');
    document.getElementById('gantry-marker').style.setProperty("top", yp.toString() + '%');
}

// mouse position is off
function send_grasper_data_left() {
    let mouseY = event.clientY;
    let rect = event.target.getBoundingClientRect();

    let y = rect.bottom - mouseY

    let value = (100*(y/rect.height)) // need to convert to percent
    let value_str = value.toString();

    $.ajax({
            url: '/participant/dual-grasper',
            type: 'POST',
            data: { 'grasper_l': value_str },
            success: function(response) {
                // document.getElementById('output').innerHTML = response;
                console.log(value);
                return;
            },
            error: function(error) {
                console.log(error);
            }
    });

    let display_value = Math.round(value, 2) + '%';

    document.getElementById('grasper_l').innerHTML=display_value;
    document.getElementById('grasper_l_slider').style.width=value_str + '%';
}

function send_grasper_data_right() {
    let mouseY = event.clientY;
    let rect = event.target.getBoundingClientRect();

    let y = rect.bottom - mouseY

    let value = (100*(y/rect.height)) // need to convert to percent
    let value_str = value.toString();

    $.ajax({
            url: '/participant/dual-grasper',
            type: 'POST',
            data: { 'grasper_r': value },
            success: function(response) {
                // document.getElementById('output').innerHTML = response;
                console.log(value);
                return;
            },
            error: function(error) {
                console.log(error);
            }
    });

    let display_value = Math.round(value, 2) + '%';

    document.getElementById('grasper_r').innerHTML=display_value;
    document.getElementById('grasper_r_slider').style.width=value_str + '%';

}

let graspser_on = -1;

function activate_grasper(){
    graspser_on *= -1;
    let value = graspser_on.toString();

    $.ajax({
        url: '/participant/dual-grasper',
        type: 'POST',
        data: { 'grasper_on': value },
        success: function(response) {
            // document.getElementById('output').innerHTML = response;
            console.log(value);
            return;
        },
        error: function(error) {
            console.log(error);
        }
    });
}
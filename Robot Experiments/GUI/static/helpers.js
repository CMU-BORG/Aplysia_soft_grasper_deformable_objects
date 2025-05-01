function map(val, ilo, ihi, flo, fhi) {
    return flo + ((fhi - flo) / (ihi - ilo)) * (val - ilo)
}

function mouse_to_mm(mouse){
    if (mouse == undefined) return 0;

    let f_mouse = Number(mouse)
    return Math.round(map(f_mouse, 0, 860, -255, 255))
}

function mm_to_mouse(mm){
    if (mm == undefined) return 0;

    let f_mm = Number(mm);
    return Math.round(map(f_mm, -255, 255, 0, 860))
}

function includes_all(attempts) {
    let attempt_nums = [];
    for (let i = 0; i < attempts.length; i++){
        attempt_nums.push(attempts[i]["n"])
    }

    for (let i = 0; i < 9; i++){
        if (!attempt_nums.includes(i)){
            return false;
        }
    }

    return true;
}

function count_successes(attempts) {

    let successes = 0;
    for (let i = 0; i < 9; i++){ // for each item
        for (let j = 0; j < attempts.length; j++){
            if (attempts[j]['n'] == i && attempts[j]['typ'] == 'success'){ // check if a succesful attempt was made on that item
                successes++; // if so increment # of successes
                break; // dont double count
            }
        }
    }

    return successes;

}

function choose_random_item(target, test){
    if (includes_all(attempts)) return;

    let attempted_items = [];
    for (let i = 0; i < attempts.length; i++) {
        attempted_items.push(attempts[i]['n']);
    }

    let n = Math.floor(9 * Math.random());
    while (attempted_items.includes(n)) {
        n = Math.floor(9 * Math.random());
    }
    let item = document.getElementById('item' + n.toString() + '-' + target + '-' + test);
    item.src = "static/public/attempt-target.svg";

    return n;
}
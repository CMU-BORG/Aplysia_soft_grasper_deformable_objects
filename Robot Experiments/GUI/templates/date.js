let date = new Date();

let day = date.getDate();
let month = date.getMonth();
let year = date.getFullYear();
let months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

let currentDate = `${day}-${months[month]}-${year}`;
document.getElementById('date').innerHTML=currentDate;
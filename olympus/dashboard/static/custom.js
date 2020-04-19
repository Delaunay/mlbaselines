var socket = io();

log = function(str) {
    console.info(str);
};


socket.on("connect", function() {
    socket.emit("handshake");
});

socket.on('disconnect', function() {
    socket.emit("disconnect");
});

// bind callback when the event is fired; forward it to the server
// so it can reply
bind_callback = function(event, id, attr, prop){
    return function(){
        var event_name = "bind_" + event + "_" + id;
        log("notifying " + event_name + " " + attr + " " + prop);

        data = null;
        if (attr !== null){
            data = document.getElementById(id).getAttribute(attr);
        }
        if (prop !== null){
            data = document.getElementById(id)[prop];
        }
        log('sending '+ data)
        socket.emit(event_name, data);
    };
};

// bind an element so we receive events when it is modified
// supported events: https://www.w3schools.com/jsref/dom_obj_event.asp
socket.on("bind", function(data) {
    log("binding " + data);

    var id = data["id"];
    var event = data["event"];
    var attr = data["attribute"];
    var prop = data["property"];

    var element = document.getElementById(id);
    element.addEventListener(event, bind_callback(event, id, attr, prop));
});


socket.on("set_html", function(data) {
    log("set_html " + data);

    var id = data["id"];
    var data = data["html"];

    var element = document.getElementById(id)
    element.innerHTML = data
});


socket.on("set_attribute", function(data) {
    log("set_attribute " + data);

    var id = data["id"];
    var attribute = data["attribute"];
    var value = data["value"];

    var element = document.getElementById(id)
    element.setAttribute(attribute, value);
});


socket.on("set_text", function(data) {
    log("set_text " + data);

    var id = data["id"];
    var data = data["html"];

    var element = document.getElementById(id)
    element.innerText = data
});

log("setup is done");
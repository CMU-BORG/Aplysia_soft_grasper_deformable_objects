from aiohttp import web

import socketio

sio = socketio.AsyncServer(async_mode='aiohttp')
app = web.Application()
sio.attach(app)





@sio.event
async def ping_from_client(sid):
    print('Pong')
    await sio.emit('pong_from_server', room=sid)






if __name__ == '__main__':
    web.run_app(app)
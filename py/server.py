# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Server"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import math
import json
import logging
import os
import inspect
import time
import getpass
import argparse
import copy

from os.path import expanduser
from zmq.eventloop import ioloop
ioloop.install()  # Needs to happen before any tornado imports!

import tornado.ioloop
import tornado.web
import tornado.websocket
import tornado.escape

logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser(description='Start the visdom server.')
parser.add_argument('-port', metavar='port', type=int, default=8097,
                    help='port to run the server on.')
parser.add_argument('-env_path', metavar='env_path', type=str,
                    default='%s/.visdom/' % expanduser("~"),
                    help='path to serialized session to reload (end with /).')
FLAGS = parser.parse_args()
class treeObj(object):
    def __init__(self,id="",name = "",type = ""):
        self._id = id
        self._name = name
        self._type = type
        self._parent = None
        self._child = []
        self._files = []

    def getId(self):
        return self._id

    def setId(self,id):
        self._id = id

    def getType(self):
        return self._type

    def getDir(self):
        return self._type+self._id

    def setChilds(self,childs):
        self._child = childs

    def getChilds(self):
        return self._child

    def setFiles(self,files):
        self._files = files

    def getFiles(self):
        return self._files

    def addChild(self,child):
        if not(child in self._child):
            self._child.append(child)
        #child._addParent(self)

    def addParent(self,Parent,over=True):
        if self._parent and over:
            self._parent = Parent

    def addChildAndParent(self,child,parent):
        if child:
            self.addChild(child)
        if parent:
            self.addParent(parent)

    def printRecc(self):
        print(self._id,self._name,self._type,self._child,self._files)
        print("---")
        if len(self._child) == 0:
            return

        for child in self._child:
            child.printRecc()

    def createTree(self):
        children = []
        print("id {}, name {}".format(self._id,self._name))
        print(self._files)
        for el in self._child:
            children.append(el.createTree())
        for el in self._files:
            children.append({"id":self._id+el,"text":el,"icon":"glyphicon glyphicon-file"})
        return {"id":self._id,"text":self._name,"children":children}
def get_rand_id():
    return str(hex(int(time.time() * 10000000))[2:])


def ensure_dir_exists(path):
    """Make sure the parent dir exists for path so we can write a file."""
    try:
        os.makedirs(path)
    except OSError as e1:
        assert e1.errno == 17  # errno.EEXIST
        pass


def get_path(filename):
    """Get the path to an asset."""
    cwd = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    return os.path.join(cwd, filename)


def escape_eid(eid):
    """Replace slashes with underscores, to avoid recognizing them
    as directories.
    """
    return eid
    #return eid.replace('/', '_')


def extract_eid(args):
    """Extract eid from args. If eid does not exist in args,
    it returns 'main'."""

    eid = 'main' if args.get('eid') is None else args.get('eid')
    return escape_eid(eid)


tornado_settings = {
    "autoescape": None,
    "debug": "/dbg/" in __file__,
    "static_path": get_path('static'),
    "template_path": get_path('static'),
    "compiled_template_cache": False
}


def serialize_env(state, eids):
    l = [i for i in eids if i in state]
    for i in l:
        print(i)
        splitted = i.split("/")[:-1]
        print(splitted)
        newpath = ""
        for el in splitted:
            newpath += el + "/"
        path = os.path.join(FLAGS.env_path, newpath)
        ensure_dir_exists(path)
        p = '%s/%s.json' % (FLAGS.env_path, i)
        open(p, 'w').write(json.dumps(state[i]))
    return l


def serialize_all(state):
    serialize_env(state, list(state.keys()))


def buildTree(path):
    """
    {"id":"SAD","text":"Root node","children":[{"id":2,"text":"Child node 1","children":[{"id":8,"text":"Child node 1","children":[{"id":9,"text":"Child node 1","children":[{"id":10,"text":"Child node 1"}]}]}]},{"id":3,"text":"Child node 2","icon":"glyphicon glyphicon-file"}]},
    {"id":4,"text":"Hola","icon":"glyphicon glyphicon-file"}
    """
    print(path)
    files = {}
    trees = []
    treeDir ={}
    for root, dr, fl in os.walk(os.path.expanduser(path)):
        root = root.replace(os.path.expanduser(path), "")
        root = root+"/" if root else root
        root = root[1:] if root and root[0] == '/' else root

        if not(root in treeDir.keys()):
            y = treeObj(id=root, name=root, type="Folder")
        else:
            y = treeDir[root]
        for el in dr:
            id = root+el+"/"
            print(id)
            aux = treeObj(id=id,name=el,type="Folder")
            treeDir[id] = aux
            y.addChild(treeDir[id])
            treeDir[id].addParent(y)
        files[y.getId()] = []
        for el in fl:
            files[y.getId()].append(el.replace('.json',""))

        print("====files: {}".format(files))
        y.setFiles(files[y.getId()])
        if not (y.getId() in treeDir.keys()):
            trees.append(y)
    print("Printing Tree Dir")
    print(treeDir)
    print("=========\n Printing tree recc")
    trees[0].printRecc()

    tree = trees[0].createTree()
    print("=======TREEE CREATED==========")
    return tree


class Application(tornado.web.Application):
    def __init__(self):
        state = {}
        subs = {}
        dirs = {}
        files = {}
        # reload state
        dirs = buildTree(FLAGS.env_path)
        ensure_dir_exists(FLAGS.env_path)
        l = [i for i in os.listdir(FLAGS.env_path) if '.json' in i]
        j = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(FLAGS.env_path)) for f in fn if '.json' in f]
        print(j)
        for i in j:
            p = i
            f = tornado.escape.json_decode(open(p, 'r').read())
            eid = i.replace('.json', '').replace(os.path.expanduser(FLAGS.env_path),'')
            if eid[0] == '/':
                eid = eid[1:]
            state[eid] = {'jsons': f['jsons'], 'reload': f['reload']}

        if 'main' not in state and 'main.json' not in l:
            state['main'] = {'jsons': {}, 'reload': {}}
            serialize_env(state, ['main'])

        handlers = [
            (r"/events", PostHandler, dict(state=state, subs=subs)),
            (r"/update", UpdateHandler, dict(state=state, subs=subs)),
            (r"/close", CloseHandler, dict(state=state, subs=subs)),
            (r"/socket", SocketHandler, dict(state=state, subs=subs)),
            (r"/env/(.*)", EnvHandler, dict(state=state, subs=subs)),
            (r"/save", SaveHandler, dict(state=state, subs=subs)),
            (r"/error/(.*)", ErrorHandler),
            (r"/(.*)", IndexHandler, dict(state=state,subs=dirs)),
        ]
        super(Application, self).__init__(handlers, **tornado_settings)


class SocketHandler(tornado.websocket.WebSocketHandler):
    def initialize(self, state, subs):
        self.state = state
        self.subs = subs

    def check_origin(self, origin):
        return True

    def open(self):
        self.sid = str(hex(int(time.time() * 10000000))[2:])
        if self not in list(self.subs.values()):
            self.eid = 'main'
            self.subs[self.sid] = self
        print('Opened new socket from ip:', self.request.remote_ip)

        self.write_message(
            json.dumps({'command': 'register', 'data': self.sid}))

    def on_message(self, message):
        print('from web client: ', message)
        msg = tornado.escape.json_decode(tornado.escape.to_basestring(message))

        cmd = msg.get('cmd')
        if cmd == 'close':
            if 'data' in msg and 'eid' in msg:
                print('closing pane ', msg['data'])
                self.state[msg['eid']]['jsons'].pop(msg['data'], None)

        elif cmd == 'save':
            # save localStorage pane metadata
            if 'data' in msg and 'eid' in msg:
                msg['eid'] = escape_eid(msg['eid'])
                self.state[msg['eid']] = copy.deepcopy(self.state[msg['prev_eid']])
                self.state[msg['eid']]['reload'] = msg['data']
                self.eid = msg['eid']
                if '/' in self.eid:
                    splitted = self.eid.split("/")[:-1]
                    print(splitted)
                    newpath = ""
                    for el in splitted:
                        newpath += el + "/"
                    path = os.path.join(FLAGS.env_path,newpath)
                    ensure_dir_exists(path)
                serialize_env(self.state, [self.eid])

    def on_close(self):
        if self in list(self.subs.values()):
            self.subs.pop(self.sid, None)


class BaseHandler(tornado.web.RequestHandler):
    def __init__(self, *request, **kwargs):
        self.include_host = True
        super(BaseHandler, self).__init__(*request, **kwargs)

    def write_error(self, status_code, **kwargs):
        logging.error("ERROR: %s: %s" % (status_code, kwargs))
        if self.settings.get("debug") and "exc_info" in kwargs:
            logging.error("rendering error page")
            import traceback
            exc_info = kwargs["exc_info"]
            # exc_info is a tuple consisting of:
            # 1. The class of the Exception
            # 2. The actual Exception that was thrown
            # 3. The traceback opbject
            try:
                params = dict(
                    error=exc_info[1],
                    trace_info=traceback.format_exception(*exc_info),
                    request=self.request.__dict__
                )

                self.render("error.html", **params)
                logging.error("rendering complete")
            except Exception as e:
                logging.error(e)


def pane(args):
    """ Build a pane dict structure for sending to client """

    if args.get('win') is None:
        uid = 'pane_' + get_rand_id()
    else:
        uid = args['win']

    return {
        'command': 'pane',
        'id': str(uid),
        'title': '' if args.get('title') is None else args['title'],
        'contentID': get_rand_id(),   # to detected updated panes
    }


def broadcast(self, msg, eid):
    for s in self.subs:
        if self.subs[s].eid == eid:
            self.subs[s].write_message(msg)


def register_pane(self, p, eid):
    # in case env doesn't exist
    self.state[eid] = self.state.get(eid, {'jsons': {}, 'reload': {}})
    env = self.state[eid]['jsons']

    if p['id'] in env:
        p['i'] = env[p['id']]['i']
    else:
        p['i'] = len(env)

    env[p['id']] = p

    broadcast(self, p, eid)
    self.write(p['id'])


class PostHandler(BaseHandler):
    def initialize(self, state, subs):
        self.state = state
        self.subs = subs

    def post(self):
        args = tornado.escape.json_decode(tornado.escape.to_basestring(self.request.body))

        ptype = args['data'][0]['type']
        print(ptype)
        p = pane(args)
        eid = extract_eid(args)

        if type(eid) == type([]):
            eid = eid[0]
        print(eid)
        if ptype in ['image', 'text']:
            p.update(dict(content=args['data'][0]['content'], type=ptype))
        else:
            p['content'] = dict(data=args['data'], layout=args['layout'])
            p['type'] = 'plot'
        register_pane(self, p, eid)


class UpdateHandler(BaseHandler):
    def initialize(self, state, subs):
        self.state = state
        self.subs = subs

    def update(self, p, args):
        pdata = p['content']['data']

        new_data = args['data']
        name = args.get('name')
        append = args.get('append')

        idxs = list(range(len(pdata)))

        if name is not None:
            assert len(new_data['x']) == 1
            idxs = [i for i in idxs if pdata[i]['name'] == name]

        # inject new trace
        if len(idxs) == 0:
            idx = len(pdata)
            pdata.append(dict(pdata[0]))   # plot is not empty, clone an entry
            pdata[idx]['name'] = name
            idxs = [idx]
            append = False
            if 'marker' in new_data:
                pdata[idx]['marker']['color'] = new_data['marker']

        for n, idx in enumerate(idxs):    # update traces
            if all([math.isnan(i) or i is None for i in new_data['x'][n]]):
                continue

            pdata[idx]['x'] = (pdata[idx]['x'] + new_data['x'][n]) if append \
                else new_data['x'][n]
            pdata[idx]['y'] = (pdata[idx]['y'] + new_data['y'][n]) if append \
                else new_data['y'][n]

        return p

    def post(self):
        args = tornado.escape.json_decode(tornado.escape.to_basestring(self.request.body))
        eid = extract_eid(args)

        if args['win'] not in self.state[eid]['jsons']:
            self.write('win does not exist')
            return

        p = self.state[eid]['jsons'][args['win']]

        if not p['content']['data'][0]['type'] == 'scatter':
            self.write('win is not scatter')
            return

        p = self.update(p, args)

        p['contentID'] = get_rand_id()
        broadcast(self, p, eid)
        self.write(p['id'])


class CloseHandler(BaseHandler):
    def initialize(self, state, subs):
        self.state = state
        self.subs = subs

    def post(self):
        args = tornado.escape.json_decode(tornado.escape.to_basestring(self.request.body))

        eid = extract_eid(args)
        win = args.get('win')

        keys = list(self.state[eid]['jsons'].keys()) if win is None else [win]
        for win in keys:
            self.state[eid]['jsons'].pop(win, None)
            broadcast(
                self, json.dumps({'command': 'close', 'data': win}), eid
            )


def load_env(state, eid, socket):
    """ load an environment to a client by socket """

    env = {}
    if eid in state:
        env = state.get(eid)
    else:
        p = os.path.join(FLAGS.env_path, eid.strip(), '.json')
        if os.path.exists(p):
            env = tornado.escape.json_decode(open(p, 'r').read())
            state[eid] = env

    if 'reload' in env:
        socket.write_message(
            json.dumps({'command': 'reload', 'data': env['reload']})
        )

    jsons = list(env.get('jsons', {}).values())
    panes = sorted(jsons, key=lambda k: ('i' not in k, k.get('i', None)))
    for v in panes:
        socket.write_message(v)

    socket.write_message(json.dumps({'command': 'layout'}))
    socket.eid = eid


def gather_envs(state):
    print(os.listdir(FLAGS.env_path))
    items = [i.replace('.json', '') for i in os.listdir(FLAGS.env_path)
                if '.json' in i]
    return sorted(list(set(items + list(state.keys()))))

def gather_envs2(state,filter):
    items = [i for i in state.keys() if filter in i]
    return sorted(items)

class EnvHandler(BaseHandler):
    def initialize(self, state, subs):
        self.state = state
        self.subs = subs

    def get(self, eid):
        if '/' in eid:
            splitted = eid.split("/")[:-1]
            newpath = ""
            for el in splitted:
                newpath += el + "/"
            #TODO: Arreglar cuando no existe el directiorio
            items = gather_envs2(self.state, newpath)
            print(len(items))
            active = items[0] if (items and eid not in items) else eid
        else:
            items = gather_envs(self.state)
            active = 'main' if eid not in items else eid
        self.render(
            'index.html',
            user=getpass.getuser(),
            items=items,
            testvar=json.dumps([
                      {"id":1,"text":"Root node","children":[{"id":2,"text":"Child node 1","children":[{"id":8,"text":"Child node 1","children":[{"id":9,"text":"Child node 1","children":[{"id":10,"text":"Child node 1"}]}]}]},{"id":3,"text":"Child node 2","icon":"glyphicon glyphicon-file"}]},
                      {'id':4,'text':'Hola','icon':'glyphicon glyphicon-file'}
                    ]),
            active_item=active
        )

    def post(self, args):
        sid = tornado.escape.json_decode(tornado.escape.to_basestring(self.request.body))['sid']
        args = args.replace(FLAGS.env_path+"/","").replace(".json","")
        print(args)
        load_env(self.state, args, self.subs[sid])


class SaveHandler(BaseHandler):
    def initialize(self, state, subs):
        self.state = state
        self.subs = subs

    def post(self):
        envs = tornado.escape.json_decode(tornado.escape.to_basestring(self.request.body))['data']
        envs = [escape_eid(eid) for eid in envs]
        ret = serialize_env(self.state, envs)  # this ignores invalid env ids
        self.write(json.dumps(ret))


class IndexHandler(BaseHandler):
    def initialize(self, state,subs):
        self.state = state
        self.dirs = subs
        print("========")
        print(self.dirs)
        print("///")

    def get(self, args, **kwargs):
        items = gather_envs(self.state)
        self.render(
            'index.html',
            user=getpass.getuser(),
            items=items,
            testvar=json.dumps(self.dirs["children"]),
            active_item='main'
        )


class ErrorHandler(BaseHandler):
    def get(self, text):
        error_text = text or "test error"
        raise Exception(error_text)


def main():
    print("It's Alive!")
    app = Application()
    app.listen(FLAGS.port, max_buffer_size=1024 ** 3)
    ioloop.IOLoop.instance().start()
    logging.info("Application Started")


if __name__ == "__main__":
    main()


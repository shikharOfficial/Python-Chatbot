# app/cli.py
import click
from flask.cli import with_appcontext
from app import create_app

app = create_app()

@click.command(name='runserver')
@with_appcontext
def runserver():
    """Run the Flask development server with Gunicorn"""
    import gunicorn.app.base
    from six import iteritems

    class GunicornApp(gunicorn.app.base.BaseApplication):
        def __init__(self, app, options=None):
            self.application = app
            self.options = options or {}
            super().__init__()

        def load_config(self):
            config = {key: value for key, value in iteritems(self.options) if key in self.cfg.settings}
            for key, value in iteritems(config):
                self.cfg.set(key.lower(), value)

        def load_application(self):
            return self.application

    options = {
        'bind': '127.0.0.1:8000',
        'workers': 4,
        'accesslog': 'access.log',
        'errorlog': 'error.log',
        'loglevel': 'debug',
        'timeout': 600
    }

    GunicornApp(app, options).run()

if __name__ == '__main__':
    runserver()

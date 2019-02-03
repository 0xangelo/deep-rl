import os
import tempfile
os.environ['OPENAI_LOGDIR'] = os.path.join(tempfile.gettempdir(), 'deep-rl')

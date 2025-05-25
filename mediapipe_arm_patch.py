"""
Patch for MediaPipe to work without TensorFlow/JAX on ARM
"""
import sys
import types

# Create dummy tensorflow module
tensorflow = types.ModuleType('tensorflow')
tensorflow.tools = types.ModuleType('tensorflow.tools')
tensorflow.tools.docs = types.ModuleType('tensorflow.tools.docs')
tensorflow.tools.docs.doc_controls = types.ModuleType('tensorflow.tools.docs.doc_controls')

# Add dummy decorators
tensorflow.tools.docs.doc_controls.do_not_generate_docs = lambda: lambda x: x
tensorflow.tools.docs.doc_controls.do_not_doc_inheritable = lambda: lambda x: x

# Inject into sys.modules
sys.modules['tensorflow'] = tensorflow
sys.modules['tensorflow.tools'] = tensorflow.tools
sys.modules['tensorflow.tools.docs'] = tensorflow.tools.docs
sys.modules['tensorflow.tools.docs.doc_controls'] = tensorflow.tools.docs.doc_controls

print("MediaPipe ARM patch applied successfully!")

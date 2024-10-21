# note, these shaders are run in @react-three/drei, they are not raw GLSL

VERTEX_SHADER_PATH = "res/shaders/object_color_solid.vert"
FRAGMENT_SHADER_PATH = "res/shaders/object_color_solid.frag"

def get_vertex_shader() -> str:
    with open(VERTEX_SHADER_PATH) as fd:
        vert_src: str = fd.read()
        return vert_src

def get_fragment_shader() -> str:
    with open(FRAGMENT_SHADER_PATH) as fd:
        frag_src : str = fd.read()
        return frag_src





def test_shader_load():
    print("fragment shader source:")
    print(get_fragment_shader())
    print("vertex shader source:")
    print(get_vertex_shader())

if __name__ == "__main__":
    test_shader_load()

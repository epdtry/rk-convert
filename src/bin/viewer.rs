use std::collections::HashMap;
use std::env;
use std::f32::consts::PI;
use std::ffi::{CString, CStr};
use std::fs::{self, File};
use std::io;
use std::path::Path;
use std::ptr;
use gl::types::{GLenum, GLint, GLuint, GLsizei, GLvoid};
use rkengine::model::ModelFile;
use rkengine::modify;
use rkengine::pvr::PvrFile;


const VERTEX_SHADER_SOURCE: &str = r"
#version 330 core

layout (location = 0) in vec4 pos;
layout (location = 1) in vec2 uv_in;

out vec2 uv;

void main() {
    //gl_Position = vec4(pos.xyz / 100.0, 1.0);
    gl_Position = pos;
    //gl_Position = vec4(pos.xy, 5.0, 5.0);
    uv = vec2(uv_in.x, 1.0 - uv_in.y);
    //gl_Position = vec4(pos.x / 20.0, pos.y / 20.0, 0.5, 1.0);
}
";

const FRAGMENT_SHADER_SOURCE: &str = r"
#version 330 core

uniform sampler2D tex;
in vec2 uv;
out vec4 color;

void main() {
    //color = vec4(1.0, 0.0, 0.0, 1.0);
    color = texture2D(tex, uv);
}
";

fn compile_shader(kind: GLenum, src: &str) -> GLuint {
    unsafe {
        let shader = gl::CreateShader(kind);
        let src = CString::new(src).unwrap();
        gl::ShaderSource(shader, 1, &src.as_ptr(), ptr::null_mut());
        gl::CompileShader(shader);

        let mut ok = 0;
        gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut ok);
        if ok == 0 {
            let mut len = 0;
            gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut len);
            let mut buf = vec![0; len as usize];
            gl::GetShaderInfoLog(shader, len, &mut len, buf.as_mut_ptr() as *mut _);
            let msg = String::from_utf8_lossy(&buf);
            panic!("shader error:\n{}", msg);
        }

        shader
    }
}

fn compile_shader_program() -> GLuint {
    let vert = compile_shader(gl::VERTEX_SHADER, VERTEX_SHADER_SOURCE);
    let frag = compile_shader(gl::FRAGMENT_SHADER, FRAGMENT_SHADER_SOURCE);

    unsafe {
        let prog = gl::CreateProgram();
        gl::AttachShader(prog, vert);
        gl::AttachShader(prog, frag);
        gl::LinkProgram(prog);

        let mut ok = 0;
        gl::GetProgramiv(prog, gl::LINK_STATUS, &mut ok);
        if ok == 0 {
            let mut len = 0;
            gl::GetProgramiv(prog, gl::INFO_LOG_LENGTH, &mut len);
            let mut buf = vec![0; len as usize];
            gl::GetProgramInfoLog(prog, len, &mut len, buf.as_mut_ptr() as *mut _);
            let msg = String::from_utf8_lossy(&buf);
            panic!("shader error:\n{}", msg);
        }

        prog
    }
}

fn compute_perspective_matrix(fov: f32, w: i32, h: i32) -> [f32; 16] {
    let near = 0.1;
    let far = 10.;
    let top = near * (fov / 2.).tan();
    let bottom = -top;
    let left = bottom * w as f32 / h as f32;
    let right = top * w as f32 / h as f32;

    let (n, f, t, b, l, r) = (near, far, top, bottom, left, right);
    /*
    [
        2 * n / (r - l), 0., 0., 0.,
        0., 2 * n / (t - b), 0., 0.,
        (r + l) / (r - l), (t + b) / (t - b), -(f + n) / (f - n), -1.,
        0., 0., -2 * f * n / (f - n), 0.,
    ]
    */

    /*
    [
        2. * n / (r - l), 0., (r + l) / (r - l), 0.,
        0., 2. * n / (t - b), (t + b) / (t - b), 0.,
        0., 0., -(f + n) / (f - n), -2. * f * n / (f - n),
        0., 0., -1., 0.,
    ]
    */

    [
        n / r, 0., 0., 0.,
        0., n / t, 0., 0.,
        0., 0., -(f + n) / (f - n), -2. * f * n / (f - n),
        0., 0., -1., 0.,
    ]

}

fn scale(x: f32, y: f32, z: f32) -> [f32; 16] {
    [
        x, 0., 0., 0.,
        0., y, 0., 0.,
        0., 0., z, 0.,
        0., 0., 0., 1.,
    ]
}

fn scale_uniform(c: f32) -> [f32; 16] {
    scale(c, c, c)
}

fn translate(x: f32, y: f32, z: f32) -> [f32; 16] {
    [
        1., 0., 0., x,
        0., 1., 0., y,
        0., 0., 1., z,
        0., 0., 0., 1.,
    ]
}

fn rotate_x(theta: f32) -> [f32; 16] {
    let s = theta.sin();
    let c = theta.cos();
    [
        1., 0., 0., 0.,
        0., c, -s, 0.,
        0., s, c, 0.,
        0., 0., 0., 1.,
    ]
}

fn rotate_y(theta: f32) -> [f32; 16] {
    let s = theta.sin();
    let c = theta.cos();
    [
        c, 0., s, 0.,
        0., 1., 0., 0.,
        -s, 0., c, 0.,
        0., 0., 0., 1.,
    ]
}

fn rotate_z(theta: f32) -> [f32; 16] {
    let s = theta.sin();
    let c = theta.cos();
    [
        c, -s, 0., 0.,
        s, c, 0., 0.,
        0., 0., 1., 0.,
        0., 0., 0., 1.,
    ]
}

fn matmat(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut c = [0.; 16];
    for i in 0..4 {
        for j in 0..4 {
            let mut acc = 0.;
            for k in 0..4 {
                acc += a[4 * i + k] * b[4 * k + j];
            }
            c[i * 4 + j] = acc;
        }
    }
    c
}

fn matmats(ms: &[[f32; 16]]) -> [f32; 16] {
    let mut out = [
        1., 0., 0., 0.,
        0., 1., 0., 0.,
        0., 0., 1., 0.,
        0., 0., 0., 1.,
    ];
    for &m in ms {
        out = matmat(out, m);
    }
    out
}

fn matvec(m: [f32; 16], v: [f32; 4]) -> [f32; 4] {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3] * v[3],
        m[4] * v[0] + m[5] * v[1] + m[6] * v[2] + m[7] * v[3],
        m[8] * v[0] + m[9] * v[1] + m[10] * v[2] + m[11] * v[3],
        m[12] * v[0] + m[13] * v[1] + m[14] * v[2] + m[15] * v[3],
    ]
}

fn main() -> io::Result<()> {
    // Load model

    let args = env::args_os().collect::<Vec<_>>();
    assert!(args.len() == 2, "usage: {} <input.rk>", args[0].to_string_lossy());

    let model_path = Path::new(&args[1]);

    eprintln!("load object from {}", model_path.display());
    let mut mf = ModelFile::new(File::open(model_path)?);
    let mut o = mf.read_object()?;

    //modify::flip_normals(&mut o);

    o.models.retain(|m| !m.name.contains("eyes_") || m.name.contains("open"));

    let mut material_images = HashMap::new();
    for m in &o.models {
        eprintln!("model {} material = {}", m.name, m.material);
        if m.material.len() == 0 {
            continue;
        }
        if material_images.contains_key(&m.material) {
            continue;
        }
        let texture_path = model_path.with_file_name(format!("{}.pvr", m.material));
        eprintln!("  load {} from {}", m.material, texture_path.display());
        // TODO: read {name}.rkm and extract DiffuseTexture name
        let mut pf = PvrFile::new(File::open(texture_path)?);
        let img = pf.read_image()?;
        material_images.insert(&m.material, img);
    }


    // SDL init

    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    {
        let gl_attr = video_subsystem.gl_attr();
        gl_attr.set_double_buffer(true);
        gl_attr.set_depth_size(16);
        gl_attr.set_context_profile(sdl2::video::GLProfile::Core);
        gl_attr.set_context_version(3, 3);
    }


    let window = video_subsystem.window("rkengine viewer", 1200, 900)
        .position_centered()
        .opengl()
        .build()
        .unwrap();

    let gl_context = window.gl_create_context().unwrap();
    window.gl_make_current(&gl_context).unwrap();

    gl::load_with(|name| video_subsystem.gl_get_proc_address(name) as *const _);


    // View setup
    let m_proj = compute_perspective_matrix(PI / 2., 1200, 900);
    let m_combined = matmats(&[
        m_proj,
        translate(0., -1., -2.),
        rotate_x(PI / 12.),
        rotate_y(5. * PI / 4.),
        scale_uniform(1. / 100.),
        scale(1., -1., 1.),
    ]);


    // OpenGL init
    let prog = compile_shader_program();

    let vbo = unsafe {
        let mut id = 0;
        gl::GenBuffers(1, &mut id);
        assert_ne!(id, 0);
        id
    };

    let uv_vbo = unsafe {
        let mut id = 0;
        gl::GenBuffers(1, &mut id);
        assert_ne!(id, 0);
        id
    };

    let vao = unsafe {
        let mut id = 0;
        gl::GenVertexArrays(1, &mut id);
        assert_ne!(id, 0);
        id
    };

    unsafe {
        gl::UseProgram(prog);

        gl::BindVertexArray(vao);

        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
        gl::EnableVertexAttribArray(0);
        gl::VertexAttribPointer(
            0,
            4,              // components
            gl::FLOAT,
            gl::FALSE,      // normalized
            16,             // stride
            ptr::null(),    // offset
        );

        gl::BindBuffer(gl::ARRAY_BUFFER, uv_vbo);
        gl::EnableVertexAttribArray(1);
        gl::VertexAttribPointer(
            1,
            2,              // components
            gl::FLOAT,
            gl::FALSE,      // normalized
            8,              // stride
            ptr::null(),    // offset
        );

        gl::Enable(gl::DEPTH_TEST);
        gl::DepthFunc(gl::LESS);
        gl::Enable(gl::CULL_FACE);
        gl::CullFace(gl::BACK);
        gl::Enable(gl::BLEND);
        gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
    }


    // Load textures
    let mut material_textures = HashMap::with_capacity(material_images.len());
    for (&name, img) in &material_images {
        unsafe {
            let mut id = 0;
            gl::GenTextures(1, &mut id);
            assert_ne!(id, 0);
            let tex = id;

            gl::BindTexture(gl::TEXTURE_2D, tex);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::REPEAT as _);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::REPEAT as _);
            gl::TexParameteri(
                gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR_MIPMAP_LINEAR as _);
            gl::TexParameteri(
                gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as _);
            let (w, h) = img.size;
            gl::TexImage2D(
                gl::TEXTURE_2D,
                0,  // mipmap level
                gl::RGBA as _,
                w as _,
                h as _,
                0,
                gl::RGBA,
                gl::UNSIGNED_BYTE,
                img.data.as_ptr() as *const _,
            );
            gl::GenerateMipmap(gl::TEXTURE_2D);

            material_textures.insert(name, tex);
        }
    }


    // Initialize vertex data

    struct DrawOp {
        start: GLint,
        len: GLsizei,
        tex: GLuint,
    }

    let mut verts = Vec::with_capacity(o.models.iter().map(|m| m.tris.len() * 3).sum());
    let mut uvs = Vec::with_capacity(verts.capacity());
    let mut draw_ops = Vec::with_capacity(o.models.len());
    for m in &o.models {
        let start = verts.len();

        for t in &m.tris {
            for &vi in &t.verts {
                //verts.push(m.verts[vi].pos);
                let [x, y, z] = m.verts[vi].pos;
                verts.push(matvec(m_combined, [x, y, z, 1.]));
            }
            for &uv in &t.uvs {
                uvs.push(uv);
            }
        }

        let count = verts.len() - start;

        let tex = if m.material.len() != 0 {
            material_textures[&m.material]
        } else {
            0
        };

        draw_ops.push(DrawOp {
            start: start as _,
            len: count as _,
            tex,
        });
    }

    /*
    let mut verts = Vec::new();
    let uvs = vec![[0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]];
    let draw_ops = vec![DrawOp {
        start: 0,
        len: 6,
        tex: 0,
    }];
    */

    unsafe {
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
        gl::BufferData(
            gl::ARRAY_BUFFER,
            (16 * verts.len()) as _,
            verts.as_ptr() as *const GLvoid,
            gl::STATIC_DRAW,
        );

        gl::BindBuffer(gl::ARRAY_BUFFER, uv_vbo);
        gl::BufferData(
            gl::ARRAY_BUFFER,
            (8 * uvs.len()) as _,
            uvs.as_ptr() as *const GLvoid,
            gl::STATIC_DRAW,
        );
    }


    // SDL event loop

    let mut event_pump = sdl_context.event_pump().unwrap();
    let mut t = 0.0_f32;

    'main: loop {
        for event in event_pump.poll_iter() {
            match event {
                sdl2::event::Event::Quit {..} => break 'main,
                sdl2::event::Event::Window {
                    win_event: sdl2::event::WindowEvent::Resized(w, h),
                    ..
                } => {
                    eprintln!("resize viewport to {}, {}", w, h);
                    unsafe {
                        gl::Viewport(0, 0, w, h);
                        // TODO: update projection matrix
                    }
                },
                _ => {},
            }
        }


        /*
        // Update vertex data
        t += 1./60.;

        verts.clear();

        verts.push(matvec(m_combined, [0., 0., 1. * t.sin(), 1.]));
        verts.push(matvec(m_combined, [1., 0., 1. * t.sin(), 1.]));
        verts.push(matvec(m_combined, [0., 1., 1. * t.sin(), 1.]));

        verts.push(matvec(m_combined, [0., 0., 0., 1.]));
        verts.push(matvec(m_combined, [t.cos(), t.sin(), 0., 1.]));
        verts.push(matvec(m_combined, [-t.sin(), t.cos(), 0., 1.]));

        dbg!(&verts);
        */

        unsafe {
            gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
            gl::BufferData(
                gl::ARRAY_BUFFER,
                (16 * verts.len()) as _,
                verts.as_ptr() as *const GLvoid,
                gl::STATIC_DRAW,
            );
        }


        unsafe {
            gl::ClearColor(0.0, 0.0, 1.0, 0.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

            for op in &draw_ops {
                gl::BindTexture(gl::TEXTURE_2D, op.tex);
                gl::DrawArrays(gl::TRIANGLES, 0, verts.len() as _);
            }
        }

        window.gl_swap_window();
    }

    Ok(())
}

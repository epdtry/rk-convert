use std::cmp;
use std::collections::HashMap;
use std::env;
use std::f32::consts::PI;
use std::ffi::{CString, CStr, OsStr};
use std::fs::{self, File};
use std::io;
use std::path::Path;
use std::ptr;
use std::time::Instant;
use gl::types::{GLenum, GLint, GLuint, GLsizei, GLvoid};
use nalgebra::{Vector3, Vector4, Matrix4, Quaternion, UnitQuaternion};
use rkengine::anim::AnimFile;
use rkengine::model::ModelFile;
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

fn compute_perspective_matrix(fov: f32, w: i32, h: i32) -> Matrix4<f32> {
    Matrix4::new_perspective(w as f32 / h as f32, fov, 0.1, 10.)
}

fn main() -> io::Result<()> {
    // Load model

    let args = env::args_os().collect::<Vec<_>>();
    assert!(
        args.len() == 2 || args.len() == 3,
        "usage: {} <model.rk> [anim.csv|anim.anim]",
        args[0].to_string_lossy(),
    );

    let model_path = Path::new(&args[1]);
    let anim_path = args.get(2).map(|s| Path::new(s));

    eprintln!("load object from {}", model_path.display());
    let mut mf = ModelFile::new(File::open(model_path)?);
    let mut o = mf.read_object()?;

    eprintln!("bones:");
    for b in &o.bones {
        if let Some(i) = b.parent {
            eprintln!("  {}, parent = {}", b.name, o.bones[i].name);
        } else {
            eprintln!("  {}", b.name);
        }
    }

    struct AnimRange {
        name: String,
        start: usize,
        end: usize,
        frame_rate: u32,
    }
    let (anim, anim_ranges) = if let Some(anim_path) = anim_path {
        if anim_path.extension() == Some(OsStr::new("csv")) {
            let mut ranges = Vec::new();
            let mut reader = csv::ReaderBuilder::new()
                .has_headers(false)
                .from_path(anim_path)?;
            for row in reader.deserialize() {
                let (name, start, end, frame_rate): (String, usize, usize, u32) = row?;
                ranges.push(AnimRange { name, start, end, frame_rate });
            }

            let mut af = AnimFile::new(File::open(anim_path.with_extension("anim"))?);
            (Some(af.read_anim()?), ranges)
        } else {
            let mut af = AnimFile::new(File::open(anim_path)?);
            let anim = af.read_anim()?;
            let ranges = vec![AnimRange {
                name: "all".into(),
                start: 0,
                end: anim.frames.len(),
                frame_rate: 15,
            }];
            (Some(anim), ranges)
        }
    } else {
        (None, Vec::new())
    };

    o.models.retain(|m| !m.name.contains("eyes_") || m.name.contains("open"));
    o.models.retain(|m| m.name != "a_rainbowdash_cloud");

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
    let m_combined =
        m_proj *
        Matrix4::new_translation(&Vector3::new(0., -1., -2.)) *
        Matrix4::new_rotation(Vector3::new(PI / 12., 0., 0.)) *
        Matrix4::new_rotation(Vector3::new(0., 5. * PI / 4., 0.)) *
        Matrix4::new_scaling(1. / 100.) *
        Matrix4::from_diagonal(&Vector4::new(1., -1., 1., 1.));


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
                verts.push(m_combined * Vector4::new(x, y, z, 1.));
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


    // Misc initialization

    let num_frames = match anim {
        Some(ref anim) => anim.frames.len(),
        None => 1,
    };


    // SDL event loop

    let mut event_pump = sdl_context.event_pump().unwrap();
    let mut range_counter = 0;
    let mut start_time = Instant::now();
    if anim.is_some() {
        let r = &anim_ranges[0];
        eprintln!("{}: {} frames", r.name, r.end - r.start);
    }

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


        if let Some(anim) = anim.as_ref() {
            let r = &anim_ranges[range_counter];
            let len = cmp::max(1, r.end - r.start);

            let t = start_time.elapsed().as_secs_f32();
            let frame_num_rel: i32 = ((t - 0.5) * r.frame_rate as f32).floor() as i32;
            let frame_num: usize = r.start +
                cmp::max(0, cmp::min(len as i32 - 1, frame_num_rel)) as usize;

            if t >= len as f32 / r.frame_rate as f32 + 1. {
                start_time = Instant::now();
                range_counter = (range_counter + 1) % anim_ranges.len();
                let r = &anim_ranges[range_counter];
                eprintln!("{}: {} frames", r.name, r.end - r.start);
            }

            // Update vertices using bones
            let frame = &anim.frames[frame_num];

            assert_eq!(o.bones.len(), frame.bones.len());
            let mut bone_matrix = Vec::with_capacity(o.bones.len());
            for (bone, pose) in o.bones.iter().zip(frame.bones.iter()) {
                let bone_mat_inv = Matrix4::from_column_slice(&bone.matrix)
                    .try_inverse().unwrap();
                let [a, b, c, d] = pose.quat;
                let quat = UnitQuaternion::from_quaternion(Quaternion::new(a, b, c, d));
                let [x, y, z] = pose.pos;
                let pose_mat = Matrix4::new_translation(&Vector3::new(x, y, z)) *
                    Matrix4::from(quat);

                // We multiply the vertex position by `bone_mat_inv` to convert it to bone space,
                // then multiply by `pose_mat` to get the posed position.
                bone_matrix.push(pose_mat * bone_mat_inv);
            }

            verts.clear();
            for m in &o.models {
                for t in &m.tris {
                    for &vi in &t.verts {
                        let [x0, y0, z0] = m.verts[vi].pos;
                        let v0 = Vector4::new(x0, y0, z0, 1.);
                        let mut v = Vector4::new(0., 0., 0., 0.);
                        let mut total_weight = 0.;
                        for bw in &m.verts[vi].bone_weights {
                            let weight = bw.weight as f32;
                            v += bone_matrix[bw.bone] * v0 * weight;
                            total_weight += weight;
                        }

                        verts.push(m_combined * (v / total_weight));
                    }
                }
            }
        }


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
            gl::ClearColor(0.3, 0.3, 0.3, 0.0);
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

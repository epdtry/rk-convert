use std::collections::hash_map::{HashMap, Entry};
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;
use std::str::FromStr;
use xml::name::OwnedName;
use xml::reader::{self, EventReader, XmlEvent};


#[derive(Debug, Default)]
pub struct AnimObjects {
    /// Map from .rk `Object` name to subobject info.
    pub subobjects: HashMap<String, Vec<Subobject>>,
    /// Map from .csv `AnimRange` name to animation frame info.
    pub anims: HashMap<String, Vec<FrameObjects>>,
}

#[derive(Debug)]
pub struct Subobject {
    /// The ID used in the `AnimObjects` XML file to refer to this subobject.
    pub xml_id: String,
    /// The .rk `Model` name for this subobject.
    pub model_name: String,
    /// Whether the subobject should be visible by default.
    pub default_visible: bool,
}

#[derive(Debug)]
pub struct FrameObjects {
    /// The 1-based frame index when these object settings should take effect.
    pub index: usize,
    /// Which eye objects should be shown starting with this frame.
    pub eye_state: Option<EyeState>,
    /// Which subobjects should be shown or hidden starting with this frame.  Keys are `Subobject`
    /// `xml_id`s, not `Model` names.
    pub subobject_state: HashMap<String, bool>,
}

#[derive(Debug)]
pub struct EyeState {
    pub mode: EyeMode,
    pub blink: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum EyeMode {
    None,
    Open,
    Closed,
    Happy,
    Frown,
}


pub fn read_anim_xml(path: impl AsRef<Path>) -> io::Result<AnimObjects> {
    let f = File::open(path)?;
    read_anim_xml_inner(f).map_err(|e| {
        let kind = match e.kind() {
            reader::ErrorKind::Io(ref e) => e.kind(),
            _ => io::ErrorKind::Other,
        };
        io::Error::new(kind, e)
    })
}

fn read_anim_xml_inner(r: impl Read) -> reader::Result<AnimObjects> {
    let mut r = EventReader::new(r);
    let mut it = r.into_iter();

    let mut ao = read_top_level(&mut it)?;

    for frames in ao.anims.values_mut() {
        frames.sort_by_key(|f| f.index);
    }

    Ok(ao)
}

fn read_top_level(
    it: &mut impl Iterator<Item = reader::Result<XmlEvent>>,
) -> reader::Result<AnimObjects> {
    let mut mesh_list = None;
    let mut anim_list = None;
    loop {
        let evt = match it.next() {
            Some(x) => x?,
            None => break,
        };
        match evt {
            XmlEvent::StartDocument { .. } => {},
            XmlEvent::EndDocument => break,
            XmlEvent::ProcessingInstruction { .. } => {},
            XmlEvent::StartElement { ref name, .. } if name.local_name == "MeshList" => {
                assert!(mesh_list.is_none(), "found multiple MeshList at top level");
                mesh_list = Some(read_mesh_list(it)?);
            },
            XmlEvent::StartElement { ref name, .. } if name.local_name == "AnimationList" => {
                assert!(anim_list.is_none(), "found multiple AnimationList at top level");
                anim_list = Some(read_animation_list(it)?);
            },
            XmlEvent::StartElement { ref name, .. } => {
                panic!("unexpected element {:?} at top level", name);
            },
            XmlEvent::EndElement { .. } => {},
            XmlEvent::CData(..) => {},
            XmlEvent::Comment(..) => {},
            XmlEvent::Characters(..) => {},
            XmlEvent::Whitespace(..) => {},
        }
    }

    Ok(AnimObjects {
        subobjects: mesh_list.unwrap_or_else(HashMap::new),
        anims: anim_list.unwrap_or_else(HashMap::new),
    })
}

fn read_mesh_list(
    it: &mut impl Iterator<Item = reader::Result<XmlEvent>>,
) -> reader::Result<HashMap<String, Vec<Subobject>>> {
    let mut m = HashMap::new();
    loop {
        let evt = match it.next() {
            Some(x) => x?,
            None => break,
        };
        match evt {
            XmlEvent::StartDocument { .. } => {},
            XmlEvent::EndDocument => break,
            XmlEvent::ProcessingInstruction { .. } => {},
            XmlEvent::StartElement { ref name, .. } => {
                let key = name.local_name.clone();
                let value = read_mesh_list_entry(it, name)?;
                assert!(!m.contains_key(&key), "duplicate MeshList entry for {:?}", key);
                m.insert(key, value);
            },
            XmlEvent::EndElement { ref name, .. } if name.local_name == "MeshList" => break,
            XmlEvent::EndElement { .. } => {},
            XmlEvent::CData(..) => {},
            XmlEvent::Comment(..) => {},
            XmlEvent::Characters(..) => {},
            XmlEvent::Whitespace(..) => {},
        }
    }
    Ok(m)
}

fn read_mesh_list_entry(
    it: &mut impl Iterator<Item = reader::Result<XmlEvent>>,
    tag: &OwnedName,
) -> reader::Result<Vec<Subobject>> {
    let mut v = Vec::new();
    loop {
        let evt = match it.next() {
            Some(x) => x?,
            None => break,
        };
        match evt {
            XmlEvent::StartDocument { .. } => {},
            XmlEvent::EndDocument => break,
            XmlEvent::ProcessingInstruction { .. } => {},
            XmlEvent::StartElement { ref name, ref attributes, .. }
            if name.local_name == "SubObject" => {
                let mut id = None;
                let mut name = None;
                let mut default_visible = None;
                for attr in attributes {
                    match &attr.name.local_name as &str {
                        "ID" => {
                            assert!(id.is_none(), "duplicate attribute ID");
                            id = Some(attr.value.clone());
                        },
                        "Name" => {
                            assert!(name.is_none(), "duplicate attribute Name");
                            name = Some(attr.value.clone());
                        },
                        "DefaultVisible" => {
                            assert!(default_visible.is_none(),
                                "duplicate attribute DefaultVisible");
                            default_visible = Some(match &attr.value as &str {
                                "0" => false,
                                "1" => true,
                                s => panic!("unexpected value {:?} for DefaultVisible", s),
                            });
                        },
                        _ => {},
                    }
                }
                v.push(Subobject {
                    xml_id: id.unwrap_or_else(|| panic!("missing ID in SubObject")),
                    model_name: name.unwrap_or_else(|| panic!("missing ID in SubObject")),
                    default_visible: default_visible
                        .unwrap_or_else(|| panic!("missing DefaultVisible in SubObject")),
                });
            },
            XmlEvent::StartElement { ref name, .. } => {
                panic!("unexpected element {:?} in MeshList entry", name);
            },
            XmlEvent::EndElement { ref name, .. } if name.local_name == "SubObject" => {},
            XmlEvent::EndElement { ref name, .. } if name == tag => break,
            XmlEvent::EndElement { ref name, .. } => {
                panic!("unexpected end of element {:?} inside mesh list entry", name);
            },
            XmlEvent::CData(..) => {},
            XmlEvent::Comment(..) => {},
            XmlEvent::Characters(..) => {},
            XmlEvent::Whitespace(..) => {},
        }
    }
    Ok(v)
}

fn read_animation_list(
    it: &mut impl Iterator<Item = reader::Result<XmlEvent>>,
) -> reader::Result<HashMap<String, Vec<FrameObjects>>> {
    let mut m = HashMap::new();
    loop {
        let evt = match it.next() {
            Some(x) => x?,
            None => break,
        };
        match evt {
            XmlEvent::StartDocument { .. } => {},
            XmlEvent::EndDocument => break,
            XmlEvent::ProcessingInstruction { .. } => {},
            XmlEvent::StartElement { ref name, ref attributes, .. }
            if name.local_name == "Animation" => {
                let mut name = None;
                for attr in attributes {
                    match &attr.name.local_name as &str {
                        "Name" => {
                            assert!(name.is_none(), "duplicate attribute Name");
                            name = Some(attr.value.clone());
                        },
                        _ => {},
                    }
                }
                let key = name.unwrap_or_else(|| panic!("missing Name in Animation"));
                let value = read_animation(it)?;
                match m.entry(key) {
                    Entry::Vacant(e) => { e.insert(value); },
                    Entry::Occupied(e) => { e.into_mut().extend(value); },
                }
            },
            XmlEvent::StartElement { ref name, .. } => {
                panic!("unexpected element {:?} in AnimationList", name);
            },
            XmlEvent::EndElement { ref name, .. } if name.local_name == "AnimationList" => break,
            XmlEvent::EndElement { ref name, .. } => {
                panic!("unexpected end of element {:?} in AnimationList", name);
            },
            XmlEvent::CData(..) => {},
            XmlEvent::Comment(..) => {},
            XmlEvent::Characters(..) => {},
            XmlEvent::Whitespace(..) => {},
        }
    }
    Ok(m)
}

fn read_animation(
    it: &mut impl Iterator<Item = reader::Result<XmlEvent>>,
) -> reader::Result<Vec<FrameObjects>> {
    let mut v = Vec::new();
    loop {
        let evt = match it.next() {
            Some(x) => x?,
            None => break,
        };
        match evt {
            XmlEvent::StartDocument { .. } => {},
            XmlEvent::EndDocument => break,
            XmlEvent::ProcessingInstruction { .. } => {},
            XmlEvent::StartElement { ref name, ref attributes, .. }
            if name.local_name == "Frame" => {
                let mut index = None;
                for attr in attributes {
                    match &attr.name.local_name as &str {
                        "Index" => {
                            assert!(index.is_none(), "duplicate attribute Name");
                            let index_val = usize::from_str(&attr.value).unwrap_or_else(|_| {
                                panic!("bad Index value {:?} in Animation", attr.value);
                            });
                            index = Some(index_val);
                        },
                        _ => {},
                    }
                }
                let index = index.unwrap_or_else(|| panic!("missing Index in Frame"));
                let frame = read_frame(it, index)?;
                v.push(frame);
            },
            XmlEvent::StartElement { ref name, .. } => {
                panic!("unexpected element {:?} in Animation", name);
            },
            XmlEvent::EndElement { ref name, .. } if name.local_name == "Animation" => break,
            XmlEvent::EndElement { ref name, .. } => {
                panic!("unexpected end of element {:?} in Animation", name);
            },
            XmlEvent::CData(..) => {},
            XmlEvent::Comment(..) => {},
            XmlEvent::Characters(..) => {},
            XmlEvent::Whitespace(..) => {},
        }
    }
    Ok(v)
}

fn read_frame(
    it: &mut impl Iterator<Item = reader::Result<XmlEvent>>,
    index: usize,
) -> reader::Result<FrameObjects> {
    let mut eye_state = None;
    let mut subobject_state = HashMap::new();
    loop {
        let evt = match it.next() {
            Some(x) => x?,
            None => break,
        };
        match evt {
            XmlEvent::StartDocument { .. } => {},
            XmlEvent::EndDocument => break,
            XmlEvent::ProcessingInstruction { .. } => {},
            XmlEvent::StartElement { ref name, ref attributes, .. }
            if name.local_name == "EyeSet" => {
                assert!(eye_state.is_none(), "duplicate element EyeSet in Frame");
                let mut open = None;
                let mut enable_blink = None;
                for attr in attributes {
                    match &attr.name.local_name as &str {
                        "Open" => {
                            assert!(open.is_none(), "duplicate attribute Open");
                            open = Some(match &attr.value as &str {
                                "None" => EyeMode::None,
                                "Open" => EyeMode::Open,
                                "Closed" => EyeMode::Closed,
                                "Happy" => EyeMode::Happy,
                                "Frown" => EyeMode::Frown,
                                s => panic!("unexpected value {:?} for Open", s),
                            });
                        },
                        "EnableBlink" => {
                            assert!(enable_blink.is_none(), "duplicate attribute EnableBlink");
                            enable_blink = Some(match &attr.value as &str {
                                "0" => false,
                                "1" => true,
                                s => panic!("unexpected value {:?} for EnableBlink", s),
                            });
                        },
                        _ => {},
                    }
                }
                eye_state = Some(EyeState {
                    mode: open.unwrap_or_else(|| panic!("missing attribute Open in EyeSet")),
                    blink: enable_blink.unwrap_or(true),
                });
            },
            XmlEvent::StartElement { ref name, ref attributes, .. }
            if name.local_name == "SubObject" => {
                let mut id = None;
                let mut show = None;
                for attr in attributes {
                    match &attr.name.local_name as &str {
                        "ID" => {
                            assert!(id.is_none(), "duplicate attribute ID");
                            id = Some(attr.value.clone());
                        },
                        "Show" => {
                            assert!(show.is_none(), "duplicate attribute Show");
                            show = Some(match &attr.value as &str {
                                "0" => false,
                                "1" => true,
                                s => panic!("unexpected value {:?} for Show", s),
                            });
                        },
                        _ => {},
                    }
                }
                let id = id.unwrap_or_else(|| panic!("missing attribute ID in SubObject"));
                let show = show.unwrap_or_else(|| panic!("missing attribute Show in SubObject"));
                assert!(!subobject_state.contains_key(&id),
                    "duplicate SubObject entry for {:?} in Frame", id);
                subobject_state.insert(id, show);
            },
            XmlEvent::StartElement { ref name, .. } if name.local_name == "Special" => {},
            XmlEvent::StartElement { ref name, .. } if name.local_name == "InteractReaction" => {},
            XmlEvent::StartElement { ref name, .. } if name.local_name == "Emitter" => {},
            XmlEvent::StartElement { ref name, .. } => {
                panic!("unexpected element {:?} in Frame", name);
            },
            XmlEvent::EndElement { ref name, .. } if name.local_name == "Frame" => break,
            XmlEvent::EndElement { ref name, .. } if name.local_name == "EyeSet" => {},
            XmlEvent::EndElement { ref name, .. } if name.local_name == "SubObject" => {},
            XmlEvent::EndElement { ref name, .. } if name.local_name == "Special" => {},
            XmlEvent::EndElement { ref name, .. } if name.local_name == "InteractReaction" => {},
            XmlEvent::EndElement { ref name, .. } if name.local_name == "Emitter" => {},
            XmlEvent::EndElement { ref name, .. } => {
                panic!("unexpected end of element {:?} in Frame", name);
            },
            XmlEvent::CData(..) => {},
            XmlEvent::Comment(..) => {},
            XmlEvent::Characters(..) => {},
            XmlEvent::Whitespace(..) => {},
        }
    }
    Ok(FrameObjects {
        index,
        eye_state,
        subobject_state,
    })
}

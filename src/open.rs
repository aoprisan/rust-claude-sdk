//! Open enums: the API adds block, citation, event and delta types over time, and clients
//! must handle unknown ones gracefully. An enum declared with `#[serde(remote = "Self")]`,
//! `#[serde(tag = "type")]` and a last `#[serde(skip)] Other(serde_json::Value)` variant gets
//! these impls: a known `type` decodes into its variant (and a malformed one is an error),
//! anything else is kept verbatim in `Other` and serialises back unchanged.
macro_rules! open_enum {
    ($ty:ident, [$($tag:literal),+ $(,)?]) => {
        impl $ty {
            /// The `type` tags this crate decodes; any other is kept in `Other`.
            pub const KNOWN_TYPES: &'static [&'static str] = &[$($tag),+];
        }

        impl serde::Serialize for $ty {
            fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                match self {
                    $ty::Other(v) => serde::Serialize::serialize(v, s),
                    known => $ty::serialize(known, s),
                }
            }
        }

        impl<'de> serde::Deserialize<'de> for $ty {
            fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                let v = <serde_json::Value as serde::Deserialize>::deserialize(d)?;
                let known = matches!(v.get("type").and_then(serde_json::Value::as_str), Some($($tag)|+));
                if !known {
                    return Ok($ty::Other(v));
                }
                $ty::deserialize(v).map_err(serde::de::Error::custom)
            }
        }
    };
}

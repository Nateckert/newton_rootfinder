//! What a model says of its iteratives and residuals in a log

use super::json;

/// The names and the texts a model gives its iteratives and its residuals,
/// written at the start of each resolution a log records: see [crate::model::Model::describe]
///
/// An index beyond the dimension of the problem is ignored.
#[derive(Debug)]
pub struct Description {
    iteratives: Vec<Entry>,
    residuals: Vec<Entry>,
    ignored: Entry,
}

/// The name of an iterative or of a residual, and texts about it
#[derive(Debug, Default)]
pub struct Entry {
    name: Option<String>,
    details: Vec<(String, String)>,
}

impl Description {
    pub(crate) fn new(dimension: usize) -> Self {
        Description {
            iteratives: (0..dimension).map(|_| Entry::default()).collect(),
            residuals: (0..dimension).map(|_| Entry::default()).collect(),
            ignored: Entry::default(),
        }
    }

    /// Name the iterative `index`
    pub fn iterative(&mut self, index: usize, name: &str) -> &mut Entry {
        Self::entry(&mut self.iteratives, &mut self.ignored, index, name)
    }

    /// Name the residual `index`, the equation it measures for instance
    pub fn residual(&mut self, index: usize, name: &str) -> &mut Entry {
        Self::entry(&mut self.residuals, &mut self.ignored, index, name)
    }

    fn entry<'e>(
        entries: &'e mut [Entry],
        ignored: &'e mut Entry,
        index: usize,
        name: &str,
    ) -> &'e mut Entry {
        let entry = entries.get_mut(index).unwrap_or(ignored);
        entry.name = Some(name.to_owned());
        entry
    }

    /// The fields an iterative of the start line gets before its settings
    pub(crate) fn write_iterative(&self, line: &mut String, index: usize) {
        if let Some(entry) = self.iteratives.get(index) {
            entry.write(line);
        }
    }

    /// The fields a residual of the start line gets before its settings
    pub(crate) fn write_residual(&self, line: &mut String, index: usize) {
        if let Some(entry) = self.residuals.get(index) {
            entry.write(line);
        }
    }
}

impl Entry {
    /// A text about the iterative or the residual: its unit, the source of an equation...
    pub fn with(&mut self, key: &str, text: &str) -> &mut Entry {
        self.details.push((key.to_owned(), text.to_owned()));
        self
    }

    /// `"name":...,"details":{...},`, the fields given
    fn write(&self, line: &mut String) {
        if let Some(name) = &self.name {
            line.push_str("\"name\":");
            json::string(line, name);
            line.push(',');
        }
        if !self.details.is_empty() {
            line.push_str("\"details\":{");
            for (i, (key, text)) in self.details.iter().enumerate() {
                if i > 0 {
                    line.push(',');
                }
                json::string(line, key);
                line.push(':');
                json::string(line, text);
            }
            line.push_str("},");
        }
    }
}

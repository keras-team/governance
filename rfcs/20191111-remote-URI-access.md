# DataFrameIterator() Support for Remote URIs

| Status        | proposed 										       |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | James DiPadua (james.dipadua@gmail.com)			   |
| **Sponsor**   | 								                       |
| **Updated**   | 2019-11-11                                           |

## Objective

The goal of this effort is to extend the DataFrameIterator to support calls to 
remote URIs, rather than restricting image loading from local filesystems. 

By making this extension, machine learning practioners working with cloud-based
storage (AWS, Azure, etc) or in regulated markets will be able to utilize 
the existing Keras infrastructure.

## Motivation

The `DataFrameIterator` has been a very useful extension to Keras by supporting
an integration with data processing workflows that involve Pandas DataFrames. 

The existing design implies the image paths within the DataFrame are stored 
on the _local system_. With the rapid growth in dataset sizes, image storage has 
moved from local filesystems to cloud-based storage. 

This is particularly acute for enterprise applications in which machine learning 
researchers may not be the "data owners" but merely _one of many stakeholders_, all 
accessing data via cloud interfaces. Further, as practioners move to cloud enviornments,
the local filesystem may be ephemeral: only up during training, which means migrating all
images to the cloud environment before training can begin. Last, for many industry 
practitioners, there are *regulatory conditions which **prohibit** direct storage* of image data. 

Indeed, regulatory conditions motivated making these proposed changes on a Keras fork.
The result was immediate access to the Keras suite, including transfer learning 
via pretrained networks. A production-grade model was developed without infringing upon
regulatory mandates while meeting stakeholder needs.

Because the change is merely one of _file access_ and not system architecture, existing 
users of `flow_from_dataframe()` should experience no impact.

## User Benefit

"Keras eliminates storage needs and removes regulatory obstacles by supporting cloud storage."

Keras developers will have access to images stored _anywhere_ with an `http` or `https` URI. 

This type of extension will allow more practioners access to datasets that are too large
for local storage or due to regulatatory conditions restricted from storing locally.

## Design Proposal
 
### Proposed Option
The current design for the `DataFrameIterator` within Keras Preprocessing, makes the proposed
change fairly small -- similar to the `directory` keyword argument, utilize `is_remote` 
to signal the system should make a remote call, via the Python `requests` library 
to the URI specified in the `x_col` of `iterator.flow_from_dataframe()`.

### Alternative Option
One option that was _initially_ evaluated for supporting remote URI calls was making a 
new `Iterator`. While this option was not technically wrong, it added unncessary complexity 
to the Keras Preprocessing API. 

Rather that introduce what ended up being fairly redundant code, it seems **more** straightforward 
from an API design and usibility perspective to extend the existing `DataFrameIterator` 
with a flag for remote URIs. 

### Example Implementation, by touchpoint
1. Modify the DataFrameIterator `__init__()` to accept a new keyword argument:
```python
is_remote: boolean to flag Keras load_img() to access a remote URI for image load
            default: False
...
def __init__(self,
             dataframe,
             directory=None,
             is_remote=False,
              )
```
2. Modify `DataFrameIterator._filter_valid_filepaths()` and `DataFrameIterator.validate_filename()` to validate remote URI files against a whitelist of URI prefixes
```python
mask = filepaths.apply(validate_filename, 
                                args=(self.input_mode, self.is_remote, self.white_list_formats, self.uri_formats)
```
Where URI formats are defined within `Iterator.py`  as `uri_formats = ('http://', 'https://')`. Note this 
is the existing pattern for image validation. 

3. Within `iterator.py`, modify the call to `keras-preprocessing.utils.load_img` to include the `is_remote` flag:
```python
img = load_img(filepaths[j][d],
                               color_mode=self.color_mode,
                               target_size=self.target_size,
                               interpolation=self.interpolation,
                               remote_uri = self.is_remote)
```
4. Update `keras-preprocessing.utils.load_img` to accept the new `is_remote` argument
```python 
def load_img(path, grayscale=False, color_mode='rgb', target_size=None,
             interpolation='nearest', remote_uri=False)
```

5. Change `keras.preprocessing.image.DataFrameIterator()` to reflect the extensions within Keras Preprocessing
```python
def __init__(self,
             dataframe,
             directory=None,
             is_remote = False,
             ..)
```
6. Last, update `keras.preprocessing.image.flow_from_dataframe()`
```python
def flow_from_dataframe(self,
                        dataframe,
                        directory=None,
                        is_remote=False,
                        )
...
return DataFrameIterator(
            dataframe,
            directory,
            is_remote,
            self,
            x_col=x_col,
            y_col=y_col,
            weight_col=weight_col,
            target_size=target_size,
            color_mode=color_mode,
            classes=classes,
            class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            subset=subset,
            interpolation=interpolation,
            validate_filenames=validate_filenames,
            **kwargs
        )
```

#### Design Justification 

The major benefit to this approach is that existing users of the `DataFrameIterator` will see 
no change. Meaning, the change is fully backwards compatible while introducing little
maintenance overhead. 

Nonetheless, one drawback is that `requests` becomes a dependency for Keras Preprocessing. It is 
arguable that this drawback is largely mitgated because `requests` is a dependency within Keras Applications. 
Further, a similar dependency on external libraries is pre-existing for the `flow_from_dataframe()`
workflow, namely `Pandas` and `Pillow`. 

From a performance perspective, there should be low impact beyond the first batch load. That is, while 
there _is an I/O cost_ related to fetching remote data and migrating it to local process, due to 
multithreading already supported within the Keras `ImageGenerator()`, batches subsequent to the first 
should have no noticable performance impact. This is because subsequent batches will be fetched 
while the model is training on an existing batch. 

At this time, no forward-looking maintenance is likely incurred, as web requests are well-defined and the 
`requests` library is stable and itself well-supported within the Python community. 

One outstanding limitation with this approach is that the system a) makes no notion of a valid URI 
and image format beyond prefix ('http://', 'https://') and file suffix ('png', 'jpg', etc) and b) training
will fail if a remote file fails to load. While a check can be made for an approriate http-response code (`200`), failed requests would result in failed training. This is not a desirable behavior and should be addressed.

## Questions and Discussion Topics

* What is the best way to handle failed requests, be that due to remote-system downtime or incorrect file contents?
	> The current design implies the files are of a specified type but if `PIL` fails to load the file because it's not *really* an image, training fails. **Is that sufficient for remote calls as well?**

* What are approriate unit tests for this design? 

* Should there be a preset "throttle" via a call to `time.sleep({some_default})` in order prevent making too many simultaneous requests? Should the throttle be user-configurable so practioners can choose what is approriate 
for their network conditions? 

* Perhaps worth noting, loading a response would also require `BytesIO`. Should this be considered a potential gotcha?


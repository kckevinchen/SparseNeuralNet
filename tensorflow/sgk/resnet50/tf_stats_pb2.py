# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tf_stats.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='tf_stats.proto',
  package='tensorflow.profiler',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x0etf_stats.proto\x12\x13tensorflow.profiler\"\xa7\x01\n\x0fTfStatsDatabase\x12\x34\n\twith_idle\x18\x04 \x01(\x0b\x32!.tensorflow.profiler.TfStatsTable\x12\x37\n\x0cwithout_idle\x18\x05 \x01(\x0b\x32!.tensorflow.profiler.TfStatsTable\x12\x13\n\x0b\x64\x65vice_type\x18\x06 \x01(\tJ\x04\x08\x01\x10\x02J\x04\x08\x02\x10\x03J\x04\x08\x03\x10\x04\"\x83\x01\n\x0cTfStatsTable\x12;\n\x0ftf_stats_record\x18\x01 \x03(\x0b\x32\".tensorflow.profiler.TfStatsRecord\x12\x19\n\x11host_tf_pprof_key\x18\x02 \x01(\t\x12\x1b\n\x13\x64\x65vice_tf_pprof_key\x18\x03 \x01(\t\"\xbb\x04\n\rTfStatsRecord\x12\x0c\n\x04rank\x18\x01 \x01(\x04\x12\x16\n\x0ehost_or_device\x18\x02 \x01(\t\x12\x0f\n\x07op_type\x18\x03 \x01(\t\x12\x0f\n\x07op_name\x18\x04 \x01(\t\x12\x13\n\x0boccurrences\x18\x05 \x01(\x03\x12\x18\n\x10total_time_in_us\x18\x06 \x01(\x01\x12\x16\n\x0e\x61vg_time_in_us\x18\x07 \x01(\x01\x12\x1d\n\x15total_self_time_in_us\x18\x08 \x01(\x01\x12\x1b\n\x13\x61vg_self_time_in_us\x18\t \x01(\x01\x12*\n\"device_total_self_time_as_fraction\x18\n \x01(\x01\x12\x35\n-device_cumulative_total_self_time_as_fraction\x18\x0b \x01(\x01\x12(\n host_total_self_time_as_fraction\x18\x0c \x01(\x01\x12\x33\n+host_cumulative_total_self_time_as_fraction\x18\r \x01(\x01\x12\x1a\n\x12measured_flop_rate\x18\x0e \x01(\x01\x12\x1a\n\x12measured_memory_bw\x18\x0f \x01(\x01\x12\x1d\n\x15operational_intensity\x18\x10 \x01(\x01\x12\x10\n\x08\x62ound_by\x18\x11 \x01(\t\x12\x10\n\x08is_eager\x18\x12 \x01(\x08\x12\"\n\x1agpu_tensorcore_utilization\x18\x13 \x01(\x01\x62\x06proto3')
)




_TFSTATSDATABASE = _descriptor.Descriptor(
  name='TfStatsDatabase',
  full_name='tensorflow.profiler.TfStatsDatabase',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='with_idle', full_name='tensorflow.profiler.TfStatsDatabase.with_idle', index=0,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='without_idle', full_name='tensorflow.profiler.TfStatsDatabase.without_idle', index=1,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='device_type', full_name='tensorflow.profiler.TfStatsDatabase.device_type', index=2,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=40,
  serialized_end=207,
)


_TFSTATSTABLE = _descriptor.Descriptor(
  name='TfStatsTable',
  full_name='tensorflow.profiler.TfStatsTable',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='tf_stats_record', full_name='tensorflow.profiler.TfStatsTable.tf_stats_record', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='host_tf_pprof_key', full_name='tensorflow.profiler.TfStatsTable.host_tf_pprof_key', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='device_tf_pprof_key', full_name='tensorflow.profiler.TfStatsTable.device_tf_pprof_key', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=210,
  serialized_end=341,
)


_TFSTATSRECORD = _descriptor.Descriptor(
  name='TfStatsRecord',
  full_name='tensorflow.profiler.TfStatsRecord',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='rank', full_name='tensorflow.profiler.TfStatsRecord.rank', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='host_or_device', full_name='tensorflow.profiler.TfStatsRecord.host_or_device', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='op_type', full_name='tensorflow.profiler.TfStatsRecord.op_type', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='op_name', full_name='tensorflow.profiler.TfStatsRecord.op_name', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='occurrences', full_name='tensorflow.profiler.TfStatsRecord.occurrences', index=4,
      number=5, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='total_time_in_us', full_name='tensorflow.profiler.TfStatsRecord.total_time_in_us', index=5,
      number=6, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='avg_time_in_us', full_name='tensorflow.profiler.TfStatsRecord.avg_time_in_us', index=6,
      number=7, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='total_self_time_in_us', full_name='tensorflow.profiler.TfStatsRecord.total_self_time_in_us', index=7,
      number=8, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='avg_self_time_in_us', full_name='tensorflow.profiler.TfStatsRecord.avg_self_time_in_us', index=8,
      number=9, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='device_total_self_time_as_fraction', full_name='tensorflow.profiler.TfStatsRecord.device_total_self_time_as_fraction', index=9,
      number=10, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='device_cumulative_total_self_time_as_fraction', full_name='tensorflow.profiler.TfStatsRecord.device_cumulative_total_self_time_as_fraction', index=10,
      number=11, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='host_total_self_time_as_fraction', full_name='tensorflow.profiler.TfStatsRecord.host_total_self_time_as_fraction', index=11,
      number=12, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='host_cumulative_total_self_time_as_fraction', full_name='tensorflow.profiler.TfStatsRecord.host_cumulative_total_self_time_as_fraction', index=12,
      number=13, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='measured_flop_rate', full_name='tensorflow.profiler.TfStatsRecord.measured_flop_rate', index=13,
      number=14, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='measured_memory_bw', full_name='tensorflow.profiler.TfStatsRecord.measured_memory_bw', index=14,
      number=15, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='operational_intensity', full_name='tensorflow.profiler.TfStatsRecord.operational_intensity', index=15,
      number=16, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bound_by', full_name='tensorflow.profiler.TfStatsRecord.bound_by', index=16,
      number=17, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='is_eager', full_name='tensorflow.profiler.TfStatsRecord.is_eager', index=17,
      number=18, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='gpu_tensorcore_utilization', full_name='tensorflow.profiler.TfStatsRecord.gpu_tensorcore_utilization', index=18,
      number=19, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=344,
  serialized_end=915,
)

_TFSTATSDATABASE.fields_by_name['with_idle'].message_type = _TFSTATSTABLE
_TFSTATSDATABASE.fields_by_name['without_idle'].message_type = _TFSTATSTABLE
_TFSTATSTABLE.fields_by_name['tf_stats_record'].message_type = _TFSTATSRECORD
DESCRIPTOR.message_types_by_name['TfStatsDatabase'] = _TFSTATSDATABASE
DESCRIPTOR.message_types_by_name['TfStatsTable'] = _TFSTATSTABLE
DESCRIPTOR.message_types_by_name['TfStatsRecord'] = _TFSTATSRECORD
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TfStatsDatabase = _reflection.GeneratedProtocolMessageType('TfStatsDatabase', (_message.Message,), dict(
  DESCRIPTOR = _TFSTATSDATABASE,
  __module__ = 'tf_stats_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.profiler.TfStatsDatabase)
  ))
_sym_db.RegisterMessage(TfStatsDatabase)

TfStatsTable = _reflection.GeneratedProtocolMessageType('TfStatsTable', (_message.Message,), dict(
  DESCRIPTOR = _TFSTATSTABLE,
  __module__ = 'tf_stats_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.profiler.TfStatsTable)
  ))
_sym_db.RegisterMessage(TfStatsTable)

TfStatsRecord = _reflection.GeneratedProtocolMessageType('TfStatsRecord', (_message.Message,), dict(
  DESCRIPTOR = _TFSTATSRECORD,
  __module__ = 'tf_stats_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.profiler.TfStatsRecord)
  ))
_sym_db.RegisterMessage(TfStatsRecord)


# @@protoc_insertion_point(module_scope)

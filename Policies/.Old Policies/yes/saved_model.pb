ПЗ
║К
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8ил
Ж
sequential/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namesequential/dense_2/bias

+sequential/dense_2/bias/Read/ReadVariableOpReadVariableOpsequential/dense_2/bias*
_output_shapes
:*
dtype0
О
sequential/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d**
shared_namesequential/dense_2/kernel
З
-sequential/dense_2/kernel/Read/ReadVariableOpReadVariableOpsequential/dense_2/kernel*
_output_shapes

:d*
dtype0
Ж
sequential/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*(
shared_namesequential/dense_1/bias

+sequential/dense_1/bias/Read/ReadVariableOpReadVariableOpsequential/dense_1/bias*
_output_shapes
:d*
dtype0
О
sequential/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd**
shared_namesequential/dense_1/kernel
З
-sequential/dense_1/kernel/Read/ReadVariableOpReadVariableOpsequential/dense_1/kernel*
_output_shapes

:dd*
dtype0
В
sequential/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_namesequential/dense/bias
{
)sequential/dense/bias/Read/ReadVariableOpReadVariableOpsequential/dense/bias*
_output_shapes
:d*
dtype0
Л
sequential/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	цd*(
shared_namesequential/dense/kernel
Д
+sequential/dense/kernel/Read/ReadVariableOpReadVariableOpsequential/dense/kernel*
_output_shapes
:	цd*
dtype0
d
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0	

NoOpNoOp
п
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ъ
valueрB▌ B╓
╪

train_step
metadata
model_variables
_all_assets

signatures
#_self_saveable_object_factories

action
distribution
	get_initial_state

get_metadata
get_train_step*
GA
VARIABLE_VALUEVariable%train_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
0
1
2
3
4
5*

_wrapped_policy*
* 
* 

trace_0
trace_1* 

trace_0* 

trace_0* 
* 
* 
]W
VARIABLE_VALUEsequential/dense/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEsequential/dense/bias,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEsequential/dense_1/kernel,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEsequential/dense_1/bias,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEsequential/dense_2/kernel,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEsequential/dense_2/bias,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
9

_q_network
#_self_saveable_object_factories*
* 
* 
* 
* 
¤
	variables
trainable_variables
regularization_losses
	keras_api
_layer_state_is_list
_sequential_layers
_layer_has_state
# _self_saveable_object_factories
!__call__
*"&call_and_return_all_conditional_losses*
* 
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
* 
╕
#non_trainable_variables

$layers
%metrics
&layer_regularization_losses
'layer_metrics
	variables
trainable_variables
regularization_losses
#(_self_saveable_object_factories
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*
* 

)0
*1
+2*
* 
* 
* 
* 
* 

)0
*1
+2*
* 
* 
* 
* 
╦
,	variables
-trainable_variables
.regularization_losses
/	keras_api

kernel
bias
#0_self_saveable_object_factories
1__call__
*2&call_and_return_all_conditional_losses*
╦
3	variables
4trainable_variables
5regularization_losses
6	keras_api

kernel
bias
#7_self_saveable_object_factories
8__call__
*9&call_and_return_all_conditional_losses*
╦
:	variables
;trainable_variables
<regularization_losses
=	keras_api

kernel
bias
#>_self_saveable_object_factories
?__call__
*@&call_and_return_all_conditional_losses*

0
1*

0
1*
* 
╕
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
,	variables
-trainable_variables
.regularization_losses
#F_self_saveable_object_factories
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*
* 
* 
* 

0
1*

0
1*
* 
╕
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
3	variables
4trainable_variables
5regularization_losses
#L_self_saveable_object_factories
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses*
* 
* 
* 

0
1*

0
1*
* 
╕
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
:	variables
;trainable_variables
<regularization_losses
#R_self_saveable_object_factories
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╧
StatefulPartitionedCallStatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOp+sequential/dense/kernel/Read/ReadVariableOp)sequential/dense/bias/Read/ReadVariableOp-sequential/dense_1/kernel/Read/ReadVariableOp+sequential/dense_1/bias/Read/ReadVariableOp-sequential/dense_2/kernel/Read/ReadVariableOp+sequential/dense_2/bias/Read/ReadVariableOpConst*
Tin
2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__traced_save_2405909
└
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameVariablesequential/dense/kernelsequential/dense/biassequential/dense_1/kernelsequential/dense_1/biassequential/dense_2/kernelsequential/dense_2/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference__traced_restore_2405940╤Ў
╛
7
%__inference_get_initial_state_2395249

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
■D
Й
)__inference_polymorphic_action_fn_2395246
	step_type

reward
discount
observationB
/sequential_dense_matmul_readvariableop_resource:	цd>
0sequential_dense_biasadd_readvariableop_resource:dC
1sequential_dense_1_matmul_readvariableop_resource:dd@
2sequential_dense_1_biasadd_readvariableop_resource:dC
1sequential_dense_2_matmul_readvariableop_resource:d@
2sequential_dense_2_biasadd_readvariableop_resource:
identityИв'sequential/dense/BiasAdd/ReadVariableOpв&sequential/dense/MatMul/ReadVariableOpв)sequential/dense_1/BiasAdd/ReadVariableOpв(sequential/dense_1/MatMul/ReadVariableOpв)sequential/dense_2/BiasAdd/ReadVariableOpв(sequential/dense_2/MatMul/ReadVariableOpЧ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	цd*
dtype0Р
sequential/dense/MatMulMatMulobservation.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dФ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0й
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dr
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:         dЪ
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0м
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dШ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0п
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dv
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         dЪ
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0о
sequential/dense_2/MatMulMatMul%sequential/dense_1/Relu:activations:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ш
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0п
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         l
!Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
         а
Categorical/mode/ArgMaxArgMax#sequential/dense_2/BiasAdd:output:0*Categorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:         |
Categorical/mode/CastCast Categorical/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:         T
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : T
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB c
Deterministic/sample/ShapeShapeCategorical/mode/Cast:y:0*
T0*
_output_shapes
:\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╕
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB о
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : З
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:в
 Deterministic/sample/BroadcastToBroadcastToCategorical/mode/Cast:y:0$Deterministic/sample/concat:output:0*
T0*'
_output_shapes
:         u
Deterministic/sample/Shape_1Shape)Deterministic/sample/BroadcastTo:output:0*
T0*
_output_shapes
:t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:└
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_1:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ▀
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:и
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0*#
_output_shapes
:         Y
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :Ч
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:         Q
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:         ╟
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 \
IdentityIdentityclip_by_value:z:0^NoOp*
T0*#
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:         :         :         :         ц: : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp:N J
#
_output_shapes
:         
#
_user_specified_name	step_type:KG
#
_output_shapes
:         
 
_user_specified_namereward:MI
#
_output_shapes
:         
"
_user_specified_name
discount:UQ
(
_output_shapes
:         ц
%
_user_specified_nameobservation
╪E
▒
)__inference_polymorphic_action_fn_2395353
time_step_step_type
time_step_reward
time_step_discount
time_step_observationB
/sequential_dense_matmul_readvariableop_resource:	цd>
0sequential_dense_biasadd_readvariableop_resource:dC
1sequential_dense_1_matmul_readvariableop_resource:dd@
2sequential_dense_1_biasadd_readvariableop_resource:dC
1sequential_dense_2_matmul_readvariableop_resource:d@
2sequential_dense_2_biasadd_readvariableop_resource:
identityИв'sequential/dense/BiasAdd/ReadVariableOpв&sequential/dense/MatMul/ReadVariableOpв)sequential/dense_1/BiasAdd/ReadVariableOpв(sequential/dense_1/MatMul/ReadVariableOpв)sequential/dense_2/BiasAdd/ReadVariableOpв(sequential/dense_2/MatMul/ReadVariableOpЧ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	цd*
dtype0Ъ
sequential/dense/MatMulMatMultime_step_observation.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dФ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0й
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dr
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:         dЪ
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0м
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dШ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0п
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dv
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         dЪ
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0о
sequential/dense_2/MatMulMatMul%sequential/dense_1/Relu:activations:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ш
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0п
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         l
!Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
         а
Categorical/mode/ArgMaxArgMax#sequential/dense_2/BiasAdd:output:0*Categorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:         |
Categorical/mode/CastCast Categorical/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:         T
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : T
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB c
Deterministic/sample/ShapeShapeCategorical/mode/Cast:y:0*
T0*
_output_shapes
:\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╕
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB о
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : З
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:в
 Deterministic/sample/BroadcastToBroadcastToCategorical/mode/Cast:y:0$Deterministic/sample/concat:output:0*
T0*'
_output_shapes
:         u
Deterministic/sample/Shape_1Shape)Deterministic/sample/BroadcastTo:output:0*
T0*
_output_shapes
:t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:└
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_1:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ▀
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:и
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0*#
_output_shapes
:         Y
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :Ч
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:         Q
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:         ╟
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 \
IdentityIdentityclip_by_value:z:0^NoOp*
T0*#
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:         :         :         :         ц: : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp:X T
#
_output_shapes
:         
-
_user_specified_nametime_step/step_type:UQ
#
_output_shapes
:         
*
_user_specified_nametime_step/reward:WS
#
_output_shapes
:         
,
_user_specified_nametime_step/discount:_[
(
_output_shapes
:         ц
/
_user_specified_nametime_step/observation
]

__inference_<lambda>_2395290*(
_construction_contextkEagerRuntime*
_input_shapes 
╧ 
ъ
#__inference__traced_restore_2405940
file_prefix#
assignvariableop_variable:	 =
*assignvariableop_1_sequential_dense_kernel:	цd6
(assignvariableop_2_sequential_dense_bias:d>
,assignvariableop_3_sequential_dense_1_kernel:dd8
*assignvariableop_4_sequential_dense_1_bias:d>
,assignvariableop_5_sequential_dense_2_kernel:d8
*assignvariableop_6_sequential_dense_2_bias:

identity_8ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_2вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6╚
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ю
valueфBсB%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHА
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B ╞
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*4
_output_shapes"
 ::::::::*
dtypes

2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:Д
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_1AssignVariableOp*assignvariableop_1_sequential_dense_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_2AssignVariableOp(assignvariableop_2_sequential_dense_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_3AssignVariableOp,assignvariableop_3_sequential_dense_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_4AssignVariableOp*assignvariableop_4_sequential_dense_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_5AssignVariableOp,assignvariableop_5_sequential_dense_2_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_6AssignVariableOp*assignvariableop_6_sequential_dense_2_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ы

Identity_7Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_8IdentityIdentity_7:output:0^NoOp_1*
T0*
_output_shapes
: ┘
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6*"
_acd_function_control_output(*
_output_shapes
 "!

identity_8Identity_8:output:0*#
_input_shapes
: : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_6:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Э)
п
/__inference_polymorphic_distribution_fn_2395425
	step_type

reward
discount
observationB
/sequential_dense_matmul_readvariableop_resource:	цd>
0sequential_dense_biasadd_readvariableop_resource:dC
1sequential_dense_1_matmul_readvariableop_resource:dd@
2sequential_dense_1_biasadd_readvariableop_resource:dC
1sequential_dense_2_matmul_readvariableop_resource:d@
2sequential_dense_2_biasadd_readvariableop_resource:
identity

identity_1

identity_2Ив'sequential/dense/BiasAdd/ReadVariableOpв&sequential/dense/MatMul/ReadVariableOpв)sequential/dense_1/BiasAdd/ReadVariableOpв(sequential/dense_1/MatMul/ReadVariableOpв)sequential/dense_2/BiasAdd/ReadVariableOpв(sequential/dense_2/MatMul/ReadVariableOpЧ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	цd*
dtype0Р
sequential/dense/MatMulMatMulobservation.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dФ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0й
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dr
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:         dЪ
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0м
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dШ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0п
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dv
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         dЪ
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0о
sequential/dense_2/MatMulMatMul%sequential/dense_1/Relu:activations:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ш
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0п
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         l
!Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
         а
Categorical/mode/ArgMaxArgMax#sequential/dense_2/BiasAdd:output:0*Categorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:         |
Categorical/mode/CastCast Categorical/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:         T
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : T
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : ╟
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 Y
IdentityIdentityDeterministic/atol:output:0^NoOp*
T0*
_output_shapes
: f

Identity_1IdentityCategorical/mode/Cast:y:0^NoOp*
T0*#
_output_shapes
:         [

Identity_2IdentityDeterministic/rtol:output:0^NoOp*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:         :         :         :         ц: : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp:N J
#
_output_shapes
:         
#
_user_specified_name	step_type:KG
#
_output_shapes
:         
 
_user_specified_namereward:MI
#
_output_shapes
:         
"
_user_specified_name
discount:UQ
(
_output_shapes
:         ц
%
_user_specified_nameobservation
Ў
c
__inference_<lambda>_2395254!
readvariableop_resource:	 
identity	ИвReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	W
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 T
IdentityIdentityReadVariableOp:value:0^NoOp*
T0	*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2 
ReadVariableOpReadVariableOp
ч
ш
 __inference__traced_save_2405909
file_prefix'
#savev2_variable_read_readvariableop	6
2savev2_sequential_dense_kernel_read_readvariableop4
0savev2_sequential_dense_bias_read_readvariableop8
4savev2_sequential_dense_1_kernel_read_readvariableop6
2savev2_sequential_dense_1_bias_read_readvariableop8
4savev2_sequential_dense_2_kernel_read_readvariableop6
2savev2_sequential_dense_2_bias_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ┼
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ю
valueфBсB%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH}
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B Ц
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop2savev2_sequential_dense_kernel_read_readvariableop0savev2_sequential_dense_bias_read_readvariableop4savev2_sequential_dense_1_kernel_read_readvariableop2savev2_sequential_dense_1_bias_read_readvariableop4savev2_sequential_dense_2_kernel_read_readvariableop2savev2_sequential_dense_2_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes

2	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*J
_input_shapes9
7: : :	цd:d:dd:d:d:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :%!

_output_shapes
:	цd: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::

_output_shapes
: "╡	J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:ЎN
Є

train_step
metadata
model_variables
_all_assets

signatures
#_self_saveable_object_factories

action
distribution
	get_initial_state

get_metadata
get_train_step"
_generic_user_object
:	 (2Variable
 "
trackable_dict_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
5
_wrapped_policy"
trackable_dict_wrapper
"
signature_map
 "
trackable_dict_wrapper
┴
trace_0
trace_12К
)__inference_polymorphic_action_fn_2395246
)__inference_polymorphic_action_fn_2395353▒
к▓ж
FullArgSpec(
args Ъ
j	time_step
jpolicy_state
varargs
 
varkw
 
defaultsв
в 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ztrace_0ztrace_1
В
trace_02х
/__inference_polymorphic_distribution_fn_2395425▒
к▓ж
FullArgSpec(
args Ъ
j	time_step
jpolicy_state
varargs
 
varkw
 
defaultsв
в 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ztrace_0
у
trace_02╞
%__inference_get_initial_state_2395249Ь
Х▓С
FullArgSpec
argsЪ
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ztrace_0
▓Bп
__inference_<lambda>_2395290"О
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▓Bп
__inference_<lambda>_2395254"О
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
*:(	цd2sequential/dense/kernel
#:!d2sequential/dense/bias
+:)dd2sequential/dense_1/kernel
%:#d2sequential/dense_1/bias
+:)d2sequential/dense_2/kernel
%:#2sequential/dense_2/bias
S

_q_network
#_self_saveable_object_factories"
_generic_user_object
ОBЛ
)__inference_polymorphic_action_fn_2395246	step_typerewarddiscountobservation"▒
к▓ж
FullArgSpec(
args Ъ
j	time_step
jpolicy_state
varargs
 
varkw
 
defaultsв
в 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╢B│
)__inference_polymorphic_action_fn_2395353time_step/step_typetime_step/rewardtime_step/discounttime_step/observation"▒
к▓ж
FullArgSpec(
args Ъ
j	time_step
jpolicy_state
varargs
 
varkw
 
defaultsв
в 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ФBС
/__inference_polymorphic_distribution_fn_2395425	step_typerewarddiscountobservation"▒
к▓ж
FullArgSpec(
args Ъ
j	time_step
jpolicy_state
varargs
 
varkw
 
defaultsв
в 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╫B╘
%__inference_get_initial_state_2395249
batch_size"Ь
Х▓С
FullArgSpec
argsЪ
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ч
	variables
trainable_variables
regularization_losses
	keras_api
_layer_state_is_list
_sequential_layers
_layer_has_state
# _self_saveable_object_factories
!__call__
*"&call_and_return_all_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
╥
#non_trainable_variables

$layers
%metrics
&layer_regularization_losses
'layer_metrics
	variables
trainable_variables
regularization_losses
#(_self_saveable_object_factories
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╪2╒╥
╦▓╟
FullArgSpec&
argsЪ
jinputs
jnetwork_state
varargs
 
varkwjkwargs
defaultsв
в 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
╪2╒╥
╦▓╟
FullArgSpec&
argsЪ
jinputs
jnetwork_state
varargs
 
varkwjkwargs
defaultsв
в 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
 "
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
х
,	variables
-trainable_variables
.regularization_losses
/	keras_api

kernel
bias
#0_self_saveable_object_factories
1__call__
*2&call_and_return_all_conditional_losses"
_generic_user_object
х
3	variables
4trainable_variables
5regularization_losses
6	keras_api

kernel
bias
#7_self_saveable_object_factories
8__call__
*9&call_and_return_all_conditional_losses"
_generic_user_object
х
:	variables
;trainable_variables
<regularization_losses
=	keras_api

kernel
bias
#>_self_saveable_object_factories
?__call__
*@&call_and_return_all_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
╥
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
,	variables
-trainable_variables
.regularization_losses
#F_self_saveable_object_factories
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
╥
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
3	variables
4trainable_variables
5regularization_losses
#L_self_saveable_object_factories
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
╥
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
:	variables
;trainable_variables
<regularization_losses
#R_self_saveable_object_factories
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper;
__inference_<lambda>_2395254в

в 
к "К 	4
__inference_<lambda>_2395290в

в 
к "к R
%__inference_get_initial_state_2395249)"в
в
К

batch_size 
к "в ь
)__inference_polymorphic_action_fn_2395246╛▀в█
╙в╧
╟▓├
TimeStep,
	step_typeК
	step_type         &
rewardК
reward         *
discountК
discount         5
observation&К#
observation         ц
в 
к "R▓O

PolicyStep&
actionК
action         
stateв 
infoв Ф
)__inference_polymorphic_action_fn_2395353цЗвГ
√вў
я▓ы
TimeStep6
	step_type)К&
time_step/step_type         0
reward&К#
time_step/reward         4
discount(К%
time_step/discount         ?
observation0К-
time_step/observation         ц
в 
к "R▓O

PolicyStep&
actionК
action         
stateв 
infoв ╧
/__inference_polymorphic_distribution_fn_2395425Ы▀в█
╙в╧
╟▓├
TimeStep,
	step_typeК
	step_type         &
rewardК
reward         *
discountК
discount         5
observation&К#
observation         ц
в 
к "о▓к

PolicyStepА
actionїТё╜в╣
`
Bк?

atolК 

locК         

rtolК 
JкG

allow_nan_statsp

namejDeterministic_1

validate_argsp 
в
j
parameters
в 
в
jname+tfp.distributions.Deterministic_ACTTypeSpec 
stateв 
infoв 
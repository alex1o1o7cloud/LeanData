import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_meet_conditions_l491_49116

/-- The circumference of the circular track in meters -/
noncomputable def track_circumference : ℝ := 360

/-- The speed of runner A -/
noncomputable def speed_A : ℝ := 1

/-- The speed of runner B relative to A's speed -/
noncomputable def speed_B : ℝ := 4 * speed_A

/-- The distance A runs before B and C start -/
noncomputable def distance_A_before_BC_start : ℝ := 90

/-- The time it takes for A and B to meet for the first time -/
noncomputable def time_AB_meet : ℝ := (track_circumference / 2) / (speed_A + speed_B)

/-- The distance C is behind A and B when they first meet -/
noncomputable def distance_C_behind_AB : ℝ := track_circumference / 2

/-- The speed of runner C (to be determined) -/
noncomputable def speed_C : ℝ := sorry

/-- The time it takes for A and C to meet for the first time after B and C start -/
noncomputable def time_AC_meet : ℝ := (track_circumference / 2) / (speed_A + speed_C)

/-- The distance B is behind A and C when they first meet -/
noncomputable def distance_B_behind_AC : ℝ := track_circumference / 2

theorem runners_meet_conditions :
  ∃ (speed_C : ℝ),
    speed_C > 0 ∧
    distance_C_behind_AB = track_circumference / 2 ∧
    distance_B_behind_AC = track_circumference / 2 ∧
    distance_A_before_BC_start = 90 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_meet_conditions_l491_49116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_side_length_l491_49121

/-- Given a triangle ABC with two sides of lengths 8 and 15, 
    if cos(2A) + cos(2B) + cos(2C) = 0, 
    then the maximum length of the third side is √(289 - 120√2) -/
theorem triangle_max_side_length (A B C : ℝ) (a b c : ℝ) :
  a = 8 → b = 15 →
  Real.cos (2 * A) + Real.cos (2 * B) + Real.cos (2 * C) = 0 →
  c ≤ Real.sqrt (289 - 120 * Real.sqrt 2) := by
  sorry

#check triangle_max_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_side_length_l491_49121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_problem_l491_49198

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the cross product operation
variable (cross : V → V → V)

-- Define the transformation T
variable (T : V → V)

-- Define the properties of T
variable (h1 : ∀ (a b : ℝ) (v w : V), T (a • v + b • w) = a • T v + b • T w)
variable (h2 : ∀ (v w : V), T (cross v w) = cross (T v) (T v))

-- Define basis vectors
variable (e₁ e₂ e₃ : V)

-- Define the properties of T on specific vectors
variable (h3 : T (6 • e₁ + 6 • e₂ + 3 • e₃) = 4 • e₁ + (-1) • e₂ + 8 • e₃)
variable (h4 : T ((-6) • e₁ + 3 • e₂ + 6 • e₃) = 4 • e₁ + 8 • e₂ + (-1) • e₃)

theorem transformation_problem :
  T (3 • e₁ + 9 • e₂ + 12 • e₃) = 7 • e₁ + 8 • e₂ + 11 • e₃ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_problem_l491_49198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_next_game_shots_l491_49168

/-- Represents the shooting statistics for a basketball player --/
structure ShootingStats where
  initialMade : ℕ
  initialTotal : ℕ
  additionalShots : ℕ
  initialAverage : ℚ
  newAverage : ℚ

/-- Calculates the number of shots made in the next game --/
def shotsInNextGame (stats : ShootingStats) : ℕ :=
  ⌊(stats.newAverage * (stats.initialTotal + stats.additionalShots : ℚ) - stats.initialMade)⌋.toNat

/-- Theorem stating that given the initial conditions and new average, 
    the number of shots made in the next game is 9 --/
theorem next_game_shots (stats : ShootingStats) 
  (h1 : stats.initialMade = 24)
  (h2 : stats.initialTotal = 60)
  (h3 : stats.additionalShots = 15)
  (h4 : stats.initialAverage = 2/5)
  (h5 : stats.newAverage = 9/20) :
  shotsInNextGame stats = 9 := by
  sorry

#eval shotsInNextGame { initialMade := 24, initialTotal := 60, additionalShots := 15, initialAverage := 2/5, newAverage := 9/20 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_next_game_shots_l491_49168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y₁_gt_y₂_range_l491_49169

/-- Linear function y₁ -/
noncomputable def y₁ (x : ℝ) : ℝ := x - 3

/-- Inverse proportion function y₂ -/
noncomputable def y₂ (x : ℝ) : ℝ := 4 / x

/-- The graphs of y₁ and y₂ intersect at two points -/
axiom intersect_points : ∃ (a b : ℝ), a ≠ b ∧ y₁ a = y₂ a ∧ y₁ b = y₂ b

/-- Theorem stating the range of x values where y₁ > y₂ -/
theorem y₁_gt_y₂_range (x : ℝ) : 
  y₁ x > y₂ x ↔ ((-1 < x ∧ x < 0) ∨ x > 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y₁_gt_y₂_range_l491_49169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_interior_angle_l491_49179

/-- The measure of an interior angle of a regular hexagon is 120 degrees. -/
def interior_angle_regular_hexagon : ℚ := 120

/-- A regular hexagon has 6 sides. -/
def regular_hexagon_sides : ℕ := 6

/-- Formula for the sum of interior angles of a polygon with n sides. -/
def sum_interior_angles (n : ℕ) : ℚ := (n - 2) * 180

/-- The measure of a single interior angle in a regular polygon. -/
def interior_angle_measure (n : ℕ) : ℚ :=
  (sum_interior_angles n) / n

/-- Theorem stating that the interior angle of a regular hexagon is 120 degrees. -/
theorem regular_hexagon_interior_angle :
  interior_angle_measure regular_hexagon_sides = interior_angle_regular_hexagon :=
by
  -- Unfold definitions
  unfold interior_angle_measure
  unfold sum_interior_angles
  unfold regular_hexagon_sides
  unfold interior_angle_regular_hexagon
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_interior_angle_l491_49179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_theorem_l491_49153

/-- Represents the capacity of a water tank in litres. -/
def TankCapacity : ℝ → Prop := sorry

/-- Represents the time in hours it takes to empty the tank through the leak. -/
def LeakEmptyTime : ℝ → Prop := sorry

/-- Represents the fill rate of Inlet Pipe A in litres per minute. -/
def InletPipeARate : ℝ → Prop := sorry

/-- Represents the fill rate of Inlet Pipe B in litres per minute. -/
def InletPipeBRate : ℝ → Prop := sorry

/-- Represents the time in hours it takes to empty the tank with both inlet pipes open and the leak active. -/
def BothPipesLeakEmptyTime : ℝ → Prop := sorry

theorem tank_capacity_theorem (C : ℝ) :
  TankCapacity C →
  LeakEmptyTime 3 →
  InletPipeARate 6 →
  InletPipeBRate 4 →
  BothPipesLeakEmptyTime 12 →
  C = 1440 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_theorem_l491_49153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_production_calculation_l491_49190

/-- Given initial production conditions and new production conditions, 
    calculate the number of articles produced under new conditions -/
theorem production_calculation 
  (a b c d f p q r g : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : d > 0) (h5 : f > 0) (h6 : p > 0) 
  (h7 : q > 0) (h8 : r > 0) (h9 : g > 0) : 
  (p * q * r * d * g) / (a * b * c * f) = (p * q * r * d * g) / (a * b * c * f) := by
  -- The proof goes here
  sorry

#check production_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_production_calculation_l491_49190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_condition_l491_49178

-- Define the vector type
def Vector2D := ℝ × ℝ

-- Define vector addition
def add_vectors (v w : Vector2D) : Vector2D :=
  (v.1 + w.1, v.2 + w.2)

-- Define vector subtraction
def sub_vectors (v w : Vector2D) : Vector2D :=
  (v.1 - w.1, v.2 - w.2)

-- Define parallel vectors
def parallel (v w : Vector2D) : Prop :=
  ∃ (k : ℝ), v = (k * w.1, k * w.2)

-- Theorem statement
theorem vector_parallel_condition (x : ℝ) :
  let a : Vector2D := (1, 1)
  let b : Vector2D := (2, x)
  parallel (add_vectors a b) (sub_vectors a b) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_condition_l491_49178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_speed_l491_49138

/-- Calculates the average speed of a round trip given the distance and speeds -/
noncomputable def average_speed (n : ℝ) : ℝ :=
  let feet_per_mile : ℝ := 5280
  let minutes_per_hour : ℝ := 60
  let north_speed : ℝ := 1 / 5  -- miles per minute
  let south_speed : ℝ := 3      -- miles per minute
  let distance : ℝ := n / feet_per_mile
  let total_distance : ℝ := 2 * distance
  let north_time : ℝ := distance / north_speed
  let south_time : ℝ := distance / south_speed
  let total_time : ℝ := (north_time + south_time) / minutes_per_hour
  total_distance / total_time

/-- Theorem stating that the average speed for the round trip is 135 miles per hour -/
theorem round_trip_speed (n : ℝ) (h : n > 0) : average_speed n = 135 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_speed_l491_49138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_for_even_function_l491_49192

def det (a b c d : ℝ) : ℝ := a * d - b * c

noncomputable def f (x : ℝ) : ℝ := det 2 (2 * Real.sin x) (Real.sqrt 3) (Real.cos x)

theorem min_phi_for_even_function :
  ∃ (φ : ℝ), φ > 0 ∧
  (∀ x, f (x + φ) = f (-x + φ)) ∧
  (∀ ψ, 0 < ψ ∧ ψ < φ → ∃ y, f (y + ψ) ≠ f (-y + ψ)) ∧
  φ = 2 * Real.pi / 3 := by
  sorry

#check min_phi_for_even_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_for_even_function_l491_49192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ecommerce_model_properties_l491_49137

/-- Represents the e-commerce company's product pricing and sales model -/
structure ECommerceModel where
  costPrice : ℚ
  maxPrice : ℚ
  basePrice : ℚ
  baseSales : ℚ
  priceReductionEffect : ℚ

/-- The e-commerce model satisfies the given conditions -/
def validModel (m : ECommerceModel) : Prop :=
  m.costPrice = 70 ∧
  m.maxPrice = 99 ∧
  m.basePrice = 110 ∧
  m.baseSales = 20 ∧
  m.priceReductionEffect = 2

/-- Calculates the selling price for a given sales volume -/
def sellingPrice (m : ECommerceModel) (salesVolume : ℚ) : ℚ :=
  m.basePrice - (salesVolume - m.baseSales) / m.priceReductionEffect

/-- Represents the relationship between sales volume and selling price -/
def salesVolumeFunction (m : ECommerceModel) (x : ℚ) : ℚ :=
  -m.priceReductionEffect * x + (m.basePrice * m.priceReductionEffect + m.baseSales)

/-- Calculates the daily profit for a given selling price -/
def dailyProfit (m : ECommerceModel) (sellingPrice : ℚ) : ℚ :=
  (sellingPrice - m.costPrice) * (salesVolumeFunction m sellingPrice)

theorem ecommerce_model_properties (m : ECommerceModel) (h : validModel m) :
  sellingPrice m 30 = 105 ∧
  (∀ x, m.costPrice ≤ x ∧ x ≤ m.maxPrice → salesVolumeFunction m x = -2 * x + 240) ∧
  (∃ x, m.costPrice ≤ x ∧ x ≤ m.maxPrice ∧ dailyProfit m x = 1200 ∧ x = 90) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ecommerce_model_properties_l491_49137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_sequence_properties_l491_49156

/-- Ink length of a figure in the pentagon sequence -/
def ink_length (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 5
  | n+1 => ink_length n + 5 * (n+1) - 2 * n

/-- Difference in ink length between two consecutive figures -/
def ink_length_diff (n : ℕ) : ℕ :=
  ink_length (n + 1) - ink_length n

theorem pentagon_sequence_properties :
  (ink_length 4 = 38) ∧
  (ink_length_diff 8 = 29) ∧
  (ink_length 100 = 15350) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_sequence_properties_l491_49156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_l491_49100

/-- Represents a digit from 0 to 9 -/
def Digit := Fin 10

/-- A function that maps Chinese characters to digits -/
def ChineseToDigit : Char → Digit := sorry

/-- The equation holds true -/
axiom equation_holds :
  (ChineseToDigit '学').val * (ChineseToDigit '数').val * (ChineseToDigit '学').val * 7 *
  (ChineseToDigit '迎').val * (ChineseToDigit '春').val * (ChineseToDigit '杯').val =
  (ChineseToDigit '加').val * 1000 + (ChineseToDigit '油').val * 100 +
  (ChineseToDigit '加').val * 10 + (ChineseToDigit '油').val * (ChineseToDigit '吧').val

/-- Each Chinese character represents a unique digit -/
axiom unique_digits : ∀ c1 c2 : Char, c1 ≠ c2 → ChineseToDigit c1 ≠ ChineseToDigit c2

/-- No Chinese character represents 7 -/
axiom no_seven : ∀ c : Char, (ChineseToDigit c).val ≠ 7

/-- "迎", "春", and "杯" are not equal to 1 -/
axiom not_one :
  (ChineseToDigit '迎').val ≠ 1 ∧ (ChineseToDigit '春').val ≠ 1 ∧ (ChineseToDigit '杯').val ≠ 1

/-- The sum of the digits represented by "迎", "春", and "杯" is 18 -/
theorem sum_of_digits :
  (ChineseToDigit '迎').val + (ChineseToDigit '春').val + (ChineseToDigit '杯').val = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_l491_49100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l491_49175

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-10 * x^2 - 11 * x + 6)

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f x = y} = {x : ℝ | -3/2 ≤ x ∧ x ≤ 2/5} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l491_49175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lamp_cycle_theorem_l491_49129

/-- Represents the state of a lamp (ON or OFF) -/
inductive LampState
| On
| Off

/-- Represents the configuration of n lamps -/
def LampConfig (n : ℕ) := Fin n → LampState

/-- Performs one step of the lamp changing process -/
def step (n : ℕ) (config : LampConfig n) : LampConfig n :=
  sorry

/-- Returns true if all lamps in the configuration are ON -/
def allOn (n : ℕ) (config : LampConfig n) : Prop :=
  ∀ i, config i = LampState.On

/-- The number of steps needed to return to all lamps ON -/
def M (n : ℕ) : ℕ :=
  sorry

/-- Main theorem about the lamp cycle -/
theorem lamp_cycle_theorem (n : ℕ) (h : n > 1) :
  ∃ (M : ℕ), M > 0 ∧ 
    (let initialConfig : LampConfig n := λ _ ↦ LampState.On
     allOn n ((step n)^[M] initialConfig)) ∧
    (∀ k : ℕ, n = 2^k → M = n^2 - 1) ∧
    (∀ k : ℕ, n = 2^k + 1 → M = n^2 - n + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lamp_cycle_theorem_l491_49129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_arrangement_l491_49118

-- Define a type for the triangular arrangement
structure TriangularArrangement where
  top : Nat
  bottom_left : Nat
  bottom_right : Nat
  middle_left : Nat
  middle_right : Nat
  bottom : Nat

-- Define a predicate to check if an arrangement is valid
def is_valid_arrangement (arr : TriangularArrangement) : Prop :=
  arr.top ∈ [1, 2, 3, 4, 5, 6] ∧
  arr.bottom_left ∈ [1, 2, 3, 4, 5, 6] ∧
  arr.bottom_right ∈ [1, 2, 3, 4, 5, 6] ∧
  arr.middle_left ∈ [1, 2, 3, 4, 5, 6] ∧
  arr.middle_right ∈ [1, 2, 3, 4, 5, 6] ∧
  arr.bottom ∈ [1, 2, 3, 4, 5, 6] ∧
  arr.top ≠ arr.bottom_left ∧
  arr.top ≠ arr.bottom_right ∧
  arr.top ≠ arr.middle_left ∧
  arr.top ≠ arr.middle_right ∧
  arr.top ≠ arr.bottom ∧
  arr.bottom_left ≠ arr.bottom_right ∧
  arr.bottom_left ≠ arr.middle_left ∧
  arr.bottom_left ≠ arr.middle_right ∧
  arr.bottom_left ≠ arr.bottom ∧
  arr.bottom_right ≠ arr.middle_left ∧
  arr.bottom_right ≠ arr.middle_right ∧
  arr.bottom_right ≠ arr.bottom ∧
  arr.middle_left ≠ arr.middle_right ∧
  arr.middle_left ≠ arr.bottom ∧
  arr.middle_right ≠ arr.bottom

-- Define a predicate to check if the sums are correct
def sums_are_ten (arr : TriangularArrangement) : Prop :=
  arr.top + arr.middle_left + arr.bottom_right = 10 ∧
  arr.top + arr.middle_right + arr.bottom_left = 10 ∧
  arr.bottom_left + arr.bottom + arr.bottom_right = 10

-- Theorem statement
theorem exists_valid_arrangement :
  ∃ (arr : TriangularArrangement), is_valid_arrangement arr ∧ sums_are_ten arr := by
  sorry

#check exists_valid_arrangement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_arrangement_l491_49118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_face_diagonal_angle_is_60_degrees_l491_49106

-- Define a cube
def Cube := {c : ℝ // c > 0}

-- Define the length of a face diagonal
noncomputable def face_diagonal_length (c : Cube) : ℝ := c.val * Real.sqrt 2

-- Define the angle between two face diagonals
noncomputable def face_diagonal_angle (c : Cube) : ℝ := Real.arccos (1 / 2)

-- Theorem statement
theorem face_diagonal_angle_is_60_degrees (c : Cube) :
  face_diagonal_angle c = 60 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_face_diagonal_angle_is_60_degrees_l491_49106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_positive_condition_l491_49124

theorem sin_positive_condition : 
  (∀ α : Real, 0 < α ∧ α < Real.pi → Real.sin α > 0) ∧
  (∃ β : Real, Real.sin β > 0 ∧ (β ≤ 0 ∨ β ≥ Real.pi)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_positive_condition_l491_49124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_C_l491_49187

structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right_angled : (C.1 - A.1) * (C.1 - B.1) + (C.2 - A.2) * (C.2 - B.2) = 0

def O : ℝ × ℝ := (0, 0)

def slides_along_axes (t : RightTriangle) : Prop :=
  t.A.1 = 0 ∧ t.B.2 = 0

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def on_circle (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Prop :=
  distance center point = radius

theorem locus_of_C (t : RightTriangle) :
  slides_along_axes t →
  on_circle O (distance t.A t.B) t.C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_C_l491_49187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l491_49193

-- Define the function
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 8

-- Define the interval
def interval : Set ℝ := Set.Icc (-2) 3

-- State the theorem
theorem function_properties :
  -- Tangent line at x = 1
  (∀ x y, 12 * x + y - 7 = 0 ↔ y = (deriv f) 1 * (x - 1) + f 1) ∧
  -- Maximum value
  (∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x ∧ f x = 15) ∧
  -- Minimum value
  (∃ x ∈ interval, ∀ y ∈ interval, f x ≤ f y ∧ f x = -12) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l491_49193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_true_states_l491_49181

/-- Represents a grid of binary states -/
def Grid (n : ℕ) := Fin n → Fin n → Bool

/-- The operation of flipping states in a row and column except their intersection -/
def flipGrid (grid : Grid n) (i j : Fin n) : Grid n :=
  fun x y => if x = i ∧ y ≠ j ∨ x ≠ i ∧ y = j then !(grid x y) else grid x y

/-- Counts the number of true states in the grid -/
def countTrue (grid : Grid n) : ℕ :=
  (Finset.univ.sum fun i => (Finset.univ.sum fun j => if grid i j then 1 else 0))

/-- The initial grid with all states set to false -/
def initialGrid (n : ℕ) : Grid n := fun _ _ => false

theorem min_true_states (n : ℕ) (h : n > 0) :
  ∀ (grid : Grid n), ∃ (m : ℕ), (∃ (seq : List (Fin n × Fin n)),
    grid = seq.foldl (fun g p => flipGrid g p.1 p.2) (initialGrid n) ∧
    countTrue grid > 0) → countTrue grid ≥ 2 * n - 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_true_states_l491_49181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_flow_maximum_and_range_l491_49166

/-- The traffic flow function -/
noncomputable def traffic_flow (v : ℝ) : ℝ := 920 * v / (v^2 + 3 * v + 1600)

theorem traffic_flow_maximum_and_range :
  (∀ v : ℝ, v > 0 → traffic_flow v ≤ 920 / 83) ∧
  (traffic_flow 40 = 920 / 83) ∧
  (∀ v : ℝ, v > 0 → (traffic_flow v > 10 ↔ 25 < v ∧ v < 64)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_flow_maximum_and_range_l491_49166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mosaic_unique_symmetry_center_l491_49114

/-- Represents a regular hexagon in the mosaic --/
structure Hexagon where
  angle90_count : Nat
  angle135_count : Nat
  opposing_90_angles : Bool

/-- Represents the mosaic of hexagons --/
structure Mosaic where
  hexagons : Set Hexagon
  covers_plane : Bool
  equal_parts : Nat
  nth_row_count : Nat → Nat
  three_hexagons_at_vertex : Bool

/-- Represents a point in the plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is a symmetry center of the mosaic --/
def is_symmetry_center (m : Mosaic) (c : Point) : Prop :=
  sorry -- Definition of symmetry center

/-- The shown center of symmetry --/
def shown_center : Point :=
  ⟨0, 0⟩ -- Assuming the shown center is at the origin

/-- The theorem stating that the mosaic has only one symmetry center --/
theorem mosaic_unique_symmetry_center (m : Mosaic) : 
  m.hexagons.Nonempty ∧ 
  m.covers_plane ∧ 
  (∀ h : Hexagon, h ∈ m.hexagons → h.angle90_count = 2 ∧ h.angle135_count = 4 ∧ h.opposing_90_angles) ∧
  m.equal_parts = 8 ∧
  (∀ n : Nat, m.nth_row_count n = n) ∧
  m.three_hexagons_at_vertex →
  ¬ ∃ (c : Point), c ≠ shown_center ∧ is_symmetry_center m c :=
by
  sorry -- Proof goes here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mosaic_unique_symmetry_center_l491_49114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l491_49177

variable (a b : ℝ → ℝ → ℝ → ℝ)

theorem angle_between_vectors 
  (h1 : Real.sqrt ((a 0 0 0)^2 + (a 1 0 0)^2 + (a 2 0 0)^2) = 2 * Real.sqrt ((b 0 0 0)^2 + (b 1 0 0)^2 + (b 2 0 0)^2))
  (h2 : Real.sqrt ((b 0 0 0)^2 + (b 1 0 0)^2 + (b 2 0 0)^2) ≠ 0)
  (h3 : (b 0 0 0 * (a 0 0 0 - b 0 0 0)) + (b 1 0 0 * (a 1 0 0 - b 1 0 0)) + (b 2 0 0 * (a 2 0 0 - b 2 0 0)) = 0) :
  Real.arccos (((a 0 0 0 * b 0 0 0) + (a 1 0 0 * b 1 0 0) + (a 2 0 0 * b 2 0 0)) / 
    (Real.sqrt ((a 0 0 0)^2 + (a 1 0 0)^2 + (a 2 0 0)^2) * 
     Real.sqrt ((b 0 0 0)^2 + (b 1 0 0)^2 + (b 2 0 0)^2))) = π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l491_49177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_are_imaginary_l491_49104

theorem roots_are_imaginary (k : ℝ) (h1 : k > 0) : 
  let f : ℝ → ℝ := λ x => x^2 - (4*k - 3)*x + 3*k^2 - 2
  let roots := {x : ℂ | f x.re = 0}
  (∀ x ∈ roots, ∀ y ∈ roots, (x * y).re = 10) → 
  (∀ x ∈ roots, ¬(x.im = 0)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_are_imaginary_l491_49104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_value_l491_49120

theorem tan_theta_value (θ k : ℝ) 
  (h_sin : Real.sin θ = (k + 1) / (k - 3))
  (h_cos : Real.cos θ = (k - 1) / (k - 3))
  (h_not_axis : Real.sin θ ≠ 0 ∧ Real.cos θ ≠ 0) :
  Real.tan θ = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_value_l491_49120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_m_value_l491_49143

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m + 2 / (2^x + 1)

theorem odd_function_m_value (m : ℝ) :
  (∀ x, f m (-x) = -(f m x)) → m = -1 := by
  intro h
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_m_value_l491_49143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_smallest_positive_period_l491_49171

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi/3) * Real.cos (Real.pi/6 - x)

/-- The period of the function f -/
noncomputable def period : ℝ := Real.pi

/-- Theorem stating that the period of f is the smallest positive period -/
theorem f_smallest_positive_period :
  ∀ (x : ℝ), f (x + period) = f x ∧
  ∀ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) → p ≥ period := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_smallest_positive_period_l491_49171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_cost_equation_original_cost_satisfies_equation_l491_49152

/-- Proves the equation for the original cost of an article given profit conditions -/
theorem article_cost_equation (C : ℝ) : 1.50 * C - 1.12 * C = 40 :=
  by sorry

/-- Calculates the original cost of the article -/
noncomputable def original_cost : ℝ :=
  40 / 0.38

/-- Proves that the calculated original cost satisfies the equation -/
theorem original_cost_satisfies_equation : 
  1.50 * original_cost - 1.12 * original_cost = 40 :=
  by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_cost_equation_original_cost_satisfies_equation_l491_49152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l491_49126

-- Define the circle C in polar coordinates
def circle_C (ρ θ : ℝ) : Prop := ρ = 4 * Real.sqrt 2 * Real.cos (θ - Real.pi/4)

-- Define the line l in parametric form
def line_l (t x y : ℝ) : Prop := x = t + 1 ∧ y = t - 1

-- Theorem statement
theorem circle_line_intersection :
  -- Given the circle C and line l as defined above
  ∃ (C : Set (ℝ × ℝ)) (l : Set (ℝ × ℝ)),
  -- The Cartesian equation of circle C
  (∀ x y, (x, y) ∈ C ↔ x^2 + y^2 = 4*x - 4*y) ∧
  -- The standard equation of line l
  (∀ x y, (x, y) ∈ l ↔ x - y = 2) ∧
  -- The area of triangle ABC
  (∃ A B : ℝ × ℝ, A ∈ C ∧ A ∈ l ∧ B ∈ C ∧ B ∈ l ∧ A ≠ B ∧
    ∃ (area : ℝ), area = 2 * Real.sqrt 3 ∧
    area = (1/2) * Real.sqrt ((A.fst - B.fst)^2 + (A.snd - B.snd)^2) * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l491_49126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_driving_practice_results_l491_49186

def trainee_A_record : List Int := [15, -2, 5, -1, 10, -3, -2, 12, 4, -5, 6]
def trainee_B_record : List Int := [-17, 9, -2, 8, 6, 9, -5, -1, 4, -7, -8]

def final_position (record : List Int) : Int :=
  record.sum

def total_distance (record : List Int) : Int :=
  record.map abs |>.sum

theorem driving_practice_results :
  (final_position trainee_A_record = 39) ∧
  (final_position trainee_B_record = -4) ∧
  (total_distance trainee_A_record < total_distance trainee_B_record) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_driving_practice_results_l491_49186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_for_half_meter_l491_49165

/-- The number of tiles needed for a given side length, given the area of the room -/
noncomputable def tiles_needed (side_length : ℝ) (area : ℝ) : ℝ :=
  area / (side_length * side_length)

/-- The theorem stating the number of tiles needed for 0.5m side length -/
theorem tiles_for_half_meter (room_area : ℝ) :
  (tiles_needed 0.3 room_area = 500) →
  (tiles_needed 0.5 room_area = 180) :=
by
  sorry

#check tiles_for_half_meter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_for_half_meter_l491_49165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l491_49155

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

theorem triangle_properties (t : Triangle) 
  (h1 : Real.cos t.C + (Real.cos t.A - Real.sqrt 3 * Real.sin t.A) * Real.cos t.B = 0)
  (h2 : t.a + t.c = 1) :
  t.B = π / 3 ∧ 1 / 2 ≤ t.b ∧ t.b < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l491_49155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_for_equation_l491_49117

theorem two_solutions_for_equation : 
  ∃! (s : Finset (ℕ × ℕ)), 
    (∀ (m n : ℕ), (m, n) ∈ s ↔ m > 0 ∧ n > 0 ∧ (5 : ℚ) / m + (3 : ℚ) / n = 1) ∧ 
    s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_for_equation_l491_49117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bricks_needed_for_l_shaped_wall_l491_49109

/-- Represents the dimensions of a rectangular object in centimeters -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular object in cubic centimeters -/
def volume (d : Dimensions) : ℝ := d.length * d.width * d.height

/-- Converts cubic meters to cubic centimeters -/
def m3_to_cm3 (v : ℝ) : ℝ := v * 1000000

theorem bricks_needed_for_l_shaped_wall (brick_dim : Dimensions) 
    (section_a_dim : Dimensions) (section_b_dim : Dimensions) 
    (window_dim : Dimensions) 
    (h1 : brick_dim.length = 25)
    (h2 : brick_dim.width = 11)
    (h3 : brick_dim.height = 6)
    (h4 : section_a_dim.length = 800)
    (h5 : section_a_dim.width = 100)
    (h6 : section_a_dim.height = 5)
    (h7 : section_b_dim.length = 500)
    (h8 : section_b_dim.width = 100)
    (h9 : section_b_dim.height = 5)
    (h10 : window_dim.length = 200)
    (h11 : window_dim.width = 100)
    (h12 : window_dim.height = 5)
    : ↑(⌈(volume section_a_dim - volume window_dim + volume section_b_dim) / volume brick_dim⌉) = 334 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bricks_needed_for_l_shaped_wall_l491_49109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_battery_replacement_month_l491_49170

/-- Represents the months of the year -/
inductive Month
| january | february | march | april | may | june
| july | august | september | october | november | december

/-- Returns the next month in the sequence -/
def nextMonth (m : Month) : Month :=
  match m with
  | Month.january => Month.february
  | Month.february => Month.march
  | Month.march => Month.april
  | Month.april => Month.may
  | Month.may => Month.june
  | Month.june => Month.july
  | Month.july => Month.august
  | Month.august => Month.september
  | Month.september => Month.october
  | Month.october => Month.november
  | Month.november => Month.december
  | Month.december => Month.january

/-- Calculates the month after a given number of months have passed -/
def monthAfter (start : Month) (months : Nat) : Month :=
  match months with
  | 0 => start
  | n + 1 => monthAfter (nextMonth start) n

theorem battery_replacement_month :
  monthAfter Month.january ((15 - 1) * 3) = Month.july := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_battery_replacement_month_l491_49170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_increasing_on_interval_l491_49172

noncomputable def f (x : ℝ) : ℝ := Real.cos (4 * x - Real.pi / 3) + 2 * (Real.cos (2 * x))^2

noncomputable def g (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) + 1

def is_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem g_increasing_on_interval :
  is_increasing g (-Real.pi/4) (Real.pi/4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_increasing_on_interval_l491_49172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_of_triangle_l491_49107

/-- The longest side of a triangle with vertices at (1,2), (4,7), and (7,2) has a length of 6 units -/
theorem longest_side_of_triangle : ∃ (longest_side : ℝ), longest_side = 6 := by
  let vertex1 : ℝ × ℝ := (1, 2)
  let vertex2 : ℝ × ℝ := (4, 7)
  let vertex3 : ℝ × ℝ := (7, 2)

  let side1 := Real.sqrt ((vertex2.1 - vertex1.1)^2 + (vertex2.2 - vertex1.2)^2)
  let side2 := Real.sqrt ((vertex3.1 - vertex1.1)^2 + (vertex3.2 - vertex1.2)^2)
  let side3 := Real.sqrt ((vertex3.1 - vertex2.1)^2 + (vertex3.2 - vertex2.2)^2)

  let longest_side := max side1 (max side2 side3)

  use longest_side
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_of_triangle_l491_49107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_comparison_l491_49147

-- Define the triangles and their properties
structure Triangle where
  base : ℝ
  height : ℝ

noncomputable def triangle_Q : Triangle := { base := 1, height := 1 }

noncomputable def triangle_P : Triangle := {
  base := 1.3 * triangle_Q.base,
  height := 0.6 * triangle_Q.height
}

noncomputable def triangle_R : Triangle := {
  base := 0.55 * triangle_Q.base,
  height := 1.25 * triangle_Q.height
}

-- Define the area calculation function
noncomputable def area (t : Triangle) : ℝ := t.base * t.height / 2

-- Define the percentage difference function
noncomputable def percentage_difference (a b : ℝ) : ℝ := (b - a) / b * 100

-- Theorem statement
theorem triangle_area_comparison :
  percentage_difference (area triangle_P) (area triangle_Q) = 22 ∧
  percentage_difference (area triangle_R) (area triangle_Q) = 31.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_comparison_l491_49147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_percent_problem_l491_49161

theorem certain_percent_problem (x y : ℕ) (p : ℚ) : 
  y = 125 →
  y = (50 : ℚ) / 100 * (p / 100) * x →
  (y : ℚ) / 100 * x = 100 →
  p = 312.5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_percent_problem_l491_49161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l491_49103

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x^2 - 1) / (x^2 + 2)

-- State the theorem
theorem f_range :
  ∀ y : ℝ, (∃ x : ℝ, x > -1 ∧ f x = y) ↔ -1/2 ≤ y ∧ y < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l491_49103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_signal_post_l491_49142

/-- The time (in seconds) it takes for a train to pass a signal post -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  train_length / (train_speed_kmh * 1000 / 3600)

/-- Theorem stating that a 400-meter long train traveling at 60 km/hour takes approximately 24 seconds to pass a signal post -/
theorem train_passing_signal_post :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ 
  |train_passing_time 400 60 - 24| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_signal_post_l491_49142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_relationship_l491_49133

theorem remainder_relationship : 
  ∃ (A B M : ℕ) (S T s t : ℕ), 
    A > B ∧
    S = A % M ∧
    T = B % M ∧
    s = (A^2) % M ∧
    t = (B^2) % M ∧
    ((s > t) ∨ (s < t)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_relationship_l491_49133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_for_special_function_l491_49151

/-- Given a function f(x) = k - √(x + 2), where k is a real constant, and there exist real 
    numbers a and b with a < b such that the range of f(x) on [a, b] is [a, b], 
    prove that the range of possible values for k is (-5/4, -1]. -/
theorem range_of_k_for_special_function : 
  ∀ (k a b : ℝ), a < b → 
  (∀ x ∈ Set.Icc a b, ∃ y ∈ Set.Icc a b, k - Real.sqrt (x + 2) = y) →
  (∀ y ∈ Set.Icc a b, ∃ x ∈ Set.Icc a b, k - Real.sqrt (x + 2) = y) →
  k ∈ Set.Ioc (-5/4) (-1) := by
  sorry

#check range_of_k_for_special_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_for_special_function_l491_49151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_density2_is_2160_l491_49132

/-- Represents an object with two parts -/
structure TwoPartObject where
  totalVolume : ℝ
  totalMass : ℝ
  density1 : ℝ
  volumeRatio1 : ℝ
  massRatio1 : ℝ

/-- Calculates the density of the second part of a two-part object -/
noncomputable def density2 (obj : TwoPartObject) : ℝ :=
  (obj.totalMass * (1 - obj.massRatio1)) / (obj.totalVolume * (1 - obj.volumeRatio1))

/-- Theorem stating that under given conditions, the density of the second part is 2160 kg/m³ -/
theorem density2_is_2160 (obj : TwoPartObject) 
    (h1 : obj.density1 = 2700)
    (h2 : obj.volumeRatio1 = 0.25)
    (h3 : obj.massRatio1 = 0.4) :
  density2 obj = 2160 := by
  sorry

#check density2_is_2160

end NUMINAMATH_CALUDE_ERRORFEEDBACK_density2_is_2160_l491_49132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l491_49163

theorem problem_solution (x y m n : ℝ) : 
  (x^m = 2 ∧ y^n = 3 → x^(3*m) + y^(2*n) = 17) ∧
  (x + 2*y - 2 = 0 → (2:ℝ)^x * (4:ℝ)^y = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l491_49163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_pell_like_equation_l491_49160

theorem infinite_solutions_pell_like_equation (D : ℤ) (hD : D ≠ 0) :
  ∃ f : ℕ → ℕ × ℕ × ℕ,
    Function.Injective f ∧
    ∀ n : ℕ,
      let (x, y, z) := f n
      x > 0 ∧ y > 0 ∧ z > 0 ∧
      x^2 - D * y^2 = z^2 ∧
      Nat.gcd x y = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_pell_like_equation_l491_49160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_quadrilateral_angle_cosine_l491_49144

/-- For a quadrilateral inscribed in a circle with sides a, b, c, and d,
    the cosine of the angle α enclosed between sides a and b is given by
    (a² + b² - c² - d²) / (2(ab + cd)) -/
theorem inscribed_quadrilateral_angle_cosine
  (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ∃ α : ℝ, Real.cos α = (a^2 + b^2 - c^2 - d^2) / (2 * (a * b + c * d)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_quadrilateral_angle_cosine_l491_49144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_seven_percentage_l491_49197

theorem divisible_by_seven_percentage (n : ℕ) (hn : n = 140) :
  (Finset.filter (λ x => x % 7 = 0) (Finset.range n)).card / n = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_seven_percentage_l491_49197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_max_a_l491_49199

noncomputable def f (x : ℝ) : ℝ := x^2 + 1

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := 2 * a * Real.log x + 1

theorem common_tangent_max_a :
  ∃ (a : ℝ), ∀ (a' : ℝ),
    (∃ (x y : ℝ), (deriv f x = deriv (g a) y) ∧ (f x = g a x) ∧ (f y = g a y)) →
    a' ≤ a ∧ a = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_max_a_l491_49199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l491_49158

def A : Set ℝ := {x | (2*x - 1)/(x - 2) < 0}
def B : Set ℝ := Set.range (fun n : ℕ => (n : ℝ))

theorem intersection_A_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l491_49158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_photos_theorem_l491_49146

/-- Represents a tourist's photo collection --/
def PhotoCollection := Fin 3 → Bool

/-- The number of tourists --/
def num_tourists : Nat := 42

/-- The condition that any two tourists together have photos of all monuments --/
def covers_all_monuments (photos : Fin num_tourists → PhotoCollection) : Prop :=
  ∀ i j : Fin num_tourists, i ≠ j → ∀ k : Fin 3, photos i k || photos j k

/-- The total number of photos taken --/
def total_photos (photos : Fin num_tourists → PhotoCollection) : Nat :=
  Finset.sum (Finset.univ : Finset (Fin num_tourists)) (λ i =>
    Finset.sum (Finset.univ : Finset (Fin 3)) (λ j => if photos i j then 1 else 0))

/-- The main theorem --/
theorem min_photos_theorem :
  ∃ (photos : Fin num_tourists → PhotoCollection),
    covers_all_monuments photos ∧
    total_photos photos = 123 ∧
    (∀ photos' : Fin num_tourists → PhotoCollection,
      covers_all_monuments photos' → total_photos photos' ≥ 123) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_photos_theorem_l491_49146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_term_a_seq_l491_49123

def a (n : ℕ) : ℤ := n^2 - 9*n - 100

theorem smallest_term_a_seq :
  ∃ k ∈ ({4, 5} : Set ℕ), ∀ n : ℕ, n ≥ 1 → a k ≤ a n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_term_a_seq_l491_49123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_iff_a_eq_neg_one_or_three_l491_49110

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def IsPerp (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- Definition of the first line l1 -/
def l1 (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y => a * x + (1 + a) * y = 3

/-- Definition of the second line l2 -/
def l2 (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y => (a + 1) * x + (3 - 2 * a) * y = 2

/-- Definition of perpendicularity for our specific lines -/
def lines_perpendicular (a : ℝ) : Prop :=
  IsPerp (-a / (1 + a)) (-(a + 1) / (3 - 2 * a))

/-- The main theorem: l1 and l2 are perpendicular if and only if a is -1 or 3 -/
theorem perpendicular_lines_iff_a_eq_neg_one_or_three :
  ∀ a : ℝ, lines_perpendicular a ↔ (a = -1 ∨ a = 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_iff_a_eq_neg_one_or_three_l491_49110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_calculation_l491_49134

/-- Calculates the simple interest earned given principal, rate, and time -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Calculates the compound interest earned given principal, rate, and time -/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

theorem simple_interest_calculation (totalSavings : ℝ) (compoundInterestEarned : ℝ) (time : ℝ) :
  totalSavings = 1800 →
  compoundInterestEarned = 124 →
  time = 2 →
  ∃ (rate : ℝ),
    compoundInterest (totalSavings / 2) rate time = compoundInterestEarned ∧
    abs (simpleInterest (totalSavings / 2) rate time - 119.7) < 0.1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_calculation_l491_49134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_bucket_problem_l491_49128

/-- Proves that under given conditions, the initial weight of water in each bucket was 3.5 kg -/
theorem water_bucket_problem (initial_weight : ℝ) : 
  initial_weight > 0 →  -- Assuming positive initial weight
  initial_weight + 2.5 = 6 * (initial_weight - 2.5) → 
  initial_weight = 3.5 := by
  intro h1 h2
  -- Proof steps would go here
  sorry

#check water_bucket_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_bucket_problem_l491_49128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_grades_2_and_3_l491_49113

theorem total_students_grades_2_and_3 : ℕ := by
  let boys_grade_2 : ℕ := 20
  let girls_grade_2 : ℕ := 11
  let students_grade_2 : ℕ := boys_grade_2 + girls_grade_2
  let students_grade_3 : ℕ := 2 * students_grade_2
  have h : students_grade_2 + students_grade_3 = 93 := by
    -- Proof steps would go here
    sorry
  exact students_grade_2 + students_grade_3


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_grades_2_and_3_l491_49113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_and_function_property_l491_49182

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / (3^x + Real.sqrt 3)

-- State the theorem
theorem inequalities_and_function_property 
  (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : a + b + c = 0) :
  (Real.sqrt (b^2 - a*c) < Real.sqrt 3 * a) ∧ 
  (∀ x : ℝ, f x + f (1 - x) = Real.sqrt 3 / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_and_function_property_l491_49182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_a_bound_l491_49112

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := -1/x + x
noncomputable def g (x : ℝ) : ℝ := x^2 - 4*x - 3

-- State the theorem
theorem function_inequality_implies_a_bound (a : ℝ) :
  (∀ x₁ ∈ Set.Icc (-2) (-1), ∃ x₂ ∈ Set.Icc 1 a, f x₁ < g x₂) →
  a > 2 + Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_a_bound_l491_49112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_queue_adjustment_ways_l491_49162

/-- The number of ways to select k items from n items -/
def combination (n k : ℕ) : ℕ := 
  if k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  else 0

/-- The number of ways to insert m items into n + 1 spaces -/
def insertion_ways (n m : ℕ) : ℕ := 
  (List.range m).foldl (λ acc i => acc * (n + i + 1)) 1

theorem queue_adjustment_ways :
  let total_students : ℕ := 10
  let front_row_initial : ℕ := 3
  let back_row_initial : ℕ := 7
  let students_to_move : ℕ := 2
  combination back_row_initial students_to_move * 
    insertion_ways front_row_initial students_to_move = 420 := by
  sorry

#eval combination 7 2 * insertion_ways 3 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_queue_adjustment_ways_l491_49162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_black_cell_arrangement_l491_49164

/-- Represents a 4x4 board where each cell contains the count of neighboring black cells --/
def Board := Matrix (Fin 4) (Fin 4) Nat

/-- The given board configuration --/
def givenBoard : Board :=
  Matrix.of ![
    ![1, 2, 1, 1],
    ![0, 2, 1, 2],
    ![2, 3, 3, 1],
    ![1, 0, 2, 1]
  ]

/-- Represents the coordinates of a cell on the board --/
structure Cell where
  row : Fin 4
  col : Fin 4

/-- A list of black cell coordinates --/
def BlackCells := List Cell

/-- Counts the number of black neighbors for a given cell --/
def countBlackNeighbors (blackCells : BlackCells) (cell : Cell) : Nat :=
  sorry

/-- Checks if the given black cell arrangement satisfies the board configuration --/
def isValidArrangement (board : Board) (blackCells : BlackCells) : Prop :=
  ∀ i j, board i j = countBlackNeighbors blackCells ⟨i, j⟩

/-- The main theorem to prove --/
theorem unique_black_cell_arrangement :
  ∃! blackCells : BlackCells,
    blackCells.length = 4 ∧
    isValidArrangement givenBoard blackCells :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_black_cell_arrangement_l491_49164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_product_bound_l491_49125

theorem adjacent_product_bound (a b c d e : ℝ) 
  (non_neg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 0 ≤ e) 
  (sum_one : a + b + c + d + e = 1) :
  ∃ (p : Fin 5 → Fin 5), Function.Bijective p ∧
    let arr := [a, b, c, d, e]
    (arr[p 0]! * arr[p 1]! ≤ 1/9) ∨ 
    (arr[p 1]! * arr[p 2]! ≤ 1/9) ∨ 
    (arr[p 2]! * arr[p 3]! ≤ 1/9) ∨ 
    (arr[p 3]! * arr[p 4]! ≤ 1/9) ∨ 
    (arr[p 4]! * arr[p 0]! ≤ 1/9) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_product_bound_l491_49125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_y_coordinate_l491_49180

/-- The y-coordinate of the vertex of the parabola y = 2x^2 + 8x + 5 is -3 -/
theorem parabola_vertex_y_coordinate :
  let f : ℝ → ℝ := λ x ↦ 2 * x^2 + 8 * x + 5
  ∃ p : ℝ, (∀ x, f p ≤ f x) ∧ f p = -3 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_y_coordinate_l491_49180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_count_l491_49139

def number_of_boys : ℕ := 9
def number_of_girls : ℕ := 3
def separation : ℕ := 4

def number_of_arrangements : ℕ :=
  (Nat.choose number_of_boys separation) * Nat.factorial separation * 2 * Nat.factorial 7

theorem arrangement_count :
  number_of_arrangements = 3024 * Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_count_l491_49139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l491_49149

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.A + t.B + t.C = Real.pi ∧
  t.B = Real.pi/4 ∧
  Real.cos t.A - Real.cos (2 * t.A) = 0 ∧
  t.b^2 + t.c^2 = t.a - t.b * t.c + 2

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h : triangle_conditions t) : 
  t.C = Real.pi/12 ∧ 
  (1/2 * t.a * t.c * Real.sin t.B : Real) = 1 - Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l491_49149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_when_minimized_l491_49108

/-- Definition of an ellipse passing through (3,2) -/
def is_ellipse_through_point (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧ (9 / a^2 + 4 / b^2 = 1)

/-- Definition of eccentricity for an ellipse -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

/-- Theorem stating the eccentricity of the ellipse when a^2 + b^2 is minimized -/
theorem ellipse_eccentricity_when_minimized :
  ∃ (a b : ℝ), is_ellipse_through_point a b ∧
  (∀ (a' b' : ℝ), is_ellipse_through_point a' b' → a^2 + b^2 ≤ a'^2 + b'^2) →
  eccentricity a b = Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_when_minimized_l491_49108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l491_49148

theorem constant_term_expansion (x : ℝ) : 
  ∃ (c : ℝ), c = -160 ∧ 
  ∃ (f : ℝ → ℝ), (λ x => (x - 2/x)^6) = (λ x => f x + c) ∧ 
  ∀ x, x ≠ 0 → ∃ p : Polynomial ℝ, f x = p.eval x :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l491_49148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l491_49145

theorem trigonometric_equation_solution :
  ∃ m : ℝ, Real.sin (π / 18) + m * Real.cos (π / 18) = 2 * Real.cos (7 * π / 9) ∧ m = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l491_49145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_increasing_f_min_value_of_f_when_a_is_1_l491_49140

-- Define the function f and its derivative
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x^3/3 + x^2/2 + 2*a*x
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := -x^2 + x + 2*a

-- Theorem 1: Range of a for monotonically increasing f
theorem range_of_a_for_increasing_f :
  ∀ a : ℝ, (∀ x : ℝ, x > 2/3 → f' a x > 0) ↔ a > -1/8 := by
  sorry

-- Theorem 2: Minimum value of f when a=1
theorem min_value_of_f_when_a_is_1 :
  ∀ x : ℝ, x ∈ Set.Icc 1 4 → f 1 x ≥ -16/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_increasing_f_min_value_of_f_when_a_is_1_l491_49140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_value_l491_49119

/-- The function g(x) is defined as the minimum of three linear functions. -/
noncomputable def g (x : ℝ) : ℝ := min (3 * x + 3) (min ((2/3) * x + 2) (-x + 9))

/-- The maximum value of g(x) is 24/5. -/
theorem g_max_value : ∃ (M : ℝ), M = 24/5 ∧ ∀ (x : ℝ), g x ≤ M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_value_l491_49119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_sum_l491_49188

/-- Given a triangle with sides a, b, c and corresponding angles α, β, γ,
    prove that (b/c + c/b)cos(α) + (c/a + a/c)cos(β) + (a/b + b/a)cos(γ) = 3 -/
theorem triangle_cosine_sum (a b c : ℝ) (α β γ : ℝ) 
    (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b)
    (h_angles : 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = Real.pi)
    (h_law_of_cosines_α : Real.cos α = (b^2 + c^2 - a^2) / (2*b*c))
    (h_law_of_cosines_β : Real.cos β = (a^2 + c^2 - b^2) / (2*a*c))
    (h_law_of_cosines_γ : Real.cos γ = (a^2 + b^2 - c^2) / (2*a*b)) :
  (b/c + c/b) * Real.cos α + (c/a + a/c) * Real.cos β + (a/b + b/a) * Real.cos γ = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_sum_l491_49188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equation_l491_49122

variable {n : ℕ}

theorem matrix_equation (B : Matrix (Fin n) (Fin n) ℝ) 
  (h_inv : Invertible B) 
  (h_eq : (B - 3 • 1) * (B - 5 • 1) = 0) : 
  B + 10 • B⁻¹ = 8 • 1 + 5 • B⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equation_l491_49122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_g_equals_one_third_l491_49154

/-- g(n) is defined as the sum of 1/k^n for k from 3 to infinity -/
noncomputable def g (n : ℕ) : ℝ := ∑' k : ℕ, (1 : ℝ) / ((k + 2) ^ n)

/-- The sum of g(n) from n = 3 to infinity equals 1/3 -/
theorem sum_of_g_equals_one_third : ∑' n : ℕ, g (n + 3) = (1 : ℝ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_g_equals_one_third_l491_49154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_not_four_l491_49141

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem omega_not_four 
  (ω φ : ℝ) 
  (h_ω_pos : ω > 0)
  (h_φ_bound : |φ| < π / 2)
  (h_left_shift : ∀ x, f ω φ (x + π/12) = f ω φ (-x + π/12))
  (h_right_shift : ∀ x, f ω φ (x - π/6) = -f ω φ (-x - π/6)) :
  ω ≠ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_not_four_l491_49141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_l491_49176

/-- Represents the distance from a point P on AB to D -/
def x : ℝ := sorry

/-- Represents the distance from a corresponding point P' on A'B' to D' -/
def y : ℝ := sorry

/-- Theorem stating the relationship between x and y -/
theorem distance_sum (a : ℝ) (h : x = a) : x + y = 17 - 3 * a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_l491_49176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_interval_l491_49101

-- Define θ as an internal angle of a triangle
variable (θ : Real)

-- Define the function y
noncomputable def y (x : Real) : Real := Real.cos θ * x^2 - 4 * Real.sin θ * x + 6

-- State the theorem
theorem cos_theta_interval :
  (∀ x : Real, y θ x > 0) →
  Real.cos θ ∈ Set.Ioo (1/2 : Real) 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_interval_l491_49101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_four_valid_configs_l491_49184

/-- Represents a configuration of numbers in the triangles --/
structure TriangleConfig where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  f : ℕ

/-- The set of remaining numbers to be placed --/
def RemainingNumbers : Finset ℕ := {3, 5, 6, 7, 8, 9}

/-- Checks if a configuration is valid --/
def isValidConfig (config : TriangleConfig) : Prop :=
  config.a ∈ RemainingNumbers ∧
  config.b ∈ RemainingNumbers ∧
  config.c ∈ RemainingNumbers ∧
  config.d ∈ RemainingNumbers ∧
  config.e ∈ RemainingNumbers ∧
  config.f ∈ RemainingNumbers ∧
  config.a ≠ config.b ∧ config.a ≠ config.c ∧ config.a ≠ config.d ∧ config.a ≠ config.e ∧ config.a ≠ config.f ∧
  config.b ≠ config.c ∧ config.b ≠ config.d ∧ config.b ≠ config.e ∧ config.b ≠ config.f ∧
  config.c ≠ config.d ∧ config.c ≠ config.e ∧ config.c ≠ config.f ∧
  config.d ≠ config.e ∧ config.d ≠ config.f ∧
  config.e ≠ config.f

/-- Checks if a configuration satisfies the sum condition --/
def satisfiesSumCondition (config : TriangleConfig) : Prop :=
  config.a + config.b + 11 = 23 ∧
  config.b + config.c + config.d + 2 = 23 ∧
  config.b + config.e + config.f + 4 = 23

/-- The main theorem stating there are exactly 4 valid configurations --/
theorem exactly_four_valid_configs :
  ∃! (configs : Finset TriangleConfig),
    (∀ config ∈ configs, isValidConfig config ∧ satisfiesSumCondition config) ∧
    configs.card = 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_four_valid_configs_l491_49184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l491_49115

/-- The area of a triangle given by three points in a 2D plane. -/
noncomputable def triangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

/-- Coordinates of point P -/
def P : ℝ × ℝ := (1, 1)

/-- Coordinates of point Q -/
def Q : ℝ × ℝ := (4, 5)

/-- Coordinates of point R -/
def R : ℝ × ℝ := (7, 2)

theorem triangle_PQR_area :
  triangleArea P.1 P.2 Q.1 Q.2 R.1 R.2 = 10.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l491_49115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l491_49130

-- Define the solution sets M and N
def M : Set ℝ := {x : ℝ | 2 * |x - 1| + x - 1 ≤ 1}
def N : Set ℝ := {x : ℝ | 16 * x^2 - 8 * x + 1 ≤ 4}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = Set.Icc 0 (3/4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l491_49130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_roots_l491_49185

noncomputable def f (x : ℝ) : ℝ := 
  if x > -2 then Real.exp (x + 1) - 2 
  else Real.exp (-(x + 1)) - 2

theorem even_function_roots (k : ℤ) : 
  (∀ x, f (-x) = f x) ∧ 
  (∃ x₀ : ℝ, f x₀ = 0 ∧ (k - 1 : ℝ) < x₀ ∧ x₀ < k) ↔ 
  k = -3 ∨ k = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_roots_l491_49185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_correctness_l491_49135

open Real

noncomputable def is_in_square (x y : ℝ) : Prop :=
  -π ≤ x ∧ x ≤ π ∧ 0 ≤ y ∧ y ≤ 2*π

noncomputable def satisfies_equations (x y : ℝ) : Prop :=
  sin x + sin y = sin 2 ∧ cos x + cos y = cos 2

noncomputable def solution_point (x y : ℝ) : Prop :=
  is_in_square x y ∧ satisfies_equations x y

def number_of_solutions : ℕ := 2

noncomputable def smallest_ordinate_point : ℝ × ℝ := (2 + π/3, 2 - π/3)

theorem solution_correctness :
  (∃ (points : Finset (ℝ × ℝ)), points.card = number_of_solutions ∧
    ∀ (p : ℝ × ℝ), p ∈ points ↔ solution_point p.1 p.2) ∧
  (solution_point smallest_ordinate_point.1 smallest_ordinate_point.2 ∧
    ∀ (x y : ℝ), solution_point x y → y ≥ smallest_ordinate_point.2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_correctness_l491_49135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fitness_center_member_ratio_l491_49196

/-- Proves the ratio of female to male members in a fitness center -/
theorem fitness_center_member_ratio 
  (f m : ℚ) -- number of female and male members as rationals
  (h1 : f > 0)
  (h2 : m > 0)
  (h3 : (40 * f + 25 * m) / (f + m) = 30) :
  f / m = 1 / 2 := by
  sorry

#check fitness_center_member_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fitness_center_member_ratio_l491_49196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_doll_count_after_two_years_l491_49167

/-- Represents the number of dolls each person has. -/
structure DollCount where
  gina : ℝ
  susan : ℝ
  rene : ℝ
  natalie : ℝ
  emily : ℝ

/-- Calculates the total number of dolls after 2 years. -/
noncomputable def totalDollsAfterTwoYears (initial : DollCount) : ℝ :=
  (3 * (initial.gina + 8)) +  -- Rene's dolls
  (initial.gina + 8) +        -- Susan's dolls
  (5/2 * (initial.gina + 6)) + -- Natalie's dolls
  (Real.sqrt (initial.gina + 6) - 2) + -- Emily's dolls
  (initial.gina + 6)          -- Gina's dolls

/-- The theorem stating the relationship between initial doll counts and the total after 2 years. -/
theorem doll_count_after_two_years (initial : DollCount) 
  (h1 : initial.rene = 3 * initial.susan)
  (h2 : initial.susan = initial.gina + 2)
  (h3 : initial.natalie = 5/2 * initial.gina)
  (h4 : initial.emily = Real.sqrt initial.gina - 2) :
  totalDollsAfterTwoYears initial = 5 * initial.gina + 38 + 5/2 * (initial.gina + 6) + Real.sqrt (initial.gina + 6) - 2 := by
  sorry

#check doll_count_after_two_years

end NUMINAMATH_CALUDE_ERRORFEEDBACK_doll_count_after_two_years_l491_49167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l491_49173

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + 2 / x

-- State the theorem
theorem function_properties (a : ℝ) :
  (f a (-2) = 1) →
  (∀ x y : ℝ, 0 < x ∧ x < y → f a x > f a y) ∧
  (∀ t : ℝ, (∀ x : ℝ, x ≥ 1 → f a x ≤ (1 + t * x) / x) ↔ t ≥ 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l491_49173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_decreasing_on_open_interval_l491_49136

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 1 / x
noncomputable def g (x : ℝ) : ℝ := x^2 * f (x - 1)

-- State the theorem
theorem g_decreasing_on_open_interval :
  ∀ x ∈ Set.Ioo 1 2, ∀ y ∈ Set.Ioo 1 2, x < y → g x > g y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_decreasing_on_open_interval_l491_49136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_driver_regular_rate_l491_49157

/-- Represents the bus driver's compensation structure and worked hours --/
structure BusDriverCompensation where
  regularRate : ℝ
  overtimeMultiplier : ℝ
  regularHours : ℝ
  overtimeHours : ℝ
  totalCompensation : ℝ

/-- Calculates the total compensation for a bus driver --/
def calculateTotalCompensation (bdc : BusDriverCompensation) : ℝ :=
  bdc.regularRate * bdc.regularHours + 
  bdc.regularRate * bdc.overtimeMultiplier * bdc.overtimeHours

/-- Theorem stating that the bus driver's regular rate is approximately $13.95 --/
theorem bus_driver_regular_rate : 
  ∀ (bdc : BusDriverCompensation), 
  bdc.overtimeMultiplier = 1.75 ∧ 
  bdc.regularHours = 40 ∧ 
  bdc.overtimeHours = 18 ∧
  bdc.totalCompensation = 998 ∧
  calculateTotalCompensation bdc = bdc.totalCompensation →
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |bdc.regularRate - 13.95| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_driver_regular_rate_l491_49157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sample_count_example_l491_49195

/-- Systematic sampling function that returns the number of selected items in a given interval -/
def systematicSampleCount (populationSize : ℕ) (sampleSize : ℕ) (intervalStart : ℕ) (intervalEnd : ℕ) : ℕ :=
  let stepSize := populationSize / sampleSize
  (intervalEnd - intervalStart + 1) / stepSize

/-- Theorem stating that for the given parameters, the systematic sample count in the specified interval is 6 -/
theorem systematic_sample_count_example :
  systematicSampleCount 420 21 241 360 = 6 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sample_count_example_l491_49195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_eventually_constant_l491_49191

/-- Define the sequence a_n -/
def sequence_property (N : ℕ) (a : ℕ → ℕ) : Prop :=
  (∀ n ≤ N, 0 < a n ∧ ¬ (2^(N+1) ∣ a n)) ∧
  (∀ n > N, ∃ k < n, a n = 2 * a k ∧
    ∀ j < n, a k % 2^n ≤ a j % 2^n)

/-- The main theorem -/
theorem sequence_eventually_constant
  {N : ℕ} {a : ℕ → ℕ} (h : sequence_property N a) :
  ∃ M : ℕ, ∀ n ≥ M, a n = a M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_eventually_constant_l491_49191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l491_49102

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

noncomputable def solutions : Set ℝ := {Real.sqrt 29 / 2, Real.sqrt 189 / 2, Real.sqrt 229 / 2, Real.sqrt 269 / 2}

theorem equation_solutions :
  ∀ x : ℝ, (4 * x^2 - 40 * (floor x) + 51 = 0) ↔ (x ∈ solutions) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l491_49102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_special_lines_l491_49105

/-- A line in the 2D plane --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A parabola in the 2D plane --/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Determines if a line intersects a parabola at exactly one point --/
def intersectsOnce (l : Line) (p : Parabola) : Prop := sorry

/-- The point through which all lines must pass --/
def fixedPoint : ℝ × ℝ := (0, 1)

/-- The parabola y^2 = 4x --/
def givenParabola : Parabola := { a := 4, h := 0, k := 0 }

/-- The set of lines passing through the fixed point and intersecting the parabola once --/
def specialLines : Finset Line := 
  sorry

/-- There are exactly three special lines --/
theorem three_special_lines : specialLines.card = 3 := by
  sorry

#check three_special_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_special_lines_l491_49105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l491_49150

/-- 
Given a hyperbola with equation x²/a² - y²/b² = 1 and eccentricity √3,
prove that its asymptotes are given by the equation y = ±√2 * x
-/
theorem hyperbola_asymptotes (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →  -- Equation of the hyperbola
  (Real.sqrt 3 = Real.sqrt (a^2 + b^2) / a) →  -- Eccentricity is √3
  (∀ x y : ℝ, y = Real.sqrt 2 * x ∨ y = -(Real.sqrt 2 * x)) := -- Equation of asymptotes
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l491_49150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l491_49111

noncomputable def f (x : ℝ) : ℝ := (16 * x + 7) / (4 * x + 4)

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 3  -- a_1 = 3
  | n + 1 => f (sequence_a n)

noncomputable def sequence_b : ℕ → ℝ
  | 0 => 4  -- b_1 = 4
  | n + 1 => f (sequence_b n)

theorem part1 (a₁ : ℝ) (h : a₁ > 0) :
  (∀ n : ℕ, sequence_a (n + 1) > sequence_a n) ↔ (0 < a₁ ∧ a₁ < 7/2) := by
  sorry

theorem part2 : ∀ n : ℕ, 0 < sequence_b n - sequence_a n ∧ sequence_b n - sequence_a n ≤ (1/8)^n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l491_49111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_inequality_equivalence_l491_49194

theorem angle_inequality_equivalence (A B : ℝ) (h1 : 0 < A) (h2 : A < π) (h3 : 0 < B) (h4 : B < π) :
  (A ≠ B) ↔ (Real.cos (2 * A) ≠ Real.cos (2 * B)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_inequality_equivalence_l491_49194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_moving_circle_trajectory_l491_49174

/-- The trajectory of the center of a moving circle that passes through a fixed point
    and is internally tangent to another fixed circle is an ellipse. -/
theorem moving_circle_trajectory
  (A B : ℝ × ℝ)
  (r : ℝ)
  (h_A : A = (-3, 0))
  (h_B : B = (3, 0))
  (h_r : r = 10) :
  ∃ P : ℝ → ℝ × ℝ,
    (∀ t : ℝ, ((P t).1)^2 / 25 + ((P t).2)^2 / 16 = 1) ∧
    (∀ t : ℝ, ((P t).1 - A.1)^2 + ((P t).2 - A.2)^2 = ((P t).1 - B.1)^2 + ((P t).2 - B.2)^2) ∧
    (∀ t : ℝ, (((P t).1 - B.1)^2 + ((P t).2 - B.2)^2)^(1/2) + (((P t).1 - A.1)^2 + ((P t).2 - A.2)^2)^(1/2) = r) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_moving_circle_trajectory_l491_49174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_when_fourth_power_divisible_by_251_l491_49183

theorem largest_divisor_when_fourth_power_divisible_by_251 (n : ℕ) 
  (hn : n > 0)
  (h : n^4 % 251 = 0) : 
  ∀ m : ℕ, m ∣ n → m ≤ 251 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_when_fourth_power_divisible_by_251_l491_49183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_passes_through_point_no_zeros_on_interval_inequality_holds_l491_49127

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x

-- Statement 1
theorem tangent_passes_through_point (a : ℝ) :
  (∀ x : ℝ, ((Real.exp 0 - a) * x + (Real.exp 0 - a * 0) = 0) → x = 1) → a = 2 := by
  sorry

-- Statement 2
theorem no_zeros_on_interval (a : ℝ) :
  (∀ x : ℝ, -1 < x → f a x ≠ 0) → -1 / Real.exp 1 ≤ a ∧ a < Real.exp 1 := by
  sorry

-- Statement 3
theorem inequality_holds (x : ℝ) :
  f 1 x ≥ (1 + x) / (f 1 x + x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_passes_through_point_no_zeros_on_interval_inequality_holds_l491_49127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_theorem_l491_49131

-- Define the sphere
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

-- Define the point on the sphere's surface
def M : ℝ × ℝ × ℝ := (3, 2, 1)

-- Theorem statement
theorem sphere_radius_theorem (S : Sphere) : 
  (S.radius = |S.center.1| ∧ S.radius = |S.center.2.1| ∧ S.radius = |S.center.2.2|) → -- Sphere is tangent to three planes
  ((S.center.1 - M.1)^2 + (S.center.2.1 - M.2.1)^2 + (S.center.2.2 - M.2.2)^2 = S.radius^2) → -- M is on the sphere's surface
  S.radius = 3 + Real.sqrt 2 ∨ S.radius = 3 - Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_theorem_l491_49131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_cube_edge_length_l491_49189

-- Define the cone parameters
noncomputable def side_length : ℝ := 2
noncomputable def cone_height : ℝ := Real.sqrt 3

-- Define the function for the volume of the cube
def cube_volume (edge : ℝ) : ℝ := edge ^ 3

-- Define the constraint function based on the cone's geometry
def constraint (edge : ℝ) : Prop := 
  edge ≤ side_length ∧ edge ≤ cone_height - (edge / 2)

-- Theorem statement
theorem largest_cube_edge_length :
  ∃ (max_edge : ℝ), 
    constraint max_edge ∧ 
    (∀ (edge : ℝ), constraint edge → cube_volume edge ≤ cube_volume max_edge) ∧
    max_edge = 3 * Real.sqrt 2 - 2 * Real.sqrt 3 := by
  sorry

-- Additional helper lemmas if needed
lemma helper_lemma : True := by
  trivial

#check largest_cube_edge_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_cube_edge_length_l491_49189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_qin_jiushao_mult_count_f_l491_49159

/-- Represents a polynomial function -/
def MyPolynomial (α : Type*) [Semiring α] := List α

/-- Qin Jiushao's method for evaluating a polynomial -/
def qinJiushaoEval (p : MyPolynomial ℚ) (x : ℚ) : ℚ :=
  p.foldl (fun acc a => acc * x + a) 0

/-- Counts the number of multiplication operations in Qin Jiushao's method -/
def qinJiushaoMultCount (p : MyPolynomial ℚ) : ℕ :=
  p.length - 1

/-- The given polynomial function -/
def f : MyPolynomial ℚ := [5, 4, 3, 2, 1, 1]

theorem qin_jiushao_mult_count_f :
  qinJiushaoMultCount f = 5 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_qin_jiushao_mult_count_f_l491_49159

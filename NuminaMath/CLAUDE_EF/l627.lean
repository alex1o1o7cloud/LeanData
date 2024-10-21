import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_properties_l627_62771

/-- The numerator of the rational function -/
noncomputable def numerator : ℝ → ℝ := λ x ↦ x^3 - 3*x^2 - 4*x + 12

/-- The denominator of the rational function -/
noncomputable def q : ℝ → ℝ := λ x ↦ 2*x^2 - 8*x + 6

/-- The rational function -/
noncomputable def f : ℝ → ℝ := λ x ↦ numerator x / q x

theorem rational_function_properties :
  (∀ x : ℝ, x ≠ 3 → x ≠ 1 → q x ≠ 0) ∧
  (Filter.Tendsto f Filter.atTop Filter.atTop) ∧
  q (-1) = 16 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_properties_l627_62771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marbles_distribution_l627_62785

/-- The number of marbles each boy has, given a parameter x -/
def marbles (x : ℚ) : Fin 3 → ℚ
  | 0 => 2*x+2
  | 1 => 3*x
  | 2 => x+4

/-- The total number of marbles -/
def total_marbles : ℕ := 56

/-- Theorem stating that the value of x that satisfies the conditions is 25/3 -/
theorem marbles_distribution (x : ℚ) : 
  (Finset.sum Finset.univ (marbles x)) = total_marbles ↔ x = 25/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marbles_distribution_l627_62785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_fixed_point_l627_62758

noncomputable section

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 6

-- Define the relationship between P, Q, and M
def point_relationship (P Q M : ℝ × ℝ) : Prop :=
  (1 - Real.sqrt 3) * Q.1 = P.1 - Real.sqrt 3 * M.1 ∧
  (1 - Real.sqrt 3) * Q.2 = P.2 - Real.sqrt 3 * M.2

-- Define the trajectory E
def trajectory_E (x y : ℝ) : Prop := x^2 / 6 + y^2 / 2 = 1

-- Define the line l passing through (2,0)
def line_l (m : ℝ) (x y : ℝ) : Prop := x = m * y + 2

-- Define the constant value
def constant_value : ℝ := -5/9

theorem trajectory_and_fixed_point :
  ∀ (P Q M : ℝ × ℝ),
  circle_O P.1 P.2 →
  Q.2 = 0 →
  point_relationship P Q M →
  (∀ (x y : ℝ), trajectory_E x y ↔ ∃ (M : ℝ × ℝ), point_relationship P Q M ∧ M.1 = x ∧ M.2 = y) ∧
  (∀ (m : ℝ) (A B : ℝ × ℝ),
    line_l m A.1 A.2 →
    line_l m B.1 B.2 →
    trajectory_E A.1 A.2 →
    trajectory_E B.1 B.2 →
    let D : ℝ × ℝ := (7/3, 0)
    let DA : ℝ × ℝ := (A.1 - D.1, A.2 - D.2)
    let DB : ℝ × ℝ := (B.1 - D.1, B.2 - D.2)
    let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
    DA.1 * AB.1 + DA.2 * AB.2 + DA.1^2 + DA.2^2 = constant_value) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_fixed_point_l627_62758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_best_fit_function_l627_62783

noncomputable def data_points : List (Real × Real) := [(0.50, -0.99), (0.99, 0.01), (2.01, 0.98), (3.98, 2.00)]

def function_a (x : Real) : Real := 2 * x
def function_b (x : Real) : Real := x^2 - 1
def function_c (x : Real) : Real := 2 * x - 2
noncomputable def function_d (x : Real) : Real := Real.log x / Real.log 2

def error_sum (f : Real → Real) (points : List (Real × Real)) : Real :=
  points.foldl (fun sum (x, y) => sum + (f x - y)^2) 0

theorem best_fit_function :
  error_sum function_d data_points < min 
    (error_sum function_a data_points)
    (min (error_sum function_b data_points) (error_sum function_c data_points)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_best_fit_function_l627_62783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_pi_over_3_l627_62707

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem angle_B_is_pi_over_3 (t : Triangle) :
  (Real.sin t.A - Real.sin t.C) / (t.b + t.c) = (Real.sin t.B - Real.sin t.C) / t.a →
  t.B = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_pi_over_3_l627_62707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_addition_theorem_l627_62714

/-- Converts a list of bits to a natural number -/
def bitsToNat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a natural number to a list of bits -/
def natToBits (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec go (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: go (m / 2)
    go n

theorem binary_addition_theorem :
  let a := [true, false, true, true, false, true]  -- 101101₂
  let b := [true, true, false, true]               -- 1011₂
  let c := [true, false, false, true, true]        -- 11001₂
  let d := [true, false, true, false, true, true, true] -- 1110101₂
  let e := [true, true, true, true]                -- 1111₂
  let sum := [true, false, false, true, false, false, false, true] -- 10010001₂
  bitsToNat a + bitsToNat b + bitsToNat c + bitsToNat d + bitsToNat e = bitsToNat sum := by
  sorry

#eval natToBits (bitsToNat [true, false, true, true, false, true] +
                 bitsToNat [true, true, false, true] +
                 bitsToNat [true, false, false, true, true] +
                 bitsToNat [true, false, true, false, true, true, true] +
                 bitsToNat [true, true, true, true])

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_addition_theorem_l627_62714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_allocation_optimal_groups_final_planting_time_l627_62774

/-- Represents the time taken to plant trees based on the number of volunteers in Group A -/
def planting_time (x : ℕ) : ℚ :=
  max (60 / x) (100 / (52 - x))

/-- Theorem stating the optimal allocation of volunteers -/
theorem optimal_allocation :
  ∀ x : ℕ, 0 < x ∧ x < 52 → planting_time x ≥ planting_time 20 := by
  sorry

/-- Corollary confirming the optimal allocation -/
theorem optimal_groups :
  planting_time 20 < planting_time 19 ∧
  planting_time 20 < planting_time 21 := by
  sorry

/-- The final duration of the tree planting activity after reallocation -/
def final_duration : ℚ := 3 + 6/7

/-- Theorem stating the final duration of the tree planting activity -/
theorem final_planting_time :
  final_duration = 3 + 6/7 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_allocation_optimal_groups_final_planting_time_l627_62774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_period_f_monotonic_increasing_l627_62777

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x)^2 + Real.cos (2 * x)

-- Theorem for the minimum positive period
theorem f_min_period : ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧ p = π :=
sorry

-- Theorem for monotonic increasing interval
theorem f_monotonic_increasing :
  ∀ (x y : ℝ), 0 ≤ x ∧ x < y ∧ y ≤ π / 8 → f x < f y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_period_f_monotonic_increasing_l627_62777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_pyramid_volume_l627_62721

/-- Represents the volume of a pyramid with a square base -/
noncomputable def pyramid_volume (side_length : ℝ) (height : ℝ) : ℝ :=
  (1/3) * side_length^2 * height

theorem new_pyramid_volume 
  (original_volume : ℝ) 
  (original_side : ℝ) 
  (original_height : ℝ) 
  (h1 : original_volume = 60)
  (h2 : pyramid_volume original_side original_height = original_volume)
  : pyramid_volume (3 * original_side) (4 * original_height) = 2160 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_pyramid_volume_l627_62721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixed_fraction_division_simplification_l627_62722

theorem mixed_fraction_division_simplification :
  (7 + 4480 / 8333) / (21934 / 25909) / (1 + 18556 / 35255) = 35 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixed_fraction_division_simplification_l627_62722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_difference_bound_l627_62779

theorem product_difference_bound (n : ℕ) (hn : n > 0) :
  ∀ t k : ℕ, t > 0 → k > 0 → t * (t + k) = n^2 + n + 1 → k ≥ 2 * Real.sqrt (n : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_difference_bound_l627_62779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_formulas_correct_l627_62764

/-- The area of a curvilinear trapezoid bounded by y = a^x, x-axis, y-axis, and x = b -/
noncomputable def curvilinear_trapezoid_area (a b : ℝ) : ℝ := (a^b - 1) / Real.log a

/-- The area of a curvilinear triangle bounded by y = log_a x, x-axis, and x = b -/
noncomputable def curvilinear_triangle_area (a b : ℝ) : ℝ := (b * Real.log b - b + 1) / Real.log a

/-- Theorem stating the correctness of the area formulas -/
theorem area_formulas_correct (a b : ℝ) (ha : a > 0) (hb : b > 1) :
  (curvilinear_trapezoid_area a b = ∫ x in (0 : ℝ)..b, a^x) ∧
  (curvilinear_triangle_area a b = b * (Real.log b / Real.log a) - ∫ x in (1 : ℝ)..b, (Real.log x / Real.log a)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_formulas_correct_l627_62764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_problem_l627_62793

/-- Represents a simple interest loan -/
structure SimpleLoan where
  principal : ℚ
  rate : ℚ
  time : ℚ

/-- Calculates the simple interest for a given loan -/
def simpleInterest (loan : SimpleLoan) : ℚ :=
  (loan.principal * loan.rate * loan.time) / 100

/-- Theorem stating the conditions and conclusion of the loan problem -/
theorem loan_problem (loan : SimpleLoan) 
  (h1 : loan.time = loan.rate)
  (h2 : simpleInterest loan = 432)
  (h3 : loan.rate = 6) : 
  loan.principal = 1200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_problem_l627_62793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_suitcase_weight_l627_62782

-- Define constants for conversion rates
noncomputable def ounces_per_pound : ℝ := 16
noncomputable def pounds_per_kilogram : ℝ := 2.20462
noncomputable def ounces_per_gram : ℝ := 0.03527396

-- Define the weights of items
noncomputable def initial_weight : ℝ := 12
noncomputable def perfume_weight : ℝ := 5 * 1.2 / ounces_per_pound
noncomputable def chocolate_weight : ℝ := 4 + 1.5 + 3.25
noncomputable def soap_weight : ℝ := 2 * 5 / ounces_per_pound
noncomputable def jam_weight : ℝ := (8 + 6 + 10 + 12) / ounces_per_pound
noncomputable def sculpture_weight : ℝ := 3.5 * pounds_per_kilogram
noncomputable def shirt_weight : ℝ := 3 * 300 * ounces_per_gram / ounces_per_pound
noncomputable def cookie_weight : ℝ := 450 * ounces_per_gram / ounces_per_pound
noncomputable def wine_weight : ℝ := 190 * ounces_per_gram / ounces_per_pound

-- Theorem statement
theorem final_suitcase_weight :
  initial_weight + perfume_weight + chocolate_weight + soap_weight +
  jam_weight + sculpture_weight + shirt_weight + cookie_weight + wine_weight =
  35.111288 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_suitcase_weight_l627_62782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_2023_terms_l627_62728

def sequence_a : ℕ → ℤ
  | 0 => 1  -- Adding case for 0
  | 1 => 1
  | 2 => 2
  | (n + 3) => if sequence_a (n + 1) < sequence_a (n + 2) 
                then sequence_a (n + 2) - sequence_a (n + 1)
                else sequence_a (n + 1) - sequence_a (n + 2)

theorem sum_of_2023_terms : 
  (Finset.range 2023).sum (λ i => sequence_a i) = 1351 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_2023_terms_l627_62728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jesse_room_width_l627_62719

/-- Represents the properties of Jesse's rooms -/
structure JesseRooms where
  length : ℚ
  num_rooms : ℕ
  total_area : ℚ

/-- Calculates the width of Jesse's rooms given their properties -/
def room_width (rooms : JesseRooms) : ℚ :=
  rooms.total_area / (rooms.num_rooms * rooms.length)

/-- Theorem stating that Jesse's rooms are 18 feet wide -/
theorem jesse_room_width :
  let rooms : JesseRooms := {
    length := 19,
    num_rooms := 20,
    total_area := 6840
  }
  room_width rooms = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jesse_room_width_l627_62719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projectiles_meeting_time_l627_62796

/-- The time (in minutes) taken for two projectiles to meet, given their initial distance and speeds. -/
noncomputable def time_to_meet (initial_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) : ℝ :=
  initial_distance / (speed1 + speed2) * 60

/-- Theorem stating that two projectiles launched 1386 km apart with speeds 445 km/h and 545 km/h
    will meet in 84 minutes. -/
theorem projectiles_meeting_time :
  time_to_meet 1386 445 545 = 84 := by
  sorry

-- Remove the #eval line as it's not computable
-- #eval time_to_meet 1386 445 545

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projectiles_meeting_time_l627_62796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l627_62712

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 3 * Real.sin (2 * ω * x + Real.pi / 3)

theorem function_properties (ω : ℝ) (h1 : 0 < ω) (h2 : ω < 2)
  (h3 : ∀ x, f ω x = f ω (-x - Real.pi / 3)) :
  ω = 1 ∧
  (∀ x, f ω (x + Real.pi) = f ω x) ∧
  (∀ k : ℤ, StrictMonoOn (f ω) (Set.Icc (Real.pi / 12 + k * Real.pi) (5 * Real.pi / 12 + k * Real.pi))) ∧
  (∀ x, f ω x ≥ 3 / 2 ↔ ∃ k : ℤ, x ∈ Set.Icc (Real.pi / 12 + k * Real.pi) (5 * Real.pi / 12 + k * Real.pi)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l627_62712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_permutation_equals_seven_l627_62768

/-- Represents the three possible operations -/
inductive Operation
  | Add
  | Sub
  | Div
  deriving DecidableEq

/-- Applies an operation to two numbers -/
def applyOp (op : Operation) (a b : ℚ) : ℚ :=
  match op with
  | Operation.Add => a + b
  | Operation.Sub => a - b
  | Operation.Div => a / b

/-- Evaluates the expression given a list of operations -/
def evaluateExpression (ops : List Operation) : ℚ :=
  match ops with
  | [op1, op2, op3] => applyOp op3 (applyOp op2 (applyOp op1 7 2) 8) 4
  | _ => 0  -- Invalid case, should not happen

/-- There exists a permutation of operations that results in 7 -/
theorem exists_permutation_equals_seven :
  ∃ (ops : List Operation),
    ops.length = 3 ∧
    ops.toFinset.card = 3 ∧
    evaluateExpression ops = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_permutation_equals_seven_l627_62768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_inequality_l627_62744

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  O : Point3D
  A : Point3D
  B : Point3D
  C : Point3D

/-- Checks if three planes are parallel -/
def are_parallel_planes (p1 p2 p3 : Point3D × Point3D × Point3D) : Prop := sorry

/-- Calculates the volume of a tetrahedron -/
noncomputable def volume (t : Tetrahedron) : ℝ := sorry

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point3D) : ℝ := sorry

/-- Main theorem -/
theorem tetrahedron_volume_inequality
  (O A₁ B₁ C₁ : Point3D)
  (A₂ A₃ : Point3D)
  (h_on_line : ∃ t₂ t₃ : ℝ, 0 ≤ t₂ ∧ t₂ ≤ 1 ∧ 0 ≤ t₃ ∧ t₃ ≤ 1 ∧ 
    A₂ = ⟨O.x + t₂ * (A₁.x - O.x), O.y + t₂ * (A₁.y - O.y), O.z + t₂ * (A₁.z - O.z)⟩ ∧
    A₃ = ⟨O.x + t₃ * (A₁.x - O.x), O.y + t₃ * (A₁.y - O.y), O.z + t₃ * (A₁.z - O.z)⟩)
  (h_parallel : are_parallel_planes (A₁, B₁, C₁) (A₂, B₂, C₂) (A₃, B₃, C₃))
  (h_order : distance O A₁ > distance O A₂ ∧ distance O A₂ > distance O A₃ ∧ distance O A₃ > 0)
  (V₁ := volume ⟨O, A₁, B₁, C₁⟩)
  (V₂ := volume ⟨O, A₂, B₂, C₂⟩)
  (V₃ := volume ⟨O, A₃, B₃, C₃⟩)
  (V := volume ⟨O, A₁, B₂, C₃⟩) :
  V₁ + V₂ + V₃ ≥ 3 * V := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_inequality_l627_62744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_cos_relation_l627_62706

noncomputable section

open Real

def IsAngleOfTriangle (θ : Real) : Prop := 0 < θ ∧ θ < Real.pi

theorem triangle_angle_cos_relation (A B : Real) (h_triangle : IsAngleOfTriangle A ∧ IsAngleOfTriangle B) :
  A > B ↔ cos A < cos B := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_cos_relation_l627_62706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_rate_proof_l627_62790

/-- The diameter of the circular field in meters -/
def diameter : ℝ := 22

/-- The total cost of fencing in Rupees -/
def total_cost : ℝ := 207.34511513692632

/-- Pi (π) constant -/
noncomputable def π : ℝ := Real.pi

/-- The circumference of the circular field -/
noncomputable def circumference : ℝ := π * diameter

/-- The rate per meter for fencing -/
noncomputable def rate_per_meter : ℝ := total_cost / circumference

theorem fencing_rate_proof : 
  ∀ ε > 0, |rate_per_meter - 3| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_rate_proof_l627_62790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_p_trajectory_l627_62775

/-- The trajectory of point P satisfying the given conditions -/
theorem point_p_trajectory (x y : ℝ) : 
  ((-2 - x) * (2 - x) + (-y) * (-y) = -x^2) → (x^2 / 2 + y^2 / 4 = 1) := by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_p_trajectory_l627_62775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_value_problem_l627_62743

theorem least_value_problem (x y z w : ℕ) 
  (hx1 : x % 9 = 2) (hx2 : x % 7 = 4)
  (hy1 : y % 11 = 3) (hy2 : y % 13 = 12)
  (hz1 : z % 17 = 8) (hz2 : z % 19 = 6)
  (hw1 : w % 23 = 5) (hw2 : w % 29 = 10) :
  ∃ (x' y' z' w' : ℕ), 
    (x' % 9 = 2) ∧ (x' % 7 = 4) ∧
    (y' % 11 = 3) ∧ (y' % 13 = 12) ∧
    (z' % 17 = 8) ∧ (z' % 19 = 6) ∧
    (w' % 23 = 5) ∧ (w' % 29 = 10) ∧
    (y' + z' : ℤ) - (x' + w') = -326 ∧
    ∀ (a b c d : ℕ), 
      (a % 9 = 2) → (a % 7 = 4) →
      (b % 11 = 3) → (b % 13 = 12) →
      (c % 17 = 8) → (c % 19 = 6) →
      (d % 23 = 5) → (d % 29 = 10) →
      (b + c : ℤ) - (a + d) ≥ -326 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_value_problem_l627_62743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_l_l627_62736

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x-2) + 1

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y - 7 = 0

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y - 2 = k*(x - 1)

-- State the theorem
theorem slope_of_line_l (a : ℝ) (k : ℝ) :
  a > 0 →
  a ≠ 1 →
  f a 2 = 2 →
  (∃ x y : ℝ, circle_C x y ∧ line_l k x y) →
  (∃ x1 y1 x2 y2 : ℝ,
    circle_C x1 y1 ∧ circle_C x2 y2 ∧
    line_l k x1 y1 ∧ line_l k x2 y2 ∧
    (x2 - x1)^2 + (y2 - y1)^2 = 18) →
  k = -1 ∨ k = -7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_l_l627_62736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_of_2_l627_62795

noncomputable def t (x : ℝ) : ℝ := 2 * x^2 - 5 * x + 1

noncomputable def s (y : ℝ) : ℝ := 
  let x := (5 + Real.sqrt 33) / 4
  x^3 - 4 * x^2 + x + 6

theorem s_of_2 : s 2 = ((5 + Real.sqrt 33) / 4)^3 - 4 * ((5 + Real.sqrt 33) / 4)^2 + ((5 + Real.sqrt 33) / 4) + 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_of_2_l627_62795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_steven_farmland_acres_l627_62787

/-- Represents the farmer's capabilities and land information -/
structure FarmerData where
  plow_rate : ℚ  -- acres of farmland plowed per day
  mow_rate : ℚ   -- acres of grassland mowed per day
  total_days : ℚ -- total days to complete both tasks
  grassland : ℚ  -- acres of grassland

/-- Calculates the acres of farmland given farmer data -/
def farmland_acres (data : FarmerData) : ℚ :=
  (data.total_days - data.grassland / data.mow_rate) * data.plow_rate

/-- Theorem stating that Steven has 55 acres of farmland -/
theorem steven_farmland_acres :
  let data : FarmerData := {
    plow_rate := 10,
    mow_rate := 12,
    total_days := 8,
    grassland := 30
  }
  farmland_acres data = 55 := by
  -- Proof goes here
  sorry

#eval farmland_acres {
  plow_rate := 10,
  mow_rate := 12,
  total_days := 8,
  grassland := 30
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_steven_farmland_acres_l627_62787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l627_62702

/-- A right isosceles triangle ABC with side length 4 -/
structure RightIsoscelesTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  right_angle : A.1 = 0 ∧ A.2 = 0 ∧ B.1 = 4 ∧ B.2 = 0 ∧ C.1 = 0 ∧ C.2 = 4

/-- A point D inside the triangle ABC -/
structure PointInTriangle (t : RightIsoscelesTriangle) where
  D : ℝ × ℝ
  inside : 0 < D.1 ∧ D.1 < 4 ∧ 0 < D.2 ∧ D.2 < 4 ∧ D.2 < -D.1 + 4

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The theorem statement -/
theorem min_distance_sum (t : RightIsoscelesTriangle) (p : PointInTriangle t) :
  distance t.A p.D = Real.sqrt 2 →
  distance t.B p.D + distance t.C p.D ≥ 2 * Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l627_62702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beat_distance_approx_87_l627_62760

/-- A kilometer race where runner A beats runner B --/
structure Race where
  length : ℝ  -- Race length in meters
  time_A : ℝ  -- Time taken by runner A in seconds
  time_diff : ℝ  -- Time difference between A and B in seconds

/-- Calculate the distance by which runner A beats runner B --/
noncomputable def beatDistance (race : Race) : ℝ :=
  (race.length / race.time_A) * race.time_diff

/-- Theorem stating that in the given race conditions, A beats B by approximately 87 meters --/
theorem beat_distance_approx_87 (race : Race) 
  (h1 : race.length = 1000) 
  (h2 : race.time_A = 92) 
  (h3 : race.time_diff = 8) : 
  ∃ ε > 0, |beatDistance race - 87| < ε := by
  sorry

#eval Float.round ((1000 / 92) * 8)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beat_distance_approx_87_l627_62760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_nine_equals_899_l627_62709

-- Define the polynomial f
def f (x : ℝ) : ℝ := x^3 + x + 1

-- Define g as a cubic polynomial with the given properties
noncomputable def g (x : ℝ) : ℝ := sorry

-- State the properties of g
axiom g_cubic : ∃ a b c d : ℝ, ∀ x, g x = a*x^3 + b*x^2 + c*x + d
axiom g_zero : g 0 = -1
axiom g_roots : ∀ r : ℝ, f r = 0 → g (r^2) = 0

-- The theorem to prove
theorem g_nine_equals_899 : g 9 = 899 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_nine_equals_899_l627_62709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_problems_l627_62753

theorem absolute_value_problems :
  (∀ (x : ℤ), |10 - (-6)| = 16) ∧
  (∀ (m : ℤ), |m - 3| = 5 ↔ (m = 8 ∨ m = -2)) ∧
  (∀ (m : ℤ), |m - 4| + |m + 2| = 6 ↔ m ∈ ({-2, -1, 0, 1, 2, 3, 4} : Set ℤ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_problems_l627_62753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersections_theorem_l627_62746

/-- Given a circle of radius R and four smaller circles constructed on its radii,
    this theorem states that the total area of pairwise intersections of the smaller circles
    is equal to πR²/8, which is also equal to the area of the original circle
    outside these four smaller circles. -/
theorem circle_intersections_theorem (R : ℝ) (R_pos : R > 0) :
  let big_circle_area := π * R^2
  let small_circle_radius := R / 2
  let small_circle_area := π * small_circle_radius^2
  let total_small_circles_area := 4 * small_circle_area
  let pairwise_intersection_area := π * R^2 / 8
  let area_outside_small_circles := big_circle_area - (total_small_circles_area - 2 * pairwise_intersection_area)
  pairwise_intersection_area = π * R^2 / 8 ∧
  area_outside_small_circles = π * R^2 / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersections_theorem_l627_62746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_rounds_for_change_l627_62701

/-- Represents the state of the tennis tournament after each round. -/
structure TournamentState (N : ℕ) where
  playerPositions : Fin (2 * N) → Fin N

/-- Represents the movement of players after a round. -/
def nextRound (N : ℕ) (state : TournamentState N) : TournamentState N :=
  sorry

/-- Checks if players 2 to N+1 have all changed courts. -/
def allPlayersChanged (N : ℕ) (initialState finalState : TournamentState N) : Prop :=
  sorry

/-- The main theorem stating the minimum number of rounds required. -/
theorem min_rounds_for_change (N : ℕ) (h : N ≥ 2) :
  ∃ M : ℕ, (∀ initialState : TournamentState N,
    ∃ finalState : TournamentState N,
      (finalState = (Nat.iterate (nextRound N) M initialState)) ∧
      (allPlayersChanged N initialState finalState)) ∧
    (M = if N = 2 then 2 else N + 1) := by
  sorry

/-- Helper lemma: There exists a minimum number of rounds for all players to change courts. -/
lemma exists_min_rounds (N : ℕ) (h : N ≥ 2) :
  ∃ M : ℕ, ∀ initialState : TournamentState N,
    ∃ finalState : TournamentState N,
      (finalState = (Nat.iterate (nextRound N) M initialState)) ∧
      (allPlayersChanged N initialState finalState) := by
  sorry

/-- Helper lemma: The minimum number of rounds is at most N+1 for N ≥ 3. -/
lemma min_rounds_upper_bound (N : ℕ) (h : N ≥ 3) :
  ∃ M : ℕ, M ≤ N + 1 ∧ ∀ initialState : TournamentState N,
    ∃ finalState : TournamentState N,
      (finalState = (Nat.iterate (nextRound N) M initialState)) ∧
      (allPlayersChanged N initialState finalState) := by
  sorry

/-- Helper lemma: For N = 2, the minimum number of rounds is 2. -/
lemma min_rounds_n_eq_two :
  ∃ M : ℕ, M = 2 ∧ ∀ initialState : TournamentState 2,
    ∃ finalState : TournamentState 2,
      (finalState = (Nat.iterate (nextRound 2) M initialState)) ∧
      (allPlayersChanged 2 initialState finalState) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_rounds_for_change_l627_62701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_money_loses_value_without_partners_l627_62750

/-- Represents the presence of other individuals to trade with -/
def HasTradingPartners : Prop := sorry

/-- Represents the function of money as a medium of exchange -/
def MoneyAsExchangeMedium : Prop := sorry

/-- Represents the value of money in a given context -/
def MoneyHasValue : Prop := sorry

/-- The primary economic function of money is to serve as a medium of exchange -/
axiom money_primary_function : MoneyAsExchangeMedium → MoneyHasValue

/-- Money requires trading partners to function as a medium of exchange -/
axiom exchange_requires_partners : MoneyAsExchangeMedium → HasTradingPartners

/-- Theorem: Money loses its value when there are no trading partners -/
theorem money_loses_value_without_partners : ¬HasTradingPartners → ¬MoneyHasValue := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_money_loses_value_without_partners_l627_62750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2theta_value_l627_62754

theorem cos_2theta_value (θ : ℝ) (h : ∑' n, (Real.cos θ)^(2 * n) = 5) : Real.cos (2 * θ) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2theta_value_l627_62754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_operations_l627_62784

def horner_polynomial (x : ℝ) : ℝ := 1 + 2*x + x^2 - 3*x^3 + 2*x^4

def horner_operations : ℕ := 8

-- Define a function to count operations (this is a placeholder)
def count_operations (f : ℝ → ℝ) (x : ℝ) : ℕ := sorry

theorem horner_method_operations :
  horner_operations = count_operations horner_polynomial (-1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_operations_l627_62784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_is_pi_over_four_side_a_is_sqrt_two_l627_62748

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions of the problem
axiom triangle_exists : ∃ (t : Triangle),
  0 < t.A ∧ t.A < Real.pi ∧
  0 < t.B ∧ t.B < Real.pi ∧
  0 < t.C ∧ t.C < Real.pi ∧
  t.b * (Real.cos t.A) - t.a * (Real.sin t.B) = 0 ∧
  t.b = Real.sqrt 2 ∧
  (1/2) * t.a * t.b * (Real.sin t.C) = 1

-- Theorem 1: Prove that angle A is π/4
theorem angle_A_is_pi_over_four (t : Triangle) 
  (h : t.b * (Real.cos t.A) - t.a * (Real.sin t.B) = 0 ∧ 
       0 < t.A ∧ t.A < Real.pi ∧ 
       0 < t.B ∧ t.B < Real.pi) : 
  t.A = Real.pi/4 := by sorry

-- Theorem 2: Prove that side a is √2
theorem side_a_is_sqrt_two (t : Triangle) 
  (h : t.b = Real.sqrt 2 ∧ 
       t.A = Real.pi/4 ∧ 
       (1/2) * t.a * t.b * (Real.sin t.C) = 1) : 
  t.a = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_is_pi_over_four_side_a_is_sqrt_two_l627_62748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l627_62794

theorem diophantine_equation_solutions : 
  {(m, n) : ℕ × ℕ | 200 * m + 6 * n = 2006} = 
  {(1, 301), (4, 201), (7, 101), (10, 1)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l627_62794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l627_62762

-- Define the line equation
def line_equation (k x y : ℝ) : Prop := k * x - y + 2 + k = 0

-- Define the condition for not passing through the fourth quadrant
def not_in_fourth_quadrant (k : ℝ) : Prop := k ≥ 0

-- Define the intersection points
noncomputable def point_A (k : ℝ) : ℝ × ℝ := (-((k + 2) / k), 0)
noncomputable def point_B (k : ℝ) : ℝ × ℝ := (0, k + 2)

-- Define the area of triangle AOB
noncomputable def triangle_area (k : ℝ) : ℝ := (1 / 2) * (k + 4 + 4 / k)

-- Theorem statement
theorem line_properties :
  ∀ k : ℝ,
  not_in_fourth_quadrant k →
  (∃ A B : ℝ × ℝ, A = point_A k ∧ B = point_B k) →
  (k ∈ Set.Ici 0) ∧
  (∀ S : ℝ, triangle_area k ≥ 4) ∧
  (∃ k₀ : ℝ, k₀ = 2 ∧ ∀ x y : ℝ, line_equation k₀ x y ↔ y = 2 * x + 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l627_62762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l627_62745

-- Define the power function
noncomputable def power_function (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

-- Theorem statement
theorem power_function_value : 
  ∃ α : ℝ, (power_function α 2 = 1/4) ∧ (power_function α (-3) = 1/9) := by
  -- Introduce α and prove its existence
  let α := -2
  use α
  
  -- Split the conjunction
  constructor
  
  -- Prove power_function α 2 = 1/4
  · simp [power_function]
    norm_num
  
  -- Prove power_function α (-3) = 1/9
  · simp [power_function]
    norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l627_62745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_circumference_l627_62769

/-- The circumference of the base of a right circular cone -/
noncomputable def base_circumference (volume : ℝ) (height : ℝ) : ℝ :=
  2 * Real.pi * (3 * volume / (Real.pi * height))^(1/2)

/-- Theorem: The circumference of the base of a right circular cone with volume 18π and height 6 is 6π -/
theorem cone_base_circumference : 
  base_circumference (18 * Real.pi) 6 = 6 * Real.pi := by
  -- Unfold the definition of base_circumference
  unfold base_circumference
  -- Simplify the expression
  simp [Real.pi]
  -- The rest of the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_circumference_l627_62769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_squared_l627_62738

/-- Represents a cuboctahedron -/
structure Cuboctahedron where
  side_length : ℝ
  square_faces : ℕ
  triangle_faces : ℕ
  hSquares : square_faces = 6
  hTriangles : triangle_faces = 8

/-- Represents an octahedron -/
structure Octahedron where
  side_length : ℝ

/-- Calculates the volume of a cuboctahedron -/
noncomputable def cuboctahedron_volume (c : Cuboctahedron) : ℝ := 
  (5 * Real.sqrt 2 * c.side_length ^ 3) / 3

/-- Calculates the volume of an octahedron -/
noncomputable def octahedron_volume (o : Octahedron) : ℝ := 
  (Real.sqrt 2 * o.side_length ^ 3) / 3

/-- The main theorem to prove -/
theorem volume_ratio_squared (c : Cuboctahedron) (o : Octahedron) 
    (h : c.side_length = o.side_length) : 
    100 * ((octahedron_volume o) / (cuboctahedron_volume c))^2 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_squared_l627_62738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concyclic_iff_perpendicular_l627_62788

-- Define the basic structures
structure Line where
  p : ℝ × ℝ
  q : ℝ × ℝ

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a membership relation for points in lines and circles
def pointOnLine (point : ℝ × ℝ) (line : Line) : Prop := sorry
def pointOnCircle (point : ℝ × ℝ) (circle : Circle) : Prop := sorry

-- Define the given points and lines
noncomputable def P : ℝ × ℝ := sorry
noncomputable def l₁ : Line := sorry
noncomputable def l₂ : Line := sorry

-- Define the circles
noncomputable def S₁ : Circle := sorry
noncomputable def S₂ : Circle := sorry
noncomputable def T₁ : Circle := sorry
noncomputable def T₂ : Circle := sorry

-- Define the intersection points
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry
noncomputable def C : ℝ × ℝ := sorry
noncomputable def D : ℝ × ℝ := sorry

-- Define the properties of the configuration
axiom circles_tangent_at_P :
  (S₁.center = P ∨ S₁.radius = 0) ∧
  (S₂.center = P ∨ S₂.radius = 0) ∧
  (T₁.center = P ∨ T₁.radius = 0) ∧
  (T₂.center = P ∨ T₂.radius = 0)

axiom circles_tangent_to_lines :
  (pointOnLine S₁.center l₁ ∨ S₁.radius = 0) ∧
  (pointOnLine S₂.center l₁ ∨ S₂.radius = 0) ∧
  (pointOnLine T₁.center l₂ ∨ T₁.radius = 0) ∧
  (pointOnLine T₂.center l₂ ∨ T₂.radius = 0)

axiom A_is_intersection : pointOnCircle A S₁ ∧ pointOnCircle A T₁ ∧ A ≠ P
axiom B_is_intersection : pointOnCircle B S₁ ∧ pointOnCircle B T₂ ∧ B ≠ P
axiom C_is_intersection : pointOnCircle C S₂ ∧ pointOnCircle C T₁ ∧ C ≠ P
axiom D_is_intersection : pointOnCircle D S₂ ∧ pointOnCircle D T₂ ∧ D ≠ P

-- Define concyclicity and perpendicularity
def are_concyclic (a b c d : ℝ × ℝ) : Prop := sorry
def are_perpendicular (l₁ l₂ : Line) : Prop := sorry

-- The main theorem
theorem concyclic_iff_perpendicular :
  are_concyclic A B C D ↔ are_perpendicular l₁ l₂ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concyclic_iff_perpendicular_l627_62788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_k_values_l627_62797

/-- Represents a tile shape -/
inductive TileShape
  | L
  | Rectangle
deriving BEq, Repr

/-- Represents a tiling configuration -/
structure Tiling where
  tiles : List TileShape
  valid : tiles.length = 12

/-- Checks if a tiling is valid for a 6x6 square -/
def isValidTiling (t : Tiling) : Prop :=
  t.tiles.length = 12 ∧
  t.tiles.count TileShape.L + 2 * t.tiles.count TileShape.Rectangle = 18

/-- Theorem stating the possible values of k for valid 6x6 tilings -/
theorem valid_k_values :
  ∀ k : ℕ,
  (∃ t : Tiling, isValidTiling t ∧ t.tiles.count TileShape.L = k) ↔
  k ∈ ({2, 4, 5, 6, 7, 8, 9, 10, 11, 12} : Set ℕ) :=
by
  sorry

#check valid_k_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_k_values_l627_62797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_but_not_sufficient_l627_62789

theorem necessary_but_not_sufficient (m a : ℝ) (ha : a ≠ 0) :
  (∀ m, |m| = a → m ∈ ({-a, a} : Set ℝ)) ∧
  (∃ m, m ∈ ({-a, a} : Set ℝ) ∧ |m| ≠ a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_but_not_sufficient_l627_62789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_equation_l627_62756

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0.5 then 1 / (0.5 - x) else 0.5

-- State the theorem
theorem f_satisfies_equation :
  ∀ x : ℝ, f x + (0.5 + x) * f (1 - x) = 1 := by
  intro x
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_equation_l627_62756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l627_62705

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (2*x + f y) = x + y + f x) →
  (∀ x : ℝ, f x = x) :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l627_62705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l627_62725

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sqrt 3 * (cos (x / 2))^2 - (1 / 2) * sin x - sqrt 3 / 2

-- State the theorem
theorem f_monotone_increasing :
  ∀ x y, x ∈ Set.Icc (5 * π / 6) π → y ∈ Set.Icc (5 * π / 6) π → x ≤ y → f x ≤ f y :=
by
  sorry

-- Optionally, you can add a lemma to show that the function is equivalent to cos(x + π/6)
lemma f_eq_cos_shifted :
  ∀ x, f x = cos (x + π / 6) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l627_62725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l627_62799

def f (x : ℕ) : ℕ := Finset.prod (Finset.range (x / 2 + 1)) (fun i => 2 * (i + 1))

theorem problem_statement (x : ℕ) : 
  Even x → 
  (∃ (y : ℕ), f x + 12 = 13 * y ∧ ∀ p, Nat.Prime p → p ∣ (f x + 12) → p ≤ 13) → 
  x = 6 :=
by sorry

#eval f 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l627_62799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonals_bisect_in_special_quadrilaterals_l627_62763

-- Define the basic shapes
class Quadrilateral (α : Type*) where
  -- Add any necessary properties here

class Rectangle (α : Type*) extends Quadrilateral α where
  -- Add any necessary properties here

class Rhombus (α : Type*) extends Quadrilateral α where
  -- Add any necessary properties here

class Square (α : Type*) extends Rectangle α, Rhombus α where
  -- Add any necessary properties here

-- Define the property of diagonals bisecting each other
def diagonalsBisectEachOther {α : Type*} (q : Quadrilateral α) : Prop := sorry

-- Theorem stating that rectangles, rhombuses, and squares have diagonals that bisect each other
theorem diagonals_bisect_in_special_quadrilaterals 
  {α : Type*} (r : Rectangle α) (h : Rhombus α) (s : Square α) :
  diagonalsBisectEachOther (r.toQuadrilateral) ∧ 
  diagonalsBisectEachOther (h.toQuadrilateral) ∧ 
  diagonalsBisectEachOther (s.toQuadrilateral) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonals_bisect_in_special_quadrilaterals_l627_62763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_infinite_sum_equals_four_thirds_l627_62727

theorem double_infinite_sum_equals_four_thirds :
  (∑' j : ℕ, ∑' k : ℕ, (2 : ℝ) ^ (-(3 * k + j + (k + j)^2 : ℤ))) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_infinite_sum_equals_four_thirds_l627_62727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sequence_sum_l627_62740

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sequence_term (n : ℕ) : ℕ := factorial n + n

def sequence_sum : ℕ := (List.range 11).map sequence_term |>.sum

theorem units_digit_of_sequence_sum :
  sequence_sum % 10 = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sequence_sum_l627_62740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_theorem_l627_62767

structure MagicalTree where
  bananas : Nat
  oranges : Nat

inductive PickAction
  | one_fruit : PickAction
  | two_identical : PickAction
  | two_different : PickAction

def apply_action (t : MagicalTree) (a : PickAction) : MagicalTree :=
  match a with
  | PickAction.one_fruit => t
  | PickAction.two_identical => { bananas := t.bananas, oranges := t.oranges + 1 }
  | PickAction.two_different => { bananas := t.bananas + 1, oranges := t.oranges }

def initial_tree : MagicalTree := { bananas := 15, oranges := 20 }

theorem tree_theorem :
  ∃ (actions : List PickAction),
    (actions.foldl apply_action initial_tree).bananas + (actions.foldl apply_action initial_tree).oranges = 1 ∧
    (actions.foldl apply_action initial_tree).bananas = 1 ∧
    ∀ (actions : List PickAction),
      (actions.foldl apply_action initial_tree).bananas + (actions.foldl apply_action initial_tree).oranges ≠ 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_theorem_l627_62767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_1000_in_column_C_l627_62720

def column_sequence : ℕ → Fin 6
| n => match n % 10 with
  | 0 => 0  -- A
  | 1 => 1  -- B
  | 2 => 2  -- C
  | 3 => 3  -- D
  | 4 => 4  -- E
  | 5 => 5  -- F
  | 6 => 4  -- E
  | 7 => 3  -- D
  | 8 => 2  -- C
  | _ => 1  -- B (for 9 and any other case)

theorem integer_1000_in_column_C :
  column_sequence 998 = 2 := by
  -- Proof goes here
  sorry

#eval column_sequence 998  -- This will evaluate to 2, confirming our theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_1000_in_column_C_l627_62720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_monotone_increasing_f_monotone_on_pos_reals_l627_62752

-- Define the function f(x) = √x
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

-- State the theorem
theorem sqrt_monotone_increasing :
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry

-- Additional lemma to show f is monotone on (0, +∞)
theorem f_monotone_on_pos_reals :
  Monotone (fun x => f x) := by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_monotone_increasing_f_monotone_on_pos_reals_l627_62752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_range_l627_62734

/-- The curve f(x) = -e^x - x -/
noncomputable def f (x : ℝ) : ℝ := -Real.exp x - x

/-- The curve g(x) = 3ax + 2cos x -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := 3 * a * x + 2 * Real.cos x

/-- The derivative of f(x) -/
noncomputable def f' (x : ℝ) : ℝ := -Real.exp x - 1

/-- The derivative of g(x) -/
noncomputable def g' (a : ℝ) (x : ℝ) : ℝ := 3 * a - 2 * Real.sin x

/-- Theorem stating the range of a given the perpendicularity condition -/
theorem perpendicular_tangents_range (a : ℝ) : 
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, f' x₁ * g' a x₂ = -1) → 
  a ∈ Set.Icc (-1/3) (2/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_range_l627_62734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_parameter_l627_62718

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Distance between two points -/
noncomputable def distance (a b : Point) : ℝ :=
  Real.sqrt ((a.x - b.x)^2 + (a.y - b.y)^2)

/-- Area of a triangle given three points -/
noncomputable def triangleArea (a b c : Point) : ℝ :=
  abs (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)) / 2

theorem parabola_parameter (par : Parabola) 
  (O : Point) 
  (F : Point) 
  (M : Point) :
  O.x = 0 ∧ O.y = 0 →  -- Origin at (0, 0)
  F.x = par.p / 2 ∧ F.y = 0 →  -- Focus at (p/2, 0)
  M.y^2 = 2 * par.p * M.x →  -- M is on the parabola
  distance M F = 4 * distance O F →  -- |MF| = 4|OF|
  triangleArea M F O = 4 * Real.sqrt 3 →  -- Area of triangle MFO is 4√3
  par.p = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_parameter_l627_62718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_4_pow_6_plus_8_pow_5_l627_62731

theorem greatest_prime_factor_of_4_pow_6_plus_8_pow_5 :
  (Nat.factors (4^6 + 8^5)).maximum? = some 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_4_pow_6_plus_8_pow_5_l627_62731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_prism_volume_l627_62776

/-- The volume of an oblique prism with an equilateral triangle base of side length a
    and a lateral rhombus face with diagonal b perpendicular to the base -/
theorem oblique_prism_volume (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : b < 2*a) :
  ∃ V : ℝ, V = (1/8) * a * b * Real.sqrt (12 * a^2 - 3 * b^2) :=
by
  -- The proof would go here
  sorry

#check oblique_prism_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_prism_volume_l627_62776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_less_than_8_of_72_l627_62739

def factors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ x ↦ x ∣ n) (Finset.range (n + 1))

def factorsLessThan (n k : ℕ) : Finset ℕ :=
  Finset.filter (λ x ↦ x < k) (factors n)

theorem probability_factor_less_than_8_of_72 :
  let allFactors := factors 72
  let factorsLess8 := factorsLessThan 72 8
  (Finset.card factorsLess8 : ℚ) / (Finset.card allFactors : ℚ) = 5 / 12 := by
  sorry

#eval factors 72
#eval factorsLessThan 72 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_less_than_8_of_72_l627_62739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_wrt_x_axis_l627_62735

/-- 
Given a point (2, 3, 4) in 3D space, this theorem states that its symmetric point 
with respect to the x-axis has coordinates (2, -3, -4).
-/
theorem symmetric_point_wrt_x_axis : 
  let original_point : ℝ × ℝ × ℝ := (2, 3, 4)
  let symmetric_point : ℝ × ℝ × ℝ := (2, -3, -4)
  (∀ (p : ℝ × ℝ × ℝ), 
    p.1 = original_point.1 ∧ 
    (p.2).1 = -(original_point.2).1 ∧ 
    (p.2).2 = -(original_point.2).2 → 
    p = symmetric_point) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_wrt_x_axis_l627_62735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_equation_l627_62772

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  -- The major and minor axes are aligned with the coordinate axes
  axesAligned : Bool
  -- One endpoint of the minor axis and the two foci form an equilateral triangle
  equilateralTriangle : Bool
  -- The shortest distance from a focus to a point on the ellipse
  shortestDistance : ℝ

/-- The equation of an ellipse -/
def EllipseEquation := ℝ → ℝ → Prop

/-- Theorem: The equation of the special ellipse -/
theorem special_ellipse_equation (e : SpecialEllipse) 
  (h1 : e.axesAligned = true) 
  (h2 : e.equilateralTriangle = true)
  (h3 : e.shortestDistance = Real.sqrt 3) :
  ∃ (eq : EllipseEquation), 
    (eq = λ x y ↦ x^2 / 12 + y^2 / 9 = 1) ∨ 
    (eq = λ x y ↦ y^2 / 12 + x^2 / 9 = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_equation_l627_62772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_common_points_implies_a_range_l627_62751

-- Define the line l: kx-y-k+2=0
def line_l (k x y : ℝ) : Prop := k * x - y - k + 2 = 0

-- Define the circle C: x^2+2ax+y^2-a+2=0
def circle_C (a x y : ℝ) : Prop := x^2 + 2*a*x + y^2 - a + 2 = 0

-- Statement of the theorem
theorem no_common_points_implies_a_range :
  ∀ a : ℝ, (∃ k : ℝ, ∀ x y : ℝ, ¬(line_l k x y ∧ circle_C a x y)) →
  (-7 < a ∧ a < -2) ∨ (a > 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_common_points_implies_a_range_l627_62751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_transformed_plane_l627_62723

/-- The similarity transformation of a plane with coefficient k -/
def transform_plane (a b c d k : ℝ) : ℝ → ℝ → ℝ → Prop :=
  λ x y z ↦ a * x + b * y + c * z + k * d = 0

/-- The point A -/
def A : ℝ × ℝ × ℝ := (2, -5, 4)

/-- The coefficient of similarity transformation -/
noncomputable def k : ℝ := 4 / 3

/-- The original plane a -/
def plane_a : ℝ → ℝ → ℝ → Prop :=
  λ x y z ↦ 5 * x + 2 * y - z + 3 = 0

theorem point_on_transformed_plane :
  transform_plane 5 2 (-1) 3 k A.1 A.2.1 A.2.2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_transformed_plane_l627_62723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_set_l627_62773

theorem existence_of_special_set (n : ℕ) (hn : n > 0) : 
  ∃ (S : Finset ℕ), 
    Finset.card S = n ∧ 
    (∀ a b : ℕ, a ∈ S → b ∈ S → a ≠ b → 
      (a - b ∣ a) ∧ 
      (a - b ∣ b) ∧ 
      (∀ c : ℕ, c ∈ S → c ≠ a → c ≠ b → ¬(a - b ∣ c))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_set_l627_62773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_prediction_for_10_year_old_l627_62730

/-- Represents a linear regression model for children's height based on age -/
structure HeightModel where
  slope : ℝ
  intercept : ℝ

/-- Predicts the height of a child given their age and a height model -/
def predict_height (model : HeightModel) (age : ℝ) : ℝ :=
  model.slope * age + model.intercept

/-- Checks if a given value is approximately equal to another value within a certain tolerance -/
def is_approximately (x y : ℝ) (tolerance : ℝ) : Prop :=
  |x - y| ≤ tolerance

theorem height_prediction_for_10_year_old (model : HeightModel)
    (h_slope : model.slope = 7.2)
    (h_intercept : model.intercept = 74)
    (h_age_range : ∀ x, 3 ≤ x ∧ x ≤ 9 → predict_height model x = predict_height model x)
    : is_approximately (predict_height model 10) 146 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_prediction_for_10_year_old_l627_62730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parentheses_correction_l627_62792

theorem parentheses_correction : 
  (1 * 2 * 3 + 4) * 5 = 50 := by
  -- Evaluate the expression inside the parentheses
  have h1 : 1 * 2 * 3 + 4 = 10 := by
    ring
  
  -- Multiply the result by 5
  calc
    (1 * 2 * 3 + 4) * 5 = 10 * 5 := by rw [h1]
    _ = 50 := by ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parentheses_correction_l627_62792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_closed_form_l627_62761

noncomputable def a : ℕ → ℝ
  | 0 => 1  -- Adding the base case for 0
  | 1 => 1
  | n + 1 => (1/16) * (1 + 4 * a n + Real.sqrt (1 + 24 * a n))

theorem a_closed_form (n : ℕ) (h : n ≥ 1) : 
  a n = (1 + 3 * 2^(n-1) + 2^(2*n-1)) / (3 * 2^(2*n-1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_closed_form_l627_62761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l627_62716

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = (2 * k + 1) * 8213) :
  Int.gcd (8 * b^2 + 63 * b + 144) (2 * b + 15) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l627_62716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_sum_l627_62780

theorem sine_cosine_sum (θ : ℝ) (b : ℝ) (h1 : 0 < θ ∧ θ < π/2) (h2 : Real.cos (2*θ) = b) :
  Real.sin θ + Real.cos θ = Real.sqrt (2 - b) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_sum_l627_62780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_area_l627_62766

/-- The volume of a right circular cone -/
noncomputable def cone_volume (r : ℝ) (h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- The area of a circle -/
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

theorem cone_base_area (V : ℝ) (h : ℝ) (hV : V = 24 * Real.pi) (hh : h = 6) :
  ∃ (r : ℝ), cone_volume r h = V ∧ circle_area r = 12 * Real.pi := by
  sorry

#check cone_base_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_area_l627_62766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l627_62700

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi

-- Define the area of a triangle using two sides and the included angle
noncomputable def triangle_area (a b : ℝ) (C : ℝ) : ℝ :=
  1/2 * a * b * Real.sin C

-- Theorem statement
theorem area_of_triangle_ABC :
  ∀ (A B C : ℝ),
  triangle_ABC A B C →
  C = Real.pi/3 →
  triangle_area 2 3 C = 3 * Real.sqrt 3 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l627_62700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_mile_time_speed_inverse_relation_speed_constant_per_mile_l627_62715

/-- Represents the speed of a particle at a given mile -/
noncomputable def speed (n : ℕ) : ℝ := 
  if n = 1 then 1  -- Speed for the first mile is not specified, so we set it to 1
  else 5 / (3 * (2 * n - 1))

/-- The time taken to traverse the nth mile -/
noncomputable def time (n : ℕ) : ℝ := 1 / speed n

theorem nth_mile_time (n : ℕ) (h : n > 0) : 
  time n = (3 * (2 * n - 1)) / 5 := by
  sorry

/-- The third mile is traversed in 3 hours -/
axiom third_mile_time : time 3 = 3

theorem speed_inverse_relation (n : ℕ) (h : n > 1) : 
  ∃ k : ℝ, speed n = k / (2 * (n - 1) + 1) := by
  sorry

theorem speed_constant_per_mile (n : ℕ) (h : n > 0) : 
  ∀ x : ℝ, 0 ≤ x ∧ x < 1 → 
    (speed n) * x = (speed n) * (1 - x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_mile_time_speed_inverse_relation_speed_constant_per_mile_l627_62715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_first_option_like_terms_l627_62710

/-- A monomial is represented as a pair of integers (coefficient, exponent) for each variable --/
def Monomial := List (ℤ × ℕ)

/-- Check if two monomials have the same variables with the same exponents --/
def like_terms (m1 m2 : Monomial) : Prop :=
  m1.map (λ (_, e) => e) = m2.map (λ (_, e) => e)

/-- The given monomial a^2b --/
def given_monomial : Monomial := [(1, 2), (1, 1)]

/-- The list of option monomials --/
def option_monomials : List Monomial := [
  [(-2, 2), (1, 1)],  -- -2a^2b
  [(1, 2), (2, 1)],   -- a^2b^2
  [(1, 1), (2, 1)],   -- ab^2
  [(3, 1), (1, 1)]    -- 3ab
]

theorem only_first_option_like_terms : 
  ∃! m, m ∈ option_monomials ∧ like_terms m given_monomial :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_first_option_like_terms_l627_62710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_one_condition_l627_62770

theorem tan_one_condition (θ : ℝ) : 
  (∃ k : ℤ, θ = 2 * k * Real.pi + Real.pi / 4) → Real.tan θ = 1 ∧ 
  ¬(Real.tan θ = 1 → ∃ k : ℤ, θ = 2 * k * Real.pi + Real.pi / 4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_one_condition_l627_62770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l627_62713

/-- The time taken for a train to pass a man moving in the same direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 300 →
  train_speed = 68 * (1000 / 3600) →
  man_speed = 8 * (1000 / 3600) →
  let relative_speed := train_speed - man_speed
  abs (train_length / relative_speed - 18) < 0.1 := by
  sorry

-- Remove the #eval statement as it's causing issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l627_62713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_13_l627_62737

theorem remainder_sum_mod_13 (a b c d e : ℕ) 
  (ha : a % 13 = 3)
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9)
  (he : e % 13 = 11) :
  (a + b + c + d + e) % 13 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_13_l627_62737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friendship_theorem_l627_62733

/-- A graph with 11 vertices where each vertex has degree at least 6 -/
structure FriendshipGraph where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)
  vertex_count : vertices.card = 11
  min_degree : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card ≥ 6

/-- A triangle in a graph is a set of three vertices that are all connected to each other -/
def HasTriangle (G : FriendshipGraph) : Prop :=
  ∃ a b c, a ∈ G.vertices ∧ b ∈ G.vertices ∧ c ∈ G.vertices ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (a, b) ∈ G.edges ∧ (b, c) ∈ G.edges ∧ (a, c) ∈ G.edges

/-- The main theorem: any FriendshipGraph contains a triangle -/
theorem friendship_theorem (G : FriendshipGraph) : HasTriangle G := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_friendship_theorem_l627_62733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_exponential_translation_l627_62742

noncomputable def e_to_x (x : ℝ) := Real.exp x

def is_symmetric_about_y_axis (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g (-x)

def translate_right (f : ℝ → ℝ) (units : ℝ) : ℝ → ℝ :=
  fun x ↦ f (x - units)

theorem symmetric_exponential_translation (f : ℝ → ℝ) :
  (translate_right (fun x ↦ e_to_x (-x)) 1 = f) →
  (is_symmetric_about_y_axis f e_to_x) →
  (f = fun x ↦ e_to_x (-x + 1)) :=
by
  sorry

#check symmetric_exponential_translation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_exponential_translation_l627_62742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_a_values_l627_62704

-- Define the line l
noncomputable def line_l (a t : ℝ) : ℝ × ℝ := (a + 3/5 * t, 1 + 4/5 * t)

-- Define the curve C
def curve_C (x y : ℝ) : Prop := y^2 = 8*x

-- Define point P
noncomputable def point_P (a : ℝ) : ℝ × ℝ := (a, 1)

-- Define the intersection condition
def intersects (a : ℝ) : Prop := ∃ t₁ t₂ : ℝ, 
  curve_C (line_l a t₁).1 (line_l a t₁).2 ∧
  curve_C (line_l a t₂).1 (line_l a t₂).2 ∧
  t₁ ≠ t₂

-- Define the distance ratio condition
def distance_ratio (a : ℝ) : Prop := ∃ t₁ t₂ : ℝ,
  intersects a ∧
  (line_l a t₁).1^2 + (line_l a t₁).2^2 = 9 * ((line_l a t₂).1^2 + (line_l a t₂).2^2)

-- Main theorem
theorem intersection_points_a_values : 
  ∀ a : ℝ, distance_ratio a ↔ (a = 13/8 ∨ a = -1/4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_a_values_l627_62704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_winning_strategy_l627_62717

/-- Represents the number of spaces a player can move in one turn -/
def PlayerMoves := Fin 6

/-- Represents the game board configuration -/
def BoardConfig := ℕ → Bool

/-- Calculates the win probability for the second player given the game parameters -/
noncomputable def secondPlayerWinProbability (s₁ s₂ : PlayerMoves) (board : BoardConfig) : ℝ :=
  sorry

/-- Theorem stating that the second player can always choose s₂ and a board configuration 
    to win with probability > 1/2, given any choice of s₁ by the first player -/
theorem second_player_winning_strategy :
  ∀ (s₁ : PlayerMoves), 
  ∃ (s₂ : PlayerMoves) (board : BoardConfig), 
  secondPlayerWinProbability s₁ s₂ board > (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_winning_strategy_l627_62717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_range_l627_62726

-- Define the functions f and g
def f (a x : ℝ) : ℝ := -x^3 + 1 + a
noncomputable def g (x : ℝ) : ℝ := 3 * Real.log x

-- State the theorem
theorem symmetric_points_range (e : ℝ) (h_e : e = Real.exp 1) :
  ∃ (a : ℝ), ∀ x : ℝ, 1/e ≤ x ∧ x ≤ e →
    (∃ y : ℝ, 1/e ≤ y ∧ y ≤ e ∧ f a x = -g y) →
      0 ≤ a ∧ a ≤ e^3 - 4 :=
by
  sorry

-- Additional helper lemmas if needed
lemma helper_lemma (x : ℝ) (hx : x > 0) : Real.log x < x := by
  sorry

-- You can add more helper lemmas here if required for the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_range_l627_62726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_result_l627_62729

noncomputable def dilation (center : ℂ) (scale : ℝ) (point : ℂ) : ℂ :=
  center + scale • (point - center)

theorem dilation_result : 
  dilation (1 + 2*Complex.I) 4 (0 - 2*Complex.I) = -3 - 14*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_result_l627_62729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_coins_is_one_l627_62757

/-- A game where a fair coin is flipped until a head appears, and the number of coins won
    is equal to the number of tails flipped before the first head. -/
def coin_flip_game : Type := Unit

/-- The probability of getting a head in a single flip. -/
noncomputable def prob_head : ℝ := 1 / 2

/-- The probability of getting a tail in a single flip. -/
noncomputable def prob_tail : ℝ := 1 - prob_head

/-- The expected number of gold coins won in the game. -/
noncomputable def expected_coins (game : coin_flip_game) : ℝ :=
  sorry

/-- Theorem stating that the expected number of gold coins won is 1. -/
theorem expected_coins_is_one (game : coin_flip_game) :
  expected_coins game = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_coins_is_one_l627_62757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_exists_l627_62778

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in 3D space -/
structure Plane where
  normal : Point3D
  d : ℝ

/-- A line in 3D space -/
structure Line where
  point : Point3D
  direction : Point3D

/-- The projection axis -/
noncomputable def projectionAxis : Line :=
  { point := ⟨0, 0, 0⟩, direction := ⟨1, 0, 0⟩ }

/-- The second projection plane -/
noncomputable def secondProjectionPlane : Plane :=
  { normal := ⟨0, 0, 1⟩, d := 0 }

/-- Determines if a plane passes through the projection axis -/
def planePassesThroughProjectionAxis (p : Plane) : Prop := sorry

/-- Determines if a plane is parallel to another plane -/
def planeParallelToPlane (p1 p2 : Plane) : Prop := sorry

/-- Finds the line of intersection between two planes -/
noncomputable def intersectionLine (p1 p2 : Plane) : Line := sorry

/-- Determines if a point is on a plane -/
def pointOnPlane (point : Point3D) (plane : Plane) : Prop := sorry

theorem intersection_line_exists 
  (p1 p2 : Plane) 
  (givenPoint : Point3D) 
  (h1 : planePassesThroughProjectionAxis p1)
  (h2 : p1.normal ≠ p2.normal)
  (h3 : pointOnPlane givenPoint p1)
  (h4 : planeParallelToPlane p2 secondProjectionPlane ∨ True) :
  ∃ (l : Line), l = intersectionLine p1 p2 := by
  sorry

#check intersection_line_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_exists_l627_62778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_condition_l627_62711

theorem square_condition (n : ℕ) : ∃ (a : ℕ), n * 2^n + 1 = a^2 ↔ n = 2 ∨ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_condition_l627_62711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_l627_62765

theorem product_of_roots : 
  (64 : ℝ)^(1/2 : ℝ) * (125 : ℝ)^(1/3 : ℝ) * (16 : ℝ)^(1/4 : ℝ) = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_l627_62765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l627_62703

def a : ℕ → ℚ
  | 0 => 1  -- Adding case for 0
  | 1 => 1
  | 2 => 3/7
  | n+3 => (a (n+1) * a (n+2)) / (2 * a (n+1) - a (n+2))

theorem a_formula (n : ℕ) : n ≥ 1 → a n = 3 / (4*n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l627_62703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_numParts_bounds_l627_62781

/-- A line drawn in a rectangle --/
structure Line where
  -- Add necessary properties for a line
  id : Nat  -- Unique identifier for each line

/-- A rectangle with lines drawn in it --/
structure RectangleWithLines where
  lines : List Line
  noOverlap : ∀ l1 l2, l1 ∈ lines → l2 ∈ lines → l1 ≠ l2 → l1.id ≠ l2.id

/-- The number of parts a rectangle is divided into by lines --/
def numParts (r : RectangleWithLines) : ℕ :=
  sorry

/-- Theorem stating the bounds on the number of parts --/
theorem numParts_bounds (r : RectangleWithLines) (h : r.lines.length = 3) :
  4 ≤ numParts r ∧ numParts r ≤ 7 := by
  sorry

#check numParts_bounds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_numParts_bounds_l627_62781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_addition_formula_l627_62724

theorem sine_addition_formula (x y : Real) :
  Real.sin x * Real.cos y + Real.cos x * Real.sin y = Real.sin (x + y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_addition_formula_l627_62724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_2023_l627_62759

-- Define the point type
structure Point where
  x : ℝ
  y : ℝ

-- Define the reflection operations
def reflect_x (p : Point) : Point := ⟨p.x, -p.y⟩
def reflect_y (p : Point) : Point := ⟨-p.x, p.y⟩

-- Define the composite reflection operation
def reflect_cycle (p : Point) : Point := reflect_y (reflect_x (reflect_y (reflect_x p)))

-- Define a function to apply reflect_cycle n times
def apply_n_times (n : ℕ) (p : Point) : Point :=
  match n with
  | 0 => p
  | n + 1 => reflect_cycle (apply_n_times n p)

-- Define the theorem
theorem reflection_2023 (m n : ℝ) (h : m < 0 ∧ n > 0) :
  (reflect_y (reflect_x (reflect_y (apply_n_times 505 ⟨m, n⟩)))) = ⟨-m, -n⟩ := by
  sorry

#check reflection_2023

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_2023_l627_62759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_transformation_l627_62749

-- Define the original ellipse
def original_ellipse (x y : ℝ) : Prop := 9 * x^2 + 4 * y^2 = 36

-- Define the focal distance of the original ellipse
noncomputable def focal_distance : ℝ := Real.sqrt 5

-- Define the new ellipse with the same foci and minor axis of length 2
def new_ellipse (x y : ℝ) : Prop :=
  x^2 + y^2 / 6 = 1

theorem ellipse_transformation :
  ∀ x y : ℝ,
  original_ellipse x y →
  (∃ x' y' : ℝ, new_ellipse x' y' ∧
    (x'^2 + y'^2 = x^2 + y^2) ∧
    (new_ellipse 0 1 ∨ new_ellipse 1 0)) :=
by
  sorry

#check ellipse_transformation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_transformation_l627_62749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_four_percent_l627_62798

/-- Given a principal amount, time period, and interest amount, 
    calculate the simple interest rate per annum. -/
noncomputable def calculate_interest_rate (principal : ℝ) (time : ℝ) (interest : ℝ) : ℝ :=
  (interest / (principal * time)) * 100

/-- Theorem stating that for the given conditions, the interest rate is 4% -/
theorem interest_rate_is_four_percent 
  (principal : ℝ) 
  (time : ℝ) 
  (interest : ℝ) 
  (h1 : principal = 1500)
  (h2 : time = 4)
  (h3 : interest = 240) :
  calculate_interest_rate principal time interest = 4 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculate_interest_rate 1500 4 240

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_four_percent_l627_62798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_equilateral_polyhedron_l627_62791

/-- A polyhedron with the equilateral triangle property -/
structure EquilateralPolyhedron where
  vertices : Finset (Fin 3 → ℝ)
  vertex_count : vertices.card = 5
  equilateral_property : ∀ (v1 v2 : Fin 3 → ℝ), v1 ∈ vertices → v2 ∈ vertices → v1 ≠ v2 →
    ∃ (v3 : Fin 3 → ℝ), v3 ∈ vertices ∧ v3 ≠ v1 ∧ v3 ≠ v2 ∧
    ‖v1 - v2‖ = ‖v2 - v3‖ ∧ ‖v2 - v3‖ = ‖v3 - v1‖

/-- There exists a polyhedron with 5 vertices satisfying the equilateral triangle property -/
theorem exists_equilateral_polyhedron : ∃ (p : EquilateralPolyhedron), True := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_equilateral_polyhedron_l627_62791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_det_and_count_l627_62741

/-- The set of 3x3 matrices where each row and column contain a, b, c -/
def M (a b c : ℝ) : Set (Matrix (Fin 3) (Fin 3) ℝ) :=
  { A | ∀ i j : Fin 3, ({A i 0, A i 1, A i 2} : Set ℝ) = {a, b, c} ∧ ({A 0 j, A 1 j, A 2 j} : Set ℝ) = {a, b, c} }

theorem max_det_and_count (a b c : ℝ) (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a) (hsum : a + b + c > 0) :
  ∃ (max_det : ℝ) (count : ℕ),
    max_det = a^3 + b^3 + c^3 - 3*a*b*c ∧
    count = 6 ∧
    (∀ A ∈ M a b c, Matrix.det A ≤ max_det) ∧
    (∃ (matrices : Finset (Matrix (Fin 3) (Fin 3) ℝ)),
      matrices.card = count ∧
      ∀ A ∈ matrices, A ∈ M a b c ∧ Matrix.det A = max_det) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_det_and_count_l627_62741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l627_62732

def geometric_sequence (a₁ : ℕ) (r : ℕ) : ℕ → ℕ
  | 0 => a₁
  | n + 1 => r * geometric_sequence a₁ r n

def sum_geometric_sequence (a₁ : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  (a₁ * (1 - r^n)) / (1 - r)

theorem sequence_properties :
  geometric_sequence 1 2 4 = 16 ∧ sum_geometric_sequence 1 2 8 = 255 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l627_62732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_is_67_point_5_l627_62755

/-- Represents a spherical triangle with angles in a 3:4:5 ratio --/
structure SphericalTriangle where
  angle_sum : ℝ
  angle_sum_gt_180 : angle_sum > 180
  angle_sum_lt_540 : angle_sum < 540

/-- The smallest angle in the spherical triangle --/
noncomputable def smallest_angle (t : SphericalTriangle) : ℝ :=
  t.angle_sum / 12

theorem smallest_angle_is_67_point_5 (t : SphericalTriangle) :
  smallest_angle t = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_is_67_point_5_l627_62755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equiv_g_l627_62708

/-- The domain of the functions f and g -/
noncomputable def Domain : Set ℝ := {x : ℝ | x ≠ 0}

/-- Function f(x) = x + 1/x -/
noncomputable def f (x : ℝ) : ℝ := x + 1/x

/-- Function g(x) = (x^2 + 1)/x -/
noncomputable def g (x : ℝ) : ℝ := (x^2 + 1)/x

/-- Theorem stating that f and g are equivalent on their domain -/
theorem f_equiv_g : ∀ x ∈ Domain, f x = g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equiv_g_l627_62708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_characterization_l627_62786

-- Define a triangle as a structure with three points
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a function to check if a point is on the circumcircle of a triangle
def isOnCircumcircle (t : Triangle) (M : ℝ × ℝ) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    dist center t.A = radius ∧
    dist center t.B = radius ∧
    dist center t.C = radius ∧
    dist center M = radius ∧
    M ≠ t.A ∧ M ≠ t.B ∧ M ≠ t.C

-- Define a function to check if two line segments are perpendicular
def isPerpendicular (seg1 : (ℝ × ℝ) × (ℝ × ℝ)) (seg2 : (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  let (a1, a2) := seg1
  let (b1, b2) := seg2
  let v1 := (a2.1 - a1.1, a2.2 - a1.2)
  let v2 := (b2.1 - b1.1, b2.2 - b1.2)
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Define a function to check if perpendiculars intersect at a single point
def perpendicularsIntersect (t : Triangle) (M : ℝ × ℝ) : Prop :=
  ∃ (M' : ℝ × ℝ),
    isPerpendicular (t.A, M) (t.A, M') ∧
    isPerpendicular (t.B, M) (t.B, M') ∧
    isPerpendicular (t.C, M) (t.C, M')

-- The main theorem
theorem circumcircle_characterization (t : Triangle) (M : ℝ × ℝ) :
  isOnCircumcircle t M ↔ perpendicularsIntersect t M :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_characterization_l627_62786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_divisor_l627_62747

theorem fourth_divisor (X : ℕ) (h1 : X = 200) 
  (h2 : ∀ (n : ℕ), n ∈ ({15, 30, 45} : Set ℕ) → (X - 20) % n = 0) : 
  ∃ (d : ℕ), d ∣ (X - 20) ∧ d ∉ ({15, 30, 45} : Set ℕ) ∧ d = 4 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_divisor_l627_62747

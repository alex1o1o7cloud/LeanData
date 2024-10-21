import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_major_premise_incorrect_l810_81043

open Function Real

-- Define a differentiable function
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define a point x₀
variable (x₀ : ℝ)

-- Define IsExtremeValue
def IsExtremeValue (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x, |x - x₀| < ε → f x₀ ≤ f x ∨ f x ≤ f x₀

-- State the theorem to be proven
theorem major_premise_incorrect :
  ¬(∀ f : ℝ → ℝ, ∀ x₀ : ℝ, Differentiable ℝ f → (deriv f x₀ = 0 → IsExtremeValue f x₀)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_major_premise_incorrect_l810_81043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l810_81052

/-- The area of a right triangle with base 12 cm and height 9 cm is 54 square centimeters. -/
theorem right_triangle_area : 
  ∀ (A B C : ℝ × ℝ) (base height area : ℝ),
    (B.1 = 0 ∧ B.2 = 0) →  -- B is at the origin
    (C.1 = base ∧ C.2 = 0) →  -- C is base units along x-axis from B
    (A.1 = base ∧ A.2 = height) →  -- A is height units above C
    base = 12 →
    height = 9 →
    area = (1 / 2) * base * height →
    area = 54 :=
by
  intros A B C base height area hB hC hA hbase hheight harea
  simp [hbase, hheight] at harea
  norm_num at harea
  exact harea


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l810_81052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_seq_min_S_value_l810_81008

-- Define the sequence a_n and its sum S_n
def a : ℕ → ℝ := sorry
def S : ℕ → ℝ := sorry

-- Given condition
axiom condition (n : ℕ) : 2 * S n / n + n = 2 * a n + 1

-- a_4, a_7, and a_9 form a geometric sequence
axiom geometric_seq : (a 7) ^ 2 = (a 4) * (a 9)

-- Theorem 1: a_n is an arithmetic sequence with common difference 1
theorem arithmetic_seq : ∀ n : ℕ, a (n + 1) = a n + 1 := by
  sorry

-- Theorem 2: The minimum value of S_n is -78
theorem min_S_value : ∃ n : ℕ, S n = -78 ∧ ∀ m : ℕ, S m ≥ -78 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_seq_min_S_value_l810_81008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gyroscope_initial_speed_l810_81057

/-- Represents the speed of a gyroscope that doubles every 15 seconds -/
noncomputable def gyroscope_speed (initial_speed : ℝ) (time : ℝ) : ℝ :=
  initial_speed * (2 ^ (time / 15))

/-- Theorem: If a gyroscope's speed doubles every 15 seconds and reaches 400 m/s after 90 seconds,
    then its initial speed was 6.25 m/s -/
theorem gyroscope_initial_speed :
  ∃ (initial_speed : ℝ),
    gyroscope_speed initial_speed 90 = 400 ∧
    initial_speed = 6.25 := by
  use 6.25
  constructor
  · simp [gyroscope_speed]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gyroscope_initial_speed_l810_81057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trirectangular_tetrahedron_max_volume_l810_81086

/-- A trirectangular tetrahedron is a tetrahedron with three right angles at a single vertex. -/
structure TrirectangularTetrahedron where
  /-- The sum of the lengths of the six edges -/
  S : ℝ
  /-- Assumption that S is positive -/
  S_pos : S > 0

/-- The maximum volume of a trirectangular tetrahedron -/
noncomputable def max_volume (t : TrirectangularTetrahedron) : ℝ :=
  (t.S^3 * (Real.sqrt 2 - 1)^3) / 162

/-- Theorem stating that the maximum volume of a trirectangular tetrahedron
    with edge sum S is (S³(√2-1)³)/162 -/
theorem trirectangular_tetrahedron_max_volume (t : TrirectangularTetrahedron) :
  ∀ V : ℝ, V ≤ max_volume t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trirectangular_tetrahedron_max_volume_l810_81086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_angle_l810_81046

/-- The equation of a line l is x + √3y - 1 = 0. -/
def line_equation (x y : ℝ) : Prop := x + Real.sqrt 3 * y - 1 = 0

/-- The slope angle of a line with equation ax + by + c = 0 is the angle between the line and the positive x-axis. -/
noncomputable def slope_angle (a b c : ℝ) : ℝ := Real.arctan (-a / b)

/-- The slope angle of the line x + √3y - 1 = 0 is 150°. -/
theorem line_slope_angle :
  slope_angle 1 (Real.sqrt 3) (-1) = 150 * Real.pi / 180 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_angle_l810_81046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_experiment_result_l810_81097

/-- Represents the germination experiment with two plots of seeds. -/
structure SeedExperiment where
  plot1_seeds : ℕ
  plot2_seeds : ℕ
  plot1_germination_rate : ℚ
  overall_germination_rate : ℚ

/-- Calculates the germination rate of the second plot. -/
def plot2_germination_rate (e : SeedExperiment) : ℚ :=
  ((e.overall_germination_rate * (e.plot1_seeds + e.plot2_seeds : ℚ)
    - e.plot1_germination_rate * e.plot1_seeds) 
   / e.plot2_seeds)

/-- Theorem stating that for the given experiment parameters, 
    the germination rate of the second plot is 40%. -/
theorem experiment_result : 
  let e : SeedExperiment := {
    plot1_seeds := 300,
    plot2_seeds := 200,
    plot1_germination_rate := 1/4,
    overall_germination_rate := 31/100
  }
  plot2_germination_rate e = 2/5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_experiment_result_l810_81097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_function_derivative_zero_l810_81011

theorem constant_function_derivative_zero
  {a b : ℝ} (f : ℝ → ℝ) (hf : ContinuousOn f (Set.Icc a b)) :
  (∃ c, ∀ x ∈ Set.Icc a b, f x = c) →
  ∀ x ∈ Set.Icc a b, HasDerivAt f 0 x :=
by
  intro h
  intro x hx
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_function_derivative_zero_l810_81011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_x_intercept_l810_81032

/-- Definition of an ellipse with given foci and one x-intercept -/
structure Ellipse where
  foci : (ℝ × ℝ) × (ℝ × ℝ)
  x_intercept : ℝ × ℝ
  sum_distances : ℝ

/-- The ellipse in the problem -/
def problem_ellipse : Ellipse :=
  { foci := ((0, 3), (4, 0)),
    x_intercept := (0, 0),
    sum_distances := 7 }

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: The other x-intercept of the ellipse is (56/11, 0) -/
theorem other_x_intercept (e : Ellipse) (h : e = problem_ellipse) :
  ∃ x : ℝ, x = 56 / 11 ∧
  distance (x, 0) e.foci.1 + distance (x, 0) e.foci.2 = e.sum_distances :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_x_intercept_l810_81032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_value_l810_81063

theorem cos_2x_value (x θ : ℝ) 
  (h1 : Real.sin (2 * x) = (Real.sin θ + Real.cos θ) / 2)
  (h2 : Real.cos x ^ 2 - Real.sin θ * Real.cos θ = 0) :
  Real.cos (2 * x) = (-1 - Real.sqrt 33) / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_value_l810_81063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beth_finishes_first_l810_81038

/-- Represents the area of a lawn -/
structure LawnArea where
  size : ℝ
  size_pos : size > 0

/-- Represents the mowing rate of a lawn mower -/
structure MowingRate where
  rate : ℝ
  rate_pos : rate > 0

/-- Represents a person with their lawn area and mowing rate -/
structure Person where
  name : String
  lawn : LawnArea
  mower : MowingRate

/-- Calculates the time taken to mow a lawn -/
noncomputable def mowingTime (p : Person) : ℝ := p.lawn.size / p.mower.rate

theorem beth_finishes_first (beth andy carlos : Person)
  (h1 : andy.lawn.size = 1.5 * beth.lawn.size)
  (h2 : andy.lawn.size = 2 * carlos.lawn.size)
  (h3 : carlos.mower.rate = 0.5 * andy.mower.rate)
  (h4 : beth.mower.rate = andy.mower.rate) :
  mowingTime beth < mowingTime andy ∧ mowingTime beth < mowingTime carlos := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_beth_finishes_first_l810_81038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bank_account_balance_bank_account_balance_proof_l810_81093

theorem bank_account_balance (initial_deposit : ℝ) (first_year_interest : ℝ) 
  (second_year_rate : ℝ) (total_increase_rate : ℝ) : ℝ :=
  let first_year_balance := initial_deposit + first_year_interest
  let second_year_balance := first_year_balance * (1 + second_year_rate)
  let total_increase := initial_deposit * total_increase_rate
  let final_balance := initial_deposit + total_increase
  
  if initial_deposit = 500 ∧ 
     first_year_interest = 100 ∧ 
     second_year_rate = 0.1 ∧ 
     total_increase_rate = 0.32 ∧
     second_year_balance = final_balance
  then
    first_year_balance
  else
    0

theorem bank_account_balance_proof : 
  bank_account_balance 500 100 0.1 0.32 = 600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bank_account_balance_bank_account_balance_proof_l810_81093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_plus_xcosx_equals_pi_half_l810_81088

open Real MeasureTheory

theorem integral_sqrt_plus_xcosx_equals_pi_half :
  ∫ x in (-1 : ℝ)..1, (Real.sqrt (1 - x^2) + x * Real.cos x) = π / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_plus_xcosx_equals_pi_half_l810_81088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lamps_eventually_off_iff_power_of_two_minus_one_l810_81007

/-- Represents the state of a lamp (on or off) -/
inductive LampState
| On : LampState
| Off : LampState

/-- Represents a row of lamps -/
def LampRow (n : ℕ) := Fin n → LampState

/-- Function to update the state of lamps according to the rules -/
def updateLamps (n : ℕ) (row : LampRow n) : LampRow n :=
  sorry

/-- Predicate to check if all lamps are off -/
def allLampsOff (n : ℕ) (row : LampRow n) : Prop :=
  ∀ i, row i = LampState.Off

/-- Predicate to check if a given number of lamps will eventually turn off -/
def eventuallyAllOff (n : ℕ) : Prop :=
  ∀ initial : LampRow n, ∃ t : ℕ, allLampsOff n (Nat.iterate (updateLamps n) t initial)

/-- Main theorem: all lamps will eventually turn off if and only if n is of the form 2^k - 1 -/
theorem lamps_eventually_off_iff_power_of_two_minus_one (n : ℕ) :
  (∃ k : ℕ, n = 2^k - 1) ↔ eventuallyAllOff n :=
sorry

#check lamps_eventually_off_iff_power_of_two_minus_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lamps_eventually_off_iff_power_of_two_minus_one_l810_81007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_inside_circle_outside_square_l810_81016

/-- The area inside a circle of radius 1 but outside a square of side length 2,
    where the circle and square share the same center. -/
noncomputable def areaInsideCircleOutsideSquare : ℝ :=
  Real.pi - (4 * (Real.pi / 4 - 1 / 2))

/-- Theorem stating that the area inside the circle but outside the square is 2. -/
theorem area_inside_circle_outside_square :
  areaInsideCircleOutsideSquare = 2 := by
  sorry

-- This evaluation won't work in Lean 4 as it involves non-computable real numbers
-- #eval areaInsideCircleOutsideSquare

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_inside_circle_outside_square_l810_81016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_decrease_l810_81028

/-- Theorem: When the radius of a circle is decreased by 50%, the area of the circle decreases by 75%. -/
theorem circle_area_decrease (r : ℝ) (hr : r > 0) : 
  (π * r^2 - π * (r/2)^2) / (π * r^2) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_decrease_l810_81028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_and_expected_value_l810_81080

-- Define the data set
def days : List ℚ := [1, 2, 3, 4, 5, 6, 7]
def heights : List ℚ := [0, 4, 7, 9, 11, 12, 13]

-- Define the average function
noncomputable def average (l : List ℚ) : ℚ := (l.sum) / (l.length : ℚ)

-- Define the linear regression coefficients
noncomputable def b_hat (x : List ℚ) (y : List ℚ) : ℚ :=
  let x_avg := average x
  let y_avg := average y
  (List.sum (List.zipWith (· * ·) x y) - x.length * x_avg * y_avg) /
  (List.sum (List.map (· ^ 2) x) - x.length * x_avg ^ 2)

noncomputable def a_hat (x : List ℚ) (y : List ℚ) : ℚ :=
  average y - b_hat x y * average x

-- Define the random variable ξ
noncomputable def ξ (selected : List ℚ) : ℕ :=
  let y_avg := average heights
  (selected.filter (· > y_avg)).length

-- Theorem to prove
theorem linear_regression_and_expected_value :
  b_hat days heights = 59 / 28 ∧
  a_hat days heights = -3 / 7 ∧
  (1 / 35 * 0 + 12 / 35 * 1 + 18 / 35 * 2 + 4 / 35 * 3 : ℚ) = 12 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_and_expected_value_l810_81080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_under_sine_curve_l810_81071

-- Define the bounds of the integral
noncomputable def lower_bound : ℝ := 0
noncomputable def upper_bound : ℝ := Real.pi / 2

-- Define the function representing the curve
noncomputable def f (x : ℝ) : ℝ := Real.sin x

-- State the theorem
theorem area_under_sine_curve :
  (∫ x in lower_bound..upper_bound, f x) =
  (∫ x in lower_bound..upper_bound, Real.sin x) := by
  -- The proof is skipped using 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_under_sine_curve_l810_81071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_to_asymptote_distance_l810_81039

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 2 - y^2 / 3 = 1

-- Define the focus of the hyperbola
noncomputable def focus : ℝ × ℝ := (Real.sqrt 5, 0)

-- Define the asymptote of the hyperbola
def asymptote (x y : ℝ) : Prop := 2 * y - Real.sqrt 6 * x = 0

-- Theorem statement
theorem focus_to_asymptote_distance :
  ∃ (d : ℝ), d = Real.sqrt 3 ∧
  ∀ (x y : ℝ), hyperbola x y →
  d = abs (2 * focus.1 - Real.sqrt 6 * focus.2) /
      Real.sqrt (2^2 + (Real.sqrt 6)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_to_asymptote_distance_l810_81039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l810_81054

/-- The function we're maximizing -/
noncomputable def f (t : ℝ) : ℝ := (2^(t+1) - 4*t)*t / 16^t

/-- Theorem stating that 1/16 is the maximum value of f -/
theorem f_max_value :
  (∀ t : ℝ, f t ≤ 1/16) ∧ (∃ t : ℝ, f t = 1/16) := by
  sorry

#check f_max_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l810_81054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_log_is_exp_rotated_graph_is_exp_l810_81030

-- Define the original function
noncomputable def original_function (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the rotation transformation
def rotate_90_ccw (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, p.1)

-- State the theorem
theorem rotated_log_is_exp :
  ∀ x y : ℝ, x > 0 → y = original_function x →
  rotate_90_ccw (x, y) = (-y, x) ∧ x = 2^(-y) := by
  sorry

-- Additional theorem to connect the rotation to the new function
theorem rotated_graph_is_exp :
  ∀ x y : ℝ, x > 0 →
  (∃ t : ℝ, t > 0 ∧ rotate_90_ccw (t, original_function t) = (x, y)) →
  y = 2^x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_log_is_exp_rotated_graph_is_exp_l810_81030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_and_inequality_l810_81094

-- Define the function f
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- State the theorem
theorem extreme_values_and_inequality (a b c : ℝ) :
  (∀ x, f a b c x = x^3 + a*x^2 + b*x + c) →
  (∃ y, (deriv (f a b c)) y = 0 ∧ y = -1) →
  (∃ z, (deriv (f a b c)) z = 0 ∧ z = 2) →
  (∀ x ∈ Set.Icc (-2) 3, f a b c x + (3/2)*c < c^2) →
  (a = -3/2 ∧ b = -6) ∧
  (∀ x ∈ Set.Ioo (-1) 2, (deriv (f a b c)) x < 0) ∧
  (∀ x ∈ Set.Iic (-1), (deriv (f a b c)) x > 0) ∧
  (∀ x ∈ Set.Ioi 2, (deriv (f a b c)) x > 0) ∧
  (c ∈ Set.Iio (-1) ∪ Set.Ioi (7/2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_and_inequality_l810_81094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_divisible_by_36_and_largest_l810_81018

def f (n : ℕ+) : ℕ := (2 * n.val + 7) * 3^n.val + 9

theorem f_divisible_by_36_and_largest (n : ℕ+) :
  ∃ (k : ℕ), f n = 36 * k ∧
  ∀ (m : ℕ), m > 36 → ¬(∀ (j : ℕ+), ∃ (l : ℕ), f j = m * l) :=
by sorry

#eval f 1
#eval f 2
#eval f 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_divisible_by_36_and_largest_l810_81018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l810_81033

open Real

theorem trigonometric_identities (α : ℝ) 
  (h1 : cos α - sin α = (5 * sqrt 2) / 13)
  (h2 : 0 < α ∧ α < π / 4) :
  sin α * cos α = 119 / 338 ∧
  cos (2 * α) / cos (π / 4 + α) = 24 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l810_81033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_hexagon_angle_l810_81001

/-- The measure of an interior angle of a regular polygon with n sides -/
noncomputable def interior_angle (n : ℕ) : ℝ := (n - 2 : ℝ) * 180 / n

/-- The measure of an exterior angle between two regular polygons -/
noncomputable def exterior_angle (n m : ℕ) : ℝ := 360 - (interior_angle n + interior_angle m)

theorem pentagon_hexagon_angle :
  let pentagon_angle : ℝ := interior_angle 5
  let hexagon_angle : ℝ := interior_angle 6
  let ext_angle : ℝ := exterior_angle 5 6
  let triangle_angle : ℝ := (180 - ext_angle) / 2
  triangle_angle = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_hexagon_angle_l810_81001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_one_l810_81083

/-- A type representing positive rational numbers -/
def PositiveRational := { q : ℚ // 0 < q }

instance : Mul PositiveRational where
  mul a b := ⟨a.val * b.val, mul_pos a.property b.property⟩

instance : Pow PositiveRational ℕ where
  pow a n := ⟨a.val ^ n, pow_pos a.property n⟩

instance : OfNat PositiveRational (nat_lit 1) where
  ofNat := ⟨1, by norm_num⟩

/-- The functional equation property -/
def SatisfiesFunctionalEquation (f : PositiveRational → PositiveRational) :=
  ∀ x y : PositiveRational, f (x^2 * (f y)^2) = (f x)^2 * f y

/-- Theorem: The only function satisfying the functional equation is the constant function f(x) = 1 -/
theorem unique_solution_is_one (f : PositiveRational → PositiveRational) 
  (h : SatisfiesFunctionalEquation f) : 
  ∀ x : PositiveRational, f x = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_one_l810_81083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_of_legs_l810_81020

-- Define the right triangle XYZ
structure RightTriangle where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  is_right_angle : (Z.1 - X.1) * (Z.1 - Y.1) + (Z.2 - X.2) * (Z.2 - Y.2) = 0

-- Define the sequence of equilateral triangles
structure EquilateralTriangleSequence where
  X : ℕ → ℝ × ℝ
  Y : ℕ → ℝ × ℝ
  T : ℕ → ℝ × ℝ
  is_equilateral : ∀ i, 
    ((X i).1 - (Y i).1)^2 + ((X i).2 - (Y i).2)^2 = 
    ((X i).1 - (T i).1)^2 + ((X i).2 - (T i).2)^2 ∧
    ((X i).1 - (T i).1)^2 + ((X i).2 - (T i).2)^2 = 
    ((Y i).1 - (T i).1)^2 + ((Y i).2 - (T i).2)^2

-- Define the properties of the sequence
def SequenceProperties (triangle : RightTriangle) (seq : EquilateralTriangleSequence) : Prop :=
  (seq.X 0 = triangle.X) ∧
  (seq.Y 0 = triangle.Y) ∧
  (∀ i, ∃ t : ℝ, seq.X i = (triangle.X.1 + t * (triangle.Z.1 - triangle.X.1), triangle.X.2 + t * (triangle.Z.2 - triangle.X.2))) ∧
  (∀ i, ∃ t : ℝ, seq.Y i = (triangle.Y.1 + t * (triangle.Z.1 - triangle.Y.1), triangle.Y.2 + t * (triangle.Z.2 - triangle.Y.2))) ∧
  (∀ i, (seq.X i).1 * (triangle.Z.1 - triangle.Y.1) + (seq.X i).2 * (triangle.Z.2 - triangle.Y.2) = 0) ∧
  (∀ i, (seq.T i).2 ≠ triangle.Y.2) ∧
  (∀ i > 0, ∃ t : ℝ, seq.X i = ((seq.Y (i-1)).1 + t * ((seq.T (i-1)).1 - (seq.Y (i-1)).1), (seq.Y (i-1)).2 + t * ((seq.T (i-1)).2 - (seq.Y (i-1)).2)))

-- Define the area equality condition
noncomputable def AreaEquality (triangle : RightTriangle) (seq : EquilateralTriangleSequence) : Prop :=
  -- This is a placeholder for the actual area equality condition
  True

-- The main theorem
theorem equality_of_legs (triangle : RightTriangle) (seq : EquilateralTriangleSequence) 
  (h_properties : SequenceProperties triangle seq) (h_area : AreaEquality triangle seq) :
  (triangle.Y.1 - triangle.X.1)^2 + (triangle.Y.2 - triangle.X.2)^2 = 
  (triangle.Z.1 - triangle.Y.1)^2 + (triangle.Z.2 - triangle.Y.2)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_of_legs_l810_81020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_cone_volume_ratio_l810_81037

/-- A sphere with an inscribed right circular cone where the sphere's center divides the cone's height according to the golden ratio -/
structure SphereWithCone where
  r : ℝ  -- radius of the sphere
  x : ℝ  -- distance from the center of the sphere to the base of the cone
  h : r > 0
  golden_ratio : x * (r + x) = r^2

/-- The volume of a sphere -/
noncomputable def sphere_volume (s : SphereWithCone) : ℝ := (4 / 3) * Real.pi * s.r^3

/-- The volume of the inscribed cone -/
noncomputable def cone_volume (s : SphereWithCone) : ℝ := (1 / 3) * Real.pi * (s.r^2 - s.x^2) * (s.r + s.x)

/-- The ratio of the volume of the sphere to the volume of the cone is 4:1 -/
theorem sphere_cone_volume_ratio (s : SphereWithCone) :
  sphere_volume s / cone_volume s = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_cone_volume_ratio_l810_81037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l810_81096

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x - Real.sqrt 3 * (Real.cos x)^2

theorem f_properties :
  (∃ T : ℝ, T > 0 ∧ (∀ x, f (x + T) = f x) ∧ (∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y)) ∧
  (∃ M : ℝ, (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ M) ∧ M = 1 - Real.sqrt 3 / 2) ∧
  (∃ m : ℝ, (∀ x ∈ Set.Icc 0 (Real.pi / 2), m ≤ f x) ∧ m = -Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l810_81096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_sum_times_four_l810_81010

/-- The sum of coordinates of a 2D point. -/
def sum_coordinates (p : ℝ × ℝ) : ℝ :=
  p.1 + p.2

/-- The first endpoint of the segment. -/
def p1 : ℝ × ℝ := (8, -4)

/-- The second endpoint of the segment. -/
def p2 : ℝ × ℝ := (-2, 10)

theorem midpoint_sum_times_four :
  4 * sum_coordinates ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_sum_times_four_l810_81010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_common_integer_is_one_l810_81003

def a : ℕ → ℤ
  | 0 => 1  -- Adding case for 0
  | 1 => 1
  | 2 => 1
  | (n + 3) => a (n + 2) + 2 * a (n + 1)

def b : ℕ → ℤ
  | 0 => 1  -- Adding case for 0
  | 1 => 1
  | 2 => 7
  | (n + 3) => 2 * b (n + 2) + 3 * b (n + 1)

theorem only_common_integer_is_one (n m : ℕ) (h : n ≥ 1 ∧ m ≥ 1) :
  a n = b m → a n = 1 ∧ b m = 1 := by
  sorry

#eval a 5  -- Testing the function
#eval b 5  -- Testing the function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_common_integer_is_one_l810_81003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increase_max_triangle_area_l810_81069

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x - (Real.cos (x + Real.pi/4))^2

def monotonic_increase_intervals (f : ℝ → ℝ) : Set (Set ℝ) :=
  {S | ∀ x y, x ∈ S → y ∈ S → x < y → f x < f y}

theorem f_monotonic_increase :
  monotonic_increase_intervals f = {S | ∃ k : ℤ, S = Set.Icc (-Real.pi/4 + k*Real.pi) (Real.pi/4 + k*Real.pi)} := by
  sorry

theorem max_triangle_area (A B C : ℝ) (h_acute : A + B + C = Real.pi) 
  (h_f : f (A/2) = 0) (h_a : Real.sin A = 1) :
  (Real.sin B * Real.sin C) / (2 * Real.sin A) ≤ (2 + Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increase_max_triangle_area_l810_81069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_payoff_period_is_179_days_l810_81053

/-- Represents the cryptocurrency mining setup -/
structure MiningSetup where
  system_unit_cost : ℚ
  graphics_card_cost : ℚ
  system_unit_power : ℚ
  graphics_card_power : ℚ
  graphics_card_count : ℕ
  daily_eth_per_card : ℚ
  eth_to_rub_rate : ℚ
  electricity_cost_per_kwh : ℚ

/-- Calculates the number of days required for the investment to pay off -/
noncomputable def payoff_days (setup : MiningSetup) : ℚ :=
  let total_investment := setup.system_unit_cost + setup.graphics_card_cost * setup.graphics_card_count
  let daily_eth_revenue := setup.daily_eth_per_card * setup.graphics_card_count
  let daily_rub_revenue := daily_eth_revenue * setup.eth_to_rub_rate
  let total_power_consumption := setup.system_unit_power + setup.graphics_card_power * setup.graphics_card_count
  let daily_energy_cost := (total_power_consumption / 1000) * 24 * setup.electricity_cost_per_kwh
  let daily_profit := daily_rub_revenue - daily_energy_cost
  total_investment / daily_profit

/-- Theorem stating that the payoff period for the given setup is approximately 179 days -/
theorem payoff_period_is_179_days :
  let setup : MiningSetup := {
    system_unit_cost := 9499
    graphics_card_cost := 20990
    system_unit_power := 120
    graphics_card_power := 185
    graphics_card_count := 2
    daily_eth_per_card := 63/10000
    eth_to_rub_rate := 2779037/100
    electricity_cost_per_kwh := 538/100
  }
  ⌊payoff_days setup⌋ = 179 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_payoff_period_is_179_days_l810_81053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_values_l810_81015

theorem sum_of_possible_values (x y : ℝ) 
  (h : x * y - (2 * x / y^3) - (2 * y / x^3) = 6) : 
  ∃ (S : Finset ℝ), (∀ z ∈ S, ∃ x y : ℝ, 
    (x * y - (2 * x / y^3) - (2 * y / x^3) = 6) ∧ 
    ((x - 2) * (y - 2) = z)) ∧ 
  (S.sum id = 13) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_values_l810_81015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_point_l810_81091

def point_A : ℝ × ℝ := (1, 2)
def point_B : ℝ × ℝ := (1, 5)
def point_C : ℝ × ℝ := (3, 6)
def point_D : ℝ × ℝ := (7, -1)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

noncomputable def sum_of_distances (p : ℝ × ℝ) : ℝ :=
  distance p point_A + distance p point_B + distance p point_C + distance p point_D

theorem minimum_distance_point :
  ∀ p : ℝ × ℝ, sum_of_distances (2, 4) ≤ sum_of_distances p :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_point_l810_81091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_value_l810_81062

noncomputable section

open Real

noncomputable def f (x : ℝ) := sin x * cos x - sqrt 3 * (cos x)^2

noncomputable def g (x : ℝ) := sin (2 * x + π / 3) - sqrt 3 / 2

theorem min_shift_value (k : ℝ) :
  (k > 0 ∧ ∀ x, f x = g (x - k)) →
  k ≥ π / 3 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_value_l810_81062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_z_equals_one_l810_81026

noncomputable def z : ℂ := (1 + 2*Complex.I) / (2 - Complex.I)

theorem abs_z_equals_one : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_z_equals_one_l810_81026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l810_81068

theorem problem_statement : 
  (∀ (a b c : ℝ → ℝ → ℝ → ℝ), (∀ x y z, a x y z * c x y z = b x y z * c x y z) → a = b) ∨ 
  (∀ (a b : ℝ → ℝ → ℝ → ℝ), 
    (∀ x y z, Real.sqrt ((a x y z)^2 + (a x y z)^2 + (a x y z)^2) + 
              Real.sqrt ((b x y z)^2 + (b x y z)^2 + (b x y z)^2) = 2) →
    (∀ x y z, Real.sqrt ((a x y z)^2 + (a x y z)^2 + (a x y z)^2) < 
              Real.sqrt ((b x y z)^2 + (b x y z)^2 + (b x y z)^2)) →
    (∀ x y z, (b x y z)^2 + (b x y z)^2 + (b x y z)^2 > 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l810_81068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_equals_one_l810_81064

theorem sum_reciprocals_equals_one (a b : ℝ) (h1 : (2 : ℝ)^a = 10) (h2 : (5 : ℝ)^b = 10) : 
  1/a + 1/b = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_equals_one_l810_81064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l810_81005

/-- For positive real numbers a and c, the maximum value of 3(a - x)(x + √(x^2 + c^2)) is 3/2(a^2 + c^2). -/
theorem max_value_expression (a c : ℝ) (ha : 0 < a) (hc : 0 < c) :
  (⨆ x, 3 * (a - x) * (x + Real.sqrt (x^2 + c^2))) = 3/2 * (a^2 + c^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l810_81005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_price_theorem_l810_81081

noncomputable def pencil_price_in_10000_won (original_price : ℝ) (discount : ℝ) : ℝ :=
  (original_price - discount) / 10000

theorem pencil_price_theorem (original_price : ℝ) (discount : ℝ) 
  (h1 : original_price = 5000)
  (h2 : discount = 200) :
  pencil_price_in_10000_won original_price discount = 0.48 := by
  -- Unfold the definition of pencil_price_in_10000_won
  unfold pencil_price_in_10000_won
  -- Substitute the values of original_price and discount
  rw [h1, h2]
  -- Perform the arithmetic
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_price_theorem_l810_81081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_palindromic_n_string_l810_81055

/-- An n-string is a string of digits formed by writing the numbers 1 to n in some order. -/
def IsNString (s : String) (n : ℕ) : Prop :=
  ∃ (perm : Fin n → Fin n), s = String.join (List.map (fun i => toString ((perm i).val + 1)) (List.finRange n))

/-- A string is palindromic if it reads the same forwards and backwards. -/
def IsPalindrome (s : String) : Prop :=
  s.toList = s.toList.reverse

/-- The main theorem stating that 19 is the smallest n > 1 for which a palindromic n-string exists. -/
theorem smallest_palindromic_n_string :
  (∀ n : ℕ, 1 < n ∧ n < 19 → ¬∃ s : String, IsNString s n ∧ IsPalindrome s) ∧
  (∃ s : String, IsNString s 19 ∧ IsPalindrome s) := by
  sorry

#check smallest_palindromic_n_string

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_palindromic_n_string_l810_81055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_becomes_convex_l810_81025

-- Define a polygon as a list of points in 2D space
def Polygon := List (ℝ × ℝ)

-- Define the property of being non-self-intersecting
def NonSelfIntersecting (p : Polygon) : Prop :=
  sorry

-- Define the property of being non-convex
def NonConvex (p : Polygon) : Prop :=
  sorry

-- Define the reflection operation
def ReflectPart (p : Polygon) (a b : ℝ × ℝ) : Polygon :=
  sorry

-- Define the property of being convex
def IsConvex (p : Polygon) : Prop :=
  sorry

-- Main theorem
theorem polygon_becomes_convex 
  (p : Polygon) 
  (h1 : NonSelfIntersecting p) 
  (h2 : NonConvex p) :
  ∃ (n : ℕ) (seq : ℕ → Polygon),
    seq 0 = p ∧
    (∀ i, i < n → ∃ a b, seq (i + 1) = ReflectPart (seq i) a b) ∧
    IsConvex (seq n) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_becomes_convex_l810_81025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l810_81049

/-- Given two points (m, n) and (m + 3, n + 21) in the coordinate plane,
    prove that the equation y = 7x + (n - 7m) represents the line passing through these points. -/
theorem line_equation_proof (m n : ℝ) : 
  let p : ℝ := 3
  let point1 : ℝ × ℝ := (m, n)
  let point2 : ℝ × ℝ := (m + p, n + 21)
  let slope : ℝ := 21 / p
  let line_eq : ℝ → ℝ := λ x ↦ slope * x + (n - slope * m)
  (∀ x y, ((x, y) = point1 ∨ (x, y) = point2) → y = line_eq x) ∧
  slope = 7 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l810_81049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PF₁F₂_l810_81099

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-3, 0)
def F₂ : ℝ × ℝ := (3, 0)

-- Define a point on the right branch of the hyperbola
noncomputable def P : ℝ × ℝ := sorry

-- Assume P satisfies the hyperbola equation
axiom P_on_hyperbola : hyperbola P.1 P.2

-- Assume P is on the right branch
axiom P_right_branch : P.1 > 0

-- Define the distances
noncomputable def PF₁ : ℝ := Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2)
noncomputable def PF₂ : ℝ := Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2)

-- Assume the given relationship between PF₁ and PF₂
axiom PF₁_eq_2PF₂ : PF₁ = 2 * PF₂

-- State the theorem
theorem area_of_triangle_PF₁F₂ : 
  ∃ (S : ℝ), S = 3 * Real.sqrt 15 ∧ 
  S = Real.sqrt (((PF₁ + PF₂ + 6) / 2) * 
                 ((PF₁ + PF₂ + 6) / 2 - PF₁) * 
                 ((PF₁ + PF₂ + 6) / 2 - PF₂) * 
                 ((PF₁ + PF₂ + 6) / 2 - 6)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PF₁F₂_l810_81099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_gain_per_year_l810_81089

/-- Calculate simple interest --/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem transaction_gain_per_year 
  (principal : ℝ) 
  (borrow_rate : ℝ) 
  (lend_rate : ℝ) 
  (time : ℝ) 
  (h1 : principal = 5000) 
  (h2 : borrow_rate = 4) 
  (h3 : lend_rate = 6) 
  (h4 : time = 2) :
  (simpleInterest principal lend_rate time - simpleInterest principal borrow_rate time) / time = 100 := by
  sorry

#check transaction_gain_per_year

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_gain_per_year_l810_81089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_QR_l810_81022

/-- Right triangle DEF with given side lengths -/
structure RightTriangle where
  DE : ℝ
  EF : ℝ
  DF : ℝ
  is_right : DE^2 + EF^2 = DF^2

/-- Circle centered at Q tangent to DE at D and passing through F -/
structure CircleQ where
  center : ℝ × ℝ
  radius : ℝ
  tangent_DE : True  -- Simplified condition for tangency
  passes_through_F : True  -- Simplified condition for passing through F

/-- Circle centered at R tangent to DF at F and passing through E -/
structure CircleR where
  center : ℝ × ℝ
  radius : ℝ
  tangent_DF : True  -- Simplified condition for tangency
  passes_through_E : True  -- Simplified condition for passing through E

/-- The main theorem -/
theorem length_QR (t : RightTriangle) (cq : CircleQ) (cr : CircleR) :
  t.DE = 9 ∧ t.EF = 12 ∧ t.DF = 15 →
  ∃ Q R : ℝ × ℝ, ‖Q - R‖ = 15.375 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_QR_l810_81022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l810_81027

open Real

-- Define the triangle
variable (A B C : ℝ)  -- Angles of the triangle
variable (a b c : ℝ)  -- Sides of the triangle
variable (D : ℝ)      -- Point where angle bisector intersects BC

-- Define the conditions
axiom condition1 : (sqrt 3 * cos (10 * π / 180) - sin (10 * π / 180)) * cos ((B + 35) * π / 180) = sin (80 * π / 180)
axiom condition2 : 2 * b * cos A = c - b
axiom condition3 : D = (A + B) / 2  -- Angle bisector property
axiom condition4 : 2 = sqrt ((c - D)^2 + (b * sin A)^2)  -- AD = 2 using distance formula

-- State the theorem
theorem triangle_problem :
  B = 15 * π / 180 ∧ c = sqrt 6 + sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l810_81027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greens_function_for_bvp_l810_81074

noncomputable def G (x ξ : ℝ) : ℝ :=
  if x ≤ ξ then
    (1/2*ξ - ξ^2 + 1/2*ξ^3)*x^2 - (1/6 - 1/2*ξ^2 + 1/3*ξ^3)*x^3
  else
    -1/6*ξ^3 + 1/2*ξ^2*x + (1/2*ξ^3 - ξ^2)*x^2 + (1/2*ξ^2 - 1/3*ξ^3)*x^3

theorem greens_function_for_bvp (x ξ : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → (deriv^[4] (fun x => G x ξ)) x = 0) ∧
  (G 0 ξ = 0) ∧
  ((deriv (fun x => G x ξ)) 0 = 0) ∧
  (G 1 ξ = 0) ∧
  ((deriv (fun x => G x ξ)) 1 = 0) ∧
  (∀ x ξ, G x ξ = G ξ x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greens_function_for_bvp_l810_81074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutations_with_substring_l810_81092

def original_string : String := "000011112222"
def substring : String := "2020"

def count_permutations_with_substring (s : String) (sub : String) : ℕ :=
  sorry

theorem permutations_with_substring (s : String) (sub : String) : 
  (count_permutations_with_substring s sub) = 3575 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutations_with_substring_l810_81092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_divides_polynomial_l810_81021

theorem quadratic_divides_polynomial (b : ℤ) : 
  (∃ q : Polynomial ℤ, (X^2 - 2*X + b) * q = X^15 + 2*X + 180) ↔ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_divides_polynomial_l810_81021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_real_axis_length_l810_81041

/-- Represents a hyperbola with equation x²/a² - y²/b² = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- The slope of the asymptote of a hyperbola -/
noncomputable def asymptote_slope (h : Hyperbola) : ℝ := h.b / h.a

/-- The x-coordinate of the right focus of a hyperbola -/
noncomputable def right_focus_x (h : Hyperbola) : ℝ := Real.sqrt (h.a^2 + h.b^2)

/-- The length of the real axis of a hyperbola -/
noncomputable def real_axis_length (h : Hyperbola) : ℝ := 2 * h.a

/-- Theorem stating the conditions and conclusion about the hyperbola -/
theorem hyperbola_real_axis_length 
  (h : Hyperbola) 
  (h_asymptote : asymptote_slope h = 2) 
  (h_focus : right_focus_x h = Real.sqrt 5) : 
  real_axis_length h = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_real_axis_length_l810_81041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_integer_to_x_minus_y_l810_81009

theorem nearest_integer_to_x_minus_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : |x| - y = 4) (h2 : |x| * y - x^3 = 1) : 
  ∃ (n : ℤ), n = 4 ∧ ∀ (m : ℤ), |↑m - (x - y)| ≥ |↑n - (x - y)| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_integer_to_x_minus_y_l810_81009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_proof_l810_81056

-- Define the point P
def P : ℝ × ℝ × ℝ := (2, 1, 1)

-- Define the normal vectors of the two planes that form the line
def n₁ : ℝ × ℝ × ℝ := (1, -3, 1)
def n₂ : ℝ × ℝ × ℝ := (3, -2, -2)

-- Define the normal vector of the plane we want to prove
def n : ℝ × ℝ × ℝ := (8, 5, 7)

-- Theorem statement
theorem plane_equation_proof :
  -- The plane passes through point P
  (8 * P.fst + 5 * P.snd.fst + 7 * P.snd.snd - 28 = 0) ∧
  -- The plane is perpendicular to the line (dot product of normals is zero)
  (n.fst * n₁.fst + n.snd.fst * n₁.snd.fst + n.snd.snd * n₁.snd.snd = 0) ∧
  (n.fst * n₂.fst + n.snd.fst * n₂.snd.fst + n.snd.snd * n₂.snd.snd = 0) :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_proof_l810_81056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_bound_l810_81076

theorem polynomial_bound {n : ℕ} (P : Polynomial ℝ) (h_deg : P.degree = n) :
  ∃ x₀ : ℝ, x₀ ∈ Set.Icc (-1 : ℝ) 1 ∧ |P.eval x₀| ≥ 1 / (2 ^ (n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_bound_l810_81076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_offset_length_l810_81060

/-- Represents a quadrilateral with a diagonal and two offsets -/
structure Quadrilateral where
  diagonal : ℝ
  offset1 : ℝ
  offset2 : ℝ

/-- Calculates the area of a quadrilateral given its diagonal and offsets -/
noncomputable def area (q : Quadrilateral) : ℝ :=
  (q.diagonal * (q.offset1 + q.offset2)) / 2

theorem offset_length (q : Quadrilateral) 
  (h1 : q.diagonal = 15)
  (h2 : q.offset2 = 4)
  (h3 : area q = 75) :
  q.offset1 = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_offset_length_l810_81060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_power_function_l810_81067

-- Define the power function that passes through (2, √2/2)
noncomputable def f (x : ℝ) : ℝ := x^(-1/2 : ℝ)

-- State the theorem
theorem fixed_point_power_function :
  f 2 = Real.sqrt 2 / 2 ∧ f 9 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_power_function_l810_81067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_sum_minimum_trig_sum_minimum_tight_l810_81023

theorem trig_sum_minimum (x : ℝ) : 
  |Real.sin x + Real.cos x + Real.tan x + (Real.tan x)⁻¹ + (Real.cos x)⁻¹ + (Real.sin x)⁻¹| ≥ 2 * Real.sqrt 2 - 1 :=
sorry

theorem trig_sum_minimum_tight : 
  ∃ x : ℝ, |Real.sin x + Real.cos x + Real.tan x + (Real.tan x)⁻¹ + (Real.cos x)⁻¹ + (Real.sin x)⁻¹| = 2 * Real.sqrt 2 - 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_sum_minimum_trig_sum_minimum_tight_l810_81023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lunar_rover_transport_cost_is_8325_l810_81082

/-- The cost of transporting an item to the International Space Station, in dollars per kilogram. -/
noncomputable def transport_cost_per_kg : ℚ := 25000

/-- The weight of the lunar rover in grams. -/
noncomputable def lunar_rover_weight_g : ℚ := 333

/-- Conversion factor from grams to kilograms. -/
noncomputable def grams_to_kg : ℚ := 1 / 1000

/-- The cost of transporting the lunar rover to the International Space Station. -/
noncomputable def lunar_rover_transport_cost : ℚ :=
  lunar_rover_weight_g * grams_to_kg * transport_cost_per_kg

theorem lunar_rover_transport_cost_is_8325 :
  lunar_rover_transport_cost = 8325 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lunar_rover_transport_cost_is_8325_l810_81082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_negative_300_degrees_l810_81087

/-- Given a point A(x, y) on the terminal side of angle -300°, where A is not the origin, prove that y/x = √3 -/
theorem terminal_side_negative_300_degrees (x y : ℝ) : 
  (x ≠ 0 ∨ y ≠ 0) →  -- A is not the origin
  (∃ r : ℝ, r > 0 ∧ x = r * Real.cos (-300 * Real.pi / 180) ∧ y = r * Real.sin (-300 * Real.pi / 180)) → -- A is on the terminal side of -300°
  y / x = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_negative_300_degrees_l810_81087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_focus_to_asymptote_l810_81059

/-- The parabola with equation y^2 = 8x -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- The hyperbola with equation x^2 - y^2/3 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

/-- The focus of the parabola y^2 = 8x -/
def focus : ℝ × ℝ := (2, 0)

/-- An asymptote of the hyperbola x^2 - y^2/3 = 1 -/
def asymptote (x y : ℝ) : Prop := x = (Real.sqrt 3 / 3) * y ∨ x = -(Real.sqrt 3 / 3) * y

/-- The distance from a point (a, b) to a line ax + by + c = 0 -/
noncomputable def distance_point_to_line (a b c x y : ℝ) : ℝ :=
  abs (a*x + b*y + c) / Real.sqrt (a^2 + b^2)

theorem distance_focus_to_asymptote :
  ∃ (x y : ℝ), asymptote x y ∧
  distance_point_to_line (Real.sqrt 3/3) (-1) 0 focus.1 focus.2 = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_focus_to_asymptote_l810_81059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_b_l810_81073

theorem triangle_angle_b (a b : ℝ) (A : ℝ) :
  a = 4 →
  b = 4 * Real.sqrt 3 →
  A = 30 * π / 180 →
  ∃ (B : ℝ), (B = 60 * π / 180 ∨ B = 120 * π / 180) ∧
    Real.sin B = (b * Real.sin A) / a :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_b_l810_81073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_unity_are_roots_of_polynomial_l810_81058

open Complex Real

def is_qth_root_of_unity (z : ℂ) (q : ℕ) : Prop := z^q = 1 ∧ ∀ k < q, z^k ≠ 1

theorem roots_of_unity_are_roots_of_polynomial
  (n k q : ℕ)
  (p : Polynomial ℂ)
  (hq : Nat.Prime q)
  (hn : p.degree = n)
  (hcoeff : ∀ i, p.coeff i ∈ ({-1, 0, 1} : Set ℂ))
  (hdiv : (X - 1)^k ∣ p)
  (hbound : (q : ℝ) / Real.log (q : ℝ) < (k : ℝ) / Real.log ((n + 1) : ℝ))
  (z : ℂ)
  (hz : is_qth_root_of_unity z q) :
  p.eval z = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_unity_are_roots_of_polynomial_l810_81058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_test_end_time_l810_81006

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Represents duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat

def add_time_and_duration (t : Time) (d : Duration) : Time := 
  let total_minutes := t.hours * 60 + t.minutes + d.hours * 60 + d.minutes
  let new_hours := (total_minutes / 60) % 24
  let new_minutes := total_minutes % 60
  ⟨new_hours, new_minutes, by sorry, by sorry⟩

theorem test_end_time :
  let start_time : Time := ⟨12, 35, by sorry, by sorry⟩
  let duration : Duration := ⟨4, 50⟩
  let end_time := add_time_and_duration start_time duration
  end_time = ⟨17, 25, by sorry, by sorry⟩ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_test_end_time_l810_81006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_line_tangent_to_circle_l810_81029

-- Define the initial line
def initial_line (x y : ℝ) : Prop := x + Real.sqrt 3 * y = 0

-- Define the rotation angle
noncomputable def rotation_angle : ℝ := Real.pi / 6  -- 30° in radians

-- Define the rotated line
def rotated_line (x y : ℝ) : Prop :=
  Real.sqrt 3 * x - y = 0

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 3

-- Define the tangency condition
def is_tangent (line : (ℝ → ℝ → Prop)) (circle : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x y : ℝ), line x y ∧ circle x y ∧
  ∀ (x' y' : ℝ), line x' y' → circle x' y' → (x', y') = (x, y)

-- Theorem statement
theorem rotated_line_tangent_to_circle :
  is_tangent rotated_line circle_eq :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_line_tangent_to_circle_l810_81029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l810_81095

theorem tan_alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l810_81095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l810_81044

/-- A function g with specific properties -/
noncomputable def g (D E F : ℤ) (x : ℝ) : ℝ := x^2 / (D*x^2 + E*x + F)

/-- Theorem stating that D + E + F = -8 given the properties of g -/
theorem sum_of_coefficients (D E F : ℤ) : 
  (∀ x > 3, g D E F x > 0.3) →
  (∀ x, x ≠ -3 ∧ x ≠ 2 → ContinuousAt (g D E F) x) →
  (∃ y, 0.3 < y ∧ y < 1 ∧ Filter.Tendsto (g D E F) Filter.atTop (nhds y)) →
  D + E + F = -8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l810_81044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_plane1_distance_point_to_plane2_l810_81078

/-- Calculate the distance from a point to a plane in 3D space -/
noncomputable def distancePointToPlane (x₀ y₀ z₀ a b c d : ℝ) : ℝ :=
  (|a * x₀ + b * y₀ + c * z₀ + d|) / Real.sqrt (a^2 + b^2 + c^2)

/-- Theorem: Distance from point (2, 3, -4) to plane 2x + 6y - 3z + 16 = 0 is 7 1/7 -/
theorem distance_point_to_plane1 :
  distancePointToPlane 2 3 (-4) 2 6 (-3) 16 = 7 + 1/7 := by sorry

/-- Theorem: Distance from point (2, -4, 1) to plane x - 8y + 4z = 0 is 4 2/9 -/
theorem distance_point_to_plane2 :
  distancePointToPlane 2 (-4) 1 1 (-8) 4 0 = 4 + 2/9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_plane1_distance_point_to_plane2_l810_81078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_partition_theorem_l810_81024

/-- Represents a cube with integer edge length -/
structure Cube where
  edge : ℕ

/-- Represents a partition of a larger cube into smaller cubes -/
structure CubePartition where
  original : Cube
  parts : List Cube

def validPartition (cp : CubePartition) : Prop :=
  -- The original cube has edge length 6
  cp.original.edge = 6 ∧
  -- All smaller cubes have whole number edge lengths
  (∀ c, c ∈ cp.parts → c.edge > 0) ∧
  -- The sum of volumes of smaller cubes equals the volume of the original cube
  (cp.parts.map (λ c => c.edge ^ 3)).sum = cp.original.edge ^ 3 ∧
  -- Not all smaller cubes are the same size
  ∃ c₁ c₂, c₁ ∈ cp.parts ∧ c₂ ∈ cp.parts ∧ c₁.edge ≠ c₂.edge

theorem cube_partition_theorem : 
  ∃ cp : CubePartition, validPartition cp ∧ cp.parts.length = 164 := by
  sorry

#check cube_partition_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_partition_theorem_l810_81024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_in_pascal_l810_81047

/-- Pascal's triangle is represented as a function from row and column indices to natural numbers -/
def pascal : ℕ → ℕ → ℕ := sorry

/-- A number appears in Pascal's triangle if there exist row and column indices for which pascal returns that number -/
def appearsInPascal (n : ℕ) : Prop :=
  ∃ (i j : ℕ), pascal i j = n

/-- The smallest four-digit number -/
def smallestFourDigit : ℕ := 1000

/-- Theorem: The smallest four-digit number in Pascal's triangle is 1000 -/
theorem smallest_four_digit_in_pascal :
  (∀ n, n < smallestFourDigit → ¬(appearsInPascal n)) ∧
  appearsInPascal smallestFourDigit := by
  sorry

#check smallest_four_digit_in_pascal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_in_pascal_l810_81047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_for_budget_l810_81072

/-- Represents the taxi fare structure in Metropolis -/
structure TaxiFare where
  initial_fare : ℚ
  initial_distance : ℚ
  standard_rate : ℚ
  standard_distance : ℚ
  discount_rate : ℚ

/-- Calculates the fare for a given distance -/
def calculate_fare (fare_structure : TaxiFare) (distance : ℚ) : ℚ :=
  if distance ≤ fare_structure.standard_distance then
    fare_structure.initial_fare + 
    fare_structure.standard_rate * (distance - fare_structure.initial_distance) / (1/10)
  else
    fare_structure.initial_fare + 
    fare_structure.standard_rate * (fare_structure.standard_distance - fare_structure.initial_distance) / (1/10) +
    fare_structure.discount_rate * (distance - fare_structure.standard_distance) / (1/10)

/-- Theorem stating the maximum distance that can be traveled with a given budget -/
theorem max_distance_for_budget 
  (fare_structure : TaxiFare)
  (budget : ℚ)
  (tip : ℚ)
  (h1 : fare_structure.initial_fare = 3)
  (h2 : fare_structure.initial_distance = 1/2)
  (h3 : fare_structure.standard_rate = 1/4)
  (h4 : fare_structure.standard_distance = 4)
  (h5 : fare_structure.discount_rate = 1/8)
  (h6 : budget = 12)
  (h7 : tip = 2) :
  ∃ (distance : ℚ), distance = 33/10 ∧ calculate_fare fare_structure distance + tip = budget := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_for_budget_l810_81072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_discount_rate_l810_81077

theorem max_discount_rate (cost_price selling_price : ℝ) 
  (h_cost : cost_price = 4)
  (h_selling : selling_price = 5)
  (h_profit_margin : ℝ → Prop) 
  (h_min_profit : h_profit_margin = λ x ↦ (selling_price * (1 - x / 100) - cost_price) / cost_price ≥ 0.1)
  : ∃ max_discount : ℝ, 
    max_discount = 12 ∧ 
    h_profit_margin max_discount ∧ 
    ∀ d, d > max_discount → ¬(h_profit_margin d) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_discount_rate_l810_81077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_four_thirds_l810_81012

/-- Given an infinite sequence a, where S_n is the sum of its first n terms -/
noncomputable def S (n : ℕ) : ℝ := 2^(n+1) + n - 2

/-- The n-th term of the sequence a -/
noncomputable def a (n : ℕ) : ℝ := 
  if n = 0 then 0 else S n - S (n-1)

/-- The sum of the series Σ(a_i / 4^i) from i=1 to infinity -/
noncomputable def series_sum : ℝ := ∑' i, (a i) / (4^i)

theorem series_sum_equals_four_thirds : series_sum = 4/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_four_thirds_l810_81012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l810_81084

-- Define the functions representing the curves
def f (x : ℝ) : ℝ := 2 * x + 1

noncomputable def g (x : ℝ) : ℝ := x + Real.log x

-- Define the distance function between the curves
noncomputable def distance (x : ℝ) : ℝ := f x - g x

-- Theorem statement
theorem min_distance_between_curves :
  ∃ (min_dist : ℝ), min_dist = 2 ∧
  ∀ (x : ℝ), x > 0 → distance x ≥ min_dist := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l810_81084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pratt_certificate_verification_algorithm_l810_81014

/-- Definition of a Pratt certificate -/
def is_pratt_certificate (n : ℕ) (T : List ℕ) : Prop :=
  sorry

/-- Time complexity function -/
def time_complexity (n : ℕ) : ℕ :=
  sorry

/-- Logarithm function for natural numbers -/
def log_nat (n : ℕ) : ℕ :=
  sorry

/-- The theorem statement -/
theorem pratt_certificate_verification_algorithm :
  ∃ (A : ℕ → List ℕ → Bool),
    (∀ n : ℕ, ∀ T : List ℕ,
      A n T = true ↔ is_pratt_certificate n T) ∧
    (∃ c : ℕ, ∀ n : ℕ, time_complexity n ≤ c * (log_nat n)^5) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pratt_certificate_verification_algorithm_l810_81014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_approximation_l810_81066

/-- Given a principal amount and an interest rate, calculates the simple interest for a given time period. -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Given a principal amount and an interest rate, calculates the compound interest for a given time period. -/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

/-- Theorem stating that if the simple interest for 2 years is 326 and the compound interest for 2 years is 340,
    then the interest rate is approximately 8.59%. -/
theorem interest_rate_approximation (P : ℝ) (R : ℝ) 
    (h1 : simpleInterest P R 2 = 326)
    (h2 : compoundInterest P R 2 = 340) :
    ∃ ε > 0, |R - 8.59| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_approximation_l810_81066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_r_values_l810_81004

open BigOperators

variable {n : ℕ}
variable (a r : Fin n → ℝ)
variable (x : Fin n → ℝ)

def sum_squares (v : Fin n → ℝ) : ℝ := ∑ i, (v i)^2

theorem inequality_implies_r_values
  (h_a_nonzero : ∀ i, a i ≠ 0)
  (h_inequality : ∀ x : Fin n → ℝ,
    ∑ i, r i * (x i - a i) ≤ Real.sqrt (sum_squares x) - Real.sqrt (sum_squares a)) :
  ∀ i, r i = a i / Real.sqrt (sum_squares a) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_r_values_l810_81004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_of_fortune_jeopardy_ratio_l810_81090

/-- Proves that the ratio of the length of one episode of Wheel of Fortune to one episode of Jeopardy is 2:1 --/
theorem wheel_of_fortune_jeopardy_ratio : 
  (let total_watch_time : ℕ := 2 * 60  -- 2 hours in minutes
   let jeopardy_episodes : ℕ := 2
   let wheel_of_fortune_episodes : ℕ := 2
   let jeopardy_length : ℕ := 20  -- minutes per episode
   let jeopardy_total_time : ℕ := jeopardy_episodes * jeopardy_length
   let wheel_of_fortune_total_time : ℕ := total_watch_time - jeopardy_total_time
   let wheel_of_fortune_length : ℕ := wheel_of_fortune_total_time / wheel_of_fortune_episodes
   (wheel_of_fortune_length : ℚ) / jeopardy_length = 2 / 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_of_fortune_jeopardy_ratio_l810_81090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_is_sqrt_5_l810_81048

/-- The distance from the origin to a line given by y = mx + b -/
noncomputable def distance_origin_to_line (m b : ℝ) : ℝ :=
  |b| / Real.sqrt (1 + m^2)

/-- The slope of the line -/
noncomputable def m : ℝ := -1/2

/-- The y-intercept of the line -/
noncomputable def b : ℝ := 5/2

theorem distance_to_line_is_sqrt_5 :
  distance_origin_to_line m b = Real.sqrt 5 := by
  -- Unfold definitions
  unfold distance_origin_to_line m b
  -- Simplify the expression
  simp [Real.sqrt_eq_rpow]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_is_sqrt_5_l810_81048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l810_81045

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2*a - 1)*x + 7*a - 2 else a^x

theorem a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) →
  3/8 < a ∧ a < 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l810_81045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l810_81079

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  Real.sin (Real.pi - ω * x) * Real.cos (ω * x) + (Real.cos (ω * x))^2

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem function_properties (ω : ℝ) (h1 : ω > 0) 
  (h2 : has_period (f ω) Real.pi) (h3 : ∀ p, 0 < p → p < Real.pi → ¬ has_period (f ω) p) :
  (ω = 1) ∧ 
  (∀ x ∈ Set.Icc 0 (Real.pi / 16), f ω (2 * x) ≥ 1) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 16), f ω (2 * x) = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l810_81079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_interval_l810_81017

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - x/2 > 0}
def N : Set ℝ := {x | Real.log x ≤ 0}

-- Define the interval (1/2, 1]
def target_interval : Set ℝ := Set.Ioc (1/2) 1

-- Theorem statement
theorem intersection_equals_interval : M ∩ N = target_interval := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_interval_l810_81017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_product_l810_81050

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define a line passing through the focus
def line_through_focus (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 2)

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) (k : ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
  line_through_focus k A.1 A.2 ∧ line_through_focus k B.1 B.2

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem parabola_intersection_product (A B : ℝ × ℝ) (k : ℝ) :
  intersection_points A B k →
  distance A B = 10 →
  distance A focus * distance B focus = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_product_l810_81050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_slope_tangent_line_deriv_f_l810_81085

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log x + x^2

-- State the theorem
theorem min_slope_tangent_line :
  ∃ (k : ℝ), k = 2 * Real.sqrt 2 ∧ ∀ x > 0, (1 / x + 2 * x) ≥ k := by
  -- We'll use k = 2√2 as our minimum value
  let k := 2 * Real.sqrt 2
  
  -- Prove that this k satisfies our conditions
  have h1 : k = 2 * Real.sqrt 2 := rfl
  
  -- Prove that for all x > 0, (1/x + 2x) ≥ 2√2
  have h2 : ∀ x > 0, (1 / x + 2 * x) ≥ k := by
    intro x hx
    -- This is where we'd normally prove the inequality
    -- For now, we'll use sorry to skip the proof
    sorry
  
  -- Combine our results
  exact ⟨k, h1, h2⟩

-- Optionally, we can state a theorem about the derivative of f
theorem deriv_f (x : ℝ) (hx : x > 0) : 
  deriv f x = 1/x + 2*x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_slope_tangent_line_deriv_f_l810_81085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_eccentricity_and_asymptotes_l810_81070

/-- Given hyperbola -/
noncomputable def given_hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

/-- Candidate hyperbola -/
noncomputable def candidate_hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 3 = 1

/-- Eccentricity of a hyperbola -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

/-- Asymptotes of a hyperbola -/
noncomputable def asymptotes (a b : ℝ) (x : ℝ) : Set ℝ := {y | y = b / a * x ∨ y = -b / a * x}

/-- Theorem stating that the candidate hyperbola has the same eccentricity and asymptotes as the given hyperbola -/
theorem same_eccentricity_and_asymptotes :
  let a₁ := Real.sqrt 3
  let b₁ := 1
  let a₂ := 3
  let b₂ := Real.sqrt 3
  eccentricity a₁ b₁ = eccentricity a₂ b₂ ∧
  asymptotes a₁ b₁ = asymptotes a₂ b₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_eccentricity_and_asymptotes_l810_81070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_proof_l810_81065

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  area : Real

-- Define the theorem
theorem triangle_ratio_proof (abc : Triangle) 
  (h1 : abc.a = 1)
  (h2 : abc.B = Real.pi / 4)
  (h3 : abc.area = 2) :
  abc.b / Real.sin abc.B = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_proof_l810_81065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_arithmetic_sequence_implies_a_values_l810_81036

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - 3 / x + a - 2

/-- Predicate to check if three real numbers form an arithmetic sequence -/
def isArithmeticSequence (x y z : ℝ) : Prop := y - x = z - y

theorem three_zeros_arithmetic_sequence_implies_a_values (a : ℝ) :
  (∃ x y z : ℝ, x < y ∧ y < z ∧
    f a x = 0 ∧ f a y = 0 ∧ f a z = 0 ∧
    isArithmeticSequence x y z) →
  a = (5 + 3 * Real.sqrt 33) / 8 ∨ a = -9 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_arithmetic_sequence_implies_a_values_l810_81036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_parallel_l810_81019

-- Define the vector type
def MyVector := ℝ × ℝ

-- Define parallel vectors
def parallel (v w : MyVector) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

-- Define vector addition
def add (v w : MyVector) : MyVector :=
  (v.1 + w.1, v.2 + w.2)

-- Define scalar multiplication
def smul (r : ℝ) (v : MyVector) : MyVector :=
  (r * v.1, r * v.2)

-- Theorem statement
theorem vector_sum_parallel (y : ℝ) :
  let a : MyVector := (-1, 2)
  let b : MyVector := (2, y)
  parallel a b →
  add (smul 3 a) (smul 2 b) = (1, -2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_parallel_l810_81019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_numbers_l810_81031

/-- The number of positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

/-- The least common multiple of a and b -/
def my_lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem sum_of_special_numbers (a b : ℕ) :
  a > 0 ∧ b > 0 ∧
  num_divisors a = 9 ∧
  num_divisors b = 10 ∧
  my_lcm a b = 4400 →
  a + b = 276 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_numbers_l810_81031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_conditions_l810_81002

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1/x - x + a * log x

-- State the theorem
theorem extreme_points_conditions (a k : ℝ) :
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧
    (∀ x : ℝ, x > 0 → f a x ≤ max (f a x₁) (f a x₂)) ∧
    (f a x₂ - f a x₁ ≥ k*a - 3)) →
  (a > 2 ∧ k ≤ 2 * log 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_conditions_l810_81002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_and_g_inequality_l810_81042

open Real

noncomputable def f (x : ℝ) := max (x^2 - 1) (2 * log x)
noncomputable def g (a x : ℝ) := max (x + log x) (a * x^2 + x)

theorem f_range_and_g_inequality :
  (∃ (y : ℝ), y ∈ Set.Icc (-3/4) 3 ↔ ∃ (x : ℝ), x ∈ Set.Icc (1/2) 1 ∧ f x = y) ∧
  (∃ (a : ℝ), ∀ (x : ℝ), x > 1 → g a x < (3/2) * x + 4 * a) ∧
  (∀ (a : ℝ), (∀ (x : ℝ), x > 1 → g a x < (3/2) * x + 4 * a) →
    a ∈ Set.Ioo ((log 2 - 1) / 4) 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_and_g_inequality_l810_81042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l810_81040

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (a * x) * Real.sin x

-- Define the type for natural numbers excluding 0
def PositiveNat := {n : ℕ // n > 0}

-- Define the sequence of extremum points
noncomputable def extremum_points (a : ℝ) : PositiveNat → ℝ
| ⟨n, _⟩ => n * Real.pi - Real.arctan (1 / a)

-- Statement of the theorem
theorem geometric_sequence_property (a : ℝ) (h : a > 0) :
  ∃ (r : ℝ), r = -Real.exp (a * Real.pi) ∧
  ∀ (n : PositiveNat),
    f a (extremum_points a ⟨n.1 + 1, Nat.succ_pos n.1⟩) = r * f a (extremum_points a n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l810_81040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neither_odd_nor_even_f_increasing_on_positive_reals_f_range_on_interval_l810_81061

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) / x

-- Theorem 1: f is neither odd nor even
theorem f_neither_odd_nor_even :
  ∀ x : ℝ, x ≠ 0 → (f (-x) ≠ f x ∧ f (-x) ≠ -f x) := by
  sorry

-- Theorem 2: f is monotonically increasing on (0, +∞)
theorem f_increasing_on_positive_reals :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f x₁ < f x₂ := by
  sorry

-- Theorem 3: Range of f on [3, 5] is [5/3, 9/5]
theorem f_range_on_interval :
  ∀ y : ℝ, 5/3 ≤ y ∧ y ≤ 9/5 → ∃ x : ℝ, 3 ≤ x ∧ x ≤ 5 ∧ f x = y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neither_odd_nor_even_f_increasing_on_positive_reals_f_range_on_interval_l810_81061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_inscribed_sphere_properties_l810_81013

/-- Represents a cone with an inscribed sphere -/
structure ConeWithInscribedSphere where
  /-- The ratio of the sphere's surface area to the cone's base area -/
  k : ℝ
  /-- The angle between the cone's slant height and the plane of its base -/
  α : ℝ

/-- Theorem about the properties of a cone with an inscribed sphere -/
theorem cone_inscribed_sphere_properties (cone : ConeWithInscribedSphere) :
  Real.cos cone.α = (4 - cone.k) / (4 + cone.k) ∧ 0 < cone.k ∧ cone.k < 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_inscribed_sphere_properties_l810_81013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_rectangle_perimeter_sum_l810_81000

/-- Represents a folded rectangle with given dimensions -/
structure FoldedRectangle where
  px : ℚ
  qx : ℚ
  ry : ℚ

/-- Calculate the perimeter of the folded rectangle -/
noncomputable def perimeter (r : FoldedRectangle) : ℚ :=
  2 * (((r.px^2 + r.qx^2).sqrt) + (((r.qx^2 + r.ry^2).sqrt + (r.px^2 + r.qx^2).sqrt)^2 - (r.px^2 + r.qx^2)) / (2 * (r.px^2 + r.qx^2).sqrt))

/-- The main theorem stating the sum of numerator and denominator of the perimeter fraction -/
theorem folded_rectangle_perimeter_sum (r : FoldedRectangle) 
    (h_px : r.px = 5) 
    (h_qx : r.qx = 12) 
    (h_ry : r.ry = 4) : 
  ∃ (m n : ℕ), Nat.Coprime m n ∧ perimeter r = m / n ∧ m + n = 735 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_rectangle_perimeter_sum_l810_81000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_exists_iff_a_eq_19_l810_81035

-- Define the system of equations
def system (x y a : ℝ) : Prop :=
  (x - a)^2 = 16 * (y - x + a - 3) ∧ 
  (y/3) = (x/3)^1 ∧
  x/3 > 0 ∧ 
  x ≠ 3

-- Theorem statement
theorem system_solution_exists_iff_a_eq_19 :
  ∃ a, (∃ x y, system x y a) ↔ a = 19 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_exists_iff_a_eq_19_l810_81035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_swappable_students_l810_81098

/-- Represents a school with its enrolled and present students -/
structure School where
  enrolled : Finset ℕ
  present : Finset ℕ

/-- The problem setup -/
structure SchoolProblem where
  schools : Finset School
  total_misplaced : ℕ

/-- Axioms for the school problem -/
axiom school_count (sp : SchoolProblem) : sp.schools.card = 3

axiom enrolled_count (sp : SchoolProblem) (s : School) : 
  s ∈ sp.schools → s.enrolled.card = 100

axiom present_count (sp : SchoolProblem) (s : School) :
  s ∈ sp.schools → s.present.card = 100

axiom misplaced_count (sp : SchoolProblem) : sp.total_misplaced = 40

/-- Definition of a misplaced student -/
def is_misplaced (sp : SchoolProblem) (student : ℕ) : Prop :=
  ∃ s₁ s₂ : School, s₁ ∈ sp.schools ∧ s₂ ∈ sp.schools ∧ s₁ ≠ s₂ ∧ 
    student ∈ s₁.enrolled ∧ student ∈ s₂.present

/-- The main theorem to prove -/
theorem exist_swappable_students (sp : SchoolProblem) : 
  ∃ student₁ student₂ : ℕ, 
    is_misplaced sp student₁ ∧ 
    is_misplaced sp student₂ ∧
    student₁ ≠ student₂ ∧
    (∃ s₁ s₂ s₃ : School, 
      s₁ ∈ sp.schools ∧ s₂ ∈ sp.schools ∧ s₃ ∈ sp.schools ∧
      s₁ ≠ s₂ ∧ s₂ ≠ s₃ ∧ s₁ ≠ s₃ ∧
      student₁ ∈ s₁.enrolled ∧ student₁ ∈ s₂.present ∧
      student₂ ∈ s₂.enrolled ∧ student₂ ∈ s₃.present) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_swappable_students_l810_81098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_radii_l810_81034

/-- A triangle with known sides and angles -/
structure Triangle where
  sides_angles_known : Bool
  max_side_length : ℝ
  angle_opposite_max_side : ℝ
  has_inscribed_circle : ℝ → Prop
  has_circumscribed_circle : ℝ → Prop
  has_intersecting_circle : ℝ → Prop

theorem triangle_circle_radii 
  (T : Triangle) 
  (h_sides_angles : T.sides_angles_known) :
  ∃ (r₀ R_min R_max ρ_max : ℝ),
    (∀ r : ℝ, r > 0 → r ≤ r₀ → T.has_inscribed_circle r) ∧
    (∀ R : ℝ, R_min ≤ R ∧ R ≤ R_max → T.has_circumscribed_circle R) ∧
    (∀ ρ : ℝ, r₀ ≤ ρ ∧ ρ ≤ ρ_max → T.has_intersecting_circle ρ) ∧
    r₀ > 0 ∧
    R_min = 1/2 ∧
    R_max = Real.sqrt 3 / 3 ∧
    ρ_max = (T.max_side_length) / (2 * Real.sin T.angle_opposite_max_side) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_radii_l810_81034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l810_81051

def U : Set ℕ := {1,2,3,4,5,6,7,8}

def A : Set ℕ := {x ∈ U | x^2 - 3*x + 2 = 0}

def B : Set ℕ := {x ∈ U | 1 ≤ x ∧ x ≤ 5}

def C : Set ℕ := {x ∈ U | 2 < x ∧ x < 9}

theorem set_operations :
  (A ∩ B = {1,2}) ∧
  (A ∪ (B ∩ C) = {1,2,3,4,5}) ∧
  ((U \ B) ∪ (U \ C) = {1,2,6,7,8}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l810_81051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teresa_speed_l810_81075

-- Define the distance and time
noncomputable def distance : ℝ := 25
noncomputable def time : ℝ := 5

-- Define the speed calculation
noncomputable def speed : ℝ := distance / time

-- Theorem to prove
theorem teresa_speed : speed = 5 := by
  -- Unfold the definitions
  unfold speed distance time
  -- Simplify the fraction
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teresa_speed_l810_81075

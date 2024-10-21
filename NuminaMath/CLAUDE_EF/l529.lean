import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_shared_digit_l529_52935

/-- The set of two-digit positive integers -/
def TwoDigitIntegers : Set ℕ := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

/-- The number of two-digit positive integers -/
noncomputable def numTwoDigitIntegers : ℕ := Finset.card (Finset.filter (fun n => 10 ≤ n ∧ n ≤ 99) (Finset.range 100))

/-- Two numbers share a digit -/
def shareDigit (a b : ℕ) : Prop :=
  ∃ (d : ℕ), (d < 10 ∧ ((a / 10 = d ∨ a % 10 = d) ∧ (b / 10 = d ∨ b % 10 = d)))

/-- The number of pairs of two-digit integers that share at least one digit -/
def numSharedDigitPairs : ℕ := 1377

theorem probability_shared_digit :
  (numSharedDigitPairs : ℚ) / (numTwoDigitIntegers.choose 2 : ℚ) = 153 / 445 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_shared_digit_l529_52935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_football_team_size_correct_l529_52913

/-- The number of players on a football team given water distribution information --/
def football_team_size 
  (initial_water : ℕ)     -- Initial amount of water in liters
  (player_water : ℕ)      -- Amount of water each player receives in milliliters
  (spilled_water : ℕ)     -- Amount of water spilled in milliliters
  (leftover_water : ℕ)    -- Amount of water left over in milliliters
  : ℕ :=
  let total_water_ml := initial_water * 1000
  let available_water := total_water_ml - spilled_water
  let used_water := available_water - leftover_water
  used_water / player_water

theorem football_team_size_correct
  (initial_water : ℕ)
  (player_water : ℕ)
  (spilled_water : ℕ)
  (leftover_water : ℕ)
  (h1 : initial_water = 8)
  (h2 : player_water = 200)
  (h3 : spilled_water = 250)
  (h4 : leftover_water = 1750)
  : football_team_size initial_water player_water spilled_water leftover_water = 30 := by
  sorry

#eval football_team_size 8 200 250 1750

end NUMINAMATH_CALUDE_ERRORFEEDBACK_football_team_size_correct_l529_52913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friend_meeting_problem_l529_52918

/-- The duration of stay in minutes -/
noncomputable def n : ℝ := 60 - 30 * Real.sqrt 2

/-- The probability of the two friends meeting -/
noncomputable def probability_of_meeting : ℝ := 1 - (60 - n)^2 / 3600

/-- Representation of n as d - e√f -/
def d : ℕ := 60
def e : ℕ := 30
def f : ℕ := 2

theorem friend_meeting_problem :
  probability_of_meeting = 1/2 ∧
  n = d - e * Real.sqrt f ∧
  Nat.Prime f ∧
  d + e + f = 92 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_friend_meeting_problem_l529_52918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_P_parallel_to_polar_axis_l529_52984

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Converts a polar point to Cartesian coordinates -/
noncomputable def polarToCartesian (p : PolarPoint) : ℝ × ℝ :=
  (p.r * Real.cos p.θ, p.r * Real.sin p.θ)

/-- The given point P in polar coordinates -/
noncomputable def P : PolarPoint :=
  { r := 2, θ := Real.pi / 6 }

theorem line_equation_through_P_parallel_to_polar_axis :
  ∀ (ρ θ : ℝ), (ρ * Real.sin θ = 1) ↔ 
  (∃ (t : ℝ), polarToCartesian { r := ρ, θ := θ } = 
    (t, (polarToCartesian P).2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_P_parallel_to_polar_axis_l529_52984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l529_52940

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.tan (3 * x + Real.pi / 4)

-- State the theorem about the minimum positive period of f(x)
theorem min_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  T = Real.pi / 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l529_52940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proportional_rows_or_columns_for_3x3_counterexample_properties_l529_52980

-- Define a complex matrix type
def ComplexMatrix (n : ℕ) := Matrix (Fin n) (Fin n) ℂ

-- Define a property for matrices with all elements having modulus 1
def all_elements_unit_modulus {n : ℕ} (A : ComplexMatrix n) : Prop :=
  ∀ i j, Complex.abs (A i j) = 1

-- Define a property for non-invertible matrices
def non_invertible {n : ℕ} (A : ComplexMatrix n) : Prop :=
  Matrix.det A = 0

-- Define a property for proportional rows
def has_proportional_rows {n : ℕ} (A : ComplexMatrix n) : Prop :=
  ∃ i j, i ≠ j ∧ ∃ c : ℂ, ∀ k, A i k = c * A j k

-- Define a property for proportional columns
def has_proportional_columns {n : ℕ} (A : ComplexMatrix n) : Prop :=
  ∃ i j, i ≠ j ∧ ∃ c : ℂ, ∀ k, A k i = c * A k j

-- The main theorem
theorem proportional_rows_or_columns_for_3x3 (A : ComplexMatrix 3) 
  (h1 : all_elements_unit_modulus A) (h2 : non_invertible A) :
  has_proportional_rows A ∨ has_proportional_columns A :=
sorry

-- Example of a 4x4 matrix that is non-invertible but doesn't have proportional rows or columns
def counterexample_4x4 : ComplexMatrix 4 :=
λ i j =>
  match i, j with
  | 0, 0 => -1
  | 0, 1 => 1
  | 0, 2 => Complex.I
  | 0, 3 => -Complex.I
  | 1, 0 => 1
  | 1, 1 => -1
  | 1, 2 => -Complex.I
  | 1, 3 => Complex.I
  | 2, 0 => Complex.I
  | 2, 1 => -Complex.I
  | 2, 2 => -1
  | 2, 3 => 1
  | 3, 0 => -Complex.I
  | 3, 1 => Complex.I
  | 3, 2 => 1
  | 3, 3 => -1

-- Theorem stating that the counterexample satisfies the required properties
theorem counterexample_properties :
  all_elements_unit_modulus counterexample_4x4 ∧
  non_invertible counterexample_4x4 ∧
  ¬(has_proportional_rows counterexample_4x4 ∨ has_proportional_columns counterexample_4x4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proportional_rows_or_columns_for_3x3_counterexample_properties_l529_52980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_polar_form_l529_52981

noncomputable def cis (angle : ℝ) : ℂ := Complex.exp (angle * Complex.I)

theorem complex_product_polar_form :
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧
  (5 * cis (42 * π / 180)) * (4 * cis (85 * π / 180)) = r * cis θ ∧
  r = 20 ∧ θ = 127 * π / 180 := by
  sorry

#check complex_product_polar_form

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_polar_form_l529_52981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_4_and_6_l529_52908

noncomputable def arithmetic_mean (a b : ℝ) : ℝ := (a + b) / 2

theorem arithmetic_mean_of_4_and_6 :
  arithmetic_mean 4 6 = 5 := by
  unfold arithmetic_mean
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_4_and_6_l529_52908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_with_pi_3_asymptote_l529_52975

/-- A hyperbola with center at the origin and axes along the coordinate axes. -/
structure Hyperbola where
  a : ℝ  -- distance from center to vertex along transverse axis
  b : ℝ  -- distance from center to asymptote along conjugate axis

/-- The eccentricity of a hyperbola. -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := 
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The angle of inclination of an asymptote of a hyperbola. -/
noncomputable def asymptote_angle (h : Hyperbola) : ℝ := 
  Real.arctan (h.b / h.a)

theorem hyperbola_eccentricity_with_pi_3_asymptote :
  ∀ h : Hyperbola, 
    asymptote_angle h = π / 3 → 
    eccentricity h = 2 ∨ eccentricity h = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_with_pi_3_asymptote_l529_52975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mpg_difference_approx_six_l529_52970

/-- Represents the fuel efficiency of a car in different driving conditions -/
structure CarEfficiency where
  highway_miles_per_tankful : ℚ
  city_miles_per_tankful : ℚ
  city_miles_per_gallon : ℚ

/-- Calculates the difference in miles per gallon between highway and city driving -/
noncomputable def mpg_difference (car : CarEfficiency) : ℚ :=
  let tank_size := car.city_miles_per_tankful / car.city_miles_per_gallon
  let highway_mpg := car.highway_miles_per_tankful / tank_size
  highway_mpg - car.city_miles_per_gallon

/-- Theorem stating that the difference in miles per gallon is approximately 6 -/
theorem mpg_difference_approx_six (car : CarEfficiency) 
  (h1 : car.highway_miles_per_tankful = 448)
  (h2 : car.city_miles_per_tankful = 336)
  (h3 : car.city_miles_per_gallon = 18) :
  ∃ ε : ℚ, ε > 0 ∧ |mpg_difference car - 6| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mpg_difference_approx_six_l529_52970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factorial_divisible_by_990_l529_52915

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_factorial_divisible_by_990 : 
  ∃ n : ℕ, (n.factorial % 990 = 0) ∧ (∀ m : ℕ, m < n → m.factorial % 990 ≠ 0) ∧ n = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factorial_divisible_by_990_l529_52915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_A_range_l529_52900

/-- A function f with given properties -/
noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

/-- Theorem stating the properties of the function and the range of A -/
theorem function_properties_and_A_range 
  (A ω φ : ℝ) 
  (h_A : A > 0) 
  (h_ω : ω > 0) 
  (h_φ : 0 < φ ∧ φ < Real.pi) 
  (h_period : ∀ x, f A ω φ (x + Real.pi) = f A ω φ x) 
  (h_max : ∀ x, f A ω φ x ≤ f A ω φ (Real.pi / 12)) 
  (h_solution : ∃ x ∈ Set.Icc (-Real.pi / 4) 0, f A ω φ x - 1 + A = 0) :
  ω = 2 ∧ φ = Real.pi / 3 ∧ 4 - 2 * Real.sqrt 3 ≤ A ∧ A ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_A_range_l529_52900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_graph_l529_52922

-- Define a continuous function f on the real line
variable (f : ℝ → ℝ)

-- Define the transformed function g
noncomputable def g (f : ℝ → ℝ) (x : ℝ) : ℝ := (1/3) * f x - 2

-- State the theorem
theorem transformed_graph (f : ℝ → ℝ) (x y : ℝ) :
  y = g f x ↔ (y + 2) = (1/3) * f x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_graph_l529_52922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l529_52978

-- Define the circle equation
noncomputable def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y + 1 = 0

-- Define the line equation
noncomputable def line_equation (a b x y : ℝ) : Prop :=
  2*a*x - b*y + 2 = 0

-- Define the symmetry condition
noncomputable def symmetry_condition (a b : ℝ) : Prop :=
  line_equation a b (-1) 2

-- Define the objective function
noncomputable def objective_function (a b : ℝ) : ℝ :=
  4/a + 1/b

-- State the theorem
theorem min_value_theorem (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (hsym : symmetry_condition a b) : 
  ∃ (min_val : ℝ), min_val = 9 ∧ 
  ∀ (a' b' : ℝ), a' > 0 → b' > 0 → symmetry_condition a' b' → 
  objective_function a' b' ≥ min_val := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l529_52978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_identity_l529_52992

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x) / (x + 2)

-- State the theorem
theorem composition_identity (a b : ℝ) :
  (∀ x : ℝ, x ≠ -2 → f a (f a x) = x) ↔ (a = -4 ∧ b = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_identity_l529_52992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_solutions_l529_52942

theorem count_integer_solutions : 
  ∃! (S : Finset ℤ), 
    (∀ x ∈ S, ((x^2 - x - 2 : ℤ) : ℚ)^(x+3) = 1) ∧ 
    (∀ x : ℤ, ((x^2 - x - 2 : ℤ) : ℚ)^(x+3) = 1 → x ∈ S) ∧ 
    Finset.card S = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_solutions_l529_52942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_odd_function_zero_l529_52929

-- Define the function f on the interval [-3, 3]
def f : ℝ → ℝ := sorry

-- Define the property of f being an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_property (h_odd : is_odd f) (h_f3 : f 3 = -3) :
  f (-3) + f 0 = 3 := by
  sorry

-- Prove that f(0) = 0 for odd functions
theorem odd_function_zero (f : ℝ → ℝ) (h_odd : is_odd f) : f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_odd_function_zero_l529_52929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_guests_proof_l529_52926

def total_food : ℚ := 337
def max_food_per_guest : ℚ := 2

def min_guests_required : ℕ :=
  Nat.ceil (total_food / max_food_per_guest)

#eval min_guests_required

theorem min_guests_proof :
  min_guests_required = 169 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_guests_proof_l529_52926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_problem_3_problem_4_l529_52989

theorem problem_1 : (-1 : ℤ) ^ 2023 + (Real.pi - 3.14) ^ (0 : ℕ) - (-1/2 : ℚ) ^ (-2 : ℤ) = -4 := by
  sorry

theorem problem_2 (x : ℝ) : (1/4 * x^4 + 2*x^3 - 4*x^2) / ((-2*x)^2) = 1/16 * x^2 + 1/2 * x - 1 := by
  sorry

theorem problem_3 (x y : ℝ) : (2*x + y + 1) * (2*x + y - 1) = 4*x^2 + 4*x*y + y^2 - 1 := by
  sorry

theorem problem_4 (x : ℝ) : (2*x + 3) * (2*x - 3) - (2*x - 1)^2 = 4*x - 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_problem_3_problem_4_l529_52989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_equality_l529_52954

theorem trig_expression_equality : 
  (1 / Real.sin (π / 12)) + 4 * Real.sin (π / 6) - 2 * Real.cos (π / 3) = Real.sqrt 6 + Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_equality_l529_52954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_equation_solution_l529_52969

theorem factorial_equation_solution :
  ∀ (a b c : ℕ), 
    (a > 0 ∧ b > 0 ∧ c > 0) →
    (Nat.factorial a * Nat.factorial b = Nat.factorial a + Nat.factorial b + Nat.factorial c) ↔ 
    (a = 3 ∧ b = 3 ∧ c = 4) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_equation_solution_l529_52969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_proof_l529_52965

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 16*y

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 4)

-- Define the directrix of the parabola
def directrix (y : ℝ) : Prop := y = -4

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 64

-- Theorem statement
theorem circle_equation_proof :
  ∀ x y : ℝ,
  parabola x y →
  (∃ r : ℝ, ∀ x' y' : ℝ, (x' - focus.1)^2 + (y' - focus.2)^2 = r^2 ∧
                         (∃ y_dir : ℝ, directrix y_dir ∧
                                       (x' - focus.1)^2 + (y' - focus.2)^2 = (y' - y_dir)^2)) →
  circle_eq x y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_proof_l529_52965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_values_a_neg_two_monotonous_range_a_min_value_f_l529_52974

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 3

-- Define the domain of x
def domain : Set ℝ := Set.Icc (-4) 6

-- Theorem 1: Maximum and minimum values when a = -2
theorem max_min_values_a_neg_two :
  (∀ x ∈ domain, f (-2) x ≤ 35) ∧
  (∀ x ∈ domain, f (-2) x ≥ -1) ∧
  (∃ x ∈ domain, f (-2) x = 35) ∧
  (∃ x ∈ domain, f (-2) x = -1) :=
sorry

-- Theorem 2: Range of a for monotonous function
theorem monotonous_range_a :
  ∀ a : ℝ, (∀ x y : ℝ, x ∈ domain → y ∈ domain → x < y → f a x < f a y) ∨
           (∀ x y : ℝ, x ∈ domain → y ∈ domain → x < y → f a x > f a y)
  ↔ a ∈ Set.Ici 4 ∪ Set.Iic (-6) :=
sorry

-- Theorem 3: Minimum value of f(x)
theorem min_value_f :
  ∀ a : ℝ, (∀ x ∈ domain, f a x ≥
    (if a < -6 then 39 + 12*a
     else if a ≤ 4 then -a^2 + 3
     else 19 - 8*a)) ∧
  (∃ x ∈ domain, f a x =
    (if a < -6 then 39 + 12*a
     else if a ≤ 4 then -a^2 + 3
     else 19 - 8*a)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_values_a_neg_two_monotonous_range_a_min_value_f_l529_52974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_l529_52947

-- Define the point in rectangular coordinates
def x : ℝ := 2
def y : ℝ := -2

-- Define the polar coordinates
noncomputable def r : ℝ := Real.sqrt (x^2 + y^2)
noncomputable def θ : ℝ := 2 * Real.pi - Real.arctan (|y / x|)

-- Theorem statement
theorem rectangular_to_polar :
  (r = 2 * Real.sqrt 2) ∧ 
  (θ = 7 * Real.pi / 4) ∧ 
  (r > 0) ∧ 
  (0 ≤ θ ∧ θ < 2 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_l529_52947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_equals_min_implies_x_is_one_l529_52950

noncomputable def M (a b c : ℝ) : ℝ := (a + b + c) / 3

noncomputable def min_three (a b c : ℝ) : ℝ := min a (min b c)

theorem average_equals_min_implies_x_is_one (x : ℝ) :
  M 2 (x + 1) (2 * x) = min_three 2 (x + 1) (2 * x) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_equals_min_implies_x_is_one_l529_52950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_white_truck_probability_l529_52977

theorem white_truck_probability (total white_trucks : ℕ) : 
  (total = 120 ∧ white_trucks = 15) → 
  (round ((white_trucks : ℚ) / total * 100) = 13) := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_white_truck_probability_l529_52977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_cos_α_gt_f_sin_β_l529_52991

-- Define an even function f that is increasing on [-1,0]
noncomputable def f (x : ℝ) : ℝ := sorry

-- Define the properties of f
axiom f_even : ∀ x, f (-x) = f x
axiom f_increasing_on_neg_unit_interval : ∀ x y, -1 ≤ x ∧ x < y ∧ y ≤ 0 → f x < f y

-- Define acute angles α and β
noncomputable def α : ℝ := sorry
noncomputable def β : ℝ := sorry

-- Axioms for α and β being acute angles of a triangle
axiom α_acute : 0 < α ∧ α < Real.pi / 2
axiom β_acute : 0 < β ∧ β < Real.pi / 2
axiom α_β_in_triangle : α + β < Real.pi

-- Theorem to prove
theorem f_cos_α_gt_f_sin_β : f (Real.cos α) > f (Real.sin β) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_cos_α_gt_f_sin_β_l529_52991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_archHeightAtTen_l529_52998

/-- Represents a parabolic arch -/
structure ParabolicArch where
  a : ℝ
  k : ℝ

/-- The height of the arch at a given x coordinate -/
def archHeight (arch : ParabolicArch) (x : ℝ) : ℝ :=
  arch.a * x^2 + arch.k

/-- Creates a parabolic arch given its maximum height and span -/
noncomputable def createArch (maxHeight : ℝ) (span : ℝ) : ParabolicArch :=
  { a := -8 / 75,
    k := maxHeight }

theorem archHeightAtTen (maxHeight span : ℝ) 
  (hMax : maxHeight = 24)
  (hSpan : span = 30) :
  archHeight (createArch maxHeight span) 10 = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_archHeightAtTen_l529_52998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_roots_quadratic_l529_52996

theorem complex_roots_quadratic (ω : ℂ) (hω : ω^8 = 1) (hω_ne_one : ω ≠ 1) : 
  let α : ℂ := ω + ω^3 + ω^5
  let β : ℂ := ω^2 + ω^4 + ω^6 + ω^7
  (α^2 + α + 5 = 0) ∧ (β^2 + β + 5 = 0) := by
  sorry

#check complex_roots_quadratic

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_roots_quadratic_l529_52996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_magnitude_l529_52920

/-- Two vectors a and b in ℝ² are collinear if their cross product is zero -/
def collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- The magnitude of a vector in ℝ² -/
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

/-- Given two vectors a and b in ℝ², if they are collinear,
    then the magnitude of 3a + b is √5 -/
theorem collinear_vectors_magnitude (k : ℝ) :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-2, k)
  collinear a b →
  magnitude (3 • a + b) = Real.sqrt 5 :=
by
  -- Introduce the local variables
  intro a b h
  -- Unfold the definitions
  simp [collinear, magnitude] at *
  -- The actual proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_magnitude_l529_52920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_divisibility_iff_power_of_three_l529_52901

theorem cubic_divisibility_iff_power_of_three (n : ℕ+) :
  (∀ k : ℤ, ∃ a : ℤ, (n : ℤ) ∣ (a^3 + a - k)) ↔ ∃ b : ℕ, n = 3^b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_divisibility_iff_power_of_three_l529_52901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l529_52957

-- Define the universal set U
def U : Set Int := {x | -1 ≤ x ∧ x ≤ 5}

-- Define set A
def A : Set Int := {x : Int | (x - 1) * (x - 2) = 0}

-- Define set B
def B : Set Int := {x : Int | x > 0 ∧ (4 - x) / 2 > 1}

-- Theorem statement
theorem set_operations :
  (U \ A = {-1, 0, 3, 4, 5}) ∧
  (A ∪ B = {1, 2}) ∧
  (A ∩ B = {1}) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l529_52957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_range_l529_52925

-- Define an acute triangle with angles in arithmetic sequence
structure AcuteTriangleArithAngles where
  A : Real
  B : Real
  C : Real
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sum_angles : A + B + C = π
  arithmetic_seq : 2 * B = A + C

-- Define the function y
noncomputable def y (t : AcuteTriangleArithAngles) : Real :=
  Real.sin t.A - Real.cos (t.A - t.C + 2 * t.B)

-- State the theorem
theorem y_range (t : AcuteTriangleArithAngles) :
  ∃ (a b : Real), a < b ∧ ∀ (x : Real), y t = x → a < x ∧ x < b :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_range_l529_52925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_zero_solution_l529_52956

theorem determinant_zero_solution (b : ℝ) (h : b ≠ 0) :
  ∃ y : ℝ, Matrix.det (
    !![y + b, y - b, y;
      y - b, y + b, y;
      y, y, y + b]
  ) = 0 ↔ 
    (y = (-3 + Real.sqrt 17) / 4 * b ∨ y = (-3 - Real.sqrt 17) / 4 * b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_zero_solution_l529_52956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_five_digit_with_three_is_ten_thousand_l529_52938

/-- A function that counts the number of five-digit positive integers with the ten-thousands digit 3 -/
def count_five_digit_with_three : ℕ :=
  Finset.card (Finset.filter (λ n => 30000 ≤ n ∧ n ≤ 39999) (Finset.range 100000))

/-- Theorem stating that the count of five-digit positive integers with the ten-thousands digit 3 is 10000 -/
theorem count_five_digit_with_three_is_ten_thousand :
  count_five_digit_with_three = 10000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_five_digit_with_three_is_ten_thousand_l529_52938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_nine_l529_52976

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_arithmetic (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

theorem arithmetic_sequence_sum_nine (a : ℕ → ℚ) :
  arithmetic_sequence a → (2 * a 8 = 6 + a 11) → sum_arithmetic a 9 = 54 := by
  sorry

#check arithmetic_sequence_sum_nine

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_nine_l529_52976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipeline_project_time_saved_l529_52962

/-- Represents the time saved in a pipeline construction project. -/
noncomputable def time_saved (m : ℝ) (x : ℝ) (n : ℝ) : ℝ :=
  m / x - m / ((1 + n / 100) * x)

/-- Theorem stating the condition for 8 days saved in the pipeline project. -/
theorem pipeline_project_time_saved
  (m : ℝ) (x : ℝ) (n : ℝ)
  (h_m_pos : m > 0)
  (h_x_pos : x > 0)
  (h_n_pos : n > 0) :
  time_saved m x n = 8 ↔ m / x - m / ((1 + n / 100) * x) = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipeline_project_time_saved_l529_52962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_four_valid_polynomials_l529_52955

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The set of roots of a quadratic polynomial -/
def roots (p : QuadraticPolynomial) : Set ℝ :=
  {r : ℝ | p.a * r^2 + p.b * r + p.c = 0}

/-- The set of coefficients of a quadratic polynomial -/
def coefficients (p : QuadraticPolynomial) : Set ℝ :=
  {p.a, p.b, p.c}

/-- Predicate for a valid polynomial according to the problem conditions -/
def is_valid_polynomial (p : QuadraticPolynomial) : Prop :=
  roots p = coefficients p

/-- The main theorem stating that there are exactly 4 valid polynomials -/
theorem exactly_four_valid_polynomials :
  ∃! (s : Finset QuadraticPolynomial), (∀ p ∈ s, is_valid_polynomial p) ∧ s.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_four_valid_polynomials_l529_52955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_additive_inverse_l529_52949

theorem complex_number_additive_inverse (b : ℝ) : 
  (Complex.re ((1 + b * Complex.I) / (1 + Complex.I)) = 
   -Complex.im ((1 + b * Complex.I) / (1 + Complex.I))) → b = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_additive_inverse_l529_52949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_approximation_l529_52905

/-- The product of the given fractions -/
def product : ℚ := 2/1 * 2/3 * 4/3 * 4/5 * 6/5 * 6/7 * 8/7

/-- The result rounded to two decimal places -/
def rounded_result : ℚ := 167/100

/-- A function to approximate rounding for rationals -/
def approx_round (q : ℚ) : ℚ :=
  (q * 100).floor / 100

theorem product_approximation : 
  approx_round product = rounded_result := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_approximation_l529_52905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmer_speed_is_seven_l529_52914

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeed where
  man : ℝ  -- Speed of the man in still water
  stream : ℝ  -- Speed of the stream

/-- Given downstream and upstream distances and times, calculates the speed of the man in still water. -/
noncomputable def calculate_swimmer_speed (downstream_distance upstream_distance : ℝ) (downstream_time upstream_time : ℝ) : ℝ :=
  let downstream_speed := downstream_distance / downstream_time
  let upstream_speed := upstream_distance / upstream_time
  (downstream_speed + upstream_speed) / 2

/-- Theorem: The speed of the man in still water is 7 km/h. -/
theorem swimmer_speed_is_seven :
  let swimmer_speed := calculate_swimmer_speed 45 25 5 5
  swimmer_speed = 7 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmer_speed_is_seven_l529_52914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l529_52903

theorem exponential_equation_solution (x : ℝ) : (5 : ℝ)^(x + 2) = 625 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l529_52903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_l529_52924

/-- The y-intercept of a line is the y-coordinate of the point where the line intersects the y-axis. -/
noncomputable def y_intercept (a b c : ℝ) : ℝ := -c / b

/-- The line equation is in the form ax + by + c = 0 -/
def line_equation (a b c : ℝ) (x y : ℝ) : Prop := a * x + b * y + c = 0

theorem y_intercept_of_line :
  let a : ℝ := 2
  let b : ℝ := -3
  let c : ℝ := 6
  y_intercept a b c = -2 ∧ 
  ∀ x y : ℝ, line_equation a b c x y ↔ 2 * x - 3 * y = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_l529_52924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_level_rise_l529_52997

/-- Calculates the rise in water level when a cube is immersed in a rectangular vessel -/
theorem water_level_rise (cube_edge vessel_length vessel_width : ℝ) :
  cube_edge = 16 →
  vessel_length = 20 →
  vessel_width = 15 →
  abs ((cube_edge ^ 3) / (vessel_length * vessel_width) - 13.65) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_level_rise_l529_52997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sign_sequence_for_2012_l529_52909

/-- The sequence of squares from 1² to 2012² -/
def square_sequence : List ℕ := (List.range 2012).map (λ n => (n + 1)^2)

/-- A type representing the choice of + or - -/
inductive Sign
| plus : Sign
| minus : Sign

/-- Apply a sign to a number -/
def apply_sign (s : Sign) (n : ℤ) : ℤ :=
  match s with
  | Sign.plus => n
  | Sign.minus => -n

theorem exists_sign_sequence_for_2012 :
  ∃ (signs : List Sign),
    signs.length = square_sequence.length - 1 ∧
    (List.foldl (λ acc (pair : Sign × ℕ) => acc + apply_sign pair.1 (Int.ofNat pair.2))
      0
      (List.zip signs square_sequence.tail) + Int.ofNat square_sequence.head!) = 2012 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sign_sequence_for_2012_l529_52909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l529_52986

noncomputable section

-- Define the function h
def h (x : ℝ) : ℝ := x + 1/x + 2

-- Define the symmetry condition
def symmetric_about (f g : ℝ → ℝ) (b : ℝ) : Prop :=
  ∀ x y, f x = y ↔ g (-x) = 2*b - y

-- Main theorem
theorem f_properties (f : ℝ → ℝ) 
  (h_sym : symmetric_about f h 1) :
  (∀ x, f x = x + 1/x) ∧ 
  (∀ x ∈ Set.Ioo 0 8, f x ≥ 2) ∧
  (∀ x ∈ Set.Ioo 0 8, f x ≤ 65/8) ∧
  (f 1 = 2) ∧
  (f 8 = 65/8) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l529_52986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_five_x_minus_two_y_l529_52937

theorem power_five_x_minus_two_y (x y : ℝ) 
  (hx : (5 : ℝ)^x = 36) (hy : (5 : ℝ)^y = 2) : (5 : ℝ)^(x - 2*y) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_five_x_minus_two_y_l529_52937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_rule_cubic_equivalence_horner_rule_cubic_operations_l529_52906

/-- Horner's Rule representation of a cubic polynomial -/
def horner_cubic (a b c d : ℝ) (x : ℝ) : ℝ :=
  x * (x * (a * x + b) + c) + d

/-- Original cubic polynomial -/
def original_cubic (a b c d : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + d

/-- Complexity is represented as a pair (multiplications, additions) -/
def complexity : ℕ × ℕ := (3, 3)

theorem horner_rule_cubic_equivalence :
  ∀ (x : ℝ), horner_cubic 7 3 (-5) 11 x = original_cubic 7 3 (-5) 11 x :=
by sorry

theorem horner_rule_cubic_operations :
  complexity = (3, 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_rule_cubic_equivalence_horner_rule_cubic_operations_l529_52906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l529_52972

noncomputable def f (x : ℝ) : ℝ := Real.sqrt ((1 - x) / (1 + x))

theorem problem_1 (α : ℝ) (h : α ∈ Set.Ioo (π/2) π) :
  f (Real.cos α) + f (-Real.cos α) = 2 / Real.sin α := by
  sorry

theorem problem_2 : Real.sin (50 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l529_52972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_sphere_ratio_l529_52964

/-- A regular tetrahedron is a tetrahedron with all edges of equal length -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

/-- The radius of the circumscribed sphere of a regular tetrahedron -/
noncomputable def circumscribed_radius (t : RegularTetrahedron) : ℝ :=
  (Real.sqrt 30 / 8) * t.edge_length

/-- The radius of the inscribed sphere of a regular tetrahedron -/
noncomputable def inscribed_radius (t : RegularTetrahedron) : ℝ :=
  (3 * Real.sqrt 3 / 8) * t.edge_length

/-- The theorem stating that the ratio of the circumscribed sphere radius to the inscribed sphere radius
    of a regular tetrahedron is √10/3 -/
theorem regular_tetrahedron_sphere_ratio (t : RegularTetrahedron) :
  circumscribed_radius t / inscribed_radius t = Real.sqrt 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_sphere_ratio_l529_52964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_fixed_point_of_h_l529_52951

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := (5 * (x - 2) / 3) - 3

-- State the theorem
theorem unique_fixed_point_of_h : 
  (∃! x : ℝ, h x = x) ∧ (∀ x : ℝ, h x = x → x = 19 / 2) := by
  sorry

#check unique_fixed_point_of_h

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_fixed_point_of_h_l529_52951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trisha_price_equals_art_price_l529_52904

/-- Represents a cookie shape -/
inductive CookieShape
  | Trapezoid
  | Circle

/-- Represents a baker -/
structure Baker where
  name : String
  shape : CookieShape
  count : ℕ
  price : ℕ

/-- The amount of dough used by a baker -/
noncomputable def doughUsed (b : Baker) : ℝ :=
  match b.shape with
  | CookieShape.Trapezoid => 12 * b.count
  | CookieShape.Circle => 4 * Real.pi * b.count

/-- The total earnings of a baker -/
def totalEarnings (b : Baker) : ℕ :=
  b.count * b.price

/-- Theorem stating that if Art and Trisha use the same amount of dough and make the same number of cookies,
    Trisha should price her cookies the same as Art to earn the same amount -/
theorem trisha_price_equals_art_price (art trisha : Baker)
  (h1 : art.name = "Art")
  (h2 : trisha.name = "Trisha")
  (h3 : art.shape = CookieShape.Trapezoid)
  (h4 : trisha.shape = CookieShape.Circle)
  (h5 : art.count = trisha.count)
  (h6 : art.price = 60)
  (h7 : doughUsed art = doughUsed trisha) :
  trisha.price = art.price :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trisha_price_equals_art_price_l529_52904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_symmetry_l529_52953

/-- The sine function is symmetric about the line x = π/2 -/
theorem sine_symmetry : ∀ x : ℝ, Real.sin (π/2 + x) = Real.sin (π/2 - x) := by
  intro x
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_symmetry_l529_52953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_non_product_l529_52943

def is_product_of_two_integers_greater_than_one (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n = a * b

theorem unique_non_product (numbers : List ℕ := [6, 27, 53, 39, 77]) :
  ∃! x, x ∈ numbers ∧ ¬(is_product_of_two_integers_greater_than_one x) ∧ x = 53 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_non_product_l529_52943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_length_in_cube_l529_52961

/-- The length of a line segment within a cube of edge length 4 -/
theorem line_segment_length_in_cube : 
  let start : Fin 3 → ℝ := ![0, 0, 4]
  let end_ : Fin 3 → ℝ := ![4, 4, 8]
  let segment_length := Real.sqrt ((end_ 0 - start 0)^2 + (end_ 1 - start 1)^2 + (end_ 2 - start 2)^2)
  segment_length = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_length_in_cube_l529_52961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l529_52932

-- Define the line L
def Line (A B C : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ -A * x - B * y + C = 0

-- Define what it means for a line to pass through the origin
def PassesThroughOrigin (A B C : ℝ) : Prop :=
  Line A B C 0 0

-- Define what it means for a line to lie in the first and third quadrants
def LiesInFirstAndThirdQuadrants (A B C : ℝ) : Prop :=
  ∀ x y, Line A B C x y → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)

-- The theorem to prove
theorem line_properties (A B C : ℝ) :
  PassesThroughOrigin A B C →
  LiesInFirstAndThirdQuadrants A B C →
  A * B < 0 ∧ C = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l529_52932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C3_C4_congruent_l529_52917

noncomputable section

-- Define the setup
def AB : ℝ := 1
def P (x : ℝ) : ℝ := x
def AP (x : ℝ) : ℝ := x
def PB (x : ℝ) : ℝ := 1 - x

-- Define the circles
def C1_radius (x : ℝ) : ℝ := AP x / 2
def C2_radius (x : ℝ) : ℝ := PB x / 2

-- Define the radii of C3 and C4
def C3_radius (x : ℝ) : ℝ := x * (1 - x) / 2
def C4_radius (x : ℝ) : ℝ := x * (1 - x) / 2

-- Theorem statement
theorem C3_C4_congruent (x : ℝ) (h : 0 < x ∧ x < 1) : 
  C3_radius x = C4_radius x := by
  -- Unfold the definitions
  unfold C3_radius C4_radius
  -- The expressions are identical, so reflexivity completes the proof
  rfl

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C3_C4_congruent_l529_52917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_equal_area_division_l529_52907

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define point K on side AB
variable (K : EuclideanSpace ℝ (Fin 2))

-- Define that K is on AB
axiom K_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ K = (1 - t) • A + t • B

-- Define the area of a triangle
noncomputable def triangle_area (P Q R : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- State the theorem
theorem exists_equal_area_division :
  ∃ (M : EuclideanSpace ℝ (Fin 2)), 
    (∃ u : ℝ, 0 ≤ u ∧ u ≤ 1 ∧ M = (1 - u) • C + u • B) ∨
    (∃ v : ℝ, 0 ≤ v ∧ v ≤ 1 ∧ M = (1 - v) • C + v • A) ∧
    triangle_area A K M = triangle_area K B C :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_equal_area_division_l529_52907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_pi_sufficient_not_necessary_l529_52902

noncomputable def curve (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

def passes_through_origin (φ : ℝ) : Prop :=
  ∃ x : ℝ, curve x φ = 0 ∧ x = 0

theorem phi_pi_sufficient_not_necessary :
  (∀ φ : ℝ, φ = π → passes_through_origin φ) ∧
  ¬(∀ φ : ℝ, passes_through_origin φ → φ = π) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_pi_sufficient_not_necessary_l529_52902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l529_52958

/-- Custom operation ⊗ for positive real numbers -/
noncomputable def otimes (a b : ℝ) : ℝ := a + b - Real.sqrt (a * b)

/-- Function f(x) defined in the problem -/
noncomputable def f (k x : ℝ) : ℝ := (otimes k x) / Real.sqrt x

/-- Theorem stating the minimum value of f(x) -/
theorem min_value_of_f (k : ℝ) (h : otimes 4 k = 3) :
  ∃ (min : ℝ), min = 1 ∧ ∀ (x : ℝ), x > 0 → f k x ≥ min := by
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l529_52958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l529_52999

/-- Given an equation y = x / (x^3 + Ax^2 + Bx + C) with vertical asymptotes at x = -1, 3, 4,
    where A, B, C are integers, prove that A + B + C = 11 -/
theorem asymptote_sum (A B C : ℤ) : 
  (∀ x : ℝ, x = -1 ∨ x = 3 ∨ x = 4 → x^3 + A*x^2 + B*x + C = 0) →
  A + B + C = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l529_52999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_condition_l529_52930

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h1 : d ≠ 0
  h2 : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

theorem arithmetic_geometric_condition (seq : ArithmeticSequence) 
  (h3 : (seq.a 3) * (seq.a 8) = (seq.a 4)^2) : 
  (seq.a 1 * seq.d < 0) ∧ (seq.d * sum_n seq 4 < 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_condition_l529_52930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vector_sum_on_circle_l529_52912

/-- The circle (C) in the Cartesian plane -/
def C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 6*p.1 + 5 = 0}

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The vector sum of OA and OB -/
def vectorSum (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 + b.1, a.2 + b.2)

/-- The magnitude of a vector in ℝ² -/
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

theorem max_vector_sum_on_circle :
  ∀ (a b : ℝ × ℝ),
    a ∈ C → b ∈ C →
    distance a b = 2 * Real.sqrt 3 →
    magnitude (vectorSum a b) ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vector_sum_on_circle_l529_52912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_workers_payment_l529_52983

/-- Represents the sum of money available to pay workers. -/
def S : ℝ := sorry

/-- Represents the daily wage of worker x. -/
noncomputable def wage_x : ℝ := S / 36

/-- Represents the daily wage of worker y. -/
noncomputable def wage_y : ℝ := S / 45

/-- Represents the daily wage of worker z. -/
noncomputable def wage_z : ℝ := S / 60

/-- Represents the combined daily wage of all three workers. -/
noncomputable def combined_wage : ℝ := wage_x + wage_y + wage_z

/-- 
Theorem stating that if S can pay for worker x for 36 days, 
worker y for 45 days, and worker z for 60 days, then S can pay 
for all three workers together for 15 days.
-/
theorem workers_payment (hS : S > 0) : 
  S / combined_wage = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_workers_payment_l529_52983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_angles_l529_52948

theorem tan_sum_angles (α β : Real) : 
  (0 < α ∧ α < Real.pi / 2) →
  (0 < β ∧ β < Real.pi / 2) →
  Real.tan α = 1 / 7 →
  Real.sin β = Real.sqrt 10 / 10 →
  Real.tan (α + 2 * β) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_angles_l529_52948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_prism_volume_l529_52927

/-- A right prism with a right triangle base -/
structure RightPrism where
  α : Real  -- acute angle of the base triangle
  l : Real  -- length of the lateral edge
  β : Real  -- angle between lateral edge and diagonal of larger lateral face
  h_α_acute : 0 < α ∧ α < Real.pi/2
  h_l_pos : l > 0
  h_β_acute : 0 < β ∧ β < Real.pi/2

/-- The volume of a right prism -/
noncomputable def volume (p : RightPrism) : Real := 
  (1/4) * p.l^3 * (Real.tan p.β)^2 * Real.sin (2*p.α)

/-- Theorem stating the volume of a right prism -/
theorem right_prism_volume (p : RightPrism) : 
  volume p = (1/4) * p.l^3 * (Real.tan p.β)^2 * Real.sin (2*p.α) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_prism_volume_l529_52927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_of_tetrahedron_l529_52919

/-- A tetrahedron with vertex A and edges AB, AC, and AD. -/
structure Tetrahedron where
  AB : ℝ
  AC : ℝ
  AD : ℝ
  BC : ℝ
  CD : ℝ
  BD : ℝ

/-- The condition that the angles at vertex A are all 90 degrees. -/
def right_angles_at_A (t : Tetrahedron) : Prop :=
  t.AB ^ 2 + t.AC ^ 2 = t.BC ^ 2 ∧
  t.AB ^ 2 + t.AD ^ 2 = t.BD ^ 2 ∧
  t.AC ^ 2 + t.AD ^ 2 = t.CD ^ 2

/-- The total length of all edges is 1. -/
def total_edge_length_is_one (t : Tetrahedron) : Prop :=
  t.AB + t.AC + t.AD + t.BC + t.CD + t.BD = 1

/-- The volume of the tetrahedron. -/
noncomputable def volume (t : Tetrahedron) : ℝ :=
  (t.AB * t.AC * t.AD) / 6

/-- The theorem stating the maximum volume of the tetrahedron. -/
theorem max_volume_of_tetrahedron (t : Tetrahedron) 
  (h1 : right_angles_at_A t) (h2 : total_edge_length_is_one t) :
  volume t ≤ (5 * Real.sqrt 2 - 7) / 162 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_of_tetrahedron_l529_52919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l529_52952

theorem expression_value : ((((3 : ℚ) + 2)⁻¹ + 2)⁻¹ + 2)⁻¹ + 2 = 65 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l529_52952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_combinations_l529_52911

theorem coin_combinations : ∃! n : ℕ, 
  (∀ x y z : ℕ, x + y + z = 100 ∧ x + 2*y + 5*z = 300 → 
    y ∈ Finset.range (n + 1)) ∧ 
  (∃ x y z : ℕ, x + y + z = 100 ∧ x + 2*y + 5*z = 300 ∧ y = n) ∧
  n = 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_combinations_l529_52911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_spherical_coordinates_standard_coordinate_valid_l529_52928

/-- Given a point in spherical coordinates (ρ, θ, φ), this function returns the equivalent
    point within the standard range constraints. -/
noncomputable def standardSphericalCoordinate (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ, θ % (2 * Real.pi), φ % (2 * Real.pi))

/-- Theorem stating that (4, 3π/4, 7π/4) is equivalent to (4, 3π/4, π/4) in standard spherical coordinates. -/
theorem equivalent_spherical_coordinates :
  let (ρ, θ, φ) := standardSphericalCoordinate 4 (3 * Real.pi / 4) (7 * Real.pi / 4)
  ρ = 4 ∧ θ = 3 * Real.pi / 4 ∧ φ = Real.pi / 4 :=
by
  sorry

/-- Constraints on standard spherical coordinates -/
def validSphericalCoordinate (ρ θ φ : ℝ) : Prop :=
  ρ > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 0 ≤ φ ∧ φ ≤ Real.pi

/-- Theorem stating that the result of standardSphericalCoordinate satisfies the constraints -/
theorem standard_coordinate_valid (ρ θ φ : ℝ) (h : ρ > 0) :
  let (ρ', θ', φ') := standardSphericalCoordinate ρ θ φ
  validSphericalCoordinate ρ' θ' φ' :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_spherical_coordinates_standard_coordinate_valid_l529_52928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_unique_solutions_l529_52987

noncomputable def f (x : ℝ) : ℝ := 4 * x^2 / (1 + 4 * x^2)

theorem system_solutions :
  ∀ x y z : ℝ,
  (f x = y ∧ f y = z ∧ f z = x) →
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2)) :=
by sorry

theorem unique_solutions :
  ∃! x y z : ℝ,
  (f x = y ∧ f y = z ∧ f z = x) ∧
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_unique_solutions_l529_52987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_conversion_theorem_l529_52941

theorem base_conversion_theorem :
  let possible_bases := {b : ℕ | b ≥ 2 ∧ b^3 ≤ 256 ∧ 256 < b^4}
  Finset.card (Finset.filter (λ b => b^3 ≤ 256 ∧ 256 < b^4) (Finset.range 7)) = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_conversion_theorem_l529_52941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_negative_five_l529_52979

theorem reciprocal_of_negative_five :
  (λ x : ℚ => 1 / x) (-5) = -1 / 5 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_negative_five_l529_52979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_g_unique_zero_point_l529_52923

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x

-- Define the function g
noncomputable def g (t : ℝ) (x : ℝ) : ℝ := Real.log x + x - (2/t) * x^2

-- Theorem for part 1
theorem f_max_value :
  ∃ (x : ℝ), x ∈ Set.Icc (Real.exp 1) (Real.exp 2) ∧
  ∀ (y : ℝ), y ∈ Set.Icc (Real.exp 1) (Real.exp 2) →
  f (-1/3) y ≤ f (-1/3) x ∧
  f (-1/3) x = Real.log 3 - 1 :=
by sorry

-- Theorem for part 2
theorem g_unique_zero_point :
  (∃! (x : ℝ), g 2 x = 0) ∧
  (∀ (t : ℝ), t > 0 → (∃! (x : ℝ), g t x = 0) → t = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_g_unique_zero_point_l529_52923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_line_equation_l529_52971

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

-- Define the midpoint of the chord
def chord_midpoint : ℝ × ℝ := (2, -1)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - 2*y - 4 = 0

-- Theorem statement
theorem chord_line_equation :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  is_on_ellipse x₁ y₁ ∧ is_on_ellipse x₂ y₂ ∧ 
  ((x₁ + x₂)/2, (y₁ + y₂)/2) = chord_midpoint →
  line_equation x₁ y₁ ∧ line_equation x₂ y₂ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_line_equation_l529_52971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_propositions_l529_52968

-- Proposition 1
def proposition1 : Prop :=
  (∃ x : ℝ, x^2 + x - 1 < 0) ↔ ¬(∀ x : ℝ, x^2 + x - 1 > 0)

-- Proposition 2
def proposition2 (p q : Prop) : Prop :=
  ((q → p) ∧ ¬(p → q)) → ((¬p → ¬q) ∧ ¬(¬q → ¬p))

-- Proposition 3
def proposition3 : Prop :=
  ∀ x y : ℝ, Real.sin x ≠ Real.sin y → x ≠ y

-- Proposition 4
noncomputable def proposition4 : Prop :=
  ∀ x y : ℝ, x > 0 ∧ y > 0 → (Real.log x > Real.log y ↔ x > y)

theorem correct_propositions :
  ¬proposition1 ∧ proposition2 (True) (True) ∧ proposition3 ∧ ¬proposition4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_propositions_l529_52968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_cost_is_265000_l529_52960

/-- The cost of raising a child for John --/
def child_raising_cost (n : ℕ) : ℚ :=
  if n ≤ 8 then 10000 else 20000

/-- The total cost of raising a child until 18 years old --/
def total_raising_cost : ℚ :=
  (List.range 18).map child_raising_cost |>.sum

/-- The cost of university tuition --/
def university_tuition : ℚ := 250000

/-- The total cost including raising and university tuition --/
def total_cost : ℚ := total_raising_cost + university_tuition

/-- John's share of the total cost --/
def john_cost : ℚ := total_cost / 2

theorem john_cost_is_265000 : john_cost = 265000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_cost_is_265000_l529_52960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chameleon_distance_theorem_l529_52945

/-- A chameleon is a sequence of 3n letters, with exactly n occurrences of each of the letters a, b, and c -/
def Chameleon (n : ℕ) := { s : List Char // s.length = 3*n ∧ s.count 'a' = n ∧ s.count 'b' = n ∧ s.count 'c' = n }

/-- A swap is the transposition of two adjacent letters in a chameleon -/
def swap (s : List Char) (i : Fin s.length) : List Char :=
  if h : i.val + 1 < s.length then
    let j : Fin s.length := ⟨i.val + 1, h⟩
    s.set i (s.get j) |>.set j (s.get i)
  else
    s

/-- The distance between two chameleons is the minimum number of swaps required to transform one into the other -/
noncomputable def distance (X Y : Chameleon n) : ℕ := sorry

theorem chameleon_distance_theorem (n : ℕ) (hn : n > 0) (X : Chameleon n) :
  ∃ Y : Chameleon n, distance X Y ≥ (3 * n^2) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chameleon_distance_theorem_l529_52945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_diagonal_division_l529_52934

/-- A regular hexagon with side length 1 -/
structure RegularHexagon :=
  (A B C D E F : ℝ × ℝ)
  (is_regular : sorry)
  (side_length : sorry)

/-- A point that divides a line segment internally -/
def internal_division_point (A B : ℝ × ℝ) (r : ℝ) : ℝ × ℝ := 
  ((1 - r) * A.1 + r * B.1, (1 - r) * A.2 + r * B.2)

/-- Check if three points are collinear -/
def collinear (P Q R : ℝ × ℝ) : Prop :=
  (Q.2 - P.2) * (R.1 - Q.1) = (R.2 - Q.2) * (Q.1 - P.1)

theorem hexagon_diagonal_division (ABCDEF : RegularHexagon) :
  let AC := (ABCDEF.C.1 - ABCDEF.A.1, ABCDEF.C.2 - ABCDEF.A.2)
  let CE := (ABCDEF.E.1 - ABCDEF.C.1, ABCDEF.E.2 - ABCDEF.C.2)
  let M := internal_division_point ABCDEF.A ABCDEF.C (1 / Real.sqrt 3)
  let N := internal_division_point ABCDEF.C ABCDEF.E (1 / Real.sqrt 3)
  collinear ABCDEF.B M N := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_diagonal_division_l529_52934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_DE_is_35_l529_52988

/-- Triangle ABC with given side lengths -/
structure Triangle :=
  (A B C : ℝ × ℝ)
  (AB_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 15)
  (BC_length : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 18)
  (AC_length : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = 21)

/-- Point P on AC such that PC = 15 -/
noncomputable def P (t : Triangle) : ℝ × ℝ :=
  sorry

/-- Line BP -/
noncomputable def lineBP (t : Triangle) : ℝ → ℝ :=
  sorry

/-- Point D on line BP such that AD is parallel to BC -/
noncomputable def D (t : Triangle) : ℝ × ℝ :=
  sorry

/-- Point E on line BP such that AB is parallel to CE -/
noncomputable def E (t : Triangle) : ℝ × ℝ :=
  sorry

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_DE_is_35 (t : Triangle) : 
  distance (D t) (E t) = 35 := by
  sorry

#check distance_DE_is_35

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_DE_is_35_l529_52988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_wins_n_5_l529_52939

/-- Represents the state of a seat in the game -/
inductive SeatState
| Empty : SeatState
| Occupied : SeatState
deriving Repr, DecidableEq

/-- Represents a player in the game -/
inductive Player
| Alice : Player
| Bob : Player
deriving Repr, DecidableEq

/-- Represents the game state -/
structure GameState where
  seats : List SeatState
  currentPlayer : Player
deriving Repr

/-- Checks if a move is valid -/
def isValidMove (game : GameState) (position : Nat) : Prop :=
  position > 0 ∧ position ≤ game.seats.length ∧
  game.seats.get? (position - 1) = some SeatState.Empty ∧
  (position = 1 ∨ game.seats.get? (position - 2) = some SeatState.Empty) ∧
  (position = game.seats.length ∨ game.seats.get? position = some SeatState.Empty)

/-- Applies a move to the game state -/
def applyMove (game : GameState) (position : Nat) : GameState :=
  { seats := game.seats.set (position - 1) SeatState.Occupied,
    currentPlayer := if game.currentPlayer = Player.Alice then Player.Bob else Player.Alice }

/-- Checks if the game is over (no valid moves left) -/
def isGameOver (game : GameState) : Prop :=
  ∀ position, ¬(isValidMove game position)

/-- Defines a winning strategy for Alice -/
def aliceWinningStrategy (game : GameState) : Prop :=
  game.currentPlayer = Player.Alice →
  ∃ (move : Nat), isValidMove game move ∧
    ∀ (bobMove : Nat), isValidMove (applyMove game move) bobMove →
      isGameOver (applyMove (applyMove game move) bobMove)

/-- The main theorem: Alice has a winning strategy for n = 5 -/
theorem alice_wins_n_5 :
  let initialGame : GameState := {
    seats := [SeatState.Empty, SeatState.Empty, SeatState.Empty, SeatState.Empty, SeatState.Empty],
    currentPlayer := Player.Alice
  }
  aliceWinningStrategy initialGame := by
  sorry

#check alice_wins_n_5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_wins_n_5_l529_52939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fastest_slowest_difference_l529_52993

/-- Represents the five sheep -/
inductive Sheep
  | A | B | C | D | E
  deriving BEq, Repr

/-- The order of sheep at 7:00 -/
def order_7am : List Sheep := [Sheep.A, Sheep.B, Sheep.C, Sheep.D, Sheep.E]

/-- The order of sheep at 8:00 -/
def order_8am : List Sheep := [Sheep.B, Sheep.E, Sheep.C, Sheep.A, Sheep.D]

/-- Common difference at 7:00 -/
def diff_7am : ℕ := 20

/-- Common difference at 8:00 -/
def diff_8am : ℕ := 30

/-- Distance function at 7:00 -/
def dist_7am (s : Sheep) : ℕ → ℕ :=
  fun L => L + (order_7am.indexOf s * diff_7am)

/-- Distance function at 8:00 -/
def dist_8am (s : Sheep) : ℕ → ℕ :=
  fun L => L + (order_8am.indexOf s * diff_8am)

/-- Distance covered by a sheep in one hour -/
def distance_covered (s : Sheep) : ℕ → ℤ :=
  fun L => (dist_8am s L) - (dist_7am s L)

theorem fastest_slowest_difference :
  ∃ (L : ℕ), 
    (List.maximum? (List.map (fun s => distance_covered s L) order_7am)).isSome ∧
    (List.minimum? (List.map (fun s => distance_covered s L) order_7am)).isSome ∧
    ((List.maximum? (List.map (fun s => distance_covered s L) order_7am)).get! -
     (List.minimum? (List.map (fun s => distance_covered s L) order_7am)).get!) = 140 := by
  sorry

#eval List.map (fun s => distance_covered s 0) order_7am

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fastest_slowest_difference_l529_52993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_b_value_l529_52966

/-- Represents an ellipse with semi-major axis 2 and semi-minor axis b -/
structure Ellipse (b : ℝ) :=
  (eq : ∀ x y : ℝ, x^2/4 + y^2/b^2 = 1)
  (b_pos : 0 < b)
  (b_lt_two : b < 2)

/-- Represents a point on the ellipse -/
structure PointOnEllipse (b : ℝ) :=
  (x : ℝ)
  (y : ℝ)
  (on_ellipse : x^2/4 + y^2/b^2 = 1)

/-- The right focus of the ellipse -/
noncomputable def rightFocus (b : ℝ) : ℝ × ℝ := (Real.sqrt (4 - b^2), 0)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The maximum sum of distances from two points on the ellipse to the right focus -/
def maxSumDistances : ℝ := 5

theorem ellipse_b_value (b : ℝ) (e : Ellipse b) :
  maxSumDistances = 5 → b = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_b_value_l529_52966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_failure_system_properties_l529_52946

/-- A system of n elements where each element has an independent probability p of failing
    and causes all elements below it to fail. -/
structure FailureSystem where
  n : ℕ
  p : ℝ
  h_p_pos : 0 < p
  h_p_lt_one : p < 1

/-- The probability that exactly k elements fail in the system. -/
noncomputable def prob_k_fail (sys : FailureSystem) (k : ℕ) : ℝ :=
  sys.p * (1 - sys.p) ^ (sys.n - k)

/-- The expected number of failed elements in the system. -/
noncomputable def expected_failures (sys : FailureSystem) : ℝ :=
  sys.n + 1 - 1 / sys.p + (1 - sys.p) ^ (sys.n + 1) / sys.p

theorem failure_system_properties (sys : FailureSystem) :
  (∀ k, k ≤ sys.n → prob_k_fail sys k = sys.p * (1 - sys.p) ^ (sys.n - k)) ∧
  expected_failures sys = sys.n + 1 - 1 / sys.p + (1 - sys.p) ^ (sys.n + 1) / sys.p := by
  sorry

#check failure_system_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_failure_system_properties_l529_52946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_group_probability_l529_52921

/-- The probability of two people being in the same group when drawing from 20 cards -/
theorem same_group_probability : 
  (7 : ℚ) / 51 = 
  let total_cards : ℕ := 20
  let drawn_cards : Finset ℕ := {5, 14}
  let remaining_cards : Finset ℕ := Finset.range total_cards \ drawn_cards
  let favorable_outcomes : ℕ := (Finset.filter (λ x => x > 14) remaining_cards).card.choose 2 +
                                (Finset.filter (λ x => x < 5) remaining_cards).card.choose 2
  let total_outcomes : ℕ := remaining_cards.card.choose 2
  (favorable_outcomes : ℚ) / total_outcomes := by
  sorry

#check same_group_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_group_probability_l529_52921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_case_l529_52982

/-- The radius of a circle inscribed within three mutually externally tangent circles -/
noncomputable def inscribed_circle_radius (a b c : ℝ) : ℝ :=
  1 / (1/a + 1/b + 1/c + 2 * Real.sqrt (1/(a*b) + 1/(a*c) + 1/(b*c)))

/-- The theorem stating that for given values of a, b, and c, the inscribed circle radius is 18/17 -/
theorem inscribed_circle_radius_specific_case :
  inscribed_circle_radius 3 6 9 = 18/17 := by
  -- Expand the definition of inscribed_circle_radius
  unfold inscribed_circle_radius
  -- Simplify the expression
  simp [Real.sqrt_eq_rpow]
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_case_l529_52982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_points_exist_and_unique_l529_52936

/-- The set C₁ in the plane -/
def C₁ : Set (ℝ × ℝ) := {p | p.1 > 0 ∧ p.2 = 1 / p.1}

/-- The set C₂ in the plane -/
def C₂ : Set (ℝ × ℝ) := {p | p.1 < 0 ∧ p.2 = -1 + 1 / p.1}

/-- Distance function between two points in the plane -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem stating the existence and uniqueness of minimum distance points -/
theorem min_distance_points_exist_and_unique :
  ∃! (p₀ : ℝ × ℝ) (q₀ : ℝ × ℝ),
    p₀ ∈ C₁ ∧ q₀ ∈ C₂ ∧
    (∀ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ C₁ → q ∈ C₂ → distance p₀ q₀ ≤ distance p q) ∧
    p₀.1 = -q₀.1 ∧ p₀.1 > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_points_exist_and_unique_l529_52936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l529_52963

/-- The area of a triangle given its vertices -/
noncomputable def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

/-- Theorem: The area of a triangle with vertices at (0,0), (0,6), and (8,14) is 24 square units -/
theorem triangle_area_specific : triangle_area 0 0 0 6 8 14 = 24 := by
  -- Unfold the definition of triangle_area
  unfold triangle_area
  -- Simplify the expression
  simp
  -- Evaluate the absolute value
  rw [abs_of_nonneg]
  -- Prove that the result is nonnegative
  · norm_num
  -- Complete the calculation
  · norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l529_52963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcf_and_ratio_of_numbers_l529_52990

theorem gcf_and_ratio_of_numbers : 
  ∃ (gcf : ℕ) (ratio : ℚ), 
    (Nat.gcd 4536 14280 = gcf) ∧ 
    (gcf = 504) ∧ 
    (ratio = 4536 / 14280) ∧ 
    (abs (ratio - 1/3) < 1/100) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcf_and_ratio_of_numbers_l529_52990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_and_roots_l529_52916

noncomputable def f (a b x : ℝ) : ℝ := (1/3) * x^3 + a * x^2 + 3 * x + b

theorem max_value_and_roots (a b : ℝ) :
  (∀ x, f a b x ≤ 2) ∧ (f a b 1 = 2) →
  (a = -2 ∧ b = 2/3) ∧
  (∀ k, (∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    f (-2) (2/3) x = -x^2 + 6*x + k ∧
    f (-2) (2/3) y = -y^2 + 6*y + k ∧
    f (-2) (2/3) z = -z^2 + 6*z + k) ↔
    -25/3 < k ∧ k < 7/3) :=
by sorry

#check max_value_and_roots

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_and_roots_l529_52916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_c_l529_52931

theorem unique_solution_c : ∃! c : ℝ, 
  (2 * (Int.floor c)^2 + 14 * (Int.floor c) - 48 = 0) ∧ 
  (9 * (c - Int.floor c)^2 - 25 * (c - Int.floor c) + 6 = 0) ∧ 
  (0 ≤ c - Int.floor c) ∧ (c - Int.floor c < 1) ∧
  (c = 20 / 9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_c_l529_52931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_market_costs_for_100_uniforms_cost_effectiveness_comparison_l529_52944

noncomputable section

-- Define the prices
def football_price : ℝ := 90
def uniform_price : ℝ := 150

-- Define the condition that uniforms cost 60 more than footballs
axiom uniform_football_price_diff : uniform_price = football_price + 60

-- Define the condition that 3 uniforms cost the same as 5 footballs
axiom uniform_football_cost_relation : 3 * uniform_price = 5 * football_price

-- Define Market A's offer
def market_a_cost (uniforms : ℕ) (footballs : ℕ) : ℝ :=
  uniform_price * (uniforms : ℝ) + football_price * ((footballs : ℝ) - (uniforms / 10 : ℝ))

-- Define Market B's offer
def market_b_cost (uniforms : ℕ) (footballs : ℕ) : ℝ :=
  uniform_price * (uniforms : ℝ) + 
  if uniforms > 60 then 0.8 * football_price * (footballs : ℝ) else football_price * (footballs : ℝ)

-- Theorem for the specific case of 100 uniforms and y footballs
theorem market_costs_for_100_uniforms (y : ℕ) (h : y > 10) :
  market_a_cost 100 y = 90 * (y : ℝ) + 14100 ∧
  market_b_cost 100 y = 72 * (y : ℝ) + 15000 :=
by sorry

-- Theorem for cost-effectiveness comparison
theorem cost_effectiveness_comparison (y : ℕ) (h : y > 10) :
  (y < 50 → market_a_cost 100 y < market_b_cost 100 y) ∧
  (y > 50 → market_a_cost 100 y > market_b_cost 100 y) ∧
  (y = 50 → market_a_cost 100 y = market_b_cost 100 y) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_market_costs_for_100_uniforms_cost_effectiveness_comparison_l529_52944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_expression_l529_52973

theorem cubic_root_expression (x₁ x₂ x₃ : ℝ) : 
  x₁^3 - x₁^2 - 5*x₁ - 1 = 0 →
  x₂^3 - x₂^2 - 5*x₂ - 1 = 0 →
  x₃^3 - x₃^2 - 5*x₃ - 1 = 0 →
  x₁ ≠ x₂ →
  x₂ ≠ x₃ →
  x₃ ≠ x₁ →
  (x₁^2 - 4*x₁*x₂ + x₂^2) * (x₂^2 - 4*x₂*x₃ + x₃^2) * (x₃^2 - 4*x₃*x₁ + x₁^2) = 444 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_expression_l529_52973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_diameter_segment_l529_52995

theorem circle_diameter_segment (BC BD DA x : ℝ) : 
  BC = Real.sqrt 901 →
  BD = 1 →
  DA = 16 →
  (∃ (E : ℝ × ℝ), 
    let C : ℝ × ℝ := (Real.sqrt 901, 0)
    let B : ℝ × ℝ := (0, 0)
    let D : ℝ × ℝ := (1, Real.sqrt (BC^2 - 1))
    let A : ℝ × ℝ := (1, Real.sqrt (BC^2 - 1) + 16)
    E.1^2 + E.2^2 = (BC/2)^2 ∧
    (E.1 - C.1)^2 + E.2^2 = x^2 ∧
    (A.1 - E.1)^2 + (A.2 - E.2)^2 = (BC - x)^2) →
  x = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_diameter_segment_l529_52995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_one_zero_l529_52994

open Real

/-- A monotonic function f on (0, +∞) satisfying f[f(x) - log x] = e + 1 -/
noncomputable def f : ℝ → ℝ := sorry

/-- f is monotonic on (0, +∞) -/
axiom f_monotonic : Monotone f

/-- Domain of f is (0, +∞) -/
axiom f_domain (x : ℝ) : 0 < x → ∃ y, f x = y

/-- f satisfies the given functional equation -/
axiom f_equation (x : ℝ) (h : 0 < x) : f (f x - log x) = Real.exp 1 + 1

/-- g is defined as f(x) - f'(x) -/
noncomputable def g (x : ℝ) : ℝ := f x - deriv f x

/-- Theorem: g has exactly one zero -/
theorem g_has_one_zero : ∃! x, g x = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_one_zero_l529_52994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_stars_necessary_and_sufficient_l529_52933

/-- Represents a 4x4 grid with stars -/
def Grid := Fin 4 → Fin 4 → Bool

/-- Counts the number of stars in a grid -/
def count_stars (g : Grid) : ℕ :=
  Finset.sum (Finset.univ : Finset (Fin 4)) (λ i =>
    Finset.sum (Finset.univ : Finset (Fin 4)) (λ j =>
      if g i j then 1 else 0))

/-- Checks if a 2x2 subgrid contains at least one star -/
def subgrid_has_star (g : Grid) (r c : Fin 2) : Prop :=
  ∃ (i j : Fin 2), g (r.val + i.val) (c.val + j.val)

/-- The main theorem: 7 stars are necessary and sufficient -/
theorem seven_stars_necessary_and_sufficient :
  (∀ g : Grid, count_stars g ≥ 7 → ∀ (r c : Fin 2), subgrid_has_star g r c) ∧
  (∀ n : ℕ, n < 7 → ∃ g : Grid, count_stars g = n ∧ ∃ (r c : Fin 2), ¬subgrid_has_star g r c) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_stars_necessary_and_sufficient_l529_52933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stable_performance_comparison_l529_52959

/-- Represents a student's long jump performance -/
structure StudentPerformance where
  variance : ℝ

/-- Determines if the first student's performance is more stable than the second -/
def moreStable (a b : StudentPerformance) : Prop :=
  a.variance < b.variance

/-- Theorem: Given the variances of two students' long jump results, 
    the student with the smaller variance has more stable performance -/
theorem stable_performance_comparison 
  (studentA studentB : StudentPerformance) 
  (h : studentA.variance < studentB.variance) : 
  moreStable studentA studentB :=
by
  exact h

/-- Example application of the theorem -/
def exampleComparison : Prop :=
  let studentA : StudentPerformance := ⟨0.04⟩
  let studentB : StudentPerformance := ⟨0.13⟩
  moreStable studentA studentB

#check exampleComparison

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stable_performance_comparison_l529_52959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_standard_deviation_l529_52910

noncomputable def sample : List ℝ := [4, 5, 6]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  (xs.map (fun x => (x - mean xs)^2)).sum / xs.length

noncomputable def standardDeviation (xs : List ℝ) : ℝ :=
  Real.sqrt (variance xs)

theorem sample_standard_deviation :
  standardDeviation sample = Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_standard_deviation_l529_52910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_condition_l529_52985

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

/-- The condition for the eccentricity of the hyperbola (x^2/m - y^2/3 = 1) to be 2 -/
theorem hyperbola_eccentricity_condition (m : ℝ) : 
  (m > 0) → (m = 1 ↔ eccentricity (Real.sqrt m) (Real.sqrt 3) = 2) := by
  sorry

#check hyperbola_eccentricity_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_condition_l529_52985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_polynomial_and_x_l529_52967

/-- Given that x is a multiple of 37521, prove that the greatest common divisor
    of (3x+4)(8x+5)(15x+9)(x+15) and x is 1 -/
theorem gcd_of_polynomial_and_x (x : ℤ) (h : ∃ k : ℤ, x = 37521 * k) :
  Int.gcd ((3*x+4)*(8*x+5)*(15*x+9)*(x+15)) x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_polynomial_and_x_l529_52967

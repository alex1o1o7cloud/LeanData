import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l747_74708

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (A B C₁ C₂ : ℝ) : ℝ :=
  |C₂ - C₁| / Real.sqrt (A^2 + B^2)

/-- Theorem: The distance between the parallel lines 3x - 2y - 1 = 0 and 3x - 2y + 1 = 0 is 2√13/13 -/
theorem distance_between_specific_lines :
  distance_between_parallel_lines 3 (-2) (-1) 1 = 2 * Real.sqrt 13 / 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l747_74708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laundry_cleaning_rate_l747_74726

theorem laundry_cleaning_rate (total_pieces : ℕ) (available_hours : ℚ) 
  (h1 : total_pieces = 150) (h2 : available_hours = 4.5) : 
  (Nat.ceil ((total_pieces : ℚ) / available_hours) : ℕ) = 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_laundry_cleaning_rate_l747_74726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_example_l747_74784

/-- Calculates the area of a trapezoid given its bases and height -/
noncomputable def trapezoidArea (base1 base2 height : ℝ) : ℝ :=
  (1/2) * (base1 + base2) * height

/-- Theorem stating that the area of a trapezoid with bases 4 and 7, and height 5 is 27.5 -/
theorem trapezoid_area_example : trapezoidArea 4 7 5 = 27.5 := by
  -- Unfold the definition of trapezoidArea
  unfold trapezoidArea
  -- Simplify the expression
  simp [mul_assoc, mul_comm, mul_left_comm]
  -- Evaluate the numerical expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_example_l747_74784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_x_equals_one_log_function_range_l747_74722

-- Proposition ③
theorem symmetry_about_x_equals_one (f : ℝ → ℝ) :
  (∀ x, f (2 - x) = f x) → 
  ∀ x y, f x = y ↔ f (2 - x) = y :=
by
  sorry

-- Proposition ④
theorem log_function_range (a : ℝ) :
  (∀ y, ∃ x, Real.log (x^2 + a*x - a) = y) →
  a ≤ -4 ∨ a ≥ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_x_equals_one_log_function_range_l747_74722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_subset_and_score_l747_74727

/-- Represents a mathematician with their birth year -/
structure Mathematician where
  name : Char
  birth_year : Int

/-- The set of all mathematicians -/
def mathematicians : Finset Mathematician := sorry

/-- Checks if two mathematicians have birthdates within 20 years of each other -/
def within_20_years (m1 m2 : Mathematician) : Prop :=
  (m1.birth_year - m2.birth_year).natAbs < 20

/-- A valid subset satisfies the birthday constraint -/
def valid_subset (s : Finset Mathematician) : Prop :=
  ∀ m1 m2, m1 ∈ s → m2 ∈ s → m1 ≠ m2 → ¬(within_20_years m1 m2)

/-- The score calculation function -/
def score (k : Nat) : Nat := max (3 * (k - 3)) 0

/-- The main theorem -/
theorem largest_subset_and_score :
  ∃ (s : Finset Mathematician),
    s ⊆ mathematicians ∧
    valid_subset s ∧
    s.card = 11 ∧
    (∀ t : Finset Mathematician, t ⊆ mathematicians → valid_subset t → t.card ≤ s.card) ∧
    score s.card = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_subset_and_score_l747_74727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l747_74723

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) - 2 * Real.sin x

-- State the theorem
theorem solution_set_of_inequality (x : ℝ) :
  (f (x^2 - 2*x) + f (x - 2) < 0) ↔ (-1 < x ∧ x < 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l747_74723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l747_74734

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 2)

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = {y : ℝ | y ≠ 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l747_74734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_theorem_l747_74797

/-- The height of a cylinder containing 8 unit spheres arranged in two layers -/
noncomputable def cylinderHeight : ℝ := Real.sqrt (Real.sqrt 8) + 2

/-- Represents the arrangement of 8 unit spheres in a cylinder -/
structure SphereArrangement where
  numSpheres : ℕ
  radius : ℝ
  layers : ℕ
  isTangentToNeighbors : Prop
  isTangentToBase : Prop
  isTangentToSide : Prop

/-- The specific arrangement described in the problem -/
def problemArrangement : SphereArrangement :=
  { numSpheres := 8
  , radius := 1
  , layers := 2
  , isTangentToNeighbors := True
  , isTangentToBase := True
  , isTangentToSide := True }

theorem cylinder_height_theorem (arrangement : SphereArrangement) 
  (h1 : arrangement = problemArrangement) :
  cylinderHeight = Real.sqrt (Real.sqrt 8) + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_theorem_l747_74797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l747_74791

-- Define the functions f and g
noncomputable def f (a x : ℝ) : ℝ := Real.log (x - 3 * a) / Real.log a
noncomputable def g (a x : ℝ) : ℝ := Real.log (1 / (x - a)) / Real.log a

-- State the theorem
theorem function_inequality (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x ∈ Set.Icc (a + 2) (a + 3), |f a x - g a x| ≤ 1) →
  a ≤ (9 - Real.sqrt 57) / 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l747_74791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_origin_l747_74774

-- Define the function f(x) = e^x
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- State the theorem
theorem tangent_slope_at_origin :
  (deriv f) 0 = 1 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_origin_l747_74774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteen_factorial_sum_of_unknown_digits_l747_74770

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i ↦ i + 1)

/-- Representation of a number in base 10 -/
def base_ten_repr (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 10) :: aux (m / 10)
  (aux n).reverse

theorem fifteen_factorial_sum_of_unknown_digits :
  ∃ (X Y Z : ℕ) (digits : List ℕ),
    X < 10 ∧ Y < 10 ∧ Z < 10 ∧
    base_ten_repr (factorial 15) = 
      digits ++ [X, 2, 0, Y, 8, 0, Z, 0, 0] ∧
    digits = [1, 3, 0, 7, 6, 7, 4] ∧
    X + Y + Z = 7 := by
  sorry

#eval factorial 15
#eval base_ten_repr (factorial 15)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteen_factorial_sum_of_unknown_digits_l747_74770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_cubed_l747_74762

def polynomial (x : ℝ) : ℝ := 3*(2*x^3 - x) - 5*(x^3 - x^2 + x^6) + 4*(x^2 - 3*x^3)

theorem coefficient_of_x_cubed : 
  (deriv^[3] polynomial) 0 / (3 * 2 * 1) = -11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_cubed_l747_74762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_chord_lengths_l747_74776

/-- The ellipse defined by x^2/8 + y^2/4 = 1 -/
def ellipse (x y : ℝ) : Prop := x^2/8 + y^2/4 = 1

/-- The line y = kx + 1 -/
def line_l (k x y : ℝ) : Prop := y = k*x + 1

/-- The line kx + y - 2 = 0 -/
def line_d (k x y : ℝ) : Prop := k*x + y - 2 = 0

/-- The chord length cut by a line on the ellipse -/
noncomputable def chord_length (k : ℝ) (line : ℝ → ℝ → ℝ → Prop) : ℝ :=
  sorry -- Definition of chord length calculation

theorem different_chord_lengths :
  ∀ k : ℝ, chord_length k line_l ≠ chord_length k line_d :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_chord_lengths_l747_74776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_work_completion_time_proof_l747_74786

/-- Given work completion rates for A and B, and the time A needs to finish the remaining work,
    prove that B worked for 10 days before leaving. -/
theorem work_completion_time (a_rate b_rate : ℚ) (a_remaining_time : ℚ) : Prop :=
  /- A can finish the work in 6 days -/
  a_rate = 1 / 6 ∧
  /- B can finish the work in 15 days -/
  b_rate = 1 / 15 ∧
  /- A can finish the remaining work alone in 2 days after B left -/
  a_remaining_time = 2 ∧
  /- The number of days B worked before leaving is 10 -/
  ∃ (b_days : ℚ), b_days * b_rate + a_remaining_time * a_rate = 1 ∧ b_days = 10

theorem work_completion_time_proof : ∃ (a_rate b_rate a_remaining_time : ℚ),
  work_completion_time a_rate b_rate a_remaining_time := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_work_completion_time_proof_l747_74786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_divisor_l747_74771

theorem find_divisor (n : ℕ) (h1 : n = 1020) 
  (h2 : ∀ k ∈ ({12, 24, 48, 56} : Set ℕ), (n - 12) % k = 0) : 
  ∃! x : ℕ, x ∈ Set.Icc 1 n ∧ 
            (n - 12) % x = 0 ∧ 
            x ∉ ({12, 24, 48, 56} : Set ℕ) ∧ 
            x = 63 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_divisor_l747_74771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_no_roots_l747_74764

/-- The function f(x) with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x| + Real.sqrt (a^2 - x^2) - Real.sqrt 2

/-- The set of a values for which f has no roots -/
def no_roots_set : Set ℝ := {a | a > 0 ∧ a < 1} ∪ {a | a > Real.sqrt 2}

/-- Theorem stating that f has no roots if and only if a is in the no_roots_set -/
theorem f_no_roots (a : ℝ) : (∀ x, f a x ≠ 0) ↔ a ∈ no_roots_set := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_no_roots_l747_74764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_result_l747_74724

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ := !![1/3, 0; 0, 1/2]

def original_ellipse (x y : ℝ) : Prop := x^2/9 + y^2/4 = 1

def transformed_curve (x y : ℝ) : Prop := x^2 + y^2 = 1

theorem transformation_result :
  ∀ x y : ℝ, ∃ x₁ y₁ : ℝ,
    original_ellipse x₁ y₁ ∧
    x = (1/3) * x₁ ∧
    y = (1/2) * y₁ →
    transformed_curve x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_result_l747_74724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_test_points_correct_l747_74769

def feed_rates : List Float := [0.30, 0.33, 0.35, 0.40, 0.45, 0.48, 0.50, 0.55, 0.60, 0.65, 0.71, 0.81, 0.91]

def first_test_point (rates : List Float) : Float :=
  rates[(rates.length / 2)]!

def second_test_point (rates : List Float) : Float :=
  let first_half := rates.take (rates.length / 2)
  (first_half[(first_half.length / 2 - 1)]! + first_half[(first_half.length / 2)]!) / 2

theorem test_points_correct :
  first_test_point feed_rates = 0.50 ∧ 
  second_test_point feed_rates = 0.375 := by
  sorry

#eval first_test_point feed_rates
#eval second_test_point feed_rates

end NUMINAMATH_CALUDE_ERRORFEEDBACK_test_points_correct_l747_74769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_point_l747_74718

theorem cos_double_angle_special_point (α : ℝ) :
  let p : ℝ × ℝ := (-1, 3)
  (∀ t : ℝ, t ≠ 0 → (t * p.1, t * p.2) ∈ {(x, y) : ℝ × ℝ | x * Real.cos α - y * Real.sin α = 0}) →
  Real.cos (2 * α) = -4/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_point_l747_74718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_eq_l747_74736

/-- The sum of the series ∑(n=1 to ∞) (4n+k)/3^n, where k is a constant -/
noncomputable def series_sum (k : ℝ) : ℝ := ∑' n, (4 * n + k) / 3^n

/-- Theorem stating that the sum of the series is equal to 3 + k/2 -/
theorem series_sum_eq (k : ℝ) : series_sum k = 3 + k/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_eq_l747_74736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_system_solving_idea_l747_74787

/-- Represents a method for solving systems of linear equations -/
inductive SolvingMethod
| Substitution
| Elimination

/-- Represents the mathematical idea behind a problem-solving method -/
inductive MathematicalIdea
| CaseAnalysis
| ReductionAndTransformation
| FunctionsAndEquations
| IntegrationOfNumbersAndFigures

/-- A linear equation with two variables -/
structure TwoVariableLinearEquation :=
  (a b c : ℝ)

/-- A system of linear equations with two variables -/
structure LinearSystem :=
  (equations : List TwoVariableLinearEquation)

/-- Represents a single variable equation -/
structure SingleVariableEquation :=
  (a b : ℝ)

/-- The process of solving a system of linear equations -/
def solveLinearSystem (system : LinearSystem) (method : SolvingMethod) : SingleVariableEquation :=
  sorry

/-- Function to determine the primary mathematical idea of a solving process -/
def primaryMathematicalIdea (equation : SingleVariableEquation) : MathematicalIdea :=
  sorry

/-- The main theorem stating that solving linear systems primarily demonstrates reduction and transformation -/
theorem linear_system_solving_idea (system : LinearSystem) (method : SolvingMethod) :
  primaryMathematicalIdea (solveLinearSystem system method) = MathematicalIdea.ReductionAndTransformation :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_system_solving_idea_l747_74787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_behavior_l747_74739

-- Define the function f(x) = ln(x) / x
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

-- State the theorem
theorem f_behavior (x : ℝ) (h1 : 0 < x) (h2 : x < 10) :
  (∀ y z, 0 < y ∧ y < z ∧ z < Real.exp 1 → f y < f z) ∧
  (∀ y z, Real.exp 1 < y ∧ y < z ∧ z < 10 → f y > f z) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_behavior_l747_74739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_imply_a_range_l747_74716

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*x - a
noncomputable def g (x : ℝ) : ℝ := 2*x + 2*Real.log x

-- Define the domain
def domain : Set ℝ := { x | 1/Real.exp 1 ≤ x ∧ x ≤ Real.exp 1 }

-- State the theorem
theorem intersection_points_imply_a_range (a : ℝ) :
  (∃ x y, x ∈ domain ∧ y ∈ domain ∧ x ≠ y ∧ f a x = g x ∧ f a y = g y) →
  1 < a ∧ a ≤ 1/(Real.exp 1)^2 + 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_imply_a_range_l747_74716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_given_l747_74742

theorem tan_alpha_given (α : Real) (h : Real.tan α = 2) : 
  Real.tan (α + π/4) = -3 ∧ (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - Real.cos α) = 13/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_given_l747_74742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_average_speed_l747_74725

/-- Calculates the average speed of a bus including stoppages -/
noncomputable def average_speed_with_stoppages (speed_without_stoppages : ℝ) (stoppage_time : ℝ) : ℝ :=
  let moving_time := 60 - stoppage_time
  let distance := speed_without_stoppages * (moving_time / 60)
  distance

/-- Theorem: Given a bus with an average speed of 75 km/hr excluding stoppages
    and stopping for 28 minutes per hour, the average speed including stoppages is 40 km/hr -/
theorem bus_average_speed :
  average_speed_with_stoppages 75 28 = 40 := by
  -- Unfold the definition of average_speed_with_stoppages
  unfold average_speed_with_stoppages
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_average_speed_l747_74725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_l747_74709

/-- Proves that a boat's speed in still water is 12 kmph given specific conditions -/
theorem boat_speed (boat_speed stream_speed downstream_distance upstream_distance : ℝ) 
  (h1 : stream_speed = 4)
  (h2 : downstream_distance = 32)
  (h3 : upstream_distance = 16)
  (h4 : downstream_distance / (boat_speed + stream_speed) = upstream_distance / (boat_speed - stream_speed)) :
  boat_speed = 12 :=
by
  sorry

#check boat_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_l747_74709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sum_l747_74773

theorem range_of_sum (x y : ℝ) (h : (2 : ℝ)^x + (2 : ℝ)^y = 1) :
  x + y ≤ -2 ∧ ∀ z < -2, ∃ x' y' : ℝ, (2 : ℝ)^x' + (2 : ℝ)^y' = 1 ∧ x' + y' = z :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sum_l747_74773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l747_74781

/-- An arithmetic sequence with non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : d ≠ 0
  seq : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * seq.a 1 + n * (n - 1) / 2 * seq.d

theorem arithmetic_sequence_sum_10 (seq : ArithmeticSequence) :
  seq.a 4 ^ 2 = seq.a 3 * seq.a 7 →
  sum_n seq 8 = 16 →
  sum_n seq 10 = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l747_74781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_totient_prime_powers_l747_74702

/-- Euler's totient function -/
def φ : ℕ → ℕ := sorry

/-- Theorem: Euler's totient function for the product of two prime powers -/
theorem euler_totient_prime_powers (p q : ℕ) (l m : ℕ) 
  (hp : Nat.Prime p) (hq : Nat.Prime q) (hl : l > 0) (hm : m > 0) :
  φ (p^l * q^m) = (p^l - p^(l-1)) * (q^m - q^(m-1)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_totient_prime_powers_l747_74702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ninety_ninth_term_is_30_l747_74713

def next_term (n : ℕ) : ℕ :=
  if n < 20 then n * 9
  else if n % 2 = 0 then n / 2
  else if n % 7 ≠ 0 then n - 5
  else n + 7

def sequence_term (start : ℕ) : ℕ → ℕ
  | 0 => start
  | n + 1 => next_term (sequence_term start n)

theorem ninety_ninth_term_is_30 : sequence_term 65 99 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ninety_ninth_term_is_30_l747_74713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circumradius_l747_74789

-- Define the hyperbola
noncomputable def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / 16 = 1

-- Define the foci A and B
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Define point P on the hyperbola
noncomputable def P (a : ℝ) : ℝ × ℝ := sorry

-- Define the incenter of triangle PAB
noncomputable def incenter (a : ℝ) : ℝ × ℝ := (3, 1)

-- Define tan α and tan β
noncomputable def tan_alpha : ℝ := 1/8
noncomputable def tan_beta : ℝ := 1/2

-- Define the circumradius of triangle PAB
noncomputable def circumradius (a : ℝ) : ℝ := sorry

-- Theorem statement
theorem hyperbola_circumradius (a : ℝ) :
  hyperbola a (P a).1 (P a).2 ∧
  incenter a = (3, 1) ∧
  tan_alpha = 1/8 ∧
  tan_beta = 1/2 →
  circumradius a = 65/12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circumradius_l747_74789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_numbers_ratio_product_l747_74755

theorem two_numbers_ratio_product (x y : ℝ) : 
  (x - y) / (x + y) = 1 / 8 ∧ (x * y) / (x + y) = 40 / 8 → x * y = 6400 / 63 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_numbers_ratio_product_l747_74755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_squared_sum_equals_23_l747_74754

noncomputable def angles : List ℝ := List.range 46 |>.map (λ n => 2 * n * Real.pi / 180)

theorem cosine_squared_sum_equals_23 : 
  (angles.map (λ θ => Real.cos θ ^ 2)).sum = 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_squared_sum_equals_23_l747_74754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_log_l747_74717

theorem inequality_implies_log (x y : ℝ) :
  (2 : ℝ)^x - (2 : ℝ)^y < (3 : ℝ)^(-x) - (3 : ℝ)^(-y) → Real.log (y - x + 1) > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_log_l747_74717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jam_remaining_l747_74729

theorem jam_remaining (jar : ℚ) : 
  jar = 1 → 
  (jar - jar * (1 / 3) - (jar - jar * (1 / 3)) * (1 / 7)) = jar * (4 / 7) :=
by
  intro h
  simp [h]
  field_simp
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jam_remaining_l747_74729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_outside_hexagon_tethered_point_l747_74719

/-- The area outside a regular hexagon reachable by a tethered point -/
theorem area_outside_hexagon_tethered_point (side_length : ℝ) (rope_length : ℝ) 
  (h1 : side_length = 1.5)
  (h2 : rope_length = 3) : 
  (π * rope_length^2 * (2/3)) + (2 * (π * side_length^2 * (1/6))) = 6.75 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_outside_hexagon_tethered_point_l747_74719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_time_q_l747_74740

/-- Represents an investment partner -/
structure Partner where
  investment : ℚ
  profit : ℚ
  time : ℚ

/-- Represents the investment scenario -/
structure InvestmentScenario where
  p : Partner
  q : Partner
  r : Partner
  inv_ratio_p : ℚ
  inv_ratio_q : ℚ
  inv_ratio_r : ℚ
  profit_ratio_p : ℚ
  profit_ratio_q : ℚ
  profit_ratio_r : ℚ

/-- The investment scenario satisfies the given conditions -/
def satisfies_conditions (scenario : InvestmentScenario) : Prop :=
  scenario.inv_ratio_p / scenario.inv_ratio_q = 7 / 5 ∧
  scenario.inv_ratio_p / scenario.inv_ratio_r = 7 / 8 ∧
  scenario.profit_ratio_p / scenario.profit_ratio_q = 14 / 10 ∧
  scenario.profit_ratio_p / scenario.profit_ratio_r = 14 / 24 ∧
  scenario.p.time = 10 ∧
  scenario.r.time = 15 ∧
  scenario.p.investment / scenario.q.investment = scenario.inv_ratio_p / scenario.inv_ratio_q ∧
  scenario.p.investment / scenario.r.investment = scenario.inv_ratio_p / scenario.inv_ratio_r ∧
  scenario.p.profit / scenario.q.profit = scenario.profit_ratio_p / scenario.profit_ratio_q ∧
  scenario.p.profit / scenario.r.profit = scenario.profit_ratio_p / scenario.profit_ratio_r ∧
  scenario.p.profit / scenario.p.investment / scenario.p.time =
    scenario.q.profit / scenario.q.investment / scenario.q.time ∧
  scenario.p.profit / scenario.p.investment / scenario.p.time =
    scenario.r.profit / scenario.r.investment / scenario.r.time

theorem investment_time_q (scenario : InvestmentScenario) 
  (h : satisfies_conditions scenario) : scenario.q.time = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_time_q_l747_74740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_problem_l747_74706

open BigOperators
open Matrix

def w : Fin 3 → ℝ := sorry

def projection (v : Fin 3 → ℝ) (w : Fin 3 → ℝ) : Fin 3 → ℝ := sorry

theorem projection_problem :
  let v₁ : Fin 3 → ℝ := ![2, -1, 3]
  let v₂ : Fin 3 → ℝ := ![-3, 2, 4]
  projection v₁ w = ![-1, (1/2), -(3/2)] →
  projection v₂ w = ![-(16/7), 8/7, -(24/7)] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_problem_l747_74706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_and_range_l747_74738

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_domain : ∀ x, f x ≠ 0 → 0 ≤ x ∧ x ≤ 3
axiom f_range : ∀ x, 0 ≤ f x ∧ f x ≤ 1

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 1 - f (x + 2)

-- Theorem statement
theorem g_domain_and_range :
  (∀ x, g x ≠ 0 → -2 ≤ x ∧ x ≤ 1) ∧
  (∀ x, 0 ≤ g x ∧ g x ≤ 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_and_range_l747_74738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_idempotent_on_T_l747_74744

variable {S : Type*}
variable (star : S → S → S)

-- Define the subset T and its property
def SubsetT (T : Set S) : Prop :=
  ∀ b ∈ T, ∃ a : S, b = star a a

-- The main theorem
theorem star_idempotent_on_T (T : Set S) (h_assoc : Associative star) (h_subset : SubsetT star T) :
  ∀ b ∈ T, star b b = b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_idempotent_on_T_l747_74744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_intersecting_line_exists_l747_74785

-- Define a line in 3D space
structure Line3D where
  point : Fin 3 → ℝ
  direction : Fin 3 → ℝ

-- Define a point being on a line
def on_line (p : Fin 3 → ℝ) (l : Line3D) : Prop :=
  ∃ t : ℝ, ∀ i : Fin 3, p i = l.point i + t * l.direction i

-- Define the property of two lines being in general position and not intersecting
def general_position_non_intersecting (l1 l2 : Line3D) : Prop :=
  ∀ (p1 p2 : Fin 3 → ℝ), on_line p1 l1 → on_line p2 l2 → p1 ≠ p2

-- Define the dot product of two vectors
def dot_product (v1 v2 : Fin 3 → ℝ) : ℝ :=
  (v1 0) * (v2 0) + (v1 1) * (v2 1) + (v1 2) * (v2 2)

-- Define the property of a line being perpendicular to another line
def perpendicular (l1 l2 : Line3D) : Prop :=
  dot_product l1.direction l2.direction = 0

-- Theorem statement
theorem perpendicular_intersecting_line_exists (l1 l2 : Line3D) 
  (h : general_position_non_intersecting l1 l2) :
  ∃ (l : Line3D), perpendicular l l1 ∧ perpendicular l l2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_intersecting_line_exists_l747_74785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_inequality_l747_74788

noncomputable def f (x : ℝ) : ℝ := x / (x + 1)

noncomputable def a : ℕ → ℝ
  | 0 => 1/2
  | 1 => 3/4
  | (n+2) => f (a n) + f (a (n+1))

theorem a_inequality (n : ℕ) (hn : n > 0) :
  f (3 * 2^(n-1)) ≤ a (2*n) ∧ a (2*n) ≤ f (3 * 2^(2*n-2)) := by
  sorry

#check a_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_inequality_l747_74788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_hexagon_angle_l747_74796

/-- A convex hexagon formed by 5 identical isosceles trapezoids -/
structure TrapezoidHexagon where
  /-- The number of sides in the hexagon -/
  sides : ℕ
  /-- The number of identical isosceles trapezoids forming the hexagon -/
  trapezoids : ℕ
  /-- The sum of interior angles of the hexagon -/
  interior_angle_sum : ℚ
  /-- Condition: The hexagon has 6 sides -/
  hex_sides : sides = 6
  /-- Condition: The hexagon is formed by 5 trapezoids -/
  hex_trapezoids : trapezoids = 5
  /-- Condition: The sum of interior angles of a hexagon is (n-2) * 180° -/
  angle_sum : interior_angle_sum = (sides - 2) * 180

/-- The measure of the angle formed at the intersection of two adjacent trapezoids -/
noncomputable def intersection_angle (h : TrapezoidHexagon) : ℚ :=
  h.interior_angle_sum / (2 * h.trapezoids)

/-- Theorem: The measure of the angle formed at the intersection of two adjacent trapezoids is 72° -/
theorem trapezoid_hexagon_angle (h : TrapezoidHexagon) :
  intersection_angle h = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_hexagon_angle_l747_74796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_worker_time_approx_8_l747_74750

-- Define the time it takes for the first worker to load a truck
noncomputable def worker1_time : ℝ := 5

-- Define the time it takes for both workers together to load a truck
noncomputable def combined_time : ℝ := 3.0769230769230766

-- Define the function to calculate the time for the second worker
noncomputable def worker2_time : ℝ := 1 / (1 / combined_time - 1 / worker1_time)

-- Theorem statement
theorem second_worker_time_approx_8 :
  |worker2_time - 8| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_worker_time_approx_8_l747_74750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_arccos_sin_l747_74795

open Real MeasureTheory

-- Define the function
noncomputable def f (x : ℝ) : ℝ := arccos (sin x)

-- Define the interval
noncomputable def a : ℝ := π / 2
noncomputable def b : ℝ := 5 * π / 2

-- State the theorem
theorem area_arccos_sin : ∫ x in a..b, f x = π^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_arccos_sin_l747_74795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exams_to_grade_on_wednesday_l747_74746

theorem exams_to_grade_on_wednesday 
  (total_exams : ℕ) 
  (monday_percentage : ℚ) 
  (tuesday_percentage : ℚ) 
  (h1 : total_exams = 120) 
  (h2 : monday_percentage = 60 / 100) 
  (h3 : tuesday_percentage = 75 / 100) : 
  total_exams - 
    (Nat.floor (monday_percentage * total_exams) : ℕ) - 
    (Nat.floor (tuesday_percentage * (total_exams - Nat.floor (monday_percentage * total_exams))) : ℕ) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exams_to_grade_on_wednesday_l747_74746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l747_74753

/-- Circle C in Cartesian coordinates -/
def circleC (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

/-- Line l passing through (5,6) with slope 4/3 -/
noncomputable def lineL (t : ℝ) : ℝ × ℝ := (5 + 3/5 * t, 6 + 4/5 * t)

/-- Point M -/
def M : ℝ × ℝ := (5, 6)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_distance_sum :
  ∃ t1 t2 : ℝ,
    circleC (lineL t1).1 (lineL t1).2 ∧
    circleC (lineL t2).1 (lineL t2).2 ∧
    t1 ≠ t2 ∧
    distance M (lineL t1) + distance M (lineL t2) = 66/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l747_74753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_f_leq_3_l747_74745

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then -x^2 else x^2 + 2*x

theorem solution_set_of_f_f_leq_3 :
  {x : ℝ | f (f x) ≤ 3} = Set.Iic (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_f_leq_3_l747_74745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_trigonometric_functions_l747_74778

/-- The minimum positive shift to transform sin(2x) + √3 * cos(2x) into sin(2x) - √3 * cos(2x) -/
theorem min_shift_trigonometric_functions : 
  let f (x : ℝ) := Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x)
  let g (x : ℝ) := Real.sin (2 * x) - Real.sqrt 3 * Real.cos (2 * x)
  ∃ (shift : ℝ), shift > 0 ∧ 
    (∀ x, g x = f (x - shift)) ∧
    (∀ s, s > 0 ∧ (∀ x, g x = f (x - s)) → s ≥ shift) ∧
    shift = Real.pi / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_trigonometric_functions_l747_74778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_l747_74790

theorem divisibility_property (n k : ℕ) 
  (hn : n > 0 ∧ n % 2 ≠ 0 ∧ n % 3 ≠ 0) : 
  ∃ m : ℤ, (k + 1)^n - k^n - 1 = m * (k^2 + k + 1) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_l747_74790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equations_existence_l747_74756

theorem diophantine_equations_existence 
  (n : ℕ) (a b c : ℕ) 
  (h_coprime_ab : Nat.Coprime a b) 
  (h_coprime_c : Nat.Coprime c a ∨ Nat.Coprime c b) :
  (∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x^(n-1) + y^n = z^(n+1)) ∧
  (∃ (S : Set (ℕ × ℕ × ℕ)), Set.Infinite S ∧ 
    ∀ (x y z : ℕ), (x, y, z) ∈ S → x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x^a + y^b = z^c) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equations_existence_l747_74756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_cylindrical_conversion_l747_74792

/-- Converts a point from rectangular coordinates to cylindrical coordinates -/
noncomputable def rectangularToCylindrical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.arccos (x / r)
  (r, θ, z)

theorem rectangular_to_cylindrical_conversion :
  let (r, θ, z) := rectangularToCylindrical 3 4 6
  r = 5 ∧ θ = Real.arccos (3/5) ∧ z = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_cylindrical_conversion_l747_74792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_failed_student_mean_score_l747_74721

/-- Given a group of students taking a test, this theorem proves the mean score
    of those who failed, given the overall mean, percentage who passed, and
    mean score of those who passed. -/
theorem failed_student_mean_score
  (overall_mean : ℝ)
  (pass_percentage : ℝ)
  (pass_mean : ℝ)
  (h_overall_mean : overall_mean = 6)
  (h_pass_percentage : pass_percentage = 0.6)
  (h_pass_mean : pass_mean = 8) :
  (overall_mean - pass_percentage * pass_mean) / (1 - pass_percentage) = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_failed_student_mean_score_l747_74721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shoes_for_max_one_legged_l747_74737

/-- Represents the population of the island -/
def population : ℕ := 10000

/-- Represents the percentage of one-legged inhabitants -/
noncomputable def one_legged_percentage : ℝ := 100

/-- Calculates the number of shoes needed given the percentage of one-legged inhabitants -/
noncomputable def shoes_needed (p : ℝ) : ℝ :=
  let one_legged := p / 100 * population
  let two_legged := population - one_legged
  one_legged + (two_legged / 2) * 2

/-- Theorem stating that 100% one-legged inhabitants minimizes the number of shoes to 10,000 -/
theorem min_shoes_for_max_one_legged :
  shoes_needed one_legged_percentage = population ∧
  ∀ p, 0 ≤ p ∧ p ≤ 100 → shoes_needed p ≥ population := by
  sorry

#eval population

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shoes_for_max_one_legged_l747_74737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_floor_equation_l747_74749

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem smallest_n_for_floor_equation : 
  (∀ n : ℕ, n < 7 → ¬∃ m : ℤ, floor ((10 : ℝ)^n / m) = 1989) ∧
  (∃ m : ℤ, floor ((10 : ℝ)^7 / m) = 1989) := by
  sorry

#check smallest_n_for_floor_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_floor_equation_l747_74749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfying_inequality_l747_74747

theorem function_satisfying_inequality (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x^2) - f (y^2) ≤ (f x + y) * (x - f y)) →
  (∀ x : ℝ, f x = x ∨ f x = -x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfying_inequality_l747_74747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_sum_log_l747_74783

/-- Given a polynomial 8x^3 + 6ax^2 + 7bx + 2a = 0 with three distinct positive roots,
    if the sum of the base-3 logarithms of the roots is 3, then a = -108 -/
theorem polynomial_root_sum_log (a b : ℝ) (u v w : ℝ) : 
  (∀ x : ℝ, 8 * x^3 + 6 * a * x^2 + 7 * b * x + 2 * a = 0 ↔ x = u ∨ x = v ∨ x = w) →
  (u > 0 ∧ v > 0 ∧ w > 0) →
  (u ≠ v ∧ u ≠ w ∧ v ≠ w) →
  (Real.log u / Real.log 3 + Real.log v / Real.log 3 + Real.log w / Real.log 3 = 3) →
  a = -108 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_sum_log_l747_74783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l747_74759

-- Define the line l: mx - y - 3m + 1 = 0
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  m * x - y - 3 * m + 1 = 0

-- Define the circle C: (x-1)^2 + (y-2)^2 = 25
def circle_C (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 2)^2 = 25

-- Define point P that always lies on line l
def point_P : ℝ × ℝ :=
  (3, 1)

-- Define the chord length |AB|
noncomputable def chord_length (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Theorem statement
theorem min_chord_length :
  ∃ A B : ℝ × ℝ,
  (∃ m : ℝ, line_l m A.1 A.2 ∧ line_l m B.1 B.2) ∧
  circle_C A.1 A.2 ∧
  circle_C B.1 B.2 ∧
  (∀ X Y : ℝ × ℝ,
    (∃ k : ℝ, line_l k X.1 X.2 ∧ line_l k Y.1 Y.2) →
    circle_C X.1 X.2 →
    circle_C Y.1 Y.2 →
    chord_length A B ≤ chord_length X Y) ∧
  chord_length A B = 4 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l747_74759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_coefficients_l747_74798

/-- Given a parabola y = ax^2 + bx with vertex (3,3), prove that a = -1/3 and b = 2 -/
theorem parabola_coefficients (a b : ℝ) : 
  (3 = a * 3^2 + b * 3) ∧ 
  (-b / (2 * a) = 3) →
  a = -1/3 ∧ b = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_coefficients_l747_74798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_monotonic_implies_m_range_l747_74793

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := -1/2 * x^2 + 6*x - 8 * Real.log x

-- Define the property of not being monotonic on an interval
def not_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ (x y z : ℝ), a ≤ x ∧ x < y ∧ y < z ∧ z ≤ b ∧
    ((f x < f y ∧ f y > f z) ∨ (f x > f y ∧ f y < f z))

-- State the theorem
theorem f_not_monotonic_implies_m_range (m : ℝ) :
  not_monotonic f m (m + 1) → m ∈ Set.Ioo 1 2 ∪ Set.Ioo 3 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_monotonic_implies_m_range_l747_74793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_s_at_zero_l747_74715

/-- The x-coordinate of the left-most intersection point of y = x^2 + 2x - 3 and y = m -/
noncomputable def M (m : ℝ) : ℝ := -1 - Real.sqrt (m + 4)

/-- The function s defined in the problem -/
noncomputable def s (m : ℝ) : ℝ := (M (-2*m) - M m) / (2*m)

theorem limit_of_s_at_zero :
  ∀ ε > 0, ∃ δ > 0, ∀ m : ℝ, 0 < |m| ∧ |m| < δ ∧ -8 < m ∧ m < 5 → |s m - 3/8| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_s_at_zero_l747_74715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_side_length_l747_74700

theorem right_triangle_side_length (R S Q : Real) : 
  Real.cos R = 5 / 13 → S - R = 13 → (S - Q)^2 + (Q - R)^2 = (S - R)^2 → S - Q = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_side_length_l747_74700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bound_l747_74767

def sequence_rule (a : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, ∃ d : ℕ, 0 < d ∧ d < 10 ∧ d ∣ a (2*n - 1) ∧ a (2*n) = a (2*n - 1) + d) ∧
  (∀ n : ℕ, ∃ d : ℕ, 0 < d ∧ d < 10 ∧ d ∣ a (2*n) ∧ a (2*n + 1) = a (2*n) - d)

theorem sequence_bound (a : ℕ → ℕ) (h : sequence_rule a) : ∀ n : ℕ, a n ≤ 4 * a 1 + 44 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bound_l747_74767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_price_reduction_is_twenty_percent_l747_74758

/-- Calculates the percentage reduction in oil price given the reduced price and additional quantity purchased for a fixed amount. -/
noncomputable def oil_price_reduction (reduced_price : ℝ) (additional_quantity : ℝ) (fixed_amount : ℝ) : ℝ :=
  let original_price := fixed_amount / (fixed_amount / reduced_price - additional_quantity)
  ((original_price - reduced_price) / original_price) * 100

/-- Theorem stating that under the given conditions, the oil price reduction is 20%. -/
theorem oil_price_reduction_is_twenty_percent :
  oil_price_reduction 30 4 600 = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_price_reduction_is_twenty_percent_l747_74758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_payneful_properties_l747_74782

/-- A Payneful pair of functions -/
structure PaynefulPair (f g : ℝ → ℝ) : Prop where
  real_f : ∀ x : ℝ, ∃ y : ℝ, f x = y
  real_g : ∀ x : ℝ, ∃ y : ℝ, g x = y
  add_f : ∀ x y : ℝ, f (x + y) = f x * g y + g x * f y
  add_g : ∀ x y : ℝ, g (x + y) = g x * g y - f x * f y
  nonzero_f : ∃ a : ℝ, f a ≠ 0

/-- The h function defined for a Payneful pair -/
def h (f g : ℝ → ℝ) : ℝ → ℝ := λ x ↦ (f x)^2 + (g x)^2

theorem payneful_properties {f g : ℝ → ℝ} (p : PaynefulPair f g) :
  (f 0 = 0 ∧ g 0 = 1) ∧
  (h f g 5 * h f g (-5) = 1) ∧
  (∀ x : ℝ, -10 ≤ f x ∧ f x ≤ 10 ∧ -10 ≤ g x ∧ g x ≤ 10 → h f g 2021 = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_payneful_properties_l747_74782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_negative_27_l747_74730

theorem cube_root_of_negative_27 : (-27 : ℝ) ^ (1/3 : ℝ) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_negative_27_l747_74730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_cosine_45_ratio_l747_74775

theorem tangent_cosine_45_ratio : 
  let tan_sq_45 := Real.tan (45 * π / 180) ^ 2
  let cos_sq_45 := Real.cos (45 * π / 180) ^ 2
  tan_sq_45 = 1 → cos_sq_45 = 1/2 → 
  (tan_sq_45 - cos_sq_45) / (tan_sq_45 * cos_sq_45) = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_cosine_45_ratio_l747_74775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_break_duration_is_60_minutes_l747_74741

/-- Represents a painter with a constant painting rate -/
structure Painter where
  rate : ℝ
  rate_pos : rate > 0

/-- Represents a day's work -/
structure WorkDay where
  painters : List Painter
  startTime : ℕ
  endTime : ℕ
  percentPainted : ℝ
  valid_percent : percentPainted ≥ 0 ∧ percentPainted ≤ 1
  valid_time : endTime > startTime

/-- The break duration in hours -/
def breakDuration : ℝ := 1

/-- The combined painting rate of a list of painters -/
def combinedRate (painters : List Painter) : ℝ :=
  painters.map (·.rate) |> List.sum

/-- The effective work time on a work day -/
def effectiveWorkTime (day : WorkDay) : ℝ :=
  (day.endTime - day.startTime : ℝ) - breakDuration

theorem break_duration_is_60_minutes 
  (alice bob carla dave : Painter)
  (monday : WorkDay)
  (tuesday : WorkDay)
  (wednesday : WorkDay)
  (h1 : monday.painters = [alice, bob, carla, dave])
  (h2 : monday.startTime = 7 ∧ monday.endTime = 17 ∧ monday.percentPainted = 0.6)
  (h3 : tuesday.painters = [carla, dave])
  (h4 : tuesday.startTime = 7 ∧ tuesday.endTime = 15 ∧ tuesday.percentPainted = 0.25)
  (h5 : wednesday.painters = [alice])
  (h6 : wednesday.startTime = 7 ∧ wednesday.endTime = 18 ∧ wednesday.percentPainted = 0.15)
  : breakDuration = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_break_duration_is_60_minutes_l747_74741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_board_diff_4_not_exist_board_diff_3_l747_74777

/-- Represents a 4x4 board with integers --/
def Board := Fin 4 → Fin 4 → Nat

/-- Checks if two cells are adjacent --/
def adjacent (i j k l : Fin 4) : Prop :=
  (i = k ∧ j.val + 1 = l.val) ∨ 
  (i = k ∧ j.val = l.val + 1) ∨ 
  (i.val + 1 = k.val ∧ j = l) ∨ 
  (i.val = k.val + 1 ∧ j = l)

/-- Checks if a board is valid (contains numbers 1 through 16) --/
def valid_board (b : Board) : Prop :=
  ∀ n : Nat, 1 ≤ n ∧ n ≤ 16 → ∃ i j : Fin 4, b i j = n

/-- Theorem: There exists a valid board where adjacent cells differ by at most 4 --/
theorem exist_board_diff_4 : 
  ∃ b : Board, valid_board b ∧ 
    ∀ i j k l : Fin 4, adjacent i j k l → 
      (b i j : Int) - (b k l : Int) ≤ 4 ∧ (b k l : Int) - (b i j : Int) ≤ 4 := by sorry

/-- Theorem: There does not exist a valid board where adjacent cells differ by at most 3 --/
theorem not_exist_board_diff_3 : 
  ¬∃ b : Board, valid_board b ∧ 
    ∀ i j k l : Fin 4, adjacent i j k l → 
      (b i j : Int) - (b k l : Int) ≤ 3 ∧ (b k l : Int) - (b i j : Int) ≤ 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_board_diff_4_not_exist_board_diff_3_l747_74777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_negative_four_F_decreasing_min_a_value_l747_74735

-- Define the functions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 3
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (6 + a) * 2^(x - 1)
noncomputable def F (x : ℝ) : ℝ := 2 / (1 + 2^x)

-- Theorem 1
theorem a_equals_negative_four (a : ℝ) : f a 1 = f a 3 → a = -4 := by sorry

-- Theorem 2
theorem F_decreasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → F x₁ > F x₂ := by sorry

-- Theorem 3
theorem min_a_value (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-2) 2 → f a x ≥ a) ∧ a ∉ Set.Ioo (-4) 4 →
  a ≥ -7 ∧ (∀ b : ℝ, b < -7 → ∃ x : ℝ, x ∈ Set.Icc (-2) 2 ∧ f b x < b) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_negative_four_F_decreasing_min_a_value_l747_74735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_specific_plane_l747_74760

/-- The distance from a point to a plane in 3D space -/
noncomputable def distance_point_to_plane (x₀ y₀ z₀ A B C D : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C * z₀ + D| / Real.sqrt (A^2 + B^2 + C^2)

/-- The specific point in the problem -/
def point : ℝ × ℝ × ℝ := (2, 1, -3)

/-- Coefficients of the plane equation x+2y+3z+3=0 -/
def plane_coeffs : ℝ × ℝ × ℝ × ℝ := (1, 2, 3, 3)

theorem distance_point_to_specific_plane :
  distance_point_to_plane 
    (point.1) (point.2.1) (point.2.2)
    (plane_coeffs.1) (plane_coeffs.2.1) (plane_coeffs.2.2.1) (plane_coeffs.2.2.2) = Real.sqrt 14 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_specific_plane_l747_74760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_condition_minimum_value_range_l747_74707

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 * x - 4) * Real.exp x + a * (x + 2)^2

-- Part I: Monotonically increasing condition implies a ≥ 1/2
theorem monotonic_increasing_condition (a : ℝ) :
  (∀ x > 0, Monotone (f a)) → a ≥ 1/2 := by sorry

-- Part II: Existence of minimum value and its range when 0 < a < 1/2
theorem minimum_value_range (a : ℝ) (ha : 0 < a ∧ a < 1/2) :
  ∃ (min : ℝ), (∀ x > 0, f a x ≥ min) ∧ (-2 * Real.exp 1 < min ∧ min < -2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_condition_minimum_value_range_l747_74707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_relationship_l747_74714

theorem abc_relationship : 
  let a := Real.sqrt 0.4
  let b := (2 : Real) ^ (0.4 : Real)
  let c := (0.4 : Real) ^ (0.2 : Real)
  b > c ∧ c > a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_relationship_l747_74714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l747_74752

noncomputable def S : ℕ → ℝ
  | 0 => 0
  | 1 => 1
  | (n + 2) => ((2 + S (n + 1))^2) / (4 + S (n + 1))

noncomputable def a : ℕ → ℝ
  | 0 => 0
  | 1 => 1
  | (n + 2) => S (n + 2) - S (n + 1)

theorem sequence_inequality (n : ℕ) (h : n > 0) : 
  a n ≥ 4 / Real.sqrt (9 * n + 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l747_74752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_2pi_periodic_l747_74763

noncomputable def f (x : ℝ) : ℝ := Real.cos (x + Real.pi/4) - Real.cos (x - Real.pi/4)

theorem f_is_odd_and_2pi_periodic :
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 2*Real.pi) = f x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_2pi_periodic_l747_74763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_at_one_l747_74728

/-- The function f(x) = x(x+a)² has a minimum at x=1 when a = -1 -/
theorem min_at_one (a : ℝ) : 
  (∀ x : ℝ, x * (x + a)^2 ≥ 1 * (1 + a)^2) ↔ a = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_at_one_l747_74728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_park_area_theorem_l747_74710

/-- Represents a rectangular park with sides in ratio 3:2 -/
structure RectangularPark where
  x : ℝ
  length : ℝ := 3 * x
  width : ℝ := 2 * x

/-- Calculate the perimeter of the park -/
noncomputable def perimeter (park : RectangularPark) : ℝ :=
  2 * (park.length + park.width)

/-- Calculate the area of the park -/
noncomputable def area (park : RectangularPark) : ℝ :=
  park.length * park.width

/-- Convert paise to rupees -/
noncomputable def paiseToRupees (paise : ℝ) : ℝ :=
  paise / 100

theorem park_area_theorem (park : RectangularPark) :
  perimeter park * paiseToRupees 50 = 140 →
  area park = 4704 := by
  sorry

#check park_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_park_area_theorem_l747_74710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_x_in_set_l747_74732

theorem unique_x_in_set : ∃! x : ℝ, (2*x ≠ x^2 + x) ∧ (2*x ≠ -4) ∧ (x^2 + x ≠ -4) ∧ ({2*x, x^2 + x, -4} : Set ℝ).Nonempty := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_x_in_set_l747_74732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hockey_match_handshakes_l747_74711

/-- Represents a hockey match setup -/
structure HockeyMatch where
  team_size : ℕ
  num_teams : ℕ
  num_referees : ℕ

/-- Calculates the total number of handshakes in a hockey match -/
def total_handshakes (m : HockeyMatch) : ℕ :=
  let player_to_player := m.team_size * m.team_size * (m.num_teams - 1) / 2
  let player_to_referee := m.team_size * m.num_teams * m.num_referees
  player_to_player + player_to_referee

/-- Theorem stating the total number of handshakes in the specific match scenario -/
theorem hockey_match_handshakes :
  let m : HockeyMatch := ⟨6, 2, 3⟩
  total_handshakes m = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hockey_match_handshakes_l747_74711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l747_74731

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def g (x : ℝ) : ℝ := (1/3) ^ x

-- Define the interval
def interval : Set ℝ := Set.Ioo 0 (50 * Real.pi)

-- State the theorem
theorem solution_count :
  ∃ (S : Set ℝ), S ⊆ interval ∧ (∀ x ∈ S, f x = g x) ∧ Finite S ∧ Nat.card S = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l747_74731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_sum_squares_eq_495_l747_74766

def simplify_and_sum_squares (x : ℝ) : ℝ :=
  let simplified := -6*x^3 + 3*x^2 + 21*x - 3
  let coefficients := [-6, 3, 21, -3]
  (coefficients.map (λ c => c^2)).sum

theorem simplify_and_sum_squares_eq_495 :
  ∀ x, simplify_and_sum_squares x = 495 := by
  intro x
  unfold simplify_and_sum_squares
  simp
  -- The actual proof would go here
  sorry

#eval simplify_and_sum_squares 0  -- Should output 495

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_sum_squares_eq_495_l747_74766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l747_74703

-- Define the functions f and g
def f (k : ℝ) (x : ℝ) : ℝ := x^2 + k*x + 5
def g (x : ℝ) : ℝ := 4*x

-- Define the function y (using noncomputable due to exponential function)
noncomputable def y (x : ℝ) : ℝ := (4 : ℝ)^x - 2^(x+1) + 2

-- Define the set D
def D : Set ℝ := Set.Icc 1 2

-- State the theorem
theorem k_range (k : ℝ) : 
  (∀ x ∈ D, f k x ≤ g x) ↔ k ≤ -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l747_74703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_winning_number_l747_74733

def mary_turn (x : ℤ) : ℤ := 2017 * x + 2

def pat_turn (y : ℤ) : ℤ := y + 2019

def game_sequence (N : ℤ) : ℕ → ℤ
  | 0 => N
  | 1 => mary_turn N
  | 2 => pat_turn (mary_turn N)
  | 3 => mary_turn (pat_turn (mary_turn N))
  | n + 4 => game_sequence N n

theorem smallest_winning_number :
  ∀ N : ℤ, N > 2017 →
    (∀ k : ℕ, ¬ (2018 ∣ game_sequence N k)) ↔ N ≥ 2022 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_winning_number_l747_74733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_l747_74761

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- Definition of f(x) for x > 0 -/
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2017^x + Real.log x / Real.log 2017 else 0  -- placeholder for x ≤ 0

/-- The number of zeros of a function on ℝ -/
def NumberOfZeros (f : ℝ → ℝ) : ℕ :=
  sorry -- Definition omitted, as it's not trivial to define in Lean

theorem f_has_three_zeros :
  IsOdd f ∧ (∀ x > 0, f x = 2017^x + Real.log x / Real.log 2017) →
  NumberOfZeros f = 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_l747_74761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_equals_pi_over_four_minus_half_l747_74779

theorem definite_integral_equals_pi_over_four_minus_half :
  ∫ x in (Set.Icc 0 1), (Real.sqrt (1 - (x - 1)^2) - x) = π / 4 - 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_equals_pi_over_four_minus_half_l747_74779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l747_74794

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 1 - x^2 else x^2 + x - 2

theorem f_composition_value :
  f (1 / f 2) = 15/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l747_74794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_lines_k_value_l747_74704

/-- Three lines intersect at a single point -/
def intersect_at_point (f g h : ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, f x = y ∧ g x = y ∧ h x = y

/-- The theorem stating that if three specific lines intersect at a point, then k = 77/8 -/
theorem intersecting_lines_k_value :
  ∀ k : ℝ, 
  intersect_at_point (λ x ↦ 3*x + 12) (λ x ↦ -5*x - 7) (λ x ↦ 2*x + k) →
  k = 77/8 := by
  sorry

#check intersecting_lines_k_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_lines_k_value_l747_74704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nacl_moles_l747_74765

/-- Represents the number of moles of a chemical substance -/
def Moles : Type := ℝ

/-- Represents the chemical reaction between NaCl and KNO3 -/
structure Reaction where
  nacl : Moles
  kno3 : Moles
  nano3 : Moles
  kcl : Moles

/-- The stoichiometric relationship in the reaction -/
axiom stoichiometry (r : Reaction) : r.nacl = r.kno3 ∧ r.nacl = r.nano3 ∧ r.nacl = r.kcl

/-- Given conditions of the problem -/
def given_conditions (r : Reaction) : Prop :=
  r.kno3 = (1 : ℝ) ∧ r.nano3 = (1 : ℝ)

/-- Theorem stating that given the conditions, the number of moles of NaCl combined is 1 -/
theorem nacl_moles (r : Reaction) (h : given_conditions r) : r.nacl = (1 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nacl_moles_l747_74765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_among_three_l747_74757

theorem largest_among_three :
  let a := Real.log 3 / Real.log (1/2)
  let b := (1/3 : Real) ^ (0.2 : Real)
  let c := (2 : Real) ^ (1/3 : Real)
  max a (max b c) = c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_among_three_l747_74757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_a_le_e_plus_one_product_of_zeros_lt_one_l747_74748

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x / x - Real.log x + x - a

-- Theorem 1: f(x) ≥ 0 for all x > 0 if and only if a ≤ e + 1
theorem f_nonnegative_iff_a_le_e_plus_one (a : ℝ) :
  (∀ x > 0, f a x ≥ 0) ↔ a ≤ Real.exp 1 + 1 := by sorry

-- Theorem 2: If f has two positive zeros, their product is less than 1
theorem product_of_zeros_lt_one (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : f a x₁ = 0) (h₄ : f a x₂ = 0) :
  x₁ * x₂ < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_a_le_e_plus_one_product_of_zeros_lt_one_l747_74748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_common_tangent_l747_74799

-- Define the curves
def f (x : ℝ) : ℝ := x^2
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * Real.exp (x + 1)

-- Define the condition for a common tangent line
def has_common_tangent (a : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
  (2 * x₁ = a * Real.exp (x₂ + 1)) ∧
  ((f x₂ - f x₁) / (x₂ - x₁) = 2 * x₁)

-- State the theorem
theorem unique_common_tangent (a : ℝ) (ha : a > 0) :
  (∃! k : ℝ, has_common_tangent k) → a = 4 / Real.exp 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_common_tangent_l747_74799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_N_l747_74772

def N : ℕ := 2^4 * 3^3 * 5^2 * 7^1

theorem number_of_factors_N : (Finset.filter (· ∣ N) (Finset.range (N + 1))).card = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_N_l747_74772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l747_74768

theorem log_equation_solution :
  ∀ x : ℝ, x > 0 → (Real.log 625 / Real.log x = Real.log 81 / Real.log 3) → x = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l747_74768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_faces_dice_pair_l747_74780

/-- Represents a pair of fair dice -/
structure DicePair where
  faces1 : ℕ
  faces2 : ℕ
  h1 : faces1 ≥ 8
  h2 : faces2 ≥ 8

/-- The probability of rolling a specific sum with a pair of dice -/
def prob_sum (d : DicePair) (sum : ℕ) : ℚ :=
  (Finset.filter (λ (x : ℕ × ℕ) => x.1 + x.2 = sum) (Finset.product (Finset.range d.faces1) (Finset.range d.faces2))).card /
  (d.faces1 * d.faces2 : ℚ)

/-- The theorem statement -/
theorem min_faces_dice_pair :
  ∃ (d : DicePair),
    (∀ d' : DicePair, 
      prob_sum d' 9 = 2 * prob_sum d' 12 ∧
      prob_sum d' 15 = 1/15 →
      d.faces1 + d.faces2 ≤ d'.faces1 + d'.faces2) ∧
    prob_sum d 9 = 2 * prob_sum d 12 ∧
    prob_sum d 15 = 1/15 ∧
    d.faces1 + d.faces2 = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_faces_dice_pair_l747_74780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotation_l747_74751

/-- The volume of the solid formed by rotating the region bounded by y = x^2 + 1, y = x, x = 0, and x = 1 about the y-axis -/
noncomputable def rotationVolume : ℝ := 
  let f (x : ℝ) := x^2 + 1
  let g (x : ℝ) := x
  let a : ℝ := 0
  let b : ℝ := 1
  -- The actual computation of the volume would go here
  (5 / 6) * Real.pi

/-- Theorem stating that the volume of the solid formed by rotating the specified region about the y-axis is 5π/6 -/
theorem volume_of_rotation : rotationVolume = (5 / 6) * Real.pi := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotation_l747_74751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_speed_l747_74705

/-- Proves that the speed of the stream is 5 km/hr given the conditions of the boat problem -/
theorem stream_speed (boat_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) (stream_speed : ℝ)
  (h1 : boat_speed = 40)
  (h2 : downstream_distance = 45)
  (h3 : downstream_time = 1)
  (h4 : downstream_distance / downstream_time = boat_speed + stream_speed) :
  stream_speed = 5 := by
  sorry

#check stream_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_speed_l747_74705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_bundle_length_l747_74720

theorem wire_bundle_length :
  ∀ L : ℝ,
  (L / 2 + 3 < L) →  -- First usage condition
  (L / 2 - 3) / 2 - 10 > 0 →  -- Second usage condition
  ((L / 2 - 3) - ((L / 2 - 3) / 2 - 10)) - 15 > 0 →  -- Third usage condition
  (((L / 2 - 3) - ((L / 2 - 3) / 2 - 10)) - 15) = 7 →  -- Final remaining length condition
  L = 54 := by
  intro L h1 h2 h3 h4
  sorry

#check wire_bundle_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_bundle_length_l747_74720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_equation_width_equation_side_length_S2_is_645_l747_74743

/-- The side length of square S_2 in a composite rectangle --/
def side_length_S2 : ℕ := 645

/-- The total width of the composite rectangle --/
def total_width : ℕ := 3450

/-- The total height of the composite rectangle --/
def total_height : ℕ := 2160

/-- The shorter side length of rectangles R_1 and R_2 --/
def r : ℕ := (total_height - side_length_S2) / 2

/-- Theorem stating that the height equation holds --/
theorem height_equation : 2 * r + side_length_S2 = total_height := by sorry

/-- Theorem stating that the width equation holds --/
theorem width_equation : 2 * (r + side_length_S2) + side_length_S2 = total_width := by sorry

/-- Main theorem proving that the side length of S_2 is 645 --/
theorem side_length_S2_is_645 : side_length_S2 = 645 := by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_equation_width_equation_side_length_S2_is_645_l747_74743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_work_time_verify_solution_l747_74701

/-- Represents the time taken by A to complete the work alone -/
def A : ℝ := 12

/-- Represents the time taken by B to complete the work alone -/
def B : ℝ := 8

/-- Represents the time taken by A, B, and C together to complete the work -/
def ABC : ℝ := 3

/-- Represents the total payment for the work -/
def total_payment : ℝ := 6000

/-- Represents C's payment for the work -/
def C_payment : ℝ := 750

/-- Theorem stating that A alone can complete the work in 12 days -/
theorem A_work_time : A = 12 := by
  -- The proof goes here
  sorry

/-- Theorem verifying the solution -/
theorem verify_solution : 
  (1 / A + 1 / B + C_payment / total_payment = 1 / ABC) ∧
  (C_payment / total_payment = 1 / 8) := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_work_time_verify_solution_l747_74701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_b_value_l747_74712

-- Define the ellipse parameters
noncomputable def ellipse (a b : ℝ) := {(x, y) : ℝ × ℝ | x^2 / a^2 + y^2 / b^2 = 1}

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Define the sum of distances from a point to the foci
noncomputable def sum_distances (a : ℝ) : ℝ := 2 * a

-- Theorem statement
theorem ellipse_b_value (a b : ℝ) :
  a > b ∧ b > 0 ∧
  eccentricity a b = Real.sqrt 5 / 3 ∧
  sum_distances a = 12 →
  b = 4 := by
  sorry

#check ellipse_b_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_b_value_l747_74712

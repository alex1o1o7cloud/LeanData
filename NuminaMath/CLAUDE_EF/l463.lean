import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_vertical_asymptote_l463_46374

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (x + 2) / (x^2 - 2*x - 8)

-- Theorem statement
theorem f_has_one_vertical_asymptote :
  ∃! a : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - a| ∧ |x - a| < δ → |f x| > 1/ε :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_vertical_asymptote_l463_46374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_husband_weekly_contribution_l463_46380

def husband_contribution : ℚ → ℚ := λ H => H
def wife_contribution : ℚ → ℚ := λ _ => 225
def savings_period : ℚ := 6
def weeks_per_month : ℚ := 4
def num_children : ℚ := 4
def child_receives : ℚ := 1680

theorem husband_weekly_contribution :
  ∃ (H : ℚ),
    (H = 335) ∧
    (savings_period * weeks_per_month * (husband_contribution H + wife_contribution H) =
     2 * (num_children * child_receives)) :=
by
  use 335
  apply And.intro
  · rfl
  · norm_num
    rfl

#check husband_weekly_contribution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_husband_weekly_contribution_l463_46380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_plane_l463_46376

noncomputable def plane (x y z : ℝ) : Prop := 4*x + 3*y - 2*z = 40

noncomputable def distance (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2 + (z₂ - z₁)^2)

theorem closest_point_on_plane :
  let P : ℝ × ℝ × ℝ := (134/29, 82/29, -55/29)
  let A : ℝ × ℝ × ℝ := (2, 1, -1)
  plane P.1 P.2.1 P.2.2 ∧
  ∀ Q : ℝ × ℝ × ℝ, plane Q.1 Q.2.1 Q.2.2 →
    distance P.1 P.2.1 P.2.2 A.1 A.2.1 A.2.2 ≤ distance Q.1 Q.2.1 Q.2.2 A.1 A.2.1 A.2.2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_plane_l463_46376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_proposition_truth_l463_46377

-- Define the propositions
def p : Prop := ∀ x : ℝ, |x| = x ↔ x > 0

def q : Prop := ∀ f : ℝ → ℝ, Function.HasRightInverse f → Monotone f

-- State the theorem
theorem composite_proposition_truth :
  (∃ x : ℝ, |x| = x ∧ x ≤ 0) →
  (∃ f : ℝ → ℝ, Function.HasRightInverse f ∧ ¬Monotone f) →
  ¬(p ∧ q) ∧ ¬(p ∨ q) ∧ ¬(¬p ∧ q) ∧ (¬p ∨ q) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_proposition_truth_l463_46377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_doubling_time_l463_46390

/-- The time required for a population to double given specific birth and death rates -/
theorem population_doubling_time (birth_rate death_rate : ℝ) 
  (h_birth : birth_rate = 39.4)
  (h_death : death_rate = 19.4)
  (h_positive : birth_rate > death_rate) :
  (Real.log 2) / ((birth_rate - death_rate) / 1000) = (Real.log 2) / 0.02 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_doubling_time_l463_46390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_other_intercept_l463_46323

/-- An ellipse with foci at (0,3) and (4,0) that passes through the origin -/
structure Ellipse where
  /-- The sum of distances from any point on the ellipse to the two foci is constant -/
  constant_sum : ℝ
  /-- The ellipse passes through the origin -/
  passes_through_origin : Real.sqrt 9 + Real.sqrt 16 = constant_sum

/-- The other x-intercept of the ellipse -/
noncomputable def other_x_intercept (e : Ellipse) : ℝ := 55 / 16

theorem ellipse_other_intercept (e : Ellipse) :
  let x := other_x_intercept e
  Real.sqrt (x^2 + 9) + Real.sqrt ((x - 4)^2) = e.constant_sum :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_other_intercept_l463_46323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_consecutive_product_l463_46362

/-- Definition of the function f(n, k) -/
def f (n k : ℕ) : ℕ := 2 * n^(3*k) + 4 * n^k + 10

/-- Theorem stating that f(n, k) cannot be expressed as the product of consecutive natural numbers -/
theorem f_not_consecutive_product (n k : ℕ) :
  ¬ ∃ (m p : ℕ), p > 0 ∧ f n k = (List.range p).foldl (· * ·) (m + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_consecutive_product_l463_46362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scheduling_methods_count_l463_46384

/-- The number of staff members -/
def n : ℕ := 7

/-- The number of days to schedule -/
def d : ℕ := 7

/-- The number of staff members who cannot be scheduled on the first two days -/
def k : ℕ := 2

/-- The function representing the number of scheduling methods -/
def scheduling_methods (n d k : ℕ) : ℕ :=
  Nat.descFactorial (n - k) k * Nat.factorial (n - k)

/-- Theorem stating that the number of scheduling methods is 2400 -/
theorem scheduling_methods_count :
  scheduling_methods n d k = 2400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scheduling_methods_count_l463_46384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l463_46333

open Real

-- Define the function f(x) = x ln(x)
noncomputable def f (x : ℝ) : ℝ := x * log x

-- State the theorem
theorem f_inequality (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₁ ≠ x₂) :
  (f x₂ - f x₁) / (x₂ - x₁) < log ((x₁ + x₂) / 2) + 1 := by
  sorry

-- Note: The derivative f'(x) = ln(x) + 1 is represented as (log ((x₁ + x₂) / 2) + 1) in the theorem statement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l463_46333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_theorem_l463_46321

theorem triangle_angle_theorem (a b c : ℝ) (h : b^2 + c^2 - a^2 = Real.sqrt 2 * b * c) :
  Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_theorem_l463_46321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C_D_l463_46327

noncomputable def C : Set ℂ := {z : ℂ | z^3 - 27 = 0}
noncomputable def D : Set ℂ := {z : ℂ | z^3 - 9*z^2 + 27*z - 27 = 0}

noncomputable def distance (z w : ℂ) : ℝ := Complex.abs (z - w)

theorem max_distance_C_D : 
  ∃ (z : ℂ) (w : ℂ), z ∈ C ∧ w ∈ D ∧
  (∀ (z' : ℂ) (w' : ℂ), z' ∈ C → w' ∈ D → distance z w ≥ distance z' w') ∧ 
  distance z w = (9 * Real.sqrt 2) / 2 :=
sorry

#check max_distance_C_D

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C_D_l463_46327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_with_12_divisors_l463_46346

theorem smallest_integer_with_12_divisors : 
  ∃ n : ℕ, (Finset.card (Nat.divisors n) = 12 ∧ ∀ m : ℕ, Finset.card (Nat.divisors m) = 12 → n ≤ m) ∧ n = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_with_12_divisors_l463_46346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_after_exclusion_l463_46330

theorem class_average_after_exclusion (total_students : ℕ) (total_average : ℚ)
  (excluded_students : ℕ) (excluded_average : ℚ) :
  total_students = 20 →
  total_average = 80 →
  excluded_students = 5 →
  excluded_average = 50 →
  (total_students - excluded_students : ℚ) * 
    ((total_students * total_average - excluded_students * excluded_average) / 
     (total_students - excluded_students)) = 90 * (total_students - excluded_students) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_after_exclusion_l463_46330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_true_l463_46389

-- Define the propositions
def prop1 : Prop := (¬ ∃ x : ℝ, x^2 + x - 1 < 0) ↔ (∀ x : ℝ, x^2 + x - 1 > 0)

def prop2 : Prop := 
  (∀ m : ℝ, m = -1 → (∀ x y : ℝ, m*x + (2*m-1)*y + 1 = 0 ↔ 3*x + m*y + 3 = 0)) ∧
  (∃ m : ℝ, m ≠ -1 ∧ (∀ x y : ℝ, m*x + (2*m-1)*y + 1 = 0 ↔ 3*x + m*y + 3 = 0))

noncomputable def prop3 : Prop := 
  (∀ x y : ℝ, Real.sin x ≠ Real.sin y → x ≠ y) ∧
  (∃ x y : ℝ, x ≠ y ∧ Real.sin x = Real.sin y)

def prop4 : Prop := 
  (∀ f : ℝ → ℝ, (∀ x : ℝ, f (x+1) > f x) ↔ (∀ x y : ℝ, x < y → f x < f y))

-- The main theorem
theorem exactly_two_true : 
  (prop1 = false ∧ prop2 = true ∧ prop3 = true ∧ prop4 = false) ∨
  (prop1 = false ∧ prop2 = true ∧ prop3 = false ∧ prop4 = true) ∨
  (prop1 = false ∧ prop2 = false ∧ prop3 = true ∧ prop4 = true) ∨
  (prop1 = true ∧ prop2 = true ∧ prop3 = false ∧ prop4 = false) ∨
  (prop1 = true ∧ prop2 = false ∧ prop3 = true ∧ prop4 = false) ∨
  (prop1 = true ∧ prop2 = false ∧ prop3 = false ∧ prop4 = true) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_true_l463_46389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subset_sum_length_exceeds_one_l463_46309

-- Define a vector in 2D plane
def Vector2D := ℝ × ℝ

-- Define the length of a vector
noncomputable def length (v : Vector2D) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Define the sum of vectors
def vectorSum (vs : List Vector2D) : Vector2D :=
  vs.foldl (fun acc v => (acc.1 + v.1, acc.2 + v.2)) (0, 0)

theorem vector_subset_sum_length_exceeds_one 
  (vs : List Vector2D) 
  (h : (vs.map length).sum = 4) : 
  ∃ (subset : List Vector2D), subset ⊆ vs ∧ subset ≠ [] ∧ length (vectorSum subset) > 1 := by
  sorry

#check vector_subset_sum_length_exceeds_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subset_sum_length_exceeds_one_l463_46309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_firetruck_reachable_area_l463_46316

/-- Represents the speed of the firetruck in miles per hour -/
structure FiretruckSpeed where
  roadSpeed : ℝ
  sandSpeed : ℝ

/-- Represents the time limit in hours -/
noncomputable def timeLimit : ℝ := 8 / 60

/-- Calculates the distance traveled on road given the time limit -/
noncomputable def roadDistance (speed : FiretruckSpeed) : ℝ :=
  speed.roadSpeed * timeLimit

/-- Calculates the distance traveled on sand given the time limit -/
noncomputable def sandDistance (speed : FiretruckSpeed) : ℝ :=
  speed.sandSpeed * timeLimit

/-- Theorem stating the area of the region reachable by the firetruck -/
theorem firetruck_reachable_area (speed : FiretruckSpeed)
  (h1 : speed.roadSpeed = 60)
  (h2 : speed.sandSpeed = 10) :
  (4 * (((roadDistance speed) / 2) ^ 2 + 2 * (π / 4 * (sandDistance speed) ^ 2))) = 6678 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_firetruck_reachable_area_l463_46316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neg_i_equals_neg_one_plus_i_l463_46358

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the function g
noncomputable def g (x : ℂ) : ℂ := (x^3 - x) / (x - 1)

-- State the theorem
theorem g_neg_i_equals_neg_one_plus_i : g (-i) = -1 + i := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neg_i_equals_neg_one_plus_i_l463_46358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_distance_l463_46324

/-- Represents a participant in the race -/
structure Participant where
  speed : ℚ
  distance : ℚ

/-- Represents the race state -/
structure RaceState where
  participants : Fin 4 → Participant
  time : ℚ

def total_distance (state : RaceState) : ℚ :=
  Finset.sum (Finset.range 4) (fun i => (state.participants i).distance)

def race_length : ℚ := 100

theorem petya_distance (initial_state final_state : RaceState) 
  (h1 : initial_state.time = 12)
  (h2 : total_distance initial_state = 288)
  (h3 : final_state.time > initial_state.time)
  (h4 : (final_state.participants 0).distance = race_length)
  (h5 : total_distance final_state = 3 * race_length + 60)
  (h6 : ∀ i, (initial_state.participants i).speed = (final_state.participants i).speed) :
  (initial_state.participants 0).distance = 80 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_distance_l463_46324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bounded_by_curves_l463_46383

open Real MeasureTheory

/-- The area of the region bounded by y = x^2 * sqrt(16 - x^2), y = 0, and 0 ≤ x ≤ 4 is 16π. -/
theorem area_bounded_by_curves : 
  ∫ x in Set.Icc 0 4, x^2 * Real.sqrt (16 - x^2) = 16 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bounded_by_curves_l463_46383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_tangent_line_parallel_f_inequality_l463_46397

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^3 * (3 * Real.log x - 1)

-- Statement 1: The minimum value of f(x) is -1
theorem f_minimum_value : ∃ x₀ : ℝ, ∀ x : ℝ, x > 0 → f x ≥ f x₀ ∧ f x₀ = -1 :=
sorry

-- Statement 2: If the tangent line at (m, f(m)) is parallel to y = 9e^2x - 1, then f(m) = 2e^3
theorem tangent_line_parallel (m : ℝ) :
  (∃ k : ℝ, ∀ x, f m + (deriv f m) * (x - m) = 9 * Real.exp 2 * x + k) →
  f m = 2 * Real.exp 3 :=
sorry

-- Statement 3: f(ln(3e/2)) < f(3/2) < f(log_2(3))
theorem f_inequality :
  f (Real.log (3 * Real.exp 1 / 2)) < f (3/2) ∧ f (3/2) < f (Real.log 3 / Real.log 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_tangent_line_parallel_f_inequality_l463_46397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_chord_l463_46365

-- Define the circle
def myCircle (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 3 = 0

-- Define the line
def myLine (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the point P
def P : ℝ × ℝ := (0, 1)

-- Theorem statement
theorem longest_chord :
  (∀ x y, myCircle x y → myLine x y → (x, y) = P ∨ (x - 1)^2 + y^2 = 2) ∧
  (∀ x y, myCircle x y → myLine x y → ∃ x' y', myCircle x' y' ∧ myLine x' y' ∧ (x - x')^2 + (y - y')^2 ≤ 4) ∧
  myLine P.1 P.2 ∧
  myCircle P.1 P.2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_chord_l463_46365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_divisibility_l463_46302

theorem power_of_two_divisibility (n a b : ℕ) : 
  (2 : ℕ)^n = 10*a + b → b < 10 → n > 3 → ∃ k : ℕ, a*b = 6*k :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_divisibility_l463_46302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_1_simplify_expression_2_l463_46345

-- Expression 1
theorem simplify_expression_1 (a : ℝ) : 2*(a-1)-(2*a-3)+3 = 4 := by
  ring

-- Expression 2
theorem simplify_expression_2 (x : ℝ) : 3*x^2-(7*x-(4*x-3)-2*x^2) = 5*x^2 - 3*x - 3 := by
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_1_simplify_expression_2_l463_46345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_table_cost_price_l463_46326

/-- Given a selling price and a markup percentage, calculate the cost price. -/
noncomputable def cost_price (selling_price : ℝ) (markup_percentage : ℝ) : ℝ :=
  selling_price / (1 + markup_percentage)

/-- Theorem: The cost price of an item with a selling price of 3000 and a 20% markup is 2500. -/
theorem computer_table_cost_price :
  cost_price 3000 0.20 = 2500 := by
  -- Unfold the definition of cost_price
  unfold cost_price
  -- Simplify the expression
  simp
  -- Check that the result is equal to 2500
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_table_cost_price_l463_46326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l463_46319

open Real

-- Define the function (marked as noncomputable due to dependency on Real)
noncomputable def f (x : ℝ) : ℝ := tan (π / 2 * x - π / 3)

-- State the theorem
theorem min_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l463_46319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_k_geq_one_l463_46367

-- Define the functions f and g as noncomputable
noncomputable def f (x : ℝ) : ℝ := (Real.exp 2 * x^2 + 1) / x
noncomputable def g (x : ℝ) : ℝ := (Real.exp 2 * x) / Real.exp x

-- State the theorem
theorem inequality_implies_k_geq_one :
  ∀ (k : ℝ), k > 0 →
  (∀ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 → g x₁ / k ≤ f x₂ / (k + 1)) →
  k ≥ 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_k_geq_one_l463_46367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l463_46396

/-- An arithmetic sequence {a_n} -/
def a : ℕ → ℝ := sorry

/-- A geometric sequence {b_n} with positive terms -/
def b : ℕ → ℝ := sorry

/-- Sum of the first n terms of the geometric sequence {b_n} -/
def T : ℕ → ℝ := sorry

/-- Theorem stating the properties of the sequences and their relation -/
theorem sequence_properties :
  (a 5 = 8) →
  (a 7 = 12) →
  (∀ n : ℕ, b n > 0) →
  (b 3 = a 3) →
  (T 2 = 3) →
  ((∀ n : ℕ, a n = 2 * n - 2) ∧ (∀ n : ℕ, T n = 2^n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l463_46396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_interval_l463_46388

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3) + Real.cos (2 * x - Real.pi / 6)

theorem monotonically_decreasing_interval (k : ℤ) :
  ∀ x ∈ Set.Icc (Real.pi / 12 + k * Real.pi) (k * Real.pi + 7 * Real.pi / 12),
    ∀ y ∈ Set.Icc (Real.pi / 12 + k * Real.pi) (k * Real.pi + 7 * Real.pi / 12),
      x ≤ y → f x ≥ f y := by
  sorry

#check monotonically_decreasing_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_interval_l463_46388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_has_four_digits_l463_46304

/-- A digit is a natural number from 1 to 9 -/
def Digit : Type := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

/-- Convert a two-digit number represented by two digits to a natural number -/
def twoDigitToNat (tens : Digit) (ones : ℕ) : ℕ := tens.val * 10 + ones

/-- The sum of the four numbers in the problem -/
def problemSum (C D : Digit) : ℕ := 7654 + twoDigitToNat C 7 + twoDigitToNat D 9 + 81

/-- The number of digits in a natural number -/
def numDigits (n : ℕ) : ℕ := 
  if n = 0 then 1 else Nat.log n 10 + 1

theorem sum_has_four_digits (C D : Digit) : numDigits (problemSum C D) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_has_four_digits_l463_46304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_range_l463_46357

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def monotone_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem function_inequality_range (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_mono : monotone_increasing_on f (-1) 1)
  (h_bound : f (-1) = -1) :
  {t : ℝ | ∀ x a, x ∈ Set.Icc (-1) 1 → a ∈ Set.Icc (-1) 1 → f x ≤ t^2 + 2*a*t + 1} =
  Set.Iic (-2) ∪ {0} ∪ Set.Ici 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_range_l463_46357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_always_odd_l463_46385

theorem expression_always_odd (a b c : ℕ) (ha : Even a) (hc : Even c) (hpos : 0 < a ∧ 0 < b ∧ 0 < c) :
  Odd (3^a + (b+1)^2*c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_always_odd_l463_46385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l463_46315

open Real

/-- The function f(x) defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := (sin x^3 + 2*sin x^2 + 5*sin x + 3*cos x^2 - 12) / (sin x + 2)

/-- The theorem stating the range of f(x) -/
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, sin x ≠ -2 ∧ f x = y) ↔ -10.5 ≤ y ∧ y ≤ 4.5 :=
by
  sorry

#check range_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l463_46315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_area_theorem_l463_46360

/-- A right triangle with sides 8, 6, and 10 -/
structure RightTriangle where
  AB : ℝ
  AC : ℝ
  BC : ℝ
  right_triangle : AB^2 + AC^2 = BC^2
  AB_eq : AB = 8
  AC_eq : AC = 6
  BC_eq : BC = 10

/-- A point randomly placed inside the triangle -/
structure RandomPoint where
  x : ℝ
  y : ℝ
  in_triangle : x ≥ 0 ∧ y ≥ 0 ∧ y ≤ (6 / 8) * x

/-- The probability of the area of triangle PBC being less than one-third of ABC -/
noncomputable def probability_area_less_than_third (t : RightTriangle) (p : RandomPoint) : ℝ := 4 / 15

/-- The main theorem stating the probability -/
theorem probability_area_theorem (t : RightTriangle) (p : RandomPoint) :
  probability_area_less_than_third t p = 4 / 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_area_theorem_l463_46360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_solution_l463_46310

-- Define ω as a complex number satisfying ω² + ω + 1 = 0
noncomputable def ω : ℂ := Complex.exp ((2 * Real.pi * Complex.I) / 3)

-- Define the cubic equation
def cubic_equation (p q x : ℝ) : Prop := x^3 + p * x + q = 0

-- Define the conditions for y and z
def y_z_conditions (p q y z : ℝ) : Prop := -3 * y * z = p ∧ y^3 + z^3 = q

-- Define the roots x₁, x₂, x₃
def x₁ (y z : ℝ) : ℝ := -(y + z)
noncomputable def x₂ (y z : ℝ) : ℂ := -(ω * y + ω^2 * z)
noncomputable def x₃ (y z : ℝ) : ℂ := -(ω^2 * y + ω * z)

-- State the theorem
theorem cubic_solution (p q : ℝ) :
  ∀ y z : ℝ, y_z_conditions p q y z →
  cubic_equation p q (x₁ y z) ∧
  cubic_equation p q (Complex.re (x₂ y z)) ∧
  cubic_equation p q (Complex.re (x₃ y z)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_solution_l463_46310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_to_equation_l463_46387

theorem solutions_to_equation :
  {(x, n) : ℕ+ × ℕ+ | 3 * 2^(x.val) + 4 = n.val^2} =
  {(⟨2, 4⟩ : ℕ+ × ℕ+), (⟨5, 10⟩ : ℕ+ × ℕ+), (⟨6, 14⟩ : ℕ+ × ℕ+)} := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_to_equation_l463_46387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_AOB_l463_46372

-- Define the polar coordinates of points A and B
noncomputable def r_A : ℝ := 6
noncomputable def θ_A : ℝ := Real.pi / 3
noncomputable def r_B : ℝ := 4
noncomputable def θ_B : ℝ := Real.pi / 6

-- Define the area of triangle AOB
noncomputable def area_AOB : ℝ := (1 / 2) * r_A * r_B * Real.sin (θ_B - θ_A)

-- Theorem statement
theorem area_triangle_AOB : area_AOB = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_AOB_l463_46372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_condition_f_decreasing_range_of_m_l463_46369

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := (-2^x - b) / (2^(x+1) + 2)

theorem odd_function_condition (b : ℝ) : 
  (∀ x : ℝ, f b x = -f b (-x)) → b = -1 := by
  sorry

noncomputable def f_final (x : ℝ) : ℝ := (1 - 2^x) / (2^(x+1) + 2)

theorem f_decreasing : 
  ∀ x y : ℝ, x < y → f_final x > f_final y := by
  sorry

theorem range_of_m : 
  ∀ m : ℝ, (∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ f_final x = m) ↔ m ∈ Set.Icc (-1/6) 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_condition_f_decreasing_range_of_m_l463_46369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_and_g_range_l463_46339

noncomputable def f (x : ℝ) : ℝ := Real.sin (x - Real.pi / 3) + Real.sqrt 3 * Real.cos (x - Real.pi / 3)

noncomputable def g (x : ℝ) : ℝ := (1 + Real.sin x) * f x

theorem f_monotone_and_g_range :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), StrictMono f) ∧
  (∀ x ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi), StrictMono f) ∧
  Set.range g = Set.Icc (-1/2) 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_and_g_range_l463_46339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_of_added_numbers_l463_46395

theorem mean_of_added_numbers (original_count : ℕ) (original_mean : ℚ) (new_count : ℕ) (new_mean : ℚ) :
  original_count = 7 →
  original_mean = 45 →
  new_count = 10 →
  new_mean = 58 →
  (new_count * new_mean - original_count * original_mean) / (new_count - original_count) = 265 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_of_added_numbers_l463_46395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_fraction_theorem_l463_46338

/-- The fraction of a project completed by two people in one hour -/
noncomputable def project_fraction (a b : ℝ) : ℝ := 1/a + 1/b

/-- Theorem: The fraction of a project completed by two people in one hour 
    is the sum of their individual work rates -/
theorem project_fraction_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  project_fraction a b = 1/a + 1/b := by
  -- Unfold the definition of project_fraction
  unfold project_fraction
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_fraction_theorem_l463_46338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_l463_46364

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- The conditions on R(x) -/
def satisfiesConditions (R : IntPolynomial) (b : ℤ) : Prop :=
  b > 0 ∧
  R.eval (2 : ℤ) = b ∧
  R.eval (4 : ℤ) = b ∧
  R.eval (6 : ℤ) = b ∧
  R.eval (8 : ℤ) = b ∧
  R.eval (3 : ℤ) = -b ∧
  R.eval (5 : ℤ) = -b ∧
  R.eval (7 : ℤ) = -b ∧
  R.eval (9 : ℤ) = -b

/-- The theorem stating the smallest possible value of b -/
theorem smallest_b : ∀ R : IntPolynomial, ∀ b : ℤ,
  satisfiesConditions R b → b ≥ 315 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_l463_46364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l463_46351

noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

noncomputable def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  -- Expand the definition of g
  simp [g, f]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l463_46351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_non_collinearity_l463_46336

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (m : ℝ) : ℝ × ℝ := (m, 3*m - 2)

theorem vector_non_collinearity (m : ℝ) : 
  (∃ (l μ : ℝ), ∀ c : ℝ × ℝ, ∃! (x y : ℝ), c = (x * vector_a.1 + y * (vector_b m).1, x * vector_a.2 + y * (vector_b m).2)) ↔ 
  (m < 2 ∨ m > 2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_non_collinearity_l463_46336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sine_ratio_range_l463_46391

theorem min_sine_ratio_range (α β γ : Real) (h1 : 0 < α) (h2 : α ≤ β) (h3 : β ≤ γ) 
  (h4 : α + β + γ = π) : 
  1 ≤ min (Real.sin β / Real.sin α) (Real.sin γ / Real.sin β) ∧ 
  min (Real.sin β / Real.sin α) (Real.sin γ / Real.sin β) < (Real.sqrt 5 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sine_ratio_range_l463_46391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_pairs_l463_46349

theorem solution_pairs : 
  ∀ x y : ℝ, 
    Real.sin ((x + y) / 2) = 0 ∧ 
    abs x + abs y = 1 → 
    ((x = 1/2 ∧ y = -1/2) ∨ (x = -1/2 ∧ y = 1/2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_pairs_l463_46349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loop_iterations_specific_loop_iterations_l463_46311

/-- 
For a loop "For I From a To b", where a and b are integers and a ≤ b,
the number of iterations is equal to b - a + 1.
-/
theorem loop_iterations (a b : ℤ) (h : a ≤ b) : 
  (b - a + 1 : ℤ) = b - a + 1 := by
  -- The proof is trivial as we're equating the same expression
  rfl

/--
The number of iterations for the specific loop "For I From 2 To 20"
-/
theorem specific_loop_iterations : 
  (20 - 2 + 1 : ℤ) = 19 := by
  norm_num

#eval 20 - 2 + 1 -- This will output 19

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loop_iterations_specific_loop_iterations_l463_46311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_three_similar_parts_l463_46359

theorem no_three_similar_parts : ¬ ∃ (a b c : ℝ), 
  (a > 0 ∧ b > 0 ∧ c > 0) ∧ 
  (a + b + c = 1) ∧
  (a ≤ Real.sqrt 2 * b) ∧ (a ≤ Real.sqrt 2 * c) ∧
  (b ≤ Real.sqrt 2 * a) ∧ (b ≤ Real.sqrt 2 * c) ∧
  (c ≤ Real.sqrt 2 * a) ∧ (c ≤ Real.sqrt 2 * b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_three_similar_parts_l463_46359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vertex_product_sum_l463_46363

/-- Represents a cube with numbers assigned to its faces -/
structure NumberedCube where
  faces : Fin 6 → Nat
  face_values : ∀ i, faces i ∈ ({1, 3, 4, 6, 8, 9} : Set Nat)

/-- Calculates the sum of products at vertices for a given cube -/
def vertexProductSum (cube : NumberedCube) : Nat :=
  (cube.faces 0 + cube.faces 1) * (cube.faces 2 + cube.faces 3) * (cube.faces 4 + cube.faces 5)

/-- The maximum sum of products at vertices for any valid cube assignment -/
def maxVertexProductSum : Nat := 1100

/-- Theorem stating that the maximum sum of products at vertices is 1100 -/
theorem max_vertex_product_sum :
  ∀ cube : NumberedCube, vertexProductSum cube ≤ maxVertexProductSum := by
  sorry

#check max_vertex_product_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vertex_product_sum_l463_46363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_f_def_f_three_equals_seven_l463_46340

noncomputable def f (x : ℝ) : ℝ := 
  if 0 ≤ x ∧ x ≤ 2 then 4^x + 3/x else 0  -- default value for x outside [0,2]

theorem f_period (x : ℝ) : f (x + 2) = f x := by sorry

theorem f_def (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2) : f x = 4^x + 3/x := by sorry

theorem f_three_equals_seven : f 3 = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_f_def_f_three_equals_seven_l463_46340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_value_l463_46381

-- Define the function f(x) = log_a x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the interval [2, 3]
def interval : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem log_base_value (a : ℝ) (h : a > 0) :
  (∀ x ∈ interval, f a x ≤ f a 3) ∧
  (∀ x ∈ interval, f a 2 ≤ f a x) ∧
  (f a 3 = f a 2 + 1) →
  a = 3/2 ∨ a = 2/3 :=
by
  sorry

#check log_base_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_value_l463_46381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_ride_time_at_high_speed_l463_46317

/-- Represents a cyclist's ride with three speed stages -/
structure CyclistRide where
  totalDistance : ℝ
  totalTime : ℝ
  speedHigh : ℝ
  speedMedium : ℝ
  speedLow : ℝ

/-- Calculates the time spent at high speed given a CyclistRide -/
noncomputable def timeAtHighSpeed (ride : CyclistRide) : ℝ :=
  let x := (ride.totalDistance - ride.speedLow * ride.totalTime) / (ride.speedHigh - ride.speedLow)
  let y := (ride.totalTime * (ride.speedHigh - ride.speedLow) - x * (ride.speedHigh - ride.speedMedium)) / (ride.speedMedium - ride.speedLow)
  x

/-- The main theorem stating that for the given ride parameters, the time at high speed is 2.7 hours -/
theorem alice_ride_time_at_high_speed :
  let ride : CyclistRide := {
    totalDistance := 162,
    totalTime := 9,
    speedHigh := 25,
    speedMedium := 15,
    speedLow := 10
  }
  timeAtHighSpeed ride = 2.7 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_ride_time_at_high_speed_l463_46317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_second_smallest_and_largest_primes_l463_46343

def isPrime (n : ℕ) : Bool := Nat.Prime n

def primesBetween (a b : ℕ) : List ℕ :=
  (List.range (b - a + 1)).map (· + a) |>.filter isPrime

theorem sum_second_smallest_and_largest_primes :
  let primes := primesBetween 1 50
  primes[1]! + primes[primes.length - 2]! = 46 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_second_smallest_and_largest_primes_l463_46343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_servant_salary_l463_46371

/-- Represents the annual salary in rupees -/
def annual_salary : ℝ → Prop := sorry

/-- The number of months the servant worked -/
def months_worked : ℕ := 9

/-- The total number of months in a year -/
def months_in_year : ℕ := 12

/-- The cash amount received by the servant after 9 months -/
def cash_received : ℝ := 45

/-- The value of the turban in rupees -/
def turban_value : ℝ := 90

/-- The total value received by the servant (cash + turban) -/
def total_received : ℝ := cash_received + turban_value

theorem servant_salary :
  annual_salary 60 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_servant_salary_l463_46371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_translation_l463_46366

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_symmetric_origin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem sine_function_translation (A ω φ : ℝ) 
  (h_A : A ≠ 0) 
  (h_ω : ω > 0) 
  (h_φ : -π/2 < φ ∧ φ < π/2) 
  (h_odd : is_odd_function (fun x ↦ A * Real.sin (ω * x + φ)))
  (h_sym : is_symmetric_origin (fun x ↦ A * Real.sin (ω * (x + π/6))))
  : ω = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_translation_l463_46366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_statement_l463_46332

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)

-- State the theorem
theorem incorrect_statement
  (a b : Line) (α : Plane)
  (h1 : perpendicular a b)
  (h2 : parallel_line_plane a α) :
  ¬ (∀ (b : Line), perpendicular_line_plane b α) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_statement_l463_46332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_removed_volume_theorem_l463_46300

/-- The volume of a single tetrahedron removed from a unit cube when slicing it to make regular hexagonal faces -/
noncomputable def single_tetrahedron_volume : ℝ :=
  let x : ℝ := (Real.sqrt 3 - 1) / 2
  let base_area : ℝ := (Real.sqrt 3 / 4) * x^2
  let height : ℝ := 1 - Real.sqrt 2 * x
  (1 / 3) * base_area * height

/-- The total volume of tetrahedra removed from a unit cube to make each face a regular hexagon -/
noncomputable def total_removed_volume : ℝ := 8 * single_tetrahedron_volume

/-- Theorem stating the relationship between the removed volume and the unit cube -/
theorem removed_volume_theorem :
  total_removed_volume < 1 ∧ total_removed_volume > 0 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval total_removed_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_removed_volume_theorem_l463_46300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_infinite_irregular_set_l463_46373

/-- A set A is irregular if for any different elements x and y in A,
    there is no element of the form x + k(y - x) in A different from x and y,
    where k is an integer. -/
def Irregular (A : Set ℤ) : Prop :=
  ∀ x y, x ∈ A → y ∈ A → x ≠ y → ∀ k : ℤ, x + k * (y - x) ∈ A → x + k * (y - x) = x ∨ x + k * (y - x) = y

/-- There exists an infinite irregular subset of the integers. -/
theorem exists_infinite_irregular_set : ∃ A : Set ℤ, Infinite A ∧ Irregular A := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_infinite_irregular_set_l463_46373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximization_l463_46329

/-- Profit function for a manufacturer's promotional event -/
noncomputable def profit (x : ℝ) : ℝ := 80 - (36 / x + x)

/-- Sales volume as a function of promotional expenses -/
noncomputable def sales_volume (x : ℝ) : ℝ := 5 - 2 / x

/-- Sales price as a function of sales volume -/
noncomputable def sales_price (t : ℝ) : ℝ := 4 + 20 / t

/-- Investment cost as a function of sales volume -/
noncomputable def investment_cost (t : ℝ) : ℝ := 10 + 2 * t

/-- Theorem stating the conditions for profit maximization -/
theorem profit_maximization (a : ℝ) (ha : a > 0) :
  let max_profit := if a ≥ 6 then profit 6 else profit a
  ∀ x, 0 < x ∧ x ≤ a → profit x ≤ max_profit := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximization_l463_46329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equality_l463_46370

noncomputable def h (x : ℝ) : ℝ := ((2 * x + 5) / 5) ^ (1/3 : ℝ)

theorem h_equality (x : ℝ) : h (3 * x) = 3 * h x ↔ x = -65/24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equality_l463_46370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_specific_parabola_l463_46322

/-- The focus of a parabola given by y = ax² + bx + c --/
noncomputable def parabola_focus (a b c : ℝ) : ℝ × ℝ :=
  let h := -b / (2 * a)
  let k := c - b^2 / (4 * a)
  (h, k + 1 / (4 * a))

/-- Theorem: The focus of the parabola y = 4x² - 8x - 12 is at (1, -15.9375) --/
theorem focus_of_specific_parabola :
  parabola_focus 4 (-8) (-12) = (1, -15.9375) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_specific_parabola_l463_46322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_theorem_l463_46355

/-- Two men walking towards each other from points R and S -/
structure WalkingMen where
  initial_distance : ℝ
  r_speed : ℝ
  s_initial_speed : ℝ
  s_speed_increase : ℝ

/-- Calculate the distance walked by the man from S in h hours -/
noncomputable def distance_s (w : WalkingMen) (h : ℕ) : ℝ :=
  (h : ℝ) / 2 * (2 * w.s_initial_speed + ((h : ℝ) - 1) * w.s_speed_increase)

/-- The meeting point of the two men -/
noncomputable def meeting_point (w : WalkingMen) : ℝ :=
  w.initial_distance / 2 - (w.r_speed * 10 - distance_s w 10) / 2

/-- Theorem stating that the men meet 10 miles closer to R than S -/
theorem meeting_point_theorem (w : WalkingMen) 
  (h_initial : w.initial_distance = 100)
  (h_r_speed : w.r_speed = 4.5)
  (h_s_initial : w.s_initial_speed = 3.5)
  (h_s_increase : w.s_speed_increase = 0.5) :
  meeting_point w = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_theorem_l463_46355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_240_l463_46379

/-- A rectangle inscribed in a semicircle -/
structure InscribedRectangle where
  /-- Length of side DA of the rectangle -/
  da : ℝ
  /-- Length of segment FD (equal to AE) -/
  fd : ℝ
  /-- da is positive -/
  da_pos : 0 < da
  /-- fd is positive -/
  fd_pos : 0 < fd

/-- Calculate the area of the inscribed rectangle -/
noncomputable def area (r : InscribedRectangle) : ℝ :=
  let ef := r.fd + r.da + r.fd  -- Diameter of the semicircle
  let oc := ef / 2  -- Radius of the semicircle
  let od := r.da / 2  -- Half of DA
  let cd := Real.sqrt (oc^2 - od^2)  -- Length of CD using Pythagorean theorem
  r.da * cd

/-- Theorem stating that the area of the specific inscribed rectangle is 240 -/
theorem area_is_240 (r : InscribedRectangle) (h1 : r.da = 16) (h2 : r.fd = 9) :
  area r = 240 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_240_l463_46379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_done_is_46_l463_46352

-- Define the force function
noncomputable def F (x : ℝ) : ℝ :=
  if x ≤ 2 then 10 else 3 * x + 4

-- State the theorem
theorem work_done_is_46 :
  ∫ x in (0 : ℝ)..(4 : ℝ), F x = 46 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_done_is_46_l463_46352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_best_approximate_root_l463_46314

def f (x : ℝ) : ℝ := x^2 + 3*x - 5

def data_points : List (ℝ × ℝ) := [
  (1, -1),
  (1.1, -0.49),
  (1.2, 0.04),
  (1.3, 0.59),
  (1.4, 1.16)
]

theorem best_approximate_root :
  ∀ x ∈ [1, 1.1, 1.2, 1.3, 1.4],
    abs (f 1.2) ≤ abs (f x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_best_approximate_root_l463_46314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_alpha_minus_beta_l463_46347

theorem two_alpha_minus_beta (α β : ℝ) : 
  α ∈ Set.Ioo 0 π → 
  β ∈ Set.Ioo 0 π → 
  Real.tan (α - β) = 1/2 → 
  Real.tan β = -1/7 → 
  2*α - β = -3*π/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_alpha_minus_beta_l463_46347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_ratio_l463_46378

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where a = 4, b = 5, and c = 6, prove that sin(A+B) / sin(2A) = 1 -/
theorem triangle_angle_ratio (A B C : Real) (a b c : Real) 
    (h_triangle : (A + B + C) = Real.pi)
    (h_positive : 0 < A ∧ 0 < B ∧ 0 < C)
    (h_sides : a = 4 ∧ b = 5 ∧ c = 6)
    (h_law_of_sines : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C) :
    Real.sin (A + B) / Real.sin (2 * A) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_ratio_l463_46378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_cow_drinking_time_l463_46344

/-- Represents the scenario of cows drinking from a spring-fed pond -/
structure PondScenario where
  /-- Total volume of the pond -/
  pond_volume : ℝ
  /-- Daily water consumption by one cow -/
  cow_consumption : ℝ
  /-- Daily water inflow from the spring -/
  spring_inflow : ℝ

/-- The time it takes for a given number of cows to drink the pond -/
noncomputable def drinking_time (scenario : PondScenario) (num_cows : ℝ) : ℝ :=
  (scenario.pond_volume + scenario.spring_inflow * num_cows) / (scenario.cow_consumption * num_cows)

/-- Theorem stating that one cow will take approximately 75 days to drink the pond -/
theorem one_cow_drinking_time :
  ∀ (scenario : PondScenario),
    drinking_time scenario 17 = 3 →
    drinking_time scenario 2 = 30 →
    ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |drinking_time scenario 1 - 75| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_cow_drinking_time_l463_46344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_period_l463_46312

/-- Calculates the period for compound interest -/
noncomputable def calculate_period (principal : ℝ) (rate : ℝ) (compound_interest : ℝ) : ℝ :=
  let total := principal + compound_interest
  let annual_factor := 1 + rate
  Real.log (total / principal) / Real.log annual_factor

/-- The problem statement -/
theorem compound_interest_period :
  let principal : ℝ := 8000
  let rate : ℝ := 0.15
  let compound_interest : ℝ := 3109
  let period := calculate_period principal rate compound_interest
  ⌊period⌋ = 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_period_l463_46312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_in_terms_of_f_l463_46399

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the function g
def g : ℝ → ℝ := sorry

-- State the theorem
theorem g_in_terms_of_f :
  ∀ x : ℝ, g x = f (4 - x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_in_terms_of_f_l463_46399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_3_power_4_l463_46308

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- State the conditions
axiom fg_condition : ∀ x : ℝ, x ≥ 1 → f (g x) = x^3
axiom gf_condition : ∀ x : ℝ, x ≥ 1 → g (f x) = x^4
axiom g_81 : g 81 = 3

-- State the theorem to be proved
theorem g_3_power_4 : (g 3)^4 = 3^(3/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_3_power_4_l463_46308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_i_part_ii_l463_46393

/-- The function f(x) = x ln(x) -/
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

/-- The function g(x) = k(x-1) -/
def g (k : ℝ) (x : ℝ) : ℝ := k * (x - 1)

/-- Part I: f(x) ≥ g(x) holds for all x > 0 if and only if k = 1 -/
theorem part_i (k : ℝ) : 
  (∀ x > 0, f x ≥ g k x) ↔ k = 1 := by sorry

/-- Part II: There is no k > 1 satisfying the conditions -/
theorem part_ii : 
  ¬∃ k, k > 1 ∧ ∃ x₀ x₁, x₀ > 1 ∧ x₁ > 1 ∧
    f x₁ = g k x₁ ∧ 
    (deriv f x₀ = deriv (g k) x₀) ∧ 
    x₁ / x₀ = k := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_i_part_ii_l463_46393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_topping_cost_is_fifty_cents_l463_46307

/-- The cost of a sundae with ice cream and toppings -/
def sundae_cost (ice_cream_cost : ℝ) (topping_cost : ℝ) (num_toppings : ℕ) : ℝ :=
  ice_cream_cost + topping_cost * (num_toppings : ℝ)

/-- Theorem: The cost of each topping is $0.50 -/
theorem topping_cost_is_fifty_cents
  (ice_cream_cost : ℝ)
  (total_cost : ℝ)
  (num_toppings : ℕ)
  (h1 : ice_cream_cost = 2)
  (h2 : total_cost = 7)
  (h3 : num_toppings = 10)
  (h4 : sundae_cost ice_cream_cost (0.5 : ℝ) num_toppings = total_cost) :
  0.5 = (total_cost - ice_cream_cost) / (num_toppings : ℝ) := by
  sorry

#check topping_cost_is_fifty_cents

end NUMINAMATH_CALUDE_ERRORFEEDBACK_topping_cost_is_fifty_cents_l463_46307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_B_is_false_l463_46331

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define diagonals of a quadrilateral
def diagonals (q : Quadrilateral) : ((ℝ × ℝ) × (ℝ × ℝ)) × ((ℝ × ℝ) × (ℝ × ℝ)) :=
  ((q.vertices 0, q.vertices 2), (q.vertices 1, q.vertices 3))

-- Define equality of diagonals
def equal_diagonals (q : Quadrilateral) : Prop :=
  let ((v0, v2), (v1, v3)) := diagonals q
  (v0.1 - v2.1)^2 + (v0.2 - v2.2)^2 = (v1.1 - v3.1)^2 + (v1.2 - v3.2)^2

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  ((q.vertices 0 = (0, 0) ∧ q.vertices 1 = (a, 0) ∧ q.vertices 2 = (a, b) ∧ q.vertices 3 = (0, b)) ∨
   (q.vertices 0 = (0, 0) ∧ q.vertices 1 = (0, b) ∧ q.vertices 2 = (a, b) ∧ q.vertices 3 = (a, 0)))

-- Statement B is false
theorem statement_B_is_false : ¬(∀ q : Quadrilateral, equal_diagonals q → is_rectangle q) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_B_is_false_l463_46331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_transverse_axis_length_l463_46354

/-- Given a hyperbola and a parabola with specific properties, prove the length of the hyperbola's transverse axis -/
theorem hyperbola_transverse_axis_length 
  (a b : ℝ) 
  (ha : a > 0)
  (hb : b > 0)
  (h_hyperbola : ∀ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1)
  (h_eccentricity : Real.sqrt 2 = (Real.sqrt (a^2 + b^2)) / a)
  (h_parabola : ∀ x y : ℝ, x^2 = -4 * Real.sqrt 3 * y)
  (h_triangle_area : Real.sqrt 3 = abs (Real.sqrt (3 - a^2) * Real.sqrt 3)) :
  2 * Real.sqrt 2 = 2 * a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_transverse_axis_length_l463_46354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l463_46320

theorem cos_beta_value (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.cos α = 4/5) (h4 : Real.tan (α - β) = -1/3) : Real.cos β = 9 * Real.sqrt 10 / 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l463_46320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mr_wise_stock_worth_l463_46334

/-- Calculates the total worth of stock given the number of shares and prices -/
def total_stock_worth (shares1 shares2 : ℕ) (price1 price2 : ℚ) : ℚ :=
  shares1 * price1 + shares2 * price2

/-- Theorem stating the total worth of Mr. Wise's stock purchase -/
theorem mr_wise_stock_worth :
  ∃ (shares1 shares2 : ℕ) (price1 price2 : ℚ),
    shares1 + shares2 = 450 ∧
    (shares1 = 400 ∨ shares2 = 400) ∧
    (price1 = 3 ∧ price2 = 9/2 ∨ price1 = 9/2 ∧ price2 = 3) ∧
    total_stock_worth shares1 shares2 price1 price2 = 1425 := by
  use 400, 50, 3, 9/2
  constructor
  · exact rfl
  constructor
  · left; rfl
  constructor
  · left; constructor <;> rfl
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mr_wise_stock_worth_l463_46334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meal_cost_solution_l463_46361

/-- Represents the cost calculation for a meal --/
def meal_cost (x : ℝ) : ℝ :=
  x + 0.096 * x + 0.18 * x + 5

/-- Theorem stating the correct meal cost before tax, tip, and service charge --/
theorem meal_cost_solution :
  ∃ (x : ℝ), x > 0 ∧ meal_cost x = 40 ∧ |x - 27.43| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meal_cost_solution_l463_46361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_age_of_staff_l463_46368

theorem combined_age_of_staff (num_students : ℕ) (avg_students : ℝ) 
  (avg_with_teachers : ℝ) (avg_with_all : ℝ) 
  (h1 : num_students = 30)
  (h2 : avg_students = 18)
  (h3 : avg_with_teachers = 19)
  (h4 : avg_with_all = 20)
  : (num_students + 2) * avg_with_teachers - num_students * avg_students +
    (num_students + 4) * avg_with_all - (num_students + 2) * avg_with_teachers = 140 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_age_of_staff_l463_46368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_divisors_of_factorial_product_l463_46301

/-- The product of factorials from 1 to n -/
def factorialProduct (n : ℕ) : ℕ := (List.range n).map Nat.factorial |>.prod

/-- The number of perfect square divisors of a natural number -/
def numPerfectSquareDivisors (n : ℕ) : ℕ := sorry

theorem perfect_square_divisors_of_factorial_product :
  numPerfectSquareDivisors (factorialProduct 10) = 2592 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_divisors_of_factorial_product_l463_46301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_l463_46303

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 6)^2 = 20 ∨ (x + 7)^2 + (y - 6)^2 = 80

-- Define the line that the circle is tangent to
def tangent_line (x y : ℝ) : Prop :=
  x - 2*y - 1 = 0

-- Define point P on the circle
def point_on_circle (x y : ℝ) : Prop :=
  circle_C x y

-- Define point Q
def point_Q : ℝ × ℝ := (-3, -6)

-- Define the midpoint M of PQ
def midpoint_M (x y x₀ y₀ : ℝ) : Prop :=
  x = (x₀ + point_Q.1) / 2 ∧ y = (y₀ + point_Q.2) / 2

-- Theorem statement
theorem midpoint_trajectory :
  ∀ x y x₀ y₀ : ℝ,
  point_on_circle x₀ y₀ →
  midpoint_M x y x₀ y₀ →
  (x^2 + y^2 = 5 ∨ (x + 5)^2 + y^2 = 20) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_l463_46303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_basket_size_l463_46318

def basket_sizes : List Nat := [4, 6, 12, 13, 22, 29]
def total_eggs : Nat := 86

theorem correct_basket_size :
  ∃! n, n ∈ basket_sizes ∧ (total_eggs - n) % 3 = 0 ∧ n = 29 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_basket_size_l463_46318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_distance_between_spheres_l463_46398

/-- The largest possible distance between two points on given spheres -/
theorem largest_distance_between_spheres :
  let center1 : ℝ × ℝ × ℝ := (-5, -15, 10)
  let center2 : ℝ × ℝ × ℝ := (15, 12, -21)
  let radius1 : ℝ := 23
  let radius2 : ℝ := 91
  let center_distance : ℝ := Real.sqrt ((15 - (-5))^2 + (12 - (-15))^2 + ((-21) - 10)^2)
  ∀ (p1 p2 : ℝ × ℝ × ℝ),
    ((p1.1 - center1.1)^2 + (p1.2.1 - center1.2.1)^2 + (p1.2.2 - center1.2.2)^2 = radius1^2) →
    ((p2.1 - center2.1)^2 + (p2.2.1 - center2.2.1)^2 + (p2.2.2 - center2.2.2)^2 = radius2^2) →
    Real.sqrt ((p1.1 - p2.1)^2 + (p1.2.1 - p2.2.1)^2 + (p1.2.2 - p2.2.2)^2) ≤ radius1 + center_distance + radius2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_distance_between_spheres_l463_46398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l463_46306

theorem simplify_trig_expression :
  (Real.sin (40 * π / 180) - Real.sin (10 * π / 180)) /
  (Real.cos (40 * π / 180) - Real.cos (10 * π / 180)) =
  -(Real.cos (25 * π / 180) / Real.sin (25 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l463_46306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asha_brother_loan_l463_46392

/-- The amount of money Asha borrowed from her brother -/
noncomputable def brother_loan : ℚ := 20

/-- The amount of money Asha borrowed from her father -/
noncomputable def father_loan : ℚ := 40

/-- The amount of money Asha borrowed from her mother -/
noncomputable def mother_loan : ℚ := 30

/-- The amount of money Asha received as a gift from her granny -/
noncomputable def granny_gift : ℚ := 70

/-- The amount of money Asha had in savings -/
noncomputable def savings : ℚ := 100

/-- The fraction of total money Asha spent -/
noncomputable def spent_fraction : ℚ := 3/4

/-- The amount of money Asha was left with after spending -/
noncomputable def remaining_money : ℚ := 65

theorem asha_brother_loan : 
  brother_loan = 20 ∧
  (brother_loan + father_loan + mother_loan + granny_gift + savings) * (1 - spent_fraction) = remaining_money :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asha_brother_loan_l463_46392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_hexagon_perimeter_l463_46341

/-- An equilateral hexagon with specific interior angles and area -/
structure SpecialHexagon where
  -- The side length of the hexagon
  side : ℝ
  -- The hexagon is equilateral
  is_equilateral : True
  -- Three nonadjacent interior angles measure 45°
  special_angles : True
  -- The enclosed area of the hexagon is 12
  area : side^2 * (3 / 2 + Real.sqrt 3 / 2) = 12

/-- The perimeter of the special hexagon is 12√(6 - 2√3) -/
theorem special_hexagon_perimeter (h : SpecialHexagon) : 
  6 * h.side = 12 * Real.sqrt (6 - 2 * Real.sqrt 3) := by
  sorry

#check special_hexagon_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_hexagon_perimeter_l463_46341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l463_46382

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  B = π / 3 →
  Real.cos A = 4 / 5 →
  b = Real.sqrt 3 →
  (Real.sin C = (3 + 4 * Real.sqrt 3) / 10) ∧
  (1 / 2 * a * b * Real.sin C = (36 + 9 * Real.sqrt 3) / 50) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l463_46382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_simplification_l463_46356

theorem cube_root_simplification : (2^6 * 3^3 * 11^3 : ℝ)^(1/3) = 132 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_simplification_l463_46356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_chord_length_squared_l463_46375

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents a chord that is a common external tangent to two circles -/
structure CommonTangentChord where
  circle1 : Circle
  circle2 : Circle
  largeCircle : Circle
  length : ℝ

theorem common_tangent_chord_length_squared 
  (c4 : Circle) 
  (c8 : Circle) 
  (c12 : Circle) 
  (h1 : c4.radius = 4) 
  (h2 : c8.radius = 8) 
  (h3 : c12.radius = 12) 
  (h4 : c4.radius + c8.radius = c12.radius) 
  (chord : CommonTangentChord) 
  (h5 : chord.circle1 = c4) 
  (h6 : chord.circle2 = c8) 
  (h7 : chord.largeCircle = c12) : 
  (chord.length)^2 = 4160/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_chord_length_squared_l463_46375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stewart_farm_sheep_count_l463_46328

theorem stewart_farm_sheep_count : ℕ := by
  let sheep_to_horse_ratio : ℚ := 7 / 7
  let horse_food_per_day : ℕ := 230
  let total_horse_food_per_day : ℕ := 12880
  let num_horses : ℕ := total_horse_food_per_day / horse_food_per_day
  let num_sheep : ℕ := num_horses -- Because the ratio is 1:1
  have h : num_sheep = 56 := by sorry
  exact num_sheep

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stewart_farm_sheep_count_l463_46328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_volume_l463_46313

noncomputable def container_side : ℝ := 12
noncomputable def water_fraction : ℝ := 1/3
noncomputable def ice_cube_side : ℝ := 1.5
def num_ice_cubes : ℕ := 15

noncomputable def container_volume : ℝ := container_side ^ 3
noncomputable def water_volume : ℝ := water_fraction * container_volume
noncomputable def ice_cube_volume : ℝ := ice_cube_side ^ 3
noncomputable def total_ice_volume : ℝ := (num_ice_cubes : ℝ) * ice_cube_volume

theorem unoccupied_volume :
  container_volume - (water_volume + total_ice_volume) = 1101.375 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_volume_l463_46313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_expression_approx_l463_46353

-- Define the expression
def expression : ℝ := 81 * (16 ^ (1/3)) * (16 ^ (1/4))

-- State the theorem
theorem log_expression_approx : 
  ∃ ε > 0, |((Real.log expression) / (Real.log 4)) - 4.3| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_expression_approx_l463_46353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_positive_derivative_increasing_l463_46350

theorem continuous_positive_derivative_increasing 
  (f : ℝ → ℝ) 
  (h_cont : ContinuousOn f (Set.Icc 0 1))
  (h_deriv : ∀ x ∈ Set.Ioo 0 1, HasDerivAt f (deriv f x) x ∧ deriv f x > 0) :
  f 1 > f 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_positive_derivative_increasing_l463_46350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_concurrent_diagonals_l463_46348

theorem parallelogram_concurrent_diagonals 
  (A B C D O M N P Q : ℝ × ℝ) : 
  (∃ a b : ℝ, B - A = (a, 0) ∧ D - A = (0, b)) →  -- ABCD is a parallelogram
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ O = t • B + (1 - t) • D) →  -- O is on BD
  (∃ k : ℝ, M = A + k • (D - A) ∧ N = M + (B - A)) →  -- MN parallel to AB, M on AD
  (∃ l : ℝ, Q = A + l • (B - A) ∧ P = Q + (D - A)) →  -- PQ parallel to AD, Q on AB
  ∃ u v : ℝ, A + u • (O - A) = B + v • (P - B) ∧ 
           A + u • (O - A) = D + v • (N - D) :=  -- K is on AO, BP, and DN
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_concurrent_diagonals_l463_46348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_for_period_one_l463_46335

noncomputable def f (k : ℕ+) (x : ℝ) : ℝ := 
  (Real.sin (k.val * x / 10))^4 + (Real.cos (k.val * x / 10))^4

theorem min_k_for_period_one :
  ∀ k : ℕ+, (∀ a : ℝ, {f k x | x ∈ Set.Ioo a (a + 1)} = Set.range (f k)) ↔ k ≥ 16 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_for_period_one_l463_46335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_obtuse_if_sin_A_cos_C_negative_l463_46337

/-- If ABC is a triangle and sin A cos C < 0, then ABC is an obtuse triangle -/
theorem triangle_obtuse_if_sin_A_cos_C_negative 
  (A B C : ℝ) 
  (triangle_ABC : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (h : Real.sin A * Real.cos C < 0) : 
  π / 2 < C ∧ C < π :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_obtuse_if_sin_A_cos_C_negative_l463_46337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_meet_time_l463_46394

/-- The length of the track in meters -/
def track_length : ℝ := 400

/-- The speeds of the three runners in meters per second -/
def v₁ : ℝ := 3.8
def v₂ : ℝ := 4.0
def v₃ : ℝ := 4.2

/-- The time in seconds it takes for the runners to meet again -/
def meeting_time : ℝ := 2000

theorem runners_meet_time :
  let d₁ := v₁ * meeting_time
  let d₂ := v₂ * meeting_time
  let d₃ := v₃ * meeting_time
  (d₁ - d₂) % track_length = 0 ∧
  (d₃ - d₂) % track_length = 0 ∧
  ⌊d₁ / track_length⌋ = ⌊d₂ / track_length⌋ ∧
  ⌊d₂ / track_length⌋ = ⌊d₃ / track_length⌋ :=
by
  -- Proof goes here
  sorry

#check runners_meet_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_meet_time_l463_46394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_y_l463_46325

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3 + 2

-- Define the function y
noncomputable def y (x : ℝ) : ℝ := (f x)^2 + f (x^2)

-- Theorem statement
theorem max_value_y :
  ∀ x : ℝ, 1 ≤ x ∧ x ≤ 9 → y x ≤ 13 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_y_l463_46325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_year_interest_rate_is_4_percent_l463_46305

noncomputable def first_year_interest_rate 
  (principal : ℝ) 
  (second_year_rate : ℝ) 
  (final_amount : ℝ) : ℝ :=
  (final_amount - principal * (1 + second_year_rate)) / 
  (principal * (1 + second_year_rate))

theorem first_year_interest_rate_is_4_percent 
  (principal : ℝ)
  (second_year_rate : ℝ)
  (final_amount : ℝ)
  (h1 : principal = 9000)
  (h2 : second_year_rate = 0.05)
  (h3 : final_amount = 9828) :
  first_year_interest_rate principal second_year_rate final_amount = 0.04 := by
  sorry

#check first_year_interest_rate_is_4_percent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_year_interest_rate_is_4_percent_l463_46305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l463_46386

noncomputable section

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the conditions
def conditions (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ f a b c 0 = 1 ∧ f a b c 1 = 3 ∧
  ∃ x, ∀ y, f a b c y ≥ f a b c x ∧ f a b c x = 3/4

-- Define the maximum value function g
noncomputable def g (a : ℝ) : ℝ :=
  if a > -2/3 ∧ a ≠ 0 then 2*a + 5
  else if -2 ≤ a ∧ a ≤ -2/3 then 1 - (2-a)^2/(4*a)
  else 3

-- Theorem statement
theorem f_and_g_properties (a b c : ℝ) (h : conditions a b c) :
  ((f a b c = f 4 (-2) 1) ∨ (f a b c = f 1 1 1)) ∧
  (∀ x ∈ Set.Icc 1 2, f a b c x ≤ g a) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l463_46386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_squares_for_specific_rectangle_l463_46342

/-- Represents the dimensions of a rectangle in centimeters -/
structure RectangleDimensions where
  length : Rat
  width : Rat

/-- Calculates the minimum number of squares required to cover a rectangle -/
def min_squares_to_cover (rect : RectangleDimensions) : Nat :=
  let m := (rect.length * 2).num
  let n := (rect.width * 3).num
  (m * n / m.gcd n).toNat

/-- The main theorem stating the minimum number of squares required -/
theorem min_squares_for_specific_rectangle :
  min_squares_to_cover { length := 121/2, width := 143/3 } = 858 := by
  sorry

#eval min_squares_to_cover { length := 121/2, width := 143/3 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_squares_for_specific_rectangle_l463_46342

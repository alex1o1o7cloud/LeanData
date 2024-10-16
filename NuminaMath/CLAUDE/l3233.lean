import Mathlib

namespace NUMINAMATH_CALUDE_no_solution_for_divisibility_l3233_323397

theorem no_solution_for_divisibility (n : ℕ) : n ≥ 1 → ¬(9 ∣ (7^n + n^3)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_divisibility_l3233_323397


namespace NUMINAMATH_CALUDE_sams_original_portion_l3233_323372

theorem sams_original_portion (s j r : ℝ) :
  s + j + r = 1200 →
  s - 200 + 3 * j + 3 * r = 1800 →
  s = 800 :=
by sorry

end NUMINAMATH_CALUDE_sams_original_portion_l3233_323372


namespace NUMINAMATH_CALUDE_student_selection_probability_l3233_323363

theorem student_selection_probability : 
  let total_students : ℕ := 4
  let selected_students : ℕ := 2
  let target_group : ℕ := 2
  let favorable_outcomes : ℕ := target_group * (total_students - target_group)
  let total_outcomes : ℕ := Nat.choose total_students selected_students
  (favorable_outcomes : ℚ) / total_outcomes = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_student_selection_probability_l3233_323363


namespace NUMINAMATH_CALUDE_min_value_a_l3233_323390

theorem min_value_a (a : ℝ) : 
  (∀ x : ℝ, x > 0 → (1 / (x^2 + 1)) ≤ (a / x)) → 
  a ≥ 1/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_l3233_323390


namespace NUMINAMATH_CALUDE_linear_function_max_value_l3233_323310

/-- The maximum value of a linear function y = (5/3)x + 2 over the interval [-3, 3] is 7 -/
theorem linear_function_max_value (x : ℝ) :
  x ∈ Set.Icc (-3 : ℝ) 3 →
  (5/3 : ℝ) * x + 2 ≤ 7 ∧ ∃ x₀, x₀ ∈ Set.Icc (-3 : ℝ) 3 ∧ (5/3 : ℝ) * x₀ + 2 = 7 :=
by sorry

end NUMINAMATH_CALUDE_linear_function_max_value_l3233_323310


namespace NUMINAMATH_CALUDE_stairs_climbed_together_l3233_323336

/-- The number of stairs Samir climbed -/
def samir_stairs : ℕ := 318

/-- The number of stairs Veronica climbed -/
def veronica_stairs : ℕ := samir_stairs / 2 + 18

/-- The total number of stairs Samir and Veronica climbed together -/
def total_stairs : ℕ := samir_stairs + veronica_stairs

theorem stairs_climbed_together : total_stairs = 495 := by
  sorry

end NUMINAMATH_CALUDE_stairs_climbed_together_l3233_323336


namespace NUMINAMATH_CALUDE_geometric_and_arithmetic_sequence_problem_l3233_323386

-- Define the geometric sequence
def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)

-- Define the b_n sequence
def b_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := geometric_sequence a₁ q n + 2*n

-- Define the sum of the first n terms of b_n
def T_n (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := 
  Finset.sum (Finset.range n) (λ i => b_sequence a₁ q (i+1))

theorem geometric_and_arithmetic_sequence_problem :
  ∀ a₁ q : ℝ,
  (a₁ > 0) →
  (q > 1) →
  (a₁ * (a₁*q) * (a₁*q^2) = 8) →
  (2*((a₁*q)+2) = (a₁+1) + ((a₁*q^2)+2)) →
  (a₁ = 1 ∧ q = 2) ∧
  (∀ n : ℕ, n > 0 → T_n 1 2 n = 2^n + n^2 + n - 1) :=
by sorry

end NUMINAMATH_CALUDE_geometric_and_arithmetic_sequence_problem_l3233_323386


namespace NUMINAMATH_CALUDE_set_membership_implies_value_l3233_323357

theorem set_membership_implies_value (m : ℤ) : 3 ∈ ({1, m + 2} : Set ℤ) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_set_membership_implies_value_l3233_323357


namespace NUMINAMATH_CALUDE_probability_one_blue_is_9_22_l3233_323342

/-- Represents the number of jellybeans of each color in the bowl -/
structure JellyBeanBowl where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Calculates the probability of picking exactly one blue jellybean -/
def probability_one_blue (bowl : JellyBeanBowl) : ℚ :=
  let total := bowl.red + bowl.blue + bowl.white
  let favorable := bowl.blue * (total - bowl.blue).choose 2
  favorable / total.choose 3

/-- The main theorem stating the probability of picking exactly one blue jellybean -/
theorem probability_one_blue_is_9_22 :
  probability_one_blue ⟨5, 2, 5⟩ = 9/22 := by
  sorry

#eval probability_one_blue ⟨5, 2, 5⟩

end NUMINAMATH_CALUDE_probability_one_blue_is_9_22_l3233_323342


namespace NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l3233_323399

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  ∀ p, p < 20 → ¬(Nat.Prime p ∧ p ∣ n)

theorem smallest_composite_no_small_factors :
  (is_composite 667 ∧ has_no_prime_factors_less_than_20 667) ∧
  (∀ m : ℕ, m < 667 → ¬(is_composite m ∧ has_no_prime_factors_less_than_20 m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l3233_323399


namespace NUMINAMATH_CALUDE_integer_root_iff_a_value_l3233_323375

def polynomial (a x : ℤ) : ℤ := x^3 + 3*x^2 + a*x - 7

theorem integer_root_iff_a_value (a : ℤ) : 
  (∃ x : ℤ, polynomial a x = 0) ↔ (a = -70 ∨ a = -29 ∨ a = -5 ∨ a = 3) := by sorry

end NUMINAMATH_CALUDE_integer_root_iff_a_value_l3233_323375


namespace NUMINAMATH_CALUDE_expected_value_twelve_sided_die_l3233_323328

/-- A twelve-sided die with faces numbered from 1 to 12 -/
def TwelveSidedDie := Finset.range 12

/-- The expected value of a roll of the twelve-sided die -/
def expectedValue : ℚ := (TwelveSidedDie.sum (λ i => i + 1)) / 12

/-- Theorem: The expected value of a roll of a twelve-sided die with faces numbered from 1 to 12 is 6.5 -/
theorem expected_value_twelve_sided_die : expectedValue = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_twelve_sided_die_l3233_323328


namespace NUMINAMATH_CALUDE_line_curve_properties_l3233_323313

/-- Line passing through a point with a given direction vector -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Curve defined by an equation -/
def Curve := (ℝ × ℝ) → Prop

def line_l : Line := { point := (1, 0), direction := (2, -1) }

def curve_C : Curve := fun (x, y) ↦ x^2 + y^2 - 2*x - 4*y - 4 = 0

/-- Distance from a point to a line -/
def distance_point_to_line (p : ℝ × ℝ) (l : Line) : ℝ := sorry

/-- Check if a line intersects a curve -/
def intersects (l : Line) (c : Curve) : Prop := sorry

/-- Length of the chord formed by the intersection of a line and a curve -/
def chord_length (l : Line) (c : Curve) : ℝ := sorry

theorem line_curve_properties :
  let origin := (0, 0)
  distance_point_to_line origin line_l = 1 / Real.sqrt 5 ∧
  intersects line_l curve_C ∧
  chord_length line_l curve_C = 2 * Real.sqrt 145 / 5 := by sorry

end NUMINAMATH_CALUDE_line_curve_properties_l3233_323313


namespace NUMINAMATH_CALUDE_harold_marble_sharing_l3233_323319

theorem harold_marble_sharing (total_marbles : ℕ) (kept_marbles : ℕ) (marbles_per_friend : ℕ) 
  (h1 : total_marbles = 100)
  (h2 : kept_marbles = 20)
  (h3 : marbles_per_friend = 16)
  : (total_marbles - kept_marbles) / marbles_per_friend = 5 := by
  sorry

end NUMINAMATH_CALUDE_harold_marble_sharing_l3233_323319


namespace NUMINAMATH_CALUDE_angle_ABC_is_30_degrees_l3233_323393

theorem angle_ABC_is_30_degrees (BA BC : ℝ × ℝ) : 
  BA = (1/2, Real.sqrt 3/2) → 
  BC = (Real.sqrt 3/2, 1/2) → 
  Real.arccos ((BA.1 * BC.1 + BA.2 * BC.2) / (Real.sqrt (BA.1^2 + BA.2^2) * Real.sqrt (BC.1^2 + BC.2^2))) = π/6 := by
  sorry

end NUMINAMATH_CALUDE_angle_ABC_is_30_degrees_l3233_323393


namespace NUMINAMATH_CALUDE_parallelogram_area_l3233_323377

/-- The area of a parallelogram with one angle of 100 degrees and two consecutive sides of lengths 10 inches and 20 inches is approximately 197.0 square inches. -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) : 
  a = 10 → b = 20 → θ = 100 * π / 180 → 
  abs (a * b * Real.sin (π - θ) - 197.0) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3233_323377


namespace NUMINAMATH_CALUDE_cost_of_four_books_l3233_323346

/-- Given that two identical books cost $36, prove that four of these books cost $72. -/
theorem cost_of_four_books (cost_of_two : ℝ) (h : cost_of_two = 36) : 
  2 * cost_of_two = 72 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_four_books_l3233_323346


namespace NUMINAMATH_CALUDE_greatest_integer_gcd_18_is_9_l3233_323325

theorem greatest_integer_gcd_18_is_9 :
  ∃ n : ℕ, n < 200 ∧ n.gcd 18 = 9 ∧ ∀ m : ℕ, m < 200 ∧ m.gcd 18 = 9 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_gcd_18_is_9_l3233_323325


namespace NUMINAMATH_CALUDE_circle_with_parallel_tangents_l3233_323337

-- Define the type for points in 2D space
def Point := ℝ × ℝ

-- Define three non-collinear points
variable (A B C : Point)

-- Define the property of non-collinearity
def NonCollinear (A B C : Point) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  (x₂ - x₁) * (y₃ - y₁) ≠ (y₂ - y₁) * (x₃ - x₁)

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- Define a tangent line to a circle
def IsTangent (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

-- Define parallel lines
def Parallel (l₁ l₂ : Point → Prop) : Prop :=
  ∀ (p q : Point), l₁ p ∧ l₂ q → ∃ (k : ℝ), k ≠ 0 ∧ 
    (p.1 - q.1) * k = (p.2 - q.2)

-- Theorem statement
theorem circle_with_parallel_tangents 
  (h : NonCollinear A B C) : 
  ∃ (c : Circle), c.center = C ∧ 
    ∃ (t₁ t₂ : Point → Prop), 
      IsTangent A c ∧ IsTangent B c ∧ 
      Parallel t₁ t₂ :=
sorry

end NUMINAMATH_CALUDE_circle_with_parallel_tangents_l3233_323337


namespace NUMINAMATH_CALUDE_marble_capacity_l3233_323335

theorem marble_capacity (v₁ v₂ : ℝ) (m₁ : ℕ) (h₁ : v₁ > 0) (h₂ : v₂ > 0) :
  v₁ = 36 → m₁ = 180 → v₂ = 108 →
  (v₂ / v₁ * m₁ : ℝ) = 540 := by sorry

end NUMINAMATH_CALUDE_marble_capacity_l3233_323335


namespace NUMINAMATH_CALUDE_length_breadth_difference_l3233_323305

/-- Represents a rectangular plot with given dimensions and fencing cost. -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ
  fencing_cost_per_meter : ℝ
  total_fencing_cost : ℝ

/-- Theorem stating that for a rectangular plot with given conditions, 
    the length is 12 meters more than the breadth. -/
theorem length_breadth_difference (plot : RectangularPlot) 
  (h1 : plot.length = 56)
  (h2 : plot.fencing_cost_per_meter = 26.5)
  (h3 : plot.total_fencing_cost = 5300)
  (h4 : plot.total_fencing_cost = (2 * (plot.length + plot.breadth)) * plot.fencing_cost_per_meter) :
  plot.length - plot.breadth = 12 := by
  sorry

#check length_breadth_difference

end NUMINAMATH_CALUDE_length_breadth_difference_l3233_323305


namespace NUMINAMATH_CALUDE_equation_solution_l3233_323394

theorem equation_solution : ∃ a : ℝ, -6 * a^2 = 3 * (4 * a + 2) ∧ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3233_323394


namespace NUMINAMATH_CALUDE_min_value_of_b_l3233_323320

def S (n : ℕ) : ℕ := 2^n - 1

def a : ℕ → ℝ
  | 0 => 1
  | n + 1 => S (n + 1) - S n

def b (n : ℕ) : ℝ := (a n)^2 - 7*(a n) + 6

theorem min_value_of_b :
  ∃ (m : ℝ), ∀ (n : ℕ), b n ≥ m ∧ ∃ (k : ℕ), b k = m ∧ m = -6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_b_l3233_323320


namespace NUMINAMATH_CALUDE_egg_collection_theorem_l3233_323339

/-- The number of dozen eggs Benjamin collects per day -/
def benjamin_eggs : ℕ := 6

/-- The number of dozen eggs Carla collects per day -/
def carla_eggs : ℕ := 3 * benjamin_eggs

/-- The number of dozen eggs Trisha collects per day -/
def trisha_eggs : ℕ := benjamin_eggs - 4

/-- The total number of dozen eggs collected by Benjamin, Carla, and Trisha -/
def total_eggs : ℕ := benjamin_eggs + carla_eggs + trisha_eggs

theorem egg_collection_theorem : total_eggs = 26 := by
  sorry

end NUMINAMATH_CALUDE_egg_collection_theorem_l3233_323339


namespace NUMINAMATH_CALUDE_sum_of_numbers_in_ratio_l3233_323340

theorem sum_of_numbers_in_ratio (a b c : ℝ) : 
  a > 0 ∧ b = 2*a ∧ c = 3*a ∧ a^2 + b^2 + c^2 = 2016 → a + b + c = 72 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_in_ratio_l3233_323340


namespace NUMINAMATH_CALUDE_sin_seven_pi_thirds_l3233_323391

theorem sin_seven_pi_thirds : Real.sin (7 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_seven_pi_thirds_l3233_323391


namespace NUMINAMATH_CALUDE_fraction_equality_l3233_323344

theorem fraction_equality : (2015 : ℤ) / (2015^2 - 2016 * 2014) = 2015 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3233_323344


namespace NUMINAMATH_CALUDE_no_real_m_for_equal_roots_l3233_323387

/-- The equation whose roots we're analyzing -/
def equation (x m : ℝ) : Prop :=
  (3 * x^2 * (x - 2) - (2*m + 3)) / ((x - 2) * (m - 2)) = 2 * x^2 / m

/-- Theorem stating that there are no real values of m for which the roots of the equation are equal -/
theorem no_real_m_for_equal_roots :
  ¬ ∃ (m : ℝ), ∃ (x : ℝ), ∀ (y : ℝ), equation y m → y = x :=
sorry

end NUMINAMATH_CALUDE_no_real_m_for_equal_roots_l3233_323387


namespace NUMINAMATH_CALUDE_f_nonpositive_implies_k_geq_one_l3233_323388

open Real

/-- Given a function f(x) = ln(ex) - kx defined on (0, +∞), 
    if f(x) ≤ 0 for all x > 0, then k ≥ 1 -/
theorem f_nonpositive_implies_k_geq_one (k : ℝ) : 
  (∀ x > 0, Real.log (Real.exp 1 * x) - k * x ≤ 0) → k ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_f_nonpositive_implies_k_geq_one_l3233_323388


namespace NUMINAMATH_CALUDE_tangent_sum_l3233_323303

theorem tangent_sum (x y : ℝ) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 4) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_l3233_323303


namespace NUMINAMATH_CALUDE_f_expression_m_values_l3233_323374

/-- A quadratic function satisfying certain properties -/
def f (x : ℝ) : ℝ := sorry

/-- The properties of the quadratic function -/
axiom f_diff (x : ℝ) : f (x + 1) - f x = 2 * x - 1
axiom f_zero : f 0 = 3

/-- The expression of f(x) -/
theorem f_expression (x : ℝ) : f x = x^2 - 2*x + 3 := sorry

/-- The function y in terms of x and m -/
def y (x m : ℝ) : ℝ := f (Real.log x / Real.log 3 + m)

/-- The set of x values -/
def X : Set ℝ := Set.Icc (1/3) 3

/-- The theorem about the values of m -/
theorem m_values :
  ∀ m : ℝ, (∀ x ∈ X, y x m ≥ 3) ∧ (∃ x ∈ X, y x m = 3) →
  m = -1 ∨ m = 3 := sorry

end NUMINAMATH_CALUDE_f_expression_m_values_l3233_323374


namespace NUMINAMATH_CALUDE_andrews_age_l3233_323368

theorem andrews_age (a g s : ℝ) 
  (h1 : g = 10 * a)
  (h2 : g - s = a + 45)
  (h3 : s = 5) :
  a = 50 / 9 := by
sorry

end NUMINAMATH_CALUDE_andrews_age_l3233_323368


namespace NUMINAMATH_CALUDE_marts_income_percentage_l3233_323355

theorem marts_income_percentage (juan tim mart : ℝ) 
  (h1 : mart = 1.3 * tim) 
  (h2 : tim = 0.6 * juan) : 
  mart = 0.78 * juan := by
sorry

end NUMINAMATH_CALUDE_marts_income_percentage_l3233_323355


namespace NUMINAMATH_CALUDE_definite_integral_sqrt_minus_x_l3233_323379

open Set
open MeasureTheory
open Interval

theorem definite_integral_sqrt_minus_x :
  ∫ (x : ℝ) in (Icc 0 1), (Real.sqrt (1 - (x - 1)^2) - x) = π/4 - 1/2 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_sqrt_minus_x_l3233_323379


namespace NUMINAMATH_CALUDE_jame_gold_bars_l3233_323347

/-- The number of gold bars Jame has left after tax and divorce -/
def gold_bars_left (initial : ℕ) (tax_rate : ℚ) (divorce_loss : ℚ) : ℕ :=
  let after_tax := initial - (initial * tax_rate).floor
  (after_tax - (after_tax * divorce_loss).floor).toNat

/-- Theorem stating that Jame has 27 gold bars left after tax and divorce -/
theorem jame_gold_bars :
  gold_bars_left 60 (1/10) (1/2) = 27 := by
  sorry

end NUMINAMATH_CALUDE_jame_gold_bars_l3233_323347


namespace NUMINAMATH_CALUDE_largest_n_for_product_2016_l3233_323349

/-- An arithmetic sequence with integer terms -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem largest_n_for_product_2016 :
  ∀ a b : ℕ → ℤ,
  ArithmeticSequence a →
  ArithmeticSequence b →
  a 1 = 1 →
  b 1 = 1 →
  a 2 ≤ b 2 →
  (∃ n : ℕ, a n * b n = 2016) →
  (∀ m : ℕ, (∃ n : ℕ, n > m ∧ a n * b n = 2016) → m ≤ 32) :=
sorry

end NUMINAMATH_CALUDE_largest_n_for_product_2016_l3233_323349


namespace NUMINAMATH_CALUDE_real_part_of_z_l3233_323370

theorem real_part_of_z (z : ℂ) (h : Complex.I * (z + 1) = -3 + 2 * Complex.I) : 
  z.re = 1 := by sorry

end NUMINAMATH_CALUDE_real_part_of_z_l3233_323370


namespace NUMINAMATH_CALUDE_right_triangle_sides_l3233_323361

theorem right_triangle_sides : ∃ (a b c : ℝ), 
  (a = 1 ∧ b = Real.sqrt 3 ∧ c = 2) ∧ 
  (a^2 + b^2 = c^2) ∧
  ¬(3^2 + 4^2 = 6^2) ∧
  ¬(5^2 + 12^2 = 14^2) ∧
  ¬((Real.sqrt 2)^2 + (Real.sqrt 3)^2 = 2^2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l3233_323361


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l3233_323365

theorem sufficient_condition_for_inequality (a : ℝ) (h : a > 4) :
  ∀ x : ℝ, x ∈ Set.Icc (-1) 2 → x^2 - a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l3233_323365


namespace NUMINAMATH_CALUDE_odd_function_extension_l3233_323329

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_extension 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_neg : ∀ x < 0, f x = x^2 - 3*x - 1) : 
  ∀ x > 0, f x = -x^2 - 3*x + 1 :=
by sorry

end NUMINAMATH_CALUDE_odd_function_extension_l3233_323329


namespace NUMINAMATH_CALUDE_light_reflection_theorem_l3233_323316

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if a point is on a line segment -/
def isOnSegment (P : Point) (A : Point) (B : Point) : Prop := sorry

/-- Checks if a quadrilateral is cyclic -/
def isCyclic (q : Quadrilateral) : Prop := sorry

/-- Calculates the perimeter of a quadrilateral -/
def perimeter (q : Quadrilateral) : ℝ := sorry

/-- Represents the light ray path -/
structure LightPath (q : Quadrilateral) where
  P : Point
  Q : Point
  R : Point
  S : Point
  pOnAB : isOnSegment P q.A q.B
  qOnBC : isOnSegment Q q.B q.C
  rOnCD : isOnSegment R q.C q.D
  sOnDA : isOnSegment S q.D q.A

theorem light_reflection_theorem (q : Quadrilateral) :
  (∀ (path : LightPath q), isCyclic q) ∧
  (∃ (c : ℝ), ∀ (path : LightPath q), perimeter ⟨path.P, path.Q, path.R, path.S⟩ = c) :=
sorry

end NUMINAMATH_CALUDE_light_reflection_theorem_l3233_323316


namespace NUMINAMATH_CALUDE_hyeyoung_walk_distance_l3233_323306

/-- Given a promenade of length 6 km, prove that walking to its halfway point is 3 km. -/
theorem hyeyoung_walk_distance (promenade_length : ℝ) (hyeyoung_distance : ℝ) 
  (h1 : promenade_length = 6)
  (h2 : hyeyoung_distance = promenade_length / 2) :
  hyeyoung_distance = 3 := by
  sorry

end NUMINAMATH_CALUDE_hyeyoung_walk_distance_l3233_323306


namespace NUMINAMATH_CALUDE_milburg_adult_population_l3233_323371

theorem milburg_adult_population (total_population children : ℝ) 
  (h1 : total_population = 5256.0)
  (h2 : children = 2987.0) :
  total_population - children = 2269.0 := by
sorry

end NUMINAMATH_CALUDE_milburg_adult_population_l3233_323371


namespace NUMINAMATH_CALUDE_gcd_abc_plus_cba_l3233_323385

def is_consecutive (a b c : ℕ) : Prop := b = a + 1 ∧ c = a + 2

def abc_plus_cba (a b c : ℕ) : ℕ := 100 * a + 10 * b + c + 100 * c + 10 * b + a

theorem gcd_abc_plus_cba :
  ∀ a b c : ℕ,
  0 ≤ a ∧ a ≤ 7 →
  is_consecutive a b c →
  (∃ k : ℕ, abc_plus_cba a b c = 2 * k) ∧
  (∃ a₁ b₁ c₁ a₂ b₂ c₂ : ℕ,
    0 ≤ a₁ ∧ a₁ ≤ 7 ∧
    0 ≤ a₂ ∧ a₂ ≤ 7 ∧
    is_consecutive a₁ b₁ c₁ ∧
    is_consecutive a₂ b₂ c₂ ∧
    Nat.gcd (abc_plus_cba a₁ b₁ c₁) (abc_plus_cba a₂ b₂ c₂) = 2) :=
by sorry

end NUMINAMATH_CALUDE_gcd_abc_plus_cba_l3233_323385


namespace NUMINAMATH_CALUDE_sum_of_squared_differences_zero_l3233_323332

theorem sum_of_squared_differences_zero (x y z : ℝ) :
  (x - 3)^2 + (y - 4)^2 + (z - 5)^2 = 0 → x + y + z = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_differences_zero_l3233_323332


namespace NUMINAMATH_CALUDE_bottle_production_l3233_323343

/-- Given that 6 identical machines can produce 240 bottles per minute at a constant rate,
    prove that 10 such machines will produce 1600 bottles in 4 minutes. -/
theorem bottle_production (rate : ℕ → ℕ → ℕ) : 
  (rate 6 1 = 240) → (rate 10 4 = 1600) :=
by
  sorry

end NUMINAMATH_CALUDE_bottle_production_l3233_323343


namespace NUMINAMATH_CALUDE_math_majors_consecutive_probability_l3233_323338

/-- The number of people sitting around the table -/
def total_people : ℕ := 12

/-- The number of math majors -/
def math_majors : ℕ := 5

/-- The number of physics majors -/
def physics_majors : ℕ := 4

/-- The number of biology majors -/
def biology_majors : ℕ := 3

/-- The probability of all math majors sitting in consecutive seats -/
def prob_consecutive_math : ℚ := 2 / 55

theorem math_majors_consecutive_probability :
  (total_people = math_majors + physics_majors + biology_majors) →
  (prob_consecutive_math = 2 / 55) := by
  sorry

end NUMINAMATH_CALUDE_math_majors_consecutive_probability_l3233_323338


namespace NUMINAMATH_CALUDE_decreasing_quadratic_condition_l3233_323345

/-- A quadratic function f(x) = x^2 + ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 3

/-- The property of f being decreasing on (-∞, 2] -/
def is_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x < y → y ≤ 2 → f a x > f a y

/-- The main theorem: if f is decreasing on (-∞, 2], then a ≤ -4 -/
theorem decreasing_quadratic_condition (a : ℝ) :
  is_decreasing_on_interval a → a ≤ -4 := by sorry

end NUMINAMATH_CALUDE_decreasing_quadratic_condition_l3233_323345


namespace NUMINAMATH_CALUDE_vieta_sum_product_l3233_323353

theorem vieta_sum_product (p q : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x ≠ y) →
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x + y = 9 ∧ x * y = 15) →
  p + q = 72 :=
by sorry

end NUMINAMATH_CALUDE_vieta_sum_product_l3233_323353


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_set_l3233_323312

theorem fraction_inequality_solution_set :
  {x : ℝ | (x - 2) / (x + 1) < 0} = {x : ℝ | -1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_set_l3233_323312


namespace NUMINAMATH_CALUDE_lcm_problem_l3233_323389

theorem lcm_problem (m : ℕ+) 
  (h1 : Nat.lcm 40 m = 120) 
  (h2 : Nat.lcm m 45 = 180) : 
  m = 12 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l3233_323389


namespace NUMINAMATH_CALUDE_two_pencils_length_l3233_323334

def pencil_length : ℕ := 12

theorem two_pencils_length : pencil_length + pencil_length = 24 := by
  sorry

end NUMINAMATH_CALUDE_two_pencils_length_l3233_323334


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l3233_323356

theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : x + y = 52) (h3 : x = 3 * y) :
  ∃ y_new : ℝ, ((-10 : ℝ) * y_new = k) ∧ (y_new = -50.7) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l3233_323356


namespace NUMINAMATH_CALUDE_triangle_condition_implies_isosceles_right_l3233_323352

-- Define a triangle with side lengths a, b, and c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the condition given in the problem
def satisfiesCondition (t : Triangle) : Prop :=
  |t.c^2 - t.a^2 - t.b^2| + (t.a - t.b)^2 = 0

-- Define an isosceles right triangle
def isIsoscelesRightTriangle (t : Triangle) : Prop :=
  t.a = t.b ∧ t.a^2 + t.b^2 = t.c^2

-- The theorem to be proved
theorem triangle_condition_implies_isosceles_right (t : Triangle) 
  (h : satisfiesCondition t) : isIsoscelesRightTriangle t :=
sorry

end NUMINAMATH_CALUDE_triangle_condition_implies_isosceles_right_l3233_323352


namespace NUMINAMATH_CALUDE_pairwise_sum_problem_l3233_323359

/-- Given four numbers that when added pairwise result in specific sums, 
    prove the remaining sums and possible sets of numbers -/
theorem pairwise_sum_problem (a b c d : ℝ) : 
  a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ 
  d + c = 20 ∧ d + b = 16 ∧ 
  ((d + a = 13 ∧ c + b = 9) ∨ (d + a = 9 ∧ c + b = 13)) →
  (a + b = 2 ∧ a + c = 6) ∧
  ((a = -0.5 ∧ b = 2.5 ∧ c = 6.5 ∧ d = 13.5) ∨
   (a = -2.5 ∧ b = 4.5 ∧ c = 8.5 ∧ d = 11.5)) := by
  sorry

end NUMINAMATH_CALUDE_pairwise_sum_problem_l3233_323359


namespace NUMINAMATH_CALUDE_may_cookie_cost_l3233_323317

/-- The total amount spent on cookies in May -/
def total_cookie_cost (weekday_count : ℕ) (weekend_count : ℕ) 
  (weekday_cookie_count : ℕ) (weekend_cookie_count : ℕ)
  (weekday_cookie1_price : ℕ) (weekday_cookie2_price : ℕ)
  (weekend_cookie1_price : ℕ) (weekend_cookie2_price : ℕ) : ℕ :=
  (weekday_count * (2 * weekday_cookie1_price + 2 * weekday_cookie2_price)) +
  (weekend_count * (3 * weekend_cookie1_price + 2 * weekend_cookie2_price))

/-- Theorem stating the total amount spent on cookies in May -/
theorem may_cookie_cost : 
  total_cookie_cost 22 9 4 5 15 18 12 20 = 2136 := by
  sorry

end NUMINAMATH_CALUDE_may_cookie_cost_l3233_323317


namespace NUMINAMATH_CALUDE_complement_P_equals_two_l3233_323307

def U : Set Int := {-1, 0, 1, 2}

def P : Set Int := {x : Int | x^2 < 2}

theorem complement_P_equals_two : 
  {x ∈ U | x ∉ P} = {2} := by sorry

end NUMINAMATH_CALUDE_complement_P_equals_two_l3233_323307


namespace NUMINAMATH_CALUDE_minimum_m_value_l3233_323362

theorem minimum_m_value (a x : ℝ) (ha : |a| ≤ 1) (hx : |x| ≤ 1) :
  ∃ m : ℝ, (∀ a x, |a| ≤ 1 → |x| ≤ 1 → |x^2 - a*x - a^2| ≤ m) ∧ 
  (∀ m' : ℝ, m' < m → ∃ a x, |a| ≤ 1 ∧ |x| ≤ 1 ∧ |x^2 - a*x - a^2| > m') ∧
  m = 5/4 :=
sorry

end NUMINAMATH_CALUDE_minimum_m_value_l3233_323362


namespace NUMINAMATH_CALUDE_darias_initial_savings_l3233_323366

def couch_price : ℕ := 750
def table_price : ℕ := 100
def lamp_price : ℕ := 50
def remaining_debt : ℕ := 400

def total_furniture_cost : ℕ := couch_price + table_price + lamp_price

theorem darias_initial_savings : total_furniture_cost - remaining_debt = 500 := by
  sorry

end NUMINAMATH_CALUDE_darias_initial_savings_l3233_323366


namespace NUMINAMATH_CALUDE_hyperbola_equation_prove_hyperbola_equation_l3233_323315

/-- The standard equation of a hyperbola with given foci and passing through a specific point. -/
theorem hyperbola_equation (h : ℝ → ℝ → Prop) (f : ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
  -- Given hyperbola with equation x^2/16 - y^2/9 = 1
  (∀ x y, h x y ↔ x^2/16 - y^2/9 = 1) →
  -- The new hyperbola has the same foci as the given one
  (∃ c : ℝ, c^2 = 25 ∧ f = (c, 0) ∨ f = (-c, 0)) →
  -- The new hyperbola passes through the point P
  (p = (-Real.sqrt 5 / 2, -Real.sqrt 6)) →
  -- The standard equation of the new hyperbola is x^2/1 - y^2/24 = 1
  (∀ x y, (x^2/1 - y^2/24 = 1) ↔ 
    ((x - f.1)^2 + y^2)^(1/2) - ((x + f.1)^2 + y^2)^(1/2) = 2 * Real.sqrt (f.1^2 - 1))

/-- Proof of the hyperbola equation -/
theorem prove_hyperbola_equation : ∃ h f p, hyperbola_equation h f p := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_prove_hyperbola_equation_l3233_323315


namespace NUMINAMATH_CALUDE_circle_tangent_line_l3233_323381

/-- A circle in polar coordinates with equation ρ = 2cosθ -/
def Circle (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

/-- A line in polar coordinates with equation 3ρcosθ + 4ρsinθ + a = 0 -/
def Line (ρ θ a : ℝ) : Prop := 3 * ρ * Real.cos θ + 4 * ρ * Real.sin θ + a = 0

/-- The circle is tangent to the line -/
def IsTangent (a : ℝ) : Prop :=
  ∃! (ρ θ : ℝ), Circle ρ θ ∧ Line ρ θ a

theorem circle_tangent_line (a : ℝ) :
  IsTangent a ↔ (a = -8 ∨ a = 2) :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_line_l3233_323381


namespace NUMINAMATH_CALUDE_division_subtraction_problem_l3233_323392

theorem division_subtraction_problem (x : ℝ) : 
  (800 / x) - 154 = 6 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_division_subtraction_problem_l3233_323392


namespace NUMINAMATH_CALUDE_larger_solution_quadratic_l3233_323322

theorem larger_solution_quadratic (x : ℝ) : 
  x^2 - 7*x - 18 = 0 → x ≤ 9 ∧ (∃ y : ℝ, y ≠ x ∧ y^2 - 7*y - 18 = 0) := by
  sorry

end NUMINAMATH_CALUDE_larger_solution_quadratic_l3233_323322


namespace NUMINAMATH_CALUDE_output_is_three_l3233_323360

def program_output (a b : ℕ) : ℕ := a + b

theorem output_is_three : program_output 1 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_output_is_three_l3233_323360


namespace NUMINAMATH_CALUDE_square_area_error_l3233_323300

theorem square_area_error (x : ℝ) (h : x > 0) :
  let measured_side := 1.19 * x
  let actual_area := x^2
  let calculated_area := measured_side^2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.4161 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l3233_323300


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_range_of_a_for_A_intersect_C_eq_C_l3233_323341

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - x - 6 ≤ 0}
def B : Set ℝ := {x | (x-4)/(x+1) < 0}
def C (a : ℝ) : Set ℝ := {x | 2-a < x ∧ x < 2+a}

-- Statement for (∁_R A) ∩ B = (3, 4)
theorem complement_A_intersect_B : (Set.univ \ A) ∩ B = Set.Ioo 3 4 := by sorry

-- Statement for the range of a when A ∩ C = C
theorem range_of_a_for_A_intersect_C_eq_C :
  ∀ a : ℝ, (A ∩ C a = C a) ↔ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_range_of_a_for_A_intersect_C_eq_C_l3233_323341


namespace NUMINAMATH_CALUDE_total_fish_l3233_323383

theorem total_fish (lilly_fish rosy_fish tom_fish : ℕ) 
  (h1 : lilly_fish = 10)
  (h2 : rosy_fish = 14)
  (h3 : tom_fish = 8) :
  lilly_fish + rosy_fish + tom_fish = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_l3233_323383


namespace NUMINAMATH_CALUDE_max_value_of_sum_of_roots_l3233_323311

theorem max_value_of_sum_of_roots (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 17) :
  Real.sqrt (2 * x + 27) + Real.sqrt (17 - x) + Real.sqrt (3 * x) ≤ 14.951 ∧
  ∃ x₀, x₀ = 17 ∧ Real.sqrt (2 * x₀ + 27) + Real.sqrt (17 - x₀) + Real.sqrt (3 * x₀) = 14.951 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_of_roots_l3233_323311


namespace NUMINAMATH_CALUDE_sum_not_prime_l3233_323358

theorem sum_not_prime (a b c d : ℕ) (h : a * b = c * d) : 
  ¬(Nat.Prime (a + b + c + d)) :=
by sorry

end NUMINAMATH_CALUDE_sum_not_prime_l3233_323358


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3233_323318

/-- Given an arithmetic sequence, prove that if the difference of the average of the first 2016 terms
    and the average of the first 16 terms is 100, then the common difference is 1/10. -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (d : ℝ) 
  (h_arithmetic : ∀ n, a (n + 1) = a n + d) 
  (h_sum : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2) 
  (h_condition : S 2016 / 2016 - S 16 / 16 = 100) :
  d = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3233_323318


namespace NUMINAMATH_CALUDE_crazy_silly_school_books_to_read_l3233_323351

/-- The number of books still to read in a series -/
def books_to_read (total_books read_books : ℕ) : ℕ :=
  total_books - read_books

/-- Theorem: For the 'crazy silly school' series, the number of books still to read is 10 -/
theorem crazy_silly_school_books_to_read :
  books_to_read 22 12 = 10 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_books_to_read_l3233_323351


namespace NUMINAMATH_CALUDE_one_true_proposition_l3233_323304

-- Define propositions p and q
def p : Prop := ∀ a b : ℝ, a > b → (1 / a < 1 / b)
def q : Prop := ∀ a b : ℝ, (1 / (a * b) < 0) → (a * b < 0)

-- State the theorem
theorem one_true_proposition (h1 : ¬p) (h2 : q) :
  (p ∧ q) = false ∧ (p ∨ q) = true ∧ ((¬p) ∧ (¬q)) = false :=
sorry

end NUMINAMATH_CALUDE_one_true_proposition_l3233_323304


namespace NUMINAMATH_CALUDE_unique_triple_solution_l3233_323327

theorem unique_triple_solution :
  ∃! (a b c : ℕ), b > 1 ∧ 2^c + 2^2016 = a^b ∧ a = 3 * 2^1008 ∧ b = 2 ∧ c = 2019 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l3233_323327


namespace NUMINAMATH_CALUDE_odd_three_digit_count_l3233_323308

/-- The set of digits that can be used for the first digit -/
def first_digit_set : Finset Nat := {0, 2}

/-- The set of digits that can be used for the second and third digits -/
def odd_digit_set : Finset Nat := {1, 3, 5}

/-- A function to check if a number is odd -/
def is_odd (n : Nat) : Bool := n % 2 = 1

/-- A function to check if a three-digit number has no repeating digits -/
def no_repeats (n : Nat) : Bool :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3

/-- The main theorem to be proved -/
theorem odd_three_digit_count : 
  (Finset.filter (λ n : Nat => 
    100 ≤ n ∧ n < 1000 ∧
    (n / 100) ∈ first_digit_set ∧
    ((n / 10) % 10) ∈ odd_digit_set ∧
    (n % 10) ∈ odd_digit_set ∧
    is_odd n ∧
    no_repeats n
  ) (Finset.range 1000)).card = 18 := by
  sorry

end NUMINAMATH_CALUDE_odd_three_digit_count_l3233_323308


namespace NUMINAMATH_CALUDE_product_of_roots_plus_one_l3233_323354

theorem product_of_roots_plus_one (a b c : ℝ) : 
  (a^3 - 15*a^2 + 20*a - 8 = 0) → 
  (b^3 - 15*b^2 + 20*b - 8 = 0) → 
  (c^3 - 15*c^2 + 20*c - 8 = 0) → 
  (1 + a) * (1 + b) * (1 + c) = 28 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_plus_one_l3233_323354


namespace NUMINAMATH_CALUDE_birds_and_storks_count_l3233_323333

/-- Given initial birds, storks, and additional birds, calculates the total number of birds and storks -/
def total_birds_and_storks (initial_birds : ℕ) (storks : ℕ) (additional_birds : ℕ) : ℕ :=
  initial_birds + additional_birds + storks

/-- Proves that with 3 initial birds, 2 storks, and 5 additional birds, the total is 10 -/
theorem birds_and_storks_count : total_birds_and_storks 3 2 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_birds_and_storks_count_l3233_323333


namespace NUMINAMATH_CALUDE_anna_gets_more_candy_l3233_323324

/-- Calculates the difference in candy pieces between Anna and Billy --/
def candy_difference (anna_per_house billy_per_house anna_houses billy_houses : ℕ) : ℕ :=
  anna_per_house * anna_houses - billy_per_house * billy_houses

/-- Proves that Anna gets 15 more pieces of candy than Billy --/
theorem anna_gets_more_candy : 
  candy_difference 14 11 60 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_anna_gets_more_candy_l3233_323324


namespace NUMINAMATH_CALUDE_simplify_fraction_l3233_323330

theorem simplify_fraction (a : ℝ) (h : a ≠ 0) : (a + 1) / a - 1 / a = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3233_323330


namespace NUMINAMATH_CALUDE_david_biology_marks_l3233_323309

/-- Calculates David's marks in Biology given his marks in other subjects and his average -/
def davidsBiologyMarks (english : ℕ) (mathematics : ℕ) (physics : ℕ) (chemistry : ℕ) (average : ℕ) : ℕ :=
  5 * average - (english + mathematics + physics + chemistry)

theorem david_biology_marks :
  davidsBiologyMarks 51 65 82 67 70 = 85 := by
  sorry

end NUMINAMATH_CALUDE_david_biology_marks_l3233_323309


namespace NUMINAMATH_CALUDE_partner_a_contribution_l3233_323364

/-- A business partnership where two partners contribute capital for different durations and share profits proportionally. -/
structure BusinessPartnership where
  /-- Duration (in months) that Partner A's capital is used -/
  duration_a : ℕ
  /-- Duration (in months) that Partner B's capital is used -/
  duration_b : ℕ
  /-- Fraction of profit received by Partner B -/
  profit_share_b : ℚ
  /-- Fraction of capital contributed by Partner A -/
  capital_fraction_a : ℚ

/-- Theorem stating that under given conditions, Partner A's capital contribution is 1/4 -/
theorem partner_a_contribution
  (bp : BusinessPartnership)
  (h1 : bp.duration_a = 15)
  (h2 : bp.duration_b = 10)
  (h3 : bp.profit_share_b = 2/3)
  : bp.capital_fraction_a = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_partner_a_contribution_l3233_323364


namespace NUMINAMATH_CALUDE_rita_butterfly_hours_l3233_323380

theorem rita_butterfly_hours : ∀ (total_required hours_backstroke hours_breaststroke monthly_freestyle_sidestroke months : ℕ),
  total_required = 1500 →
  hours_backstroke = 50 →
  hours_breaststroke = 9 →
  monthly_freestyle_sidestroke = 220 →
  months = 6 →
  total_required - (hours_backstroke + hours_breaststroke + monthly_freestyle_sidestroke * months) = 121 :=
by
  sorry

#check rita_butterfly_hours

end NUMINAMATH_CALUDE_rita_butterfly_hours_l3233_323380


namespace NUMINAMATH_CALUDE_factor_t_squared_minus_144_l3233_323367

theorem factor_t_squared_minus_144 (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) := by
  sorry

end NUMINAMATH_CALUDE_factor_t_squared_minus_144_l3233_323367


namespace NUMINAMATH_CALUDE_parallel_lines_d_value_l3233_323398

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of d for which the lines y = 3x + 5 and y = (4d)x + 3 are parallel -/
theorem parallel_lines_d_value :
  (∀ x y : ℝ, y = 3 * x + 5 ↔ y = (4 * d) * x + 3) → d = 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_d_value_l3233_323398


namespace NUMINAMATH_CALUDE_prime_and_multiple_of_5_probability_l3233_323301

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if a natural number is a multiple of 5 -/
def isMultipleOf5 (n : ℕ) : Prop := sorry

/-- The set of cards numbered from 1 to 75 -/
def cardSet : Finset ℕ := sorry

/-- The probability of an event occurring when selecting from the card set -/
def probability (event : ℕ → Prop) : ℚ := sorry

theorem prime_and_multiple_of_5_probability :
  probability (fun n => n ∈ cardSet ∧ isPrime n ∧ isMultipleOf5 n) = 1 / 75 := by sorry

end NUMINAMATH_CALUDE_prime_and_multiple_of_5_probability_l3233_323301


namespace NUMINAMATH_CALUDE_minimum_packages_shipped_minimum_packages_value_l3233_323378

def sarahs_load : ℕ := 18
def ryans_load : ℕ := 11

theorem minimum_packages_shipped (n : ℕ) :
  (n % sarahs_load = 0) ∧ (n % ryans_load = 0) →
  n ≥ Nat.lcm sarahs_load ryans_load :=
by sorry

theorem minimum_packages_value :
  Nat.lcm sarahs_load ryans_load = 198 :=
by sorry

end NUMINAMATH_CALUDE_minimum_packages_shipped_minimum_packages_value_l3233_323378


namespace NUMINAMATH_CALUDE_inverse_function_point_sum_l3233_323321

-- Define a function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
variable (f_inv : ℝ → ℝ)

-- Assume f and f_inv are inverses of each other
axiom inverse_relation : ∀ x, f_inv (f x) = x

-- Given condition: (2, 4) is on the graph of y = f(x)/3
axiom point_on_f : f 2 = 12

-- Theorem to prove
theorem inverse_function_point_sum :
  ∃ a b : ℝ, f_inv a = 3 * b ∧ a + b = 38 / 3 :=
sorry

end NUMINAMATH_CALUDE_inverse_function_point_sum_l3233_323321


namespace NUMINAMATH_CALUDE_spade_sum_equals_six_l3233_323323

-- Define the ♠ operation
def spade (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem spade_sum_equals_six : 
  (spade 2 3) + (spade 5 10) = 6 := by
  sorry

end NUMINAMATH_CALUDE_spade_sum_equals_six_l3233_323323


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3233_323396

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is √(1 + b²/a²) -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt (1 + b^2 / a^2)
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1) → e = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3233_323396


namespace NUMINAMATH_CALUDE_limit_proof_l3233_323369

open Real

theorem limit_proof (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧
  ∀ x : ℝ, 0 < |x - 1/2| ∧ |x - 1/2| < δ →
    |(2*x^2 - 5*x + 2)/(x - 1/2) + 3| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_limit_proof_l3233_323369


namespace NUMINAMATH_CALUDE_projection_matrix_values_l3233_323302

/-- A projection matrix is idempotent (P² = P) -/
def IsProjectionMatrix (P : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  P * P = P

/-- The specific form of our projection matrix -/
def P (a c : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![a, 21/76],
    ![c, 55/76]]

theorem projection_matrix_values :
  ∃ (a c : ℚ), IsProjectionMatrix (P a c) ∧ a = 7/19 ∧ c = 21/76 := by
  sorry

end NUMINAMATH_CALUDE_projection_matrix_values_l3233_323302


namespace NUMINAMATH_CALUDE_calorie_calculation_l3233_323382

/-- Represents the daily calorie allowance for a certain age group -/
def average_daily_allowance : ℕ := 2000

/-- The number of calories to reduce daily to hypothetically live to 100 years -/
def calorie_reduction : ℕ := 500

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The allowed weekly calorie intake for the age group -/
def allowed_weekly_intake : ℕ := 10500

theorem calorie_calculation :
  (average_daily_allowance - calorie_reduction) * days_in_week = allowed_weekly_intake := by
  sorry

end NUMINAMATH_CALUDE_calorie_calculation_l3233_323382


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3233_323395

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → 
  a - b = -10 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3233_323395


namespace NUMINAMATH_CALUDE_libby_igloo_top_bricks_l3233_323331

/-- Represents the structure of an igloo --/
structure Igloo where
  total_rows : ℕ
  bottom_rows : ℕ
  top_rows : ℕ
  bottom_bricks_per_row : ℕ
  total_bricks : ℕ

/-- Calculates the number of bricks in each row of the top half of the igloo --/
def top_bricks_per_row (i : Igloo) : ℕ :=
  (i.total_bricks - i.bottom_rows * i.bottom_bricks_per_row) / i.top_rows

/-- Theorem stating the number of bricks in each row of the top half of Libby's igloo --/
theorem libby_igloo_top_bricks :
  let i : Igloo := {
    total_rows := 10,
    bottom_rows := 5,
    top_rows := 5,
    bottom_bricks_per_row := 12,
    total_bricks := 100
  }
  top_bricks_per_row i = 8 := by
  sorry

end NUMINAMATH_CALUDE_libby_igloo_top_bricks_l3233_323331


namespace NUMINAMATH_CALUDE_train_passing_bridge_time_l3233_323350

/-- The time it takes for a train to pass a bridge -/
theorem train_passing_bridge_time (train_length bridge_length : ℝ) (train_speed_kmh : ℝ) :
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let time := total_distance / train_speed_ms
  train_length = 250 ∧ bridge_length = 150 ∧ train_speed_kmh = 35 →
  ∃ ε > 0, |time - 41.1528| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_train_passing_bridge_time_l3233_323350


namespace NUMINAMATH_CALUDE_problem_statement_l3233_323376

-- Define proposition p
def p (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, (f x₁ - f x₂) * (x₁ - x₂) ≥ 0

-- Define proposition q
def q : Prop :=
  ∀ x y : ℝ, x + y > 2 → x > 1 ∨ y > 1

-- Define decreasing function
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

-- Theorem statement
theorem problem_statement (f : ℝ → ℝ) :
  (¬(p f ∧ q)) ∧ q → is_decreasing f :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3233_323376


namespace NUMINAMATH_CALUDE_sum_of_bases_equality_l3233_323348

/-- Converts a number from base 13 to base 10 -/
def base13ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 14 to base 10 -/
def base14ToBase10 (n : ℕ) : ℕ := sorry

/-- Represents the digit C in base 14 -/
def C : ℕ := 12

theorem sum_of_bases_equality : 
  base13ToBase10 372 + base14ToBase10 (4 * 14^2 + C * 14 + 5) = 1557 := by sorry

end NUMINAMATH_CALUDE_sum_of_bases_equality_l3233_323348


namespace NUMINAMATH_CALUDE_third_degree_polynomial_property_l3233_323384

/-- A third-degree polynomial with real coefficients -/
def ThirdDegreePolynomial : Type := ℝ → ℝ

/-- The property that the absolute value of g at specified points is 15 -/
def has_abs_value_15 (g : ThirdDegreePolynomial) : Prop :=
  |g 1| = 15 ∧ |g 3| = 15 ∧ |g 4| = 15 ∧ |g 5| = 15 ∧ |g 6| = 15 ∧ |g 7| = 15

theorem third_degree_polynomial_property (g : ThirdDegreePolynomial) 
  (h : has_abs_value_15 g) : |g 0| = 645/8 := by
  sorry


end NUMINAMATH_CALUDE_third_degree_polynomial_property_l3233_323384


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3233_323373

theorem inequality_solution_set (x : ℝ) :
  (3 ≤ x / (2 * x - 5) ∧ x / (2 * x - 5) < 8) ↔ (8/3 < x ∧ x ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3233_323373


namespace NUMINAMATH_CALUDE_equilateral_condition_isosceles_condition_l3233_323314

-- Define a triangle ABC with side lengths a, b, c
structure Triangle :=
  (a b c : ℝ)
  (positive_a : a > 0)
  (positive_b : b > 0)
  (positive_c : c > 0)
  (triangle_inequality_ab : a + b > c)
  (triangle_inequality_bc : b + c > a)
  (triangle_inequality_ca : c + a > b)

-- Define equilateral triangle
def is_equilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

-- Define isosceles triangle
def is_isosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

-- Theorem 1
theorem equilateral_condition (t : Triangle) :
  abs (t.a - t.b) + abs (t.b - t.c) = 0 → is_equilateral t :=
by sorry

-- Theorem 2
theorem isosceles_condition (t : Triangle) :
  (t.a - t.b) * (t.b - t.c) = 0 → is_isosceles t :=
by sorry

end NUMINAMATH_CALUDE_equilateral_condition_isosceles_condition_l3233_323314


namespace NUMINAMATH_CALUDE_yellow_teams_count_l3233_323326

theorem yellow_teams_count (blue_students yellow_students total_students total_teams blue_teams : ℕ)
  (h1 : blue_students = 70)
  (h2 : yellow_students = 84)
  (h3 : total_students = blue_students + yellow_students)
  (h4 : total_teams = 77)
  (h5 : total_students = 2 * total_teams)
  (h6 : blue_teams = 30) :
  ∃ yellow_teams : ℕ, yellow_teams = 37 ∧ 
    yellow_teams = total_teams - blue_teams - (blue_students + yellow_students - 2 * blue_teams) / 2 :=
by sorry

end NUMINAMATH_CALUDE_yellow_teams_count_l3233_323326

import Mathlib

namespace NUMINAMATH_CALUDE_jinyoung_fewest_marbles_l2180_218017

-- Define the number of marbles for each person
def minjeong_marbles : ℕ := 6
def joohwan_marbles : ℕ := 7
def sunho_marbles : ℕ := minjeong_marbles - 1
def jinyoung_marbles : ℕ := joohwan_marbles - 3

-- Define a function to get the number of marbles for each person
def marbles (person : String) : ℕ :=
  match person with
  | "Minjeong" => minjeong_marbles
  | "Joohwan" => joohwan_marbles
  | "Sunho" => sunho_marbles
  | "Jinyoung" => jinyoung_marbles
  | _ => 0

-- Theorem: Jinyoung has the fewest marbles
theorem jinyoung_fewest_marbles :
  ∀ person, person ≠ "Jinyoung" → marbles "Jinyoung" ≤ marbles person :=
by sorry

end NUMINAMATH_CALUDE_jinyoung_fewest_marbles_l2180_218017


namespace NUMINAMATH_CALUDE_problem_solution_l2180_218051

theorem problem_solution (a b d : ℤ) 
  (h1 : a + b = d) 
  (h2 : b + d = 7) 
  (h3 : d = 4) : 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2180_218051


namespace NUMINAMATH_CALUDE_homework_problem_count_l2180_218032

theorem homework_problem_count 
  (math_pages : ℕ) 
  (reading_pages : ℕ) 
  (problems_per_page : ℕ) 
  (h1 : math_pages = 6) 
  (h2 : reading_pages = 4) 
  (h3 : problems_per_page = 3) : 
  (math_pages + reading_pages) * problems_per_page = 30 := by
  sorry

end NUMINAMATH_CALUDE_homework_problem_count_l2180_218032


namespace NUMINAMATH_CALUDE_petri_dishes_count_l2180_218039

-- Define the total number of germs
def total_germs : ℝ := 0.037 * 10^5

-- Define the number of germs per dish
def germs_per_dish : ℕ := 25

-- Define the number of petri dishes
def num_petri_dishes : ℕ := 148

-- Theorem statement
theorem petri_dishes_count :
  (total_germs / germs_per_dish : ℝ) = num_petri_dishes := by
  sorry

end NUMINAMATH_CALUDE_petri_dishes_count_l2180_218039


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_properties_l2180_218047

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := 3 * n - 1

-- Define the geometric sequence b_n
def b (n : ℕ) : ℚ := 2 * (4 ^ (n - 1))

-- Define S_n (sum of first n terms of a_n)
def S (n : ℕ) : ℚ := (3 / 2) * n^2 + (1 / 2) * n

-- Define T_n (sum of first n terms of b_n)
def T (n : ℕ) : ℚ := (2 / 3) * (4^n - 1)

theorem arithmetic_geometric_sequence_properties :
  (∀ n : ℕ, n ≥ 1 → S n = (3 / 2) * n^2 + (1 / 2) * n) ∧
  (b 1 = a 1) ∧
  (b 2 = a 3) →
  (∀ n : ℕ, n ≥ 1 → a n = 3 * n - 1) ∧
  (∀ n : ℕ, n ≥ 1 → T n = (2 / 3) * (4^n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_properties_l2180_218047


namespace NUMINAMATH_CALUDE_james_total_distance_l2180_218044

/-- Calculates the total distance driven given a series of driving segments -/
def total_distance (speeds : List ℝ) (times : List ℝ) : ℝ :=
  List.sum (List.zipWith (· * ·) speeds times)

/-- The total distance James drove under the given conditions -/
theorem james_total_distance :
  let speeds : List ℝ := [30, 60, 75, 60, 70]
  let times : List ℝ := [0.5, 0.75, 1.5, 2, 4]
  total_distance speeds times = 572.5 := by
  sorry

#check james_total_distance

end NUMINAMATH_CALUDE_james_total_distance_l2180_218044


namespace NUMINAMATH_CALUDE_betty_beads_l2180_218053

/-- Given that Betty has 3 red beads for every 2 blue beads and she has 20 blue beads,
    prove that Betty has 30 red beads. -/
theorem betty_beads (red_blue_ratio : ℚ) (blue_beads : ℕ) (red_beads : ℕ) : 
  red_blue_ratio = 3 / 2 →
  blue_beads = 20 →
  red_beads = red_blue_ratio * blue_beads →
  red_beads = 30 := by
sorry

end NUMINAMATH_CALUDE_betty_beads_l2180_218053


namespace NUMINAMATH_CALUDE_rectangular_solid_volume_l2180_218063

/-- Given a rectangular solid with side areas 3, 5, and 15 sharing a common vertex,
    its volume is 15. -/
theorem rectangular_solid_volume (a b c : ℝ) (h1 : a * b = 3) (h2 : a * c = 5) (h3 : b * c = 15) :
  a * b * c = 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_volume_l2180_218063


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_l2180_218079

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + m * x + 3

-- State the theorem
theorem f_monotone_decreasing (m : ℝ) :
  (∀ x : ℝ, f m x = f m (-x)) →  -- f is even
  ∀ x : ℝ, x > 0 → ∀ y : ℝ, y > x → f m y < f m x :=
by sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_l2180_218079


namespace NUMINAMATH_CALUDE_angle_ratio_not_determine_right_triangle_l2180_218009

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = 180

-- Define a right triangle
def is_right_triangle (t : Triangle) : Prop :=
  t.A = 90 ∨ t.B = 90 ∨ t.C = 90

-- Define the angle ratio condition
def angle_ratio_condition (t : Triangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ t.A = 6 * k ∧ t.B = 8 * k ∧ t.C = 10 * k

-- Theorem statement
theorem angle_ratio_not_determine_right_triangle :
  ∃ (t : Triangle), angle_ratio_condition t ∧ ¬(is_right_triangle t) :=
sorry

end NUMINAMATH_CALUDE_angle_ratio_not_determine_right_triangle_l2180_218009


namespace NUMINAMATH_CALUDE_purple_car_count_l2180_218020

theorem purple_car_count (total : ℕ) (purple blue red orange yellow green : ℕ)
  (h_total : total = 987)
  (h_blue : blue = 2 * red)
  (h_red : red = 3 * orange)
  (h_yellow1 : yellow = orange / 2)
  (h_yellow2 : yellow = 3 * purple)
  (h_green : green = 5 * purple)
  (h_sum : purple + yellow + orange + red + blue + green = total) :
  purple = 14 := by
  sorry

end NUMINAMATH_CALUDE_purple_car_count_l2180_218020


namespace NUMINAMATH_CALUDE_complementary_angles_difference_l2180_218089

theorem complementary_angles_difference (a b : ℝ) : 
  a + b = 90 →  -- complementary angles
  a = 3 * b →   -- ratio of 3:1
  |a - b| = 45  -- positive difference
  := by sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_l2180_218089


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l2180_218048

theorem cube_root_equation_solution :
  ∃! x : ℝ, (5 - x / 3) ^ (1/3 : ℝ) = -4 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l2180_218048


namespace NUMINAMATH_CALUDE_min_value_of_function_equality_condition_l2180_218094

theorem min_value_of_function (x : ℝ) (h : x > -1) :
  2 * x + 1 / (x + 1) ≥ 2 * Real.sqrt 2 - 2 :=
by sorry

theorem equality_condition (x : ℝ) (h : x > -1) :
  2 * x + 1 / (x + 1) = 2 * Real.sqrt 2 - 2 ↔ x = Real.sqrt 2 / 2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_equality_condition_l2180_218094


namespace NUMINAMATH_CALUDE_solution_system_equations_l2180_218043

theorem solution_system_equations :
  ∀ x y : ℝ, x > 0 ∧ y > 0 →
  (x - 3 * Real.sqrt (x * y) - 2 * Real.sqrt (x / y) = 0 ∧
   x^2 * y^2 + x^4 = 82) →
  ((x = 3 ∧ y = 1/3) ∨ (x = Real.rpow 66 (1/4) ∧ y = 4 / Real.rpow 66 (1/4))) :=
by sorry

end NUMINAMATH_CALUDE_solution_system_equations_l2180_218043


namespace NUMINAMATH_CALUDE_inequality_proof_l2180_218085

theorem inequality_proof (a b c : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  (a - b) * c^2 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2180_218085


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l2180_218022

/-- Given a quadratic equation (a-5)x^2 - 4x - 1 = 0 with real roots, prove that a ≥ 1 and a ≠ 5 -/
theorem quadratic_real_roots (a : ℝ) : 
  (∃ x : ℝ, (a - 5) * x^2 - 4 * x - 1 = 0) → 
  (a ≥ 1 ∧ a ≠ 5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l2180_218022


namespace NUMINAMATH_CALUDE_units_digit_of_sum_l2180_218097

theorem units_digit_of_sum (n : ℕ) : n = 33^43 + 43^32 → n % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_l2180_218097


namespace NUMINAMATH_CALUDE_farm_bulls_count_l2180_218034

/-- Given a farm with cattle and a cow-to-bull ratio, calculates the number of bulls -/
def calculate_bulls (total_cattle : ℕ) (cow_ratio : ℕ) (bull_ratio : ℕ) : ℕ :=
  (total_cattle * bull_ratio) / (cow_ratio + bull_ratio)

/-- Theorem: On a farm with 555 cattle and a cow-to-bull ratio of 10:27, there are 405 bulls -/
theorem farm_bulls_count : calculate_bulls 555 10 27 = 405 := by
  sorry

end NUMINAMATH_CALUDE_farm_bulls_count_l2180_218034


namespace NUMINAMATH_CALUDE_frequency_distribution_necessary_sufficient_l2180_218045

/-- Represents a sample of data -/
structure Sample (α : Type) where
  data : List α

/-- Represents a frequency distribution of a sample -/
structure FrequencyDistribution (α : Type) where
  ranges : List (α × α)
  counts : List Nat

/-- Represents the proportion of data points falling within a range -/
def proportion {α : Type} (s : Sample α) (range : α × α) : ℝ := sorry

/-- Main theorem: The frequency distribution is necessary and sufficient to determine
    the proportion of data points falling within any range in a sample -/
theorem frequency_distribution_necessary_sufficient
  {α : Type} [LinearOrder α] (s : Sample α) :
  ∃ (fd : FrequencyDistribution α),
    (∀ (range : α × α), ∃ (p : ℝ), proportion s range = p) ↔
    (∀ (range : α × α), ∃ (count : Nat), count ∈ fd.counts) :=
  sorry

end NUMINAMATH_CALUDE_frequency_distribution_necessary_sufficient_l2180_218045


namespace NUMINAMATH_CALUDE_cos_72_minus_cos_144_l2180_218059

/-- Proves that the difference between cosine of 72 degrees and cosine of 144 degrees is 1/2 -/
theorem cos_72_minus_cos_144 : Real.cos (72 * π / 180) - Real.cos (144 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_72_minus_cos_144_l2180_218059


namespace NUMINAMATH_CALUDE_cubic_fraction_equals_three_l2180_218071

theorem cubic_fraction_equals_three (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (sum_prod_nonzero : a * b + a * c + b * c ≠ 0) :
  (a^3 + b^3 + c^3) / (a * b * c * (a * b + a * c + b * c)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_fraction_equals_three_l2180_218071


namespace NUMINAMATH_CALUDE_probability_two_heads_two_tails_l2180_218002

theorem probability_two_heads_two_tails : 
  let n : ℕ := 4  -- total number of coins
  let k : ℕ := 2  -- number of heads (or tails) we want
  let p : ℚ := 1/2  -- probability of getting heads (or tails) on a single toss
  Nat.choose n k * p^n = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_heads_two_tails_l2180_218002


namespace NUMINAMATH_CALUDE_square_root_difference_equals_two_sqrt_three_l2180_218001

theorem square_root_difference_equals_two_sqrt_three :
  Real.sqrt (7 + 4 * Real.sqrt 3) - Real.sqrt (7 - 4 * Real.sqrt 3) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_difference_equals_two_sqrt_three_l2180_218001


namespace NUMINAMATH_CALUDE_pattern_equation_l2180_218028

theorem pattern_equation (n : ℕ+) : n * (n + 2) + 1 = (n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_pattern_equation_l2180_218028


namespace NUMINAMATH_CALUDE_function_above_identity_l2180_218055

theorem function_above_identity (f : ℝ → ℝ) (hf : Continuous f) :
  (∀ a₁ ∈ Set.Ioo 0 1, ∀ n : ℕ, f^[n+1] a₁ > f^[n] a₁) →
  ∀ x ∈ Set.Ioo 0 1, f x > x :=
sorry

end NUMINAMATH_CALUDE_function_above_identity_l2180_218055


namespace NUMINAMATH_CALUDE_sequence_general_term_l2180_218003

def sequence_property (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ 
  ∀ n : ℕ, n > 0 → (n + 1 : ℝ) * a (n + 1) - n * (a n)^2 + (n + 1 : ℝ) * a n * a (n + 1) - n * a n = 0

theorem sequence_general_term (a : ℕ → ℝ) (h : sequence_property a) :
  ∀ n : ℕ, n > 0 → a n = 1 / n :=
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2180_218003


namespace NUMINAMATH_CALUDE_unique_a_for_nonnegative_f_l2180_218031

theorem unique_a_for_nonnegative_f :
  ∃! a : ℝ, a > 0 ∧ ∀ x : ℝ, x > 0 → x^2 * (Real.log x - a) + a ≥ 0 ∧ a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_for_nonnegative_f_l2180_218031


namespace NUMINAMATH_CALUDE_range_of_f_l2180_218025

def f (x : ℝ) := x^4 - 4*x^2 + 4

theorem range_of_f :
  Set.range f = Set.Ici (0 : ℝ) := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2180_218025


namespace NUMINAMATH_CALUDE_ellipse_properties_l2180_218093

/-- Properties of an ellipse with equation x²/4 + y²/2 = 1 -/
theorem ellipse_properties :
  let a := 2  -- semi-major axis
  let b := Real.sqrt 2  -- semi-minor axis
  let c := Real.sqrt (a^2 - b^2)  -- focal distance / 2
  let e := c / a  -- eccentricity
  (∀ x y, x^2/4 + y^2/2 = 1 →
    (2*a = 4 ∧  -- length of major axis
     2*c = 2*Real.sqrt 2 ∧  -- focal distance
     e = Real.sqrt 2 / 2))  -- eccentricity
  := by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2180_218093


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2180_218078

theorem train_speed_calculation (train_length : Real) (crossing_time : Real) : 
  train_length = 133.33333333333334 →
  crossing_time = 8 →
  (train_length / 1000) / (crossing_time / 3600) = 60 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l2180_218078


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l2180_218064

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem fifth_term_of_sequence (x y : ℝ) (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_first : a 0 = x + 2*y)
  (h_second : a 1 = x - 2*y)
  (h_third : a 2 = 2*x*y)
  (h_fourth : a 3 = 2*x/y)
  (h_y_neq_half : y ≠ 1/2) :
  a 4 = (-12 - 8*y^2 + 4*y) / (2*y - 1) :=
sorry

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l2180_218064


namespace NUMINAMATH_CALUDE_ceiling_sqrt_250_l2180_218070

theorem ceiling_sqrt_250 : ⌈Real.sqrt 250⌉ = 16 := by sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_250_l2180_218070


namespace NUMINAMATH_CALUDE_collinear_vectors_x_value_l2180_218060

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem collinear_vectors_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, 1)
  collinear a b → x = (1 : ℝ) / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_x_value_l2180_218060


namespace NUMINAMATH_CALUDE_power_equation_solution_l2180_218088

theorem power_equation_solution : ∃ x : ℤ, 5^3 - 7 = 6^2 + x ∧ x = 82 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2180_218088


namespace NUMINAMATH_CALUDE_negation_of_implication_l2180_218019

theorem negation_of_implication (x : ℝ) :
  ¬(x > 0 → x^2 > 0) ↔ (x ≤ 0 → x^2 ≤ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2180_218019


namespace NUMINAMATH_CALUDE_tunnel_length_specific_tunnel_length_l2180_218095

/-- The length of a tunnel given train and time information -/
theorem tunnel_length (train_length : ℝ) (time_diff : ℝ) (train_speed : ℝ) : ℝ :=
  let tunnel_length := train_speed * time_diff / 60
  by
    -- Proof goes here
    sorry

/-- The specific tunnel length for the given problem -/
theorem specific_tunnel_length : 
  tunnel_length 2 4 30 = 2 := by sorry

end NUMINAMATH_CALUDE_tunnel_length_specific_tunnel_length_l2180_218095


namespace NUMINAMATH_CALUDE_function_composition_equality_l2180_218083

/-- Given real numbers a, b, c, d, k where k ≠ 0, and functions f and g defined as
    f(x) = ax + b and g(x) = k(cx + d), this theorem states that f(g(x)) = g(f(x))
    if and only if b(1 - kc) = k(d(1 - a)). -/
theorem function_composition_equality
  (a b c d k : ℝ)
  (hk : k ≠ 0)
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (hf : ∀ x, f x = a * x + b)
  (hg : ∀ x, g x = k * (c * x + d)) :
  (∀ x, f (g x) = g (f x)) ↔ b * (1 - k * c) = k * (d * (1 - a)) :=
by sorry

end NUMINAMATH_CALUDE_function_composition_equality_l2180_218083


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2180_218000

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (a - b)^2 = 4*(a*b)^3) :
  (1/a + 1/b) ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2180_218000


namespace NUMINAMATH_CALUDE_x_plus_y_equals_483_l2180_218091

theorem x_plus_y_equals_483 (x y : ℝ) : 
  x = 300 * (1 - 0.3) → 
  y = x * (1 + 0.3) → 
  x + y = 483 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_483_l2180_218091


namespace NUMINAMATH_CALUDE_x_value_when_y_is_two_l2180_218062

theorem x_value_when_y_is_two (x y : ℚ) : 
  y = 1 / (4 * x + 2) → y = 2 → x = -3/8 := by
sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_two_l2180_218062


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l2180_218082

theorem min_value_trig_expression (θ : Real) (h : 0 < θ ∧ θ < π/2) :
  ∃ (min_val : Real), min_val = 3*Real.sqrt 2 + Real.sqrt 3 ∧
  ∀ θ', 0 < θ' ∧ θ' < π/2 →
    3 * Real.sin θ' + 2 / Real.cos θ' + Real.sqrt 3 * (Real.cos θ' / Real.sin θ') ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l2180_218082


namespace NUMINAMATH_CALUDE_set_union_problem_l2180_218023

theorem set_union_problem (a b : ℝ) :
  let A : Set ℝ := {-1, a}
  let B : Set ℝ := {2^a, b}
  A ∩ B = {1} → A ∪ B = {-1, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_set_union_problem_l2180_218023


namespace NUMINAMATH_CALUDE_parabola_properties_l2180_218016

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

theorem parabola_properties :
  (∃ (x y : ℝ), IsLocalMin f x ∧ f x = y ∧ x = -1 ∧ y = -4) ∧
  (∀ x : ℝ, x ≥ 2 → f x ≥ 5) ∧
  (∃ x : ℝ, x ≥ 2 ∧ f x = 5) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l2180_218016


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sixth_term_l2180_218030

/-- An arithmetic sequence with given properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_a1 : a 1 = 5)
  (h_a5 : a 5 = 1) :
  a 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sixth_term_l2180_218030


namespace NUMINAMATH_CALUDE_multiply_fractions_l2180_218037

theorem multiply_fractions : 12 * (1 / 15) * 30 = 24 := by sorry

end NUMINAMATH_CALUDE_multiply_fractions_l2180_218037


namespace NUMINAMATH_CALUDE_ball_hits_ground_l2180_218096

def ball_height (t : ℝ) : ℝ := -18 * t^2 + 30 * t + 60

theorem ball_hits_ground :
  ∃ t : ℝ, t > 0 ∧ ball_height t = 0 ∧ t = (5 + Real.sqrt 145) / 6 :=
sorry

end NUMINAMATH_CALUDE_ball_hits_ground_l2180_218096


namespace NUMINAMATH_CALUDE_evaluate_polynomial_l2180_218046

theorem evaluate_polynomial (y : ℝ) (h : y = 2) : y^4 + y^3 + y^2 + y + 1 = 31 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_polynomial_l2180_218046


namespace NUMINAMATH_CALUDE_pradeep_marks_l2180_218066

theorem pradeep_marks (total_marks : ℕ) (pass_percentage : ℚ) (fail_margin : ℕ) (obtained_marks : ℕ) : 
  total_marks = 550 →
  pass_percentage = 40 / 100 →
  obtained_marks = (pass_percentage * total_marks).floor - fail_margin →
  obtained_marks = 200 := by
sorry

end NUMINAMATH_CALUDE_pradeep_marks_l2180_218066


namespace NUMINAMATH_CALUDE_arithmetic_computation_l2180_218012

theorem arithmetic_computation : -7 * 3 - (-4 * -2) + (-9 * -6) / 3 = -11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l2180_218012


namespace NUMINAMATH_CALUDE_right_triangle_area_l2180_218065

/-- Given a right triangle ABC with ∠C = 90°, a + b = 14 cm, and c = 10 cm, 
    the area of the triangle is 24 cm². -/
theorem right_triangle_area (a b c : ℝ) : 
  a + b = 14 → c = 10 → a^2 + b^2 = c^2 → (1/2) * a * b = 24 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2180_218065


namespace NUMINAMATH_CALUDE_rehabilitation_centers_count_l2180_218036

/-- The number of rehabilitation centers visited by Lisa, Jude, Han, and Jane. -/
def total_rehabilitation_centers (lisa jude han jane : ℕ) : ℕ :=
  lisa + jude + han + jane

/-- Theorem stating the total number of rehabilitation centers visited. -/
theorem rehabilitation_centers_count :
  ∃ (lisa jude han jane : ℕ),
    lisa = 6 ∧
    jude = lisa / 2 ∧
    han = 2 * jude - 2 ∧
    jane = 2 * han + 6 ∧
    total_rehabilitation_centers lisa jude han jane = 27 := by
  sorry

end NUMINAMATH_CALUDE_rehabilitation_centers_count_l2180_218036


namespace NUMINAMATH_CALUDE_intersection_set_equality_l2180_218035

theorem intersection_set_equality : 
  let S := {α : ℝ | ∃ k : ℤ, α = k * π / 2 - π / 5} ∩ {α : ℝ | -π < α ∧ α < π}
  S = {-π/5, -7*π/10, 3*π/10, 4*π/5} := by sorry

end NUMINAMATH_CALUDE_intersection_set_equality_l2180_218035


namespace NUMINAMATH_CALUDE_problem_statement_l2180_218042

/-- The repeating decimal 0.8̄ -/
def repeating_decimal : ℚ := 8/9

/-- The problem statement -/
theorem problem_statement : 2 - repeating_decimal = 10/9 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2180_218042


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l2180_218014

theorem infinitely_many_solutions : Set.Infinite {n : ℤ | (n - 3) * (n + 5) > 0} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l2180_218014


namespace NUMINAMATH_CALUDE_ball_distribution_with_constraint_l2180_218040

theorem ball_distribution_with_constraint (n : ℕ) (k : ℕ) (m : ℕ) :
  n = 5 → k = 3 → m = 2 →
  (n.pow k : ℕ) - (k - 1).pow n - n * (k - 1).pow (n - 1) = 131 :=
sorry

end NUMINAMATH_CALUDE_ball_distribution_with_constraint_l2180_218040


namespace NUMINAMATH_CALUDE_count_divisible_by_eight_l2180_218081

theorem count_divisible_by_eight : ∃ n : ℕ, n = (Finset.filter (fun x => x % 8 = 0) (Finset.Icc 200 400)).card ∧ n = 24 := by
  sorry

end NUMINAMATH_CALUDE_count_divisible_by_eight_l2180_218081


namespace NUMINAMATH_CALUDE_mothers_full_time_proportion_l2180_218098

/-- The proportion of mothers holding full-time jobs -/
def proportion_mothers_full_time : ℝ := sorry

/-- The proportion of fathers holding full-time jobs -/
def proportion_fathers_full_time : ℝ := 0.75

/-- The proportion of parents who are women -/
def proportion_women : ℝ := 0.4

/-- The proportion of parents who do not hold full-time jobs -/
def proportion_not_full_time : ℝ := 0.19

theorem mothers_full_time_proportion :
  proportion_mothers_full_time = 0.9 :=
by sorry

end NUMINAMATH_CALUDE_mothers_full_time_proportion_l2180_218098


namespace NUMINAMATH_CALUDE_complex_modulus_example_l2180_218084

theorem complex_modulus_example : Complex.abs (2 - (5/6) * Complex.I) = 13/6 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_example_l2180_218084


namespace NUMINAMATH_CALUDE_modular_inverse_of_9_mod_23_l2180_218013

theorem modular_inverse_of_9_mod_23 : ∃ x : ℕ, x ∈ Finset.range 23 ∧ (9 * x) % 23 = 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_9_mod_23_l2180_218013


namespace NUMINAMATH_CALUDE_domain_of_f_l2180_218057

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(2x+1)
def domain_f_2x_plus_1 : Set ℝ := Set.Ioo (-2) 0

-- Theorem statement
theorem domain_of_f (x : ℝ) : 
  (∀ y ∈ domain_f_2x_plus_1, ∃ x, y = 2*x + 1) →
  (Set.Ioo (-3) 1).Nonempty →
  x ∈ Set.Ioo (-3) 1 ↔ f x ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_domain_of_f_l2180_218057


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l2180_218061

theorem triangle_angle_proof (a b c : ℝ) (A : ℝ) (S : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  S = (1/2) * b * c * Real.sin A →
  b^2 + c^2 = (1/3) * a^2 + (4 * Real.sqrt 3 / 3) * S →
  A = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l2180_218061


namespace NUMINAMATH_CALUDE_quadratic_inequality_and_square_property_l2180_218026

theorem quadratic_inequality_and_square_property : 
  (¬∃ x : ℝ, x^2 - x + 2 < 0) ∧ (∀ x ∈ Set.Icc 1 2, x^2 ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_and_square_property_l2180_218026


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2180_218054

theorem quadratic_inequality_solution_set :
  {x : ℝ | -x^2 - 2*x + 3 < 0} = {x : ℝ | x < -3 ∨ x > 1} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2180_218054


namespace NUMINAMATH_CALUDE_average_of_x_and_y_l2180_218021

theorem average_of_x_and_y (x y : ℝ) : 
  (4 + 6 + 9 + x + y) / 5 = 20 → (x + y) / 2 = 40.5 := by
  sorry

end NUMINAMATH_CALUDE_average_of_x_and_y_l2180_218021


namespace NUMINAMATH_CALUDE_smallest_next_divisor_after_523_l2180_218077

def is_5digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

theorem smallest_next_divisor_after_523 (m : ℕ) (h1 : is_5digit m) (h2 : Even m) (h3 : m % 523 = 0) :
  ∃ (d : ℕ), d ∣ m ∧ d > 523 ∧ (∀ (x : ℕ), x ∣ m → x > 523 → x ≥ d) ∧ d = 524 :=
sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_after_523_l2180_218077


namespace NUMINAMATH_CALUDE_arithmetic_progression_pairs_l2180_218033

-- Define what it means for four numbers to be in arithmetic progression
def is_arithmetic_progression (x y z w : ℝ) : Prop :=
  ∃ d : ℝ, y = x + d ∧ z = y + d ∧ w = z + d

-- State the theorem
theorem arithmetic_progression_pairs :
  ∀ a b : ℝ, is_arithmetic_progression 10 a b (a * b) ↔ 
  ((a = 4 ∧ b = -2) ∨ (a = 2.5 ∧ b = -5)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_pairs_l2180_218033


namespace NUMINAMATH_CALUDE_democrat_ratio_l2180_218056

theorem democrat_ratio (total_participants : ℕ) (female_democrats : ℕ) 
  (h1 : total_participants = 810)
  (h2 : female_democrats = 135)
  (h3 : female_democrats * 2 ≤ total_participants)
  (h4 : total_participants / 3 = female_democrats + (total_participants - female_democrats * 2) / 4) :
  (total_participants - female_democrats * 2) / 4 = (total_participants - female_democrats * 2) / 4 :=
by sorry

#check democrat_ratio

end NUMINAMATH_CALUDE_democrat_ratio_l2180_218056


namespace NUMINAMATH_CALUDE_largest_common_divisor_462_330_l2180_218074

theorem largest_common_divisor_462_330 : Nat.gcd 462 330 = 66 := by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_462_330_l2180_218074


namespace NUMINAMATH_CALUDE_prob_same_heads_sum_l2180_218090

/-- Represents a coin with a given probability of landing heads -/
structure Coin where
  prob_heads : ℚ
  prob_heads_nonneg : 0 ≤ prob_heads
  prob_heads_le_one : prob_heads ≤ 1

/-- The set of three coins: two fair and one biased -/
def coin_set : Finset Coin := sorry

/-- The probability of getting the same number of heads when flipping the coin set twice -/
noncomputable def prob_same_heads (coins : Finset Coin) : ℚ := sorry

/-- The sum of numerator and denominator of the reduced fraction of prob_same_heads -/
noncomputable def sum_num_denom (coins : Finset Coin) : ℕ := sorry

theorem prob_same_heads_sum (h1 : coin_set.card = 3)
  (h2 : ∃ (c : Coin), c ∈ coin_set ∧ c.prob_heads = 1/2)
  (h3 : ∃ (c : Coin), c ∈ coin_set ∧ c.prob_heads = 3/5)
  (h4 : (coin_set.filter (fun c => c.prob_heads = 1/2)).card = 2) :
  sum_num_denom coin_set = 263 := by sorry

end NUMINAMATH_CALUDE_prob_same_heads_sum_l2180_218090


namespace NUMINAMATH_CALUDE_tangent_sum_simplification_l2180_218041

theorem tangent_sum_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + 
   Real.tan (80 * π / 180) + Real.tan (70 * π / 180)) / 
  Real.cos (40 * π / 180) = 
  (4 + 2 * (1 / Real.cos (40 * π / 180))) / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_simplification_l2180_218041


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_min_value_is_2_sqrt_2_min_value_equality_l2180_218068

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a + 1)⁻¹ + (b + 1)⁻¹ = 1) : 
  ∀ x y, x > 0 → y > 0 → (x + 1)⁻¹ + (y + 1)⁻¹ = 1 → a + 2 * b ≤ x + 2 * y :=
by
  sorry

theorem min_value_is_2_sqrt_2 (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a + 1)⁻¹ + (b + 1)⁻¹ = 1) : 
  a + 2 * b ≥ 2 * Real.sqrt 2 :=
by
  sorry

theorem min_value_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a + 1)⁻¹ + (b + 1)⁻¹ = 1) : 
  (a + 2 * b = 2 * Real.sqrt 2) ↔ (a + 1 = Real.sqrt 2 * (b + 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_min_value_is_2_sqrt_2_min_value_equality_l2180_218068


namespace NUMINAMATH_CALUDE_max_value_S_l2180_218076

theorem max_value_S (x y z w : Real) 
  (hx : x ∈ Set.Icc 0 1) 
  (hy : y ∈ Set.Icc 0 1) 
  (hz : z ∈ Set.Icc 0 1) 
  (hw : w ∈ Set.Icc 0 1) : 
  (x^2*y + y^2*z + z^2*w + w^2*x - x*y^2 - y*z^2 - z*w^2 - w*x^2) ≤ 8/27 := by
  sorry

end NUMINAMATH_CALUDE_max_value_S_l2180_218076


namespace NUMINAMATH_CALUDE_unique_k_with_prime_roots_l2180_218008

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- The quadratic equation x^2 - 75x + k = 0 has two prime roots -/
def hasPrimeRoots (k : ℤ) : Prop :=
  ∃ (p q : ℕ), isPrime p ∧ isPrime q ∧ p + q = 75 ∧ p * q = k

/-- There is exactly one integer k such that x^2 - 75x + k = 0 has two prime roots -/
theorem unique_k_with_prime_roots :
  ∃! (k : ℤ), hasPrimeRoots k :=
sorry

end NUMINAMATH_CALUDE_unique_k_with_prime_roots_l2180_218008


namespace NUMINAMATH_CALUDE_screen_time_calculation_l2180_218011

/-- Calculates the remaining screen time for the evening given the total recommended time and time already used. -/
def remaining_screen_time (total_recommended : ℕ) (time_used : ℕ) : ℕ :=
  total_recommended - time_used

/-- Converts hours to minutes. -/
def hours_to_minutes (hours : ℕ) : ℕ :=
  hours * 60

theorem screen_time_calculation :
  let total_recommended := hours_to_minutes 2
  let time_used := 45
  remaining_screen_time total_recommended time_used = 75 := by
  sorry

end NUMINAMATH_CALUDE_screen_time_calculation_l2180_218011


namespace NUMINAMATH_CALUDE_square_difference_fifty_fortynine_l2180_218052

theorem square_difference_fifty_fortynine : 50^2 - 49^2 = 99 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_fifty_fortynine_l2180_218052


namespace NUMINAMATH_CALUDE_jessica_exam_progress_l2180_218080

/-- Represents the exam parameters and Jessica's progress -/
structure ExamProgress where
  total_time : ℕ  -- Total time for the exam in minutes
  total_questions : ℕ  -- Total number of questions in the exam
  time_used : ℕ  -- Time used so far in minutes
  time_remaining : ℕ  -- Time remaining when exam is finished

/-- Represents that it's impossible to determine the exact number of questions answered -/
def cannot_determine_questions_answered (ep : ExamProgress) : Prop :=
  ∀ (questions_answered : ℕ), 
    questions_answered ≤ ep.total_questions → 
    ∃ (other_answered : ℕ), 
      other_answered ≠ questions_answered ∧ 
      other_answered ≤ ep.total_questions

/-- Theorem stating that given the exam conditions, it's impossible to determine
    the exact number of questions Jessica has answered so far -/
theorem jessica_exam_progress : 
  let ep : ExamProgress := {
    total_time := 60,
    total_questions := 80,
    time_used := 12,
    time_remaining := 0
  }
  cannot_determine_questions_answered ep :=
by
  sorry

end NUMINAMATH_CALUDE_jessica_exam_progress_l2180_218080


namespace NUMINAMATH_CALUDE_cone_surface_area_ratio_l2180_218027

/-- For a cone whose lateral surface unfolds into a sector with a central angle of 120° and radius 1,
    the ratio of its surface area to its lateral surface area is 4:3 -/
theorem cone_surface_area_ratio :
  let sector_angle : Real := 120 * π / 180
  let sector_radius : Real := 1
  let lateral_surface_area : Real := π * sector_radius^2 * (sector_angle / (2 * π))
  let base_radius : Real := sector_radius * sector_angle / (2 * π)
  let base_area : Real := π * base_radius^2
  let surface_area : Real := lateral_surface_area + base_area
  (surface_area / lateral_surface_area) = 4/3 := by
sorry

end NUMINAMATH_CALUDE_cone_surface_area_ratio_l2180_218027


namespace NUMINAMATH_CALUDE_systematic_sample_fourth_element_l2180_218010

/-- Represents a systematic sampling scheme -/
structure SystematicSample where
  total : Nat
  sampleSize : Nat
  interval : Nat
  firstElement : Nat

/-- Checks if a number is in the systematic sample -/
def isInSample (s : SystematicSample) (n : Nat) : Prop :=
  ∃ k : Nat, n = s.firstElement + k * s.interval ∧ n ≤ s.total

theorem systematic_sample_fourth_element 
  (s : SystematicSample)
  (h_total : s.total = 52)
  (h_size : s.sampleSize = 4)
  (h_interval : s.interval = s.total / s.sampleSize)
  (h_first : s.firstElement = 6)
  (h_in_32 : isInSample s 32)
  (h_in_45 : isInSample s 45) :
  isInSample s 19 :=
sorry

end NUMINAMATH_CALUDE_systematic_sample_fourth_element_l2180_218010


namespace NUMINAMATH_CALUDE_thomas_needs_2000_more_l2180_218006

/-- Thomas's savings scenario over two years -/
def thomas_savings_scenario (first_year_allowance : ℕ) (second_year_hourly_rate : ℕ) 
  (second_year_weekly_hours : ℕ) (car_cost : ℕ) (weekly_expenses : ℕ) : Prop :=
  let weeks_per_year : ℕ := 52
  let total_weeks : ℕ := 2 * weeks_per_year
  let first_year_earnings : ℕ := first_year_allowance * weeks_per_year
  let second_year_earnings : ℕ := second_year_hourly_rate * second_year_weekly_hours * weeks_per_year
  let total_earnings : ℕ := first_year_earnings + second_year_earnings
  let total_expenses : ℕ := weekly_expenses * total_weeks
  let savings : ℕ := total_earnings - total_expenses
  car_cost - savings = 2000

/-- Theorem stating Thomas needs $2000 more to buy the car -/
theorem thomas_needs_2000_more :
  thomas_savings_scenario 50 9 30 15000 35 := by sorry

end NUMINAMATH_CALUDE_thomas_needs_2000_more_l2180_218006


namespace NUMINAMATH_CALUDE_power_seven_mod_nine_l2180_218029

theorem power_seven_mod_nine : 7^138 % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_seven_mod_nine_l2180_218029


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2180_218050

-- Define the universal set U
def U : Set Nat := {0, 1, 2, 3, 4, 5}

-- Define set A
def A : Set Nat := {0, 2, 4}

-- Define set B
def B : Set Nat := {0, 5}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2180_218050


namespace NUMINAMATH_CALUDE_taxi_ride_cost_l2180_218005

/-- Calculates the total cost of a taxi ride -/
def taxi_cost (base_fare : ℝ) (per_mile_rate : ℝ) (tax_rate : ℝ) (distance : ℝ) : ℝ :=
  let fare_without_tax := base_fare + per_mile_rate * distance
  let tax := tax_rate * fare_without_tax
  fare_without_tax + tax

/-- Theorem: The total cost of an 8-mile taxi ride is $4.84 -/
theorem taxi_ride_cost :
  taxi_cost 2.00 0.30 0.10 8 = 4.84 := by
  sorry

end NUMINAMATH_CALUDE_taxi_ride_cost_l2180_218005


namespace NUMINAMATH_CALUDE_square_area_problem_l2180_218049

theorem square_area_problem (a b c : ℕ) : 
  4 * a < b → c^2 = a^2 + b^2 + 10 → c^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_area_problem_l2180_218049


namespace NUMINAMATH_CALUDE_largest_n_inequality_l2180_218038

theorem largest_n_inequality : ∃ (n : ℕ), n = 14 ∧ 
  (∀ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 → 
    (a^2 / (b/29 + c/31) + b^2 / (c/29 + a/31) + c^2 / (a/29 + b/31) ≥ n * (a + b + c))) ∧
  (∀ (m : ℕ), m > 14 → 
    ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
      (a^2 / (b/29 + c/31) + b^2 / (c/29 + a/31) + c^2 / (a/29 + b/31) < m * (a + b + c))) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_inequality_l2180_218038


namespace NUMINAMATH_CALUDE_smallest_surface_area_l2180_218069

def cube_surface_area (side : ℝ) : ℝ := 6 * side^2

def min_combined_surface_area (side1 side2 side3 : ℝ) : ℝ :=
  cube_surface_area side1 + cube_surface_area side2 + cube_surface_area side3 -
  (2 * side1^2 + 2 * side2^2 + 2 * side3^2)

theorem smallest_surface_area :
  min_combined_surface_area 3 5 8 = 502 := by
  sorry

end NUMINAMATH_CALUDE_smallest_surface_area_l2180_218069


namespace NUMINAMATH_CALUDE_cherry_sweets_count_l2180_218067

/-- The number of cherry-flavored sweets initially in the packet -/
def initial_cherry : ℕ := 30

/-- The number of strawberry-flavored sweets initially in the packet -/
def initial_strawberry : ℕ := 40

/-- The number of pineapple-flavored sweets initially in the packet -/
def initial_pineapple : ℕ := 50

/-- The number of cherry-flavored sweets Aaron gives to his friend -/
def given_away : ℕ := 5

/-- The total number of sweets left in the packet after Aaron's actions -/
def remaining_total : ℕ := 55

theorem cherry_sweets_count :
  initial_cherry = 30 ∧
  (initial_cherry / 2 - given_away) + (initial_strawberry / 2) + (initial_pineapple / 2) = remaining_total :=
by sorry

end NUMINAMATH_CALUDE_cherry_sweets_count_l2180_218067


namespace NUMINAMATH_CALUDE_binary_253_property_l2180_218024

/-- Represents a binary number as a list of bits (0 or 1) -/
def BinaryRepresentation := List Nat

/-- Converts a natural number to its binary representation -/
def toBinary (n : Nat) : BinaryRepresentation :=
  sorry

/-- Counts the number of zeros in a binary representation -/
def countZeros (bin : BinaryRepresentation) : Nat :=
  sorry

/-- Counts the number of ones in a binary representation -/
def countOnes (bin : BinaryRepresentation) : Nat :=
  sorry

theorem binary_253_property :
  let bin := toBinary 253
  let a := countZeros bin
  let b := countOnes bin
  2 * b - a = 13 := by sorry

end NUMINAMATH_CALUDE_binary_253_property_l2180_218024


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l2180_218086

/-- Given two similar triangles ABC and DEF, prove that DF = 6 -/
theorem similar_triangles_side_length 
  (A B C D E F : ℝ × ℝ) -- Points in 2D space
  (AB BC AC : ℝ) -- Sides of triangle ABC
  (DE EF : ℝ) -- Known sides of triangle DEF
  (angle_BAC angle_EDF : ℝ) -- Angles in radians
  (h_AB : dist A B = 8)
  (h_BC : dist B C = 18)
  (h_AC : dist A C = 12)
  (h_DE : dist D E = 4)
  (h_EF : dist E F = 9)
  (h_angle_BAC : angle_BAC = 2 * π / 3) -- 120° in radians
  (h_angle_EDF : angle_EDF = 2 * π / 3) -- 120° in radians
  : dist D F = 6 := by
  sorry


end NUMINAMATH_CALUDE_similar_triangles_side_length_l2180_218086


namespace NUMINAMATH_CALUDE_functional_equation_proof_l2180_218075

open Real

theorem functional_equation_proof (f g : ℝ → ℝ) : 
  (∀ x y : ℝ, sin x + cos y = f x + f y + g x - g y) ↔ 
  (∃ c : ℝ, ∀ x : ℝ, f x = (sin x + cos x) / 2 ∧ g x = (sin x - cos x) / 2 + c) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_proof_l2180_218075


namespace NUMINAMATH_CALUDE_product_of_differences_of_squares_l2180_218072

theorem product_of_differences_of_squares : 
  let P := Real.sqrt 2023 + Real.sqrt 2022
  let Q := Real.sqrt 2023 - Real.sqrt 2022
  let R := Real.sqrt 2023 + Real.sqrt 2024
  let S := Real.sqrt 2023 - Real.sqrt 2024
  (P * Q) * (R * S) = -1 := by sorry

end NUMINAMATH_CALUDE_product_of_differences_of_squares_l2180_218072


namespace NUMINAMATH_CALUDE_same_color_probability_l2180_218073

/-- The probability of drawing two balls of the same color from a bag containing
    8 blue balls and 7 yellow balls, with replacement. -/
theorem same_color_probability (blue_balls yellow_balls : ℕ) 
    (h_blue : blue_balls = 8) (h_yellow : yellow_balls = 7) :
    let total_balls := blue_balls + yellow_balls
    let p_blue := blue_balls / total_balls
    let p_yellow := yellow_balls / total_balls
    p_blue ^ 2 + p_yellow ^ 2 = 113 / 225 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l2180_218073


namespace NUMINAMATH_CALUDE_equal_money_days_l2180_218092

/-- The daily interest rate when leaving money with mother -/
def mother_rate : ℕ := 300

/-- The daily interest rate when leaving money with father -/
def father_rate : ℕ := 500

/-- The initial amount Kyu-won gave to her mother -/
def kyu_won_initial : ℕ := 8000

/-- The initial amount Seok-gi left with his father -/
def seok_gi_initial : ℕ := 5000

/-- The number of days needed for Kyu-won and Seok-gi to have the same amount of money -/
def days_needed : ℕ := 15

theorem equal_money_days :
  kyu_won_initial + mother_rate * days_needed = seok_gi_initial + father_rate * days_needed :=
by sorry

end NUMINAMATH_CALUDE_equal_money_days_l2180_218092


namespace NUMINAMATH_CALUDE_congruence_solution_l2180_218018

theorem congruence_solution (x : ℤ) 
  (h1 : (2 + x) % (5^3) = 2^2 % (5^3))
  (h2 : (3 + x) % (7^3) = 3^2 % (7^3))
  (h3 : (4 + x) % (11^3) = 5^2 % (11^3)) :
  x % 385 = 307 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l2180_218018


namespace NUMINAMATH_CALUDE_slope_determines_y_coordinate_l2180_218058

/-- Given two points P and Q, if the slope of the line passing through them is 1/4,
    then the y-coordinate of Q is -3. -/
theorem slope_determines_y_coordinate 
  (x_P y_P x_Q : ℝ) (slope : ℝ) :
  x_P = -3 →
  y_P = -5 →
  x_Q = 5 →
  slope = 1/4 →
  (y_Q - y_P) / (x_Q - x_P) = slope →
  y_Q = -3 :=
by sorry

end NUMINAMATH_CALUDE_slope_determines_y_coordinate_l2180_218058


namespace NUMINAMATH_CALUDE_perseverance_permutations_count_l2180_218099

/-- The number of letters in "PERSEVERANCE" -/
def word_length : ℕ := 11

/-- The number of occurrences of 'E' in "PERSEVERANCE" -/
def e_count : ℕ := 3

/-- The number of occurrences of 'R' in "PERSEVERANCE" -/
def r_count : ℕ := 2

/-- The number of unique permutations of the letters in "PERSEVERANCE" -/
def perseverance_permutations : ℕ := word_length.factorial / (e_count.factorial * r_count.factorial * r_count.factorial)

theorem perseverance_permutations_count :
  perseverance_permutations = 1663200 :=
by sorry

end NUMINAMATH_CALUDE_perseverance_permutations_count_l2180_218099


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l2180_218007

/-- A right triangle with two 45° angles and one 90° angle, and an inscribed circle of radius 8 cm has a hypotenuse of length 16(√2 + 1) cm. -/
theorem isosceles_right_triangle_hypotenuse (r : ℝ) (h : r = 8) :
  ∃ (a : ℝ), a > 0 ∧ 
  (a * a = 2 * r * r * (2 + Real.sqrt 2)) ∧
  (a * Real.sqrt 2 = 16 * (Real.sqrt 2 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l2180_218007


namespace NUMINAMATH_CALUDE_union_of_sets_l2180_218087

theorem union_of_sets : 
  let A : Set ℕ := {1, 2}
  let B : Set ℕ := {2, 3, 4}
  A ∪ B = {1, 2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l2180_218087


namespace NUMINAMATH_CALUDE_correct_result_l2180_218004

variables {a b c : ℤ}

theorem correct_result (A : ℤ) (h : A + 2 * (a * b + 2 * b * c - 4 * a * c) = 3 * a * b - 2 * a * c + 5 * b * c) :
  A - 2 * (a * b + 2 * b * c - 4 * a * c) = -a * b + 14 * a * c - 3 * b * c := by
  sorry

end NUMINAMATH_CALUDE_correct_result_l2180_218004


namespace NUMINAMATH_CALUDE_intersection_parallel_or_intersect_intersection_parallel_implies_parallel_to_plane_l2180_218015

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the basic operations and relations
variable (belongs_to : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (intersect : Line → Line → Prop)
variable (plane_intersect : Plane → Plane → Line → Prop)

-- Theorem 1
theorem intersection_parallel_or_intersect
  (α β : Plane) (m n : Line)
  (h1 : plane_intersect α β m)
  (h2 : belongs_to n α) :
  parallel m n ∨ intersect m n :=
sorry

-- Theorem 2
theorem intersection_parallel_implies_parallel_to_plane
  (α β : Plane) (m n : Line)
  (h1 : plane_intersect α β m)
  (h2 : parallel m n) :
  parallel_plane n α ∨ parallel_plane n β :=
sorry

end NUMINAMATH_CALUDE_intersection_parallel_or_intersect_intersection_parallel_implies_parallel_to_plane_l2180_218015

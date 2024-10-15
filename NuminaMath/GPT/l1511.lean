import Mathlib

namespace NUMINAMATH_GPT_sum_of_even_factors_420_l1511_151180

def sum_even_factors (n : ℕ) : ℕ :=
  if n ≠ 420 then 0
  else 
    let even_factors_sum :=
      (2 + 4) * (1 + 3) * (1 + 5) * (1 + 7)
    even_factors_sum

theorem sum_of_even_factors_420 : sum_even_factors 420 = 1152 :=
by {
  -- Proof skipped
  sorry
}

end NUMINAMATH_GPT_sum_of_even_factors_420_l1511_151180


namespace NUMINAMATH_GPT_find_x_from_conditions_l1511_151107

theorem find_x_from_conditions (x y : ℝ)
  (h1 : (6 : ℝ) = (1 / 2 : ℝ) * x)
  (h2 : y = (1 / 2 :ℝ) * 10)
  (h3 : x * y = 60) : x = 12 := by
  sorry

end NUMINAMATH_GPT_find_x_from_conditions_l1511_151107


namespace NUMINAMATH_GPT_min_value_ab_sum_l1511_151162

theorem min_value_ab_sum (a b : ℤ) (h : a * b = 100) : a + b ≥ -101 :=
  sorry

end NUMINAMATH_GPT_min_value_ab_sum_l1511_151162


namespace NUMINAMATH_GPT_distance_to_gym_l1511_151160

theorem distance_to_gym (v d : ℝ) (h_walked_200_m: 200 / v > 0) (h_double_speed: 2 * v = 2) (h_time_diff: 200 / v - d / (2 * v) = 50) : d = 300 :=
by sorry

end NUMINAMATH_GPT_distance_to_gym_l1511_151160


namespace NUMINAMATH_GPT_problem1_solution_l1511_151135

theorem problem1_solution (x y : ℚ) (h1 : 3 * x + 2 * y = 10) (h2 : x / 2 - (y + 1) / 3 = 1) : 
  x = 3 ∧ y = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_problem1_solution_l1511_151135


namespace NUMINAMATH_GPT_exhaust_pipe_leak_time_l1511_151109

theorem exhaust_pipe_leak_time : 
  (∃ T : Real, T > 0 ∧ 
                (1 / 10 - 1 / T) = 1 / 59.999999999999964 ∧ 
                T = 12) :=
by
  sorry

end NUMINAMATH_GPT_exhaust_pipe_leak_time_l1511_151109


namespace NUMINAMATH_GPT_jamal_total_cost_l1511_151120

-- Definitions based on conditions
def dozen := 12
def half_dozen := dozen / 2
def crayons_bought := 4 * half_dozen
def cost_per_crayon := 2
def total_cost := crayons_bought * cost_per_crayon

-- Proof statement (the question translated to a Lean theorem)
theorem jamal_total_cost : total_cost = 48 := by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_jamal_total_cost_l1511_151120


namespace NUMINAMATH_GPT_ratio_divisor_to_remainder_l1511_151100

theorem ratio_divisor_to_remainder (R D Q : ℕ) (hR : R = 46) (hD : D = 10 * Q) (hdvd : 5290 = D * Q + R) :
  D / R = 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_divisor_to_remainder_l1511_151100


namespace NUMINAMATH_GPT_sum_of_two_digit_factors_of_8060_l1511_151191

theorem sum_of_two_digit_factors_of_8060 : ∃ (a b : ℕ), (10 ≤ a ∧ a < 100) ∧ (10 ≤ b ∧ b < 100) ∧ (a * b = 8060) ∧ (a + b = 127) :=
by sorry

end NUMINAMATH_GPT_sum_of_two_digit_factors_of_8060_l1511_151191


namespace NUMINAMATH_GPT_cos_arithmetic_sequence_l1511_151147

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem cos_arithmetic_sequence (a : ℕ → ℝ) (h_seq : arithmetic_sequence a) (h_sum : a 1 + a 5 + a 9 = 8 * Real.pi) :
  Real.cos (a 3 + a 7) = -1 / 2 :=
sorry

end NUMINAMATH_GPT_cos_arithmetic_sequence_l1511_151147


namespace NUMINAMATH_GPT_Seokjin_tangerines_per_day_l1511_151136

theorem Seokjin_tangerines_per_day 
  (T_initial : ℕ) (D : ℕ) (T_remaining : ℕ) 
  (h1 : T_initial = 29) 
  (h2 : D = 8) 
  (h3 : T_remaining = 5) : 
  (T_initial - T_remaining) / D = 3 := 
by
  sorry

end NUMINAMATH_GPT_Seokjin_tangerines_per_day_l1511_151136


namespace NUMINAMATH_GPT_larger_integer_of_two_with_difference_8_and_product_168_l1511_151145

theorem larger_integer_of_two_with_difference_8_and_product_168 :
  ∃ (x y : ℕ), x > y ∧ x - y = 8 ∧ x * y = 168 ∧ x = 14 :=
by
  sorry

end NUMINAMATH_GPT_larger_integer_of_two_with_difference_8_and_product_168_l1511_151145


namespace NUMINAMATH_GPT_james_main_game_time_l1511_151143

-- Define the download time in minutes
def download_time := 10

-- Define the installation time as half the download time
def installation_time := download_time / 2

-- Define the combined time for download and installation
def combined_time := download_time + installation_time

-- Define the tutorial time as triple the combined time
def tutorial_time := combined_time * 3

-- Define the total time as the combined time plus the tutorial time
def total_time := combined_time + tutorial_time

-- Statement of the problem to prove
theorem james_main_game_time : total_time = 60 := by
  sorry

end NUMINAMATH_GPT_james_main_game_time_l1511_151143


namespace NUMINAMATH_GPT_abc_prod_eq_l1511_151187

-- Define a structure for points and triangles
structure Point :=
(x : ℝ)
(y : ℝ)

structure Triangle :=
(A B C : Point)

-- Define the angles formed by points in a triangle
def angle (A B C : Point) : ℝ := sorry

-- Define the lengths between points
def length (A B : Point) : ℝ := sorry

-- Conditions of the problem
theorem abc_prod_eq (A B C D : Point) 
  (h1 : angle A D C = angle A B C + 60)
  (h2 : angle C D B = angle C A B + 60)
  (h3 : angle B D A = angle B C A + 60) : 
  length A B * length C D = length B C * length A D :=
sorry

end NUMINAMATH_GPT_abc_prod_eq_l1511_151187


namespace NUMINAMATH_GPT_solve_quadratic_l1511_151176

theorem solve_quadratic :
    ∀ x : ℝ, x^2 - 6*x + 8 = 0 ↔ (x = 2 ∨ x = 4) :=
by
  intros x
  sorry

end NUMINAMATH_GPT_solve_quadratic_l1511_151176


namespace NUMINAMATH_GPT_unique_solution_triple_l1511_151188

theorem unique_solution_triple (x y z : ℝ) (h1 : x + y = 3) (h2 : x * y = z^3) : (x = 1.5 ∧ y = 1.5 ∧ z = 0) :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_triple_l1511_151188


namespace NUMINAMATH_GPT_brody_calculator_battery_life_l1511_151159

theorem brody_calculator_battery_life (h : ∃ t : ℕ, (3 / 4) * t + 2 + 13 = t) : ∃ t : ℕ, t = 60 :=
by
  -- Define the quarters used by Brody and the remaining battery life after the exam.
  obtain ⟨t, ht⟩ := h
  -- Simplify the equation (3/4) * t + 2 + 13 = t to get t = 60
  sorry

end NUMINAMATH_GPT_brody_calculator_battery_life_l1511_151159


namespace NUMINAMATH_GPT_problem1_problem2_l1511_151113

def M := { x : ℝ | 0 < x ∧ x < 1 }

theorem problem1 :
  { x : ℝ | |2 * x - 1| < 1 } = M :=
by
  simp [M]
  sorry

theorem problem2 (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  (a * b + 1) > (a + b) :=
by
  simp [M] at ha hb
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1511_151113


namespace NUMINAMATH_GPT_perpendicular_lines_slope_product_l1511_151102

theorem perpendicular_lines_slope_product (a : ℝ) (x y : ℝ) :
  let l1 := ax + y + 2 = 0
  let l2 := x + y = 0
  ( -a * -1 = -1 ) -> a = -1 :=
sorry

end NUMINAMATH_GPT_perpendicular_lines_slope_product_l1511_151102


namespace NUMINAMATH_GPT_k_greater_than_half_l1511_151101

-- Definition of the problem conditions
variables {a b c k : ℝ}

-- Assume a, b, c are the sides of a triangle
axiom triangle_inequality : a + b > c

-- Given condition
axiom sides_condition : a^2 + b^2 = k * c^2

-- The theorem to prove k > 0.5
theorem k_greater_than_half (h1 : a + b > c) (h2 : a^2 + b^2 = k * c^2) : k > 0.5 :=
by
  sorry

end NUMINAMATH_GPT_k_greater_than_half_l1511_151101


namespace NUMINAMATH_GPT_problem_integer_solution_l1511_151151

def satisfies_condition (n : ℤ) : Prop :=
  1 + ⌊(200 * n) / 201⌋ = ⌈(198 * n) / 200⌉

theorem problem_integer_solution :
  ∃! n : ℤ, 1 ≤ n ∧ n ≤ 20200 ∧ satisfies_condition n :=
sorry

end NUMINAMATH_GPT_problem_integer_solution_l1511_151151


namespace NUMINAMATH_GPT_Tim_pays_correct_amount_l1511_151190

def pays_in_a_week (hourly_rate : ℕ) (num_bodyguards : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  hourly_rate * num_bodyguards * hours_per_day * days_per_week

theorem Tim_pays_correct_amount :
  pays_in_a_week 20 2 8 7 = 2240 := by
  sorry

end NUMINAMATH_GPT_Tim_pays_correct_amount_l1511_151190


namespace NUMINAMATH_GPT_correct_finance_specialization_l1511_151103

-- Variables representing percentages of students specializing in different subjects
variables (students : Type) -- Type of students
           (is_specializing_finance : students → Prop) -- Predicate for finance specialization
           (is_specializing_marketing : students → Prop) -- Predicate for marketing specialization

-- Given conditions
def finance_specialization_percentage : ℝ := 0.88 -- 88% of students are taking finance specialization
def marketing_specialization_percentage : ℝ := 0.76 -- 76% of students are taking marketing specialization

-- The proof statement
theorem correct_finance_specialization (h_finance : finance_specialization_percentage = 0.88) :
  finance_specialization_percentage = 0.88 :=
by
  sorry

end NUMINAMATH_GPT_correct_finance_specialization_l1511_151103


namespace NUMINAMATH_GPT_highest_and_lowest_score_average_score_l1511_151158

def std_score : ℤ := 60
def scores : List ℤ := [36, 0, 12, -18, 20]

theorem highest_and_lowest_score 
  (highest_score : ℤ) (lowest_score : ℤ) : 
  highest_score = std_score + 36 ∧ lowest_score = std_score - 18 := 
sorry

theorem average_score (avg_score : ℤ) :
  avg_score = std_score + ((36 + 0 + 12 - 18 + 20) / 5) := 
sorry

end NUMINAMATH_GPT_highest_and_lowest_score_average_score_l1511_151158


namespace NUMINAMATH_GPT_participants_in_robbery_l1511_151171

variables (A B V G : Prop)

theorem participants_in_robbery
  (h1 : ¬G → (B ∧ ¬A))
  (h2 : V → (¬A ∧ ¬B))
  (h3 : G → B)
  (h4 : B → (A ∨ V)) :
  A ∧ B ∧ G :=
by
  sorry

end NUMINAMATH_GPT_participants_in_robbery_l1511_151171


namespace NUMINAMATH_GPT_multiplication_in_P_l1511_151153

-- Define the set P as described in the problem
def P := {x : ℕ | ∃ n : ℕ, x = n^2}

-- Prove that for all a, b in P, a * b is also in P
theorem multiplication_in_P {a b : ℕ} (ha : a ∈ P) (hb : b ∈ P) : a * b ∈ P :=
sorry

end NUMINAMATH_GPT_multiplication_in_P_l1511_151153


namespace NUMINAMATH_GPT_alice_two_turns_probability_l1511_151126

def alice_to_alice_first_turn : ℚ := 2 / 3
def alice_to_bob_first_turn : ℚ := 1 / 3
def bob_to_alice_second_turn : ℚ := 1 / 4
def bob_keeps_second_turn : ℚ := 3 / 4
def alice_keeps_second_turn : ℚ := 2 / 3

def probability_alice_keeps_twice : ℚ := alice_to_alice_first_turn * alice_keeps_second_turn
def probability_alice_bob_alice : ℚ := alice_to_bob_first_turn * bob_to_alice_second_turn

theorem alice_two_turns_probability : 
  probability_alice_keeps_twice + probability_alice_bob_alice = 37 / 108 := 
by
  sorry

end NUMINAMATH_GPT_alice_two_turns_probability_l1511_151126


namespace NUMINAMATH_GPT_minimum_value_of_x_is_4_l1511_151192

-- Given conditions
variable {x : ℝ} (hx_pos : 0 < x) (h : log x ≥ log 2 + 1/2 * log x)

-- The minimum value of x is 4
theorem minimum_value_of_x_is_4 : x ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_x_is_4_l1511_151192


namespace NUMINAMATH_GPT_weight_order_l1511_151195

variables (A B C D : ℝ) -- Representing the weights of objects A, B, C, and D as real numbers.

-- Conditions given in the problem:
axiom eq1 : A + B = C + D
axiom ineq1 : D + A > B + C
axiom ineq2 : B > A + C

-- Proof stating that the weights in ascending order are C < A < B < D.
theorem weight_order (A B C D : ℝ) : C < A ∧ A < B ∧ B < D :=
by
  -- We are not providing the proof steps here.
  sorry

end NUMINAMATH_GPT_weight_order_l1511_151195


namespace NUMINAMATH_GPT_fraction_multiplication_result_l1511_151134

theorem fraction_multiplication_result :
  (5 * 7) / 8 = 4 + 3 / 8 :=
by
  sorry

end NUMINAMATH_GPT_fraction_multiplication_result_l1511_151134


namespace NUMINAMATH_GPT_maximum_side_length_l1511_151119

theorem maximum_side_length 
    (D E F : ℝ) 
    (a b c : ℝ) 
    (h_cos : Real.cos (3 * D) + Real.cos (3 * E) + Real.cos (3 * F) = 1)
    (h_a : a = 12)
    (h_perimeter : a + b + c = 40) : 
    ∃ max_side : ℝ, max_side = 7 + Real.sqrt 23 / 2 :=
by
  sorry

end NUMINAMATH_GPT_maximum_side_length_l1511_151119


namespace NUMINAMATH_GPT_li_li_age_this_year_l1511_151152

theorem li_li_age_this_year (A B : ℕ) (h1 : A + B = 30) (h2 : A = B + 6) : B = 12 := by
  sorry

end NUMINAMATH_GPT_li_li_age_this_year_l1511_151152


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1511_151106

variable {a : Type} {M : Type} (line : a → Prop) (plane : M → Prop)

-- Assume the definitions of perpendicularity
def perp_to_plane (a : a) (M : M) : Prop := sorry -- define perpendicular to plane
def perp_to_lines_in_plane (a : a) (M : M) : Prop := sorry -- define perpendicular to countless lines

-- Mathematical statement
theorem sufficient_but_not_necessary_condition (a : a) (M : M) :
  (perp_to_plane a M → perp_to_lines_in_plane a M) ∧ ¬(perp_to_lines_in_plane a M → perp_to_plane a M) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1511_151106


namespace NUMINAMATH_GPT_symmetry_center_of_tangent_l1511_151127

noncomputable def tangentFunction (x : ℝ) : ℝ := Real.tan (2 * x - (Real.pi / 3))

theorem symmetry_center_of_tangent :
  (∃ k : ℤ, (Real.pi / 6) + (k * Real.pi / 4) = 5 * Real.pi / 12 ∧ tangentFunction ((5 * Real.pi) / 12) = 0 ) :=
sorry

end NUMINAMATH_GPT_symmetry_center_of_tangent_l1511_151127


namespace NUMINAMATH_GPT_anita_apples_l1511_151111

theorem anita_apples (num_students : ℕ) (apples_per_student : ℕ) (total_apples : ℕ) 
  (h1 : num_students = 60) 
  (h2 : apples_per_student = 6) 
  (h3 : total_apples = num_students * apples_per_student) : 
  total_apples = 360 := 
by
  sorry

end NUMINAMATH_GPT_anita_apples_l1511_151111


namespace NUMINAMATH_GPT_meaningful_fraction_condition_l1511_151197

theorem meaningful_fraction_condition (x : ℝ) : x - 2 ≠ 0 ↔ x ≠ 2 := 
by 
  sorry

end NUMINAMATH_GPT_meaningful_fraction_condition_l1511_151197


namespace NUMINAMATH_GPT_coloring_possible_l1511_151170

-- Define what it means for a graph to be planar and bipartite
def planar_graph (G : Type) : Prop := sorry
def bipartite_graph (G : Type) : Prop := sorry

-- The planar graph G results after subdivision without introducing new intersections
def subdivided_graph (G : Type) : Type := sorry

-- Main theorem to prove
theorem coloring_possible (G : Type) (h1 : planar_graph G) : 
  bipartite_graph (subdivided_graph G) :=
sorry

end NUMINAMATH_GPT_coloring_possible_l1511_151170


namespace NUMINAMATH_GPT_nonnegative_fraction_interval_l1511_151182

theorem nonnegative_fraction_interval : 
  ∀ x : ℝ, (0 ≤ x ∧ x < 3) ↔ (0 ≤ (x - 15 * x^2 + 36 * x^3) / (9 - x^3)) := by
sorry

end NUMINAMATH_GPT_nonnegative_fraction_interval_l1511_151182


namespace NUMINAMATH_GPT_square_perimeter_l1511_151144

-- First, declare the side length of the square (rectangle)
variable (s : ℝ)

-- State the conditions: the area is 484 cm^2 and it's a square
axiom area_condition : s^2 = 484
axiom is_square : ∀ (s : ℝ), s > 0

-- Define the perimeter of the square
def perimeter (s : ℝ) : ℝ := 4 * s

-- State the theorem: perimeter == 88 given the conditions
theorem square_perimeter : perimeter s = 88 :=
by 
  -- Prove the statement given the axiom 'area_condition'
  sorry

end NUMINAMATH_GPT_square_perimeter_l1511_151144


namespace NUMINAMATH_GPT_cone_lateral_surface_area_l1511_151149

theorem cone_lateral_surface_area (r l : ℝ) (h_r : r = 2) (h_l : l = 3) : 
  (r * l * Real.pi = 6 * Real.pi) := by
  sorry

end NUMINAMATH_GPT_cone_lateral_surface_area_l1511_151149


namespace NUMINAMATH_GPT_base7_to_base10_l1511_151175

theorem base7_to_base10 : (2 * 7^4 + 3 * 7^3 + 4 * 7^2 + 5 * 7^1 + 6 * 7^0) = 6068 := by
  sorry

end NUMINAMATH_GPT_base7_to_base10_l1511_151175


namespace NUMINAMATH_GPT_values_of_d_l1511_151125

theorem values_of_d (a b c d : ℕ) 
  (h : (ad - 1) / (a + 1) + (bd - 1) / (b + 1) + (cd - 1) / (c + 1) = d) : 
  d = 1 ∨ d = 2 ∨ d = 3 := 
sorry

end NUMINAMATH_GPT_values_of_d_l1511_151125


namespace NUMINAMATH_GPT_triangle_is_obtuse_l1511_151199

noncomputable def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  B = 2 * A ∧ a = 1 ∧ b = 4 / 3 ∧ (a^2 + b^2 < c^2)

theorem triangle_is_obtuse (A B C : ℝ) (a b c : ℝ) (h : triangle_ABC A B C a b c) : 
  B > π / 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_is_obtuse_l1511_151199


namespace NUMINAMATH_GPT_johns_total_packs_l1511_151186

-- Defining the conditions
def classes : ℕ := 6
def students_per_class : ℕ := 30
def packs_per_student : ℕ := 2

-- Theorem statement
theorem johns_total_packs : 
  (classes * students_per_class * packs_per_student) = 360 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_johns_total_packs_l1511_151186


namespace NUMINAMATH_GPT_expression_simplifies_l1511_151198

variable {a b : ℚ}
variable (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a ≠ b)

theorem expression_simplifies : (a^2 - b^2) / (a * b) - (a * b - b^2) / (a * b - a^2) = a / b := by
  -- TODO: Proof goes here
  sorry

end NUMINAMATH_GPT_expression_simplifies_l1511_151198


namespace NUMINAMATH_GPT_find_larger_number_l1511_151122

theorem find_larger_number (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 10) (h3 : x * y = 375) (hx : x > y) : x = 25 :=
sorry

end NUMINAMATH_GPT_find_larger_number_l1511_151122


namespace NUMINAMATH_GPT_remainder_squared_mod_five_l1511_151129

theorem remainder_squared_mod_five (n k : ℤ) (h : n = 5 * k + 3) : ((n - 1) ^ 2) % 5 = 4 :=
by
  sorry

end NUMINAMATH_GPT_remainder_squared_mod_five_l1511_151129


namespace NUMINAMATH_GPT_monotonic_increasing_interval_l1511_151112

def f (x : ℝ) : ℝ := x^2 - 2

theorem monotonic_increasing_interval :
  ∀ x y: ℝ, 0 <= x -> x <= y -> f x <= f y := 
by
  -- proof would be here
  sorry

end NUMINAMATH_GPT_monotonic_increasing_interval_l1511_151112


namespace NUMINAMATH_GPT_number_of_integers_l1511_151189

theorem number_of_integers (n : ℤ) : 
  (20 < n^2) ∧ (n^2 < 150) → 
  (∃ m : ℕ, m = 16) :=
by
  sorry

end NUMINAMATH_GPT_number_of_integers_l1511_151189


namespace NUMINAMATH_GPT_photocopy_distribution_l1511_151128

-- Define the problem setting
variables {n k : ℕ}

-- Define the theorem stating the problem
theorem photocopy_distribution :
  ∀ n k : ℕ, (n > 0) → 
  (k + n).choose (n - 1) = (k + n - 1).choose (n - 1) :=
by sorry

end NUMINAMATH_GPT_photocopy_distribution_l1511_151128


namespace NUMINAMATH_GPT_box_volume_in_cubic_yards_l1511_151165

theorem box_volume_in_cubic_yards (v_feet : ℕ) (conv_factor : ℕ) (v_yards : ℕ)
  (h1 : v_feet = 216) (h2 : conv_factor = 3) (h3 : 27 = conv_factor ^ 3) : 
  v_yards = 8 :=
by
  sorry

end NUMINAMATH_GPT_box_volume_in_cubic_yards_l1511_151165


namespace NUMINAMATH_GPT_find_a12_l1511_151121

namespace ArithmeticSequence

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence (a : ℕ → α) :=
  ∃ d : α, ∀ n : ℕ, a n = a 0 + n * d

theorem find_a12 {a : ℕ → α} (h1 : a 4 = 1) (h2 : a 7 + a 9 = 16) :
  a 12 = 15 := 
sorry

end ArithmeticSequence

end NUMINAMATH_GPT_find_a12_l1511_151121


namespace NUMINAMATH_GPT_circle_through_points_l1511_151105

theorem circle_through_points :
  ∃ (D E F : ℝ), F = 0 ∧ D = -4 ∧ E = -6 ∧ ∀ (x y : ℝ),
  (x = 0 ∧ y = 0 ∨ x = 4 ∧ y = 0 ∨ x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0 :=
by
  sorry

end NUMINAMATH_GPT_circle_through_points_l1511_151105


namespace NUMINAMATH_GPT_larger_root_eq_5_over_8_l1511_151184

noncomputable def find_larger_root : ℝ := 
    let x := ((5:ℝ) / 8)
    let y := ((23:ℝ) / 48)
    if x > y then x else y

theorem larger_root_eq_5_over_8 (x : ℝ) (y : ℝ) : 
  (x - ((5:ℝ) / 8)) * (x - ((5:ℝ) / 8)) + (x - ((5:ℝ) / 8)) * (x - ((1:ℝ) / 3)) = 0 → 
  find_larger_root = ((5:ℝ) / 8) :=
by
  intro h
  -- proof goes here
  sorry

end NUMINAMATH_GPT_larger_root_eq_5_over_8_l1511_151184


namespace NUMINAMATH_GPT_marble_problem_l1511_151185

theorem marble_problem
  (M : ℕ)
  (X : ℕ)
  (h1 : M = 18 * X)
  (h2 : M = 20 * (X - 1)) :
  M = 180 :=
by
  sorry

end NUMINAMATH_GPT_marble_problem_l1511_151185


namespace NUMINAMATH_GPT_ironing_pants_each_day_l1511_151163

-- Given conditions:
def minutes_ironing_shirt := 5 -- minutes per day
def days_per_week := 5 -- days per week
def total_minutes_ironing_4_weeks := 160 -- minutes over 4 weeks

-- Target statement to prove:
theorem ironing_pants_each_day : 
  (total_minutes_ironing_4_weeks / 4 - minutes_ironing_shirt * days_per_week) /
  days_per_week = 3 :=
by 
sorry

end NUMINAMATH_GPT_ironing_pants_each_day_l1511_151163


namespace NUMINAMATH_GPT_largest_five_digit_number_divisible_by_6_l1511_151173

theorem largest_five_digit_number_divisible_by_6 : 
  ∃ n : ℕ, n < 100000 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m < 100000 ∧ m % 6 = 0 → m ≤ n :=
by
  sorry

end NUMINAMATH_GPT_largest_five_digit_number_divisible_by_6_l1511_151173


namespace NUMINAMATH_GPT_system_of_equations_solution_exists_l1511_151168

theorem system_of_equations_solution_exists :
  ∃ (x y : ℚ), (x * y^2 - 2 * y^2 + 3 * x = 18) ∧ (3 * x * y + 5 * x - 6 * y = 24) ∧ 
                ((x = 3 ∧ y = 3) ∨ (x = 75 / 13 ∧ y = -3 / 7)) :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_exists_l1511_151168


namespace NUMINAMATH_GPT_intersection_complement_eq_l1511_151110

-- Definitions of the sets M and N
def M : Set ℝ := {0, 1, 2}
def N : Set ℝ := {x | x^2 - 3*x + 2 > 0}

-- Complements with respect to the reals
def complement_R (A : Set ℝ) : Set ℝ := {x | x ∉ A}

-- Target goal to prove
theorem intersection_complement_eq :
  M ∩ (complement_R N) = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_eq_l1511_151110


namespace NUMINAMATH_GPT_solve_p_value_l1511_151157

noncomputable def solve_for_p (n m p : ℚ) : Prop :=
  (5 / 6 = n / 90) ∧ ((m + n) / 105 = (p - m) / 150) ∧ (p = 137.5)

theorem solve_p_value (n m p : ℚ) (h1 : 5 / 6 = n / 90) (h2 : (m + n) / 105 = (p - m) / 150) : 
  p = 137.5 :=
by
  sorry

end NUMINAMATH_GPT_solve_p_value_l1511_151157


namespace NUMINAMATH_GPT_at_least_one_casket_made_by_Cellini_son_l1511_151137

-- Definitions for casket inscriptions
def golden_box := "The silver casket was made by Cellini"
def silver_box := "The golden casket was made by someone other than Cellini"

-- Predicate indicating whether a box was made by Cellini
def made_by_Cellini (box : String) : Prop :=
  box = "The golden casket was made by someone other than Cellini" ∨ box = "The silver casket was made by Cellini"

-- Our goal is to prove that at least one of the boxes was made by Cellini's son
theorem at_least_one_casket_made_by_Cellini_son :
  (¬ made_by_Cellini golden_box ∧ made_by_Cellini silver_box) ∨ (made_by_Cellini golden_box ∧ ¬ made_by_Cellini silver_box) → (¬ made_by_Cellini golden_box ∨ ¬ made_by_Cellini silver_box) :=
sorry

end NUMINAMATH_GPT_at_least_one_casket_made_by_Cellini_son_l1511_151137


namespace NUMINAMATH_GPT_wrongly_read_number_l1511_151177

theorem wrongly_read_number (initial_avg correct_avg n wrong_correct_sum : ℝ) : 
  initial_avg = 23 ∧ correct_avg = 24 ∧ n = 10 ∧ wrong_correct_sum = 36
  → ∃ (X : ℝ), 36 - X = 10 ∧ X = 26 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_wrongly_read_number_l1511_151177


namespace NUMINAMATH_GPT_rhombus_diagonals_perpendicular_not_in_rectangle_l1511_151155

-- Definitions for the rhombus
structure Rhombus :=
  (diagonals_perpendicular : Prop)

-- Definitions for the rectangle
structure Rectangle :=
  (diagonals_not_perpendicular : Prop)

-- The main proof statement
theorem rhombus_diagonals_perpendicular_not_in_rectangle 
  (R : Rhombus) 
  (Rec : Rectangle) : 
  R.diagonals_perpendicular ∧ Rec.diagonals_not_perpendicular :=
by sorry

end NUMINAMATH_GPT_rhombus_diagonals_perpendicular_not_in_rectangle_l1511_151155


namespace NUMINAMATH_GPT_hypotenuse_length_l1511_151140

theorem hypotenuse_length (a b c : ℝ) (hC : (a^2 + b^2) * (a^2 + b^2 + 1) = 12) (right_triangle : a^2 + b^2 = c^2) : 
  c = Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_hypotenuse_length_l1511_151140


namespace NUMINAMATH_GPT_sum_of_products_of_roots_eq_neg3_l1511_151123

theorem sum_of_products_of_roots_eq_neg3 {p q r s : ℂ} 
  (h : ∀ {x : ℂ}, 4 * x^4 - 8 * x^3 + 12 * x^2 - 16 * x + 9 = 0 → (x = p ∨ x = q ∨ x = r ∨ x = s)) : 
  p * q + p * r + p * s + q * r + q * s + r * s = -3 := 
sorry

end NUMINAMATH_GPT_sum_of_products_of_roots_eq_neg3_l1511_151123


namespace NUMINAMATH_GPT_range_of_m_l1511_151194

theorem range_of_m (m : ℝ) : 
  (∃ (x : ℝ), ¬((x - m) < -3) ∧ (1 + 2*x)/3 ≥ x - 1) ∧ 
  (∀ (x1 x2 x3 : ℤ), 
    (¬((x1 - m) < -3) ∧ (1 + 2 * x1)/3 ≥ x1 - 1) ∧
    (¬((x2 - m) < -3) ∧ (1 + 2 * x2)/3 ≥ x2 - 1) ∧
    (¬((x3 - m) < -3) ∧ (1 + 2 * x3)/3 ≥ x3 - 1)) →
  (4 ≤ m ∧ m < 5) :=
by 
  sorry

end NUMINAMATH_GPT_range_of_m_l1511_151194


namespace NUMINAMATH_GPT_correct_option_l1511_151118

noncomputable def M : Set ℝ := {x | x > -2}

theorem correct_option : {0} ⊆ M := 
by 
  intros x hx
  simp at hx
  simp [M]
  show x > -2
  linarith

end NUMINAMATH_GPT_correct_option_l1511_151118


namespace NUMINAMATH_GPT_no_two_champion_teams_l1511_151167

theorem no_two_champion_teams
  (T : Type) 
  (M : T -> T -> Prop)
  (superior : T -> T -> Prop)
  (champion : T -> Prop)
  (h1 : ∀ A B, M A B ∨ (∃ C, M A C ∧ M C B) → superior A B)
  (h2 : ∀ A, champion A ↔ ∀ B, superior A B)
  (h3 : ∀ A B, M A B ∨ M B A)
  : ¬ ∃ A B, champion A ∧ champion B ∧ A ≠ B := 
sorry

end NUMINAMATH_GPT_no_two_champion_teams_l1511_151167


namespace NUMINAMATH_GPT_Tim_bottle_quarts_l1511_151133

theorem Tim_bottle_quarts (ounces_per_week : ℕ) (ounces_per_quart : ℕ) (days_per_week : ℕ) (additional_ounces_per_day : ℕ) (bottles_per_day : ℕ) : 
  ounces_per_week = 812 → ounces_per_quart = 32 → days_per_week = 7 → additional_ounces_per_day = 20 → bottles_per_day = 2 → 
  ∃ quarts_per_bottle : ℝ, quarts_per_bottle = 1.5 := 
by
  intros hw ho hd ha hb
  let total_quarts_per_week := (812 : ℝ) / 32 
  let total_quarts_per_day := total_quarts_per_week / 7 
  let additional_quarts_per_day := 20 / 32 
  let quarts_from_bottles := total_quarts_per_day - additional_quarts_per_day 
  let quarts_per_bottle := quarts_from_bottles / 2 
  use quarts_per_bottle 
  sorry

end NUMINAMATH_GPT_Tim_bottle_quarts_l1511_151133


namespace NUMINAMATH_GPT_circle_touch_externally_circle_one_inside_other_without_touching_circle_completely_outside_l1511_151169

-- Definitions encapsulated in theorems with conditions and desired results
theorem circle_touch_externally {d R r : ℝ} (h1 : d = 10) (h2 : R = 8) (h3 : r = 2) : 
  d = R + r :=
by 
  rw [h1, h2, h3]
  sorry

theorem circle_one_inside_other_without_touching {d R r : ℝ} (h1 : d = 4) (h2 : R = 17) (h3 : r = 11) : 
  d < R - r :=
by 
  rw [h1, h2, h3]
  sorry

theorem circle_completely_outside {d R r : ℝ} (h1 : d = 12) (h2 : R = 5) (h3 : r = 3) : 
  d > R + r :=
by 
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_circle_touch_externally_circle_one_inside_other_without_touching_circle_completely_outside_l1511_151169


namespace NUMINAMATH_GPT_work_problem_l1511_151114

theorem work_problem (W : ℝ) (d : ℝ) :
  (1 / 40) * d * W + (28 / 35) * W = W → d = 8 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_work_problem_l1511_151114


namespace NUMINAMATH_GPT_absolute_value_condition_l1511_151164

theorem absolute_value_condition (x : ℝ) : |x + 1| + |x - 2| ≤ 5 ↔ -2 ≤ x ∧ x ≤ 3 := sorry

end NUMINAMATH_GPT_absolute_value_condition_l1511_151164


namespace NUMINAMATH_GPT_car_mpg_l1511_151104

open Nat

theorem car_mpg (x : ℕ) (h1 : ∀ (m : ℕ), m = 4 * (3 * x) -> x = 27) 
                (h2 : ∀ (d1 d2 : ℕ), d2 = (4 * d1) / 3 - d1 -> d2 = 126) 
                (h3 : ∀ g : ℕ, g = 14)
                : x = 27 := 
by
  sorry

end NUMINAMATH_GPT_car_mpg_l1511_151104


namespace NUMINAMATH_GPT_find_a_value_l1511_151148

theorem find_a_value (a : ℝ) (m : ℝ) (f g : ℝ → ℝ)
  (f_def : ∀ x, f x = Real.log x / Real.log a)
  (g_def : ∀ x, g x = (2 + m) * Real.sqrt x)
  (a_pos : 0 < a) (a_neq_one : a ≠ 1)
  (max_f : ∀ x ∈ Set.Icc (1 / 2) 16, f x ≤ 4)
  (min_f : ∀ x ∈ Set.Icc (1 / 2) 16, m ≤ f x)
  (g_increasing : ∀ x y, 0 < x → x < y → g x < g y):
  a = 2 :=
sorry

end NUMINAMATH_GPT_find_a_value_l1511_151148


namespace NUMINAMATH_GPT_product_of_legs_divisible_by_12_l1511_151179

theorem product_of_legs_divisible_by_12 
  (a b c : ℕ) 
  (h_triangle : a^2 + b^2 = c^2) 
  (h_int : ∃ a b c : ℕ, a^2 + b^2 = c^2) :
  ∃ k : ℕ, a * b = 12 * k :=
sorry

end NUMINAMATH_GPT_product_of_legs_divisible_by_12_l1511_151179


namespace NUMINAMATH_GPT_find_a_l1511_151178

variable (a : ℝ) -- Declare a as a real number.

-- Define the given conditions.
def condition1 (a : ℝ) : Prop := a^2 - 2 * a = 0
def condition2 (a : ℝ) : Prop := a ≠ 2

-- Define the theorem stating that if conditions are true, then a must be 0.
theorem find_a (h1 : condition1 a) (h2 : condition2 a) : a = 0 :=
sorry -- Proof is not provided, it needs to be constructed.

end NUMINAMATH_GPT_find_a_l1511_151178


namespace NUMINAMATH_GPT_gino_popsicle_sticks_l1511_151196

variable (my_sticks : ℕ) (total_sticks : ℕ) (gino_sticks : ℕ)

def popsicle_sticks_condition (my_sticks : ℕ) (total_sticks : ℕ) (gino_sticks : ℕ) : Prop :=
  my_sticks = 50 ∧ total_sticks = 113

theorem gino_popsicle_sticks
  (h : popsicle_sticks_condition my_sticks total_sticks gino_sticks) :
  gino_sticks = 63 :=
  sorry

end NUMINAMATH_GPT_gino_popsicle_sticks_l1511_151196


namespace NUMINAMATH_GPT_rectangular_solid_volume_l1511_151156

theorem rectangular_solid_volume 
  (x y z : ℝ)
  (h1 : x * y = 20)
  (h2 : y * z = 15)
  (h3 : x * z = 12) :
  x * y * z = 60 :=
by
  sorry

end NUMINAMATH_GPT_rectangular_solid_volume_l1511_151156


namespace NUMINAMATH_GPT_directrix_of_parabola_l1511_151130

-- Define the given conditions
def parabola_eqn (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 5

-- The problem is to show that the directrix of this parabola has the equation y = 23/12
theorem directrix_of_parabola : 
  (∃ y : ℝ, ∀ x : ℝ, parabola_eqn x = y) →

  ∃ y : ℝ, y = 23 / 12 :=
sorry

end NUMINAMATH_GPT_directrix_of_parabola_l1511_151130


namespace NUMINAMATH_GPT_minimum_value_l1511_151142

theorem minimum_value (p q r s t u v w : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s)
    (ht : 0 < t) (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) (h₁ : p * q * r * s = 16) (h₂ : t * u * v * w = 25) :
    (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 ≥ 40 := 
sorry

end NUMINAMATH_GPT_minimum_value_l1511_151142


namespace NUMINAMATH_GPT_xy_value_l1511_151150

theorem xy_value : 
  ∀ (x y : ℝ),
  (∀ (A B C : ℝ × ℝ), A = (1, 8) ∧ B = (x, y) ∧ C = (6, 3) → 
  (C.1 = (A.1 + B.1) / 2) ∧ (C.2 = (A.2 + B.2) / 2)) → 
  x * y = -22 :=
sorry

end NUMINAMATH_GPT_xy_value_l1511_151150


namespace NUMINAMATH_GPT_total_arrangements_l1511_151124

theorem total_arrangements :
  let students := 6
  let venueA := 1
  let venueB := 2
  let venueC := 3
  (students.choose venueA) * ((students - venueA).choose venueB) = 60 :=
by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_total_arrangements_l1511_151124


namespace NUMINAMATH_GPT_theater_earnings_l1511_151132

theorem theater_earnings :
  let matinee_price := 5
  let evening_price := 7
  let opening_night_price := 10
  let popcorn_price := 10
  let matinee_customers := 32
  let evening_customers := 40
  let opening_night_customers := 58
  let half_of_customers_that_bought_popcorn := 
    (matinee_customers + evening_customers + opening_night_customers) / 2
  let total_earnings := 
    (matinee_price * matinee_customers) + 
    (evening_price * evening_customers) + 
    (opening_night_price * opening_night_customers) + 
    (popcorn_price * half_of_customers_that_bought_popcorn)
  total_earnings = 1670 :=
by
  sorry

end NUMINAMATH_GPT_theater_earnings_l1511_151132


namespace NUMINAMATH_GPT_rows_of_potatoes_l1511_151193

theorem rows_of_potatoes (total_potatoes : ℕ) (seeds_per_row : ℕ) (h1 : total_potatoes = 54) (h2 : seeds_per_row = 9) : total_potatoes / seeds_per_row = 6 := 
by
  sorry

end NUMINAMATH_GPT_rows_of_potatoes_l1511_151193


namespace NUMINAMATH_GPT_license_plate_difference_l1511_151139

theorem license_plate_difference :
  let california_plates := 26^4 * 10^3
  let texas_plates := 26^3 * 10^4
  california_plates - texas_plates = 281216000 :=
by
  let california_plates := 26^4 * 10^3
  let texas_plates := 26^3 * 10^4
  have h1 : california_plates = 456976 * 1000 := by sorry
  have h2 : texas_plates = 17576 * 10000 := by sorry
  have h3 : 456976000 - 175760000 = 281216000 := by sorry
  exact h3

end NUMINAMATH_GPT_license_plate_difference_l1511_151139


namespace NUMINAMATH_GPT_marcus_baseball_cards_l1511_151131

-- Define the number of baseball cards Carter has
def carter_cards : ℕ := 152

-- Define the number of additional cards Marcus has compared to Carter
def additional_cards : ℕ := 58

-- Define the number of baseball cards Marcus has
def marcus_cards : ℕ := carter_cards + additional_cards

-- The proof statement asserting Marcus' total number of baseball cards
theorem marcus_baseball_cards : marcus_cards = 210 :=
by {
  -- This is where the proof steps would go, but we are skipping with sorry
  sorry
}

end NUMINAMATH_GPT_marcus_baseball_cards_l1511_151131


namespace NUMINAMATH_GPT_train_speed_l1511_151174

theorem train_speed (length_m : ℝ) (time_s : ℝ) 
  (h1 : length_m = 120) 
  (h2 : time_s = 3.569962336897346) 
  : (length_m / 1000) / (time_s / 3600) = 121.003 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l1511_151174


namespace NUMINAMATH_GPT_quadratic_roots_l1511_151116

noncomputable def roots_quadratic : Prop :=
  ∀ (a b : ℝ), (a + b = 7) ∧ (a * b = 7) → (a^2 + b^2 = 35)

theorem quadratic_roots (a b : ℝ) (h : a + b = 7 ∧ a * b = 7) : a^2 + b^2 = 35 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_l1511_151116


namespace NUMINAMATH_GPT_internal_angles_triangle_ABC_l1511_151117

theorem internal_angles_triangle_ABC (α β γ : ℕ) (h₁ : α + β + γ = 180)
  (h₂ : α + γ = 138) (h₃ : β + γ = 108) : (α = 72) ∧ (β = 42) ∧ (γ = 66) :=
by
  sorry

end NUMINAMATH_GPT_internal_angles_triangle_ABC_l1511_151117


namespace NUMINAMATH_GPT_sum_of_x_and_y_l1511_151108

-- Define the given angles
def angle_A : ℝ := 34
def angle_B : ℝ := 74
def angle_C : ℝ := 32

-- State the theorem
theorem sum_of_x_and_y (x y : ℝ) :
  (680 - x - y) = 720 → (x + y = 40) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_sum_of_x_and_y_l1511_151108


namespace NUMINAMATH_GPT_fraction_expression_simplifies_to_313_l1511_151115

theorem fraction_expression_simplifies_to_313 :
  (12^4 + 324) * (24^4 + 324) * (36^4 + 324) * (48^4 + 324) * (60^4 + 324) * (72^4 + 324) /
  (6^4 + 324) * (18^4 + 324) * (30^4 + 324) * (42^4 + 324) * (54^4 + 324) * (66^4 + 324) = 313 :=
by
  sorry

end NUMINAMATH_GPT_fraction_expression_simplifies_to_313_l1511_151115


namespace NUMINAMATH_GPT_inequality_solution_set_l1511_151172

variable {a b x : ℝ}

theorem inequality_solution_set (h : ∀ x : ℝ, ax - b > 0 ↔ x < -1) : 
  ∀ x : ℝ, (x-2) * (ax + b) < 0 ↔ x < 1 ∨ x > 2 :=
by sorry

end NUMINAMATH_GPT_inequality_solution_set_l1511_151172


namespace NUMINAMATH_GPT_expand_product_l1511_151138

theorem expand_product : ∀ (x : ℝ), (3 * x - 4) * (2 * x + 9) = 6 * x^2 + 19 * x - 36 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_expand_product_l1511_151138


namespace NUMINAMATH_GPT_independence_test_categorical_l1511_151181

-- Define what an independence test entails
def independence_test (X Y : Type) : Prop :=  
  ∃ (P : X → Y → Prop), ∀ x y1 y2, P x y1 → P x y2 → y1 = y2

-- Define the type of variables (categorical)
def is_categorical (V : Type) : Prop :=
  ∃ (f : V → ℕ), true

-- State the proposition that an independence test checks the relationship between categorical variables
theorem independence_test_categorical (X Y : Type) (hx : is_categorical X) (hy : is_categorical Y) :
  independence_test X Y := 
sorry

end NUMINAMATH_GPT_independence_test_categorical_l1511_151181


namespace NUMINAMATH_GPT_investment_of_c_l1511_151146

-- Definitions of given conditions
def P_b: ℝ := 4000
def diff_Pa_Pc: ℝ := 1599.9999999999995
def Ca: ℝ := 8000
def Cb: ℝ := 10000

-- Goal to be proved
theorem investment_of_c (C_c: ℝ) : 
  (∃ P_a P_c, (P_a / Ca = P_b / Cb) ∧ (P_c / C_c = P_b / Cb) ∧ (P_a - P_c = diff_Pa_Pc)) → 
  C_c = 4000 :=
sorry

end NUMINAMATH_GPT_investment_of_c_l1511_151146


namespace NUMINAMATH_GPT_range_of_a_l1511_151141

theorem range_of_a (a : ℝ) : (∃ x > 0, (2 * x - a) / (x + 1) = 1) ↔ a > -1 :=
by {
    sorry
}

end NUMINAMATH_GPT_range_of_a_l1511_151141


namespace NUMINAMATH_GPT_remainder_div_38_l1511_151154

theorem remainder_div_38 (n : ℕ) (h : n = 432 * 44) : n % 38 = 32 :=
sorry

end NUMINAMATH_GPT_remainder_div_38_l1511_151154


namespace NUMINAMATH_GPT_base8_subtraction_l1511_151166

theorem base8_subtraction : (7463 - 3154 = 4317) := by sorry

end NUMINAMATH_GPT_base8_subtraction_l1511_151166


namespace NUMINAMATH_GPT_like_terms_sum_l1511_151183

theorem like_terms_sum (m n : ℤ) (h_x : 1 = m - 2) (h_y : 2 = n + 3) : m + n = 2 :=
by
  sorry

end NUMINAMATH_GPT_like_terms_sum_l1511_151183


namespace NUMINAMATH_GPT_hyperbola_equation_l1511_151161

-- Definitions based on the conditions
def parabola_focus : (ℝ × ℝ) := (2, 0)
def point_on_hyperbola : (ℝ × ℝ) := (1, 0)
def hyperbola_center : (ℝ × ℝ) := (0, 0)
def right_focus_of_hyperbola : (ℝ × ℝ) := parabola_focus

-- Given the above definitions, we should prove that the standard equation of hyperbola C is correct
theorem hyperbola_equation :
  ∃ (a b : ℝ), (a = 1) ∧ (2^2 = a^2 + b^2) ∧
  (hyperbola_center = (0, 0)) ∧ (point_on_hyperbola = (1, 0)) →
  (x^2 - (y^2 / 3) = 1) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_equation_l1511_151161

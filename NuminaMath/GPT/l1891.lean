import Mathlib

namespace NUMINAMATH_GPT_inequality_proof_l1891_189128

-- Define the conditions and the theorem statement
variables {a b c d : ℝ}

theorem inequality_proof (h1 : c < d) (h2 : a > b) (h3 : b > 0) : a - c > b - d :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1891_189128


namespace NUMINAMATH_GPT_number_of_solutions_l1891_189115

theorem number_of_solutions :
  (∃ (xs : List ℤ), (∀ x ∈ xs, |3 * x + 4| ≤ 10) ∧ xs.length = 7) := sorry

end NUMINAMATH_GPT_number_of_solutions_l1891_189115


namespace NUMINAMATH_GPT_mean_of_six_numbers_l1891_189165

theorem mean_of_six_numbers (sum_six_numbers : ℚ) (h : sum_six_numbers = 3/4) : 
  (sum_six_numbers / 6) = 1/8 := by
  -- proof can be filled in here
  sorry

end NUMINAMATH_GPT_mean_of_six_numbers_l1891_189165


namespace NUMINAMATH_GPT_percentage_of_amount_l1891_189111

theorem percentage_of_amount :
  (0.25 * 300) = 75 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_amount_l1891_189111


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1891_189132

theorem simplify_and_evaluate (x y : ℤ) (hx : x = -1) (hy : y = 2) : 
  x^2 - 2 * (3 * y^2 - x * y) + (y^2 - 2 * x * y) = -19 := 
by
  -- Proof will go here, but it's omitted as per instructions
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1891_189132


namespace NUMINAMATH_GPT_mean_of_remaining_three_l1891_189188

theorem mean_of_remaining_three (a b c : ℝ) (h₁ : (a + b + c + 105) / 4 = 93) : (a + b + c) / 3 = 89 :=
  sorry

end NUMINAMATH_GPT_mean_of_remaining_three_l1891_189188


namespace NUMINAMATH_GPT_tagged_fish_proportion_l1891_189121

def total_fish_in_pond : ℕ := 750
def tagged_fish_first_catch : ℕ := 30
def fish_second_catch : ℕ := 50
def tagged_fish_second_catch := 2

theorem tagged_fish_proportion :
  (tagged_fish_second_catch : ℤ) * (total_fish_in_pond : ℤ) = (tagged_fish_first_catch : ℤ) * (fish_second_catch : ℤ) :=
by
  -- The statement should reflect the given proportion:
  -- T * 750 = 30 * 50
  -- Given T = 2
  sorry

end NUMINAMATH_GPT_tagged_fish_proportion_l1891_189121


namespace NUMINAMATH_GPT_correct_algorithm_description_l1891_189133

def conditions_about_algorithms (desc : String) : Prop :=
  (desc = "A" → false) ∧
  (desc = "B" → false) ∧
  (desc = "C" → true) ∧
  (desc = "D" → false)

theorem correct_algorithm_description : ∃ desc : String, 
  conditions_about_algorithms desc :=
by
  use "C"
  unfold conditions_about_algorithms
  simp
  sorry

end NUMINAMATH_GPT_correct_algorithm_description_l1891_189133


namespace NUMINAMATH_GPT_max_value_of_a_l1891_189129

variable {R : Type*} [LinearOrderedField R]

def det (a b c d : R) : R := a * d - b * c

theorem max_value_of_a (a : R) :
  (∀ x : R, det (x - 1) (a - 2) (a + 1) x ≥ 1) → a ≤ (3 / 2 : R) :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_a_l1891_189129


namespace NUMINAMATH_GPT_find_single_digit_l1891_189198

def isSingleDigit (n : ℕ) : Prop := n < 10

def repeatedDigitNumber (A : ℕ) : ℕ := 10 * A + A 

theorem find_single_digit (A : ℕ) (h1 : isSingleDigit A) (h2 : repeatedDigitNumber A + repeatedDigitNumber A = 132) : A = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_single_digit_l1891_189198


namespace NUMINAMATH_GPT_additional_time_to_walk_1_mile_l1891_189180

open Real

noncomputable def additional_time_per_mile
  (distance_child : ℝ) (time_child : ℝ)
  (distance_elderly : ℝ) (time_elderly : ℝ)
  : ℝ :=
  let speed_child := distance_child / time_child
  let time_per_mile_child := (time_child * 60) / distance_child
  let speed_elderly := distance_elderly / time_elderly
  let time_per_mile_elderly := (time_elderly * 60) / distance_elderly
  time_per_mile_elderly - time_per_mile_child

theorem additional_time_to_walk_1_mile
  (h1 : 15 = 15) (h2 : 3.5 = 3.5)
  (h3 : 10 = 10) (h4 : 4 = 4)
  : additional_time_per_mile 15 3.5 10 4 = 10 :=
  by
    sorry

end NUMINAMATH_GPT_additional_time_to_walk_1_mile_l1891_189180


namespace NUMINAMATH_GPT_fraction_representation_of_2_375_l1891_189176

theorem fraction_representation_of_2_375 : 2.375 = 19 / 8 := by
  sorry

end NUMINAMATH_GPT_fraction_representation_of_2_375_l1891_189176


namespace NUMINAMATH_GPT_work_completion_together_l1891_189170

theorem work_completion_together (man_days : ℕ) (son_days : ℕ) (together_days : ℕ) 
  (h_man : man_days = 10) (h_son : son_days = 10) : together_days = 5 :=
by sorry

end NUMINAMATH_GPT_work_completion_together_l1891_189170


namespace NUMINAMATH_GPT_floor_sqrt_30_squared_eq_25_l1891_189134

theorem floor_sqrt_30_squared_eq_25 (h1 : 5 < Real.sqrt 30) (h2 : Real.sqrt 30 < 6) : Int.floor (Real.sqrt 30) ^ 2 = 25 := 
by
  sorry

end NUMINAMATH_GPT_floor_sqrt_30_squared_eq_25_l1891_189134


namespace NUMINAMATH_GPT_diff_square_mental_math_l1891_189151

theorem diff_square_mental_math :
  75 ^ 2 - 45 ^ 2 = 3600 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_diff_square_mental_math_l1891_189151


namespace NUMINAMATH_GPT_problem_solution_l1891_189122

def equal_group_B : Prop :=
  (-2)^3 = -(2^3)

theorem problem_solution : equal_group_B := by
  sorry

end NUMINAMATH_GPT_problem_solution_l1891_189122


namespace NUMINAMATH_GPT_sarah_photos_l1891_189130

theorem sarah_photos (photos_Cristina photos_John photos_Clarissa total_slots : ℕ)
  (hCristina : photos_Cristina = 7)
  (hJohn : photos_John = 10)
  (hClarissa : photos_Clarissa = 14)
  (hTotal : total_slots = 40) :
  ∃ photos_Sarah, photos_Sarah = total_slots - (photos_Cristina + photos_John + photos_Clarissa) ∧ photos_Sarah = 9 :=
by
  sorry

end NUMINAMATH_GPT_sarah_photos_l1891_189130


namespace NUMINAMATH_GPT_number_of_subsets_l1891_189141

def num_subsets (n : ℕ) : ℕ := 2 ^ n

theorem number_of_subsets (A : Finset α) (n : ℕ) (h : A.card = n) : A.powerset.card = num_subsets n :=
by
  have : A.powerset.card = 2 ^ A.card := sorry -- Proof omitted
  rw [h] at this
  exact this

end NUMINAMATH_GPT_number_of_subsets_l1891_189141


namespace NUMINAMATH_GPT_min_value_of_expression_l1891_189142

theorem min_value_of_expression (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a^2 + a * b + a * c + b * c = 4) : 2 * a + b + c ≥ 4 :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l1891_189142


namespace NUMINAMATH_GPT_action_figure_cost_l1891_189186

def initial_figures : ℕ := 7
def total_figures_needed : ℕ := 16
def total_cost : ℕ := 72

theorem action_figure_cost :
  total_cost / (total_figures_needed - initial_figures) = 8 := by
  sorry

end NUMINAMATH_GPT_action_figure_cost_l1891_189186


namespace NUMINAMATH_GPT_committee_of_4_from_10_eq_210_l1891_189149

theorem committee_of_4_from_10_eq_210 :
  (Nat.choose 10 4) = 210 :=
by
  sorry

end NUMINAMATH_GPT_committee_of_4_from_10_eq_210_l1891_189149


namespace NUMINAMATH_GPT_strawberries_taken_out_l1891_189143

theorem strawberries_taken_out : 
  ∀ (initial_total_strawberries buckets strawberries_left_per_bucket : ℕ),
  initial_total_strawberries = 300 → 
  buckets = 5 → 
  strawberries_left_per_bucket = 40 → 
  (initial_total_strawberries / buckets - strawberries_left_per_bucket = 20) :=
by
  intros initial_total_strawberries buckets strawberries_left_per_bucket h1 h2 h3
  sorry

end NUMINAMATH_GPT_strawberries_taken_out_l1891_189143


namespace NUMINAMATH_GPT_garden_length_l1891_189167

-- Define the perimeter and breadth
def perimeter : ℕ := 900
def breadth : ℕ := 190

-- Define a function to calculate the length using given conditions
def length (P : ℕ) (B : ℕ) : ℕ := (P / 2) - B

-- Theorem stating that for the given perimeter and breadth, the length is 260.
theorem garden_length : length perimeter breadth = 260 :=
by
  -- placeholder for proof
  sorry

end NUMINAMATH_GPT_garden_length_l1891_189167


namespace NUMINAMATH_GPT_sliced_meat_cost_per_type_with_rush_shipping_l1891_189107

theorem sliced_meat_cost_per_type_with_rush_shipping:
  let original_cost := 40.0
  let rush_delivery_percentage := 0.3
  let num_types := 4
  let rush_delivery_cost := rush_delivery_percentage * original_cost
  let total_cost := original_cost + rush_delivery_cost
  let cost_per_type := total_cost / num_types
  cost_per_type = 13.0 :=
by
  sorry

end NUMINAMATH_GPT_sliced_meat_cost_per_type_with_rush_shipping_l1891_189107


namespace NUMINAMATH_GPT_lucia_hiphop_classes_l1891_189104

def cost_hiphop_class : Int := 10
def cost_ballet_class : Int := 12
def cost_jazz_class : Int := 8
def num_ballet_classes : Int := 2
def num_jazz_classes : Int := 1
def total_cost : Int := 52

def num_hiphop_classes : Int := (total_cost - (num_ballet_classes * cost_ballet_class + num_jazz_classes * cost_jazz_class)) / cost_hiphop_class

theorem lucia_hiphop_classes : num_hiphop_classes = 2 := by
  sorry

end NUMINAMATH_GPT_lucia_hiphop_classes_l1891_189104


namespace NUMINAMATH_GPT_geometric_sequence_problem_l1891_189173

noncomputable def q : ℝ := 1 + Real.sqrt 2

theorem geometric_sequence_problem (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n)
  (h_geom : ∀ n, a (n + 1) = (q : ℝ) * a n)
  (h_cond : a 2 = a 0 + 2 * a 1) :
  (a 8 + a 9) / (a 6 + a 7) = 3 + 2 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l1891_189173


namespace NUMINAMATH_GPT_new_person_weight_l1891_189138

theorem new_person_weight
  (initial_avg_weight : ℝ := 57)
  (num_people : ℕ := 8)
  (weight_to_replace : ℝ := 55)
  (weight_increase_first : ℝ := 1.5)
  (weight_increase_second : ℝ := 2)
  (weight_increase_third : ℝ := 2.5)
  (weight_increase_fourth : ℝ := 3)
  (weight_increase_fifth : ℝ := 3.5)
  (weight_increase_sixth : ℝ := 4)
  (weight_increase_seventh : ℝ := 4.5) :
  ∃ x : ℝ, x = 67 :=
by
  sorry

end NUMINAMATH_GPT_new_person_weight_l1891_189138


namespace NUMINAMATH_GPT_range_of_function_l1891_189161

open Set

noncomputable def log_base_2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem range_of_function (S : Set ℝ) : 
    S = {y : ℝ | ∃ x : ℝ, x ≥ 1 ∧ y = 2 + log_base_2 x} 
    ↔ S = {y : ℝ | y ≥ 2} :=
by 
  sorry

end NUMINAMATH_GPT_range_of_function_l1891_189161


namespace NUMINAMATH_GPT_maximum_candies_karlson_l1891_189144

theorem maximum_candies_karlson (n : ℕ) (h_n : n = 40) :
  ∃ k, k = 780 :=
by
  sorry

end NUMINAMATH_GPT_maximum_candies_karlson_l1891_189144


namespace NUMINAMATH_GPT_find_x_y_sum_of_squares_l1891_189174

theorem find_x_y_sum_of_squares :
  ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ (xy + x + y = 47) ∧ (x^2 * y + x * y^2 = 506) ∧ (x^2 + y^2 = 101) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_x_y_sum_of_squares_l1891_189174


namespace NUMINAMATH_GPT_cosine_of_angle_in_third_quadrant_l1891_189113

theorem cosine_of_angle_in_third_quadrant (B : ℝ) (hB : B ∈ Set.Ioo (π : ℝ) (3 * π / 2)) (hSinB : Real.sin B = -5 / 13) :
  Real.cos B = -12 / 13 :=
sorry

end NUMINAMATH_GPT_cosine_of_angle_in_third_quadrant_l1891_189113


namespace NUMINAMATH_GPT_remaining_speed_20_kmph_l1891_189196

theorem remaining_speed_20_kmph
  (D T : ℝ)
  (H1 : (2/3 * D) / (1/3 * T) = 80)
  (H2 : T = D / 40) :
  (D / 3) / (2/3 * T) = 20 :=
by 
  sorry

end NUMINAMATH_GPT_remaining_speed_20_kmph_l1891_189196


namespace NUMINAMATH_GPT_find_all_a_l1891_189123

def digit_sum_base_4038 (n : ℕ) : ℕ :=
  n.digits 4038 |>.sum

def is_good (n : ℕ) : Prop :=
  2019 ∣ digit_sum_base_4038 n

def is_bad (n : ℕ) : Prop :=
  ¬ is_good n

def satisfies_condition (seq : ℕ → ℕ) (a : ℝ) : Prop :=
  (∀ n, seq n ≤ a * n) ∧ ∀ n, seq n = seq (n + 1) + 1

theorem find_all_a (a : ℝ) (h1 : 1 ≤ a) :
  (∀ seq, (∀ n m, n ≠ m → seq n ≠ seq m) → satisfies_condition seq a →
    ∃ n_infinitely, is_bad (seq n_infinitely)) ↔ a < 2019 := sorry

end NUMINAMATH_GPT_find_all_a_l1891_189123


namespace NUMINAMATH_GPT_find_pairs_l1891_189124

theorem find_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  (3 * a + 1 ∣ 4 * b - 1) ∧ (2 * b + 1 ∣ 3 * a - 1) ↔ (a = 2 ∧ b = 2) := 
by 
  sorry

end NUMINAMATH_GPT_find_pairs_l1891_189124


namespace NUMINAMATH_GPT_sqrt_product_l1891_189126

theorem sqrt_product (a b : ℝ) (ha : a = 20) (hb : b = 1/5) : Real.sqrt a * Real.sqrt b = 2 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_product_l1891_189126


namespace NUMINAMATH_GPT_earthquake_relief_team_selection_l1891_189160

theorem earthquake_relief_team_selection : 
    ∃ (ways : ℕ), ways = 590 ∧ 
      ∃ (orthopedic neurosurgeon internist : ℕ), 
      orthopedic + neurosurgeon + internist = 5 ∧ 
      1 ≤ orthopedic ∧ 1 ≤ neurosurgeon ∧ 1 ≤ internist ∧
      orthopedic ≤ 3 ∧ neurosurgeon ≤ 4 ∧ internist ≤ 5 := 
  sorry

end NUMINAMATH_GPT_earthquake_relief_team_selection_l1891_189160


namespace NUMINAMATH_GPT_sum_of_squares_of_ages_l1891_189169

theorem sum_of_squares_of_ages {a b c : ℕ} (h1 : 5 * a + b = 3 * c) (h2 : 3 * c^2 = 2 * a^2 + b^2) 
  (relatively_prime : Nat.gcd (Nat.gcd a b) c = 1) : 
  a^2 + b^2 + c^2 = 374 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_ages_l1891_189169


namespace NUMINAMATH_GPT_sum_first_nine_terms_arithmetic_sequence_l1891_189137

theorem sum_first_nine_terms_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : ∀ n, a (n + 1) - a n = (a 2 - a 1))
  (h1 : a 1 + a 4 + a 7 = 45)
  (h2 : a 3 + a 6 + a 9 = 27) :
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9) = 108 := 
sorry

end NUMINAMATH_GPT_sum_first_nine_terms_arithmetic_sequence_l1891_189137


namespace NUMINAMATH_GPT_slope_of_line_l1891_189116

theorem slope_of_line : ∀ (x y : ℝ), (x / 4 - y / 3 = 1) → ((3 * x / 4) - 3) = 0 → (y = (3 / 4) * x - 3) :=
by 
  intros x y h_eq h_slope 
  sorry

end NUMINAMATH_GPT_slope_of_line_l1891_189116


namespace NUMINAMATH_GPT_equation_solution_l1891_189197

theorem equation_solution : ∃ x : ℝ, (3 / 20) + (3 / x) = (8 / x) + (1 / 15) ∧ x = 60 :=
by
  use 60
  -- skip the proof
  sorry

end NUMINAMATH_GPT_equation_solution_l1891_189197


namespace NUMINAMATH_GPT_triangle_area_eq_l1891_189102

/--
Given:
1. The base of the triangle is 4 meters.
2. The height of the triangle is 5 meters.

Prove:
The area of the triangle is 10 square meters.
-/
theorem triangle_area_eq (base height : ℝ) (h_base : base = 4) (h_height : height = 5) : 
  (base * height / 2) = 10 := by
  sorry

end NUMINAMATH_GPT_triangle_area_eq_l1891_189102


namespace NUMINAMATH_GPT_equal_real_roots_iff_c_is_nine_l1891_189190

theorem equal_real_roots_iff_c_is_nine (c : ℝ) : (∃ x : ℝ, x^2 + 6 * x + c = 0 ∧ ∃ Δ, Δ = 6^2 - 4 * 1 * c ∧ Δ = 0) ↔ c = 9 :=
by
  sorry

end NUMINAMATH_GPT_equal_real_roots_iff_c_is_nine_l1891_189190


namespace NUMINAMATH_GPT_solve_quadratic_equation_l1891_189157

theorem solve_quadratic_equation (x : ℝ) :
  x^2 - 2 * x - 8 = 0 ↔ x = 4 ∨ x = -2 := by
sorry

end NUMINAMATH_GPT_solve_quadratic_equation_l1891_189157


namespace NUMINAMATH_GPT_priyas_age_l1891_189148

/-- 
  Let P be Priya's current age, and F be her father's current age. 
  Given:
  1. F = P + 31
  2. (P + 8) + (F + 8) = 69
  Prove: Priya's current age P is 11.
-/
theorem priyas_age 
  (P F : ℕ) 
  (h1 : F = P + 31) 
  (h2 : (P + 8) + (F + 8) = 69) 
  : P = 11 :=
by
  sorry

end NUMINAMATH_GPT_priyas_age_l1891_189148


namespace NUMINAMATH_GPT_intersection_or_parallel_lines_l1891_189184

structure Triangle (Point : Type) :=
  (A B C : Point)

structure Plane (Point : Type) :=
  (P1 P2 P3 P4 : Point)

variables {Point : Type}
variables (triABC triA1B1C1 : Triangle Point)
variables (plane1 plane2 plane3 : Plane Point)

-- Intersection conditions
variable (AB_intersects_A1B1 : (triABC.A, triABC.B) = (triA1B1C1.A, triA1B1C1.B))
variable (BC_intersects_B1C1 : (triABC.B, triABC.C) = (triA1B1C1.B, triA1B1C1.C))
variable (CA_intersects_C1A1 : (triABC.C, triABC.A) = (triA1B1C1.C, triA1B1C1.A))

theorem intersection_or_parallel_lines :
  ∃ P : Point, (
    (∃ A1 : Point, (triABC.A, A1) = (P, P)) ∧
    (∃ B1 : Point, (triABC.B, B1) = (P, P)) ∧
    (∃ C1 : Point, (triABC.C, C1) = (P, P))
  ) ∨ (
    (∃ d1 d2 d3 : Point, 
      (∀ A1 B1 C1 : Point,
        (triABC.A, A1) = (d1, d1) ∧ 
        (triABC.B, B1) = (d2, d2) ∧ 
        (triABC.C, C1) = (d3, d3)
      )
    )
  ) := by
  sorry

end NUMINAMATH_GPT_intersection_or_parallel_lines_l1891_189184


namespace NUMINAMATH_GPT_gcd_459_357_l1891_189166

-- Define the numbers involved
def num1 := 459
def num2 := 357

-- State the proof problem
theorem gcd_459_357 : Int.gcd num1 num2 = 51 := by
  sorry

end NUMINAMATH_GPT_gcd_459_357_l1891_189166


namespace NUMINAMATH_GPT_janice_class_girls_l1891_189125

theorem janice_class_girls : ∃ (g b : ℕ), (3 * b = 4 * g) ∧ (g + b + 2 = 32) ∧ (g = 13) := by
  sorry

end NUMINAMATH_GPT_janice_class_girls_l1891_189125


namespace NUMINAMATH_GPT_widgets_unloaded_l1891_189117
-- We import the necessary Lean library for general mathematical purposes.

-- We begin the lean statement for our problem.
theorem widgets_unloaded (n_doo n_geegaw n_widget n_yamyam : ℕ) :
  (2^n_doo) * (11^n_geegaw) * (5^n_widget) * (7^n_yamyam) = 104350400 →
  n_widget = 2 := by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_widgets_unloaded_l1891_189117


namespace NUMINAMATH_GPT_problem_equivalent_l1891_189199

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | -1 < x ∧ x < 2}

theorem problem_equivalent :
  {x : ℝ | x ≥ 2} = (U \ (M ∪ N)) := 
by sorry

end NUMINAMATH_GPT_problem_equivalent_l1891_189199


namespace NUMINAMATH_GPT_total_cookies_l1891_189189

-- Definitions from conditions
def cookies_per_guest : ℕ := 2
def number_of_guests : ℕ := 5

-- Theorem statement that needs to be proved
theorem total_cookies : cookies_per_guest * number_of_guests = 10 := by
  -- We skip the proof since only the statement is required
  sorry

end NUMINAMATH_GPT_total_cookies_l1891_189189


namespace NUMINAMATH_GPT_expression_value_l1891_189140

theorem expression_value (a b : ℝ) (h : a^2 * b^2 / (a^4 - 2 * b^4) = 1) : 
  (a^2 - b^2) / (a^2 + b^2) = 1 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_expression_value_l1891_189140


namespace NUMINAMATH_GPT_solve_for_q_l1891_189114

theorem solve_for_q (n m q: ℚ)
  (h1 : 3 / 4 = n / 88)
  (h2 : 3 / 4 = (m + n) / 100)
  (h3 : 3 / 4 = (q - m) / 150) :
  q = 121.5 :=
sorry

end NUMINAMATH_GPT_solve_for_q_l1891_189114


namespace NUMINAMATH_GPT_evaluate_expression_l1891_189119

theorem evaluate_expression:
  (-2)^2002 + (-1)^2003 + 2^2004 + (-1)^2005 = 3 * 2^2002 - 2 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1891_189119


namespace NUMINAMATH_GPT_min_value_l1891_189154

theorem min_value (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : a + b + c = 1) :
  (a + 1)^2 + 4 * b^2 + 9 * c^2 ≥ 144 / 49 :=
sorry

end NUMINAMATH_GPT_min_value_l1891_189154


namespace NUMINAMATH_GPT_part1_part2_l1891_189147

noncomputable def set_A : Set ℝ := {x | x^2 + 4 * x = 0}
noncomputable def set_B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + a^2 - 1 = 0}

theorem part1 (a : ℝ) : (set_A ∪ set_B a = set_A ∩ set_B a) → a = 1 :=
sorry

theorem part2 (a : ℝ) : (set_A ∪ set_B a = set_A) → (a ≤ -1 ∨ a = 1) :=
sorry

end NUMINAMATH_GPT_part1_part2_l1891_189147


namespace NUMINAMATH_GPT_gcd_of_54000_and_36000_l1891_189182

theorem gcd_of_54000_and_36000 : Nat.gcd 54000 36000 = 18000 := 
by sorry

end NUMINAMATH_GPT_gcd_of_54000_and_36000_l1891_189182


namespace NUMINAMATH_GPT_sum_of_digits_of_N_l1891_189185

theorem sum_of_digits_of_N :
  ∃ N : ℕ, (N * (N + 1)) / 2 = 3003 ∧ N.digits 10 = [7, 7] :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_N_l1891_189185


namespace NUMINAMATH_GPT_snow_at_mrs_hilts_house_l1891_189120

theorem snow_at_mrs_hilts_house
    (snow_at_school : ℕ)
    (extra_snow_at_house : ℕ) 
    (school_snow_amount : snow_at_school = 17) 
    (extra_snow_amount : extra_snow_at_house = 12) :
  snow_at_school + extra_snow_at_house = 29 := 
by
  sorry

end NUMINAMATH_GPT_snow_at_mrs_hilts_house_l1891_189120


namespace NUMINAMATH_GPT_cost_of_advanced_purchase_ticket_l1891_189179

theorem cost_of_advanced_purchase_ticket
  (x : ℝ)
  (door_cost : ℝ := 14)
  (total_tickets : ℕ := 140)
  (total_money : ℝ := 1720)
  (advanced_tickets_sold : ℕ := 100)
  (door_tickets_sold : ℕ := total_tickets - advanced_tickets_sold)
  (advanced_revenue : ℝ := advanced_tickets_sold * x)
  (door_revenue : ℝ := door_tickets_sold * door_cost)
  (total_revenue : ℝ := advanced_revenue + door_revenue) :
  total_revenue = total_money → x = 11.60 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_cost_of_advanced_purchase_ticket_l1891_189179


namespace NUMINAMATH_GPT_geometric_sum_common_ratio_l1891_189156

theorem geometric_sum_common_ratio (a₁ a₂ : ℕ) (q : ℕ) (S₃ : ℕ)
  (h1 : S₃ = a₁ + 3 * a₂)
  (h2: S₃ = a₁ * (1 + q + q^2)) :
  q = 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sum_common_ratio_l1891_189156


namespace NUMINAMATH_GPT_main_diagonal_squares_second_diagonal_composite_third_diagonal_composite_l1891_189164

-- Problem Statement in Lean 4

theorem main_diagonal_squares (k : ℕ) : ∃ m : ℕ, (4 * k * (k + 1) + 1 = m * m) := 
sorry

theorem second_diagonal_composite (k : ℕ) (hk : k ≥ 1) : ∃ a b : ℕ, a ≠ 1 ∧ b ≠ 1 ∧ (4 * (2 * k * (2 * k - 1) - 1) + 1 = a * b) :=
sorry

theorem third_diagonal_composite (k : ℕ) : ∃ a b : ℕ, a ≠ 1 ∧ b ≠ 1 ∧ (4 * ((4 * k + 3) * (4 * k - 1)) + 1 = a * b) :=
sorry

end NUMINAMATH_GPT_main_diagonal_squares_second_diagonal_composite_third_diagonal_composite_l1891_189164


namespace NUMINAMATH_GPT_range_of_a_in_circle_l1891_189112

theorem range_of_a_in_circle (a : ℝ) : 
  ((1 - a)^2 + (1 + a)^2 < 4) ↔ (-1 < a ∧ a < 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_in_circle_l1891_189112


namespace NUMINAMATH_GPT_value_g2_l1891_189163

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g (g (x - y)) = g x * g y - g x + g y - x^3 * y^3

theorem value_g2 : g 2 = 8 :=
by sorry

end NUMINAMATH_GPT_value_g2_l1891_189163


namespace NUMINAMATH_GPT_exp_addition_property_l1891_189103

theorem exp_addition_property (x y : ℝ) : (Real.exp (x + y)) = (Real.exp x) * (Real.exp y) := 
sorry

end NUMINAMATH_GPT_exp_addition_property_l1891_189103


namespace NUMINAMATH_GPT_math_problem_l1891_189145

noncomputable def proof : Prop :=
  ∀ (a b : ℝ), a ≠ 0 → b ≠ 0 →
  ( (1 / a + 1 / b) / (1 / a - 1 / b) = 1001 ) →
  ((a + b) / (a - b) = 1001)

theorem math_problem : proof := 
  by
    intros a b h₁ h₂ h₃
    sorry

end NUMINAMATH_GPT_math_problem_l1891_189145


namespace NUMINAMATH_GPT_probability_A_and_B_selected_l1891_189106

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end NUMINAMATH_GPT_probability_A_and_B_selected_l1891_189106


namespace NUMINAMATH_GPT_figure_can_be_cut_and_reassembled_into_square_l1891_189150

-- Define the conditions
def is_square_area (n: ℕ) : Prop := ∃ k: ℕ, k * k = n

def can_form_square (area: ℕ) : Prop :=
area = 18 ∧ ¬ is_square_area area

-- The proof statement
theorem figure_can_be_cut_and_reassembled_into_square (area: ℕ) (hf: area = 18): 
  can_form_square area → ∃ (part1 part2 part3: Set (ℕ × ℕ)), true :=
by
  sorry

end NUMINAMATH_GPT_figure_can_be_cut_and_reassembled_into_square_l1891_189150


namespace NUMINAMATH_GPT_hyperbola_properties_l1891_189139

-- Definitions from the conditions
def line_l (x y : ℝ) : Prop := 4 * x - 3 * y + 20 = 0
def asymptote_l (x y : ℝ) : Prop := 4 * x - 3 * y = 0
def foci_on_x_axis (x y : ℝ) : Prop := y = 0

-- Standard equation of the hyperbola
def hyperbola_equation (x y : ℝ) : Prop := (x^2 / 9) - (y^2 / 16) = 1

-- Define eccentricity
def eccentricity := 5 / 3

-- Proof statement
theorem hyperbola_properties :
  (∃ x y : ℝ, line_l x y ∧ foci_on_x_axis x y) →
  (∃ x y : ℝ, asymptote_l x y) →
  ∃ x y : ℝ, hyperbola_equation x y ∧ eccentricity = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_properties_l1891_189139


namespace NUMINAMATH_GPT_part1_part2_l1891_189101

def f (x : ℝ) : ℝ := x^2 - 1

theorem part1 (m x : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) (ineq : 4 * m^2 * |f x| + 4 * f m ≤ |f (x-1)|) : 
    -1/2 ≤ m ∧ m ≤ 1/2 := 
sorry

theorem part2 (x1 : ℝ) (hx1 : 1 ≤ x1 ∧ x1 ≤ 2) : 
    (∃ x2 : ℝ, 1 ≤ x2 ∧ x2 ≤ 2 ∧ f x1 = |2 * f x2 - a * x2|) →
    (0 ≤ a ∧ a ≤ 3/2 ∨ a = 3) := 
sorry

end NUMINAMATH_GPT_part1_part2_l1891_189101


namespace NUMINAMATH_GPT_find_number_l1891_189183

theorem find_number (x : ℤ) (h : 300 + 8 * x = 340) : x = 5 := by
  sorry

end NUMINAMATH_GPT_find_number_l1891_189183


namespace NUMINAMATH_GPT_unique_solution_k_l1891_189168

theorem unique_solution_k (k : ℚ) : (∀ x : ℚ, x ≠ -2 → (x + 3)/(k*x - 2) = x) ↔ k = -3/4 :=
sorry

end NUMINAMATH_GPT_unique_solution_k_l1891_189168


namespace NUMINAMATH_GPT_vector_decomposition_l1891_189100

noncomputable def x : ℝ × ℝ × ℝ := (5, 15, 0)
noncomputable def p : ℝ × ℝ × ℝ := (1, 0, 5)
noncomputable def q : ℝ × ℝ × ℝ := (-1, 3, 2)
noncomputable def r : ℝ × ℝ × ℝ := (0, -1, 1)

theorem vector_decomposition : x = (4 : ℝ) • p + (-1 : ℝ) • q + (-18 : ℝ) • r :=
by
  sorry

end NUMINAMATH_GPT_vector_decomposition_l1891_189100


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1891_189191

theorem simplify_and_evaluate (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 0) (h3 : x ≠ 1) :
  (x = -1) → ( (x-1) / (x^2 - 2*x + 1) / ((x^2 + x - 1) / (x-1) - (x + 1)) - 1 / (x - 2) = -2 / 3 ) :=
by 
  intro hx
  rw [hx]
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1891_189191


namespace NUMINAMATH_GPT_crayon_difference_l1891_189193

theorem crayon_difference:
  let karen := 639
  let cindy := 504
  let peter := 752
  let rachel := 315
  max karen (max cindy (max peter rachel)) - min karen (min cindy (min peter rachel)) = 437 :=
by
  sorry

end NUMINAMATH_GPT_crayon_difference_l1891_189193


namespace NUMINAMATH_GPT_nelly_payment_is_correct_l1891_189158

-- Given definitions and conditions
def joes_bid : ℕ := 160000
def additional_amount : ℕ := 2000

-- Nelly's total payment
def nellys_payment : ℕ := (3 * joes_bid) + additional_amount

-- The proof statement we need to prove that Nelly's payment equals 482000 dollars
theorem nelly_payment_is_correct : nellys_payment = 482000 :=
by
  -- This is a placeholder for the actual proof.
  -- You can fill in the formal proof here.
  sorry

end NUMINAMATH_GPT_nelly_payment_is_correct_l1891_189158


namespace NUMINAMATH_GPT_find_x_if_perpendicular_l1891_189172

-- Given definitions and conditions
def a : ℝ × ℝ := (-5, 1)
def b (x : ℝ) : ℝ × ℝ := (2, x)

-- Statement to be proved
theorem find_x_if_perpendicular (x : ℝ) :
  (a.1 * (b x).1 + a.2 * (b x).2 = 0) → x = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_x_if_perpendicular_l1891_189172


namespace NUMINAMATH_GPT_intersection_A_B_l1891_189110

def A : Set ℝ := {x | abs x <= 1}

def B : Set ℝ := {y | ∃ x : ℝ, y = x^2}

theorem intersection_A_B :
  (A ∩ B) = {x | 0 ≤ x ∧ x ≤ 1} := sorry

end NUMINAMATH_GPT_intersection_A_B_l1891_189110


namespace NUMINAMATH_GPT_daily_earnings_r_l1891_189175

theorem daily_earnings_r (p q r s : ℝ)
  (h1 : p + q + r + s = 300)
  (h2 : p + r = 120)
  (h3 : q + r = 130)
  (h4 : s + r = 200)
  (h5 : p + s = 116.67) : 
  r = 75 :=
by
  sorry

end NUMINAMATH_GPT_daily_earnings_r_l1891_189175


namespace NUMINAMATH_GPT_cubic_inequality_l1891_189177

theorem cubic_inequality (x p q : ℝ) (h : x^3 + p * x + q = 0) : 4 * q * x ≤ p^2 := 
  sorry

end NUMINAMATH_GPT_cubic_inequality_l1891_189177


namespace NUMINAMATH_GPT_snacks_in_3h40m_l1891_189159

def minutes_in_hours (hours : ℕ) : ℕ := hours * 60

def snacks_in_time (total_minutes : ℕ) (snack_interval : ℕ) : ℕ := total_minutes / snack_interval

theorem snacks_in_3h40m : snacks_in_time (minutes_in_hours 3 + 40) 20 = 11 :=
by
  sorry

end NUMINAMATH_GPT_snacks_in_3h40m_l1891_189159


namespace NUMINAMATH_GPT_find_g_two_fifths_l1891_189192

noncomputable def g : ℝ → ℝ :=
sorry -- The function g(x) is not explicitly defined.

theorem find_g_two_fifths :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g x = 0 → g 0 = 0) ∧
  (∀ x y, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g (1 - x) = 1 - g x) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g (x / 5) = g x / 3)
  → g (2 / 5) = 1 / 3 :=
sorry

end NUMINAMATH_GPT_find_g_two_fifths_l1891_189192


namespace NUMINAMATH_GPT_car_rental_cost_per_mile_l1891_189178

theorem car_rental_cost_per_mile
    (daily_rental_cost : ℕ)
    (daily_budget : ℕ)
    (miles_limit : ℕ)
    (cost_per_mile : ℕ) :
    daily_rental_cost = 30 →
    daily_budget = 76 →
    miles_limit = 200 →
    cost_per_mile = (daily_budget - daily_rental_cost) * 100 / miles_limit →
    cost_per_mile = 23 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  norm_num at h4
  exact h4

end NUMINAMATH_GPT_car_rental_cost_per_mile_l1891_189178


namespace NUMINAMATH_GPT_negation_of_existential_statement_l1891_189187

theorem negation_of_existential_statement :
  (¬ ∃ x : ℝ, x ≥ 1 ∨ x > 2) ↔ ∀ x : ℝ, x < 1 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_existential_statement_l1891_189187


namespace NUMINAMATH_GPT_point_on_x_axis_l1891_189136

theorem point_on_x_axis (A B C D : ℝ × ℝ) : B = (3,0) → B.2 = 0 :=
by
  intros h
  subst h
  exact rfl

end NUMINAMATH_GPT_point_on_x_axis_l1891_189136


namespace NUMINAMATH_GPT_find_a_l1891_189155

theorem find_a (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_b : b = 1)
    (h_ab_ccb : (10 * a + b)^2 = 100 * c + 10 * c + b) (h_ccb_gt_300 : 100 * c + 10 * c + b > 300) :
    a = 2 :=
sorry

end NUMINAMATH_GPT_find_a_l1891_189155


namespace NUMINAMATH_GPT_solve_log_eq_l1891_189195

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem solve_log_eq (x : ℝ) (hx1 : x + 1 > 0) (hx2 : x - 1 > 0) :
  log_base (x + 1) (x^3 - 9 * x + 8) * log_base (x - 1) (x + 1) = 3 ↔ x = 3 := by
  sorry

end NUMINAMATH_GPT_solve_log_eq_l1891_189195


namespace NUMINAMATH_GPT_find_m_l1891_189146

variables (a : ℕ → ℝ) (r : ℝ) (m : ℕ)

-- Define the conditions of the problem
def exponential_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * r

def condition_1 (a : ℕ → ℝ) (r : ℝ) : Prop :=
  a 5 * a 6 + a 4 * a 7 = 18

def condition_2 (a : ℕ → ℝ) (m : ℕ) : Prop :=
  a 1 * a m = 9

-- The theorem to prove based on the given conditions
theorem find_m
  (h_exp : exponential_sequence a r)
  (h_r_ne_1 : r ≠ 1)
  (h_cond1 : condition_1 a r)
  (h_cond2 : condition_2 a m) :
  m = 10 :=
sorry

end NUMINAMATH_GPT_find_m_l1891_189146


namespace NUMINAMATH_GPT_joan_books_correct_l1891_189153

def sam_books : ℕ := 110
def total_books : ℕ := 212

def joan_books : ℕ := total_books - sam_books

theorem joan_books_correct : joan_books = 102 := by
  sorry

end NUMINAMATH_GPT_joan_books_correct_l1891_189153


namespace NUMINAMATH_GPT_area_of_R_sum_m_n_l1891_189135

theorem area_of_R_sum_m_n  (s : ℕ) 
  (square_area : ℕ) 
  (rectangle1_area : ℕ)
  (rectangle2_area : ℕ) :
  square_area = 4 → rectangle1_area = 8 → rectangle2_area = 2 → s = 6 → 
  36 - (square_area + rectangle1_area + rectangle2_area) = 22 :=
by
  intros
  sorry

end NUMINAMATH_GPT_area_of_R_sum_m_n_l1891_189135


namespace NUMINAMATH_GPT_carpet_covering_cost_l1891_189131

noncomputable def carpet_cost (floor_length floor_width carpet_length carpet_width carpet_cost_per_square : ℕ) : ℕ :=
  let floor_area := floor_length * floor_width
  let carpet_area := carpet_length * carpet_width
  let num_of_squares := floor_area / carpet_area
  num_of_squares * carpet_cost_per_square

theorem carpet_covering_cost :
  carpet_cost 6 10 2 2 15 = 225 :=
by
  sorry

end NUMINAMATH_GPT_carpet_covering_cost_l1891_189131


namespace NUMINAMATH_GPT_find_fourth_number_l1891_189105

variable (a : ℕ → ℕ)

theorem find_fourth_number (h₁ : a 7 = 42) (h₂ : a 9 = 110)
    (h₃ : ∀ n, n ≥ 3 → a n = a (n-1) + a (n-2)) : a 4 = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_fourth_number_l1891_189105


namespace NUMINAMATH_GPT_intersection_point_is_correct_l1891_189108

def line1 (x y : ℝ) := x - 2 * y + 7 = 0
def line2 (x y : ℝ) := 2 * x + y - 1 = 0

theorem intersection_point_is_correct : line1 (-1) 3 ∧ line2 (-1) 3 :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_is_correct_l1891_189108


namespace NUMINAMATH_GPT_shop_dimension_is_100_l1891_189181

-- Given conditions
def monthly_rent : ℕ := 1300
def annual_rent_per_sqft : ℕ := 156

-- Define annual rent
def annual_rent : ℕ := monthly_rent * 12

-- Define dimension to prove
def dimension_of_shop : ℕ := annual_rent / annual_rent_per_sqft

-- The theorem statement
theorem shop_dimension_is_100 :
  dimension_of_shop = 100 :=
by
  sorry

end NUMINAMATH_GPT_shop_dimension_is_100_l1891_189181


namespace NUMINAMATH_GPT_set_S_infinite_l1891_189127

-- Definition of a power
def is_power (n : ℕ) : Prop := 
  ∃ (a k : ℕ), a > 0 ∧ k ≥ 2 ∧ n = a^k

-- Definition of the set S, those integers which cannot be expressed as the sum of two powers
def in_S (n : ℕ) : Prop := 
  ¬ ∃ (a b k m : ℕ), a > 0 ∧ b > 0 ∧ k ≥ 2 ∧ m ≥ 2 ∧ n = a^k + b^m

-- The theorem statement asserting that S is infinite
theorem set_S_infinite : Infinite {n : ℕ | in_S n} :=
sorry

end NUMINAMATH_GPT_set_S_infinite_l1891_189127


namespace NUMINAMATH_GPT_right_triangle_acute_angles_l1891_189109

theorem right_triangle_acute_angles (α β : ℝ) 
  (h1 : α + β = 90)
  (h2 : ∀ (δ1 δ2 ε1 ε2 : ℝ), δ1 + ε1 = 135 ∧ δ1 / ε1 = 13 / 17 
                       ∧ ε2 = 180 - ε1 ∧ δ2 = 180 - δ1) :
  α = 63 ∧ β = 27 := 
  sorry

end NUMINAMATH_GPT_right_triangle_acute_angles_l1891_189109


namespace NUMINAMATH_GPT_find_special_integers_l1891_189118

theorem find_special_integers (n : ℕ) (h : n > 1) :
  (∀ d, d ∣ n ∧ d > 1 → ∃ a r, a > 0 ∧ r > 1 ∧ d = a^r + 1) ↔ (n = 10 ∨ ∃ a, a > 0 ∧ n = a^2 + 1) :=
by
  sorry

end NUMINAMATH_GPT_find_special_integers_l1891_189118


namespace NUMINAMATH_GPT_length_of_faster_train_l1891_189152

/-- Define the speeds of the trains in kmph -/
def speed_faster_train := 180 -- in kmph
def speed_slower_train := 90  -- in kmph

/-- Convert speeds to m/s -/
def kmph_to_mps (speed : ℕ) : ℕ := speed * 5 / 18

/-- Define the relative speed in m/s -/
def relative_speed := kmph_to_mps speed_faster_train - kmph_to_mps speed_slower_train

/-- Define the time it takes for the faster train to cross the man in seconds -/
def crossing_time := 15 -- in seconds

/-- Define the length of the train calculation in meters -/
noncomputable def length_faster_train := relative_speed * crossing_time

theorem length_of_faster_train :
  length_faster_train = 375 :=
by
  sorry

end NUMINAMATH_GPT_length_of_faster_train_l1891_189152


namespace NUMINAMATH_GPT_find_number_l1891_189171

theorem find_number (x : ℝ) : 2.75 + 0.003 + x = 2.911 -> x = 0.158 := 
by
  intros h
  sorry

end NUMINAMATH_GPT_find_number_l1891_189171


namespace NUMINAMATH_GPT_egg_count_l1891_189162

theorem egg_count :
  ∃ x : ℕ, 
    (∀ e1 e10 e100 : ℤ, 
      (e1 = 1 ∨ e1 = -1) →
      (e10 = 10 ∨ e10 = -10) →
      (e100 = 100 ∨ e100 = -100) →
      7 * x + e1 + e10 + e100 = 3162) → 
    x = 439 :=
by 
  sorry

end NUMINAMATH_GPT_egg_count_l1891_189162


namespace NUMINAMATH_GPT_ali_less_nada_l1891_189194

variable (Ali Nada John : ℕ)

theorem ali_less_nada
  (h_total : Ali + Nada + John = 67)
  (h_john_nada : John = 4 * Nada)
  (h_john : John = 48) :
  Nada - Ali = 5 :=
by
  sorry

end NUMINAMATH_GPT_ali_less_nada_l1891_189194

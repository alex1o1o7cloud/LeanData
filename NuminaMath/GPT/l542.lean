import Mathlib

namespace NUMINAMATH_GPT_alpha_beta_inequality_l542_54218

theorem alpha_beta_inequality (α β : ℝ) (h1 : -1 < α) (h2 : α < β) (h3 : β < 1) : 
  -2 < α - β ∧ α - β < 0 := 
sorry

end NUMINAMATH_GPT_alpha_beta_inequality_l542_54218


namespace NUMINAMATH_GPT_altered_solution_water_amount_l542_54289

def initial_bleach_ratio := 2
def initial_detergent_ratio := 40
def initial_water_ratio := 100

def new_bleach_to_detergent_ratio := 3 * initial_bleach_ratio
def new_detergent_to_water_ratio := initial_detergent_ratio / 2

def detergent_amount := 60
def water_amount := 75

theorem altered_solution_water_amount :
  (initial_detergent_ratio / new_detergent_to_water_ratio) * detergent_amount / new_bleach_to_detergent_ratio = water_amount :=
by
  sorry

end NUMINAMATH_GPT_altered_solution_water_amount_l542_54289


namespace NUMINAMATH_GPT_hypothesis_test_l542_54222

def X : List ℕ := [3, 4, 6, 10, 13, 17]
def Y : List ℕ := [1, 2, 5, 7, 16, 20, 22]

def alpha : ℝ := 0.01
def W_lower : ℕ := 24
def W_upper : ℕ := 60
def W1 : ℕ := 41

-- stating the null hypothesis test condition
theorem hypothesis_test : (24 < 41) ∧ (41 < 60) :=
by
  sorry

end NUMINAMATH_GPT_hypothesis_test_l542_54222


namespace NUMINAMATH_GPT_geometric_series_sum_l542_54243

theorem geometric_series_sum :
  let a := 4 / 5
  let r := 4 / 5
  let n := 15
  let S := (a * (1 - r^n)) / (1 - r)
  S = 117775277204 / 30517578125 := by
  let a := 4 / 5
  let r := 4 / 5
  let n := 15
  let S := (a * (1 - r^n)) / (1 - r)
  have : S = 117775277204 / 30517578125 := sorry
  exact this

end NUMINAMATH_GPT_geometric_series_sum_l542_54243


namespace NUMINAMATH_GPT_unique_positive_integers_sum_l542_54204

noncomputable def x : ℝ := Real.sqrt ((Real.sqrt 77) / 3 + 5 / 3)

theorem unique_positive_integers_sum :
  ∃ (a b c : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c),
    x^100 = 3 * x^98 + 17 * x^96 + 13 * x^94 - 2 * x^50 + (a : ℝ) * x^46 + (b : ℝ) * x^44 + (c : ℝ) * x^40
    ∧ a + b + c = 167 := by
  sorry

end NUMINAMATH_GPT_unique_positive_integers_sum_l542_54204


namespace NUMINAMATH_GPT_remainder_of_sum_l542_54229

theorem remainder_of_sum (x y z : ℕ) (h1 : x % 15 = 6) (h2 : y % 15 = 9) (h3 : z % 15 = 3) : 
  (x + y + z) % 15 = 3 := 
  sorry

end NUMINAMATH_GPT_remainder_of_sum_l542_54229


namespace NUMINAMATH_GPT_catering_budget_l542_54256

namespace CateringProblem

variables (s c : Nat) (cost_steak cost_chicken : Nat)

def total_guests (s c : Nat) : Prop := s + c = 80

def steak_to_chicken_ratio (s c : Nat) : Prop := s = 3 * c

def total_cost (s c cost_steak cost_chicken : Nat) : Nat := s * cost_steak + c * cost_chicken

theorem catering_budget :
  ∃ (s c : Nat), (total_guests s c) ∧ (steak_to_chicken_ratio s c) ∧ (total_cost s c 25 18) = 1860 :=
by
  sorry

end CateringProblem

end NUMINAMATH_GPT_catering_budget_l542_54256


namespace NUMINAMATH_GPT_find_point_B_find_line_BC_l542_54299

-- Define the coordinates of point A
def point_A : ℝ × ℝ := (2, -1)

-- Define the equation of the median on side AB
def median_AB (x y : ℝ) : Prop := x + 3 * y = 6

-- Define the equation of the internal angle bisector of ∠ABC
def bisector_BC (x y : ℝ) : Prop := x - y = -1

-- Prove the coordinates of point B
theorem find_point_B :
  (a b : ℝ) →
  (median_AB ((a + 2) / 2) ((b - 1) / 2)) →
  (a - b = -1) →
  a = 5 / 2 ∧ b = 7 / 2 :=
sorry

-- Define the line equation BC
def line_BC (x y : ℝ) : Prop := x - 9 * y + 29 = 0

-- Prove the equation of the line containing side BC
theorem find_line_BC :
  (x0 y0 : ℝ) →
  bisector_BC x0 y0 →
  (x0, y0) = (-2, 3) →
  line_BC x0 y0 :=
sorry

end NUMINAMATH_GPT_find_point_B_find_line_BC_l542_54299


namespace NUMINAMATH_GPT_product_of_terms_l542_54227

variable {α : Type*} [LinearOrderedField α]

namespace GeometricSequence

def is_geometric_sequence (a : ℕ → α) :=
  ∃ r : α, ∀ n : ℕ, a (n + 1) = r * a n

theorem product_of_terms (a : ℕ → α) (r : α) (h_geo : is_geometric_sequence a) :
  (a 4) * (a 8) = 16 → (a 2) * (a 10) = 16 :=
by
  intro h1
  sorry

end GeometricSequence

end NUMINAMATH_GPT_product_of_terms_l542_54227


namespace NUMINAMATH_GPT_evaluate_custom_operation_l542_54254

def custom_operation (A B : ℕ) : ℕ :=
  (A + 2 * B) * (A - B)

theorem evaluate_custom_operation : custom_operation 7 5 = 34 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_custom_operation_l542_54254


namespace NUMINAMATH_GPT_probability_of_matching_pair_l542_54274
-- Import the necessary library for probability and combinatorics

def probability_matching_pair (pairs : ℕ) (total_shoes : ℕ) : ℚ :=
  if total_shoes = 2 * pairs then
    (pairs : ℚ) / ((total_shoes * (total_shoes - 1) / 2) : ℚ)
  else 0

theorem probability_of_matching_pair (pairs := 6) (total_shoes := 12) : 
  probability_matching_pair pairs total_shoes = 1 / 11 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_matching_pair_l542_54274


namespace NUMINAMATH_GPT_colors_used_l542_54225

theorem colors_used (total_blocks number_per_color : ℕ) (h1 : total_blocks = 196) (h2 : number_per_color = 14) : 
  total_blocks / number_per_color = 14 :=
by
  sorry

end NUMINAMATH_GPT_colors_used_l542_54225


namespace NUMINAMATH_GPT_system_sampling_arithmetic_sequence_l542_54284

theorem system_sampling_arithmetic_sequence :
  ∃ (seq : Fin 5 → ℕ), seq 0 = 8 ∧ seq 3 = 104 ∧ seq 1 = 40 ∧ seq 2 = 72 ∧ seq 4 = 136 ∧ 
    (∀ n m : Fin 5, 0 < n.val - m.val → seq n.val = seq m.val + 32 * (n.val - m.val)) :=
sorry

end NUMINAMATH_GPT_system_sampling_arithmetic_sequence_l542_54284


namespace NUMINAMATH_GPT_coeff_of_quadratic_term_eq_neg5_l542_54221

theorem coeff_of_quadratic_term_eq_neg5 (a b c : ℝ) (h_eq : -5 * x^2 + 5 * x + 6 = a * x^2 + b * x + c) :
  a = -5 :=
by
  sorry

end NUMINAMATH_GPT_coeff_of_quadratic_term_eq_neg5_l542_54221


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l542_54208

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x = 0 → (x^2 - 2 * x = 0)) ∧ (∃ y : ℝ, y ≠ 0 ∧ y ^ 2 - 2 * y = 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l542_54208


namespace NUMINAMATH_GPT_problem_statement_l542_54215

theorem problem_statement (a b : ℝ) (h : a < b) : a - b < 0 :=
sorry

end NUMINAMATH_GPT_problem_statement_l542_54215


namespace NUMINAMATH_GPT_range_of_log2_sin_squared_l542_54207

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

noncomputable def sin_squared_log_range (x : ℝ) : ℝ :=
  log2 ((Real.sin x) ^ 2)

theorem range_of_log2_sin_squared (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ Real.pi) :
  ∃ y, y = sin_squared_log_range x ∧ y ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_log2_sin_squared_l542_54207


namespace NUMINAMATH_GPT_initial_yellow_hard_hats_count_l542_54214

noncomputable def initial_yellow_hard_hats := 24

theorem initial_yellow_hard_hats_count
  (initial_pink: ℕ)
  (initial_green: ℕ)
  (carl_pink: ℕ)
  (john_pink: ℕ)
  (john_green: ℕ)
  (total_remaining: ℕ)
  (remaining_pink: ℕ)
  (remaining_green: ℕ)
  (initial_yellow: ℕ) :
  initial_pink = 26 →
  initial_green = 15 →
  carl_pink = 4 →
  john_pink = 6 →
  john_green = 2 * john_pink →
  total_remaining = 43 →
  remaining_pink = initial_pink - carl_pink - john_pink →
  remaining_green = initial_green - john_green →
  initial_yellow = total_remaining - remaining_pink - remaining_green →
  initial_yellow = initial_yellow_hard_hats :=
by
  intros
  sorry

end NUMINAMATH_GPT_initial_yellow_hard_hats_count_l542_54214


namespace NUMINAMATH_GPT_total_chairs_l542_54244

theorem total_chairs (living_room_chairs kitchen_chairs : ℕ) (h1 : living_room_chairs = 3) (h2 : kitchen_chairs = 6) :
  living_room_chairs + kitchen_chairs = 9 := by
  sorry

end NUMINAMATH_GPT_total_chairs_l542_54244


namespace NUMINAMATH_GPT_EquivalenceStatements_l542_54217

-- Define real numbers and sets P, Q
variables {x a b c : ℝ} {P Q : Set ℝ}

-- Prove the necessary equivalences
theorem EquivalenceStatements :
  ((x > 1) → (abs x > 1)) ∧ ((∃ x, x < -1) → (abs x > 1)) ∧
  ((a ∈ P ∩ Q) ↔ (a ∈ P ∧ a ∈ Q)) ∧
  (¬ (∀ x : ℝ, x^2 + x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 + x + 1 < 0)) ∧
  (x = 1 ↔ a + b + c = 0) :=
by
  sorry

end NUMINAMATH_GPT_EquivalenceStatements_l542_54217


namespace NUMINAMATH_GPT_extreme_value_h_tangent_to_both_l542_54251

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a*x + 1
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := Real.log x - a
noncomputable def h (x : ℝ) : ℝ := f x 1 - g x 1

theorem extreme_value_h : h (1/2) = 11/4 + Real.log 2 := by
  sorry

theorem tangent_to_both : ∀ (a : ℝ), ∃ x₁ x₂ : ℝ, (2 * x₁ + a = 1 / x₂) ∧ 
  ((x₁ = (1 / (2 * x₂)) - (a / 2)) ∧ (a ≥ -1)) := by
  sorry

end NUMINAMATH_GPT_extreme_value_h_tangent_to_both_l542_54251


namespace NUMINAMATH_GPT_exponent_of_4_l542_54283

theorem exponent_of_4 (x : ℕ) (h₁ : (1 / 4 : ℚ) ^ 2 = 1 / 16) (h₂ : 16384 * (1 / 16 : ℚ) = 1024) :
  4 ^ x = 1024 → x = 5 :=
by
  sorry

end NUMINAMATH_GPT_exponent_of_4_l542_54283


namespace NUMINAMATH_GPT_new_apps_added_l542_54219

theorem new_apps_added (x : ℕ) (h1 : 15 + x - (x + 1) = 14) : x = 0 :=
by
  sorry

end NUMINAMATH_GPT_new_apps_added_l542_54219


namespace NUMINAMATH_GPT_wrong_value_l542_54275

-- Definitions based on the conditions
def initial_mean : ℝ := 32
def corrected_mean : ℝ := 32.5
def num_observations : ℕ := 50
def correct_observation : ℝ := 48

-- We need to prove that the wrong value of the observation was 23
theorem wrong_value (sum_initial : ℝ) (sum_corrected : ℝ) : 
  sum_initial = num_observations * initial_mean ∧ 
  sum_corrected = num_observations * corrected_mean →
  48 - (sum_corrected - sum_initial) = 23 :=
by
  sorry

end NUMINAMATH_GPT_wrong_value_l542_54275


namespace NUMINAMATH_GPT_xyz_inequality_l542_54255

theorem xyz_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  (x / (y + z) + y / (z + x) + z / (x + y) ≥ 3 / 2) :=
sorry

end NUMINAMATH_GPT_xyz_inequality_l542_54255


namespace NUMINAMATH_GPT_right_triangle_perimeter_l542_54226

theorem right_triangle_perimeter
  (a b : ℝ)
  (h_area : 0.5 * 30 * b = 150)
  (h_leg : a = 30) :
  a + b + Real.sqrt (a^2 + b^2) = 40 + 10 * Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_perimeter_l542_54226


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l542_54286

theorem arithmetic_sequence_sum (a_n : ℕ → ℝ) (h1 : a_n 1 + a_n 2 + a_n 3 + a_n 4 = 30) 
                               (h2 : a_n 1 + a_n 4 = a_n 2 + a_n 3) :
  a_n 2 + a_n 3 = 15 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l542_54286


namespace NUMINAMATH_GPT_cistern_water_breadth_l542_54238

theorem cistern_water_breadth 
  (length width : ℝ) (wet_surface_area : ℝ) 
  (hl : length = 9) (hw : width = 6) (hwsa : wet_surface_area = 121.5) : 
  ∃ h : ℝ, 54 + 18 * h + 12 * h = 121.5 ∧ h = 2.25 := 
by 
  sorry

end NUMINAMATH_GPT_cistern_water_breadth_l542_54238


namespace NUMINAMATH_GPT_fruit_bowl_l542_54263

variable {A P B : ℕ}

theorem fruit_bowl : (P = A + 2) → (B = P + 3) → (A + P + B = 19) → B = 9 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_fruit_bowl_l542_54263


namespace NUMINAMATH_GPT_vertex_x_coordinate_l542_54281

theorem vertex_x_coordinate (a b c : ℝ) :
  (∀ x, x = 0 ∨ x = 4 ∨ x = 7 →
    (0 ≤ x ∧ x ≤ 7 →
      (x = 0 → c = 1) ∧
      (x = 4 → 16 * a + 4 * b + c = 1) ∧
      (x = 7 → 49 * a + 7 * b + c = 5))) →
  (2 * x = 2 * 2 - b / a) ∧ (0 ≤ x ∧ x ≤ 7) :=
sorry

end NUMINAMATH_GPT_vertex_x_coordinate_l542_54281


namespace NUMINAMATH_GPT_distinct_x_intercepts_l542_54258

theorem distinct_x_intercepts : 
  let f (x : ℝ) := ((x - 8) * (x^2 + 4*x + 3))
  (∃ x1 x2 x3 : ℝ, f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) :=
by
  sorry

end NUMINAMATH_GPT_distinct_x_intercepts_l542_54258


namespace NUMINAMATH_GPT_smallest_positive_number_among_options_l542_54205

theorem smallest_positive_number_among_options :
  (10 > 3 * Real.sqrt 11) →
  (51 > 10 * Real.sqrt 26) →
  min (10 - 3 * Real.sqrt 11) (51 - 10 * Real.sqrt 26) = 51 - 10 * Real.sqrt 26 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_smallest_positive_number_among_options_l542_54205


namespace NUMINAMATH_GPT_seq_eleven_l542_54277

noncomputable def seq (n : ℕ) : ℤ := sorry

axiom seq_add (p q : ℕ) (hp : 0 < p) (hq : 0 < q) : seq (p + q) = seq p + seq q
axiom seq_two : seq 2 = -6

theorem seq_eleven : seq 11 = -33 := by
  sorry

end NUMINAMATH_GPT_seq_eleven_l542_54277


namespace NUMINAMATH_GPT_television_price_reduction_l542_54233

variable (P : ℝ) (F : ℝ)
variable (h : F = 0.56 * P - 50)

theorem television_price_reduction :
  F / P = 0.56 - 50 / P :=
by {
  sorry
}

end NUMINAMATH_GPT_television_price_reduction_l542_54233


namespace NUMINAMATH_GPT_intersection_complement_B_and_A_l542_54250

open Set Real

def A : Set ℝ := { x | x^2 - 4 * x + 3 < 0 }
def B : Set ℝ := { x | x > 2 }
def CR_B : Set ℝ := { x | x ≤ 2 }

theorem intersection_complement_B_and_A : CR_B ∩ A = { x | 1 < x ∧ x ≤ 2 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_B_and_A_l542_54250


namespace NUMINAMATH_GPT_water_tank_capacity_l542_54296

theorem water_tank_capacity (rate : ℝ) (time : ℝ) (fraction : ℝ) (capacity : ℝ) : 
(rate = 10) → (time = 300) → (fraction = 3/4) → 
(rate * time = fraction * capacity) → 
capacity = 4000 := 
by
  intros h_rate h_time h_fraction h_equation
  rw [h_rate, h_time, h_fraction] at h_equation
  linarith

end NUMINAMATH_GPT_water_tank_capacity_l542_54296


namespace NUMINAMATH_GPT_fred_earned_from_car_wash_l542_54247

def weekly_allowance : ℕ := 16
def spent_on_movies : ℕ := weekly_allowance / 2
def amount_after_movies : ℕ := weekly_allowance - spent_on_movies
def final_amount : ℕ := 14
def earned_from_car_wash : ℕ := final_amount - amount_after_movies

theorem fred_earned_from_car_wash : earned_from_car_wash = 6 := by
  sorry

end NUMINAMATH_GPT_fred_earned_from_car_wash_l542_54247


namespace NUMINAMATH_GPT_area_of_triangle_LEF_l542_54293

noncomputable
def radius : ℝ := 10
def chord_length : ℝ := 10
def diameter_parallel_chord : Prop := True -- this condition ensures EF is parallel to LM
def LZ_length : ℝ := 20
def collinear_points : Prop := True -- this condition ensures L, M, O, Z are collinear

theorem area_of_triangle_LEF : 
  radius = 10 ∧
  chord_length = 10 ∧
  diameter_parallel_chord ∧
  LZ_length = 20 ∧ 
  collinear_points →
  (∃ area : ℝ, area = 50 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_LEF_l542_54293


namespace NUMINAMATH_GPT_general_formula_l542_54297

def sum_of_terms (a : ℕ → ℕ) (n : ℕ) : ℕ := 3 / 2 * a n - 3

def sequence_term (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  if n = 0 then 6 
  else a (n - 1) * 3

theorem general_formula (a : ℕ → ℕ) (n : ℕ) :
  (∀ n, sum_of_terms a n = 3 / 2 * a n - 3) →
  (∀ n, n = 0 → a n = 6) →
  (∀ n, n > 0 → a n = a (n - 1) * 3) →
  a n = 2 * 3^n := by
  sorry

end NUMINAMATH_GPT_general_formula_l542_54297


namespace NUMINAMATH_GPT_arcsin_neg_one_l542_54246

theorem arcsin_neg_one : Real.arcsin (-1) = -Real.pi / 2 := by
  sorry

end NUMINAMATH_GPT_arcsin_neg_one_l542_54246


namespace NUMINAMATH_GPT_triangle_side_length_l542_54200

theorem triangle_side_length (A B : ℝ) (b : ℝ) (a : ℝ) 
  (hA : A = 60) (hB : B = 45) (hb : b = 2) 
  (h : a = b * (Real.sin A) / (Real.sin B)) :
  a = Real.sqrt 6 := by
  sorry

end NUMINAMATH_GPT_triangle_side_length_l542_54200


namespace NUMINAMATH_GPT_josh_remaining_marbles_l542_54231

def initial_marbles : ℕ := 16
def lost_marbles : ℕ := 7
def remaining_marbles : ℕ := 9

theorem josh_remaining_marbles : initial_marbles - lost_marbles = remaining_marbles := by
  sorry

end NUMINAMATH_GPT_josh_remaining_marbles_l542_54231


namespace NUMINAMATH_GPT_spies_denounced_each_other_l542_54264

theorem spies_denounced_each_other :
  ∃ (pairs : Finset (ℕ × ℕ)), pairs.card ≥ 10 ∧ 
  (∀ (u v : ℕ), (u, v) ∈ pairs → (v, u) ∈ pairs) :=
sorry

end NUMINAMATH_GPT_spies_denounced_each_other_l542_54264


namespace NUMINAMATH_GPT_max_min_values_of_function_l542_54257

theorem max_min_values_of_function :
  ∀ (x : ℝ), -5 ≤ 4 * Real.sin x + 3 * Real.cos x ∧ 4 * Real.sin x + 3 * Real.cos x ≤ 5 :=
by
  sorry

end NUMINAMATH_GPT_max_min_values_of_function_l542_54257


namespace NUMINAMATH_GPT_f_is_monotonic_decreasing_l542_54278

noncomputable def f (x : ℝ) : ℝ := Real.sin (1/2 * x + Real.pi / 6)

theorem f_is_monotonic_decreasing : ∀ x y : ℝ, (2 * Real.pi / 3 ≤ x ∧ x ≤ 8 * Real.pi / 3) → (2 * Real.pi / 3 ≤ y ∧ y ≤ 8 * Real.pi / 3) → x < y → f y ≤ f x :=
sorry

end NUMINAMATH_GPT_f_is_monotonic_decreasing_l542_54278


namespace NUMINAMATH_GPT_max_value_of_f_l542_54216

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 + 4 * (Real.cos x)

theorem max_value_of_f : ∃ x : ℝ, f x ≤ 4 :=
sorry

end NUMINAMATH_GPT_max_value_of_f_l542_54216


namespace NUMINAMATH_GPT_square_side_length_l542_54267

theorem square_side_length :
  ∀ (w l : ℕ) (area : ℕ),
  w = 9 → l = 27 → area = w * l →
  ∃ s : ℝ, s^2 = area ∧ s = 9 * Real.sqrt 3 :=
by
  intros w l area hw hl harea
  sorry

end NUMINAMATH_GPT_square_side_length_l542_54267


namespace NUMINAMATH_GPT_reimbursement_calculation_l542_54245

variable (total_paid : ℕ) (pieces : ℕ) (cost_per_piece : ℕ)

theorem reimbursement_calculation
  (h1 : total_paid = 20700)
  (h2 : pieces = 150)
  (h3 : cost_per_piece = 134) :
  total_paid - (pieces * cost_per_piece) = 600 := 
by
  sorry

end NUMINAMATH_GPT_reimbursement_calculation_l542_54245


namespace NUMINAMATH_GPT_saree_discount_l542_54242

theorem saree_discount (x : ℝ) : 
  let original_price := 495
  let final_price := 378.675
  let discounted_price := original_price * ((100 - x) / 100) * 0.9
  discounted_price = final_price -> x = 15 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_saree_discount_l542_54242


namespace NUMINAMATH_GPT_bleach_contains_chlorine_l542_54213

noncomputable def element_in_bleach (mass_percentage : ℝ) (substance : String) : String :=
  if mass_percentage = 31.08 ∧ substance = "sodium hypochlorite" then "Chlorine"
  else "unknown"

theorem bleach_contains_chlorine : element_in_bleach 31.08 "sodium hypochlorite" = "Chlorine" :=
by
  sorry

end NUMINAMATH_GPT_bleach_contains_chlorine_l542_54213


namespace NUMINAMATH_GPT_quadratic_equation_transformation_l542_54249

theorem quadratic_equation_transformation (x : ℝ) :
  (-5 * x ^ 2 = 2 * x + 10) →
  (x ^ 2 + (2 / 5) * x + 2 = 0) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_quadratic_equation_transformation_l542_54249


namespace NUMINAMATH_GPT_road_trip_ratio_l542_54260

theorem road_trip_ratio (D R: ℝ) (h1 : 1 / 2 * D = 40) (h2 : 2 * (D + R * D + 40) = 560 - (D + R * D + 40)) :
  R = 5 / 6 := by
  sorry

end NUMINAMATH_GPT_road_trip_ratio_l542_54260


namespace NUMINAMATH_GPT_henry_books_donation_l542_54262

theorem henry_books_donation
  (initial_books : ℕ := 99)
  (room_books : ℕ := 21)
  (coffee_table_books : ℕ := 4)
  (cookbook_books : ℕ := 18)
  (boxes : ℕ := 3)
  (picked_up_books : ℕ := 12)
  (final_books : ℕ := 23) :
  (initial_books - final_books + picked_up_books - (room_books + coffee_table_books + cookbook_books)) / boxes = 15 :=
by
  sorry

end NUMINAMATH_GPT_henry_books_donation_l542_54262


namespace NUMINAMATH_GPT_gen_sequence_term_l542_54209

theorem gen_sequence_term (a : ℕ → ℕ) (n : ℕ) (h1 : a 1 = 1) (h2 : ∀ k, a (k + 1) = 3 * a k + 1) :
  a n = (3^n - 1) / 2 := by
  sorry

end NUMINAMATH_GPT_gen_sequence_term_l542_54209


namespace NUMINAMATH_GPT_square_area_max_l542_54288

theorem square_area_max (perimeter : ℝ) (h_perimeter : perimeter = 32) : 
  ∃ (area : ℝ), area = 64 :=
by
  sorry

end NUMINAMATH_GPT_square_area_max_l542_54288


namespace NUMINAMATH_GPT_cost_of_bought_movie_l542_54271

theorem cost_of_bought_movie 
  (ticket_cost : ℝ)
  (ticket_count : ℕ)
  (rental_cost : ℝ)
  (total_spent : ℝ)
  (bought_movie_cost : ℝ) :
  ticket_cost = 10.62 →
  ticket_count = 2 →
  rental_cost = 1.59 →
  total_spent = 36.78 →
  bought_movie_cost = total_spent - (ticket_cost * ticket_count + rental_cost) →
  bought_movie_cost = 13.95 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_cost_of_bought_movie_l542_54271


namespace NUMINAMATH_GPT_sales_tax_difference_l542_54268

theorem sales_tax_difference :
  let price : ℝ := 30
  let tax_rate1 : ℝ := 0.0675
  let tax_rate2 : ℝ := 0.055
  let sales_tax1 : ℝ := price * tax_rate1
  let sales_tax2 : ℝ := price * tax_rate2
  let difference : ℝ := sales_tax1 - sales_tax2
  difference = 0.375 :=
by
  let price : ℝ := 30
  let tax_rate1 : ℝ := 0.0675
  let tax_rate2 : ℝ := 0.055
  let sales_tax1 : ℝ := price * tax_rate1
  let sales_tax2 : ℝ := price * tax_rate2
  let difference : ℝ := sales_tax1 - sales_tax2
  exact sorry

end NUMINAMATH_GPT_sales_tax_difference_l542_54268


namespace NUMINAMATH_GPT_number_of_fills_l542_54261

-- Definitions based on conditions
def needed_flour : ℚ := 4 + 3 / 4
def cup_capacity : ℚ := 1 / 3

-- The proof statement
theorem number_of_fills : (needed_flour / cup_capacity).ceil = 15 := by
  sorry

end NUMINAMATH_GPT_number_of_fills_l542_54261


namespace NUMINAMATH_GPT_Bo_needs_to_learn_per_day_l542_54235

theorem Bo_needs_to_learn_per_day
  (total_flashcards : ℕ)
  (known_percentage : ℚ)
  (days_to_learn : ℕ)
  (h1 : total_flashcards = 800)
  (h2 : known_percentage = 0.20)
  (h3 : days_to_learn = 40) : 
  total_flashcards * (1 - known_percentage) / days_to_learn = 16 := 
by
  sorry

end NUMINAMATH_GPT_Bo_needs_to_learn_per_day_l542_54235


namespace NUMINAMATH_GPT_leading_digit_not_necessarily_one_l542_54237

-- Define a condition to check if the leading digit of a number is the same
def same_leading_digit (x: ℕ) (n: ℕ) : Prop :=
  (Nat.digits 10 x).head? = (Nat.digits 10 (x^n)).head?

-- Theorem stating the digit does not need to be 1 under given conditions
theorem leading_digit_not_necessarily_one :
  (∃ x: ℕ, x > 1 ∧ same_leading_digit x 2 ∧ same_leading_digit x 3) ∧ 
  (∃ x: ℕ, x > 1 ∧ ∀ n: ℕ, 1 ≤ n ∧ n ≤ 2015 → same_leading_digit x n) :=
sorry

end NUMINAMATH_GPT_leading_digit_not_necessarily_one_l542_54237


namespace NUMINAMATH_GPT_students_need_to_walk_distance_l542_54265

-- Define distance variables and the relationships
def teacher_initial_distance : ℝ := 235
def xiao_ma_initial_distance : ℝ := 87
def xiao_lu_initial_distance : ℝ := 59
def xiao_zhou_initial_distance : ℝ := 26
def speed_ratio : ℝ := 1.5

-- Prove the distance x students need to walk
theorem students_need_to_walk_distance (x : ℝ) :
  teacher_initial_distance - speed_ratio * x =
  (xiao_ma_initial_distance - x) + (xiao_lu_initial_distance - x) + (xiao_zhou_initial_distance - x) →
  x = 42 :=
by
  sorry

end NUMINAMATH_GPT_students_need_to_walk_distance_l542_54265


namespace NUMINAMATH_GPT_workshop_worker_allocation_l542_54272

theorem workshop_worker_allocation :
  ∃ (x y : ℕ), 
    x + y = 22 ∧
    6 * x = 5 * y ∧
    x = 10 ∧ y = 12 :=
by
  sorry

end NUMINAMATH_GPT_workshop_worker_allocation_l542_54272


namespace NUMINAMATH_GPT_seventh_grade_male_students_l542_54206

theorem seventh_grade_male_students:
  ∃ x : ℤ, (48 = x + (4*x)/5 + 3) ∧ x = 25 :=
by
  sorry

end NUMINAMATH_GPT_seventh_grade_male_students_l542_54206


namespace NUMINAMATH_GPT_ship_passengers_round_trip_tickets_l542_54285

theorem ship_passengers_round_trip_tickets (total_passengers : ℕ) (p1 : ℝ) (p2 : ℝ) :
  (p1 = 0.25 * total_passengers) ∧ (p2 = 0.6 * (p * total_passengers)) →
  (p * total_passengers = 62.5 / 100 * total_passengers) :=
by
  sorry

end NUMINAMATH_GPT_ship_passengers_round_trip_tickets_l542_54285


namespace NUMINAMATH_GPT_distribute_tourists_l542_54280

theorem distribute_tourists (guides tourists : ℕ) (hguides : guides = 3) (htourists : tourists = 8) :
  ∃ k, k = 5796 := by
  sorry

end NUMINAMATH_GPT_distribute_tourists_l542_54280


namespace NUMINAMATH_GPT_range_of_a_l542_54295

theorem range_of_a (x y a : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) (hy : 1 ≤ y ∧ y ≤ 2)
    (hxy : x * y = 2) (h : ∀ x y, 2 - x ≥ a / (4 - y)) : a ≤ 0 :=
sorry

end NUMINAMATH_GPT_range_of_a_l542_54295


namespace NUMINAMATH_GPT_simplify_expression_l542_54287

theorem simplify_expression (a b c d x y : ℝ) (h : cx ≠ -dy) :
  (cx * (b^2 * x^2 + 3 * b^2 * y^2 + a^2 * y^2) + dy * (b^2 * x^2 + 3 * a^2 * x^2 + a^2 * y^2)) / (cx + dy)
  = (b^2 + 3 * a^2) * x^2 + (a^2 + 3 * b^2) * y^2 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l542_54287


namespace NUMINAMATH_GPT_min_distance_l542_54211

theorem min_distance (x y z : ℝ) :
  ∃ (m : ℝ), m = (Real.sqrt (x^2 + y^2 + z^2) + Real.sqrt ((x+1)^2 + (y-2)^2 + (z-1)^2)) ∧ m = Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_min_distance_l542_54211


namespace NUMINAMATH_GPT_stratified_sampling_l542_54202

theorem stratified_sampling (n : ℕ) : 100 + 600 + 500 = 1200 → 500 ≠ 0 → 40 / 500 = n / 1200 → n = 96 :=
by
  intros total_population nonzero_div divisor_eq
  sorry

end NUMINAMATH_GPT_stratified_sampling_l542_54202


namespace NUMINAMATH_GPT_find_x_l542_54282

-- Definitions based on the conditions
def remaining_scores_after_removal (s: List ℕ) : List ℕ :=
  s.erase 87 |>.erase 94

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

-- Converting the given problem into a Lean 4 theorem statement
theorem find_x (x : ℕ) (s : List ℕ) :
  s = [94, 87, 89, 88, 92, 90, x, 93, 92, 91] →
  average (remaining_scores_after_removal s) = 91 →
  x = 2 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_find_x_l542_54282


namespace NUMINAMATH_GPT_average_marks_mathematics_chemistry_l542_54259

theorem average_marks_mathematics_chemistry (M P C B : ℕ) 
    (h1 : M + P = 80) 
    (h2 : C + B = 120) 
    (h3 : C = P + 20) 
    (h4 : B = M - 15) : 
    (M + C) / 2 = 50 :=
by
  sorry

end NUMINAMATH_GPT_average_marks_mathematics_chemistry_l542_54259


namespace NUMINAMATH_GPT_expression_evaluation_l542_54269

theorem expression_evaluation (k : ℚ) (h : 3 * k = 10) : (6 / 5) * k - 2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l542_54269


namespace NUMINAMATH_GPT_actual_diameter_of_tissue_l542_54292

theorem actual_diameter_of_tissue (magnification: ℝ) (magnified_diameter: ℝ) :
  magnification = 1000 ∧ magnified_diameter = 1 → magnified_diameter / magnification = 0.001 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_actual_diameter_of_tissue_l542_54292


namespace NUMINAMATH_GPT_ott_fraction_l542_54212

/-- 
Moe, Loki, Nick, and Pat each give $2 to Ott.
Moe gave Ott one-seventh of his money.
Loki gave Ott one-fifth of his money.
Nick gave Ott one-fourth of his money.
Pat gave Ott one-sixth of his money.
-/
def fraction_of_money_ott_now_has (A : ℕ) (B : ℕ) (C : ℕ) (D : ℕ) : Prop :=
  A = 14 ∧ B = 10 ∧ C = 8 ∧ D = 12 ∧ (2 * (1 / 7 : ℚ)) = 2 ∧ (2 * (1 / 5 : ℚ)) = 2 ∧ (2 * (1 / 4 : ℚ)) = 2 ∧ (2 * (1 / 6 : ℚ)) = 2

theorem ott_fraction (A : ℕ) (B : ℕ) (C : ℕ) (D : ℕ) (h : fraction_of_money_ott_now_has A B C D) : 
  8 = (2 / 11 : ℚ) * (A + B + C + D) :=
by sorry

end NUMINAMATH_GPT_ott_fraction_l542_54212


namespace NUMINAMATH_GPT_cos_of_F_in_def_l542_54223

theorem cos_of_F_in_def (E F : ℝ) (h₁ : E + F = π / 2) (h₂ : Real.sin E = 3 / 5) : Real.cos F = 3 / 5 :=
sorry

end NUMINAMATH_GPT_cos_of_F_in_def_l542_54223


namespace NUMINAMATH_GPT_average_salary_of_feb_mar_apr_may_l542_54228

theorem average_salary_of_feb_mar_apr_may
  (avg_salary_jan_feb_mar_apr : ℝ)
  (salary_jan : ℝ)
  (salary_may : ℝ)
  (total_salary_feb_mar_apr : ℝ)
  (total_salary_feb_mar_apr_may: ℝ)
  (n_months: ℝ): 
  avg_salary_jan_feb_mar_apr = 8000 ∧ 
  salary_jan = 6100 ∧ 
  salary_may = 6500 ∧ 
  total_salary_feb_mar_apr = (avg_salary_jan_feb_mar_apr * 4 - salary_jan) ∧
  total_salary_feb_mar_apr_may = (total_salary_feb_mar_apr + salary_may) ∧
  n_months = (total_salary_feb_mar_apr_may / 8100) →
  n_months = 4 :=
by
  intros 
  sorry

end NUMINAMATH_GPT_average_salary_of_feb_mar_apr_may_l542_54228


namespace NUMINAMATH_GPT_projection_problem_l542_54232

noncomputable def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let dot_vv := v.1 * v.1 + v.2 * v.2
  (dot_uv / dot_vv * v.1, dot_uv / dot_vv * v.2)

theorem projection_problem :
  let v : ℝ × ℝ := (1, -1/2)
  let sum_v := (v.1 + 1, v.2 + 1)
  projection (3, 5) sum_v = (104/17, 26/17) :=
by
  sorry

end NUMINAMATH_GPT_projection_problem_l542_54232


namespace NUMINAMATH_GPT_baseball_cards_remaining_l542_54210

-- Define the number of baseball cards Mike originally had
def original_cards : ℕ := 87

-- Define the number of baseball cards Sam bought from Mike
def cards_bought : ℕ := 13

-- Prove that the remaining number of baseball cards Mike has is 74
theorem baseball_cards_remaining : original_cards - cards_bought = 74 := by
  sorry

end NUMINAMATH_GPT_baseball_cards_remaining_l542_54210


namespace NUMINAMATH_GPT_juan_marbles_l542_54234

-- Conditions
def connie_marbles : ℕ := 39
def extra_marbles_juan : ℕ := 25

-- Theorem statement: Total marbles Juan has
theorem juan_marbles : connie_marbles + extra_marbles_juan = 64 :=
by
  sorry

end NUMINAMATH_GPT_juan_marbles_l542_54234


namespace NUMINAMATH_GPT_total_income_l542_54230

-- Definitions of conditions
def charge_per_meter : ℝ := 0.2
def number_of_fences : ℝ := 50
def length_of_each_fence : ℝ := 500

-- Theorem statement
theorem total_income :
  let total_length := number_of_fences * length_of_each_fence
  let total_income := total_length * charge_per_meter
  total_income = 5000 := 
by
  sorry

end NUMINAMATH_GPT_total_income_l542_54230


namespace NUMINAMATH_GPT_total_yarn_length_is_1252_l542_54276

/-- Defining the lengths of the yarns according to the conditions --/
def green_yarn : ℕ := 156
def red_yarn : ℕ := 3 * green_yarn + 8
def blue_yarn : ℕ := (green_yarn + red_yarn) / 2
def average_yarn_length : ℕ := (green_yarn + red_yarn + blue_yarn) / 3
def yellow_yarn : ℕ := average_yarn_length - 12

/-- Proving the total length of the four pieces of yarn is 1252 cm --/
theorem total_yarn_length_is_1252 :
  green_yarn + red_yarn + blue_yarn + yellow_yarn = 1252 := by
  sorry

end NUMINAMATH_GPT_total_yarn_length_is_1252_l542_54276


namespace NUMINAMATH_GPT_percentage_fullness_before_storms_l542_54220

def capacity : ℕ := 200 -- capacity in billion gallons
def water_added_by_storms : ℕ := 15 + 30 + 75 -- total water added by storms in billion gallons
def percentage_after : ℕ := 80 -- percentage of fullness after storms
def amount_of_water_after_storms : ℕ := capacity * percentage_after / 100

theorem percentage_fullness_before_storms :
  (amount_of_water_after_storms - water_added_by_storms) * 100 / capacity = 20 := by
  sorry

end NUMINAMATH_GPT_percentage_fullness_before_storms_l542_54220


namespace NUMINAMATH_GPT_distance_along_stream_l542_54273
-- Define the problem in Lean 4

noncomputable def speed_boat_still : ℝ := 11   -- Speed of the boat in still water
noncomputable def distance_against_stream : ℝ := 9  -- Distance traveled against the stream in one hour

theorem distance_along_stream : 
  ∃ (v_s : ℝ), (speed_boat_still - v_s = distance_against_stream) ∧ (11 + v_s) * 1 = 13 := 
by
  use 2
  sorry

end NUMINAMATH_GPT_distance_along_stream_l542_54273


namespace NUMINAMATH_GPT_standard_eq_of_ellipse_value_of_k_l542_54266

-- Definitions and conditions
def is_ellipse (a b : ℝ) : Prop :=
  a > b ∧ b > 0

def eccentricity (a b : ℝ) (e : ℝ) : Prop :=
  e = (Real.sqrt 2) / 2 ∧ a^2 = b^2 + (a * e)^2

def minor_axis_length (b : ℝ) : Prop :=
  2 * b = 2

def is_tangency (k m : ℝ) : Prop := 
  m^2 = 1 + k^2

def line_intersect_ellipse (k m : ℝ) : Prop :=
  (4 * k * m)^2 - 4 * (1 + 2 * k^2) * (2 * m^2 - 2) > 0

def dot_product_condition (k m : ℝ) : Prop :=
  let x1 := -(4 * k * m) / (1 + 2 * k^2)
  let x2 := (2 * m^2 - 2) / (1 + 2 * k^2)
  let y1 := k * x1 + m
  let y2 := k * x2 + m
  x1 * x2 + y1 * y2 = 2 / 3

-- To prove the standard equation of the ellipse
theorem standard_eq_of_ellipse {a b : ℝ} (h_ellipse : is_ellipse a b)
  (h_eccentricity : eccentricity a b ((Real.sqrt 2) / 2)) 
  (h_minor_axis : minor_axis_length b) : 
  ∃ a, a = Real.sqrt 2 ∧ b = 1 ∧ (∀ x y, (x^2 / 2 + y^2 = 1)) := 
sorry

-- To prove the value of k
theorem value_of_k {k m : ℝ} (h_tangency : is_tangency k m) 
  (h_intersect : line_intersect_ellipse k m)
  (h_dot_product : dot_product_condition k m) :
  k = 1 ∨ k = -1 :=
sorry

end NUMINAMATH_GPT_standard_eq_of_ellipse_value_of_k_l542_54266


namespace NUMINAMATH_GPT_expression_exists_l542_54248

theorem expression_exists (a b : ℤ) (h : 5 * a = 3125) (hb : 5 * b = 25) : b = 5 := by
  sorry

end NUMINAMATH_GPT_expression_exists_l542_54248


namespace NUMINAMATH_GPT_counter_represents_number_l542_54294

theorem counter_represents_number (a b : ℕ) : 10 * a + b = 10 * a + b := 
by 
  sorry

end NUMINAMATH_GPT_counter_represents_number_l542_54294


namespace NUMINAMATH_GPT_original_cards_l542_54279

-- Define the number of cards Jason gave away
def cards_given_away : ℕ := 9

-- Define the number of cards Jason now has
def cards_now : ℕ := 4

-- Prove the original number of Pokemon cards Jason had
theorem original_cards (x : ℕ) : x = cards_given_away + cards_now → x = 13 :=
by {
    sorry
}

end NUMINAMATH_GPT_original_cards_l542_54279


namespace NUMINAMATH_GPT_pick_peanut_cluster_percentage_l542_54236

def total_chocolates := 100
def typeA_caramels := 5
def typeB_caramels := 6
def typeC_caramels := 4
def typeD_nougats := 2 * typeA_caramels
def typeE_nougats := 2 * typeB_caramels
def typeF_truffles := typeA_caramels + 6
def typeG_truffles := typeB_caramels + 6
def typeH_truffles := typeC_caramels + 6

def total_non_peanut_clusters := 
  typeA_caramels + typeB_caramels + typeC_caramels + typeD_nougats + typeE_nougats + typeF_truffles + typeG_truffles + typeH_truffles

def number_peanut_clusters := total_chocolates - total_non_peanut_clusters

def percent_peanut_clusters := (number_peanut_clusters * 100) / total_chocolates

theorem pick_peanut_cluster_percentage : percent_peanut_clusters = 30 := 
by {
  sorry
}

end NUMINAMATH_GPT_pick_peanut_cluster_percentage_l542_54236


namespace NUMINAMATH_GPT_impossible_15_cents_l542_54290

theorem impossible_15_cents (a b c d : ℕ) (ha : a ≤ 4) (hb : b ≤ 4) (hc : c ≤ 4) (hd : d ≤ 4) (h : a + b + c + d = 4) : 
  1 * a + 5 * b + 10 * c + 25 * d ≠ 15 :=
by
  sorry

end NUMINAMATH_GPT_impossible_15_cents_l542_54290


namespace NUMINAMATH_GPT_scientific_notation_189100_l542_54241

  theorem scientific_notation_189100 :
    (∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 189100 = a * 10^n) ∧ (∃ (a : ℝ) (n : ℤ), a = 1.891 ∧ n = 5) :=
  by {
    sorry
  }
  
end NUMINAMATH_GPT_scientific_notation_189100_l542_54241


namespace NUMINAMATH_GPT_quadratic_root_2020_l542_54201

theorem quadratic_root_2020 (a b : ℝ) (h₀ : a ≠ 0) (h₁ : a * 2019^2 + b * 2019 - 1 = 0) :
    ∃ x : ℝ, (a * (x - 1)^2 + b * (x - 1) = 1) ∧ x = 2020 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_2020_l542_54201


namespace NUMINAMATH_GPT_total_shaded_area_l542_54239

theorem total_shaded_area (side_len : ℝ) (segment_len : ℝ) (h : ℝ) :
  side_len = 8 ∧ segment_len = 1 ∧ 0 ≤ h ∧ h ≤ 8 →
  (segment_len * h / 2 + segment_len * (side_len - h) / 2) = 4 := 
by
  intro h_cond
  rcases h_cond with ⟨h_side_len, h_segment_len, h_nonneg, h_le⟩
  -- Directly state the simplified computation
  sorry

end NUMINAMATH_GPT_total_shaded_area_l542_54239


namespace NUMINAMATH_GPT_andrew_age_l542_54270

/-- 
Andrew and his five cousins are ages 4, 6, 8, 10, 12, and 14. 
One afternoon two of his cousins whose ages sum to 18 went to the movies. 
Two cousins younger than 12 but not including the 8-year-old went to play baseball. 
Andrew and the 6-year-old stayed home. How old is Andrew?
-/
theorem andrew_age (ages : Finset ℕ) (andrew_age: ℕ)
  (h_ages : ages = {4, 6, 8, 10, 12, 14})
  (movies : Finset ℕ) (baseball : Finset ℕ)
  (h_movies1 : movies.sum id = 18)
  (h_baseball1 : ∀ x ∈ baseball, x < 12 ∧ x ≠ 8)
  (home : Finset ℕ) (h_home : home = {6, andrew_age}) :
  andrew_age = 12 :=
sorry

end NUMINAMATH_GPT_andrew_age_l542_54270


namespace NUMINAMATH_GPT_magnification_factor_l542_54253

variable (diameter_magnified : ℝ)
variable (diameter_actual : ℝ)
variable (M : ℝ)

theorem magnification_factor
    (h_magnified : diameter_magnified = 0.3)
    (h_actual : diameter_actual = 0.0003) :
    M = diameter_magnified / diameter_actual ↔ M = 1000 := by
  sorry

end NUMINAMATH_GPT_magnification_factor_l542_54253


namespace NUMINAMATH_GPT_probability_A_wins_l542_54224

variable (P_A_not_lose : ℝ) (P_draw : ℝ)
variable (h1 : P_A_not_lose = 0.8)
variable (h2 : P_draw = 0.5)

theorem probability_A_wins : P_A_not_lose - P_draw = 0.3 := by
  sorry

end NUMINAMATH_GPT_probability_A_wins_l542_54224


namespace NUMINAMATH_GPT_bucket_proof_l542_54298

variable (CA : ℚ) -- capacity of Bucket A
variable (CB : ℚ) -- capacity of Bucket B
variable (SA_init : ℚ) -- initial amount of sand in Bucket A
variable (SB_init : ℚ) -- initial amount of sand in Bucket B

def bucket_conditions : Prop := 
  CB = (1 / 2) * CA ∧
  SA_init = (1 / 4) * CA ∧
  SB_init = (3 / 8) * CB

theorem bucket_proof (h : bucket_conditions CA CB SA_init SB_init) : 
  (SA_init + SB_init) / CA = 7 / 16 := 
  by sorry

end NUMINAMATH_GPT_bucket_proof_l542_54298


namespace NUMINAMATH_GPT_largest_quadrilateral_angle_l542_54252

theorem largest_quadrilateral_angle (x : ℝ)
  (h1 : 3 * x + 4 * x + 5 * x + 6 * x = 360) :
  6 * x = 120 :=
by
  sorry

end NUMINAMATH_GPT_largest_quadrilateral_angle_l542_54252


namespace NUMINAMATH_GPT_inequality_not_always_true_l542_54203

variables {a b c d : ℝ}

theorem inequality_not_always_true
  (h1 : a > b) (h2 : b > 0) (h3 : c > 0) (h4 : d ≠ 0) :
  ¬ ∀ (a b d : ℝ), (a > b) → (d ≠ 0) → (a + d)^2 > (b + d)^2 :=
by
  intro H
  specialize H a b d h1 h4
  sorry

end NUMINAMATH_GPT_inequality_not_always_true_l542_54203


namespace NUMINAMATH_GPT_problem_statement_l542_54291

open Set

def M : Set ℝ := {x | x^2 - 2008 * x - 2009 > 0}
def N (a b : ℝ) : Set ℝ := {x | x^2 + a * x + b ≤ 0}

theorem problem_statement (a b : ℝ) :
  (M ∪ N a b = univ) →
  (M ∩ N a b = {x | 2009 < x ∧ x ≤ 2010}) →
  (a = 2009 ∧ b = 2010) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l542_54291


namespace NUMINAMATH_GPT_arithmetic_sequence_third_term_l542_54240

theorem arithmetic_sequence_third_term {a d : ℝ} (h : 2 * a + 4 * d = 10) : a + 2 * d = 5 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_third_term_l542_54240

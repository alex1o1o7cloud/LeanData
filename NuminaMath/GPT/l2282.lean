import Mathlib

namespace inequality_solution_set_l2282_228258

variable {a b x : ℝ}

theorem inequality_solution_set (h : ∀ x : ℝ, ax - b > 0 ↔ x < -1) : 
  ∀ x : ℝ, (x-2) * (ax + b) < 0 ↔ x < 1 ∨ x > 2 :=
by sorry

end inequality_solution_set_l2282_228258


namespace fraction_multiplication_result_l2282_228271

theorem fraction_multiplication_result :
  (5 * 7) / 8 = 4 + 3 / 8 :=
by
  sorry

end fraction_multiplication_result_l2282_228271


namespace highest_and_lowest_score_average_score_l2282_228280

def std_score : ℤ := 60
def scores : List ℤ := [36, 0, 12, -18, 20]

theorem highest_and_lowest_score 
  (highest_score : ℤ) (lowest_score : ℤ) : 
  highest_score = std_score + 36 ∧ lowest_score = std_score - 18 := 
sorry

theorem average_score (avg_score : ℤ) :
  avg_score = std_score + ((36 + 0 + 12 - 18 + 20) / 5) := 
sorry

end highest_and_lowest_score_average_score_l2282_228280


namespace trigonometric_identity_l2282_228241

theorem trigonometric_identity (α : ℝ) 
  (h : Real.cos (5 * Real.pi / 12 - α) = Real.sqrt 2 / 3) :
  Real.sqrt 3 * Real.cos (2 * α) - Real.sin (2 * α) = 10 / 9 := sorry

end trigonometric_identity_l2282_228241


namespace li_li_age_this_year_l2282_228264

theorem li_li_age_this_year (A B : ℕ) (h1 : A + B = 30) (h2 : A = B + 6) : B = 12 := by
  sorry

end li_li_age_this_year_l2282_228264


namespace remainder_div_38_l2282_228253

theorem remainder_div_38 (n : ℕ) (h : n = 432 * 44) : n % 38 = 32 :=
sorry

end remainder_div_38_l2282_228253


namespace value_of_ab_l2282_228243

theorem value_of_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 25) : ab = 8 :=
by
  sorry

end value_of_ab_l2282_228243


namespace circle_touch_externally_circle_one_inside_other_without_touching_circle_completely_outside_l2282_228282

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

end circle_touch_externally_circle_one_inside_other_without_touching_circle_completely_outside_l2282_228282


namespace initial_avg_income_l2282_228204

theorem initial_avg_income (A : ℝ) :
  (4 * A - 990 = 3 * 650) → (A = 735) :=
by
  sorry

end initial_avg_income_l2282_228204


namespace base8_subtraction_l2282_228275

theorem base8_subtraction : (7463 - 3154 = 4317) := by sorry

end base8_subtraction_l2282_228275


namespace xiao_ming_final_score_correct_l2282_228217

/-- Xiao Ming's scores in image, content, and effect are 9, 8, and 8 points, respectively.
    The weights (ratios) for these scores are 3:4:3.
    Prove that Xiao Ming's final competition score is 8.3 points. -/
def xiao_ming_final_score : Prop :=
  let image_score := 9
  let content_score := 8
  let effect_score := 8
  let image_weight := 3
  let content_weight := 4
  let effect_weight := 3
  let total_weight := image_weight + content_weight + effect_weight
  let weighted_score := (image_score * image_weight) + (content_score * content_weight) + (effect_score * effect_weight)
  weighted_score / total_weight = 8.3

theorem xiao_ming_final_score_correct : xiao_ming_final_score := by
  sorry

end xiao_ming_final_score_correct_l2282_228217


namespace lightest_pumpkin_weight_l2282_228212

theorem lightest_pumpkin_weight 
  (A B C : ℕ)
  (h₁ : A + B = 12)
  (h₂ : B + C = 15)
  (h₃ : A + C = 13) :
  A = 5 :=
by
  sorry

end lightest_pumpkin_weight_l2282_228212


namespace square_perimeter_l2282_228249

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

end square_perimeter_l2282_228249


namespace line_through_point_area_T_l2282_228209

variable (a T : ℝ)

def equation_of_line (x y : ℝ) : Prop := 2 * T * x - a^2 * y + 2 * a * T = 0

theorem line_through_point_area_T :
  ∃ (x y : ℝ), equation_of_line a T x y ∧ x = -a ∧ y = (2 * T) / a :=
by
  sorry

end line_through_point_area_T_l2282_228209


namespace product_of_legs_divisible_by_12_l2282_228265

theorem product_of_legs_divisible_by_12 
  (a b c : ℕ) 
  (h_triangle : a^2 + b^2 = c^2) 
  (h_int : ∃ a b c : ℕ, a^2 + b^2 = c^2) :
  ∃ k : ℕ, a * b = 12 * k :=
sorry

end product_of_legs_divisible_by_12_l2282_228265


namespace jerry_age_proof_l2282_228234

variable (J : ℝ)

/-- Mickey's age is 4 years less than 400% of Jerry's age. Mickey is 18 years old. Prove that Jerry is 5.5 years old. -/
theorem jerry_age_proof (h : 18 = 4 * J - 4) : J = 5.5 :=
by
  sorry

end jerry_age_proof_l2282_228234


namespace sufficient_but_not_necessary_condition_l2282_228219

noncomputable def P := {x : ℝ | 0 < x ∧ x < 3}
noncomputable def Q := {x : ℝ | -3 < x ∧ x < 3}

theorem sufficient_but_not_necessary_condition :
  (∀ x, x ∈ P → x ∈ Q) ∧ ¬(∀ x, x ∈ Q → x ∈ P) := by
  sorry

end sufficient_but_not_necessary_condition_l2282_228219


namespace solve_for_a_l2282_228216

theorem solve_for_a (a : ℕ) (h : a^3 = 21 * 25 * 35 * 63) : a = 105 :=
sorry

end solve_for_a_l2282_228216


namespace larger_root_eq_5_over_8_l2282_228266

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

end larger_root_eq_5_over_8_l2282_228266


namespace james_main_game_time_l2282_228288

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

end james_main_game_time_l2282_228288


namespace expand_product_l2282_228297

theorem expand_product : ∀ (x : ℝ), (3 * x - 4) * (2 * x + 9) = 6 * x^2 + 19 * x - 36 :=
by
  intro x
  sorry

end expand_product_l2282_228297


namespace independence_test_categorical_l2282_228262

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

end independence_test_categorical_l2282_228262


namespace domain_of_function_l2282_228220

section
variable (x : ℝ)

def condition_1 := x + 4 ≥ 0
def condition_2 := x + 2 ≠ 0
def domain := { x : ℝ | x ≥ -4 ∧ x ≠ -2 }

theorem domain_of_function : (condition_1 x ∧ condition_2 x) ↔ (x ∈ domain) :=
by
  sorry
end

end domain_of_function_l2282_228220


namespace range_of_k_intersection_l2282_228236

theorem range_of_k_intersection (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (k^2 - 1) * x1^2 + 4 * k * x1 + 10 = 0 ∧ (k^2 - 1) * x2^2 + 4 * k * x2 + 10 = 0) ↔ (-1 < k ∧ k < 1) :=
by
  sorry

end range_of_k_intersection_l2282_228236


namespace number_of_integers_l2282_228252

theorem number_of_integers (n : ℤ) : 
  (20 < n^2) ∧ (n^2 < 150) → 
  (∃ m : ℕ, m = 16) :=
by
  sorry

end number_of_integers_l2282_228252


namespace solve_p_value_l2282_228245

noncomputable def solve_for_p (n m p : ℚ) : Prop :=
  (5 / 6 = n / 90) ∧ ((m + n) / 105 = (p - m) / 150) ∧ (p = 137.5)

theorem solve_p_value (n m p : ℚ) (h1 : 5 / 6 = n / 90) (h2 : (m + n) / 105 = (p - m) / 150) : 
  p = 137.5 :=
by
  sorry

end solve_p_value_l2282_228245


namespace hypotenuse_length_l2282_228263

theorem hypotenuse_length (a b c : ℝ) (hC : (a^2 + b^2) * (a^2 + b^2 + 1) = 12) (right_triangle : a^2 + b^2 = c^2) : 
  c = Real.sqrt 3 := 
by
  sorry

end hypotenuse_length_l2282_228263


namespace validate_assignment_l2282_228233

-- Define the statements as conditions
def S1 := "x = x + 1"
def S2 := "b ="
def S3 := "x = y = 10"
def S4 := "x + y = 10"

-- A function to check if a statement is a valid assignment
def is_valid_assignment (s : String) : Prop :=
  s = S1

-- The theorem statement proving that S1 is the only valid assignment
theorem validate_assignment : is_valid_assignment S1 ∧
                              ¬is_valid_assignment S2 ∧
                              ¬is_valid_assignment S3 ∧
                              ¬is_valid_assignment S4 :=
by
  sorry

end validate_assignment_l2282_228233


namespace find_b_minus_c_l2282_228228

variable (a b c: ℤ)

theorem find_b_minus_c (h1: a - b - c = 1) (h2: a - (b - c) = 13) (h3: (b - c) - a = -9) : b - c = 1 :=
by {
  sorry
}

end find_b_minus_c_l2282_228228


namespace no_two_champion_teams_l2282_228291

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

end no_two_champion_teams_l2282_228291


namespace bike_ride_time_good_l2282_228205

theorem bike_ride_time_good (x : ℚ) :
  (20 * x + 12 * (8 - x) = 122) → x = 13 / 4 :=
by
  intro h
  sorry

end bike_ride_time_good_l2282_228205


namespace sum_of_two_digit_factors_of_8060_l2282_228286

theorem sum_of_two_digit_factors_of_8060 : ∃ (a b : ℕ), (10 ≤ a ∧ a < 100) ∧ (10 ≤ b ∧ b < 100) ∧ (a * b = 8060) ∧ (a + b = 127) :=
by sorry

end sum_of_two_digit_factors_of_8060_l2282_228286


namespace rectangular_solid_volume_l2282_228248

theorem rectangular_solid_volume 
  (x y z : ℝ)
  (h1 : x * y = 20)
  (h2 : y * z = 15)
  (h3 : x * z = 12) :
  x * y * z = 60 :=
by
  sorry

end rectangular_solid_volume_l2282_228248


namespace brody_calculator_battery_life_l2282_228278

theorem brody_calculator_battery_life (h : ∃ t : ℕ, (3 / 4) * t + 2 + 13 = t) : ∃ t : ℕ, t = 60 :=
by
  -- Define the quarters used by Brody and the remaining battery life after the exam.
  obtain ⟨t, ht⟩ := h
  -- Simplify the equation (3/4) * t + 2 + 13 = t to get t = 60
  sorry

end brody_calculator_battery_life_l2282_228278


namespace simplify_complex_fraction_l2282_228206

theorem simplify_complex_fraction : 
  (6 - 3 * Complex.I) / (-2 + 5 * Complex.I) = (-27 / 29) - (24 / 29) * Complex.I := 
by 
  sorry

end simplify_complex_fraction_l2282_228206


namespace range_of_m_l2282_228247

theorem range_of_m (m : ℝ) : 
  (∃ (x : ℝ), ¬((x - m) < -3) ∧ (1 + 2*x)/3 ≥ x - 1) ∧ 
  (∀ (x1 x2 x3 : ℤ), 
    (¬((x1 - m) < -3) ∧ (1 + 2 * x1)/3 ≥ x1 - 1) ∧
    (¬((x2 - m) < -3) ∧ (1 + 2 * x2)/3 ≥ x2 - 1) ∧
    (¬((x3 - m) < -3) ∧ (1 + 2 * x3)/3 ≥ x3 - 1)) →
  (4 ≤ m ∧ m < 5) :=
by 
  sorry

end range_of_m_l2282_228247


namespace participants_in_robbery_l2282_228257

variables (A B V G : Prop)

theorem participants_in_robbery
  (h1 : ¬G → (B ∧ ¬A))
  (h2 : V → (¬A ∧ ¬B))
  (h3 : G → B)
  (h4 : B → (A ∨ V)) :
  A ∧ B ∧ G :=
by
  sorry

end participants_in_robbery_l2282_228257


namespace unique_solution_triple_l2282_228272

theorem unique_solution_triple (x y z : ℝ) (h1 : x + y = 3) (h2 : x * y = z^3) : (x = 1.5 ∧ y = 1.5 ∧ z = 0) :=
by
  sorry

end unique_solution_triple_l2282_228272


namespace maximize_integral_l2282_228223
open Real

noncomputable def integral_to_maximize (a b : ℝ) : ℝ :=
  ∫ x in a..b, exp (cos x) * (380 - x - x^2)

theorem maximize_integral :
  ∀ (a b : ℝ), a ≤ b → integral_to_maximize a b ≤ integral_to_maximize (-20) 19 :=
by
  intros a b h
  sorry

end maximize_integral_l2282_228223


namespace problem_solving_example_l2282_228214

theorem problem_solving_example (α β : ℝ) (h1 : α + β = 3) (h2 : α * β = 1) (h3 : α^2 - 3 * α + 1 = 0) (h4 : β^2 - 3 * β + 1 = 0) :
  7 * α^5 + 8 * β^4 = 1448 :=
sorry

end problem_solving_example_l2282_228214


namespace multiplication_in_P_l2282_228279

-- Define the set P as described in the problem
def P := {x : ℕ | ∃ n : ℕ, x = n^2}

-- Prove that for all a, b in P, a * b is also in P
theorem multiplication_in_P {a b : ℕ} (ha : a ∈ P) (hb : b ∈ P) : a * b ∈ P :=
sorry

end multiplication_in_P_l2282_228279


namespace box_volume_in_cubic_yards_l2282_228274

theorem box_volume_in_cubic_yards (v_feet : ℕ) (conv_factor : ℕ) (v_yards : ℕ)
  (h1 : v_feet = 216) (h2 : conv_factor = 3) (h3 : 27 = conv_factor ^ 3) : 
  v_yards = 8 :=
by
  sorry

end box_volume_in_cubic_yards_l2282_228274


namespace sophomores_selected_l2282_228213

variables (total_students freshmen sophomores juniors selected_students : ℕ)
def high_school_data := total_students = 2800 ∧ freshmen = 970 ∧ sophomores = 930 ∧ juniors = 900 ∧ selected_students = 280

theorem sophomores_selected (h : high_school_data total_students freshmen sophomores juniors selected_students) :
  (930 / 2800 : ℚ) * 280 = 93 := by
  sorry

end sophomores_selected_l2282_228213


namespace coloring_possible_l2282_228256

-- Define what it means for a graph to be planar and bipartite
def planar_graph (G : Type) : Prop := sorry
def bipartite_graph (G : Type) : Prop := sorry

-- The planar graph G results after subdivision without introducing new intersections
def subdivided_graph (G : Type) : Type := sorry

-- Main theorem to prove
theorem coloring_possible (G : Type) (h1 : planar_graph G) : 
  bipartite_graph (subdivided_graph G) :=
sorry

end coloring_possible_l2282_228256


namespace kanul_cash_spending_percentage_l2282_228238

theorem kanul_cash_spending_percentage :
  ∀ (spent_raw_materials spent_machinery total_amount spent_cash : ℝ),
    spent_raw_materials = 500 →
    spent_machinery = 400 →
    total_amount = 1000 →
    spent_cash = total_amount - (spent_raw_materials + spent_machinery) →
    (spent_cash / total_amount) * 100 = 10 :=
by
  intros spent_raw_materials spent_machinery total_amount spent_cash
  intro h1 h2 h3 h4
  sorry

end kanul_cash_spending_percentage_l2282_228238


namespace problem_integer_solution_l2282_228298

def satisfies_condition (n : ℤ) : Prop :=
  1 + ⌊(200 * n) / 201⌋ = ⌈(198 * n) / 200⌉

theorem problem_integer_solution :
  ∃! n : ℤ, 1 ≤ n ∧ n ≤ 20200 ∧ satisfies_condition n :=
sorry

end problem_integer_solution_l2282_228298


namespace like_terms_sum_l2282_228260

theorem like_terms_sum (m n : ℤ) (h_x : 1 = m - 2) (h_y : 2 = n + 3) : m + n = 2 :=
by
  sorry

end like_terms_sum_l2282_228260


namespace rows_of_potatoes_l2282_228244

theorem rows_of_potatoes (total_potatoes : ℕ) (seeds_per_row : ℕ) (h1 : total_potatoes = 54) (h2 : seeds_per_row = 9) : total_potatoes / seeds_per_row = 6 := 
by
  sorry

end rows_of_potatoes_l2282_228244


namespace riley_pawns_lost_l2282_228226

theorem riley_pawns_lost (initial_pawns : ℕ) (kennedy_lost : ℕ) (total_pawns_left : ℕ)
  (kennedy_initial_pawns : ℕ) (riley_initial_pawns : ℕ) : 
  kennedy_initial_pawns = initial_pawns ∧
  riley_initial_pawns = initial_pawns ∧
  kennedy_lost = 4 ∧
  total_pawns_left = 11 →
  riley_initial_pawns - (total_pawns_left - (kennedy_initial_pawns - kennedy_lost)) = 1 :=
by
  sorry

end riley_pawns_lost_l2282_228226


namespace max_pairs_of_corner_and_squares_l2282_228208

def rectangle : ℕ := 3 * 100
def unit_squares_per_pair : ℕ := 4 + 3

-- Given conditions
def conditions := rectangle = 300 ∧ unit_squares_per_pair = 7

-- Proof statement
theorem max_pairs_of_corner_and_squares (h: conditions) : ∃ n, n = 33 ∧ n * unit_squares_per_pair ≤ rectangle := 
sorry

end max_pairs_of_corner_and_squares_l2282_228208


namespace hyperbola_equation_l2282_228290

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

end hyperbola_equation_l2282_228290


namespace license_plate_difference_l2282_228255

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

end license_plate_difference_l2282_228255


namespace Tim_bottle_quarts_l2282_228299

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

end Tim_bottle_quarts_l2282_228299


namespace carolyn_fewer_stickers_l2282_228218

theorem carolyn_fewer_stickers :
  let belle_stickers := 97
  let carolyn_stickers := 79
  carolyn_stickers < belle_stickers →
  belle_stickers - carolyn_stickers = 18 :=
by
  intros
  sorry

end carolyn_fewer_stickers_l2282_228218


namespace gino_popsicle_sticks_l2282_228277

variable (my_sticks : ℕ) (total_sticks : ℕ) (gino_sticks : ℕ)

def popsicle_sticks_condition (my_sticks : ℕ) (total_sticks : ℕ) (gino_sticks : ℕ) : Prop :=
  my_sticks = 50 ∧ total_sticks = 113

theorem gino_popsicle_sticks
  (h : popsicle_sticks_condition my_sticks total_sticks gino_sticks) :
  gino_sticks = 63 :=
  sorry

end gino_popsicle_sticks_l2282_228277


namespace johns_total_packs_l2282_228294

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

end johns_total_packs_l2282_228294


namespace min_value_ab_sum_l2282_228292

theorem min_value_ab_sum (a b : ℤ) (h : a * b = 100) : a + b ≥ -101 :=
  sorry

end min_value_ab_sum_l2282_228292


namespace solve_system_l2282_228222

variable (x y z : ℝ)

def equation1 : Prop := x^2 + 25 * y + 19 * z = -471
def equation2 : Prop := y^2 + 23 * x + 21 * z = -397
def equation3 : Prop := z^2 + 21 * x + 21 * y = -545

theorem solve_system : equation1 (-22) (-23) (-20) ∧ equation2 (-22) (-23) (-20) ∧ equation3 (-22) (-23) (-20) := by
  sorry

end solve_system_l2282_228222


namespace sin_maximum_value_l2282_228207

theorem sin_maximum_value (c : ℝ) :
  (∀ x : ℝ, x = -π/4 → 3 * Real.sin (2 * x + c) = 3) → c = π :=
by
 sorry

end sin_maximum_value_l2282_228207


namespace aarti_bina_work_l2282_228231

theorem aarti_bina_work (days_aarti : ℚ) (days_bina : ℚ) (D : ℚ)
  (ha : days_aarti = 5) (hb : days_bina = 8)
  (rate_aarti : 1 / days_aarti = 1/5) 
  (rate_bina : 1 / days_bina = 1/8)
  (combine_rate : (1 / days_aarti) + (1 / days_bina) = 13 / 40) :
  3 / (13 / 40) = 120 / 13 := 
by
  sorry

end aarti_bina_work_l2282_228231


namespace wrongly_read_number_l2282_228270

theorem wrongly_read_number (initial_avg correct_avg n wrong_correct_sum : ℝ) : 
  initial_avg = 23 ∧ correct_avg = 24 ∧ n = 10 ∧ wrong_correct_sum = 36
  → ∃ (X : ℝ), 36 - X = 10 ∧ X = 26 :=
by
  intro h
  sorry

end wrongly_read_number_l2282_228270


namespace find_a_value_l2282_228284

theorem find_a_value (a : ℝ) (m : ℝ) (f g : ℝ → ℝ)
  (f_def : ∀ x, f x = Real.log x / Real.log a)
  (g_def : ∀ x, g x = (2 + m) * Real.sqrt x)
  (a_pos : 0 < a) (a_neq_one : a ≠ 1)
  (max_f : ∀ x ∈ Set.Icc (1 / 2) 16, f x ≤ 4)
  (min_f : ∀ x ∈ Set.Icc (1 / 2) 16, m ≤ f x)
  (g_increasing : ∀ x y, 0 < x → x < y → g x < g y):
  a = 2 :=
sorry

end find_a_value_l2282_228284


namespace average_of_numbers_eq_x_l2282_228221

theorem average_of_numbers_eq_x (x : ℝ) (h : (2 + x + 10) / 3 = x) : x = 6 := 
by sorry

end average_of_numbers_eq_x_l2282_228221


namespace system_of_equations_solution_exists_l2282_228281

theorem system_of_equations_solution_exists :
  ∃ (x y : ℚ), (x * y^2 - 2 * y^2 + 3 * x = 18) ∧ (3 * x * y + 5 * x - 6 * y = 24) ∧ 
                ((x = 3 ∧ y = 3) ∨ (x = 75 / 13 ∧ y = -3 / 7)) :=
by
  sorry

end system_of_equations_solution_exists_l2282_228281


namespace rhombus_diagonals_perpendicular_not_in_rectangle_l2282_228254

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

end rhombus_diagonals_perpendicular_not_in_rectangle_l2282_228254


namespace B_finish_in_54_days_l2282_228201

-- Definitions based on conditions
variables (A B : ℝ) -- A and B are the amount of work done in one day
axiom h1 : A = 2 * B -- A is twice as good as workman as B
axiom h2 : (A + B) * 18 = 1 -- Together, A and B finish the piece of work in 18 days

-- Prove that B alone will finish the work in 54 days.
theorem B_finish_in_54_days : (1 / B) = 54 :=
by 
  sorry

end B_finish_in_54_days_l2282_228201


namespace train_speed_l2282_228296

theorem train_speed (length_m : ℝ) (time_s : ℝ) 
  (h1 : length_m = 120) 
  (h2 : time_s = 3.569962336897346) 
  : (length_m / 1000) / (time_s / 3600) = 121.003 :=
by
  sorry

end train_speed_l2282_228296


namespace solve_quadratic_l2282_228269

theorem solve_quadratic :
    ∀ x : ℝ, x^2 - 6*x + 8 = 0 ↔ (x = 2 ∨ x = 4) :=
by
  intros x
  sorry

end solve_quadratic_l2282_228269


namespace stratified_sampling_l2282_228211

theorem stratified_sampling (teachers male_students female_students total_pop sample_female_students proportion_total n : ℕ)
    (h_teachers : teachers = 200)
    (h_male_students : male_students = 1200)
    (h_female_students : female_students = 1000)
    (h_total_pop : total_pop = teachers + male_students + female_students)
    (h_sample_female_students : sample_female_students = 80)
    (h_proportion_total : proportion_total = female_students / total_pop)
    (h_proportion_equation : sample_female_students = proportion_total * n) :
  n = 192 :=
by
  sorry

end stratified_sampling_l2282_228211


namespace Tim_pays_correct_amount_l2282_228285

def pays_in_a_week (hourly_rate : ℕ) (num_bodyguards : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  hourly_rate * num_bodyguards * hours_per_day * days_per_week

theorem Tim_pays_correct_amount :
  pays_in_a_week 20 2 8 7 = 2240 := by
  sorry

end Tim_pays_correct_amount_l2282_228285


namespace cost_of_each_television_l2282_228239

-- Define the conditions
def number_of_televisions : Nat := 5
def number_of_figurines : Nat := 10
def cost_per_figurine : Nat := 1
def total_spent : Nat := 260

-- Define the proof problem
theorem cost_of_each_television (T : Nat) :
  (number_of_televisions * T + number_of_figurines * cost_per_figurine = total_spent) → (T = 50) :=
by
  sorry

end cost_of_each_television_l2282_228239


namespace sin_double_angle_ineq_l2282_228229

theorem sin_double_angle_ineq (α : ℝ) (h1 : Real.sin (π - α) = 1 / 3) (h2 : π / 2 ≤ α) (h3 : α ≤ π) :
  Real.sin (2 * α) = - (4 * Real.sqrt 2) / 9 := 
by
  sorry

end sin_double_angle_ineq_l2282_228229


namespace absolute_value_condition_l2282_228273

theorem absolute_value_condition (x : ℝ) : |x + 1| + |x - 2| ≤ 5 ↔ -2 ≤ x ∧ x ≤ 3 := sorry

end absolute_value_condition_l2282_228273


namespace weight_order_l2282_228276

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

end weight_order_l2282_228276


namespace smallest_nonprime_greater_than_with_no_prime_factors_less_than_15_l2282_228235

-- Definitions for the given conditions.
def is_prime (p : ℕ) : Prop := (p > 1) ∧ ∀ d : ℕ, d ∣ p → (d = 1 ∨ d = p)

def has_no_prime_factors_less_than (n : ℕ) (m : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ m

def is_nonprime (n : ℕ) : Prop := ¬ is_prime n

-- The main theorem based on the proof problem.
theorem smallest_nonprime_greater_than_with_no_prime_factors_less_than_15 
  (n : ℕ) (h1 : n > 1) (h2 : has_no_prime_factors_less_than n 15) (h3 : is_nonprime n) : 
  280 < n ∧ n ≤ 290 :=
by
  sorry

end smallest_nonprime_greater_than_with_no_prime_factors_less_than_15_l2282_228235


namespace chicken_coop_problem_l2282_228215

-- Definitions of conditions
def available_area : ℝ := 240
def area_per_chicken : ℝ := 4
def area_per_chick : ℝ := 2
def max_daily_feed : ℝ := 8000
def feed_per_chicken : ℝ := 160
def feed_per_chick : ℝ := 40

-- Variables representing the number of chickens and chicks
variables (x y : ℕ)

-- Condition expressions
def space_condition (x y : ℕ) : Prop := 
  (2 * x + y = (available_area / area_per_chick))

def feed_condition (x y : ℕ) : Prop := 
  ((4 * x + y) * feed_per_chick <= max_daily_feed / feed_per_chick)

-- Given conditions and queries proof problem
theorem chicken_coop_problem : 
  (∃ x y : ℕ, space_condition x y ∧ feed_condition x y ∧ (x = 20 ∧ y = 80)) 
  ∧
  (¬ ∃ x y : ℕ, space_condition x y ∧ feed_condition x y ∧ (x = 30 ∧ y = 100))
  ∧
  (∃ x y : ℕ, space_condition x y ∧ feed_condition x y ∧ (x = 40 ∧ y = 40))
  ∧
  (∃ x y : ℕ, space_condition x y ∧ feed_condition x y ∧ (x = 0 ∧ y = 120)) :=
by
  sorry  -- The proof will be provided here.


end chicken_coop_problem_l2282_228215


namespace problem1_solution_l2282_228267

theorem problem1_solution (x y : ℚ) (h1 : 3 * x + 2 * y = 10) (h2 : x / 2 - (y + 1) / 3 = 1) : 
  x = 3 ∧ y = 1 / 2 :=
by
  sorry

end problem1_solution_l2282_228267


namespace largest_five_digit_number_divisible_by_6_l2282_228259

theorem largest_five_digit_number_divisible_by_6 : 
  ∃ n : ℕ, n < 100000 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m < 100000 ∧ m % 6 = 0 → m ≤ n :=
by
  sorry

end largest_five_digit_number_divisible_by_6_l2282_228259


namespace missed_questions_l2282_228224

theorem missed_questions (F U : ℕ) (h1 : U = 5 * F) (h2 : F + U = 216) : U = 180 :=
by
  sorry

end missed_questions_l2282_228224


namespace find_angle_l2282_228242

theorem find_angle (x : ℝ) (h : 180 - x = 6 * (90 - x)) : x = 72 := 
by 
    sorry

end find_angle_l2282_228242


namespace larger_integer_of_two_with_difference_8_and_product_168_l2282_228250

theorem larger_integer_of_two_with_difference_8_and_product_168 :
  ∃ (x y : ℕ), x > y ∧ x - y = 8 ∧ x * y = 168 ∧ x = 14 :=
by
  sorry

end larger_integer_of_two_with_difference_8_and_product_168_l2282_228250


namespace retailer_profit_percentage_l2282_228230

theorem retailer_profit_percentage
  (wholesale_price : ℝ)
  (retail_price : ℝ)
  (discount_rate : ℝ)
  (h_wholesale_price : wholesale_price = 108)
  (h_retail_price : retail_price = 144)
  (h_discount_rate : discount_rate = 0.10) :
  (retail_price * (1 - discount_rate) - wholesale_price) / wholesale_price * 100 = 20 :=
by
  sorry

end retailer_profit_percentage_l2282_228230


namespace flower_shop_types_l2282_228225

variable (C V T R F : ℕ)

-- Define the conditions
def condition1 : Prop := V = C / 3
def condition2 : Prop := T = V / 4
def condition3 : Prop := R = T
def condition4 : Prop := C = (2 / 3) * F

-- The main statement we need to prove: the shop stocks 4 types of flowers
theorem flower_shop_types
  (h1 : condition1 C V)
  (h2 : condition2 V T)
  (h3 : condition3 T R)
  (h4 : condition4 C F) :
  4 = 4 :=
by 
  sorry

end flower_shop_types_l2282_228225


namespace algebraic_expression_value_l2282_228240

theorem algebraic_expression_value
  (a b x y : ℤ)
  (h1 : x = a)
  (h2 : y = b)
  (h3 : x - 2 * y = 7) :
  -a + 2 * b + 1 = -6 :=
by
  -- the proof steps are omitted as instructed
  sorry

end algebraic_expression_value_l2282_228240


namespace distance_to_gym_l2282_228289

theorem distance_to_gym (v d : ℝ) (h_walked_200_m: 200 / v > 0) (h_double_speed: 2 * v = 2) (h_time_diff: 200 / v - d / (2 * v) = 50) : d = 300 :=
by sorry

end distance_to_gym_l2282_228289


namespace abc_prod_eq_l2282_228295

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

end abc_prod_eq_l2282_228295


namespace minimum_value_of_x_is_4_l2282_228283

-- Given conditions
variable {x : ℝ} (hx_pos : 0 < x) (h : log x ≥ log 2 + 1/2 * log x)

-- The minimum value of x is 4
theorem minimum_value_of_x_is_4 : x ≥ 4 :=
by
  sorry

end minimum_value_of_x_is_4_l2282_228283


namespace investment_of_c_l2282_228246

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

end investment_of_c_l2282_228246


namespace find_a_l2282_228251

variable (a : ℝ) -- Declare a as a real number.

-- Define the given conditions.
def condition1 (a : ℝ) : Prop := a^2 - 2 * a = 0
def condition2 (a : ℝ) : Prop := a ≠ 2

-- Define the theorem stating that if conditions are true, then a must be 0.
theorem find_a (h1 : condition1 a) (h2 : condition2 a) : a = 0 :=
sorry -- Proof is not provided, it needs to be constructed.

end find_a_l2282_228251


namespace base7_to_base10_l2282_228268

theorem base7_to_base10 : (2 * 7^4 + 3 * 7^3 + 4 * 7^2 + 5 * 7^1 + 6 * 7^0) = 6068 := by
  sorry

end base7_to_base10_l2282_228268


namespace sequence_geometric_proof_l2282_228202

theorem sequence_geometric_proof (a : ℕ → ℕ) (h1 : a 1 = 5) (h2 : ∀ n, a (n + 1) = 2 * a n) :
  ∀ n, a n = 5 * 2 ^ (n - 1) :=
by
  sorry

end sequence_geometric_proof_l2282_228202


namespace calc_derivative_at_pi_over_2_l2282_228237

noncomputable def f (x: ℝ) : ℝ := Real.exp x * Real.cos x

theorem calc_derivative_at_pi_over_2 : (deriv f) (Real.pi / 2) = -Real.exp (Real.pi / 2) :=
by
  sorry

end calc_derivative_at_pi_over_2_l2282_228237


namespace nonnegative_fraction_interval_l2282_228293

theorem nonnegative_fraction_interval : 
  ∀ x : ℝ, (0 ≤ x ∧ x < 3) ↔ (0 ≤ (x - 15 * x^2 + 36 * x^3) / (9 - x^3)) := by
sorry

end nonnegative_fraction_interval_l2282_228293


namespace minimum_value_x3_plus_y3_minus_5xy_l2282_228200

theorem minimum_value_x3_plus_y3_minus_5xy (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
  x^3 + y^3 - 5 * x * y ≥ -125 / 27 := 
sorry

end minimum_value_x3_plus_y3_minus_5xy_l2282_228200


namespace number_exceeds_fraction_l2282_228232

theorem number_exceeds_fraction (x : ℝ) (hx : x = 0.45 * x + 1000) : x = 1818.18 := 
by
  sorry

end number_exceeds_fraction_l2282_228232


namespace sum_of_even_factors_420_l2282_228261

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

end sum_of_even_factors_420_l2282_228261


namespace minimum_value_l2282_228287

theorem minimum_value (p q r s t u v w : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s)
    (ht : 0 < t) (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) (h₁ : p * q * r * s = 16) (h₂ : t * u * v * w = 25) :
    (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 ≥ 40 := 
sorry

end minimum_value_l2282_228287


namespace vans_needed_l2282_228203

-- Given Conditions
def van_capacity : ℕ := 4
def students : ℕ := 2
def adults : ℕ := 6
def total_people : ℕ := students + adults

-- Theorem to prove
theorem vans_needed : total_people / van_capacity = 2 :=
by
  -- Proof will be added here
  sorry

end vans_needed_l2282_228203


namespace students_neither_art_nor_music_l2282_228210

def total_students := 75
def art_students := 45
def music_students := 50
def both_art_and_music := 30

theorem students_neither_art_nor_music : 
  total_students - (art_students - both_art_and_music + music_students - both_art_and_music + both_art_and_music) = 10 :=
by 
  sorry

end students_neither_art_nor_music_l2282_228210


namespace pumac_grader_remainder_l2282_228227

/-- A PUMaC grader is grading the submissions of forty students s₁, s₂, ..., s₄₀ for the
    individual finals round, which has three problems.
    After grading a problem of student sᵢ, the grader either:
    * grades another problem of the same student, or
    * grades the same problem of the student sᵢ₋₁ or sᵢ₊₁ (if i > 1 and i < 40, respectively).
    He grades each problem exactly once, starting with the first problem of s₁
    and ending with the third problem of s₄₀.
    Let N be the number of different orders the grader may grade the students’ problems in this way.
    Prove: N ≡ 78 [MOD 100] -/

noncomputable def grading_orders_mod : ℕ := 2 * (3 ^ 38) % 100

theorem pumac_grader_remainder :
  grading_orders_mod = 78 :=
by
  sorry

end pumac_grader_remainder_l2282_228227

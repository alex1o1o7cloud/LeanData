import Mathlib

namespace NUMINAMATH_GPT_crackers_per_friend_l331_33114

theorem crackers_per_friend (Total_crackers Left_crackers Friends : ℕ) (h1 : Total_crackers = 23) (h2 : Left_crackers = 11) (h3 : Friends = 2):
  (Total_crackers - Left_crackers) / Friends = 6 :=
by
  sorry

end NUMINAMATH_GPT_crackers_per_friend_l331_33114


namespace NUMINAMATH_GPT_point_coordinates_in_second_quadrant_l331_33148

theorem point_coordinates_in_second_quadrant (P : ℝ × ℝ)
  (hx : P.1 ≤ 0)
  (hy : P.2 ≥ 0)
  (dist_x_axis : abs P.2 = 3)
  (dist_y_axis : abs P.1 = 10) :
  P = (-10, 3) :=
by
  sorry

end NUMINAMATH_GPT_point_coordinates_in_second_quadrant_l331_33148


namespace NUMINAMATH_GPT_sally_total_expense_l331_33184

-- Definitions based on the problem conditions
def peaches_price_after_coupon : ℝ := 12.32
def peaches_coupon : ℝ := 3.00
def cherries_weight : ℝ := 2.00
def cherries_price_per_kg : ℝ := 11.54
def apples_weight : ℝ := 4.00
def apples_price_per_kg : ℝ := 5.00
def apples_discount_percentage : ℝ := 0.15
def oranges_count : ℝ := 6.00
def oranges_price_per_unit : ℝ := 1.25
def oranges_promotion : ℝ := 3.00 -- Buy 2, get 1 free means she pays for 4 out of 6

-- Calculation of the total expense
def total_expense : ℝ :=
  (peaches_price_after_coupon + peaches_coupon) + 
  (cherries_weight * cherries_price_per_kg) + 
  ((apples_weight * apples_price_per_kg) * (1 - apples_discount_percentage)) +
  (4 * oranges_price_per_unit)

-- Statement to verify total expense
theorem sally_total_expense : total_expense = 60.40 := by
  sorry

end NUMINAMATH_GPT_sally_total_expense_l331_33184


namespace NUMINAMATH_GPT_max_value_of_expr_l331_33146

theorem max_value_of_expr  
  (a b c : ℝ) 
  (h₀ : 0 ≤ a)
  (h₁ : 0 ≤ b)
  (h₂ : 0 ≤ c)
  (h₃ : a + 2 * b + 3 * c = 1) :
  a + b^3 + c^4 ≤ 0.125 := 
sorry

end NUMINAMATH_GPT_max_value_of_expr_l331_33146


namespace NUMINAMATH_GPT_ten_differences_le_100_exists_l331_33140

theorem ten_differences_le_100_exists (s : Finset ℤ) (h_card : s.card = 101) (h_range : ∀ x ∈ s, 0 ≤ x ∧ x ≤ 1000) :
∃ S : Finset ℕ, S.card = 10 ∧ (∀ y ∈ S, y ≤ 100) :=
by {
  sorry
}

end NUMINAMATH_GPT_ten_differences_le_100_exists_l331_33140


namespace NUMINAMATH_GPT_star_7_3_eq_neg_5_l331_33174

def star_operation (a b : ℤ) : ℤ := 4 * a + 3 * b - 2 * a * b

theorem star_7_3_eq_neg_5 : star_operation 7 3 = -5 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_star_7_3_eq_neg_5_l331_33174


namespace NUMINAMATH_GPT_find_b_l331_33186

theorem find_b (a b c y1 y2 : ℝ) (h1 : y1 = a * 2^2 + b * 2 + c) 
              (h2 : y2 = a * (-2)^2 + b * (-2) + c) 
              (h3 : y1 - y2 = -12) : b = -3 :=
by 
  sorry

end NUMINAMATH_GPT_find_b_l331_33186


namespace NUMINAMATH_GPT_sticks_picked_up_l331_33144

variable (original_sticks left_sticks picked_sticks : ℕ)

theorem sticks_picked_up :
  original_sticks = 99 → left_sticks = 61 → picked_sticks = original_sticks - left_sticks → picked_sticks = 38 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_sticks_picked_up_l331_33144


namespace NUMINAMATH_GPT_polynomial_identity_and_sum_of_squares_l331_33199

theorem polynomial_identity_and_sum_of_squares :
  ∃ (p q r s t u : ℤ), (∀ (x : ℤ), 512 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) ∧
    p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 5472 :=
sorry

end NUMINAMATH_GPT_polynomial_identity_and_sum_of_squares_l331_33199


namespace NUMINAMATH_GPT_percentage_of_percentage_l331_33195

theorem percentage_of_percentage (a b : ℝ) (h_a : a = 0.03) (h_b : b = 0.05) : (a / b) * 100 = 60 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_percentage_l331_33195


namespace NUMINAMATH_GPT_burgers_ordered_l331_33115

theorem burgers_ordered (H : ℕ) (Ht : H + 2 * H = 45) : 2 * H = 30 := by
  sorry

end NUMINAMATH_GPT_burgers_ordered_l331_33115


namespace NUMINAMATH_GPT_square_side_length_l331_33183

-- Define the given dimensions and total length
def rectangle_width : ℕ := 2
def total_length : ℕ := 7

-- Define the unknown side length of the square
variable (Y : ℕ)

-- State the problem and provide the conclusion
theorem square_side_length : Y + rectangle_width = total_length -> Y = 5 :=
by 
  sorry

end NUMINAMATH_GPT_square_side_length_l331_33183


namespace NUMINAMATH_GPT_classroom_position_l331_33190

theorem classroom_position (a b c d : ℕ) (h : (1, 2) = (a, b)) : (3, 2) = (c, d) :=
by
  sorry

end NUMINAMATH_GPT_classroom_position_l331_33190


namespace NUMINAMATH_GPT_snow_probability_at_least_once_l331_33133

theorem snow_probability_at_least_once :
  let p := 3 / 4
  let prob_no_snow_single_day := 1 - p
  let prob_no_snow_all_days := prob_no_snow_single_day ^ 5
  let prob_snow_at_least_once := 1 - prob_no_snow_all_days
  prob_snow_at_least_once = 1023 / 1024 :=
by
  sorry

end NUMINAMATH_GPT_snow_probability_at_least_once_l331_33133


namespace NUMINAMATH_GPT_quadratic_roots_expression_l331_33153

theorem quadratic_roots_expression :
  ∀ (x₁ x₂ : ℝ), 
  (x₁ + x₂ = 3) →
  (x₁ * x₂ = -1) →
  (x₁^2 * x₂ + x₁ * x₂^2 = -3) :=
by
  intros x₁ x₂ h1 h2
  sorry

end NUMINAMATH_GPT_quadratic_roots_expression_l331_33153


namespace NUMINAMATH_GPT_percentage_less_than_l331_33130

theorem percentage_less_than (x y : ℝ) (h : y = 1.80 * x) : (x / y) * 100 = 100 - 44.44 :=
by
  sorry

end NUMINAMATH_GPT_percentage_less_than_l331_33130


namespace NUMINAMATH_GPT_drawings_in_five_pages_l331_33154

theorem drawings_in_five_pages :
  let a₁ := 5
  let a₂ := 2 * a₁
  let a₃ := 2 * a₂
  let a₄ := 2 * a₃
  let a₅ := 2 * a₄
  a₁ + a₂ + a₃ + a₄ + a₅ = 155 :=
by
  let a₁ := 5
  let a₂ := 2 * a₁
  let a₃ := 2 * a₂
  let a₄ := 2 * a₃
  let a₅ := 2 * a₄
  sorry

end NUMINAMATH_GPT_drawings_in_five_pages_l331_33154


namespace NUMINAMATH_GPT_wrapping_paper_amount_l331_33155

theorem wrapping_paper_amount (x : ℝ) (h : x + (3/4) * x + (x + (3/4) * x) = 7) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_wrapping_paper_amount_l331_33155


namespace NUMINAMATH_GPT_vehicle_count_l331_33116

theorem vehicle_count (T B : ℕ) (h1 : T + B = 15) (h2 : 3 * T + 2 * B = 40) : T = 10 ∧ B = 5 :=
by
  sorry

end NUMINAMATH_GPT_vehicle_count_l331_33116


namespace NUMINAMATH_GPT_sequence_a_n_sequence_b_n_range_k_l331_33182

-- Define the geometric sequence {a_n} with initial conditions
def a (n : ℕ) : ℕ :=
  3 * 2^(n-1)

-- Define the sequence {b_n} with the given recurrence relation
def b : ℕ → ℕ
| 0 => 1
| (n+1) => 2 * (b n) + 1

theorem sequence_a_n (n : ℕ) : 
  (a n = 3 * 2^(n-1)) := sorry

theorem sequence_b_n (n : ℕ) :
  (b n = 2^n - 1) := sorry

-- Define the condition for k and the inequality
def condition_k (k : ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → (k * (↑(b n) + 5) / 2 - 3 * 2^(n-1) ≥ 8*n + 2*k - 24)

-- Prove the range for k
theorem range_k (k : ℝ) :
  (condition_k k ↔ k ≥ 4) := sorry

end NUMINAMATH_GPT_sequence_a_n_sequence_b_n_range_k_l331_33182


namespace NUMINAMATH_GPT_propositions_correctness_l331_33150

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def P : Prop := ∃ x : ℝ, x^2 - x - 1 > 0
def negP : Prop := ∀ x : ℝ, x^2 - x - 1 ≤ 0

theorem propositions_correctness :
    (∀ a, a ∈ M → a ∈ N) = false ∧
    (∀ a b, (a ∈ M → b ∉ M) ↔ (b ∈ M → a ∉ M)) ∧
    (∀ p q, ¬(p ∧ q) → ¬p ∧ ¬q) = false ∧ 
    (¬P ↔ negP) :=
by
  sorry

end NUMINAMATH_GPT_propositions_correctness_l331_33150


namespace NUMINAMATH_GPT_find_x_l331_33192

theorem find_x (x y : ℤ) (hx : x > y) (hy : y > 0)
  (coins_megan : ℤ := 42)
  (coins_shana : ℤ := 35)
  (shana_win : ℕ := 2)
  (total_megan : shana_win * x + (total_races - shana_win) * y = coins_shana)
  (total_shana : (total_races - shana_win) * x + shana_win * y = coins_megan) :
  x = 4 := by
  sorry

end NUMINAMATH_GPT_find_x_l331_33192


namespace NUMINAMATH_GPT_greatest_power_of_2_factor_of_expr_l331_33134

theorem greatest_power_of_2_factor_of_expr :
  (∃ k, 2 ^ k ∣ 12 ^ 600 - 8 ^ 400 ∧ ∀ m, 2 ^ m ∣ 12 ^ 600 - 8 ^ 400 → m ≤ 1204) :=
sorry

end NUMINAMATH_GPT_greatest_power_of_2_factor_of_expr_l331_33134


namespace NUMINAMATH_GPT_reflect_across_y_axis_l331_33117

theorem reflect_across_y_axis (x y : ℝ) :
  (x, y) = (1, 2) → (-x, y) = (-1, 2) :=
by
  intro h
  cases h
  sorry

end NUMINAMATH_GPT_reflect_across_y_axis_l331_33117


namespace NUMINAMATH_GPT_part1_part2_l331_33132

-- Definitions corresponding to the conditions
def angle_A := 35
def angle_B1 := 40
def three_times_angle_triangle (A B C : ℕ) : Prop :=
  A + B + C = 180 ∧ (A = 3 * B ∨ B = 3 * A ∨ C = 3 * A ∨ A = 3 * C ∨ B = 3 * C ∨ C = 3 * B)

-- Part 1: Checking if triangle ABC is a "three times angle triangle".
theorem part1 : three_times_angle_triangle angle_A angle_B1 (180 - angle_A - angle_B1) :=
  sorry

-- Definitions corresponding to the new conditions
def angle_B2 := 60

-- Part 2: Finding the smallest interior angle in triangle ABC.
theorem part2 (angle_A angle_C : ℕ) :
  three_times_angle_triangle angle_A angle_B2 angle_C → (angle_A = 20 ∨ angle_A = 30 ∨ angle_C = 20 ∨ angle_C = 30) :=
  sorry

end NUMINAMATH_GPT_part1_part2_l331_33132


namespace NUMINAMATH_GPT_sin_pow_cos_pow_eq_l331_33121

theorem sin_pow_cos_pow_eq (x : ℝ) (h : Real.sin x ^ 10 + Real.cos x ^ 10 = 11 / 36) : 
  Real.sin x ^ 14 + Real.cos x ^ 14 = 41 / 216 := by
  sorry

end NUMINAMATH_GPT_sin_pow_cos_pow_eq_l331_33121


namespace NUMINAMATH_GPT_find_face_value_l331_33111

-- Define the conditions as variables in Lean
variable (BD TD FV : ℝ)
variable (hBD : BD = 36)
variable (hTD : TD = 30)
variable (hRel : BD = TD + (TD * BD / FV))

-- State the theorem we want to prove
theorem find_face_value (BD TD : ℝ) (FV : ℝ) 
  (hBD : BD = 36) (hTD : TD = 30) (hRel : BD = TD + (TD * BD / FV)) : 
  FV = 180 := 
  sorry

end NUMINAMATH_GPT_find_face_value_l331_33111


namespace NUMINAMATH_GPT_boat_ratio_l331_33181

theorem boat_ratio (b c d1 d2 : ℝ) 
  (h1 : b = 20) 
  (h2 : c = 4) 
  (h3 : d1 = 4) 
  (h4 : d2 = 2) : 
  (d1 + d2) / ((d1 / (b + c)) + (d2 / (b - c))) / b = 36 / 35 :=
by 
  sorry

end NUMINAMATH_GPT_boat_ratio_l331_33181


namespace NUMINAMATH_GPT_chocolates_difference_l331_33124

theorem chocolates_difference (robert_chocolates : ℕ) (nickel_chocolates : ℕ)
  (h1 : robert_chocolates = 7) (h2 : nickel_chocolates = 3) :
  robert_chocolates - nickel_chocolates = 4 :=
by
  sorry

end NUMINAMATH_GPT_chocolates_difference_l331_33124


namespace NUMINAMATH_GPT_parameterized_line_equation_l331_33171

theorem parameterized_line_equation (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 * t + 6) 
  (h2 : y = 5 * t - 7) : 
  y = (5 / 3) * x - 17 :=
sorry

end NUMINAMATH_GPT_parameterized_line_equation_l331_33171


namespace NUMINAMATH_GPT_gamma_suff_not_nec_for_alpha_l331_33138

variable {α β γ : Prop}

theorem gamma_suff_not_nec_for_alpha
  (h1 : β → α)
  (h2 : γ ↔ β) :
  (γ → α) ∧ (¬(α → γ)) :=
by {
  sorry
}

end NUMINAMATH_GPT_gamma_suff_not_nec_for_alpha_l331_33138


namespace NUMINAMATH_GPT_greater_than_neg4_1_l331_33187

theorem greater_than_neg4_1 (k : ℤ) (h1 : k = -4) : k > (-4.1 : ℝ) :=
by sorry

end NUMINAMATH_GPT_greater_than_neg4_1_l331_33187


namespace NUMINAMATH_GPT_statement1_statement2_statement3_statement4_statement5_statement6_l331_33128

/-
Correct syntax statements in pseudo code
-/

def correct_assignment1 (A B : ℤ) : Prop :=
  B = A ∧ A = 50

def correct_assignment2 (x y z : ℕ) : Prop :=
  x = 1 ∧ y = 2 ∧ z = 3

def correct_input1 (s : String) (x : ℕ) : Prop :=
  s = "How old are you?" ∧ x ≥ 0

def correct_input2 (x : ℕ) : Prop :=
  x ≥ 0

def correct_print1 (s1 : String) (C : ℤ) : Prop :=
  s1 = "A+B=" ∧ C < 100  -- additional arbitrary condition for C

def correct_print2 (s2 : String) : Prop :=
  s2 = "Good-bye!"

theorem statement1 (A : ℤ) : ∃ B, correct_assignment1 A B :=
sorry

theorem statement2 : ∃ (x y z : ℕ), correct_assignment2 x y z :=
sorry

theorem statement3 (x : ℕ) : ∃ s, correct_input1 s x :=
sorry

theorem statement4 (x : ℕ) : correct_input2 x :=
sorry

theorem statement5 (C : ℤ) : ∃ s1, correct_print1 s1 C :=
sorry

theorem statement6 : ∃ s2, correct_print2 s2 :=
sorry

end NUMINAMATH_GPT_statement1_statement2_statement3_statement4_statement5_statement6_l331_33128


namespace NUMINAMATH_GPT_ypsilon_calendar_l331_33119

theorem ypsilon_calendar (x y z : ℕ) 
  (h1 : 28 * x + 30 * y + 31 * z = 365) : x + y + z = 12 :=
sorry

end NUMINAMATH_GPT_ypsilon_calendar_l331_33119


namespace NUMINAMATH_GPT_r_can_complete_work_in_R_days_l331_33180

theorem r_can_complete_work_in_R_days (W : ℝ) : 
  (∀ p q r P Q R : ℝ, 
    (P = W / 24) ∧
    (Q = W / 9) ∧
    (10.000000000000002 * (W / 24) + 3 * (W / 9 + W / R) = W) 
  -> R = 12) :=
by
  intros
  sorry

end NUMINAMATH_GPT_r_can_complete_work_in_R_days_l331_33180


namespace NUMINAMATH_GPT_new_ratio_boarders_to_day_students_l331_33123

-- Given conditions
def initial_ratio_boarders_to_day_students : ℚ := 2 / 5
def initial_boarders : ℕ := 120
def new_boarders : ℕ := 30

-- Derived definitions
def initial_day_students : ℕ :=
  (initial_boarders * (5 : ℕ)) / 2

def total_boarders : ℕ := initial_boarders + new_boarders
def total_day_students : ℕ := initial_day_students

-- Theorem to prove the new ratio
theorem new_ratio_boarders_to_day_students : total_boarders / total_day_students = 1 / 2 :=
  sorry

end NUMINAMATH_GPT_new_ratio_boarders_to_day_students_l331_33123


namespace NUMINAMATH_GPT_mrs_franklin_gave_38_packs_l331_33122

-- Define the initial number of Valentines
def initial_valentines : Int := 450

-- Define the remaining Valentines after giving some away
def remaining_valentines : Int := 70

-- Define the size of each pack
def pack_size : Int := 10

-- Define the number of packs given away
def packs_given (initial remaining pack_size : Int) : Int :=
  (initial - remaining) / pack_size

theorem mrs_franklin_gave_38_packs :
  packs_given 450 70 10 = 38 := sorry

end NUMINAMATH_GPT_mrs_franklin_gave_38_packs_l331_33122


namespace NUMINAMATH_GPT_xy_condition_l331_33139

theorem xy_condition (x y : ℝ) (h : x * y + x / y + y / x = -3) : (x - 2) * (y - 2) = 3 :=
sorry

end NUMINAMATH_GPT_xy_condition_l331_33139


namespace NUMINAMATH_GPT_resistor_parallel_l331_33177

theorem resistor_parallel (x y r : ℝ)
  (h1 : x = 5)
  (h2 : r = 2.9166666666666665)
  (h3 : 1 / r = 1 / x + 1 / y) : y = 7 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_resistor_parallel_l331_33177


namespace NUMINAMATH_GPT_total_fish_l331_33142

theorem total_fish (goldfish bluefish : ℕ) (h1 : goldfish = 15) (h2 : bluefish = 7) : goldfish + bluefish = 22 := 
by
  sorry

end NUMINAMATH_GPT_total_fish_l331_33142


namespace NUMINAMATH_GPT_sum_of_three_consecutive_odds_l331_33160

theorem sum_of_three_consecutive_odds (a : ℤ) (h : a % 2 = 1) (ha_mod : (a + 4) % 2 = 1) (h_sum : a + (a + 4) = 150) : a + (a + 2) + (a + 4) = 225 :=
sorry

end NUMINAMATH_GPT_sum_of_three_consecutive_odds_l331_33160


namespace NUMINAMATH_GPT_hyperbola_center_l331_33126

theorem hyperbola_center :
  ∃ (h : ℝ × ℝ), h = (9 / 2, 2) ∧
  (∃ (x y : ℝ), 9 * x^2 - 81 * x - 16 * y^2 + 64 * y + 144 = 0) :=
  sorry

end NUMINAMATH_GPT_hyperbola_center_l331_33126


namespace NUMINAMATH_GPT_cost_of_painting_l331_33147

def area_of_house : ℕ := 484
def price_per_sqft : ℕ := 20

theorem cost_of_painting : area_of_house * price_per_sqft = 9680 := by
  sorry

end NUMINAMATH_GPT_cost_of_painting_l331_33147


namespace NUMINAMATH_GPT_parabola_chord_length_l331_33198

theorem parabola_chord_length (x₁ x₂ : ℝ) (y₁ y₂ : ℝ) 
(h1 : y₁^2 = 4 * x₁) 
(h2 : y₂^2 = 4 * x₂) 
(h3 : x₁ + x₂ = 6) : 
|y₁ - y₂| = 8 :=
sorry

end NUMINAMATH_GPT_parabola_chord_length_l331_33198


namespace NUMINAMATH_GPT_correct_value_of_3_dollar_neg4_l331_33165

def special_operation (x y : Int) : Int :=
  x * (y + 2) + x * y + x

theorem correct_value_of_3_dollar_neg4 : special_operation 3 (-4) = -15 :=
by
  sorry

end NUMINAMATH_GPT_correct_value_of_3_dollar_neg4_l331_33165


namespace NUMINAMATH_GPT_product_sum_divisible_by_1987_l331_33173

theorem product_sum_divisible_by_1987 :
  let A : ℕ :=
    List.prod (List.filter (λ x => x % 2 = 1) (List.range (1987 + 1)))
  let B : ℕ :=
    List.prod (List.filter (λ x => x % 2 = 0) (List.range (1987 + 1)))
  A + B ≡ 0 [MOD 1987] := by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_product_sum_divisible_by_1987_l331_33173


namespace NUMINAMATH_GPT_anns_age_l331_33189

theorem anns_age (a b : ℕ) (h1 : a + b = 54) 
(h2 : b = a - (a - b) + (a - b)): a = 29 :=
sorry

end NUMINAMATH_GPT_anns_age_l331_33189


namespace NUMINAMATH_GPT_blue_to_red_face_area_ratio_l331_33136

theorem blue_to_red_face_area_ratio :
  let original_cube_dim := 13
  let red_face_area := 6 * original_cube_dim^2
  let total_faces := 6 * original_cube_dim^3
  let blue_face_area := total_faces - red_face_area
  (blue_face_area / red_face_area) = 12 :=
by
  sorry

end NUMINAMATH_GPT_blue_to_red_face_area_ratio_l331_33136


namespace NUMINAMATH_GPT_paintable_area_correct_l331_33194

-- Defining lengths
def bedroom_length : ℕ := 15
def bedroom_width : ℕ := 11
def bedroom_height : ℕ := 9

-- Defining the number of bedrooms
def num_bedrooms : ℕ := 4

-- Defining the total area not to be painted per bedroom
def area_not_painted_per_bedroom : ℕ := 80

-- The total wall area calculation
def total_wall_area_per_bedroom : ℕ :=
  2 * (bedroom_length * bedroom_height) + 2 * (bedroom_width * bedroom_height)

-- The paintable wall area per bedroom calculation
def paintable_area_per_bedroom : ℕ :=
  total_wall_area_per_bedroom - area_not_painted_per_bedroom

-- The total paintable area across all bedrooms calculation
def total_paintable_area : ℕ :=
  paintable_area_per_bedroom * num_bedrooms

-- The theorem statement
theorem paintable_area_correct : total_paintable_area = 1552 := by
  sorry -- Proof is omitted

end NUMINAMATH_GPT_paintable_area_correct_l331_33194


namespace NUMINAMATH_GPT_val_of_7c_plus_7d_l331_33104

noncomputable def h (x : ℝ) : ℝ := 7 * x - 6

noncomputable def f_inv (x : ℝ) : ℝ := 7 * x - 4

noncomputable def f (c d x : ℝ) : ℝ := c * x + d

theorem val_of_7c_plus_7d (c d : ℝ) (h_eq : ∀ x, h x = f_inv x - 2) 
  (inv_prop : ∀ x, f c d (f_inv x) = x) : 7 * c + 7 * d = 5 :=
by
  sorry

end NUMINAMATH_GPT_val_of_7c_plus_7d_l331_33104


namespace NUMINAMATH_GPT_inequality_proof_l331_33100

theorem inequality_proof (a b : ℝ) (h : a + b ≠ 0) :
  (a + b) / (a^2 - a * b + b^2) ≤ 4 / |a + b| ∧
  ((a + b) / (a^2 - a * b + b^2) = 4 / |a + b| ↔ a = b) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l331_33100


namespace NUMINAMATH_GPT_largest_x_value_l331_33137

-- Definition of the equation
def equation (x : ℚ) : Prop := 3 * (9 * x^2 + 10 * x + 11) = x * (9 * x - 45)

-- The problem to prove is that the largest value of x satisfying the equation is -1/2
theorem largest_x_value : ∃ x : ℚ, equation x ∧ ∀ y : ℚ, equation y → y ≤ -1/2 := by
  sorry

end NUMINAMATH_GPT_largest_x_value_l331_33137


namespace NUMINAMATH_GPT_min_sum_of_factors_l331_33113

theorem min_sum_of_factors (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y * z = 1806) :
  x + y + z ≥ 72 := 
sorry

end NUMINAMATH_GPT_min_sum_of_factors_l331_33113


namespace NUMINAMATH_GPT_p_is_sufficient_but_not_necessary_l331_33178

-- Definitions based on conditions
def p (x y : Int) : Prop := x + y ≠ -2
def q (x y : Int) : Prop := ¬(x = -1 ∧ y = -1)

theorem p_is_sufficient_but_not_necessary (x y : Int) : 
  (p x y → q x y) ∧ ¬(q x y → p x y) :=
by
  sorry

end NUMINAMATH_GPT_p_is_sufficient_but_not_necessary_l331_33178


namespace NUMINAMATH_GPT_grant_received_money_l331_33157

theorem grant_received_money :
  let total_teeth := 20
  let lost_teeth := 2
  let first_tooth_amount := 20
  let other_tooth_amount_per_tooth := 2
  let remaining_teeth := total_teeth - lost_teeth - 1
  let total_amount_received := first_tooth_amount + remaining_teeth * other_tooth_amount_per_tooth
  total_amount_received = 54 :=
by  -- Start the proof mode
  sorry  -- This is where the actual proof would go

end NUMINAMATH_GPT_grant_received_money_l331_33157


namespace NUMINAMATH_GPT_dive_has_five_judges_l331_33127

noncomputable def number_of_judges 
  (scores : List ℝ)
  (difficulty : ℝ)
  (point_value : ℝ) : ℕ := sorry

theorem dive_has_five_judges :
  number_of_judges [7.5, 8.0, 9.0, 6.0, 8.8] 3.2 77.76 = 5 :=
by
  sorry

end NUMINAMATH_GPT_dive_has_five_judges_l331_33127


namespace NUMINAMATH_GPT_jane_donuts_l331_33156

def croissant_cost := 60
def donut_cost := 90
def days := 6

theorem jane_donuts (c d k : ℤ) 
  (h1 : c + d = days)
  (h2 : donut_cost * d + croissant_cost * c = 100 * k + 50) :
  d = 3 :=
sorry

end NUMINAMATH_GPT_jane_donuts_l331_33156


namespace NUMINAMATH_GPT_min_dist_circle_to_line_l331_33197

noncomputable def circle_eq (x y : ℝ) := x^2 + y^2 - 2*x - 2*y

noncomputable def line_eq (x y : ℝ) := x + y - 8

theorem min_dist_circle_to_line : 
  (∀ x y : ℝ, circle_eq x y = 0 → ∃ d : ℝ, d ≥ 0 ∧ 
    (∀ x₁ y₁ : ℝ, circle_eq x₁ y₁ = 0 → ∀ x₂ y₂ : ℝ, line_eq x₂ y₂ = 0 → d ≤ dist (x₁, y₁) (x₂, y₂)) ∧ 
    d = 2 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_min_dist_circle_to_line_l331_33197


namespace NUMINAMATH_GPT_intersection_correct_l331_33188

noncomputable def set_M : Set ℝ := { x | x^2 + x - 6 ≤ 0 }
noncomputable def set_N : Set ℝ := { x | abs (2 * x + 1) > 3 }
noncomputable def set_intersection : Set ℝ := { x | (x ∈ set_M) ∧ (x ∈ set_N) }

theorem intersection_correct : 
  set_intersection = { x : ℝ | (-3 ≤ x ∧ x < -2) ∨ (1 < x ∧ x ≤ 2) } := 
by 
  sorry

end NUMINAMATH_GPT_intersection_correct_l331_33188


namespace NUMINAMATH_GPT_restore_original_price_l331_33167

theorem restore_original_price (original_price promotional_price : ℝ) (h₀ : original_price = 1) (h₁ : promotional_price = original_price * 0.8) : (original_price - promotional_price) / promotional_price = 0.25 :=
by sorry

end NUMINAMATH_GPT_restore_original_price_l331_33167


namespace NUMINAMATH_GPT_find_speed_of_current_l331_33175

variable {m c : ℝ}

theorem find_speed_of_current
  (h1 : m + c = 15)
  (h2 : m - c = 10) :
  c = 2.5 :=
sorry

end NUMINAMATH_GPT_find_speed_of_current_l331_33175


namespace NUMINAMATH_GPT_sqrt_of_16_is_4_l331_33170

def arithmetic_square_root (x : ℕ) : ℕ :=
  if x = 0 then 0 else Nat.sqrt x

theorem sqrt_of_16_is_4 : arithmetic_square_root 16 = 4 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_of_16_is_4_l331_33170


namespace NUMINAMATH_GPT_percentage_of_invalid_votes_calculation_l331_33109

theorem percentage_of_invalid_votes_calculation
  (total_votes_poled : ℕ)
  (valid_votes_B : ℕ)
  (additional_percent_votes_A : ℝ)
  (Vb : ℝ)
  (total_valid_votes : ℝ)
  (P : ℝ) :
  total_votes_poled = 8720 →
  valid_votes_B = 2834 →
  additional_percent_votes_A = 0.15 →
  Vb = valid_votes_B →
  total_valid_votes = (2 * Vb) + (additional_percent_votes_A * total_votes_poled) →
  total_valid_votes / total_votes_poled = 1 - P/100 →
  P = 20 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_percentage_of_invalid_votes_calculation_l331_33109


namespace NUMINAMATH_GPT_jack_bill_age_difference_l331_33172

theorem jack_bill_age_difference :
  ∃ (a b : ℕ), (0 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (7 * a - 29 * b = 14) ∧ ((10 * a + b) - (10 * b + a) = 36) :=
by
  sorry

end NUMINAMATH_GPT_jack_bill_age_difference_l331_33172


namespace NUMINAMATH_GPT_last_three_digits_l331_33101

theorem last_three_digits (n : ℕ) : 7^106 % 1000 = 321 :=
by
  sorry

end NUMINAMATH_GPT_last_three_digits_l331_33101


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l331_33169

theorem problem1 : (-3 + 8 - 7 - 15) = -17 := 
sorry

theorem problem2 : (23 - 6 * (-3) + 2 * (-4)) = 33 := 
sorry

theorem problem3 : (-8 / (4 / 5) * (-2 / 3)) = 20 / 3 := 
sorry

theorem problem4 : (-2^2 - 9 * (-1 / 3)^2 + abs (-4)) = -1 := 
sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l331_33169


namespace NUMINAMATH_GPT_total_distance_traveled_l331_33110

/--
A spider is on the edge of a ceiling of a circular room with a radius of 65 feet. 
The spider walks straight across the ceiling to the opposite edge, passing through 
the center. It then walks straight to another point on the edge of the circle but 
not back through the center. The third part of the journey is straight back to the 
original starting point. If the third part of the journey was 90 feet long, then 
the total distance traveled by the spider is 313.81 feet.
-/
theorem total_distance_traveled (r : ℝ) (d1 d2 d3 : ℝ) (h1 : r = 65) (h2 : d1 = 2 * r) (h3 : d3 = 90) :
  d1 + d2 + d3 = 313.81 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_traveled_l331_33110


namespace NUMINAMATH_GPT_alice_score_l331_33106

variables (correct_answers wrong_answers unanswered_questions : ℕ)
variables (points_correct points_incorrect : ℚ)

def compute_score (correct_answers wrong_answers : ℕ) (points_correct points_incorrect : ℚ) : ℚ :=
    (correct_answers : ℚ) * points_correct + (wrong_answers : ℚ) * points_incorrect

theorem alice_score : 
    correct_answers = 15 → 
    wrong_answers = 5 → 
    unanswered_questions = 10 → 
    points_correct = 1 → 
    points_incorrect = -0.25 → 
    compute_score 15 5 1 (-0.25) = 13.75 := 
by intros; sorry

end NUMINAMATH_GPT_alice_score_l331_33106


namespace NUMINAMATH_GPT_range_of_m_l331_33103

noncomputable def f (x m : ℝ) : ℝ := x^2 - x + m * (2 * x + 1)

theorem range_of_m (m : ℝ) : (∀ x > 1, 0 < 2 * x + (2 * m - 1)) ↔ (m ≥ -1/2) := by
  sorry

end NUMINAMATH_GPT_range_of_m_l331_33103


namespace NUMINAMATH_GPT_find_x_l331_33191

variable (x : ℝ)

theorem find_x (h : 2 * x - 12 = -(x + 3)) : x = 3 := 
sorry

end NUMINAMATH_GPT_find_x_l331_33191


namespace NUMINAMATH_GPT_perpendicular_line_and_plane_implication_l331_33163

variable (l m : Line)
variable (α β : Plane)

-- Given conditions
def line_perpendicular_to_plane (l : Line) (α : Plane) : Prop :=
sorry -- Assume this checks if line l is perpendicular to plane α

def line_in_plane (m : Line) (α : Plane) : Prop :=
sorry -- Assume this checks if line m is included in plane α

def line_perpendicular_to_line (l m : Line) : Prop :=
sorry -- Assume this checks if line l is perpendicular to line m

-- Lean statement for the proof problem
theorem perpendicular_line_and_plane_implication
  (h1 : line_perpendicular_to_plane l α)
  (h2 : line_in_plane m α) :
  line_perpendicular_to_line l m :=
sorry

end NUMINAMATH_GPT_perpendicular_line_and_plane_implication_l331_33163


namespace NUMINAMATH_GPT_unique_integer_solution_l331_33196

theorem unique_integer_solution :
  ∃! (z : ℤ), 5 * z ≤ 2 * z - 8 ∧ -3 * z ≥ 18 ∧ 7 * z ≤ -3 * z - 21 :=
by
  sorry

end NUMINAMATH_GPT_unique_integer_solution_l331_33196


namespace NUMINAMATH_GPT_max_stamps_without_discount_theorem_l331_33162

def total_money := 5000
def price_per_stamp := 50
def max_stamps_without_discount := 100

theorem max_stamps_without_discount_theorem :
  price_per_stamp * max_stamps_without_discount ≤ total_money ∧
  ∀ n, n > max_stamps_without_discount → price_per_stamp * n > total_money := by
  sorry

end NUMINAMATH_GPT_max_stamps_without_discount_theorem_l331_33162


namespace NUMINAMATH_GPT_sum_of_special_right_triangle_areas_l331_33168

noncomputable def is_special_right_triangle (a b : ℕ) : Prop :=
  let area := (a * b) / 2
  area = 3 * (a + b)

noncomputable def special_right_triangle_areas : List ℕ :=
  [(18, 9), (9, 18), (15, 10), (10, 15), (12, 12)].map (λ p => (p.1 * p.2) / 2)

theorem sum_of_special_right_triangle_areas : 
  special_right_triangle_areas.eraseDups.sum = 228 := by
  sorry

end NUMINAMATH_GPT_sum_of_special_right_triangle_areas_l331_33168


namespace NUMINAMATH_GPT_distinct_primes_p_q_r_l331_33164

theorem distinct_primes_p_q_r (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) (eqn : r * p^3 + p^2 + p = 2 * r * q^2 + q^2 + q) : p * q * r = 2014 :=
by
  sorry

end NUMINAMATH_GPT_distinct_primes_p_q_r_l331_33164


namespace NUMINAMATH_GPT_intersection_of_lines_l331_33102

theorem intersection_of_lines : 
  (∃ x y : ℚ, y = -3 * x + 1 ∧ y = 5 * x + 4) ↔ 
  (∃ x y : ℚ, x = -3 / 8 ∧ y = 17 / 8) :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_lines_l331_33102


namespace NUMINAMATH_GPT_june_spent_on_music_books_l331_33151

theorem june_spent_on_music_books
  (total_budget : ℤ)
  (math_books_cost : ℤ)
  (science_books_cost : ℤ)
  (art_books_cost : ℤ)
  (music_books_cost : ℤ)
  (h_total_budget : total_budget = 500)
  (h_math_books_cost : math_books_cost = 80)
  (h_science_books_cost : science_books_cost = 100)
  (h_art_books_cost : art_books_cost = 160)
  (h_total_cost : music_books_cost = total_budget - (math_books_cost + science_books_cost + art_books_cost)) :
  music_books_cost = 160 :=
sorry

end NUMINAMATH_GPT_june_spent_on_music_books_l331_33151


namespace NUMINAMATH_GPT_total_cost_cardshop_l331_33152

theorem total_cost_cardshop : 
  let price_A := 1.25
  let price_B := 1.50
  let price_C := 2.25
  let price_D := 2.50
  let discount_10_percent := 0.10
  let discount_15_percent := 0.15
  let sales_tax_rate := 0.06
  let qty_A := 6
  let qty_B := 4
  let qty_C := 10
  let qty_D := 12
  let total_before_discounts := qty_A * price_A + qty_B * price_B + qty_C * price_C + qty_D * price_D
  let discount_A := if qty_A >= 5 then qty_A * price_A * discount_10_percent else 0
  let discount_C := if qty_C >= 8 then qty_C * price_C * discount_15_percent else 0
  let discount_D := if qty_D >= 8 then qty_D * price_D * discount_15_percent else 0
  let total_discounts := discount_A + discount_C + discount_D
  let total_after_discounts := total_before_discounts - total_discounts
  let tax := total_after_discounts * sales_tax_rate
  let total_cost := total_after_discounts + tax
  total_cost = 60.82
:= 
by
  have price_A : ℝ := 1.25
  have price_B : ℝ := 1.50
  have price_C : ℝ := 2.25
  have price_D : ℝ := 2.50
  have discount_10_percent : ℝ := 0.10
  have discount_15_percent : ℝ := 0.15
  have sales_tax_rate : ℝ := 0.06
  have qty_A : ℕ := 6
  have qty_B : ℕ := 4
  have qty_C : ℕ := 10
  have qty_D : ℕ := 12
  let total_before_discounts := qty_A * price_A + qty_B * price_B + qty_C * price_C + qty_D * price_D
  let discount_A := if qty_A >= 5 then qty_A * price_A * discount_10_percent else 0
  let discount_C := if qty_C >= 8 then qty_C * price_C * discount_15_percent else 0
  let discount_D := if qty_D >= 8 then qty_D * price_D * discount_15_percent else 0
  let total_discounts := discount_A + discount_C + discount_D
  let total_after_discounts := total_before_discounts - total_discounts
  let tax := total_after_discounts * sales_tax_rate
  let total_cost := total_after_discounts + tax
  sorry

end NUMINAMATH_GPT_total_cost_cardshop_l331_33152


namespace NUMINAMATH_GPT_find_theta_l331_33129

theorem find_theta (R h : ℝ) (θ : ℝ) 
  (r1_def : r1 = R * Real.cos θ)
  (r2_def : r2 = (R + h) * Real.cos θ)
  (s_def : s = 2 * π * h * Real.cos θ)
  (s_eq_h : s = h) : 
  θ = Real.arccos (1 / (2 * π)) :=
by
  sorry

end NUMINAMATH_GPT_find_theta_l331_33129


namespace NUMINAMATH_GPT_quadrilateral_divided_similarity_iff_trapezoid_l331_33131

noncomputable def convex_quadrilateral (A B C D : Type) : Prop := sorry
noncomputable def is_trapezoid (A B C D : Type) : Prop := sorry
noncomputable def similar_quadrilaterals (E F A B C D : Type) : Prop := sorry

theorem quadrilateral_divided_similarity_iff_trapezoid {A B C D E F : Type}
  (h1 : convex_quadrilateral A B C D)
  (h2 : similar_quadrilaterals E F A B C D): 
  is_trapezoid A B C D ↔ similar_quadrilaterals E F A B C D :=
sorry

end NUMINAMATH_GPT_quadrilateral_divided_similarity_iff_trapezoid_l331_33131


namespace NUMINAMATH_GPT_circle_radius_doubling_l331_33141

theorem circle_radius_doubling (r : ℝ) : 
  let new_radius := 2 * r
  let original_circumference := 2 * Real.pi * r
  let new_circumference := 2 * Real.pi * new_radius
  let original_area := Real.pi * r^2
  let new_area := Real.pi * (new_radius)^2
  (new_circumference = 2 * original_circumference) ∧ (new_area = 4 * original_area) :=
by
  let new_radius := 2 * r
  let original_circumference := 2 * Real.pi * r
  let new_circumference := 2 * Real.pi * new_radius
  let original_area := Real.pi * r^2
  let new_area := Real.pi * (new_radius)^2
  have hc : new_circumference = 2 * original_circumference := by
    sorry
  have ha : new_area = 4 * original_area := by
    sorry
  exact ⟨hc, ha⟩

end NUMINAMATH_GPT_circle_radius_doubling_l331_33141


namespace NUMINAMATH_GPT_drink_total_amount_l331_33159

theorem drink_total_amount (parts_coke parts_sprite parts_mountain_dew ounces_coke total_parts : ℕ)
  (h1 : parts_coke = 2) (h2 : parts_sprite = 1) (h3 : parts_mountain_dew = 3)
  (h4 : total_parts = parts_coke + parts_sprite + parts_mountain_dew)
  (h5 : ounces_coke = 6) :
  ( ounces_coke * total_parts ) / parts_coke = 18 :=
by
  sorry

end NUMINAMATH_GPT_drink_total_amount_l331_33159


namespace NUMINAMATH_GPT_clock_strikes_l331_33179

theorem clock_strikes (t n : ℕ) (h_t : 13 * t = 26) (h_n : 2 * n - 1 * t = 22) : n = 6 :=
by
  sorry

end NUMINAMATH_GPT_clock_strikes_l331_33179


namespace NUMINAMATH_GPT_students_taking_geometry_or_science_but_not_both_l331_33143

def students_taking_both : ℕ := 15
def students_taking_geometry : ℕ := 30
def students_taking_science_only : ℕ := 18

theorem students_taking_geometry_or_science_but_not_both : students_taking_geometry - students_taking_both + students_taking_science_only = 33 := by
  sorry

end NUMINAMATH_GPT_students_taking_geometry_or_science_but_not_both_l331_33143


namespace NUMINAMATH_GPT_sum_of_roots_eq_3n_l331_33107

variable {n : ℝ} 

-- Define the conditions
def quadratic_eq (x : ℝ) (m : ℝ) (n : ℝ) : Prop :=
  x^2 - (m + n) * x + m * n = 0

theorem sum_of_roots_eq_3n (m : ℝ) (n : ℝ) 
  (hm : m = 2 * n)
  (hroot_m : quadratic_eq m m n)
  (hroot_n : quadratic_eq n m n) :
  m + n = 3 * n :=
by sorry

end NUMINAMATH_GPT_sum_of_roots_eq_3n_l331_33107


namespace NUMINAMATH_GPT_abs_not_eq_three_implies_x_not_eq_three_l331_33158

theorem abs_not_eq_three_implies_x_not_eq_three (x : ℝ) (h : |x| ≠ 3) : x ≠ 3 :=
sorry

end NUMINAMATH_GPT_abs_not_eq_three_implies_x_not_eq_three_l331_33158


namespace NUMINAMATH_GPT_opposite_of_neg2_l331_33120

def opposite (y : ℤ) : ℤ := -y

theorem opposite_of_neg2 : opposite (-2) = 2 := by
  sorry

end NUMINAMATH_GPT_opposite_of_neg2_l331_33120


namespace NUMINAMATH_GPT_triangle_side_length_x_l331_33193

theorem triangle_side_length_x
  (y : ℝ) (z : ℝ) (cos_Y_minus_Z : ℝ)
  (hy : y = 7)
  (hz : z = 3)
  (hcos : cos_Y_minus_Z = 7 / 8) :
  ∃ x : ℝ, x = Real.sqrt 18.625 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_length_x_l331_33193


namespace NUMINAMATH_GPT_y_work_duration_l331_33108

theorem y_work_duration (x_rate y_rate : ℝ) (d : ℝ) :
  -- 1. x and y together can do the work in 20 days.
  (x_rate + y_rate = 1/20) →
  -- 2. x started the work alone and after 4 days y joined him till the work completed.
  -- 3. The total work lasted 10 days.
  (4 * x_rate + 6 * (x_rate + y_rate) = 1) →
  -- Prove: y can do the work alone in 12 days.
  y_rate = 1/12 :=
by {
  sorry
}

end NUMINAMATH_GPT_y_work_duration_l331_33108


namespace NUMINAMATH_GPT_georgia_makes_muffins_l331_33185

-- Definitions based on conditions
def muffinRecipeMakes : ℕ := 6
def numberOfStudents : ℕ := 24
def durationInMonths : ℕ := 9

-- Theorem to prove the given problem
theorem georgia_makes_muffins :
  (numberOfStudents / muffinRecipeMakes) * durationInMonths = 36 :=
by
  -- We'll skip the proof with sorry
  sorry

end NUMINAMATH_GPT_georgia_makes_muffins_l331_33185


namespace NUMINAMATH_GPT_washing_machine_regular_wash_l331_33176

variable {R : ℕ}

/-- A washing machine uses 20 gallons of water for a heavy wash,
2 gallons of water for a light wash, and an additional light wash
is added when bleach is used. Given conditions:
- Two heavy washes are done.
- Three regular washes are done.
- One light wash is done.
- Two loads are bleached.
- Total water used is 76 gallons.
Prove the washing machine uses 10 gallons of water for a regular wash. -/
theorem washing_machine_regular_wash (h : 2 * 20 + 3 * R + 1 * 2 + 2 * 2 = 76) : R = 10 :=
by
  sorry

end NUMINAMATH_GPT_washing_machine_regular_wash_l331_33176


namespace NUMINAMATH_GPT_oak_trees_in_park_l331_33112

theorem oak_trees_in_park (planting_today : ℕ) (total_trees : ℕ) 
  (h1 : planting_today = 4) (h2 : total_trees = 9) : 
  total_trees - planting_today = 5 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_oak_trees_in_park_l331_33112


namespace NUMINAMATH_GPT_ratio_of_ages_three_years_from_now_l331_33105

theorem ratio_of_ages_three_years_from_now :
  ∃ L B : ℕ,
  (L + B = 6) ∧ 
  (L = (1/2 : ℝ) * B) ∧ 
  (L + 3 = 5) ∧ 
  (B + 3 = 7) → 
  (L + 3) / (B + 3) = (5/7 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_ages_three_years_from_now_l331_33105


namespace NUMINAMATH_GPT_center_circle_sum_l331_33118

theorem center_circle_sum (h k : ℝ) :
  (∃ h k : ℝ, h + k = 6 ∧ ∃ R, (x - h)^2 + (y - k)^2 = R^2) ↔ ∃ h k : ℝ, h = 3 ∧ k = 3 ∧ h + k = 6 := 
by
  sorry

end NUMINAMATH_GPT_center_circle_sum_l331_33118


namespace NUMINAMATH_GPT_Jamie_correct_percentage_l331_33149

theorem Jamie_correct_percentage (y : ℕ) : ((8 * y - 2 * y : ℕ) / (8 * y : ℕ) : ℚ) * 100 = 75 := by
  sorry

end NUMINAMATH_GPT_Jamie_correct_percentage_l331_33149


namespace NUMINAMATH_GPT_time_after_2345_minutes_l331_33135

-- Define the constants
def minutesInHour : Nat := 60
def hoursInDay : Nat := 24
def startTime : Nat := 0 -- midnight on January 1, 2022, treated as 0 minutes.

-- Prove the equivalent time after 2345 minutes
theorem time_after_2345_minutes :
    let totalMinutes := 2345
    let totalHours := totalMinutes / minutesInHour
    let remainingMinutes := totalMinutes % minutesInHour
    let totalDays := totalHours / hoursInDay
    let remainingHours := totalHours % hoursInDay
    startTime + totalDays * hoursInDay * minutesInHour + remainingHours * minutesInHour + remainingMinutes = startTime + 1 * hoursInDay * minutesInHour + 15 * minutesInHour + 5 :=
    by
    sorry

end NUMINAMATH_GPT_time_after_2345_minutes_l331_33135


namespace NUMINAMATH_GPT_mikes_age_is_18_l331_33166

-- Define variables for Mike's age (m) and his uncle's age (u)
variables (m u : ℕ)

-- Condition 1: Mike is 18 years younger than his uncle
def condition1 : Prop := m = u - 18

-- Condition 2: The sum of their ages is 54 years
def condition2 : Prop := m + u = 54

-- Statement: Prove that Mike's age is 18 given the conditions
theorem mikes_age_is_18 (h1 : condition1 m u) (h2 : condition2 m u) : m = 18 :=
by
  -- Proof skipped with sorry
  sorry

end NUMINAMATH_GPT_mikes_age_is_18_l331_33166


namespace NUMINAMATH_GPT_dot_product_vec1_vec2_l331_33145

-- Define the vectors
def vec1 := (⟨-4, -1⟩ : ℤ × ℤ)
def vec2 := (⟨6, 8⟩ : ℤ × ℤ)

-- Define the dot product function
def dot_product (v1 v2 : ℤ × ℤ) : ℤ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Prove that the dot product of vec1 and vec2 is -32
theorem dot_product_vec1_vec2 : dot_product vec1 vec2 = -32 :=
by
  sorry

end NUMINAMATH_GPT_dot_product_vec1_vec2_l331_33145


namespace NUMINAMATH_GPT_non_deg_ellipse_b_l331_33161

theorem non_deg_ellipse_b (b : ℝ) : 
  (∃ x y : ℝ, x^2 + 9*y^2 - 6*x + 27*y = b ∧ (∀ x y : ℝ, (x - 3)^2 + 9*(y + 3/2)^2 = b + 145/4)) → b > -145/4 :=
sorry

end NUMINAMATH_GPT_non_deg_ellipse_b_l331_33161


namespace NUMINAMATH_GPT_least_positive_number_of_linear_combination_of_24_20_l331_33125

-- Define the conditions as integers
def problem_statement (x y : ℤ) : Prop := 24 * x + 20 * y = 4

theorem least_positive_number_of_linear_combination_of_24_20 :
  ∃ (x y : ℤ), (24 * x + 20 * y = 4) := 
by
  sorry

end NUMINAMATH_GPT_least_positive_number_of_linear_combination_of_24_20_l331_33125

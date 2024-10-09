import Mathlib

namespace complement_M_l802_80206

open Set

-- Define the universal set U as the set of all real numbers
def U := ℝ

-- Define the set M as {x | |x| > 2}
def M : Set ℝ := {x | |x| > 2}

-- State that the complement of M (in the universal set U) is [-2, 2]
theorem complement_M : Mᶜ = {x | -2 ≤ x ∧ x ≤ 2} :=
by
  sorry

end complement_M_l802_80206


namespace minimum_a_plus_b_l802_80208

theorem minimum_a_plus_b (a b : ℤ) (h : a * b = 144) : a + b ≥ -24 :=
by sorry

end minimum_a_plus_b_l802_80208


namespace probability_grade_A_l802_80284

-- Defining probabilities
def P_B : ℝ := 0.05
def P_C : ℝ := 0.03

-- Theorem: proving the probability of Grade A
theorem probability_grade_A : 1 - P_B - P_C = 0.92 :=
by
  -- Placeholder for proof
  sorry

end probability_grade_A_l802_80284


namespace max_shirt_price_l802_80226

theorem max_shirt_price (total_budget : ℝ) (entrance_fee : ℝ) (num_shirts : ℝ) 
  (discount_rate : ℝ) (tax_rate : ℝ) (max_price : ℝ) 
  (budget_after_fee : total_budget - entrance_fee = 195)
  (shirt_discount : num_shirts > 15 → discounted_price = num_shirts * max_price * (1 - discount_rate))
  (price_with_tax : discounted_price * (1 + tax_rate) ≤ 195) : 
  max_price ≤ 10 := 
sorry

end max_shirt_price_l802_80226


namespace uniform_heights_l802_80207

theorem uniform_heights (varA varB : ℝ) (hA : varA = 0.56) (hB : varB = 2.1) : varA < varB := by
  rw [hA, hB]
  exact (by norm_num)

end uniform_heights_l802_80207


namespace solve_for_angle_a_l802_80279

theorem solve_for_angle_a (a b c d e : ℝ) (h1 : a + b + c + d = 360) (h2 : e = 360 - (a + d)) : a = 360 - e - b - c :=
by
  sorry

end solve_for_angle_a_l802_80279


namespace identity_1_over_n_n_plus_1_sum_series_1_over_k_k_plus_1_sum_series_1_over_3k_minus_2_3k_plus_1_l802_80232

-- Question 1: Prove the given identity for 1/(n(n+1))
theorem identity_1_over_n_n_plus_1 (n : ℕ) (hn : n ≠ 0) : 
  (1 : ℚ) / (n * (n + 1)) = (1 : ℚ) / n - (1 : ℚ) / (n + 1) :=
by
  sorry

-- Question 2: Prove the sum of series 1/k(k+1) from k=1 to k=2021
theorem sum_series_1_over_k_k_plus_1 : 
  (Finset.range 2021).sum (λ k => (1 : ℚ) / (k+1) / (k+2)) = 2021 / 2022 :=
by
  sorry

-- Question 3: Prove the sum of series 1/(3k-2)(3k+1) from k=1 to k=673
theorem sum_series_1_over_3k_minus_2_3k_plus_1 : 
  (Finset.range 673).sum (λ k => (1 : ℚ) / ((3 * k + 1 - 2) * (3 * k + 1))) = 674 / 2023 :=
by
  sorry

end identity_1_over_n_n_plus_1_sum_series_1_over_k_k_plus_1_sum_series_1_over_3k_minus_2_3k_plus_1_l802_80232


namespace triangle_area_l802_80274

theorem triangle_area : 
  ∀ (A B C : ℝ × ℝ), 
  A = (0, 0) → 
  B = (4, 0) → 
  C = (2, 6) → 
  (1 / 2 : ℝ) * (4 : ℝ) * (6 : ℝ) = (12.0 : ℝ) := 
by 
  intros A B C hA hB hC
  simp [hA, hB, hC]
  norm_num

end triangle_area_l802_80274


namespace ratio_female_to_male_l802_80282

-- Definitions for the conditions
def average_age_female (f : ℕ) : ℕ := 40 * f
def average_age_male (m : ℕ) : ℕ := 25 * m
def average_age_total (f m : ℕ) : ℕ := (30 * (f + m))

-- Statement to prove
theorem ratio_female_to_male (f m : ℕ) 
  (h_avg_f: average_age_female f = 40 * f)
  (h_avg_m: average_age_male m = 25 * m)
  (h_avg_total: average_age_total f m = 30 * (f + m)) : 
  f / m = 1 / 2 :=
by
  sorry

end ratio_female_to_male_l802_80282


namespace same_sign_abc_l802_80213
open Classical

theorem same_sign_abc (a b c : ℝ) (h1 : (b / a) * (c / a) > 1) (h2 : (b / a) + (c / a) ≥ -2) : 
  (a > 0 ∧ b > 0 ∧ c > 0) ∨ (a < 0 ∧ b < 0 ∧ c < 0) :=
sorry

end same_sign_abc_l802_80213


namespace luke_total_points_l802_80250

/-- Luke gained 327 points in each round of a trivia game. 
    He played 193 rounds of the game. 
    How many points did he score in total? -/
theorem luke_total_points : 327 * 193 = 63111 :=
by
  sorry

end luke_total_points_l802_80250


namespace probability_exact_n_points_l802_80202

open Classical

noncomputable def probability_of_n_points (n : ℕ) : ℚ :=
  1/3 * (2 + (-1/2)^n)

theorem probability_exact_n_points (n : ℕ) :
  ∀ n : ℕ, probability_of_n_points n = 1/3 * (2 + (-1/2)^n) :=
sorry

end probability_exact_n_points_l802_80202


namespace arithmetic_sequence_sum_l802_80234

theorem arithmetic_sequence_sum (a₁ d S : ℤ)
  (ha : 10 * a₁ + 24 * d = 37) :
  19 * (a₁ + 2 * d) + (a₁ + 10 * d) = 74 :=
by
  sorry

end arithmetic_sequence_sum_l802_80234


namespace sum_of_possible_ks_l802_80287

theorem sum_of_possible_ks :
  ∃ S : Finset ℕ, (∀ (j k : ℕ), j > 0 ∧ k > 0 → (1 / j + 1 / k = 1 / 4) ↔ k ∈ S) ∧ S.sum id = 51 :=
  sorry

end sum_of_possible_ks_l802_80287


namespace distance_to_x_axis_P_l802_80295

-- The coordinates of point P
def P : ℝ × ℝ := (3, -2)

-- The distance from point P to the x-axis
def distance_to_x_axis (point : ℝ × ℝ) : ℝ :=
  abs (point.snd)

theorem distance_to_x_axis_P : distance_to_x_axis P = 2 :=
by
  -- Use the provided point P and calculate the distance
  sorry

end distance_to_x_axis_P_l802_80295


namespace intersection_of_lines_l802_80244

-- Define the conditions of the problem
def first_line (x y : ℝ) : Prop := y = -3 * x + 1
def second_line (x y : ℝ) : Prop := y + 1 = 15 * x

-- Prove the intersection point of the two lines
theorem intersection_of_lines : 
  ∃ (x y : ℝ), first_line x y ∧ second_line x y ∧ x = 1 / 9 ∧ y = 2 / 3 :=
by
  sorry

end intersection_of_lines_l802_80244


namespace line_problems_l802_80204

noncomputable def l1 : (ℝ → ℝ) := λ x => x - 1
noncomputable def l2 (k : ℝ) : (ℝ → ℝ) := λ x => -(k + 1) / k * x - 1

theorem line_problems (k : ℝ) :
  ∃ k, k = 0 → (l2 k 1) = 90 →      -- A
  (∀ k, (l1 1 = l2 k 1 → True)) →   -- B
  (∀ k, (l1 1 ≠ l2 k 1 → True)) →   -- C (negated conclusion from False in C)
  (∀ k, (l1 1 * l2 k 1 ≠ -1))       -- D
:=
sorry

end line_problems_l802_80204


namespace bar_graph_proportion_correct_l802_80203

def white : ℚ := 1/2
def black : ℚ := 1/4
def gray : ℚ := 1/8
def light_gray : ℚ := 1/16

theorem bar_graph_proportion_correct :
  (white = 1 / 2) ∧
  (black = white / 2) ∧
  (gray = black / 2) ∧
  (light_gray = gray / 2) →
  (white = 1 / 2) ∧
  (black = 1 / 4) ∧
  (gray = 1 / 8) ∧
  (light_gray = 1 / 16) :=
by
  intros
  sorry

end bar_graph_proportion_correct_l802_80203


namespace vertex_of_quadratic_function_l802_80260

-- Define the function and constants
variables (p q : ℝ)
  (hp : p > 0)
  (hq : q > 0)

-- State the theorem
theorem vertex_of_quadratic_function : 
  ∀ p q : ℝ, p > 0 → q > 0 → 
  (∀ x : ℝ, x = - (2 * p) / (2 : ℝ) → x = -p) := 
sorry

end vertex_of_quadratic_function_l802_80260


namespace simplify_radical_expression_l802_80239

theorem simplify_radical_expression :
  (Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7) :=
by
  sorry

end simplify_radical_expression_l802_80239


namespace carol_first_to_roll_six_l802_80264

def probability_roll (x : ℕ) (success : ℕ) : ℚ := success / x

def first_to_roll_six_probability (a b c : ℕ) : ℚ :=
  let p_six : ℚ := probability_roll 6 1
  let p_not_six : ℚ := 1 - p_six
  let cycle_prob : ℚ := p_not_six * p_not_six * p_six
  let continue_prob : ℚ := p_not_six * p_not_six * p_not_six
  let geometric_sum : ℚ := cycle_prob / (1 - continue_prob)
  geometric_sum

theorem carol_first_to_roll_six :
  first_to_roll_six_probability 1 1 1 = 25 / 91 := 
sorry

end carol_first_to_roll_six_l802_80264


namespace geometric_series_sum_l802_80220

noncomputable def T (r : ℝ) := 15 / (1 - r)

theorem geometric_series_sum (b : ℝ) (hb1 : -1 < b) (hb2 : b < 1) (H : T b * T (-b) = 3240) : T b + T (-b) = 432 := 
by sorry

end geometric_series_sum_l802_80220


namespace quadratic_roots_l802_80238

theorem quadratic_roots (c : ℝ) 
  (h : ∀ x : ℝ, (x^2 - 3*x + c = 0) ↔ (x = (3 + Real.sqrt c) / 2 ∨ x = (3 - Real.sqrt c) / 2)) :
  c = 9 / 5 :=
by
  sorry

end quadratic_roots_l802_80238


namespace find_a_l802_80276

variable {x y a : ℝ}

theorem find_a (h1 : 2 * x - y + a ≥ 0) (h2 : 3 * x + y ≤ 3) (h3 : ∀ (x y : ℝ), 4 * x + 3 * y ≤ 8) : a = 2 := 
sorry

end find_a_l802_80276


namespace total_weight_of_ripe_fruits_correct_l802_80201

-- Definitions based on conditions
def total_apples : ℕ := 14
def total_pears : ℕ := 10
def total_lemons : ℕ := 5

def ripe_apple_weight : ℕ := 150
def ripe_pear_weight : ℕ := 200
def ripe_lemon_weight : ℕ := 100

def unripe_apples : ℕ := 6
def unripe_pears : ℕ := 4
def unripe_lemons : ℕ := 2

def total_weight_of_ripe_fruits : ℕ :=
  (total_apples - unripe_apples) * ripe_apple_weight +
  (total_pears - unripe_pears) * ripe_pear_weight +
  (total_lemons - unripe_lemons) * ripe_lemon_weight

theorem total_weight_of_ripe_fruits_correct :
  total_weight_of_ripe_fruits = 2700 :=
by
  -- proof goes here (use sorry to skip the actual proof)
  sorry

end total_weight_of_ripe_fruits_correct_l802_80201


namespace roots_of_quadratic_expression_l802_80296

theorem roots_of_quadratic_expression :
    (∀ x: ℝ, (x^2 + 3 * x - 2 = 0) → ∃ x₁ x₂: ℝ, x = x₁ ∨ x = x₂) ∧ 
    (∀ x₁ x₂ : ℝ, (x₁ + x₂ = -3) ∧ (x₁ * x₂ = -2) → x₁^2 + 2 * x₁ - x₂ = 5) :=
by
  sorry

end roots_of_quadratic_expression_l802_80296


namespace y1_gt_y2_l802_80223

theorem y1_gt_y2 (k : ℝ) (y1 y2 : ℝ) (hA : y1 = k * (-3) + 3) (hB : y2 = k * 1 + 3) (hK : k < 0) : y1 > y2 :=
by 
  sorry

end y1_gt_y2_l802_80223


namespace smallest_positive_integer_l802_80285

theorem smallest_positive_integer :
  ∃ x : ℕ,
    x % 5 = 4 ∧
    x % 7 = 5 ∧
    x % 11 = 9 ∧
    x % 13 = 11 ∧
    (∀ y : ℕ, (y % 5 = 4 ∧ y % 7 = 5 ∧ y % 11 = 9 ∧ y % 13 = 11) → y ≥ x) ∧ x = 999 :=
by
  sorry

end smallest_positive_integer_l802_80285


namespace mean_and_variance_l802_80242

def scores_A : List ℝ := [8, 9, 14, 15, 15, 16, 21, 22]
def scores_B : List ℝ := [7, 8, 13, 15, 15, 17, 22, 23]

noncomputable def mean (l : List ℝ) : ℝ := (l.sum) / (l.length)
noncomputable def variance (l : List ℝ) : ℝ := mean (l.map (λ x => (x - (mean l)) ^ 2))

theorem mean_and_variance :
  (mean scores_A = mean scores_B) ∧ (variance scores_A < variance scores_B) :=
by
  sorry

end mean_and_variance_l802_80242


namespace first_digit_after_decimal_correct_l802_80271

noncomputable def first_digit_after_decimal (n: ℕ) : ℕ :=
  if n % 2 = 0 then 9 else 4

theorem first_digit_after_decimal_correct (n : ℕ) :
  (first_digit_after_decimal n = 9 ↔ n % 2 = 0) ∧ (first_digit_after_decimal n = 4 ↔ n % 2 = 1) :=
by
  sorry

end first_digit_after_decimal_correct_l802_80271


namespace minimum_boxes_to_eliminate_l802_80275

theorem minimum_boxes_to_eliminate (total_boxes remaining_boxes : ℕ) 
  (high_value_boxes : ℕ) (h1 : total_boxes = 30) (h2 : high_value_boxes = 10)
  (h3 : remaining_boxes = total_boxes - 20) :
  remaining_boxes ≥ high_value_boxes → remaining_boxes = 10 :=
by 
  sorry

end minimum_boxes_to_eliminate_l802_80275


namespace work_done_in_days_l802_80253

theorem work_done_in_days (M B : ℕ) (x : ℕ) 
  (h1 : 12 * 2 * B + 16 * B = 200 * B / 5) 
  (h2 : 13 * 2 * B + 24 * B = 50 * x * B)
  (h3 : M = 2 * B) : 
  x = 4 := 
by
  sorry

end work_done_in_days_l802_80253


namespace find_c_l802_80221

theorem find_c (a b c : ℚ) (h_eqn : ∀ y, a * y^2 + b * y + c = y^2 / 12 + 5 * y / 6 + 145 / 12)
  (h_vertex : ∀ x, x = a * (-5)^2 + b * (-5) + c)
  (h_pass : a * (-1 + 5)^2 + 1 = 4) :
  c = 145 / 12 := by
sorry

end find_c_l802_80221


namespace find_teacher_age_l802_80289

noncomputable def age_of_teacher (avg_age_students : ℕ) (num_students : ℕ) 
                                (avg_age_inclusive : ℕ) (num_people_inclusive : ℕ) : ℕ :=
  let total_age_students := num_students * avg_age_students
  let total_age_inclusive := num_people_inclusive * avg_age_inclusive
  total_age_inclusive - total_age_students

theorem find_teacher_age : age_of_teacher 15 10 16 11 = 26 := 
by 
  sorry

end find_teacher_age_l802_80289


namespace count_numbers_with_digit_sum_10_l802_80217

theorem count_numbers_with_digit_sum_10 : 
  ∃ n : ℕ, 
  (n = 66) ∧ ∀ (a b c : ℕ), 
  0 ≤ a ∧ a ≤ 9 ∧ 
  0 ≤ b ∧ b ≤ 9 ∧ 
  0 ≤ c ∧ c ≤ 9 ∧ 
  a + b + c = 10 → 
  true :=
by
  sorry

end count_numbers_with_digit_sum_10_l802_80217


namespace ratio_of_typing_speeds_l802_80233

-- Defining Tim's and Tom's typing speeds
variables (T M : ℝ)

-- Conditions given in the problem
def condition1 : Prop := T + M = 15
def condition2 : Prop := T + 1.6 * M = 18

-- Conclusion to be proved: the ratio of M to T is 1:2
theorem ratio_of_typing_speeds (h1 : condition1 T M) (h2 : condition2 T M) :
  M / T = 1 / 2 :=
by
  -- skip the proof
  sorry

end ratio_of_typing_speeds_l802_80233


namespace mother_younger_than_father_l802_80218

variable (total_age : ℕ) (father_age : ℕ) (brother_age : ℕ) (sister_age : ℕ) (kaydence_age : ℕ) (mother_age : ℕ)

noncomputable def family_data : Prop :=
  total_age = 200 ∧
  father_age = 60 ∧
  brother_age = father_age / 2 ∧
  sister_age = 40 ∧
  kaydence_age = 12 ∧
  mother_age = total_age - (father_age + brother_age + sister_age + kaydence_age)

theorem mother_younger_than_father :
  family_data total_age father_age brother_age sister_age kaydence_age mother_age →
  father_age - mother_age = 2 :=
sorry

end mother_younger_than_father_l802_80218


namespace find_coordinates_of_P_l802_80265

/-- Let the curve C be defined by the equation y = x^3 - 10x + 3 and point P lies on this curve in the second quadrant.
We are given that the slope of the tangent line to the curve at point P is 2. We need to find the coordinates of P.
--/
theorem find_coordinates_of_P :
  ∃ (x y : ℝ), (y = x ^ 3 - 10 * x + 3) ∧ (3 * x ^ 2 - 10 = 2) ∧ (x < 0) ∧ (x = -2) ∧ (y = 15) :=
by
  sorry

end find_coordinates_of_P_l802_80265


namespace find_num_male_general_attendees_l802_80286

def num_attendees := 1000
def num_presenters := 420
def total_general_attendees := num_attendees - num_presenters

variables (M_p F_p M_g F_g : ℕ)

axiom condition1 : M_p = F_p + 20
axiom condition2 : M_p + F_p = 420
axiom condition3 : F_g = M_g + 56
axiom condition4 : M_g + F_g = total_general_attendees

theorem find_num_male_general_attendees :
  M_g = 262 :=
by
  sorry

end find_num_male_general_attendees_l802_80286


namespace smallest_possible_N_l802_80235

theorem smallest_possible_N (N : ℕ) (h : ∀ m : ℕ, m ≤ 60 → m % 3 = 0 → ∃ i : ℕ, i < 20 ∧ m = 3 * i + 1 ∧ N = 20) :
    N = 20 :=
by 
  sorry

end smallest_possible_N_l802_80235


namespace unique_two_digit_solution_l802_80252

theorem unique_two_digit_solution :
  ∃! (u : ℕ), 9 < u ∧ u < 100 ∧ 13 * u % 100 = 52 := 
sorry

end unique_two_digit_solution_l802_80252


namespace awareness_not_related_to_education_level_l802_80299

def low_education : ℕ := 35 + 35 + 80 + 40 + 60 + 150
def high_education : ℕ := 55 + 64 + 6 + 110 + 140 + 25

def a : ℕ := 150
def b : ℕ := 125
def c : ℕ := 250
def d : ℕ := 275
def n : ℕ := 800

-- K^2 calculation
def K2 : ℚ := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Critical value for 95% confidence
def critical_value_95 : ℚ := 3.841

theorem awareness_not_related_to_education_level : K2 < critical_value_95 :=
by
  -- proof to be added here
  sorry

end awareness_not_related_to_education_level_l802_80299


namespace determine_guilty_defendant_l802_80270

-- Define the defendants
inductive Defendant
| A
| B
| C

open Defendant

-- Define the guilty defendant
def guilty_defendant : Defendant := C

-- Define the conditions
def condition1 (d : Defendant) : Prop :=
d ≠ A ∧ d ≠ B ∧ d ≠ C → false  -- "There were three defendants, and only one of them was guilty."

def condition2 (d : Defendant) : Prop :=
d = A → d ≠ B  -- "Defendant A accused defendant B."

def condition3 (d : Defendant) : Prop :=
d = B → d = B  -- "Defendant B admitted to being guilty."

def condition4 (d : Defendant) : Prop :=
d = C → (d = C ∨ d = A)  -- "Defendant C either admitted to being guilty or accused A."

-- The proof problem statement
theorem determine_guilty_defendant :
  (∃ d : Defendant, condition1 d ∧ condition2 d ∧ condition3 d ∧ condition4 d) → guilty_defendant = C :=
by {
  sorry
}

end determine_guilty_defendant_l802_80270


namespace total_weight_of_carrots_and_cucumbers_is_875_l802_80267

theorem total_weight_of_carrots_and_cucumbers_is_875 :
  ∀ (carrots : ℕ) (cucumbers : ℕ),
    carrots = 250 →
    cucumbers = (5 * carrots) / 2 →
    carrots + cucumbers = 875 := 
by
  intros carrots cucumbers h_carrots h_cucumbers
  rw [h_carrots, h_cucumbers]
  sorry

end total_weight_of_carrots_and_cucumbers_is_875_l802_80267


namespace range_of_4x_plus_2y_l802_80292

theorem range_of_4x_plus_2y (x y : ℝ) 
  (h₁ : 1 ≤ x + y ∧ x + y ≤ 3)
  (h₂ : -1 ≤ x - y ∧ x - y ≤ 1) : 
  2 ≤ 4 * x + 2 * y ∧ 4 * x + 2 * y ≤ 10 :=
sorry

end range_of_4x_plus_2y_l802_80292


namespace garageHasWheels_l802_80261

-- Define the conditions
def bikeWheelsPerBike : Nat := 2
def bikesInGarage : Nat := 10

-- State the theorem to be proved
theorem garageHasWheels : bikesInGarage * bikeWheelsPerBike = 20 := by
  sorry

end garageHasWheels_l802_80261


namespace solve_equations_l802_80269

theorem solve_equations :
  (∀ x : ℝ, (1 / 2) * (2 * x - 5) ^ 2 - 2 = 0 ↔ x = 7 / 2 ∨ x = 3 / 2) ∧
  (∀ x : ℝ, x ^ 2 - 4 * x - 4 = 0 ↔ x = 2 + 2 * Real.sqrt 2 ∨ x = 2 - 2 * Real.sqrt 2) :=
by
  sorry

end solve_equations_l802_80269


namespace calculate_expression_l802_80251

theorem calculate_expression : 
  (0.25 ^ 16) * ((-4) ^ 17) = -4 := 
by
  sorry

end calculate_expression_l802_80251


namespace largest_exterior_angle_l802_80268

theorem largest_exterior_angle (x : ℝ) (h1 : 3 * x + 4 * x + 5 * x = 180) : 180 - 3 * (180 / 12) = 135 :=
by {
  -- Sorry is a placeholder for the actual proof
  sorry
}

end largest_exterior_angle_l802_80268


namespace geometric_sequence_sum_l802_80231

theorem geometric_sequence_sum :
  let a := (1/2 : ℚ)
  let r := (1/3 : ℚ)
  let n := 6
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 364 / 243 :=
by
  sorry

end geometric_sequence_sum_l802_80231


namespace M_intersect_N_equals_M_l802_80256

-- Define the sets M and N
def M := { x : ℝ | x^2 - 3 * x + 2 = 0 }
def N := { x : ℝ | x * (x - 1) * (x - 2) = 0 }

-- The theorem we want to prove
theorem M_intersect_N_equals_M : M ∩ N = M := 
by 
  sorry

end M_intersect_N_equals_M_l802_80256


namespace scientific_notation_of_130944000000_l802_80291

theorem scientific_notation_of_130944000000 :
  130944000000 = 1.30944 * 10^11 :=
by sorry

end scientific_notation_of_130944000000_l802_80291


namespace find_multiplier_l802_80211

theorem find_multiplier (x : ℤ) : 
  30 * x - 138 = 102 ↔ x = 8 := 
by
  sorry

end find_multiplier_l802_80211


namespace intersection_at_7_m_l802_80229

def f (x : Int) (d : Int) : Int := 4 * x + d

theorem intersection_at_7_m (d m : Int) (h₁ : f 7 d = m) (h₂ : 7 = f m d) : m = 7 := by
  sorry

end intersection_at_7_m_l802_80229


namespace eval_expr_l802_80200

theorem eval_expr : (2/5) + (3/8) - (1/10) = 27/40 :=
by
  sorry

end eval_expr_l802_80200


namespace num_of_positive_divisors_l802_80263

-- Given conditions
variables {x y z : ℕ}
variables (p1 p2 p3 : ℕ) -- primes
variables (h1 : x = p1 ^ 3) (h2 : y = p2 ^ 3) (h3 : z = p3 ^ 3)
variables (hx : x ≠ y) (hy : y ≠ z) (hz : z ≠ x)

-- Lean statement to prove
theorem num_of_positive_divisors (hx3 : x = p1 ^ 3) (hy3 : y = p2 ^ 3) (hz3 : z = p3 ^ 3) 
    (Hdist : p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) : 
    ∃ n : ℕ, n = 10 * 13 * 7 ∧ n = (x^3 * y^4 * z^2).factors.length :=
sorry

end num_of_positive_divisors_l802_80263


namespace aunt_angela_nieces_l802_80259

theorem aunt_angela_nieces (total_jellybeans : ℕ)
                           (jellybeans_per_child : ℕ)
                           (num_nephews : ℕ)
                           (num_nieces : ℕ) 
                           (total_children : ℕ) 
                           (h1 : total_jellybeans = 70)
                           (h2 : jellybeans_per_child = 14)
                           (h3 : num_nephews = 3)
                           (h4 : total_children = total_jellybeans / jellybeans_per_child)
                           (h5 : total_children = num_nephews + num_nieces) :
                           num_nieces = 2 :=
by
  sorry

end aunt_angela_nieces_l802_80259


namespace equilateral_triangle_fixed_area_equilateral_triangle_max_area_l802_80230

theorem equilateral_triangle_fixed_area (a b c : ℝ) (Δ : ℝ) (s : ℝ) (R : ℝ) (is_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  (s = (a + b + c) / 2 ∧ Δ = Real.sqrt (s * (s - a) * (s - b) * (s - c))) →
  (a * b * c = minimized ∨ a + b + c = minimized ∨ a^2 + b^2 + c^2 = minimized ∨ R = minimized) →
    (a = b ∧ b = c) :=
by
  sorry

theorem equilateral_triangle_max_area (a b c : ℝ) (Δ : ℝ) (s : ℝ) (R : ℝ) (is_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  (s = (a + b + c) / 2 ∧ Δ = Real.sqrt (s * (s - a) * (s - b) * (s - c))) →
  (a * b * c = fixed ∨ a + b + c = fixed ∨ a^2 + b^2 + c^2 = fixed ∨ R = fixed) →
  (Δ = maximized) →
    (a = b ∧ b = c) :=
by
  sorry

end equilateral_triangle_fixed_area_equilateral_triangle_max_area_l802_80230


namespace pages_and_cost_calculation_l802_80277

noncomputable def copy_pages_cost (cents_per_5_pages : ℕ) (total_cents : ℕ) (discount_threshold : ℕ) (discount_rate : ℝ) : ℝ :=
if total_cents < discount_threshold * (cents_per_5_pages / 5) then
  total_cents / (cents_per_5_pages / 5)
else
  let num_pages_before_discount := discount_threshold
  let remaining_pages := total_cents / (cents_per_5_pages / 5) - num_pages_before_discount
  let cost_before_discount := num_pages_before_discount * (cents_per_5_pages / 5)
  let discounted_cost := remaining_pages * (cents_per_5_pages / 5) * (1 - discount_rate)
  cost_before_discount + discounted_cost

theorem pages_and_cost_calculation :
  let cents_per_5_pages := 10
  let total_cents := 5000
  let discount_threshold := 1000
  let discount_rate := 0.10
  let num_pages := (cents_per_5_pages * 2500) / 5
  let cost := copy_pages_cost cents_per_5_pages total_cents discount_threshold discount_rate
  (num_pages = 2500) ∧ (cost = 4700) :=
by
  sorry

end pages_and_cost_calculation_l802_80277


namespace products_arrangement_count_l802_80262

/--
There are five different products: A, B, C, D, and E arranged in a row on a shelf.
- Products A and B must be adjacent.
- Products C and D must not be adjacent.
Prove that there are a total of 24 distinct valid arrangements under these conditions.
-/
theorem products_arrangement_count : 
  ∃ (n : ℕ), 
  (∀ (A B C D E : Type), n = 24 ∧
  ∀ l : List (Type), l = [A, B, C, D, E] ∧
  -- A and B must be adjacent
  (∀ p : List (Type), p = [A, B] ∨ p = [B, A]) ∧
  -- C and D must not be adjacent
  ¬ (∀ q : List (Type), q = [C, D] ∨ q = [D, C])) :=
sorry

end products_arrangement_count_l802_80262


namespace evaluate_expression_l802_80210

theorem evaluate_expression : (5 * 3 ^ 4 + 6 * 4 ^ 3 = 789) :=
by
  sorry

end evaluate_expression_l802_80210


namespace inequality_solution_l802_80290

theorem inequality_solution (x : ℝ) :
  (2 * x - 1 > 0 ∧ x + 1 ≤ 3) ↔ (1 / 2 < x ∧ x ≤ 2) :=
by
  sorry

end inequality_solution_l802_80290


namespace lollipop_count_l802_80237

theorem lollipop_count (total_cost one_lollipop_cost : ℚ) (h1 : total_cost = 90) (h2 : one_lollipop_cost = 0.75) : total_cost / one_lollipop_cost = 120 :=
by
  sorry

end lollipop_count_l802_80237


namespace prove_sufficient_and_necessary_l802_80273

-- The definition of the focus of the parabola y^2 = 4x.
def focus_parabola : (ℝ × ℝ) := (1, 0)

-- The condition that the line passes through a given point.
def line_passes_through (m b : ℝ) (p : ℝ × ℝ) : Prop := 
  p.2 = m * p.1 + b

-- Let y = x + b and the equation of the parabola be y^2 = 4x.
def sufficient_and_necessary (b : ℝ) : Prop :=
  line_passes_through 1 b focus_parabola ↔ b = -1

theorem prove_sufficient_and_necessary : sufficient_and_necessary (-1) :=
by
  sorry

end prove_sufficient_and_necessary_l802_80273


namespace solve_for_t_l802_80241

variable (S₁ S₂ u t : ℝ)

theorem solve_for_t 
  (h₀ : u ≠ 0) 
  (h₁ : u = (S₁ - S₂) / (t - 1)) :
  t = (S₁ - S₂ + u) / u :=
by
  sorry

end solve_for_t_l802_80241


namespace sec_225_eq_neg_sqrt2_csc_225_eq_neg_sqrt2_l802_80288

noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ
noncomputable def csc (θ : ℝ) : ℝ := 1 / Real.sin θ

theorem sec_225_eq_neg_sqrt2 :
  sec (225 * Real.pi / 180) = -Real.sqrt 2 := sorry

theorem csc_225_eq_neg_sqrt2 :
  csc (225 * Real.pi / 180) = -Real.sqrt 2 := sorry

end sec_225_eq_neg_sqrt2_csc_225_eq_neg_sqrt2_l802_80288


namespace phil_packs_duration_l802_80278

noncomputable def total_cards_left_after_fire : ℕ := 520
noncomputable def total_cards_initially : ℕ := total_cards_left_after_fire * 2
noncomputable def cards_per_pack : ℕ := 20
noncomputable def packs_bought_weeks : ℕ := total_cards_initially / cards_per_pack

theorem phil_packs_duration : packs_bought_weeks = 52 := by
  sorry

end phil_packs_duration_l802_80278


namespace circle_area_from_circumference_l802_80209

theorem circle_area_from_circumference
  (c : ℝ)    -- the circumference
  (hc : c = 36)    -- condition: circumference is 36 cm
  : 
  ∃ A : ℝ,   -- there exists an area A
    A = 324 / π :=   -- conclusion: area is 324/π
by
  sorry   -- proof goes here

end circle_area_from_circumference_l802_80209


namespace min_students_l802_80257

theorem min_students (M D : ℕ) (hD : D = 5) (h_ratio : (M: ℚ) / (M + D) > 0.6) : M + D = 13 :=
by 
  sorry

end min_students_l802_80257


namespace derivative_y_l802_80249

noncomputable def y (x : ℝ) : ℝ := Real.sin x - Real.exp (x * Real.log 2)

theorem derivative_y (x : ℝ) : 
  deriv y x = Real.cos x - Real.exp (x * Real.log 2) * Real.log 2 := 
by 
  sorry

end derivative_y_l802_80249


namespace log_eq_solution_l802_80272

theorem log_eq_solution (x : ℝ) (h : Real.log 8 / Real.log x = Real.log 5 / Real.log 125) : x = 512 := by
  sorry

end log_eq_solution_l802_80272


namespace min_omega_l802_80216

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem min_omega (ω φ T : ℝ) (hω : ω > 0) (hφ1 : 0 < φ) (hφ2 : φ < Real.pi / 2)
  (hT : f ω φ T = Real.sqrt 3 / 2)
  (hx : f ω φ (Real.pi / 6) = 0) :
  ω = 4 := by
  sorry

end min_omega_l802_80216


namespace hexagon_diagonal_length_is_twice_side_l802_80246

noncomputable def regular_hexagon_side_length : ℝ := 12

def diagonal_length_in_regular_hexagon (s : ℝ) : ℝ :=
2 * s

theorem hexagon_diagonal_length_is_twice_side :
  diagonal_length_in_regular_hexagon regular_hexagon_side_length = 2 * regular_hexagon_side_length :=
by 
  -- Simplify and check the computation according to the understanding of the properties of the hexagon
  sorry

end hexagon_diagonal_length_is_twice_side_l802_80246


namespace marked_vertices_coincide_l802_80222

theorem marked_vertices_coincide :
  ∀ (P Q : Fin 16 → Prop),
  (∃ A B C D E F G : Fin 16, P A ∧ P B ∧ P C ∧ P D ∧ P E ∧ P F ∧ P G) →
  (∃ A' B' C' D' E' F' G' : Fin 16, Q A' ∧ Q B' ∧ Q C' ∧ Q D' ∧ Q E' ∧ Q F' ∧ Q G') →
  ∃ (r : Fin 16), ∃ (A B C D : Fin 16), 
  (Q ((A + r) % 16) ∧ Q ((B + r) % 16) ∧ Q ((C + r) % 16) ∧ Q ((D + r) % 16)) :=
by
  sorry

end marked_vertices_coincide_l802_80222


namespace incorrect_statement_l802_80280

def data_set : List ℤ := [10, 8, 6, 9, 8, 7, 8]

theorem incorrect_statement : 
  let mode := 8
  let median := 8
  let mean := 8
  let variance := 8
  (∃ x ∈ data_set, x ≠ 8) → -- suppose there is at least one element in the dataset not equal to 8
  (1 / 7 : ℚ) * (4 + 0 + 4 + 1 + 0 + 1 + 0) ≠ 8 := -- calculating real variance from dataset
by
  sorry

end incorrect_statement_l802_80280


namespace seokjin_higher_than_jungkook_l802_80293

variable (Jungkook_yoojeong_seokjin_stairs : ℕ)

def jungkook_stair := 19
def yoojeong_stair := jungkook_stair + 8
def seokjin_stair := yoojeong_stair - 5

theorem seokjin_higher_than_jungkook : seokjin_stair - jungkook_stair = 3 :=
by sorry

end seokjin_higher_than_jungkook_l802_80293


namespace simplify_expression_l802_80219

theorem simplify_expression :
  ((0.3 * 0.2) / (0.4 * 0.5)) - (0.1 * 0.6) = 0.24 :=
by
  sorry

end simplify_expression_l802_80219


namespace min_value_sq_sum_l802_80205

theorem min_value_sq_sum (x1 x2 : ℝ) (h : x1 * x2 = 2013) : (x1 + x2)^2 ≥ 8052 :=
by
  sorry

end min_value_sq_sum_l802_80205


namespace average_student_headcount_l802_80281

theorem average_student_headcount (headcount_03_04 headcount_04_05 : ℕ) 
  (h1 : headcount_03_04 = 10500) 
  (h2 : headcount_04_05 = 10700) : 
  (headcount_03_04 + headcount_04_05) / 2 = 10600 := 
by
  sorry

end average_student_headcount_l802_80281


namespace number_of_guest_cars_l802_80225

-- Definitions and conditions
def total_wheels : ℕ := 48
def mother_car_wheels : ℕ := 4
def father_jeep_wheels : ℕ := 4
def wheels_per_car : ℕ := 4

-- Theorem statement
theorem number_of_guest_cars (total_wheels mother_car_wheels father_jeep_wheels wheels_per_car : ℕ) : ℕ :=
  (total_wheels - (mother_car_wheels + father_jeep_wheels)) / wheels_per_car

-- Specific instance for the problem
example : number_of_guest_cars 48 4 4 4 = 10 := 
by
  sorry

end number_of_guest_cars_l802_80225


namespace find_principal_l802_80247

-- Define the conditions
variables (P R : ℝ) -- Define P and R as real numbers
variable (h : (P * 50) / 100 = 300) -- Introduce the equation obtained from the conditions

-- State the theorem
theorem find_principal (P R : ℝ) (h : (P * 50) / 100 = 300) : P = 600 :=
sorry

end find_principal_l802_80247


namespace minimum_distance_to_recover_cost_l802_80215

theorem minimum_distance_to_recover_cost 
  (initial_consumption : ℝ) (modification_cost : ℝ) (modified_consumption : ℝ) (gas_cost : ℝ) : 
  22000 < (modification_cost / gas_cost) / (initial_consumption - modified_consumption) * 100 ∧ 
  (modification_cost / gas_cost) / (initial_consumption - modified_consumption) * 100 < 26000 :=
by
  let initial_consumption := 8.4
  let modified_consumption := 6.3
  let modification_cost := 400.0
  let gas_cost := 0.80
  sorry

end minimum_distance_to_recover_cost_l802_80215


namespace ab_eq_neg_one_l802_80294

variable (a b : ℝ)

-- Condition for the inequality (x >= 0) -> (0 ≤ x^4 - x^3 + ax + b ≤ (x^2 - 1)^2)
def condition (a b : ℝ) : Prop :=
  ∀ x : ℝ, x ≥ 0 → 
    0 ≤ x^4 - x^3 + a * x + b ∧ 
    x^4 - x^3 + a * x + b ≤ (x^2 - 1)^2

-- Main statement to prove that assuming the condition, a * b = -1
theorem ab_eq_neg_one (h : condition a b) : a * b = -1 := 
  sorry

end ab_eq_neg_one_l802_80294


namespace sum_of_first_15_even_positive_integers_l802_80227

theorem sum_of_first_15_even_positive_integers :
  let a := 2
  let l := 30
  let n := 15
  let S := (a + l) / 2 * n
  S = 240 := by
  sorry

end sum_of_first_15_even_positive_integers_l802_80227


namespace incenter_circumcenter_identity_l802_80248

noncomputable def triangle : Type := sorry
noncomputable def incenter (t : triangle) : Type := sorry
noncomputable def circumcenter (t : triangle) : Type := sorry
noncomputable def inradius (t : triangle) : ℝ := sorry
noncomputable def circumradius (t : triangle) : ℝ := sorry
noncomputable def distance (A B : Type) : ℝ := sorry

theorem incenter_circumcenter_identity (t : triangle) (I O : Type)
  (hI : I = incenter t) (hO : O = circumcenter t)
  (r : ℝ) (h_r : r = inradius t)
  (R : ℝ) (h_R : R = circumradius t) :
  distance I O ^ 2 = R ^ 2 - 2 * R * r :=
sorry

end incenter_circumcenter_identity_l802_80248


namespace wire_around_field_l802_80258

theorem wire_around_field 
  (area_square : ℕ)
  (total_length_wire : ℕ)
  (h_area : area_square = 69696)
  (h_total_length : total_length_wire = 15840) :
  (total_length_wire / (4 * Int.natAbs (Int.sqrt area_square))) = 15 :=
  sorry

end wire_around_field_l802_80258


namespace skipping_rope_equation_correct_l802_80283

-- Definitions of constraints
variable (x : ℕ) -- Number of skips per minute by Xiao Ji
variable (H1 : 0 < x) -- The number of skips per minute by Xiao Ji is positive
variable (H2 : 100 / x * x = 100) -- Xiao Ji skips exactly 100 times

-- Xiao Fan's conditions
variable (H3 : 100 + 20 = 120) -- Xiao Fan skips 20 more times than Xiao Ji
variable (H4 : x + 30 > 0) -- Xiao Fan skips 30 more times per minute than Xiao Ji

-- Prove the equation is correct
theorem skipping_rope_equation_correct :
  100 / x = 120 / (x + 30) :=
by
  sorry

end skipping_rope_equation_correct_l802_80283


namespace Rachel_drinks_correct_glasses_l802_80255

def glasses_Sunday : ℕ := 2
def glasses_Monday : ℕ := 4
def glasses_TuesdayToFriday : ℕ := 3
def days_TuesdayToFriday : ℕ := 4
def ounces_per_glass : ℕ := 10
def total_goal : ℕ := 220
def glasses_Saturday : ℕ := 4

theorem Rachel_drinks_correct_glasses :
  ounces_per_glass * (glasses_Sunday + glasses_Monday + days_TuesdayToFriday * glasses_TuesdayToFriday + glasses_Saturday) = total_goal :=
sorry

end Rachel_drinks_correct_glasses_l802_80255


namespace lines_coinicide_l802_80243

open Real

theorem lines_coinicide (k m n : ℝ) :
  (∃ (x y : ℝ), y = k * x + m ∧ y = m * x + n ∧ y = n * x + k) →
  k = m ∧ m = n :=
by
  sorry

end lines_coinicide_l802_80243


namespace paths_from_A_to_B_via_C_l802_80266

open Classical

-- Definitions based on conditions
variables (lattice : Type) [PartialOrder lattice]
variables (A B C : lattice)
variables (first_red first_blue second_red second_blue first_green second_green orange : lattice)

-- Conditions encoded as hypotheses
def direction_changes : Prop :=
  -- Arrow from first green to orange is now one way from orange to green
  ∀ x : lattice, x = first_green → orange < x ∧ ¬ (x < orange) ∧
  -- Additional stop at point C located directly after the first blue arrows
  (C < first_blue ∨ first_blue < C)

-- Now stating the proof problem
theorem paths_from_A_to_B_via_C :
  direction_changes lattice first_green orange first_blue C →
  -- Total number of paths from A to B via C is 12
  (2 + 2) * 3 * 1 = 12 :=
by
  sorry

end paths_from_A_to_B_via_C_l802_80266


namespace perpendicular_lines_b_value_l802_80240

theorem perpendicular_lines_b_value :
  ( ∀ x y : ℝ, 2 * x + 3 * y + 4 = 0)  →
  ( ∀ x y : ℝ, b * x + 3 * y - 1 = 0) →
  ( - (2 : ℝ) / (3 : ℝ) * - b / (3 : ℝ) = -1 ) →
  b = - (9 : ℝ) / (2 : ℝ) :=
by
  intros h1 h2 h3
  sorry

end perpendicular_lines_b_value_l802_80240


namespace sum_of_common_ratios_eq_three_l802_80236

variable (k p r a2 a3 b2 b3 : ℝ)

-- Conditions on the sequences:
variable (h_nz_k : k ≠ 0)  -- k is nonzero as it is scaling factor
variable (h_seq1 : a2 = k * p)
variable (h_seq2 : a3 = k * p^2)
variable (h_seq3 : b2 = k * r)
variable (h_seq4 : b3 = k * r^2)
variable (h_diff_ratios : p ≠ r)

-- The given equation:
variable (h_eq : a3^2 - b3^2 = 3 * (a2^2 - b2^2))

-- The theorem statement
theorem sum_of_common_ratios_eq_three :
  p^2 + r^2 = 3 :=
by
  -- Introduce the assumptions
  sorry

end sum_of_common_ratios_eq_three_l802_80236


namespace correct_cases_needed_l802_80254

noncomputable def cases_needed (boxes_sold : ℕ) (boxes_per_case : ℕ) : ℕ :=
  (boxes_sold + boxes_per_case - 1) / boxes_per_case

theorem correct_cases_needed :
  cases_needed 10 6 = 2 ∧ -- For trefoils
  cases_needed 15 5 = 3 ∧ -- For samoas
  cases_needed 20 10 = 2  -- For thin mints
:= by
  sorry

end correct_cases_needed_l802_80254


namespace no_common_real_solution_l802_80228

theorem no_common_real_solution :
  ¬ ∃ (x y : ℝ), (x^2 - 6 * x + y + 9 = 0) ∧ (x^2 + 4 * y + 5 = 0) :=
by
  sorry

end no_common_real_solution_l802_80228


namespace cookie_distribution_l802_80214

theorem cookie_distribution:
  ∀ (initial_boxes brother_cookies sister_cookies leftover after_siblings leftover_sonny : ℕ),
    initial_boxes = 45 →
    brother_cookies = 12 →
    sister_cookies = 9 →
    after_siblings = initial_boxes - brother_cookies - sister_cookies →
    leftover_sonny = 17 →
    leftover = after_siblings - leftover_sonny →
    leftover = 7 :=
by
  intros initial_boxes brother_cookies sister_cookies leftover after_siblings leftover_sonny
  intros h1 h2 h3 h4 h5 h6
  sorry

end cookie_distribution_l802_80214


namespace age_ratio_l802_80224

theorem age_ratio 
  (a b c : ℕ)
  (h1 : a = b + 2)
  (h2 : a + b + c = 32)
  (h3 : b = 12) :
  b = 2 * c :=
by
  sorry

end age_ratio_l802_80224


namespace train_length_is_300_l802_80298

-- Definitions based on the conditions
def trainCrossesPlatform (L V : ℝ) : Prop :=
  L + 400 = V * 42

def trainCrossesSignalPole (L V : ℝ) : Prop :=
  L = V * 18

-- The main theorem statement
theorem train_length_is_300 (L V : ℝ)
  (h1 : trainCrossesPlatform L V)
  (h2 : trainCrossesSignalPole L V) :
  L = 300 :=
by
  sorry

end train_length_is_300_l802_80298


namespace impossible_even_sum_l802_80245

theorem impossible_even_sum (n m : ℤ) (h : (n^2 + m^2) % 2 = 1) : (n + m) % 2 ≠ 0 :=
sorry

end impossible_even_sum_l802_80245


namespace digit_difference_l802_80212

theorem digit_difference (X Y : ℕ) (h1 : 10 * X + Y - (10 * Y + X) = 36) : X - Y = 4 := by
  sorry

end digit_difference_l802_80212


namespace range_of_angle_B_l802_80297

theorem range_of_angle_B {A B C : ℝ} (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a)
  (h_sinB : Real.sin B = Real.sqrt (Real.sin A * Real.sin C)) :
  0 < B ∧ B ≤ Real.pi / 3 :=
sorry

end range_of_angle_B_l802_80297

import Mathlib

namespace product_of_roots_l246_246248

theorem product_of_roots (x : ℝ) (h : (x - 1) * (x + 4) = 22) : ∃ a b, (x^2 + 3*x - 26 = 0) ∧ a * b = -26 :=
by
  -- Given the equation (x - 1) * (x + 4) = 22,
  -- We want to show that the roots of the equation when simplified are such that
  -- their product is -26.
  sorry

end product_of_roots_l246_246248


namespace problem_inequality_l246_246854

theorem problem_inequality (x y z : ℝ) (hx : 1 ≤ x) (hy : 1 ≤ y) (hz : 1 ≤ z) :
  (x^2 - 2*x + 2) * (y^2 - 2*y + 2) * (z^2 - 2*z + 2) ≤ (x*y*z)^2 - 2*(x*y*z) + 2 := sorry

end problem_inequality_l246_246854


namespace x_intercepts_count_l246_246530

def parabola_x (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 3

theorem x_intercepts_count : 
  (∃ y : ℝ, parabola_x y = 0) → 1 := sorry

end x_intercepts_count_l246_246530


namespace problem1_problem2_l246_246825

-- Problem I
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 12 = 0}
def B : Set ℝ := {-2, 4}

theorem problem1 (a : ℝ) (h : A a = B) : a = 2 :=
sorry

-- Problem II
def C (m : ℝ) : Set ℝ := {x | m * x + 1 = 0}
def B' : Set ℝ := {-2, 4}

theorem problem2 (m : ℝ) (h : B' ∪ C m = B') : 
  m = -1/2 ∨ m = -1/4 ∨ m = 0 :=
sorry

end problem1_problem2_l246_246825


namespace number_solution_l246_246020

theorem number_solution : ∃ x : ℝ, x + 9 = x^2 ∧ x = (1 + Real.sqrt 37) / 2 :=
by
  use (1 + Real.sqrt 37) / 2
  simp
  sorry

end number_solution_l246_246020


namespace pages_left_to_read_correct_l246_246603

def total_pages : Nat := 563
def pages_read : Nat := 147
def pages_left_to_read : Nat := 416

theorem pages_left_to_read_correct : total_pages - pages_read = pages_left_to_read := by
  sorry

end pages_left_to_read_correct_l246_246603


namespace Namjoon_gave_Yoongi_9_pencils_l246_246696

theorem Namjoon_gave_Yoongi_9_pencils
  (stroke_pencils : ℕ)
  (strokes : ℕ)
  (pencils_left : ℕ)
  (total_pencils : ℕ := stroke_pencils * strokes)
  (given_pencils : ℕ := total_pencils - pencils_left) :
  stroke_pencils = 12 →
  strokes = 2 →
  pencils_left = 15 →
  given_pencils = 9 := by
  sorry

end Namjoon_gave_Yoongi_9_pencils_l246_246696


namespace natural_numbers_not_divisible_by_5_or_7_l246_246827

def num_not_divisible_by_5_or_7 (n : ℕ) : ℕ :=
  let num_div_5 := n / 5
  let num_div_7 := n / 7
  let num_div_35 := n / 35
  n - (num_div_5 + num_div_7 - num_div_35)

theorem natural_numbers_not_divisible_by_5_or_7 :
  num_not_divisible_by_5_or_7 999 = 686 :=
by sorry

end natural_numbers_not_divisible_by_5_or_7_l246_246827


namespace temperature_difference_l246_246258

theorem temperature_difference :
  let T_midnight := -4
  let T_10am := 5
  T_10am - T_midnight = 9 :=
by
  let T_midnight := -4
  let T_10am := 5
  show T_10am - T_midnight = 9
  sorry

end temperature_difference_l246_246258


namespace triangle_sim_APQ_ABC_perpendicular_if_EO_perp_PQ_then_QO_perp_PE_l246_246980

-- Definitions and conditions in Lean 4

variables {α : Type*} [EuclideanGeometry α] {A B C D E O P Q : α}

-- Acute triangle ABC
axiom acute_triangle_ABC : triangle α A B C ∧ triangle.is_acute A B C

-- Points D and E are on side BC
axiom points_DE_on_BC : same_line A B C D ∧ same_line A B C E

-- O, P, Q are the circumcenters of triangles ABC, ABD, and ADC respectively
axiom circumcenters_OPQ : is_circumcenter α A B C O ∧ is_circumcenter α A B D P ∧ is_circumcenter α A D C Q

-- Proof problem (1)
theorem triangle_sim_APQ_ABC :
  similar_triangles A P Q A B C := sorry

-- Proof problem (2)
theorem perpendicular_if_EO_perp_PQ_then_QO_perp_PE
  (h1 : perpendicular EO PQ) :
  perpendicular QO PE := sorry

end triangle_sim_APQ_ABC_perpendicular_if_EO_perp_PQ_then_QO_perp_PE_l246_246980


namespace evaluate_expression_l246_246990

theorem evaluate_expression :
  (↑(2 ^ (6 / 4))) ^ 8 = 4096 :=
by sorry

end evaluate_expression_l246_246990


namespace geometric_series_first_term_l246_246046

theorem geometric_series_first_term (r : ℚ) (S : ℚ) (a : ℚ) (h_r : r = 1 / 4) (h_S : S = 80) (h_sum : S = a / (1 - r)) : a = 60 :=
by 
  sorry

end geometric_series_first_term_l246_246046


namespace g_is_odd_l246_246407

noncomputable def g (x : ℝ) : ℝ := (1 / (3^x - 1)) + (1 / 3)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x :=
by
  intro x
  sorry

end g_is_odd_l246_246407


namespace range_of_squared_function_l246_246712

theorem range_of_squared_function (x : ℝ) (hx : 2 * x - 1 ≥ 0) : x ≥ 1 / 2 :=
sorry

end range_of_squared_function_l246_246712


namespace total_students_in_faculty_l246_246304

theorem total_students_in_faculty (N A B : ℕ) (hN : N = 230) (hA : A = 423) (hB : B = 134)
  (h80_percent : (N + A - B) = 80 / 100 * T) : T = 649 := 
by
  sorry

end total_students_in_faculty_l246_246304


namespace greene_family_total_spent_l246_246140

def adm_cost : ℕ := 45

def food_cost : ℕ := adm_cost - 13

def total_cost : ℕ := adm_cost + food_cost

theorem greene_family_total_spent : total_cost = 77 := 
by 
  sorry

end greene_family_total_spent_l246_246140


namespace possible_faulty_keys_l246_246957

theorem possible_faulty_keys (d : ℕ) (digits : list ℕ) (len_d : digits.length = 10) 
  (registered : list ℕ) (len_r : registered.length = 7) :
  (∃ d ∈ digits, d = 7 ∨ d = 9) :=
by
  sorry

end possible_faulty_keys_l246_246957


namespace quotient_korean_english_l246_246884

theorem quotient_korean_english (K M E : ℝ) (h1 : K / M = 1.2) (h2 : M / E = 5 / 6) : K / E = 1 :=
sorry

end quotient_korean_english_l246_246884


namespace popsicle_melting_ratio_l246_246029

theorem popsicle_melting_ratio (S : ℝ) (r : ℝ) (h : r^5 = 32) : r = 2 :=
by
  sorry

end popsicle_melting_ratio_l246_246029


namespace kolya_or_leva_wins_l246_246551

-- Definitions for segment lengths
variables (k l : ℝ)

-- Definition of the condition when Kolya wins
def kolya_wins (k l : ℝ) : Prop :=
  k > l

-- Definition of the condition when Leva wins
def leva_wins (k l : ℝ) : Prop :=
  k ≤ l

-- Theorem statement for the proof problem
theorem kolya_or_leva_wins (k l : ℝ) : kolya_wins k l ∨ leva_wins k l :=
sorry

end kolya_or_leva_wins_l246_246551


namespace erasers_pens_markers_cost_l246_246042

theorem erasers_pens_markers_cost 
  (E P M : ℝ)
  (h₁ : E + 3 * P + 2 * M = 240)
  (h₂ : 2 * E + 4 * M + 5 * P = 440) :
  3 * E + 4 * P + 6 * M = 520 :=
sorry

end erasers_pens_markers_cost_l246_246042


namespace pascal_row_with_ratio_456_exists_at_98_l246_246106

theorem pascal_row_with_ratio_456_exists_at_98 :
  ∃ n, ∃ r, 0 ≤ r ∧ r + 2 ≤ n ∧ 
  ((Nat.choose n r : ℚ) / Nat.choose n (r + 1) = 4 / 5) ∧
  ((Nat.choose n (r + 1) : ℚ) / Nat.choose n (r + 2) = 5 / 6) ∧ 
  n = 98 := by
  sorry

end pascal_row_with_ratio_456_exists_at_98_l246_246106


namespace sum_not_zero_l246_246148

theorem sum_not_zero (a b c d : ℝ) (h1 : a * b * c - d = 1) (h2 : b * c * d - a = 2) 
  (h3 : c * d * a - b = 3) (h4 : d * a * b - c = -6) : a + b + c + d ≠ 0 :=
sorry

end sum_not_zero_l246_246148


namespace coprime_exists_pow_divisible_l246_246897

theorem coprime_exists_pow_divisible (a n : ℕ) (h_coprime : Nat.gcd a n = 1) : 
  ∃ m : ℕ, n ∣ a^m - 1 :=
by
  sorry

end coprime_exists_pow_divisible_l246_246897


namespace jenna_practice_minutes_l246_246852

theorem jenna_practice_minutes :
  ∀ (practice_6_days practice_2_days target_total target_average: ℕ),
    practice_6_days = 6 * 80 →
    practice_2_days = 2 * 105 →
    target_average = 100 →
    target_total = 9 * target_average →
  ∃ practice_9th_day, (practice_6_days + practice_2_days + practice_9th_day = target_total) ∧ practice_9th_day = 210 :=
by sorry

end jenna_practice_minutes_l246_246852


namespace popsicle_stick_count_l246_246698

variable (Sam Sid Steve : ℕ)

def number_of_sticks (Sam Sid Steve : ℕ) : ℕ :=
  Sam + Sid + Steve

theorem popsicle_stick_count 
  (h1 : Sam = 3 * Sid)
  (h2 : Sid = 2 * Steve)
  (h3 : Steve = 12) :
  number_of_sticks Sam Sid Steve = 108 :=
by
  sorry

end popsicle_stick_count_l246_246698


namespace smallest_int_with_18_divisors_l246_246750

theorem smallest_int_with_18_divisors : ∃ n : ℕ, (∀ d : ℕ, 0 < d ∧ d ≤ n → d = 288) ∧ (∃ a1 a2 a3 : ℕ, a1 + 1 * a2 + 1 * a3 + 1 = 18) := 
by 
  sorry

end smallest_int_with_18_divisors_l246_246750


namespace find_faulty_keys_l246_246955

-- Define the conditions given in the problem
def total_digits : ℕ := 10
def registered_digits : ℕ := 7
def missing_digits : ℕ := 3
def defective_key_min_presses : ℕ := 5
def defective_key_successful_presses : ℕ := 2

-- Define that we need to find which keys could be faulty
def possible_faulty_keys : List ℤ := [7, 9]

-- The main theorem statement
theorem find_faulty_keys (total_digits = 10) (registered_digits = 7) (missing_digits = 3)
  (defective_key_min_presses = 5) (defective_key_successful_presses ≥ 2) :
  possible_faulty_keys = [7, 9] :=
by
  sorry

end find_faulty_keys_l246_246955


namespace total_ingredients_used_l246_246713

theorem total_ingredients_used (water oliveOil salt : ℕ) 
  (h_ratio : water / oliveOil = 3 / 2) 
  (h_salt : water / salt = 3 / 1)
  (h_water_cups : water = 15) : 
  water + oliveOil + salt = 30 :=
sorry

end total_ingredients_used_l246_246713


namespace total_cost_food_l246_246115

theorem total_cost_food
  (beef_pounds : ℕ)
  (beef_cost_per_pound : ℕ)
  (chicken_pounds : ℕ)
  (chicken_cost_per_pound : ℕ)
  (h_beef : beef_pounds = 1000)
  (h_beef_cost : beef_cost_per_pound = 8)
  (h_chicken : chicken_pounds = 2 * beef_pounds)
  (h_chicken_cost : chicken_cost_per_pound = 3) :
  (beef_pounds * beef_cost_per_pound + chicken_pounds * chicken_cost_per_pound = 14000) :=
by
  sorry

end total_cost_food_l246_246115


namespace find_solutions_l246_246208

noncomputable def cuberoot (x : ℝ) : ℝ := x^(1/3)

theorem find_solutions :
    {x : ℝ | cuberoot x = 15 / (8 - cuberoot x)} = {125, 27} :=
by
  sorry

end find_solutions_l246_246208


namespace line_equation_l246_246706

theorem line_equation (A : (ℝ × ℝ)) (hA_x : A.1 = 2) (hA_y : A.2 = 0)
  (h_intercept : ∀ B : (ℝ × ℝ), B.1 = 0 → 2 * B.1 + B.2 + 2 = 0 → B = (0, -2)) :
  ∃ (l : ℝ × ℝ → Prop), (l A ∧ l (0, -2)) ∧ 
    (∀ x y : ℝ, l (x, y) ↔ x - y - 2 = 0) :=
by
  sorry

end line_equation_l246_246706


namespace solve_system_eqns_l246_246137

theorem solve_system_eqns :
  ∀ x y z : ℝ, 
  (x * y + 5 * y * z - 6 * x * z = -2 * z) ∧
  (2 * x * y + 9 * y * z - 9 * x * z = -12 * z) ∧
  (y * z - 2 * x * z = 6 * z) →
  x = -2 ∧ y = 2 ∧ z = 1 / 6 ∨
  y = 0 ∧ z = 0 ∨
  x = 0 ∧ z = 0 :=
by
  sorry

end solve_system_eqns_l246_246137


namespace find_x_set_l246_246797

theorem find_x_set (x : ℝ) : ((x - 2) ^ 2 < 3 * x + 4) ↔ (0 ≤ x ∧ x < 7) := 
sorry

end find_x_set_l246_246797


namespace team_selection_ways_l246_246451

theorem team_selection_ways :
  let ways (n k : ℕ) := Nat.choose n k
  (ways 6 3) * (ways 6 3) = 400 := 
by
  let ways := Nat.choose
  -- Proof is omitted
  sorry

end team_selection_ways_l246_246451


namespace events_equally_likely_iff_N_eq_18_l246_246396

variable (N : ℕ)

-- Define the number of combinations in the draws
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the sums of selecting balls
noncomputable def S_63_10 (N: ℕ) : ℕ := sorry -- placeholder definition
noncomputable def S_44_8 (N: ℕ) : ℕ := sorry 

-- Condition for the events being equally likely
theorem events_equally_likely_iff_N_eq_18 : 
  (S_63_10 N * C N 8 = S_44_8 N * C N 10) ↔ N = 18 :=
sorry

end events_equally_likely_iff_N_eq_18_l246_246396


namespace max_radius_of_inscribable_circle_l246_246200

theorem max_radius_of_inscribable_circle
  (AB BC CD DA : ℝ) (x y z w : ℝ)
  (h1 : AB = 10) (h2 : BC = 12) (h3 : CD = 8) (h4 : DA = 14)
  (h5 : x + y = 10) (h6 : y + z = 12)
  (h7 : z + w = 8) (h8 : w + x = 14)
  (h9 : x + z = y + w) :
  ∃ r : ℝ, r = Real.sqrt 24.75 :=
by
  sorry

end max_radius_of_inscribable_circle_l246_246200


namespace triangle_area_l246_246768

variable (a b c : ℕ)
variable (s : ℕ := 21)
variable (area : ℕ := 84)

theorem triangle_area 
(h1 : c = a + b - 12) 
(h2 : (a + b + c) / 2 = s) 
(h3 : c - a = 2) 
: (21 * (21 - a) * (21 - b) * (21 - c)).sqrt = area := 
sorry

end triangle_area_l246_246768


namespace compute_expression_l246_246793

theorem compute_expression (x : ℤ) (h : x = 3) : (x^8 + 24 * x^4 + 144) / (x^4 + 12) = 93 :=
by
  rw [h]
  sorry

end compute_expression_l246_246793


namespace repeating_decimal_ratio_eq_4_l246_246725

-- Definitions for repeating decimals
def rep_dec_36 := 0.36 -- 0.\overline{36}
def rep_dec_09 := 0.09 -- 0.\overline{09}

-- Lean 4 statement of proof problem
theorem repeating_decimal_ratio_eq_4 :
  (rep_dec_36 / rep_dec_09) = 4 :=
sorry

end repeating_decimal_ratio_eq_4_l246_246725


namespace value_of_s_l246_246372

-- Conditions: (m - 8) is a factor of m^2 - sm - 24

theorem value_of_s (s : ℤ) (m : ℤ) (h : (m - 8) ∣ (m^2 - s*m - 24)) : s = 5 :=
by
  sorry

end value_of_s_l246_246372


namespace michael_left_money_l246_246125

def michael_initial_money : Nat := 100
def michael_spent_on_snacks : Nat := 25
def michael_spent_on_rides : Nat := 3 * michael_spent_on_snacks
def michael_spent_on_games : Nat := 15
def total_expenditure : Nat := michael_spent_on_snacks + michael_spent_on_rides + michael_spent_on_games
def michael_money_left : Nat := michael_initial_money - total_expenditure

theorem michael_left_money : michael_money_left = 15 := by
  sorry

end michael_left_money_l246_246125


namespace compare_abc_l246_246655

noncomputable def a : ℝ := - Real.logb 2 (1/5)
noncomputable def b : ℝ := Real.logb 8 27
noncomputable def c : ℝ := Real.exp (-3)

theorem compare_abc : a = Real.logb 2 5 ∧ 1 < b ∧ b < 2 ∧ c = Real.exp (-3) → a > b ∧ b > c :=
by
  sorry

end compare_abc_l246_246655


namespace green_or_yellow_probability_l246_246727

-- Given the number of marbles of each color
def green_marbles : ℕ := 4
def yellow_marbles : ℕ := 3
def white_marbles : ℕ := 6

-- The total number of marbles
def total_marbles : ℕ := green_marbles + yellow_marbles + white_marbles

-- The number of favorable outcomes (green or yellow marbles)
def favorable_marbles : ℕ := green_marbles + yellow_marbles

-- The probability of drawing a green or yellow marble as a fraction
def probability_of_green_or_yellow : Rat := favorable_marbles / total_marbles

theorem green_or_yellow_probability :
  probability_of_green_or_yellow = 7 / 13 :=
by
  sorry

end green_or_yellow_probability_l246_246727


namespace find_number_and_remainder_l246_246313

theorem find_number_and_remainder :
  ∃ (N r : ℕ), (3927 + 2873) * (3 * (3927 - 2873)) + r = N ∧ r < (3927 + 2873) :=
sorry

end find_number_and_remainder_l246_246313


namespace rectangle_width_is_4_l246_246146

-- Definitions of conditions
variable (w : ℝ) -- width of the rectangle
def length := w + 2 -- length of the rectangle
def perimeter := 2 * w + 2 * (w + 2) -- perimeter of the rectangle, using given conditions

-- The theorem to be proved
theorem rectangle_width_is_4 (h : perimeter = 20) : w = 4 :=
by {
  sorry -- To be proved
}

end rectangle_width_is_4_l246_246146


namespace bird_count_l246_246612

theorem bird_count (num_cages : ℕ) (parrots_per_cage : ℕ) (parakeets_per_cage : ℕ) 
  (total_birds : ℕ) (h1 : num_cages = 8) (h2 : parrots_per_cage = 2) (h3 : parakeets_per_cage = 7) 
  (h4 : total_birds = num_cages * (parrots_per_cage + parakeets_per_cage)) : 
  total_birds = 72 := 
  by
  sorry

end bird_count_l246_246612


namespace g_18_equals_324_l246_246413

def is_strictly_increasing (g : ℕ → ℕ) :=
  ∀ n : ℕ, n > 0 → g (n + 1) > g n

def multiplicative (g : ℕ → ℕ) :=
  ∀ m n : ℕ, m > 0 → n > 0 → g (m * n) = g m * g n

def m_n_condition (g : ℕ → ℕ) :=
  ∀ m n : ℕ, m ≠ n → m > 0 → n > 0 → m ^ n = n ^ m → (g m = n ∨ g n = m)

noncomputable def g : ℕ → ℕ := sorry

theorem g_18_equals_324 :
  is_strictly_increasing g →
  multiplicative g →
  m_n_condition g →
  g 18 = 324 :=
sorry

end g_18_equals_324_l246_246413


namespace find_square_sum_l246_246828

theorem find_square_sum (x y : ℝ) (h1 : 2 * x * (x + y) = 54) (h2 : 3 * y * (x + y) = 81) : (x + y) ^ 2 = 135 :=
sorry

end find_square_sum_l246_246828


namespace graduation_messages_total_l246_246608

/-- Define the number of students in the class -/
def num_students : ℕ := 40

/-- Define the combination formula C(n, 2) for choosing 2 out of n -/
def combination (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Prove that the total number of graduation messages written is 1560 -/
theorem graduation_messages_total : combination num_students = 1560 :=
by
  sorry

end graduation_messages_total_l246_246608


namespace regression_prediction_l246_246071

-- Define the linear regression model as a function
def linear_regression (x : ℝ) : ℝ :=
  7.19 * x + 73.93

-- State that using this model, the predicted height at age 10 is approximately 145.83
theorem regression_prediction :
  abs (linear_regression 10 - 145.83) < 0.01 :=
by 
  sorry

end regression_prediction_l246_246071


namespace length_of_AE_l246_246382

noncomputable def AE_calculation (AB AC AD : ℝ) (h : ℝ) (AE : ℝ) : Prop :=
  AB = 3.6 ∧ AC = 3.6 ∧ AD = 1.2 ∧ 
  (0.5 * AC * h = 0.5 * AE * (1/3) * h) →
  AE = 10.8

theorem length_of_AE {h : ℝ} : AE_calculation 3.6 3.6 1.2 h 10.8 :=
sorry

end length_of_AE_l246_246382


namespace sum_seq_formula_l246_246223

open Nat

def seq (n : ℕ) : ℕ :=
  Nat.recOn n 2 (λ _ a_n, 4 * a_n - 3 * n + 1)

noncomputable def sum_seq (n : ℕ) : ℕ :=
  (range n).sum (seq ∘ (λ i, i + 1))

theorem sum_seq_formula (n : ℕ) : 
  sum_seq n = (range n).sum (λ i, binomial n i * 3^(n - i) + i + 1) := 
by
  sorry

end sum_seq_formula_l246_246223


namespace find_natural_number_l246_246754

variable {A : ℕ}

theorem find_natural_number (h1 : A = 8 * 2 + 7) : A = 23 :=
sorry

end find_natural_number_l246_246754


namespace lucky_sum_equal_prob_l246_246394

theorem lucky_sum_equal_prob (N : ℕ) (S63_main_draw : Finset ℕ) (S44_additional_draw : Finset ℕ)
  (hN_pos : 0 < N)
  (hS63 : S63_main_draw.card = 10)
  (hSum63 : S63_main_draw.sum id = 63)
  (hS44 : S44_additional_draw.card = 8)
  (hSum44 : S44_additional_draw.sum id = 44)
  (hS_distinct : ∀ (n ∈ S63_main_draw ∪ S44_additional_draw), n ∈ Finset.range (N + 1)) :
  N = 18 :=
by
  sorry

end lucky_sum_equal_prob_l246_246394


namespace female_members_count_l246_246387

theorem female_members_count (M F : ℕ) (h1 : F = 2 * M) (h2 : F + M = 18) : F = 12 :=
by
  -- the proof will go here
  sorry

end female_members_count_l246_246387


namespace quadrilateral_EFGH_area_l246_246874

-- Definitions based on conditions
def quadrilateral_EFGH_right_angles (F H : ℝ) : Prop :=
  ∃ E G, E - F = 0 ∧ H - G = 0

def quadrilateral_length_hypotenuse (E G : ℝ) : Prop :=
  E - G = 5

def distinct_integer_lengths (EF FG EH HG : ℝ) : Prop :=
  EF ≠ FG ∧ EH ≠ HG ∧ ∃ a b : ℕ, EF = a ∧ FG = b ∧ EH = b ∧ HG = a ∧ a * a + b * b = 25

-- Proof statement
theorem quadrilateral_EFGH_area (F H : ℝ) 
  (EF FG EH HG E G : ℝ) 
  (h1 : quadrilateral_EFGH_right_angles F H) 
  (h2 : quadrilateral_length_hypotenuse E G)
  (h3 : distinct_integer_lengths EF FG EH HG) 
: 
  EF * FG / 2 + EH * HG / 2 = 12 := 
sorry

end quadrilateral_EFGH_area_l246_246874


namespace train_crossing_time_l246_246483

open Real

noncomputable def time_to_cross_bridge 
  (length_train : ℝ) (speed_train_kmh : ℝ) (length_bridge : ℝ) : ℝ :=
  let total_distance := length_train + length_bridge
  let speed_train_ms := speed_train_kmh * (1000/3600)
  total_distance / speed_train_ms

theorem train_crossing_time
  (length_train : ℝ) (speed_train_kmh : ℝ) (length_bridge : ℝ)
  (h_length_train : length_train = 160)
  (h_speed_train_kmh : speed_train_kmh = 45)
  (h_length_bridge : length_bridge = 215) :
  time_to_cross_bridge length_train speed_train_kmh length_bridge = 30 :=
sorry

end train_crossing_time_l246_246483


namespace opposite_of_neg5_is_pos5_l246_246910

theorem opposite_of_neg5_is_pos5 : -(-5) = 5 := 
by
  sorry

end opposite_of_neg5_is_pos5_l246_246910


namespace opposite_of_neg_five_l246_246925

theorem opposite_of_neg_five : -(-5) = 5 := by
  sorry

end opposite_of_neg_five_l246_246925


namespace x1_x2_gt_one_l246_246659

noncomputable def f (x : ℝ) : ℝ := log x - x

def g (m x : ℝ) : ℝ := log x + 1/(2*x) - m

theorem x1_x2_gt_one {m x1 x2 : ℝ} (hx1 : 0 < x1) (hx2 : 0 < x2) (h_x1_lt_x2 : x1 < x2)
  (hx1_zero : g m x1 = 0) (hx2_zero : g m x2 = 0) : x1 + x2 > 1 := by
  sorry

end x1_x2_gt_one_l246_246659


namespace cicely_100th_birthday_l246_246625

-- Definition of the conditions
def birth_year (birthday_year : ℕ) (birthday_age : ℕ) : ℕ :=
  birthday_year - birthday_age

def birthday (birth_year : ℕ) (age : ℕ) : ℕ :=
  birth_year + age

-- The problem restatement in Lean 4
theorem cicely_100th_birthday (birthday_year : ℕ) (birthday_age : ℕ) (expected_year : ℕ) :
  birthday_year = 1939 → birthday_age = 21 → expected_year = 2018 → birthday (birth_year birthday_year birthday_age) 100 = expected_year :=
by
  intros h1 h2 h3
  rw [birthday, birth_year]
  rw [h1, h2]
  sorry

end cicely_100th_birthday_l246_246625


namespace part1_l246_246512

theorem part1 (x : ℝ) (hx : x > 0) : 
  (1 / (2 * Real.sqrt (x + 1))) < (Real.sqrt (x + 1) - Real.sqrt x) ∧ (Real.sqrt (x + 1) - Real.sqrt x) < (1 / (2 * Real.sqrt x)) := 
sorry

end part1_l246_246512


namespace repair_cost_is_5000_l246_246281

-- Define the initial cost of the machine
def initial_cost : ℝ := 9000

-- Define the transportation charges
def transportation_charges : ℝ := 1000

-- Define the selling price
def selling_price : ℝ := 22500

-- Define the profit percentage as a decimal
def profit_percentage : ℝ := 0.5

-- Define the total cost including repairs
def total_cost (repair_cost : ℝ) : ℝ :=
  initial_cost + transportation_charges + repair_cost

-- Define the equation for selling price with 50% profit
def selling_price_equation (repair_cost : ℝ) : Prop :=
  selling_price = (1 + profit_percentage) * total_cost repair_cost

-- State the proof problem in Lean
theorem repair_cost_is_5000 : selling_price_equation 5000 :=
by 
  sorry

end repair_cost_is_5000_l246_246281


namespace necessary_but_not_sufficient_condition_l246_246819

theorem necessary_but_not_sufficient_condition (x : ℝ) (h : x > 0) : 
  ((x > 2 ∧ x < 4) ↔ (2 < x ∧ x < 4)) :=
by {
    sorry
}

end necessary_but_not_sufficient_condition_l246_246819


namespace primes_with_prime_remainders_count_l246_246367

-- Define a predicate to check for prime numbers
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Define the set of prime remainders when divided by 12
def prime_remainders := {1, 2, 3, 5, 7, 11}

-- Function to list primes between 50 and 100
def primes_between_50_and_100 := [53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

-- Function to count such primes with prime remainder when divided by 12
noncomputable def count_primes_with_prime_remainder : ℕ :=
  list.count (λ n, n % 12 ∈ prime_remainders) primes_between_50_and_100

-- The theorem to state the problem in Lean
theorem primes_with_prime_remainders_count : count_primes_with_prime_remainder = 10 :=
by {
  /- proof steps to be provided here, if required. -/
 sorry
}

end primes_with_prime_remainders_count_l246_246367


namespace c_less_than_a_l246_246216

variable (a b c : ℝ)

-- Conditions definitions
def are_negative : Prop := a < 0 ∧ b < 0 ∧ c < 0
def eq1 : Prop := c = 2 * (a + b)
def eq2 : Prop := c = 3 * (b - a)

-- Theorem statement
theorem c_less_than_a (h_neg : are_negative a b c) (h_eq1 : eq1 a b c) (h_eq2 : eq2 a b c) : c < a :=
  sorry

end c_less_than_a_l246_246216


namespace expansion_three_times_expansion_six_times_l246_246242

-- Definition for the rule of expansion
def expand (a b : Nat) : Nat := a * b + a + b

-- Problem 1: Expansion with a = 1, b = 3 for 3 times results in 255.
theorem expansion_three_times : expand (expand (expand 1 3) 7) 31 = 255 := sorry

-- Problem 2: After 6 operations, the expanded number matches the given pattern.
theorem expansion_six_times (p q : ℕ) (hp : p > q) (hq : q > 0) : 
  ∃ m n, m = 8 ∧ n = 13 ∧ (expand (expand (expand (expand (expand (expand q (expand p q)) (expand p q)) (expand p q)) (expand p q)) (expand p q)) (expand p q)) = (q + 1) ^ m * (p + 1) ^ n - 1 :=
sorry

end expansion_three_times_expansion_six_times_l246_246242


namespace solve_for_x_l246_246970

theorem solve_for_x :
  ∀ x : ℤ, (35 - (23 - (15 - x)) = (12 * 2) / 1 / 2) → x = -21 :=
by
  intro x
  sorry

end solve_for_x_l246_246970


namespace min_value_m_l246_246299

open Finset

-- Define the set S and the number of colors
def S : Finset ℕ := (finset.range 61).map (λ x : ℕ, x + 1)

-- Let m be the number of non-empty subsets of S
def m (coloring : Π (x : ℕ), x ∈ S → ℕ) : ℕ :=
  (∑ i in range 25, (2 ^ ((S.filter (λ x, coloring x (mem_map_of_mem _ (mem_range.mpr (lt_trans (self_lt_succ _) bot_lt_self))))) : ℕ).card - 1))

-- Prove that the minimum value of m is 119
theorem min_value_m : ∃ (coloring : Π (x : ℕ), x ∈ S → ℕ), m coloring = 119 :=
by {
  -- sorry to indicate skipping the proof steps
  sorry
}

end min_value_m_l246_246299


namespace opposite_of_negative_five_l246_246907

theorem opposite_of_negative_five : ∀ x : Int, -5 + x = 0 → x = 5 :=
by
  intros x h
  sorry

end opposite_of_negative_five_l246_246907


namespace find_k_l246_246238

-- Define the problem parameters
variables {x y k : ℝ}

-- The conditions given in the problem
def system_of_equations (x y k : ℝ) : Prop :=
  (x + 2 * y = k - 1) ∧ (2 * x + y = 5 * k + 4)

def solution_condition (x y : ℝ) : Prop :=
  x + y = 5

-- The proof statement
theorem find_k (x y k : ℝ) (h1 : system_of_equations x y k) (h2 : solution_condition x y) :
  k = 2 :=
sorry

end find_k_l246_246238


namespace bricks_in_wall_l246_246188

theorem bricks_in_wall (h : ℕ) 
  (brenda_rate : ℕ := h / 8)
  (brandon_rate : ℕ := h / 12)
  (combined_rate : ℕ := (5 * h) / 24)
  (decreased_combined_rate : ℕ := combined_rate - 15)
  (work_time : ℕ := 6) :
  work_time * decreased_combined_rate = h → h = 360 := by
  intros h_eq
  sorry

end bricks_in_wall_l246_246188


namespace nonnegative_difference_of_roots_l246_246014

theorem nonnegative_difference_of_roots :
  ∀ (x : ℝ), x^2 + 40 * x + 300 = -50 → (∃ a b : ℝ, x^2 + 40 * x + 350 = 0 ∧ x = a ∧ x = b ∧ |a - b| = 25) := 
by 
sorry

end nonnegative_difference_of_roots_l246_246014


namespace determine_n_l246_246771

noncomputable def average_value (n : ℕ) : ℚ :=
  (n * (n + 1) * (2 * n + 1) : ℚ) / (6 * (n * (n + 1) / 2))

theorem determine_n :
  ∃ n : ℕ, average_value n = 2020 ∧ n = 3029 :=
sorry

end determine_n_l246_246771


namespace initial_pigeons_l246_246296

theorem initial_pigeons (n : ℕ) (h : n + 1 = 2) : n = 1 := 
sorry

end initial_pigeons_l246_246296


namespace apples_count_l246_246561

def mangoes_oranges_apples_ratio (mangoes oranges apples : Nat) : Prop :=
  mangoes / 10 = oranges / 2 ∧ mangoes / 10 = apples / 3

theorem apples_count (mangoes oranges apples : Nat) (h_ratio : mangoes_oranges_apples_ratio mangoes oranges apples) (h_mangoes : mangoes = 120) : apples = 36 :=
by
  sorry

end apples_count_l246_246561


namespace range_of_a_l246_246662

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, a * x^2 + 2 * a * x - 4 < 0) : -4 < a ∧ a ≤ 0 := 
sorry

end range_of_a_l246_246662


namespace horizontal_asymptote_of_rational_function_l246_246364

theorem horizontal_asymptote_of_rational_function :
  ∀ (x : ℝ), (y = (7 * x^2 - 5) / (4 * x^2 + 6 * x + 3)) → (∃ b : ℝ, b = 7 / 4) :=
by
  intro x y
  sorry

end horizontal_asymptote_of_rational_function_l246_246364


namespace least_number_subtraction_l246_246954

theorem least_number_subtraction (n : ℕ) (h₀ : n = 3830) (k : ℕ) (h₁ : k = 5) : (n - k) % 15 = 0 :=
by {
  sorry
}

end least_number_subtraction_l246_246954


namespace range_of_M_l246_246218

theorem range_of_M (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a + b + c = 1) :
    ( (1 / a - 1) * (1 / b - 1) * (1 / c - 1) )  ≥ 8 := 
  sorry

end range_of_M_l246_246218


namespace largest_five_digit_number_tens_place_l246_246455

theorem largest_five_digit_number_tens_place :
  ∀ (n : ℕ), n = 87315 → (n % 100) / 10 = 1 := 
by
  intros n h
  sorry

end largest_five_digit_number_tens_place_l246_246455


namespace bitcoin_donation_l246_246410

theorem bitcoin_donation (x : ℝ) (h : 3 * (80 - x) / 2 - 10 = 80) : x = 20 :=
sorry

end bitcoin_donation_l246_246410


namespace last_non_zero_digit_of_40_l246_246211

def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def last_non_zero_digit (n : ℕ) : ℕ :=
  let p := factorial n
  let digits : List ℕ := List.filter (λ d => d ≠ 0) (p.digits 10)
  digits.headD 0

theorem last_non_zero_digit_of_40 : last_non_zero_digit 40 = 6 := by
  sorry

end last_non_zero_digit_of_40_l246_246211


namespace original_number_is_15_l246_246264

theorem original_number_is_15 (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) (N : ℕ) (h4 : 100 * a + 10 * b + c = m)
  (h5 : 100 * a +  10 * b +   c +
        100 * a +   c + 10 * b + 
        100 * b +  10 * a +   c +
        100 * b +   c + 10 * a + 
        100 * c +  10 * a +   b +
        100 * c +   b + 10 * a = 3315) :
  m = 15 :=
sorry

end original_number_is_15_l246_246264


namespace solve_equation_l246_246285

noncomputable def f (x : ℝ) := (1 / (x^2 + 17 * x + 20)) + (1 / (x^2 + 12 * x + 20)) + (1 / (x^2 - 15 * x + 20))

theorem solve_equation :
  {x : ℝ | f x = 0} = {-1, -4, -5, -20} :=
by
  sorry

end solve_equation_l246_246285


namespace reservoir_full_percentage_after_storm_l246_246615

theorem reservoir_full_percentage_after_storm 
  (original_contents water_added : ℤ) 
  (percentage_full_before_storm: ℚ) 
  (total_capacity new_contents : ℚ) 
  (H1 : original_contents = 220 * 10^9) 
  (H2 : water_added = 110 * 10^9) 
  (H3 : percentage_full_before_storm = 0.40)
  (H4 : total_capacity = original_contents / percentage_full_before_storm)
  (H5 : new_contents = original_contents + water_added) :
  (new_contents / total_capacity) = 0.60 := 
by 
  sorry

end reservoir_full_percentage_after_storm_l246_246615


namespace positive_difference_between_prob_3_and_prob_5_l246_246591

/-- Probability of a coin landing heads up exactly 3 times out of 5 flips -/
def prob_3_heads : ℚ := (nat.choose 5 3) * (1/2)^3 * (1/2)^(5-3)

/-- Probability of a coin landing heads up exactly 5 times out of 5 flips -/
def prob_5_heads : ℚ := (1/2)^5

/-- Positive difference between the probabilities -/
theorem positive_difference_between_prob_3_and_prob_5 : 
  |prob_3_heads - prob_5_heads| = 9 / 32 :=
by sorry

end positive_difference_between_prob_3_and_prob_5_l246_246591


namespace smallest_next_divisor_l246_246683

def isOddFourDigitNumber (n : ℕ) : Prop :=
  n % 2 = 1 ∧ 1000 ≤ n ∧ n < 10000

noncomputable def divisors (n : ℕ) : List ℕ :=
  (List.range (n + 1)).filter (λ d => d > 0 ∧ n % d = 0)

theorem smallest_next_divisor (m : ℕ) (h₁ : isOddFourDigitNumber m) (h₂ : 437 ∈ divisors m) :
  ∃ k, k > 437 ∧ k ∈ divisors m ∧ k % 2 = 1 ∧ ∀ n, n > 437 ∧ n < k → n ∉ divisors m := by
  sorry

end smallest_next_divisor_l246_246683


namespace hyperbola_eccentricity_is_2_l246_246661

noncomputable def hyperbola_eccentricity (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0)
  (H1 : b^2 = c^2 - a^2)
  (H2 : 3 * c^2 = 4 * b^2) : ℝ :=
c / a

theorem hyperbola_eccentricity_is_2 (a b c : ℝ)
  (h : a > 0 ∧ b > 0 ∧ c > 0)
  (H1 : b^2 = c^2 - a^2)
  (H2 : 3 * c^2 = 4 * b^2) :
  hyperbola_eccentricity a b c h H1 H2 = 2 :=
sorry

end hyperbola_eccentricity_is_2_l246_246661


namespace smallest_integer_with_18_divisors_l246_246752

theorem smallest_integer_with_18_divisors :
  ∃ n : ℕ, (∀ m : ℕ, m > 0 → (∀ d, d ∣ n → d ∣ m) → n = 288) ∧ (nat.divisors_count 288 = 18) := by
  sorry

end smallest_integer_with_18_divisors_l246_246752


namespace solve_inequality_find_m_range_l246_246824

noncomputable def f (x : ℝ) : ℝ := |x - 2|
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := -|x + 3| + m

theorem solve_inequality (a : ℝ) : 
  ∀ x : ℝ, f x + a - 1 > 0 ↔ 
    (a = 1 ∧ x ≠ 2) ∨ 
    (a > 1) ∨ 
    (a < 1 ∧ (x > 3 - a ∨ x < a + 1)) :=
sorry

theorem find_m_range (m : ℝ) : 
  (∀ x : ℝ, f x > g x m) ↔ m < 5 :=
sorry

end solve_inequality_find_m_range_l246_246824


namespace find_constants_l246_246691

open Nat

variables {n : ℕ} (b c : ℤ)
def S (n : ℕ) := n^2 + b * n + c
def a (n : ℕ) := S n - S (n - 1)

theorem find_constants (a2a3_sum_eq_4 : a 2 + a 3 = 4) : 
  c = 0 ∧ b = -2 := 
by 
  sorry

end find_constants_l246_246691


namespace problem_statement_l246_246086

open Set

noncomputable def A : Set ℤ := {-2, -1, 0, 1, 2}

theorem problem_statement :
  {y : ℤ | ∃ x ∈ A, y = |x + 1|} = {0, 1, 2, 3} :=
by
  sorry

end problem_statement_l246_246086


namespace evaluate_polynomial_at_3_l246_246723

def f (x : ℕ) : ℕ := 3 * x ^ 3 + x - 3

theorem evaluate_polynomial_at_3 : f 3 = 28 :=
by
  sorry

end evaluate_polynomial_at_3_l246_246723


namespace minimum_distinct_lines_l246_246389

theorem minimum_distinct_lines (n : ℕ) (h : n = 31) : 
  ∃ (k : ℕ), k = 9 :=
by
  sorry

end minimum_distinct_lines_l246_246389


namespace opposite_of_negative_five_l246_246934

theorem opposite_of_negative_five : ∃ x : ℤ, -5 + x = 0 ∧ x = 5 :=
by
  use 5
  split
  {
    sorry,
  }
  {
    refl,
  }

end opposite_of_negative_five_l246_246934


namespace min_distance_feasible_region_line_l246_246231

def point (x y : ℝ) : Type := ℝ × ℝ 

theorem min_distance_feasible_region_line :
  ∃ (M N : ℝ × ℝ),
    (2 * M.1 + M.2 - 4 >= 0) ∧
    (M.1 - M.2 - 2 <= 0) ∧
    (M.2 - 3 <= 0) ∧
    (N.2 = -2 * N.1 + 2) ∧
    (dist M N = (2 * Real.sqrt 5)/5) :=
by 
  sorry

end min_distance_feasible_region_line_l246_246231


namespace point_A_in_fourth_quadrant_l246_246845

def Point := ℤ × ℤ

def is_in_fourth_quadrant (p : Point) : Prop :=
  p.1 > 0 ∧ p.2 < 0

def point_A : Point := (3, -2)
def point_B : Point := (2, 5)
def point_C : Point := (-1, -2)
def point_D : Point := (-2, 2)

theorem point_A_in_fourth_quadrant : is_in_fourth_quadrant point_A :=
  sorry

end point_A_in_fourth_quadrant_l246_246845


namespace maximize_volume_l246_246155

-- Define the problem-specific constants
def bar_length : ℝ := 0.18
def length_to_width_ratio : ℝ := 2

-- Function to define volume of the rectangle frame
def volume (length width height : ℝ) : ℝ := length * width * height

theorem maximize_volume :
  ∃ (length width height : ℝ), 
  (length / width = length_to_width_ratio) ∧ 
  (2 * (length + width) = bar_length) ∧ 
  ((length = 2) ∧ (height = 1.5)) :=
sorry

end maximize_volume_l246_246155


namespace only_prime_such_that_2p_plus_one_is_perfect_power_l246_246506

theorem only_prime_such_that_2p_plus_one_is_perfect_power :
  ∃ (p : ℕ), p ≤ 1000 ∧ Prime p ∧ ∃ (m n : ℕ), n ≥ 2 ∧ 2 * p + 1 = m^n ∧ p = 13 :=
by
  sorry

end only_prime_such_that_2p_plus_one_is_perfect_power_l246_246506


namespace statements_evaluation_l246_246181

-- Define the statements A, B, C, D, E as propositions
def A : Prop := ∀ (A B C D E : Prop), (A → ¬B ∧ ¬C ∧ ¬D ∧ ¬E)
def B : Prop := sorry  -- Assume we have some way to read the statement B under special conditions
def C : Prop := ∀ (A B C D E : Prop), (A ∧ B ∧ C ∧ D ∧ E)
def D : Prop := sorry  -- Assume we have some way to read the statement D under special conditions
def E : Prop := A

-- Prove the conditions
theorem statements_evaluation : ¬ A ∧ ¬ C ∧ ¬ E ∧ B ∧ D :=
by
  sorry

end statements_evaluation_l246_246181


namespace intersection_M_N_l246_246088

def M : Set ℤ := {0, 1, 2}
def N : Set ℤ := {x | -1 ≤ x ∧ x ≤ 1}

theorem intersection_M_N : M ∩ N = {0, 1} := 
by
  sorry

end intersection_M_N_l246_246088


namespace flower_bee_relationship_l246_246718

def numberOfBees (flowers : ℕ) (fewer_bees : ℕ) : ℕ :=
  flowers - fewer_bees

theorem flower_bee_relationship :
  numberOfBees 5 2 = 3 := by
  sorry

end flower_bee_relationship_l246_246718


namespace value_of_expression_l246_246376

theorem value_of_expression (n : ℝ) (h : n + 1/n = 10) : n^2 + (1/n^2) + 6 = 104 :=
by
  sorry

end value_of_expression_l246_246376


namespace find_s_value_l246_246499

noncomputable def value_of_s (s : ℝ) : Prop :=
  let v := (λ s => (1 + 5 * s, -2 + 3 * s, 4 - 2 * s))
  let a := (-3, 6, 7)
  -- Orthogonality condition
  let d := (λ s => (1 + 5 * s + 3, -2 + 3 * s - 6, 4 - 2 * s - 7))
  let direction := (5, 3, -2)
  (d s).fst * direction.fst + (d s).snd.fst * direction.snd.fst + (d s).snd.snd * direction.snd.snd = 0

theorem find_s_value : ∃ s, value_of_s s ∧ s = -1 / 19 := 
by
  use -1/19
  sorry

end find_s_value_l246_246499


namespace line_integral_path_independence_l246_246699

noncomputable def vector_field (x y z : ℝ) : ℝ × ℝ × ℝ :=
  (x * y^2 * z, (1 + z^2) * y * z, (1 / 2) * x^2 * y^2)

-- Define the problem statement
theorem line_integral_path_independence :
  (∀ (x y z : ℝ), continuous (λ t : ℝ, vector_field x y z)) ∧
  (∀ (x y z : ℝ), differentiable ℝ (λ t : ℝ, vector_field x y z)) ∧
  (domain_is_simply_connected : ∀ (x y z : ℝ), ∃ u, vector_field x y z = vector_field u (x + y) (x + z)) ∧
  (∀ (x y z : ℝ), ∇ × vector_field x y z = (0, 0, 0)) →
  ∀ L : ℝ → ℝ^3, path_independent (vector_field) L :=
sorry

end line_integral_path_independence_l246_246699


namespace fourth_rectangle_area_is_112_l246_246973

def area_of_fourth_rectangle (length : ℕ) (width : ℕ) (area1 : ℕ) (area2 : ℕ) (area3 : ℕ) : ℕ :=
  length * width - area1 - area2 - area3

theorem fourth_rectangle_area_is_112 :
  area_of_fourth_rectangle 20 12 24 48 36 = 112 :=
by
  sorry

end fourth_rectangle_area_is_112_l246_246973


namespace art_museum_survey_l246_246167

theorem art_museum_survey (V E : ℕ) 
  (h1 : ∀ (x : ℕ), x = 140 → ¬ (x ≤ E))
  (h2 : E = (3 / 4) * V)
  (h3 : V = E + 140) :
  V = 560 := by
  sorry

end art_museum_survey_l246_246167


namespace largest_possible_average_l246_246693

noncomputable def ten_test_scores (a b c d e f g h i j : ℤ) : ℤ :=
  a + b + c + d + e + f + g + h + i + j

theorem largest_possible_average
  (a b c d e f g h i j : ℤ)
  (h1 : 0 ≤ a ∧ a ≤ 100)
  (h2 : 0 ≤ b ∧ b ≤ 100)
  (h3 : 0 ≤ c ∧ c ≤ 100)
  (h4 : 0 ≤ d ∧ d ≤ 100)
  (h5 : 0 ≤ e ∧ e ≤ 100)
  (h6 : 0 ≤ f ∧ f ≤ 100)
  (h7 : 0 ≤ g ∧ g ≤ 100)
  (h8 : 0 ≤ h ∧ h ≤ 100)
  (h9 : 0 ≤ i ∧ i ≤ 100)
  (h10 : 0 ≤ j ∧ j ≤ 100)
  (h11 : a + b + c + d ≤ 190)
  (h12 : b + c + d + e ≤ 190)
  (h13 : c + d + e + f ≤ 190)
  (h14 : d + e + f + g ≤ 190)
  (h15 : e + f + g + h ≤ 190)
  (h16 : f + g + h + i ≤ 190)
  (h17 : g + h + i + j ≤ 190)
  : ((ten_test_scores a b c d e f g h i j : ℚ) / 10) ≤ 44.33 := sorry

end largest_possible_average_l246_246693


namespace speed_of_train_in_kmh_l246_246179

-- Define the conditions
def time_to_cross_pole : ℝ := 6
def length_of_train : ℝ := 100
def conversion_factor : ℝ := 18 / 5

-- Using the conditions to assert the speed of the train
theorem speed_of_train_in_kmh (t : ℝ) (d : ℝ) (conv_factor : ℝ) : 
  t = time_to_cross_pole → 
  d = length_of_train → 
  conv_factor = conversion_factor → 
  (d / t) * conv_factor = 50 := 
by 
  intros h_t h_d h_conv_factor
  sorry

end speed_of_train_in_kmh_l246_246179


namespace eq_determines_ratio_l246_246420

theorem eq_determines_ratio (a b x y : ℝ) (h : a * x^3 + b * x^2 * y + b * x * y^2 + a * y^3 = 0) :
  ∃ t : ℝ, t = x / y ∧ (a * t^3 + b * t^2 + b * t + a = 0) :=
sorry

end eq_determines_ratio_l246_246420


namespace find_x_values_l246_246637

theorem find_x_values (x : ℝ) :
  (2 / (x + 2) + 8 / (x + 4) ≥ 2) ↔ (x ∈ Set.Ici 2 ∨ x ∈ Set.Iic (-4)) := by
sorry

end find_x_values_l246_246637


namespace max_last_place_score_l246_246764

theorem max_last_place_score (n : ℕ) (h : n ≥ 4) :
  ∃ k, (∀ m, m < n -> (k + m) < (n * 3)) ∧ 
     (∀ i, ∃ j, j < n ∧ i = k + j) ∧
     (n * 2 - 2) = (k + n - 1) ∧ 
     k = n - 2 := 
sorry

end max_last_place_score_l246_246764


namespace perimeter_of_rectangle_WXYZ_l246_246879

theorem perimeter_of_rectangle_WXYZ 
  (WE XF EG FH : ℝ)
  (h1 : WE = 10)
  (h2 : XF = 25)
  (h3 : EG = 20)
  (h4 : FH = 50) :
  let p := 53 -- By solving the equivalent problem, where perimeter is simplified to 53/1 which gives p = 53 and q = 1
  let q := 29
  p + q = 102 := 
by
  sorry

end perimeter_of_rectangle_WXYZ_l246_246879


namespace value_of_x_l246_246105

theorem value_of_x (x : ℝ) (h : x = 88 * 1.2) : x = 105.6 :=
by
  sorry

end value_of_x_l246_246105


namespace spherical_to_rectangular_correct_l246_246779

noncomputable def spherical_to_rectangular (rho θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (rho * Real.sin φ * Real.cos θ, rho * Real.sin φ * Real.sin θ, rho * Real.cos φ)

def initial_rectangular_coords : ℝ × ℝ × ℝ := (3, -4, 12)

noncomputable def spherical_coords (rect_coords : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let ρ := Real.sqrt (rect_coords.1 ^ 2 + rect_coords.2 ^ 2 + rect_coords.3 ^ 2)
  let θ := Real.atan rect_coords.2 rect_coords.1
  let φ := Real.acos (rect_coords.3 / ρ)
  (ρ, θ, φ)

def new_spherical_coords (spherical : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let ρ := spherical.1
  let θ := spherical.2
  let φ := spherical.3 + Real.pi / 4
  (ρ, θ, φ)

theorem spherical_to_rectangular_correct :
  let (ρ, θ, φ) := spherical_coords initial_rectangular_coords
  let (new_ρ, new_θ, new_φ) := new_spherical_coords (ρ, θ, φ)
  spherical_to_rectangular new_ρ new_θ new_φ
  = (-51 * Real.sqrt 2 / 10, -68 * Real.sqrt 2 / 10, 91 * Real.sqrt 2 / 20) := by
  sorry

end spherical_to_rectangular_correct_l246_246779


namespace sum_of_ages_l246_246977

/-- Given a woman's age is three years more than twice her son's age, 
and the son is 27 years old, prove that the sum of their ages is 84 years. -/
theorem sum_of_ages (son_age : ℕ) (woman_age : ℕ)
  (h1 : son_age = 27)
  (h2 : woman_age = 3 + 2 * son_age) :
  son_age + woman_age = 84 := 
sorry

end sum_of_ages_l246_246977


namespace pugs_cleaning_time_l246_246163

theorem pugs_cleaning_time : 
  (∀ (p t: ℕ), 15 * 12 = p * t ↔ 15 * 12 = 4 * 45) :=
by
  sorry

end pugs_cleaning_time_l246_246163


namespace fraction_spent_on_candy_l246_246597

theorem fraction_spent_on_candy (initial_quarters : ℕ) (initial_cents remaining_cents cents_per_dollar : ℕ) (fraction_spent : ℝ) :
  initial_quarters = 14 ∧ remaining_cents = 300 ∧ initial_cents = initial_quarters * 25 ∧ cents_per_dollar = 100 →
  fraction_spent = (initial_cents - remaining_cents) / cents_per_dollar →
  fraction_spent = 1 / 2 :=
by
  intro h1 h2
  sorry

end fraction_spent_on_candy_l246_246597


namespace find_defective_keys_l246_246964

-- Definitions from the conditions
def ten_digit_sequence : Type := list ℕ
def registered_digits : Type := list ℕ

axiom typed_ten_digits (s : ten_digit_sequence) : s.length = 10
axiom only_seven_registered (t : registered_digits) : t.length = 7
axiom three_missing_digits (s : ten_digit_sequence) (t : registered_digits) : 
             s.length - t.length = 3

-- This indicates that it is the same type of digits just subsets of initial sequence
axiom all_digits_in_sequence (s : ten_digit_sequence) (t : registered_digits) : 
            ∀ (d : ℕ), d ∈ t → d ∈ s

axiom defective_key_condition (s : ten_digit_sequence) (t : registered_digits) : 
            ∃ d : ℕ, (d ∈ s ∧ d ∉ t) ∧ count s d >= 5 ∧ count t d = 2

axiom multiple_defective_keys_condition (s : ten_digit_sequence) (t : registered_digits): 
           ∃ d1 d2 : ℕ, d1 ≠ d2 ∧ (d1 ∈ s ∧ d2 ∈ s) ∧ (d1 ∉ t ∧ d2 ∉ t) ∧ 
           (count s d1 >= 3 ∧ count s d2 >= 3)

-- Proving the answer:
theorem find_defective_keys (s : ten_digit_sequence) (t : registered_digits) :
  typed_ten_digits s → only_seven_registered t → three_missing_digits s t → 
  all_digits_in_sequence s t → defective_key_condition s t → multiple_defective_keys_condition s t → 
  ∃ (keys : list ℕ), keys = [7, 9] :=
begin
  sorry
end

end find_defective_keys_l246_246964


namespace cookies_eaten_l246_246314

theorem cookies_eaten (original remaining : ℕ) (h_original : original = 18) (h_remaining : remaining = 9) :
    original - remaining = 9 := by
  sorry

end cookies_eaten_l246_246314


namespace dot_product_eq_neg20_l246_246067

def a : ℝ × ℝ := (3, -1)
def b : ℝ × ℝ := (-5, 5)

def dot_product (x y : ℝ × ℝ) : ℝ :=
x.1 * y.1 + x.2 * y.2

theorem dot_product_eq_neg20 :
  dot_product a b = -20 :=
by
  sorry

end dot_product_eq_neg20_l246_246067


namespace example_theorem_l246_246673

noncomputable def P (A : Set ℕ) : ℝ := sorry

variable (A1 A2 A3 : Set ℕ)

axiom prob_A1 : P A1 = 0.2
axiom prob_A2 : P A2 = 0.3
axiom prob_A3 : P A3 = 0.5

theorem example_theorem : P (A1 ∪ A2) ≤ 0.5 := 
by {
  sorry
}

end example_theorem_l246_246673


namespace find_largest_number_l246_246785

theorem find_largest_number :
  let a := -(abs (-3) ^ 3)
  let b := -((-3) ^ 3)
  let c := (-3) ^ 3
  let d := -(3 ^ 3)
  b = 27 ∧ b > a ∧ b > c ∧ b > d := by
  sorry

end find_largest_number_l246_246785


namespace geometric_sum_eight_terms_l246_246325

theorem geometric_sum_eight_terms :
  let a0 := (1 : ℚ) / 3
  let r := (1 : ℚ) / 4
  let n := 8
  let S_n := a0 * (1 - r^n) / (1 - r)
  S_n = 65535 / 147456 := by
  sorry

end geometric_sum_eight_terms_l246_246325


namespace older_brother_catches_up_l246_246204

theorem older_brother_catches_up :
  ∃ (x : ℝ), 0 ≤ x ∧ 6 * x = 2 + 2 * x ∧ x + 1 < 1.75 :=
by
  sorry

end older_brother_catches_up_l246_246204


namespace Tamara_is_95_inches_l246_246427

/- Defining the basic entities: Kim's height (K), Tamara's height, Gavin's height -/
def Kim_height (K : ℝ) := K
def Tamara_height (K : ℝ) := 3 * K - 4
def Gavin_height (K : ℝ) := 2 * K + 6

/- Combined height equation -/
def combined_height (K : ℝ) := (Tamara_height K) + (Kim_height K) + (Gavin_height K) = 200

/- Given that Kim's height satisfies the combined height condition,
   proving that Tamara's height is 95 inches -/
theorem Tamara_is_95_inches (K : ℝ) (h : combined_height K) : Tamara_height K = 95 :=
by
  sorry

end Tamara_is_95_inches_l246_246427


namespace g_is_odd_l246_246405

def g (x : ℝ) : ℝ := 1 / (3 ^ x - 1) + 1 / 3

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  sorry

end g_is_odd_l246_246405


namespace find_natural_numbers_l246_246715

def LCM (a b : ℕ) : ℕ := (a * b) / (Nat.gcd a b)

theorem find_natural_numbers :
  ∃ a b : ℕ, a + b = 54 ∧ LCM a b - Nat.gcd a b = 114 ∧ (a = 24 ∧ b = 30 ∨ a = 30 ∧ b = 24) := by {
  sorry
}

end find_natural_numbers_l246_246715


namespace flower_bee_relationship_l246_246719

def numberOfBees (flowers : ℕ) (fewer_bees : ℕ) : ℕ :=
  flowers - fewer_bees

theorem flower_bee_relationship :
  numberOfBees 5 2 = 3 := by
  sorry

end flower_bee_relationship_l246_246719


namespace dividend_is_correct_l246_246831

def quotient : ℕ := 20
def divisor : ℕ := 66
def remainder : ℕ := 55

def dividend := (divisor * quotient) + remainder

theorem dividend_is_correct : dividend = 1375 := by
  sorry

end dividend_is_correct_l246_246831


namespace Andy_has_4_more_candies_than_Caleb_l246_246328

-- Define the initial candies each person has
def Billy_initial_candies : ℕ := 6
def Caleb_initial_candies : ℕ := 11
def Andy_initial_candies : ℕ := 9

-- Define the candies bought by the father and their distribution
def father_bought_candies : ℕ := 36
def Billy_received_from_father : ℕ := 8
def Caleb_received_from_father : ℕ := 11

-- Calculate the remaining candies for Andy after distribution
def Andy_received_from_father : ℕ := father_bought_candies - (Billy_received_from_father + Caleb_received_from_father)

-- Calculate the total candies each person has
def Billy_total_candies : ℕ := Billy_initial_candies + Billy_received_from_father
def Caleb_total_candies : ℕ := Caleb_initial_candies + Caleb_received_from_father
def Andy_total_candies : ℕ := Andy_initial_candies + Andy_received_from_father

-- Prove that Andy has 4 more candies than Caleb
theorem Andy_has_4_more_candies_than_Caleb :
  Andy_total_candies = Caleb_total_candies + 4 :=
by {
  -- Skipping the proof
  sorry
}

end Andy_has_4_more_candies_than_Caleb_l246_246328


namespace john_spent_fraction_on_snacks_l246_246682

theorem john_spent_fraction_on_snacks (x : ℚ) :
  (∀ (x : ℚ), (1 - x) * 20 - (3 / 4) * (1 - x) * 20 = 4) → (x = 1 / 5) :=
by sorry

end john_spent_fraction_on_snacks_l246_246682


namespace problem_statement_l246_246710

variables (a b : ℝ)

-- Conditions: The lines \(x = \frac{1}{3}y + a\) and \(y = \frac{1}{3}x + b\) intersect at \((3, 1)\).
def lines_intersect_at (a b : ℝ) : Prop :=
  (3 = (1/3) * 1 + a) ∧ (1 = (1/3) * 3 + b)

-- Goal: Prove that \(a + b = \frac{8}{3}\)
theorem problem_statement (H : lines_intersect_at a b) : a + b = 8 / 3 :=
by
  sorry

end problem_statement_l246_246710


namespace sqrt_frac_meaningful_l246_246381

theorem sqrt_frac_meaningful (x : ℝ) (h : 1 / (x - 1) > 0) : x > 1 :=
sorry

end sqrt_frac_meaningful_l246_246381


namespace factorize_expression_l246_246634

variable (m n : ℤ)

theorem factorize_expression : 2 * m * n^2 - 12 * m * n + 18 * m = 2 * m * (n - 3)^2 := by
  sorry

end factorize_expression_l246_246634


namespace parallel_lines_l246_246604

-- Definitions of lines and plane
variable {Line : Type}
variable {Plane : Type}
variable (a b c : Line)
variable (α : Plane)

-- Parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallelPlane : Line → Plane → Prop)

-- Given conditions
variable (h1 : parallel a c)
variable (h2 : parallel b c)

-- Theorem statement
theorem parallel_lines (a b c : Line) 
                       (α : Plane) 
                       (parallel : Line → Line → Prop) 
                       (perpendicular : Line → Line → Prop) 
                       (parallelPlane : Line → Plane → Prop)
                       (h1 : parallel a c) 
                       (h2 : parallel b c) : 
                       parallel a b :=
sorry

end parallel_lines_l246_246604


namespace sequence_properties_l246_246334

def arithmetic_sequence (a d n : ℤ) : ℤ :=
  a + (n - 1) * d

def is_multiple_of_10 (n : ℤ) : Prop :=
  n % 10 = 0

noncomputable def sum_of_multiples_of_10 (a d n : ℤ) : ℤ :=
  ∑ i in (Finset.range (n + 1)).filter (λ k, is_multiple_of_10 (arithmetic_sequence a d k)), 
  arithmetic_sequence a d i

theorem sequence_properties :
  let a := -45
  let d := 7 in
  let last := 98 in
  let n := 21 in
  -- Number of terms in the sequence
  (arithmetic_sequence a d n ≤ last) ∧
  -- Sum of numbers in the sequence that are multiples of 10
  (sum_of_multiples_of_10 a d n = 60) :=
by
  sorry

end sequence_properties_l246_246334


namespace value_of_f_a1_a3_a5_l246_246516

-- Definitions
def monotonically_increasing (f : ℝ → ℝ) :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂

def odd_function (f : ℝ → ℝ) :=
  ∀ x : ℝ, f (-x) = -f x

def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n+1) = a n + d

-- Problem statement
theorem value_of_f_a1_a3_a5 (f : ℝ → ℝ) (a : ℕ → ℝ) :
  monotonically_increasing f →
  odd_function f →
  arithmetic_sequence a →
  a 3 > 0 →
  f (a 1) + f (a 3) + f (a 5) > 0 :=
by
  intros h_mono h_odd h_arith h_a3
  sorry

end value_of_f_a1_a3_a5_l246_246516


namespace no_prime_number_between_30_and_40_mod_9_eq_7_l246_246293

theorem no_prime_number_between_30_and_40_mod_9_eq_7 : ¬ ∃ n : ℕ, 30 ≤ n ∧ n ≤ 40 ∧ Nat.Prime n ∧ n % 9 = 7 :=
by
  sorry

end no_prime_number_between_30_and_40_mod_9_eq_7_l246_246293


namespace person_is_not_sane_l246_246170

-- Definitions
def Person : Type := sorry
def sane : Person → Prop := sorry
def human : Person → Prop := sorry
def vampire : Person → Prop := sorry
def declares (p : Person) (s : String) : Prop := sorry

-- Conditions
axiom transylvanian_declares_vampire (p : Person) : declares p "I am a vampire"
axiom sane_human_never_claims_vampire (p : Person) : sane p → human p → ¬ declares p "I am a vampire"
axiom sane_vampire_never_admits_vampire (p : Person) : sane p → vampire p → ¬ declares p "I am a vampire"
axiom insane_human_might_claim_vampire (p : Person) : ¬ sane p → human p → declares p "I am a vampire"
axiom insane_vampire_might_admit_vampire (p : Person) : ¬ sane p → vampire p → declares p "I am a vampire"

-- Proof statement
theorem person_is_not_sane (p : Person) : declares p "I am a vampire" → ¬ sane p :=
by
  intros h
  sorry

end person_is_not_sane_l246_246170


namespace marbles_given_to_joan_l246_246695

def mary_original_marbles : ℝ := 9.0
def mary_marbles_left : ℝ := 6.0

theorem marbles_given_to_joan :
  mary_original_marbles - mary_marbles_left = 3 := 
by
  sorry

end marbles_given_to_joan_l246_246695


namespace opposite_of_negative_five_l246_246931

theorem opposite_of_negative_five : (-(-5) = 5) :=
by
  sorry

end opposite_of_negative_five_l246_246931


namespace range_of_a_l246_246989

variable {a : ℝ}

def A := Set.Ioo (-1 : ℝ) 1
def B (a : ℝ) := Set.Ioo a (a + 1)

theorem range_of_a :
  B a ⊆ A ↔ (-1 : ℝ) ≤ a ∧ a ≤ 0 :=
by
  sorry

end range_of_a_l246_246989


namespace algebraic_expression_evaluation_l246_246371

theorem algebraic_expression_evaluation
  (x y p q : ℝ)
  (h1 : x + y = 0)
  (h2 : p * q = 1) : (x + y) - 2 * (p * q) = -2 :=
by
  sorry

end algebraic_expression_evaluation_l246_246371


namespace number_of_years_borrowed_l246_246611

theorem number_of_years_borrowed (n : ℕ)
  (H1 : ∃ (p : ℕ), 5000 = p ∧ 4 = 4 ∧ n * 200 = 150)
  (H2 : ∃ (q : ℕ), 5000 = q ∧ 7 = 7 ∧ n * 350 = 150)
  : n = 1 :=
by
  sorry

end number_of_years_borrowed_l246_246611


namespace f_11_f_2021_eq_neg_one_l246_246458

def f (n : ℕ) : ℚ := sorry

axiom recurrence_relation (n : ℕ) : f (n + 3) = (f n - 1) / (f n + 1)
axiom f1_ne_zero : f 1 ≠ 0
axiom f1_ne_one : f 1 ≠ 1
axiom f1_ne_neg_one : f 1 ≠ -1

theorem f_11_f_2021_eq_neg_one : f 11 * f 2021 = -1 := 
by
  sorry

end f_11_f_2021_eq_neg_one_l246_246458


namespace value_of_expression_l246_246375

theorem value_of_expression (n : ℝ) (h : n + 1/n = 10) : n^2 + (1/n^2) + 6 = 104 :=
by
  sorry

end value_of_expression_l246_246375


namespace ratio_rect_prism_l246_246947

namespace ProofProblem

variables (w l h : ℕ)
def rect_prism (w l h : ℕ) : Prop := w * l * h = 128

theorem ratio_rect_prism (h1 : rect_prism w l h) :
  (w : ℕ) ≠ 0 ∧ (l : ℕ) ≠ 0 ∧ (h : ℕ) ≠ 0 ∧ 
  (∃ k, w = k ∧ l = k ∧ h = 2 * k) :=
sorry

end ProofProblem

end ratio_rect_prism_l246_246947


namespace dennis_rocks_left_l246_246988

-- Definitions based on conditions:
def initial_rocks : ℕ := 10
def rocks_eaten_by_fish (initial : ℕ) : ℕ := initial / 2
def rocks_spat_out_by_fish : ℕ := 2

-- Total rocks left:
def total_rocks_left (initial : ℕ) (spat_out : ℕ) : ℕ :=
  (rocks_eaten_by_fish initial) + spat_out

-- Statement to be proved:
theorem dennis_rocks_left : total_rocks_left initial_rocks rocks_spat_out_by_fish = 7 :=
by
  -- Conclusion by calculation (Proved in steps)
  sorry

end dennis_rocks_left_l246_246988


namespace opposite_of_neg_five_l246_246900

/-- Definition of the opposite of a number -/
def opposite (a : Int) : Int := -a

theorem opposite_of_neg_five : opposite (-5) = 5 := by
  sorry

end opposite_of_neg_five_l246_246900


namespace opposite_of_neg_five_l246_246926

theorem opposite_of_neg_five : -(-5) = 5 := by
  sorry

end opposite_of_neg_five_l246_246926


namespace coordinates_equality_l246_246891

theorem coordinates_equality (a b : ℤ) 
  (h1 : b - 1 = 2) 
  (h2 : a + 3 = -1) : a + b = -1 :=
by 
  sorry

end coordinates_equality_l246_246891


namespace negation_universal_proposition_l246_246440

theorem negation_universal_proposition :
  (¬ ∀ x : ℝ, 0 < x → x^2 + x + 1 > 0) ↔ (∃ x : ℝ, 0 < x ∧ x^2 + x + 1 ≤ 0) :=
by
  sorry

end negation_universal_proposition_l246_246440


namespace other_x_intercept_l246_246038

def foci1 := (0, -3)
def foci2 := (4, 0)
def x_intercept1 := (0, 0)

theorem other_x_intercept :
  (∃ x : ℝ, (|x - 4| + |-3| * x = 7)) → x = 11 / 4 := by
  sorry

end other_x_intercept_l246_246038


namespace remainder_when_P_divided_by_ab_l246_246215

-- Given conditions
variables {P a b c Q Q' R R' : ℕ}

-- Provided equations as conditions
def equation1 : P = a * Q + R :=
sorry

def equation2 : Q = (b + c) * Q' + R' :=
sorry

-- Proof problem statement
theorem remainder_when_P_divided_by_ab :
  P % (a * b) = (a * c * Q' + a * R' + R) % (a * b) :=
by
  sorry

end remainder_when_P_divided_by_ab_l246_246215


namespace hyperbola_eccentricity_l246_246985

def hyperbola : Prop :=
  ∀ (x y : ℝ), (x^2 / 9) - (y^2 / 16) = 1

noncomputable def eccentricity : ℝ :=
  let a := 3
  let b := 4
  let c := Real.sqrt (a^2 + b^2)
  c / a

theorem hyperbola_eccentricity : 
  (∀ (x y : ℝ), (x^2 / 9) - (y^2 / 16) = 1) → eccentricity = 5 / 3 :=
by
  intros h
  funext
  exact sorry

end hyperbola_eccentricity_l246_246985


namespace percentage_rotten_bananas_l246_246317

theorem percentage_rotten_bananas :
  let total_oranges := 600
  let total_bananas := 400
  let rotten_oranges_percentage := 0.15
  let good_condition_percentage := 0.878
  let total_fruits := total_oranges + total_bananas 
  let rotten_oranges := rotten_oranges_percentage * total_oranges 
  let good_fruits := good_condition_percentage * total_fruits
  let rotten_fruits := total_fruits - good_fruits
  let rotten_bananas := rotten_fruits - rotten_oranges
  (rotten_bananas / total_bananas) * 100 = 8 := by
  {
    -- Calculations and simplifications go here
    sorry
  }

end percentage_rotten_bananas_l246_246317


namespace opposite_of_negative_five_l246_246909

theorem opposite_of_negative_five : ∀ x : Int, -5 + x = 0 → x = 5 :=
by
  intros x h
  sorry

end opposite_of_negative_five_l246_246909


namespace area_of_EFGH_l246_246877

variables (EF FG EH HG EG : ℝ)
variables (distEFGH : EF ≠ HG ∧ EG = 5 ∧ EF^2 + FG^2 = 25 ∧ EH^2 + HG^2 = 25)

theorem area_of_EFGH : 
  ∃ EF FG EH HG : ℕ, EF ≠ HG ∧ EG = 5 
  ∧ EF^2 + FG^2 = 25 
  ∧ EH^2 + HG^2 = 25 
  ∧ EF * FG / 2 + EH * HG / 2 = 12 :=
by { sorry }

end area_of_EFGH_l246_246877


namespace part1_part2_l246_246523

open Set

variable {U : Type} [TopologicalSpace U]

def universal_set : Set ℝ := univ
def set_A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def set_B (k : ℝ) : Set ℝ := {x | x ≤ k}

theorem part1 (k : ℝ) (hk : k = 1) :
  A ∩ (univ \ set_B k) = {x | 1 < x ∧ x < 3} :=
by
  sorry

theorem part2 (k : ℝ) (h : set_A ∩ set_B k ≠ ∅) :
  k ≥ -1 :=
by
  sorry

end part1_part2_l246_246523


namespace find_coordinates_of_B_l246_246453

theorem find_coordinates_of_B (A B : ℝ × ℝ)
  (h1 : ∃ (C1 C2 : ℝ × ℝ), C1.2 = 0 ∧ C2.2 = 0 ∧ (dist C1 A = dist C1 B) ∧ (dist C2 A = dist C2 B) ∧ (A ≠ B))
  (h2 : A = (-3, 2)) :
  B = (-3, -2) :=
sorry

end find_coordinates_of_B_l246_246453


namespace part_one_part_two_l246_246525

section part_one
variables {x : ℝ}

def f (x : ℝ) : ℝ := |x - 3| + |x - 2|

theorem part_one : ∀ x : ℝ, f x ≥ 3 ↔ (x ≤ 1 ∨ x ≥ 4) := by
  sorry
end part_one

section part_two
variables {a x : ℝ}

def g (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 2|

theorem part_two : (∀ x ∈ (Set.Icc 1 2), g a x ≤ |x - 4|) → (a ∈ Set.Icc (-3) 0) := by
  sorry
end part_two

end part_one_part_two_l246_246525


namespace highest_elevation_l246_246033

   noncomputable def elevation (t : ℝ) : ℝ := 240 * t - 24 * t^2

   theorem highest_elevation : ∃ t : ℝ, elevation t = 600 ∧ ∀ x : ℝ, elevation x ≤ 600 := 
   sorry
   
end highest_elevation_l246_246033


namespace max_erasers_l246_246037

theorem max_erasers (p n e : ℕ) (h₁ : p ≥ 1) (h₂ : n ≥ 1) (h₃ : e ≥ 1) (h₄ : 3 * p + 4 * n + 8 * e = 60) :
  e ≤ 5 :=
sorry

end max_erasers_l246_246037


namespace carol_total_peanuts_l246_246792

open Nat

-- Define the conditions
def peanuts_from_tree : Nat := 48
def peanuts_from_ground : Nat := 178
def bags_of_peanuts : Nat := 3
def peanuts_per_bag : Nat := 250

-- Define the total number of peanuts Carol has to prove it equals 976
def total_peanuts (peanuts_from_tree peanuts_from_ground bags_of_peanuts peanuts_per_bag : Nat) : Nat :=
  peanuts_from_tree + peanuts_from_ground + (bags_of_peanuts * peanuts_per_bag)

theorem carol_total_peanuts : total_peanuts peanuts_from_tree peanuts_from_ground bags_of_peanuts peanuts_per_bag = 976 :=
  by
    -- proof goes here
    sorry

end carol_total_peanuts_l246_246792


namespace num_broadcasting_methods_l246_246174

theorem num_broadcasting_methods : 
  let n := 6
  let commercials := 4
  let public_services := 2
  (public_services * commercials!) = 48 :=
by
  let n := 6
  let commercials := 4
  let public_services := 2
  have total_methods : (public_services * commercials!) = 48 := sorry
  exact total_methods

end num_broadcasting_methods_l246_246174


namespace faulty_keys_l246_246966

noncomputable def faulty_digits (typed_sequence : List ℕ) : Set ℕ :=
  { d | d = 7 ∨ d = 9 }

theorem faulty_keys (typed_sequence : List ℕ) (h : typed_sequence.length = 10) :
  (∃ faulty_keys : Set ℕ, ∃ missing_digits : ℕ, missing_digits = 3 ∧ faulty_keys = {7, 9}) :=
sorry

end faulty_keys_l246_246966


namespace probability_seventh_week_A_l246_246000

/-- Define the problem parameters -/
def codes : Finset ℕ := {0, 1, 2, 3}

/-- Define that code A is used in week 1 -/
def code_first_week : ℕ := 0

/-- Define the random selection rule for subsequent weeks -/
def prob_not_prev_code (prev_code : ℕ) : ℕ → ℚ :=
  λ code, if code ≠ prev_code then (1/3 : ℚ) else 0

/-- Define the probability of selecting code A in the seventh week -/
def probability_A_seventh_week (init_code : ℕ) : ℚ :=
  let prob_week : ℕ → ℚ → ℚ
    | 0, p := p
    | k + 1, p := 
      (1/3 : ℚ) * (1 - 
        prob_week k ((1/4 : ℚ) * (-1/3 : ℚ) ^ (k - 1) + 1/4 : ℚ)
      )
  in prob_week 6 1

/-- Formal statement of the problem --/
theorem probability_seventh_week_A : probability_A_seventh_week code_first_week = (61 / 243 : ℚ) :=
  sorry

end probability_seventh_week_A_l246_246000


namespace stratified_sampling_young_employees_l246_246384

variable (total_employees elderly_employees middle_aged_employees young_employees sample_size : ℕ)

-- Conditions
axiom total_employees_eq : total_employees = 750
axiom elderly_employees_eq : elderly_employees = 150
axiom middle_aged_employees_eq : middle_aged_employees = 250
axiom young_employees_eq : young_employees = 350
axiom sample_size_eq : sample_size = 15

-- The proof problem
theorem stratified_sampling_young_employees :
  young_employees / total_employees * sample_size = 7 :=
by
  sorry

end stratified_sampling_young_employees_l246_246384


namespace part1_part2_l246_246820

open Real

variables (x a : ℝ)

def p (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2 * x - 8 > 0

theorem part1 (h : a = 1) (h_pq : p x 1 ∧ q x) : 2 < x ∧ x < 3 :=
by sorry

theorem part2 (hpq : ∀ (a x : ℝ), ¬ p x a → ¬ q x) : 1 ≤ a ∧ a ≤ 2 :=
by sorry

end part1_part2_l246_246820


namespace power_function_point_l246_246522

theorem power_function_point (m n: ℝ) (h: (m - 1) * m^n = 8) : n^(-m) = 1/9 := 
  sorry

end power_function_point_l246_246522


namespace expected_value_l246_246777

theorem expected_value (p1 p2 p3 p4 p5 p6 : ℕ) (hp1 : p1 = 1) (hp2 : p2 = 5) (hp3 : p3 = 10) 
(hp4 : p4 = 25) (hp5 : p5 = 50) (hp6 : p6 = 100) :
  (p1 / 2 + p2 / 2 + p3 / 2 + p4 / 2 + p5 / 2 + p6 / 2 : ℝ) = 95.5 := by
  sorry

end expected_value_l246_246777


namespace divisibility_proof_l246_246687

theorem divisibility_proof (n : ℕ) (hn : 0 < n) (h : n ∣ (10^n - 1)) : 
  n ∣ ((10^n - 1) / 9) :=
  sorry

end divisibility_proof_l246_246687


namespace opposite_of_neg_five_is_five_l246_246918

-- Define the condition for the opposite of a number
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- State the theorem that the opposite of -5 is 5
theorem opposite_of_neg_five_is_five : is_opposite (-5) 5 :=
by
  -- Insert the proof here
  sorry

end opposite_of_neg_five_is_five_l246_246918


namespace at_least_one_ge_two_l246_246860

theorem at_least_one_ge_two (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h_sum : a + b + c = 3) :
  (a + 1/b >= 2) ∨ (b + 1/c >= 2) ∨ (c + 1/a >= 2) :=
sorry

end at_least_one_ge_two_l246_246860


namespace mother_present_age_l246_246974

def person_present_age (P M : ℕ) : Prop :=
  P = (2 / 5) * M

def person_age_in_10_years (P M : ℕ) : Prop :=
  P + 10 = (1 / 2) * (M + 10)

theorem mother_present_age (P M : ℕ) (h1 : person_present_age P M) (h2 : person_age_in_10_years P M) : M = 50 :=
sorry

end mother_present_age_l246_246974


namespace fill_6x6_with_tetris_pieces_l246_246026

-- Definitions for Tetris pieces (represented as internal notation or lists)
-- Definition of each Tetris piece (can be expanded if necessary for proofs)
inductive TetrisPiece
| O | I | T | S | Z | L | J
-- If we assume these shapes can be represented appropriately in Lean.

-- The 6x6 grid using Fin 6 for constraint bounds
def grid : Matrix (Fin 6) (Fin 6) (Option TetrisPiece) := sorry

-- Proof statement that grid can be filled with the given pieces
theorem fill_6x6_with_tetris_pieces : 
  ∃ (g : Matrix (Fin 6) (Fin 6) (Option TetrisPiece)),
    (∀ i j, g i j ≠ none) ∧ -- Every cell is filled
    (∀ p : TetrisPiece, ∃ i j, g i j = some p) -- Each piece is used at least once
:= 
begin
  sorry
end

end fill_6x6_with_tetris_pieces_l246_246026


namespace average_of_middle_three_l246_246434

theorem average_of_middle_three
  (a b c d e : ℕ)
  (h_distinct : a < b ∧ b < c ∧ c < d ∧ d < e)
  (h_sum : a + b + c + d + e = 25)
  (h_max_diff : ∀ x y : ℕ, x + y = 24 → (e - a) ≥ (y - x)) :
  (b + c + d) / 3 = 3 :=
by
  sorry

end average_of_middle_three_l246_246434


namespace max_sumo_wrestlers_l246_246489

/-- 
At a sumo wrestling tournament, 20 sumo wrestlers participated.
The average weight of the wrestlers is 125 kg.
Individuals weighing less than 90 kg cannot participate.
Prove that the maximum possible number of sumo wrestlers weighing more than 131 kg is 17.
-/
theorem max_sumo_wrestlers : 
  ∀ (weights : Fin 20 → ℝ), 
    (∀ i, weights i ≥ 90) → 
    (∑ i, weights i = 2500) → 
    (∃ n : ℕ,  n ≤ 20 ∧ 
      (n = 17 → (∑ i in Finset.filter (λ i, weights i > 131) Finset.univ).card = n) ∧ 
      ∀ m, m > 17 → (∀ j ∈ Finset.filter (λ i, weights i > 131) Finset.univ, m = (Finset.card (Finset.filter (λ i, weights i > 131) Finset.univ) + j) → False))
:= sorry

end max_sumo_wrestlers_l246_246489


namespace smallest_non_representable_number_l246_246064

theorem smallest_non_representable_number :
  ∀ n : ℕ, (∀ a b c d : ℕ, n = (2^a - 2^b) / (2^c - 2^d) → n < 11) ∧
           (∀ a b c d : ℕ, 11 ≠ (2^a - 2^b) / (2^c - 2^d)) :=
sorry

end smallest_non_representable_number_l246_246064


namespace arithmetic_seq_a7_l246_246109

structure arith_seq (a : ℕ → ℤ) : Prop :=
  (step : ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d)

theorem arithmetic_seq_a7
  {a : ℕ → ℤ}
  (h_seq : arith_seq a)
  (h1 : a 1 = 2)
  (h2 : a 3 + a 5 = 10)
  : a 7 = 8 :=
by
  sorry

end arithmetic_seq_a7_l246_246109


namespace max_value_of_a_l246_246122

theorem max_value_of_a (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
  (h1 : a < 3 * b) (h2 : b < 2 * c) (h3 : c < 5 * d) (h4 : d < 150) : a ≤ 4460 :=
by
  sorry

end max_value_of_a_l246_246122


namespace smallest_integer_with_18_divisors_l246_246736

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, (∀ m : ℕ, (m > 0 → m ≤ 131072) → 
  (∃ d : ℕ, d = 18 → (associated (factors.count d) (num_factors m)) → m = 131072)) :=
sorry

end smallest_integer_with_18_divisors_l246_246736


namespace product_equality_l246_246147

theorem product_equality : (2.05 * 4.1 = 20.5 * 0.41) :=
by
  sorry

end product_equality_l246_246147


namespace geometric_series_sum_l246_246555

noncomputable def first_term : ℝ := 6
noncomputable def common_ratio : ℝ := -2 / 3

theorem geometric_series_sum :
  (|common_ratio| < 1) → (first_term / (1 - common_ratio) = 18 / 5) :=
by
  intros h
  simp [first_term, common_ratio]
  sorry

end geometric_series_sum_l246_246555


namespace sqrt_ax3_eq_negx_sqrt_ax_l246_246096

variable (a x : ℝ)
variable (ha : a < 0) (hx : x < 0)

theorem sqrt_ax3_eq_negx_sqrt_ax : Real.sqrt (a * x^3) = -x * Real.sqrt (a * x) := by
  sorry

end sqrt_ax3_eq_negx_sqrt_ax_l246_246096


namespace trigonometric_identity_l246_246027

theorem trigonometric_identity :
  (Real.sin (18 * Real.pi / 180) * Real.sin (78 * Real.pi / 180)) -
  (Real.cos (162 * Real.pi / 180) * Real.cos (78 * Real.pi / 180)) = 1 / 2 := by
  sorry

end trigonometric_identity_l246_246027


namespace heptagon_labeling_impossible_l246_246539

/-- 
  Let a heptagon be given with vertices labeled by integers a₁, a₂, a₃, a₄, a₅, a₆, a₇.
  The following two conditions are imposed:
  1. For every pair of consecutive vertices (aᵢ, aᵢ₊₁) (with indices mod 7), 
     at least one of aᵢ and aᵢ₊₁ divides the other.
  2. For every pair of non-consecutive vertices (aᵢ, aⱼ) where i ≠ j ± 1 mod 7, 
     neither aᵢ divides aⱼ nor aⱼ divides aᵢ. 

  Prove that such a labeling is impossible.
-/
theorem heptagon_labeling_impossible :
  ¬ ∃ (a : Fin 7 → ℕ),
    (∀ i : Fin 7, a i ∣ a ((i + 1) % 7) ∨ a ((i + 1) % 7) ∣ a i) ∧
    (∀ {i j : Fin 7}, (i ≠ j + 1 % 7) → (i ≠ j + 6 % 7) → ¬ (a i ∣ a j) ∧ ¬ (a j ∣ a i)) :=
sorry

end heptagon_labeling_impossible_l246_246539


namespace original_prop_and_contrapositive_l246_246236

theorem original_prop_and_contrapositive (m : ℝ) (h : m > 0) : 
  (∀ m : ℝ, m > 0 → ∃ x : ℝ, x^2 + x - m = 0 ∨ ∃ x y : ℝ, x^2 + x - m = 0 ∧ y^2 + y - m = 0) :=
by
  sorry

end original_prop_and_contrapositive_l246_246236


namespace all_ones_l246_246553

theorem all_ones (k : ℕ) (h₁ : k ≥ 2) (n : ℕ → ℕ) (h₂ : ∀ i, 1 ≤ i → i < k → n (i + 1) ∣ (2 ^ n i - 1))
(h₃ : n 1 ∣ (2 ^ n k - 1)) : (∀ i, 1 ≤ i → i ≤ k → n i = 1) :=
by
  sorry

end all_ones_l246_246553


namespace product_102_108_l246_246331

theorem product_102_108 : (102 = 105 - 3) → (108 = 105 + 3) → (102 * 108 = 11016) := by
  sorry

end product_102_108_l246_246331


namespace repeating_decimal_as_fraction_l246_246333

-- Define the repeating decimal 4.25252525... as x
def repeating_decimal : ℚ := 4 + 25 / 99

-- Theorem statement to prove the equivalence
theorem repeating_decimal_as_fraction :
  repeating_decimal = 421 / 99 :=
by
  sorry

end repeating_decimal_as_fraction_l246_246333


namespace four_digit_numbers_thousands_digit_5_div_by_5_l246_246243

theorem four_digit_numbers_thousands_digit_5_div_by_5 :
  ∃ (s : Finset ℕ), (∀ x ∈ s, 5000 ≤ x ∧ x ≤ 5999 ∧ x % 5 = 0) ∧ s.card = 200 :=
by
  sorry

end four_digit_numbers_thousands_digit_5_div_by_5_l246_246243


namespace a_4_is_4_l246_246366

-- Define the general term formula of the sequence
def a (n : ℕ) : ℤ := (-1)^n * n

-- State the desired proof goal
theorem a_4_is_4 : a 4 = 4 :=
by
  -- Proof to be provided here,
  -- adding 'sorry' as we are only defining the statement, not solving it
  sorry

end a_4_is_4_l246_246366


namespace irrational_pi_l246_246162

theorem irrational_pi :
  (¬ ∃ (a b : ℤ), b ≠ 0 ∧ (π = a / b)) :=
sorry

end irrational_pi_l246_246162


namespace adam_initial_money_l246_246318

theorem adam_initial_money :
  let cost_of_airplane := 4.28
  let change_received := 0.72
  cost_of_airplane + change_received = 5.00 :=
by
  sorry

end adam_initial_money_l246_246318


namespace find_largest_C_l246_246994

theorem find_largest_C : 
  ∃ (C : ℝ), 
  (∀ (x y : ℝ), x^2 + y^2 + 2 * x - 4 * y + 10 ≥ C * (x + y + 2)) 
  ∧ (∀ D : ℝ, (∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + 10 ≥ D * (x + y + 2)) → D ≤ C) 
  ∧ C = Real.sqrt 5 :=
sorry

end find_largest_C_l246_246994


namespace integral_of_quadratic_has_minimum_value_l246_246080

theorem integral_of_quadratic_has_minimum_value :
  ∃ m : ℝ, (∀ x : ℝ, x^2 + 2 * x + m ≥ -1) ∧ (∫ x in (1:ℝ)..(2:ℝ), x^2 + 2 * x = (16 / 3:ℝ)) :=
by sorry

end integral_of_quadratic_has_minimum_value_l246_246080


namespace algebraic_expression_value_l246_246596

def algebraic_expression (a b : ℤ) :=
  a + 2 * b + 2 * (a + 2 * b) + 1

theorem algebraic_expression_value :
  algebraic_expression 1 (-1) = -2 :=
by
  -- Proof skipped
  sorry

end algebraic_expression_value_l246_246596


namespace lucky_sum_probability_eq_l246_246398

/--
Given that there are N balls numbered from 1 to N,
where 10 balls are selected in the main draw with their sum being 63,
and 8 balls are selected in the additional draw with their sum being 44,
we need to prove that N = 18 such that the events are equally likely.
-/
theorem lucky_sum_probability_eq (N : ℕ) (h1 : ∃ (S : Finset ℕ), S.card = 10 ∧ S.sum id = 63) 
    (h2 : ∃ (T : Finset ℕ), T.card = 8 ∧ T.sum id = 44) : N = 18 :=
sorry

end lucky_sum_probability_eq_l246_246398


namespace equilateral_triangle_ratio_correct_l246_246734

noncomputable def equilateral_triangle_ratio : ℝ :=
  let side_length := 12
  let perimeter := 3 * side_length
  let altitude := side_length * (Real.sqrt 3 / 2)
  let area := (side_length * altitude) / 2
  area / perimeter

theorem equilateral_triangle_ratio_correct :
  equilateral_triangle_ratio = Real.sqrt 3 :=
sorry

end equilateral_triangle_ratio_correct_l246_246734


namespace angie_age_l246_246003

variables (age : ℕ)

theorem angie_age (h : 2 * age + 4 = 20) : age = 8 :=
sorry

end angie_age_l246_246003


namespace divisor_five_l246_246153

theorem divisor_five {D : ℝ} (h : 95 / D + 23 = 42) : D = 5 := by
  sorry

end divisor_five_l246_246153


namespace tan_eq_860_l246_246638

theorem tan_eq_860 (n : ℤ) (hn : -180 < n ∧ n < 180) : 
  n = -40 ↔ (Real.tan (n * Real.pi / 180) = Real.tan (860 * Real.pi / 180)) := 
sorry

end tan_eq_860_l246_246638


namespace combined_cost_l246_246196

theorem combined_cost (wallet_cost : ℕ) (purse_cost : ℕ)
    (h_wallet_cost : wallet_cost = 22)
    (h_purse_cost : purse_cost = 4 * wallet_cost - 3) :
    wallet_cost + purse_cost = 107 :=
by
  rw [h_wallet_cost, h_purse_cost]
  norm_num
  sorry

end combined_cost_l246_246196


namespace souvenirs_total_cost_l246_246323

theorem souvenirs_total_cost (T : ℝ) (H1 : 347 = T + 146) : T + 347 = 548 :=
by
  -- To ensure the validity of the Lean statement but without the proof.
  sorry

end souvenirs_total_cost_l246_246323


namespace faulty_keys_l246_246965

noncomputable def faulty_digits (typed_sequence : List ℕ) : Set ℕ :=
  { d | d = 7 ∨ d = 9 }

theorem faulty_keys (typed_sequence : List ℕ) (h : typed_sequence.length = 10) :
  (∃ faulty_keys : Set ℕ, ∃ missing_digits : ℕ, missing_digits = 3 ∧ faulty_keys = {7, 9}) :=
sorry

end faulty_keys_l246_246965


namespace sum_geometric_sequence_l246_246833

theorem sum_geometric_sequence {a : ℕ → ℝ} (ha : ∃ q, ∀ n, a n = 3 * q ^ n)
  (h1 : a 1 = 3) (h2 : a 1 + a 2 + a 3 = 9) :
  a 4 + a 5 + a 6 = 9 ∨ a 4 + a 5 + a 6 = -72 :=
sorry

end sum_geometric_sequence_l246_246833


namespace correct_propositions_l246_246320

def line (P: Type) := P → P → Prop  -- A line is a relation between points in a plane

variables (plane1 plane2: Type) -- Define two types representing two planes
variables (P1 P2: plane1) -- Points in plane1
variables (Q1 Q2: plane2) -- Points in plane2

axiom perpendicular_planes : ¬∃ l1 : line plane1, ∀ l2 : line plane2, ¬ (∀ p1 p2, l1 p1 p2 ∧ ∀ q1 q2, l2 q1 q2)

theorem correct_propositions : 3 = 3 := by
  sorry

end correct_propositions_l246_246320


namespace opposite_of_neg_five_l246_246927

theorem opposite_of_neg_five : -(-5) = 5 := by
  sorry

end opposite_of_neg_five_l246_246927


namespace cafeteria_seats_taken_l246_246449

def table1_count : ℕ := 10
def table1_seats : ℕ := 8
def table2_count : ℕ := 5
def table2_seats : ℕ := 12
def table3_count : ℕ := 5
def table3_seats : ℕ := 10
noncomputable def unseated_ratio1 : ℝ := 1/4
noncomputable def unseated_ratio2 : ℝ := 1/3
noncomputable def unseated_ratio3 : ℝ := 1/5

theorem cafeteria_seats_taken : 
  ((table1_count * table1_seats) - (unseated_ratio1 * (table1_count * table1_seats))) + 
  ((table2_count * table2_seats) - (unseated_ratio2 * (table2_count * table2_seats))) + 
  ((table3_count * table3_seats) - (unseated_ratio3 * (table3_count * table3_seats))) = 140 :=
by sorry

end cafeteria_seats_taken_l246_246449


namespace smallest_integer_in_range_l246_246294

theorem smallest_integer_in_range :
  ∃ n : ℕ, 
  1 < n ∧ 
  n % 3 = 2 ∧ 
  n % 5 = 2 ∧ 
  n % 7 = 2 ∧ 
  90 < n ∧ n < 119 :=
sorry

end smallest_integer_in_range_l246_246294


namespace range_of_f_l246_246526

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 6)

theorem range_of_f (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  ∃ y, y = f x ∧ (y ≥ -3 / 2 ∧ y ≤ 3) :=
by {
  sorry
}

end range_of_f_l246_246526


namespace man_l246_246610

theorem man's_speed_downstream (v : ℕ) (h1 : v - 3 = 8) (s : ℕ := 3) : v + s = 14 :=
by
  sorry

end man_l246_246610


namespace intersection_eq_l246_246528

-- Define sets P and Q
def setP := {y : ℝ | ∃ x : ℝ, y = -x^2 + 2}
def setQ := {y : ℝ | ∃ x : ℝ, y = -x + 2}

-- The main theorem statement
theorem intersection_eq: setP ∩ setQ = {y : ℝ | y ≤ 2} :=
by
  sorry

end intersection_eq_l246_246528


namespace plan_y_cost_effective_l246_246618

theorem plan_y_cost_effective (m : ℕ) (h1 : ∀ minutes, cost_plan_x = 15 * minutes)
(h2 : ∀ minutes, cost_plan_y = 3000 + 10 * minutes) :
m ≥ 601 → 3000 + 10 * m < 15 * m :=
by
sorry

end plan_y_cost_effective_l246_246618


namespace prob_insurance_A_or_B_prob_exactly_one_no_insurance_out_of_three_l246_246047

noncomputable def P_A : ℝ := 0.5
noncomputable def P_B_not_A : ℝ := 0.3
noncomputable def P_B : ℝ := 0.6  -- given from solution step
noncomputable def P_C : ℝ := 1 - (1 - P_A) * (1 - P_B)
noncomputable def P_D : ℝ := (1 - P_A) * (1 - P_B)
noncomputable def P_E : ℝ := 3 * P_D * (P_C ^ 2)

theorem prob_insurance_A_or_B :
  P_C = 0.8 :=
by
  sorry

theorem prob_exactly_one_no_insurance_out_of_three :
  P_E = 0.384 :=
by
  sorry

end prob_insurance_A_or_B_prob_exactly_one_no_insurance_out_of_three_l246_246047


namespace focus_of_parabola_l246_246142

theorem focus_of_parabola (m : ℝ) (m_nonzero : m ≠ 0) :
    ∃ (focus_x focus_y : ℝ), (focus_x, focus_y) = (m, 0) ∧
        ∀ (y : ℝ), (x = 1/(4*m) * y^2) := 
sorry

end focus_of_parabola_l246_246142


namespace intersection_points_area_l246_246130

noncomputable def C (x : ℝ) : ℝ := (Real.log x)^2

noncomputable def L (α : ℝ) (x : ℝ) : ℝ :=
  (2 * Real.log α / α) * x - (Real.log α)^2

noncomputable def n (α : ℝ) : ℕ :=
  if α < 1 then 0 else if α = 1 then 1 else 2

noncomputable def S (α : ℝ) : ℝ :=
  2 - 2 * α - (1 / 2) * α * (Real.log α)^2 + 2 * α * Real.log α

theorem intersection_points (α : ℝ) (h : 0 < α) : n α = if α < 1 then 0 else if α = 1 then 1 else 2 := by
  sorry

theorem area (α : ℝ) (h : 0 < α ∧ α < 1) : S α = 2 - 2 * α - (1 / 2) * α * (Real.log α)^2 + 2 * α * Real.log α := by
  sorry

end intersection_points_area_l246_246130


namespace average_of_middle_three_l246_246433

-- Let A be the set of five different positive whole numbers
def avg (lst : List ℕ) : ℚ := (lst.foldl (· + ·) 0 : ℚ) / lst.length
def maximize_diff (lst : List ℕ) : ℕ := lst.maximum' - lst.minimum'

theorem average_of_middle_three 
  (A : List ℕ) 
  (h_diff : maximize_diff A) 
  (h_avg : avg A = 5) 
  (h_len : A.length = 5) 
  (h_distinct : A.nodup) : 
  avg (A.erase_max.erase_min) = 3 := 
sorry

end average_of_middle_three_l246_246433


namespace find_sum_of_digits_l246_246717

theorem find_sum_of_digits (a c : ℕ) (h1 : 200 + 10 * a + 3 + 427 = 600 + 10 * c + 9) (h2 : (600 + 10 * c + 9) % 3 = 0) : a + c = 4 :=
sorry

end find_sum_of_digits_l246_246717


namespace given_eqn_simplification_l246_246217

theorem given_eqn_simplification (x : ℝ) (h : 6 * x^2 - 4 * x - 3 = 0) : 
  (x - 1)^2 + x * (x + 2 / 3) = 2 :=
by
  sorry

end given_eqn_simplification_l246_246217


namespace sin_alpha_of_point_P_l246_246230

theorem sin_alpha_of_point_P (α : ℝ) 
  (h1 : ∃ P : ℝ × ℝ, P = (Real.cos (π / 3), 1) ∧ P = (Real.cos α, Real.sin α) ) :
  Real.sin α = (2 * Real.sqrt 5) / 5 := by
  sorry

end sin_alpha_of_point_P_l246_246230


namespace exact_time_between_9_10_l246_246265

theorem exact_time_between_9_10
  (t : ℝ)
  (h1 : 0 ≤ t ∧ t < 60)
  (h2 : |6 * (t + 5) - (270 + 0.5 * (t - 2))| = 180) :
  t = 10 + 3 / 4 :=
sorry

end exact_time_between_9_10_l246_246265


namespace proof_2_fx_minus_11_eq_f_x_minus_d_l246_246815

def f (x : ℝ) : ℝ := 2 * x - 3
def d : ℝ := 2

theorem proof_2_fx_minus_11_eq_f_x_minus_d :
  2 * (f 5) - 11 = f (5 - d) := by
  sorry

end proof_2_fx_minus_11_eq_f_x_minus_d_l246_246815


namespace opposite_of_negative_five_l246_246930

theorem opposite_of_negative_five : (-(-5) = 5) :=
by
  sorry

end opposite_of_negative_five_l246_246930


namespace number_of_smallest_squares_l246_246173

-- Conditions
def length_cm : ℝ := 28
def width_cm : ℝ := 48
def total_lines_cm : ℝ := 6493.6

-- The main question is about the number of smallest squares
theorem number_of_smallest_squares (d : ℝ) (h_d : d = 0.4) :
  ∃ n : ℕ, n = (length_cm / d - 2) * (width_cm / d - 2) ∧ n = 8024 :=
by
  sorry

end number_of_smallest_squares_l246_246173


namespace fourteen_divisible_by_7_twenty_eight_divisible_by_7_thirty_five_divisible_by_7_forty_nine_divisible_by_7_l246_246810

def is_divisible_by_7 (n: ℕ): Prop := n % 7 = 0

theorem fourteen_divisible_by_7: is_divisible_by_7 14 :=
by
  sorry

theorem twenty_eight_divisible_by_7: is_divisible_by_7 28 :=
by
  sorry

theorem thirty_five_divisible_by_7: is_divisible_by_7 35 :=
by
  sorry

theorem forty_nine_divisible_by_7: is_divisible_by_7 49 :=
by
  sorry

end fourteen_divisible_by_7_twenty_eight_divisible_by_7_thirty_five_divisible_by_7_forty_nine_divisible_by_7_l246_246810


namespace nat_condition_l246_246803

theorem nat_condition (n : ℕ) (h : n ≥ 2) :
  (∀ i j : ℕ, 0 ≤ i → i ≤ j → j ≤ n → (i + j) % 2 = (Nat.choose n i + Nat.choose n j) % 2) ↔
  (∃ p : ℕ, n = 2^p - 2) :=
sorry

end nat_condition_l246_246803


namespace area_of_EFGH_l246_246876

variables (EF FG EH HG EG : ℝ)
variables (distEFGH : EF ≠ HG ∧ EG = 5 ∧ EF^2 + FG^2 = 25 ∧ EH^2 + HG^2 = 25)

theorem area_of_EFGH : 
  ∃ EF FG EH HG : ℕ, EF ≠ HG ∧ EG = 5 
  ∧ EF^2 + FG^2 = 25 
  ∧ EH^2 + HG^2 = 25 
  ∧ EF * FG / 2 + EH * HG / 2 = 12 :=
by { sorry }

end area_of_EFGH_l246_246876


namespace gcd_490_910_l246_246708

theorem gcd_490_910 : Nat.gcd 490 910 = 70 :=
by
  sorry

end gcd_490_910_l246_246708


namespace initial_meals_for_adults_l246_246774

theorem initial_meals_for_adults (C A : ℕ) (h1 : C = 90) (h2 : 14 * C / A = 72) : A = 18 :=
by
  sorry

end initial_meals_for_adults_l246_246774


namespace interest_calculation_l246_246892

theorem interest_calculation (P : ℝ) (r : ℝ) (CI SI : ℝ → ℝ) (n : ℝ) :
  P = 1300 →
  r = 0.10 →
  (CI n - SI n = 13) →
  (CI n = P * (1 + r)^n - P) →
  (SI n = P * r * n) →
  (1.10 ^ n - 1 - 0.10 * n = 0.01) →
  n = 2 :=
by
  intro P_eq r_eq diff_eq CI_def SI_def equation
  -- Sorry, this is just a placeholder. The proof is omitted.
  sorry

end interest_calculation_l246_246892


namespace max_omega_l246_246823

noncomputable def f (ω x : ℝ) : ℝ := Real.sin (ω * x)

theorem max_omega :
  (∃ ω > 0, (∃ k : ℤ, (f ω (2 * π / 3) = 0) ∧ (ω = 3 / 2 * k)) ∧ (0 < ω * π / 14 ∧ ω * π / 14 ≤ π / 2)) →
  ∃ ω, ω = 6 :=
by
  sorry

end max_omega_l246_246823


namespace perpendicular_line_plane_implies_perpendicular_lines_l246_246351

variables {Line Plane : Type}
variables (m n : Line) (α : Plane)

-- Assume inclusion of lines in planes, parallelism, and perpendicularity properties.
variables (parallel : Line → Plane → Prop) (perpendicular : Line → Plane → Prop) (subset : Line → Plane → Prop) (perpendicular_lines : Line → Line → Prop)

-- Given definitions based on the conditions
variable (is_perpendicular : perpendicular m α)
variable (is_subset : subset n α)

-- Prove that m is perpendicular to n
theorem perpendicular_line_plane_implies_perpendicular_lines
  (h1 : perpendicular m α)
  (h2 : subset n α) :
  perpendicular_lines m n :=
sorry

end perpendicular_line_plane_implies_perpendicular_lines_l246_246351


namespace jose_is_12_years_older_l246_246684

theorem jose_is_12_years_older (J M : ℕ) (h1 : M = 14) (h2 : J + M = 40) : J - M = 12 :=
by
  sorry

end jose_is_12_years_older_l246_246684


namespace number_of_primes_in_interval_35_to_44_l246_246714

/--
The number of prime numbers in the interval [35, 44] is 3.
-/
theorem number_of_primes_in_interval_35_to_44 : 
  (Finset.filter Nat.Prime (Finset.Icc 35 44)).card = 3 := 
by
  sorry

end number_of_primes_in_interval_35_to_44_l246_246714


namespace combined_cost_is_107_l246_246193

def wallet_cost : ℕ := 22
def purse_cost (wallet_price : ℕ) : ℕ := 4 * wallet_price - 3
def combined_cost (wallet_price : ℕ) (purse_price : ℕ) : ℕ := wallet_price + purse_price

theorem combined_cost_is_107 : combined_cost wallet_cost (purse_cost wallet_cost) = 107 := 
by 
  -- Proof
  sorry

end combined_cost_is_107_l246_246193


namespace opposite_of_negative_five_l246_246905

theorem opposite_of_negative_five : ∀ x : Int, -5 + x = 0 → x = 5 :=
by
  intros x h
  sorry

end opposite_of_negative_five_l246_246905


namespace arithmetic_sequence_common_difference_l246_246679

theorem arithmetic_sequence_common_difference  (a_n : ℕ → ℝ)
  (h1 : a_n 1 + a_n 6 = 12)
  (h2 : a_n 4 = 7) :
  ∃ d : ℝ, ∀ n : ℕ, a_n n = a_n 1 + (n - 1) * d ∧ d = 2 := 
sorry

end arithmetic_sequence_common_difference_l246_246679


namespace sin_960_eq_sqrt3_over_2_neg_l246_246582

-- Conditions
axiom sine_periodic : ∀ θ, Real.sin (θ + 360 * Real.pi / 180) = Real.sin θ

-- Theorem to prove
theorem sin_960_eq_sqrt3_over_2_neg : Real.sin (960 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  -- skipping the proof
  sorry

end sin_960_eq_sqrt3_over_2_neg_l246_246582


namespace num_bags_of_cookies_l246_246461

theorem num_bags_of_cookies (total_cookies : ℕ) (cookies_per_bag : ℕ) (h1 : total_cookies = 703) (h2 : cookies_per_bag = 19) : total_cookies / cookies_per_bag = 37 :=
by
  sorry

end num_bags_of_cookies_l246_246461


namespace double_meat_sandwich_bread_count_l246_246454

theorem double_meat_sandwich_bread_count (x : ℕ) :
  14 * 2 + 12 * x = 64 → x = 3 := by
  intro h
  sorry

end double_meat_sandwich_bread_count_l246_246454


namespace polynomial_divisible_iff_l246_246514

theorem polynomial_divisible_iff (a b : ℚ) : 
  ((a + b) * 1^5 + (a * b) * 1^2 + 1 = 0) ∧ 
  ((a + b) * 2^5 + (a * b) * 2^2 + 1 = 0) ↔ 
  (a = -1 ∧ b = 31/28) ∨ (a = 31/28 ∧ b = -1) := 
by 
  sorry

end polynomial_divisible_iff_l246_246514


namespace unique_positive_integer_n_l246_246199

-- Definitions based on conditions
def is_divisor (n a : ℕ) : Prop := a % n = 0

def is_perfect_square (m : ℕ) : Prop := ∃ k : ℕ, k * k = m

-- The main theorem statement
theorem unique_positive_integer_n : ∃ (n : ℕ), n > 0 ∧ is_divisor n 1989 ∧
    is_perfect_square (n^2 - 1989 / n) ∧ n = 13 :=
by
  sorry

end unique_positive_integer_n_l246_246199


namespace Ramu_spent_on_repairs_l246_246697

theorem Ramu_spent_on_repairs (purchase_price : ℝ) (selling_price : ℝ) (profit_percent : ℝ) (R : ℝ) 
  (h1 : purchase_price = 42000) 
  (h2 : selling_price = 61900) 
  (h3 : profit_percent = 12.545454545454545) 
  (h4 : selling_price = purchase_price + R + (profit_percent / 100) * (purchase_price + R)) : 
  R = 13000 :=
by
  sorry

end Ramu_spent_on_repairs_l246_246697


namespace simplify_expression_l246_246192

variable {a : ℝ} (h1 : a ≠ -3) (h2 : a ≠ 3) (h3 : a ≠ 2) (h4 : 2 * a + 6 ≠ 0)

theorem simplify_expression : (1 / (a + 3) + 1 / (a ^ 2 - 9)) / ((a - 2) / (2 * a + 6)) = 2 / (a - 3) :=
by
  sorry

end simplify_expression_l246_246192


namespace circleEquation_and_pointOnCircle_l246_246545

-- Definition of the Cartesian coordinate system and the circle conditions
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def inSecondQuadrant (p : ℝ × ℝ) := p.1 < 0 ∧ p.2 > 0

def tangentToLine (C : Circle) (line : ℝ → ℝ) (tangentPoint : ℝ × ℝ) :=
  let centerToLineDistance := (abs (C.center.1 - C.center.2)) / Real.sqrt 2
  C.radius = centerToLineDistance ∧ tangentPoint = (0, 0)

-- Main statements to prove
theorem circleEquation_and_pointOnCircle :
  ∃ C : Circle, ∃ Q : ℝ × ℝ,
    inSecondQuadrant C.center ∧
    C.radius = 2 * Real.sqrt 2 ∧
    tangentToLine C (fun x => x) (0, 0) ∧
    ((∃ p : ℝ × ℝ, p = (-2, 2) ∧ C = Circle.mk p (2 * Real.sqrt 2) ∧
      (∀ x y : ℝ, ((x + 2)^2 + (y - 2)^2 = 8))) ∧
    (Q = (4/5, 12/5) ∧
      ((Q.1 + 2)^2 + (Q.2 - 2)^2 = 8) ∧
      Real.sqrt ((Q.1 - 4)^2 + Q.2^2) = 4))
    := sorry

end circleEquation_and_pointOnCircle_l246_246545


namespace price_of_individual_rose_l246_246279

-- Definitions based on conditions

def price_of_dozen := 36  -- one dozen roses cost $36
def price_of_two_dozen := 50 -- two dozen roses cost $50
def total_money := 680 -- total available money
def total_roses := 317 -- total number of roses that can be purchased

-- Define the question as a theorem
theorem price_of_individual_rose : 
  ∃ (x : ℕ), (12 * (total_money / price_of_two_dozen) + 
              (total_money % price_of_two_dozen) / price_of_dozen * 12 + 
              (total_money % price_of_two_dozen % price_of_dozen) / x = total_roses) ∧ (x = 6) :=
by
  sorry

end price_of_individual_rose_l246_246279


namespace measure_of_angle_C_l246_246787

theorem measure_of_angle_C (C D : ℝ) (h1 : C + D = 360) (h2 : C = 5 * D) : C = 300 := 
by sorry

end measure_of_angle_C_l246_246787


namespace opposite_of_negative_five_l246_246933

theorem opposite_of_negative_five : (-(-5) = 5) :=
by
  sorry

end opposite_of_negative_five_l246_246933


namespace subcommittee_count_l246_246609

theorem subcommittee_count :
  let republicans := 10
  let democrats := 8
  let select_republicans := 4
  let select_democrats := 3
  let num_ways_republicans := Nat.choose republicans select_republicans
  let num_ways_democrats := Nat.choose democrats select_democrats
  let num_ways := num_ways_republicans * num_ways_democrats
  num_ways = 11760 :=
by
  sorry

end subcommittee_count_l246_246609


namespace max_wrestlers_more_than_131_l246_246488

theorem max_wrestlers_more_than_131
  (n : ℤ)
  (total_wrestlers : ℤ := 20)
  (average_weight : ℕ := 125)
  (min_weight : ℕ := 90)
  (constraint1 : n ≥ 0)
  (constraint2 : n ≤ total_wrestlers)
  (total_weight := 2500) :
  n ≤ 17 :=
by
  sorry

end max_wrestlers_more_than_131_l246_246488


namespace ratio_of_cube_volumes_l246_246628

theorem ratio_of_cube_volumes (a b : ℕ) (ha : a = 10) (hb : b = 25) :
  (a^3 : ℚ) / (b^3 : ℚ) = 8 / 125 := by
  sorry

end ratio_of_cube_volumes_l246_246628


namespace derivative_of_odd_function_is_even_l246_246277

theorem derivative_of_odd_function_is_even (f : ℝ → ℝ) (hf : ∀ x, f (-x) = -f x) :
  ∀ x, (deriv f) (-x) = (deriv f) x :=
by
  sorry

end derivative_of_odd_function_is_even_l246_246277


namespace total_steps_to_times_square_l246_246616

-- Define the conditions
def steps_to_rockefeller : ℕ := 354
def steps_to_times_square_from_rockefeller : ℕ := 228

-- State the theorem using the conditions
theorem total_steps_to_times_square : 
  steps_to_rockefeller + steps_to_times_square_from_rockefeller = 582 := 
  by 
    -- We skip the proof for now
    sorry

end total_steps_to_times_square_l246_246616


namespace ratio_of_dimensions_128_l246_246944

noncomputable def volume128 (w l h : ℕ) : Prop := w * l * h = 128

theorem ratio_of_dimensions_128 (w l h : ℕ) (h_volume : volume128 w l h) : 
  ∃ wratio lratio, (w / l = wratio) ∧ (w / h = lratio) :=
sorry

end ratio_of_dimensions_128_l246_246944


namespace positive_difference_between_prob_3_and_prob_5_l246_246592

/-- Probability of a coin landing heads up exactly 3 times out of 5 flips -/
def prob_3_heads : ℚ := (nat.choose 5 3) * (1/2)^3 * (1/2)^(5-3)

/-- Probability of a coin landing heads up exactly 5 times out of 5 flips -/
def prob_5_heads : ℚ := (1/2)^5

/-- Positive difference between the probabilities -/
theorem positive_difference_between_prob_3_and_prob_5 : 
  |prob_3_heads - prob_5_heads| = 9 / 32 :=
by sorry

end positive_difference_between_prob_3_and_prob_5_l246_246592


namespace common_fraction_difference_l246_246491

def repeating_decimal := 23 / 99
def non_repeating_decimal := 23 / 100
def fraction_difference := 23 / 9900

theorem common_fraction_difference : repeating_decimal - non_repeating_decimal = fraction_difference := 
by
  sorry

end common_fraction_difference_l246_246491


namespace combined_cost_l246_246195

theorem combined_cost (wallet_cost : ℕ) (purse_cost : ℕ)
    (h_wallet_cost : wallet_cost = 22)
    (h_purse_cost : purse_cost = 4 * wallet_cost - 3) :
    wallet_cost + purse_cost = 107 :=
by
  rw [h_wallet_cost, h_purse_cost]
  norm_num
  sorry

end combined_cost_l246_246195


namespace value_less_than_mean_by_std_dev_l246_246703

theorem value_less_than_mean_by_std_dev :
  ∀ (mean value std_dev : ℝ), mean = 16.2 → std_dev = 2.3 → value = 11.6 → 
  (mean - value) / std_dev = 2 :=
by
  intros mean value std_dev h_mean h_std_dev h_value
  -- The proof goes here, but per instructions, it is skipped
  -- So we put 'sorry' to indicate that the proof is intentionally left incomplete
  sorry

end value_less_than_mean_by_std_dev_l246_246703


namespace tan_product_identity_l246_246468

-- Lean statement for the mathematical problem
theorem tan_product_identity : 
  (1 + Real.tan (Real.pi / 12)) * (1 + Real.tan (Real.pi / 6)) = 2 := by
  sorry

end tan_product_identity_l246_246468


namespace unique_solution_system_l246_246330

theorem unique_solution_system :
  ∃! (x y z w : ℝ),
    x = Real.sin (z + w + z * w * x) ∧
    y = Real.sin (w + x + w * x * y) ∧
    z = Real.sin (x + y + x * y * z) ∧
    w = Real.sin (y + z + y * z * w) ∧
    Real.cos (x + y + z + w) = 1 :=
begin
  sorry
end

end unique_solution_system_l246_246330


namespace intersection_complement_l246_246826

open Set

variable (U : Set ℕ) (A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5}) (hA : A = {2, 3, 4}) (hB : B = {1, 2})

theorem intersection_complement :
  A ∩ (U \ B) = {3, 4} :=
by
  rw [hU, hA, hB]
  sorry

end intersection_complement_l246_246826


namespace part_I_solution_set_part_II_prove_inequality_l246_246271

-- Definition for part (I)
def f (x: ℝ) := |x - 2|
def g (x: ℝ) := 4 - |x - 1|

-- Theorem for part (I)
theorem part_I_solution_set :
  {x : ℝ | f x ≥ g x} = {x : ℝ | x ≤ -1/2} ∪ {x : ℝ | x ≥ 7/2} :=
by sorry

-- Definition for part (II)
def satisfiable_range (a: ℝ) := {x : ℝ | 0 ≤ x ∧ x ≤ 2}
def density_equation (m n a: ℝ) := (1 / m) + (1 / (2 * n)) = a

-- Theorem for part (II)
theorem part_II_prove_inequality (m n: ℝ) (hm: 0 < m) (hn: 0 < n) 
  (a: ℝ) (h_a: satisfiable_range a = {x : ℝ | abs (x - a) ≤ 1}) (h_density: density_equation m n a) :
  m + 2 * n ≥ 4 :=
by sorry

end part_I_solution_set_part_II_prove_inequality_l246_246271


namespace c_share_l246_246280

theorem c_share (a b c : ℕ) (k : ℕ) 
    (h1 : a + b + c = 1010)
    (h2 : a - 25 = 3 * k) 
    (h3 : b - 10 = 2 * k) 
    (h4 : c - 15 = 5 * k) 
    : c = 495 := 
sorry

end c_share_l246_246280


namespace avg_first_12_even_is_13_l246_246762

-- Definition of the first 12 even numbers
def first_12_even_numbers : List ℕ := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

-- The sum of the first 12 even numbers
def sum_first_12_even_numbers : ℕ := first_12_even_numbers.sum

-- Number of first 12 even numbers
def count_12_even_numbers : ℕ := first_12_even_numbers.length

-- The average of the first 12 even numbers
def average_12_even_numbers : ℕ := sum_first_12_even_numbers / count_12_even_numbers

-- Proof statement that the average of the first 12 even numbers is 13
theorem avg_first_12_even_is_13 : average_12_even_numbers = 13 := by
  sorry

end avg_first_12_even_is_13_l246_246762


namespace red_ball_count_l246_246676

theorem red_ball_count (w : ℕ) (f : ℝ) (total : ℕ) (r : ℕ) 
  (hw : w = 60)
  (hf : f = 0.25)
  (ht : total = w / (1 - f))
  (hr : r = total * f) : 
  r = 20 :=
by 
  -- Lean doesn't require a proof for the problem statement
  sorry

end red_ball_count_l246_246676


namespace cone_to_sphere_surface_area_ratio_l246_246480

noncomputable def sphere_radius (r : ℝ) := r
noncomputable def cone_height (r : ℝ) := 3 * r
noncomputable def side_length_of_triangle (r : ℝ) := 2 * Real.sqrt 3 * r
noncomputable def surface_area_of_sphere (r : ℝ) := 4 * Real.pi * r^2
noncomputable def surface_area_of_cone (r : ℝ) := 9 * Real.pi * r^2
noncomputable def ratio_of_areas (cone_surface : ℝ) (sphere_surface : ℝ) := cone_surface / sphere_surface

theorem cone_to_sphere_surface_area_ratio (r : ℝ) :
    ratio_of_areas (surface_area_of_cone r) (surface_area_of_sphere r) = 9 / 4 := sorry

end cone_to_sphere_surface_area_ratio_l246_246480


namespace solve_k_l246_246240

theorem solve_k (x y k : ℝ) (h1 : x + 2 * y = k - 1) (h2 : 2 * x + y = 5 * k + 4) (h3 : x + y = 5) :
  k = 2 :=
sorry

end solve_k_l246_246240


namespace negation_of_existential_l246_246577

theorem negation_of_existential :
  (¬ ∃ x_0 : ℝ, x_0^2 + 2 * x_0 - 3 > 0) = (∀ x : ℝ, x^2 + 2 * x - 3 ≤ 0) := 
by
  sorry

end negation_of_existential_l246_246577


namespace projection_of_difference_eq_l246_246251

noncomputable def vec_magnitude (v : ℝ × ℝ) : ℝ :=
Real.sqrt (v.1^2 + v.2^2)

noncomputable def vec_dot (v w : ℝ × ℝ) : ℝ :=
v.1 * w.1 + v.2 * w.2

noncomputable def vec_projection (v w : ℝ × ℝ) : ℝ :=
vec_dot (v - w) v / vec_magnitude v

variables (a b : ℝ × ℝ)
  (congruence_cond : vec_magnitude a / vec_magnitude b = Real.cos θ)

theorem projection_of_difference_eq (h : vec_magnitude a / vec_magnitude b = Real.cos θ) :
  vec_projection (a - b) a = (vec_dot a a - vec_dot b b) / vec_magnitude a :=
sorry

end projection_of_difference_eq_l246_246251


namespace smallest_integer_with_18_divisors_l246_246743

theorem smallest_integer_with_18_divisors : 
  ∃ n : ℕ, (n > 0 ∧ (nat.divisors_count n = 18) ∧ (∀ m : ℕ, m > 0 ∧ (nat.divisors_count m = 18) → n ≤ m)) := 
sorry

end smallest_integer_with_18_divisors_l246_246743


namespace greatest_possible_fourth_term_l246_246149

theorem greatest_possible_fourth_term {a d : ℕ} (h : 5 * a + 10 * d = 60) : a + 3 * (12 - a) ≤ 34 :=
by 
  sorry

end greatest_possible_fourth_term_l246_246149


namespace opposite_of_negative_five_l246_246928

theorem opposite_of_negative_five : (-(-5) = 5) :=
by
  sorry

end opposite_of_negative_five_l246_246928


namespace number_of_solutions_5x_plus_10y_eq_50_l246_246584

theorem number_of_solutions_5x_plus_10y_eq_50 : 
  (∃! (n : ℕ), ∃ (xy : ℕ × ℕ), xy.1 + 2 * xy.2 = 10 ∧ n = 6) :=
by
  sorry

end number_of_solutions_5x_plus_10y_eq_50_l246_246584


namespace range_of_a_for_inequality_l246_246065

theorem range_of_a_for_inequality (a : ℝ) : (∀ x : ℝ, x^2 + a * |x| + 1 ≥ 0) ↔ (a ≥ -2) :=
sorry

end range_of_a_for_inequality_l246_246065


namespace certain_number_value_l246_246575

theorem certain_number_value
  (x : ℝ)
  (y : ℝ)
  (h1 : (28 + x + 42 + 78 + y) / 5 = 62)
  (h2 : (48 + 62 + 98 + 124 + x) / 5 = 78) : 
  y = 104 :=
by
  -- Proof goes here
  sorry

end certain_number_value_l246_246575


namespace right_triangle_c_squared_value_l246_246537

theorem right_triangle_c_squared_value (a b c : ℕ) (h : a = 9) (k : b = 12) (right_triangle : True) :
  c^2 = a^2 + b^2 ∨ c^2 = b^2 - a^2 :=
by sorry

end right_triangle_c_squared_value_l246_246537


namespace geometric_series_first_term_l246_246043

theorem geometric_series_first_term (r : ℚ) (S : ℚ) (a : ℚ) (h_r : r = 1 / 4) (h_S : S = 80) (h_sum : S = a / (1 - r)) : a = 60 :=
by {
  sorry
}

end geometric_series_first_term_l246_246043


namespace fraction_equiv_l246_246259

theorem fraction_equiv (x y : ℚ) (h : (5/6) * 192 = (x/y) * 192 + 100) : x/y = 5/16 :=
sorry

end fraction_equiv_l246_246259


namespace angie_age_l246_246002

variables (age : ℕ)

theorem angie_age (h : 2 * age + 4 = 20) : age = 8 :=
sorry

end angie_age_l246_246002


namespace calculate_abs_mul_l246_246191

theorem calculate_abs_mul : |(-3 : ℤ)| * 2 = 6 := 
by 
  -- |(-3)| equals 3 and 3 * 2 equals 6.
  -- The "sorry" is used to complete the statement without proof.
  sorry

end calculate_abs_mul_l246_246191


namespace max_abs_sum_sqrt2_l246_246097

theorem max_abs_sum_sqrt2 (x y : ℝ) (h : x^2 + y^2 = 4) : 
  ∃ (a : ℝ), (a = |x| + |y| ∧ a ≤ 2 * Real.sqrt 2) ∧ 
             ∀ (x y : ℝ), x^2 + y^2 = 4 → (|x| + |y|) ≤ 2 * Real.sqrt 2 :=
sorry

end max_abs_sum_sqrt2_l246_246097


namespace daily_egg_count_per_female_emu_l246_246116

noncomputable def emus_per_pen : ℕ := 6
noncomputable def pens : ℕ := 4
noncomputable def total_eggs_per_week : ℕ := 84

theorem daily_egg_count_per_female_emu :
  (total_eggs_per_week / ((pens * emus_per_pen) / 2 * 7) = 1) :=
by
  sorry

end daily_egg_count_per_female_emu_l246_246116


namespace irrational_sum_floor_eq_iff_l246_246118

theorem irrational_sum_floor_eq_iff (a b c d : ℝ) (h_irr_a : ¬ ∃ (q : ℚ), a = q) 
                                     (h_irr_b : ¬ ∃ (q : ℚ), b = q) 
                                     (h_irr_c : ¬ ∃ (q : ℚ), c = q) 
                                     (h_irr_d : ¬ ∃ (q : ℚ), d = q) 
                                     (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
                                     (h_pos_c : 0 < c) (h_pos_d : 0 < d)
                                     (h_sum_ab : a + b = 1) :
  (c + d = 1) ↔ (∀ (n : ℕ), ⌊n * a⌋ + ⌊n * b⌋ = ⌊n * c⌋ + ⌊n * d⌋) :=
sorry

end irrational_sum_floor_eq_iff_l246_246118


namespace base_5_to_base_10_conversion_l246_246621

/-- An alien creature communicated that it produced 263_5 units of a resource. 
    Convert this quantity to base 10. -/
theorem base_5_to_base_10_conversion : ∀ (n : ℕ), n = 2 * 5^2 + 6 * 5^1 + 3 * 5^0 → n = 83 :=
by
  intros n h
  rw [h]
  sorry

end base_5_to_base_10_conversion_l246_246621


namespace joe_money_left_l246_246114

theorem joe_money_left (initial_money : ℕ) (notebook_count : ℕ) (book_count : ℕ)
    (notebook_price : ℕ) (book_price : ℕ) (total_spent : ℕ) : 
    initial_money = 56 → notebook_count = 7 → book_count = 2 → notebook_price = 4 → book_price = 7 →
    total_spent = notebook_count * notebook_price + book_count * book_price →
    (initial_money - total_spent) = 14 := 
by 
  intro h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end joe_money_left_l246_246114


namespace unique_base_for_final_digit_one_l246_246999

theorem unique_base_for_final_digit_one :
  ∃! b : ℕ, 2 ≤ b ∧ b ≤ 15 ∧ 648 % b = 1 :=
by {
  sorry
}

end unique_base_for_final_digit_one_l246_246999


namespace part1_part2_l246_246515

-- Part (1)
theorem part1 (S_n : ℕ → ℕ) (a_n : ℕ → ℕ) 
  (h1 : ∀ n, S_n n = n * a_n 1 + (n-1) * a_n 2 + 2 * a_n (n-1) + a_n n)
  (arithmetic_seq : ∀ n, a_n (n+1) = a_n n + d)
  (S1_eq : S_n 1 = 5)
  (S2_eq : S_n 2 = 18) :
  ∀ n, a_n n = 3 * n + 2 := by
  sorry

-- Part (2)
theorem part2 (S_n : ℕ → ℕ) (a_n : ℕ → ℕ) 
  (h1 : ∀ n, S_n n = n * a_n 1 + (n-1) * a_n 2 + 2 * a_n (n-1) + a_n n)
  (geometric_seq : ∃ q, ∀ n, a_n (n+1) = q * a_n n)
  (S1_eq : S_n 1 = 3)
  (S2_eq : S_n 2 = 15) :
  ∀ n, S_n n = (3^(n+2) - 6 * n - 9) / 4 := by
  sorry

end part1_part2_l246_246515


namespace total_weight_proof_l246_246721

-- Definitions of the conditions in the problem.
def bags_on_first_trip : ℕ := 10
def common_ratio : ℕ := 2
def number_of_trips : ℕ := 20
def weight_per_bag_kg : ℕ := 50

-- Function to compute the total number of bags transported.
noncomputable def total_number_of_bags : ℕ :=
  bags_on_first_trip * (1 - common_ratio^number_of_trips) / (1 - common_ratio)

-- Function to compute the total weight of onions harvested.
noncomputable def total_weight_of_onions : ℕ :=
  total_number_of_bags * weight_per_bag_kg

-- Theorem stating that the total weight of onions harvested is 524,287,500 kgs.
theorem total_weight_proof : total_weight_of_onions = 524287500 := by
  sorry

end total_weight_proof_l246_246721


namespace find_n_l246_246942

theorem find_n (a n : ℕ) 
  (h1 : a^2 % n = 8) 
  (h2 : a^3 % n = 25) 
  (h3 : n > 25) : 
  n = 113 := 
sorry

end find_n_l246_246942


namespace solve_k_l246_246241

theorem solve_k (x y k : ℝ) (h1 : x + 2 * y = k - 1) (h2 : 2 * x + y = 5 * k + 4) (h3 : x + y = 5) :
  k = 2 :=
sorry

end solve_k_l246_246241


namespace not_possible_perimeter_72_l246_246298

variable (a b : ℕ)
variable (P : ℕ)

def valid_perimeter_range (a b : ℕ) : Set ℕ := 
  { P | ∃ x, 15 < x ∧ x < 35 ∧ P = a + b + x }

theorem not_possible_perimeter_72 :
  (a = 10) → (b = 25) → ¬ (72 ∈ valid_perimeter_range 10 25) := 
by
  sorry

end not_possible_perimeter_72_l246_246298


namespace particle_position_at_2004_seconds_l246_246788

structure ParticleState where
  position : ℕ × ℕ

def initialState : ParticleState :=
  { position := (0, 0) }

def moveParticle (state : ParticleState) (time : ℕ) : ParticleState :=
  if time = 0 then initialState
  else if (time - 1) % 4 < 2 then
    { state with position := (state.position.fst + 1, state.position.snd) }
  else
    { state with position := (state.position.fst, state.position.snd + 1) }

def particlePositionAfterTime (time : ℕ) : ParticleState :=
  (List.range time).foldl moveParticle initialState

/-- The position of the particle after 2004 seconds is (20, 44) -/
theorem particle_position_at_2004_seconds :
  (particlePositionAfterTime 2004).position = (20, 44) :=
  sorry

end particle_position_at_2004_seconds_l246_246788


namespace smallest_positive_integer_with_18_divisors_l246_246741

def has_exactly_divisors (n d : ℕ) : Prop :=
  (finset.filter (λ k, n % k = 0) (finset.range (n + 1))).card = d

theorem smallest_positive_integer_with_18_divisors : ∃ n : ℕ, 0 < n ∧ has_exactly_divisors n 18 ∧ ∀ m : ℕ, 
  0 < m ∧ has_exactly_divisors m 18 → n ≤ m :=
begin
  use 90,
  split,
  { norm_num },
  split,
  { sorry },  -- Proof that 90 has exactly 18 positive divisors
  { intros m hm h_div,
    sorry }  -- Proof that for any m with exactly 18 divisors, m ≥ 90
end

end smallest_positive_integer_with_18_divisors_l246_246741


namespace exist_pair_lcm_gcd_l246_246145

theorem exist_pair_lcm_gcd (a b: ℤ) : 
  ∃ a b : ℤ, Int.lcm a b - Int.gcd a b = 19 := 
sorry

end exist_pair_lcm_gcd_l246_246145


namespace house_orderings_l246_246013

-- Define the houses as elements
inductive House : Type
| Blue
| Yellow
| Green
| Red
| Orange

open House

-- Predicate functions for conditions
def b_before_y (lst : List House) := lst.indexOf Blue < lst.indexOf Yellow
def g_not_next_to_y (lst : List House) := (lst.indexOf Green ≠ lst.indexOf Yellow - 1 ∧ lst.indexOf Green ≠ lst.indexOf Yellow + 1)
def r_before_o (lst : List House) := lst.indexOf Red < lst.indexOf Orange
def b_not_next_to_g (lst : List House) := (lst.indexOf Blue ≠ lst.indexOf Green - 1 ∧ lst.indexOf Blue ≠ lst.indexOf Green + 1)

-- The main statement
theorem house_orderings : 
  Finset.card {
        lst : List House | lst ~ List.enumFrom 0 5 ∧ 
        b_before_y lst ∧ 
        g_not_next_to_y lst ∧ 
        r_before_o lst ∧ 
        b_not_next_to_g lst 
    } = 6 :=
by sorry

end house_orderings_l246_246013


namespace pies_per_day_l246_246613

theorem pies_per_day (daily_pies total_pies : ℕ) (h1 : daily_pies = 8) (h2 : total_pies = 56) :
  total_pies / daily_pies = 7 :=
by sorry

end pies_per_day_l246_246613


namespace largest_x_plus_y_l246_246862

theorem largest_x_plus_y (x y : ℝ) (h1 : 5 * x + 3 * y ≤ 10) (h2 : 3 * x + 6 * y ≤ 12) : x + y ≤ 18 / 7 :=
by
  sorry

end largest_x_plus_y_l246_246862


namespace average_of_middle_three_is_three_l246_246432

theorem average_of_middle_three_is_three :
  ∃ (a b c d e : ℕ), 
    (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧
    (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧
    (c ≠ d) ∧ (c ≠ e) ∧ (d ≠ e) ∧
    (a + b + c + d + e = 25) ∧
    (∃ (min max : ℕ), min = min a b c d e ∧ max = max a b c d e ∧ (max - min) = 14) ∧
    (a ≠ min a b c d e ∧ a ≠ max a b c d e) ∧
    (b ≠ min a b c d e ∧ b ≠ max a b c d e) ∧
    (c ≠ min a b c d e ∧ c ≠ max a b c d e) ∧
    (d ≠ min a b c d e ∧ d ≠ max a b c d e) ∧
    (e ≠ min a b c d e ∧ e ≠ max a b c d e) ∧
    (a + b + c + d + e = 25) → 
    (a ≠ min a b c d e ∧ a ≠ max a b c d e) ∧
    (b ≠ min a b c d e ∧ b ≠ max a b c d e) ∧
    (c ≠ min a b c d e ∧ c ≠ max a b c d e) →  
  ((a + b + c) / 3 = 3) :=
by
  sorry

end average_of_middle_three_is_three_l246_246432


namespace area_bounded_region_l246_246234

theorem area_bounded_region : 
  (∃ x y : ℝ, x^2 + y^2 = 2 * abs (x - y) + 2 * abs (x + y)) →
  (bounded_area : ℝ) = 16 * Real.pi :=
by
  sorry

end area_bounded_region_l246_246234


namespace lowest_score_of_14_scores_l246_246430

theorem lowest_score_of_14_scores (mean_14 : ℝ) (new_mean_12 : ℝ) (highest_score : ℝ) (lowest_score : ℝ) :
  mean_14 = 85 ∧ new_mean_12 = 88 ∧ highest_score = 105 → lowest_score = 29 :=
by
  sorry

end lowest_score_of_14_scores_l246_246430


namespace power_modulo_remainder_l246_246300

theorem power_modulo_remainder :
  (17 ^ 2046) % 23 = 22 := 
sorry

end power_modulo_remainder_l246_246300


namespace x_intercepts_count_l246_246529

def parabola_x (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 3

theorem x_intercepts_count : 
  (∃ y : ℝ, parabola_x y = 0) → 1 := sorry

end x_intercepts_count_l246_246529


namespace cos_alpha_value_l246_246249
open Real

theorem cos_alpha_value (α : ℝ) (h0 : 0 < α ∧ α < π / 2) 
  (h1 : sin (α - π / 6) = 1 / 3) : 
  cos α = (2 * sqrt 6 - 1) / 6 := 
by 
  sorry

end cos_alpha_value_l246_246249


namespace find_value_of_2a_minus_b_l246_246237

def A : Set ℝ := {x | x < 1 ∨ x > 5}
def B (a b : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ b}

theorem find_value_of_2a_minus_b (a b : ℝ) (h1 : A ∪ B a b = Set.univ) (h2 : A ∩ B a b = {x | 5 < x ∧ x ≤ 6}) : 2 * a - b = -4 :=
by
  sorry

end find_value_of_2a_minus_b_l246_246237


namespace number_of_x_intercepts_l246_246531

def parabola (y : ℝ) : ℝ := -3 * y ^ 2 + 2 * y + 3

theorem number_of_x_intercepts : (∃ y : ℝ, parabola y = 0) ∧ (∃! x : ℝ, parabola x = 0) :=
by
  sorry

end number_of_x_intercepts_l246_246531


namespace opposite_of_negative_five_l246_246904

theorem opposite_of_negative_five : ∀ x : Int, -5 + x = 0 → x = 5 :=
by
  intros x h
  sorry

end opposite_of_negative_five_l246_246904


namespace smallest_positive_integer_with_18_divisors_l246_246740

def has_exactly_divisors (n d : ℕ) : Prop :=
  (finset.filter (λ k, n % k = 0) (finset.range (n + 1))).card = d

theorem smallest_positive_integer_with_18_divisors : ∃ n : ℕ, 0 < n ∧ has_exactly_divisors n 18 ∧ ∀ m : ℕ, 
  0 < m ∧ has_exactly_divisors m 18 → n ≤ m :=
begin
  use 90,
  split,
  { norm_num },
  split,
  { sorry },  -- Proof that 90 has exactly 18 positive divisors
  { intros m hm h_div,
    sorry }  -- Proof that for any m with exactly 18 divisors, m ≥ 90
end

end smallest_positive_integer_with_18_divisors_l246_246740


namespace sum_of_ages_l246_246024

theorem sum_of_ages (S F : ℕ) 
  (h1 : F - 18 = 3 * (S - 18)) 
  (h2 : F = 2 * S) : S + F = 108 := by 
  sorry

end sum_of_ages_l246_246024


namespace pie_selling_days_l246_246614

theorem pie_selling_days (daily_pies total_pies : ℕ) (h1 : daily_pies = 8) (h2 : total_pies = 56) :
  total_pies / daily_pies = 7 :=
by {
  rw [h1, h2],
  norm_num,
  sorry
}

end pie_selling_days_l246_246614


namespace max_abs_sum_of_squares_eq_2_sqrt_2_l246_246102

theorem max_abs_sum_of_squares_eq_2_sqrt_2 (x y : ℝ) (h : x^2 + y^2 = 4) : |x| + |y| ≤ 2 * Real.sqrt 2 :=
by
  sorry

end max_abs_sum_of_squares_eq_2_sqrt_2_l246_246102


namespace angies_age_l246_246008

variable (A : ℝ)

theorem angies_age (h : 2 * A + 4 = 20) : A = 8 :=
sorry

end angies_age_l246_246008


namespace power_graph_point_l246_246521

theorem power_graph_point :
  ∀ (m n : ℕ), (m = 2 ∧ n = 3 ∧ 8 = (m - 1) * m^n) → n^(-m) = 1 / 9 :=
by
  intros m n h,
  cases h with hm hn,
  cases hn with hn1 hn2,
  have h1 : 8 = (m - 1) * m ^ n := hn2,
  have h2 : m = 2 := hm,
  have h3 : n = 3 := hn1,
  sorry

end power_graph_point_l246_246521


namespace solve_equation_l246_246303

theorem solve_equation (x : ℝ) (h : x > 0) :
  25^(Real.log x / Real.log 4) - 5^(Real.log (x^2) / Real.log 16 + 1) = Real.log (9 * Real.sqrt 3) / Real.log (Real.sqrt 3) - 25^(Real.log x / Real.log 16) ->
  x = 4 :=
by
  sorry

end solve_equation_l246_246303


namespace number_of_pairs_l246_246578

theorem number_of_pairs : 
  (∀ n m : ℕ, (1 ≤ m ∧ m ≤ 2012) → (5^n < 2^m ∧ 2^m < 2^(m+2) ∧ 2^(m+2) < 5^(n+1))) → 
  (∃ c, c = 279) :=
by
  sorry

end number_of_pairs_l246_246578


namespace sum_first_ten_multiples_of_nine_l246_246019

theorem sum_first_ten_multiples_of_nine :
  let a := 9
  let d := 9
  let n := 10
  let S_n := n * (2 * a + (n - 1) * d) / 2
  S_n = 495 := 
by
  sorry

end sum_first_ten_multiples_of_nine_l246_246019


namespace Rachel_total_books_l246_246569

theorem Rachel_total_books :
  (8 * 15) + (4 * 15) + (3 * 15) + (5 * 15) = 300 :=
by {
  sorry
}

end Rachel_total_books_l246_246569


namespace tan_ratio_l246_246858

theorem tan_ratio (a b : ℝ)
  (h1 : Real.cos (a + b) = 1 / 3)
  (h2 : Real.cos (a - b) = 1 / 2) :
  (Real.tan a) / (Real.tan b) = 5 :=
sorry

end tan_ratio_l246_246858


namespace max_abs_sum_of_squares_eq_2_sqrt_2_l246_246101

theorem max_abs_sum_of_squares_eq_2_sqrt_2 (x y : ℝ) (h : x^2 + y^2 = 4) : |x| + |y| ≤ 2 * Real.sqrt 2 :=
by
  sorry

end max_abs_sum_of_squares_eq_2_sqrt_2_l246_246101


namespace ratio_of_area_to_perimeter_of_equilateral_triangle_l246_246731

def equilateral_triangle (s : ℝ) : Prop :=
  s = 12

theorem ratio_of_area_to_perimeter_of_equilateral_triangle :
  ∀ (s : ℝ), equilateral_triangle s → (let P := 3 * s in
  let A := (sqrt 3 / 4) * s ^ 2 in
  A / P = sqrt 3) :=
by
  intro s hs
  rw [equilateral_triangle, hs]
  let P := 3 * 12
  let A := (sqrt 3 / 4) * 12 ^ 2
  have A_eq : A = 36 * sqrt 3 := by
    calc
      (sqrt 3 / 4) * 12 ^ 2 = (sqrt 3 / 4) * 144  : by norm_num
                      ... = (sqrt 3 * 144) / 4  : by rw div_mul_cancel
                      ... = 36 * sqrt 3         : by norm_num
  have P_eq : P = 36 := by norm_num
  rw [A_eq, P_eq]
  norm_num
  rfl

end ratio_of_area_to_perimeter_of_equilateral_triangle_l246_246731


namespace excess_percentage_l246_246677

theorem excess_percentage (x : ℝ) 
  (L W : ℝ) (hL : L > 0) (hW : W > 0) 
  (h1 : L * (1 + x / 100) * W * 0.96 = L * W * 1.008) : 
  x = 5 :=
by sorry

end excess_percentage_l246_246677


namespace female_members_count_l246_246388

theorem female_members_count (M F : ℕ) (h1 : F = 2 * M) (h2 : F + M = 18) : F = 12 :=
by
  -- the proof will go here
  sorry

end female_members_count_l246_246388


namespace acute_angle_at_7_20_is_100_degrees_l246_246456

theorem acute_angle_at_7_20_is_100_degrees :
  let minute_hand_angle := 4 * 30 -- angle of the minute hand (in degrees)
  let hour_hand_progress := 20 / 60 -- progress of hour hand between 7 and 8
  let hour_hand_angle := 7 * 30 + hour_hand_progress * 30 -- angle of the hour hand (in degrees)

  ∃ angle_acute : ℝ, 
  angle_acute = abs (minute_hand_angle - hour_hand_angle) ∧
  angle_acute = 100 :=
by
  sorry

end acute_angle_at_7_20_is_100_degrees_l246_246456


namespace greatest_whole_number_satisfying_inequality_l246_246062

theorem greatest_whole_number_satisfying_inequality :
  ∃ x : ℕ, (∀ y : ℕ, y < 1 → y ≤ x) ∧ 4 * x - 3 < 2 - x :=
sorry

end greatest_whole_number_satisfying_inequality_l246_246062


namespace sqrt_polynomial_eq_l246_246341

variable (a b c : ℝ)

def polynomial := 16 * a * c + 4 * a^2 - 12 * a * b + 9 * b^2 - 24 * b * c + 16 * c^2

theorem sqrt_polynomial_eq (a b c : ℝ) : 
  (polynomial a b c) ^ (1 / 2) = (2 * a - 3 * b + 4 * c) :=
by
  sorry

end sqrt_polynomial_eq_l246_246341


namespace AndyCoordinatesAfter1500Turns_l246_246184

/-- Definition for Andy's movement rules given his starting position. -/
def AndyPositionAfterTurns (turns : ℕ) : ℤ × ℤ :=
  let rec move (x y : ℤ) (length : ℤ) (dir : ℕ) (remainingTurns : ℕ) : ℤ × ℤ :=
    match remainingTurns with
    | 0 => (x, y)
    | n+1 => 
        let (dx, dy) := match dir % 4 with
                        | 0 => (0, 1)
                        | 1 => (1, 0)
                        | 2 => (0, -1)
                        | _ => (-1, 0)
        move (x + dx * length) (y + dy * length) (length + 1) (dir + 1) n
  move (-30) 25 2 0 turns

theorem AndyCoordinatesAfter1500Turns :
  AndyPositionAfterTurns 1500 = (-280141, 280060) :=
by
  sorry

end AndyCoordinatesAfter1500Turns_l246_246184


namespace sum_reciprocals_seven_l246_246581

variable (x y : ℝ)

theorem sum_reciprocals_seven (h : x + y = 7 * x * y) (hx : x ≠ 0) (hy : y ≠ 0) :
  (1 / x) + (1 / y) = 7 := 
sorry

end sum_reciprocals_seven_l246_246581


namespace percentage_of_sum_is_14_l246_246668

-- Define variables x, y as real numbers
variables (x y P : ℝ)

-- Define condition 1: y is 17.647058823529413% of x
def y_is_percentage_of_x : Prop := y = 0.17647058823529413 * x

-- Define condition 2: 20% of (x - y) is equal to P% of (x + y)
def percentage_equation : Prop := 0.20 * (x - y) = (P / 100) * (x + y)

-- Define the statement to be proved: P is 14
theorem percentage_of_sum_is_14 (h1 : y_is_percentage_of_x x y) (h2 : percentage_equation x y P) : 
  P = 14 :=
by
  sorry

end percentage_of_sum_is_14_l246_246668


namespace shirt_cost_l246_246012

-- Definitions and conditions
def num_ten_bills : ℕ := 2
def num_twenty_bills : ℕ := num_ten_bills + 1

def ten_bill_value : ℕ := 10
def twenty_bill_value : ℕ := 20

-- Statement to prove
theorem shirt_cost :
  (num_ten_bills * ten_bill_value) + (num_twenty_bills * twenty_bill_value) = 80 :=
by
  sorry

end shirt_cost_l246_246012


namespace total_suitcases_l246_246557

-- Definitions based on the conditions in a)
def siblings : ℕ := 4
def suitcases_per_sibling : ℕ := 2
def parents : ℕ := 2
def suitcases_per_parent : ℕ := 3
def suitcases_per_Lily : ℕ := 0

-- The statement to be proved
theorem total_suitcases : (siblings * suitcases_per_sibling) + (parents * suitcases_per_parent) + suitcases_per_Lily = 14 :=
by
  sorry

end total_suitcases_l246_246557


namespace determine_h_l246_246202

noncomputable def h (x : ℝ) : ℝ := -4 * x^5 - 3 * x^3 - 4 * x^2 + 12 * x + 2

theorem determine_h (x : ℝ) :
  4 * x^5 + 5 * x^3 - 3 * x + h x = 2 * x^3 - 4 * x^2 + 9 * x + 2 :=
by
  sorry

end determine_h_l246_246202


namespace total_cost_of_supplies_l246_246039

variable (E P M : ℝ)

open Real

theorem total_cost_of_supplies (h1 : E + 3 * P + 2 * M = 240)
                                (h2 : 2 * E + 4 * M + 5 * P = 440)
                                : 3 * E + 4 * P + 6 * M = 520 := 
sorry

end total_cost_of_supplies_l246_246039


namespace opposite_of_neg5_is_pos5_l246_246914

theorem opposite_of_neg5_is_pos5 : -(-5) = 5 := 
by
  sorry

end opposite_of_neg5_is_pos5_l246_246914


namespace triangle_exists_l246_246054

theorem triangle_exists (x : ℕ) (hx : x > 0) :
  (3 * x + 10 > x * x) ∧ (x * x + 10 > 3 * x) ∧ (x * x + 3 * x > 10) ↔ (x = 3 ∨ x = 4) :=
by
  sorry

end triangle_exists_l246_246054


namespace minimum_value_w_l246_246953

theorem minimum_value_w : ∃ (x y : ℝ), ∀ w, w = 3 * x^2 + 3 * y^2 + 9 * x - 6 * y + 30 → w ≥ 20.25 :=
sorry

end minimum_value_w_l246_246953


namespace no_eight_consecutive_sums_in_circle_l246_246868

theorem no_eight_consecutive_sums_in_circle :
  ¬ ∃ (arrangement : Fin 8 → ℕ) (sums : Fin 8 → ℤ),
      (∀ i, 1 ≤ arrangement i ∧ arrangement i ≤ 8) ∧
      (∀ i, sums i = arrangement i + arrangement (⟨(i + 1) % 8, sorry⟩)) ∧
      (∃ (n : ℤ), 
        (sums 0 = n - 3) ∧ 
        (sums 1 = n - 2) ∧ 
        (sums 2 = n - 1) ∧ 
        (sums 3 = n) ∧ 
        (sums 4 = n + 1) ∧ 
        (sums 5 = n + 2) ∧ 
        (sums 6 = n + 3) ∧ 
        (sums 7 = n + 4)) := 
sorry

end no_eight_consecutive_sums_in_circle_l246_246868


namespace geometric_sequence_a4_a5_l246_246107

open BigOperators

theorem geometric_sequence_a4_a5 (a : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 1 + a 2 = 1)
  (h3 : a 3 + a 4 = 9) : 
  a 4 + a 5 = 27 ∨ a 4 + a 5 = -27 :=
sorry

end geometric_sequence_a4_a5_l246_246107


namespace angies_age_l246_246009

variable (A : ℝ)

theorem angies_age (h : 2 * A + 4 = 20) : A = 8 :=
sorry

end angies_age_l246_246009


namespace find_pairs_l246_246524

noncomputable def f (k : ℤ) (x y : ℝ) : ℝ :=
  if k = 0 then 0 else (x^k + y^k + (-1)^k * (x + y)^k) / k

theorem find_pairs (x y : ℝ) (hxy : x ≠ 0 ∧ y ≠ 0 ∧ x + y ≠ 0) :
  ∃ (m n : ℤ), m ≠ 0 ∧ n ≠ 0 ∧ m ≤ n ∧ m + n ≠ 0 ∧ 
    (∀ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ x + y ≠ 0 → f m x y * f n x y = f (m + n) x y) :=
  sorry

end find_pairs_l246_246524


namespace number_of_squares_l246_246983

def draws_88_lines (lines: ℕ) : Prop := lines = 88
def draws_triangles (triangles: ℕ) : Prop := triangles = 12
def draws_pentagons (pentagons: ℕ) : Prop := pentagons = 4

theorem number_of_squares (triangles pentagons sq_sides: ℕ) (h1: draws_88_lines (triangles * 3 + pentagons * 5 + sq_sides * 4))
    (h2: draws_triangles triangles) (h3: draws_pentagons pentagons) : sq_sides = 8 := by
  sorry

end number_of_squares_l246_246983


namespace evaluate_fraction_l246_246633

theorem evaluate_fraction (a b : ℝ) (h1 : a = 5) (h2 : b = 3) : 3 / (a + b) = 3 / 8 :=
by
  rw [h1, h2]
  sorry

end evaluate_fraction_l246_246633


namespace traditionalist_fraction_l246_246967

theorem traditionalist_fraction (T P : ℕ) 
  (h1 : ∀ prov : ℕ, prov < 6 → T = P / 9) 
  (h2 : P + 6 * T > 0) :
  6 * T / (P + 6 * T) = 2 / 5 := 
by
  sorry

end traditionalist_fraction_l246_246967


namespace sharmila_hourly_wage_l246_246305

-- Sharmila works 10 hours per day on Monday, Wednesday, and Friday.
def hours_worked_mwf : ℕ := 3 * 10

-- Sharmila works 8 hours per day on Tuesday and Thursday.
def hours_worked_tt : ℕ := 2 * 8

-- Total hours worked in a week.
def total_hours_worked : ℕ := hours_worked_mwf + hours_worked_tt

-- Sharmila earns $460 per week.
def weekly_earnings : ℕ := 460

-- Calculate and prove her hourly wage is $10 per hour.
theorem sharmila_hourly_wage : (weekly_earnings / total_hours_worked) = 10 :=
by sorry

end sharmila_hourly_wage_l246_246305


namespace time_for_one_large_division_l246_246446

/-- The clock face is divided into 12 equal parts by the 12 numbers (12 large divisions). -/
def num_large_divisions : ℕ := 12

/-- Each large division is further divided into 5 small divisions. -/
def num_small_divisions_per_large : ℕ := 5

/-- The second hand moves 1 small division every second. -/
def seconds_per_small_division : ℕ := 1

/-- The time it takes for the second hand to move one large division is 5 seconds. -/
def time_per_large_division : ℕ := num_small_divisions_per_large * seconds_per_small_division

theorem time_for_one_large_division : time_per_large_division = 5 := by
  sorry

end time_for_one_large_division_l246_246446


namespace min_additional_cells_l246_246650

-- Definitions based on conditions
def num_cells_shape : Nat := 32
def side_length_square : Nat := 9
def area_square : Nat := side_length_square * side_length_square

-- The statement to prove
theorem min_additional_cells (num_cells_given : Nat := num_cells_shape) 
(side_length : Nat := side_length_square)
(area : Nat := area_square) :
  area - num_cells_given = 49 :=
by
  sorry

end min_additional_cells_l246_246650


namespace cubic_roots_a_b_third_root_l246_246846

theorem cubic_roots_a_b_third_root (a b : ℝ) :
  (∀ x, x^3 + a * x^2 + b * x + 6 = 0 → (x = 2 ∨ x = 3 ∨ x = -1)) →
  a = -4 ∧ b = 1 :=
by
  intro h
  -- We're skipping the proof steps and focusing on definite the goal
  sorry

end cubic_roots_a_b_third_root_l246_246846


namespace opposite_of_neg_five_l246_246898

/-- Definition of the opposite of a number -/
def opposite (a : Int) : Int := -a

theorem opposite_of_neg_five : opposite (-5) = 5 := by
  sorry

end opposite_of_neg_five_l246_246898


namespace prime_count_between_50_and_100_with_prime_remainder_div_12_l246_246368

open Nat

def is_prime (n : ℕ) : Prop := nat.prime n

def primes_between (a b : ℕ) : list ℕ :=
  (list.range' a (b - a + 1)).filter is_prime

def prime_remainders (p : ℕ) : list ℕ := [2, 3, 5, 7, 11]

theorem prime_count_between_50_and_100_with_prime_remainder_div_12 : 
  (primes_between 50 100).filter (λ p, (p % 12) ∈ prime_remainders p).length = 7 :=
by
  sorry

end prime_count_between_50_and_100_with_prime_remainder_div_12_l246_246368


namespace equilateral_triangle_ratio_l246_246733

theorem equilateral_triangle_ratio (s : ℝ) (h_s : s = 12) : 
  (let A := (√3 * s^2) / 4 in let P := 3 * s in A / P = √3) :=
by
  sorry

end equilateral_triangle_ratio_l246_246733


namespace prob_both_correct_l246_246602

def prob_A : ℤ := 70
def prob_B : ℤ := 55
def prob_neither : ℤ := 20

theorem prob_both_correct : (prob_A + prob_B - (100 - prob_neither)) = 45 :=
by
  sorry

end prob_both_correct_l246_246602


namespace number_of_boxes_l246_246450

theorem number_of_boxes (eggs_per_box : ℕ) (total_eggs : ℕ) (h1 : eggs_per_box = 3) (h2 : total_eggs = 6) : total_eggs / eggs_per_box = 2 := by
  sorry

end number_of_boxes_l246_246450


namespace find_value_of_m_l246_246292

theorem find_value_of_m :
  (∃ y : ℝ, y = 20 - (0.5 * -6.7)) →
  (m : ℝ) = 3 * -6.7 + (20 - (0.5 * -6.7)) :=
by {
  sorry
}

end find_value_of_m_l246_246292


namespace determine_m_l246_246658

noncomputable def has_equal_real_roots (m : ℝ) : Prop :=
  m ≠ 0 ∧ (m^2 - 8 * m = 0)

theorem determine_m (m : ℝ) (h : has_equal_real_roots m) : m = 8 :=
  sorry

end determine_m_l246_246658


namespace similar_triangle_side_length_l246_246143

theorem similar_triangle_side_length
  (A_1 A_2 : ℕ)
  (area_diff : A_1 - A_2 = 32)
  (area_ratio : A_1 = 9 * A_2)
  (side_small_triangle : ℕ)
  (side_small_triangle_eq : side_small_triangle = 5)
  (side_ratio : ∃ r : ℕ, r = 3) :
  ∃ side_large_triangle : ℕ, side_large_triangle = side_small_triangle * 3 := by
sorry

end similar_triangle_side_length_l246_246143


namespace no_such_A_exists_l246_246212

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem no_such_A_exists :
  ¬ ∃ A : ℕ, 0 < A ∧ digit_sum A = 16 ∧ digit_sum (2 * A) = 17 :=
by 
  sorry

end no_such_A_exists_l246_246212


namespace total_number_of_coins_is_15_l246_246032

theorem total_number_of_coins_is_15 (x : ℕ) (h : 1*x + 5*x + 10*x + 25*x + 50*x = 273) : 5 * x = 15 :=
by {
  -- Proof omitted
  sorry
}

end total_number_of_coins_is_15_l246_246032


namespace positive_diff_probability_fair_coin_l246_246590

theorem positive_diff_probability_fair_coin :
  let p1 := (Nat.choose 5 3) * (1 / 2)^5
  let p2 := (1 / 2)^5
  p1 - p2 = 9 / 32 :=
by
  sorry

end positive_diff_probability_fair_coin_l246_246590


namespace triangle_angle_sum_l246_246404

open scoped Real

theorem triangle_angle_sum (A B C : ℝ) 
  (hA : A = 25) (hB : B = 55) : C = 100 :=
by
  have h1 : A + B + C = 180 := sorry
  rw [hA, hB] at h1
  linarith

end triangle_angle_sum_l246_246404


namespace smallest_number_satisfies_conditions_l246_246640

-- Define the number we are looking for
def number : ℕ := 391410

theorem smallest_number_satisfies_conditions :
  (number % 7 = 2) ∧
  (number % 11 = 2) ∧
  (number % 13 = 2) ∧
  (number % 17 = 3) ∧
  (number % 23 = 0) ∧
  (number % 5 = 0) :=
by
  -- We need to prove that 391410 satisfies all the given conditions.
  -- This proof will include detailed steps to verify each condition
  sorry

end smallest_number_satisfies_conditions_l246_246640


namespace find_y_intercept_l246_246976

theorem find_y_intercept (m x y b : ℤ) (h_slope : m = 2) (h_point : (x, y) = (259, 520)) :
  y = m * x + b → b = 2 :=
by {
  sorry
}

end find_y_intercept_l246_246976


namespace five_people_six_chairs_l246_246839

theorem five_people_six_chairs : 
  ∃ (f : Fin 6 → Bool), (∑ i, if f i then 1 else 0) = 5 ∧ 
  (∃ (g : Fin 5 → Fin 6), ∀ i j : Fin 5, i ≠ j → g i ≠ g j) →
  (5!) * (choose 6 5) = 720 :=
by
  sorry

end five_people_six_chairs_l246_246839


namespace arithmetic_sequence_ninth_term_l246_246262

-- Definitions and Conditions
variables {a : ℕ → ℝ} (S : ℕ → ℝ)

-- Problem Statement
theorem arithmetic_sequence_ninth_term
  (h1 : a 3 = 4)
  (h2 : S 11 = 110)
  (h3 : ∀ n, S n = (n * (a 1 + a n)) / 2) :
  a 9 = 16 :=
sorry

end arithmetic_sequence_ninth_term_l246_246262


namespace heptagon_isosceles_triangle_same_color_octagon_no_isosceles_triangle_same_color_general_ngon_isosceles_triangle_same_color_l246_246022

namespace PolygonColoring

/-- Define a regular n-gon and its coloring -/
def regular_ngon (n : ℕ) : Type := sorry

def isosceles_triangle {n : ℕ} (p : regular_ngon n) (v1 v2 v3 : ℕ) : Prop := sorry

def same_color {n : ℕ} (p : regular_ngon n) (v1 v2 v3 : ℕ) : Prop := sorry

/-- Part (a) statement -/
theorem heptagon_isosceles_triangle_same_color : 
  ∀ (p : regular_ngon 7), ∃ (v1 v2 v3 : ℕ), isosceles_triangle p v1 v2 v3 ∧ same_color p v1 v2 v3 := 
by
  sorry

/-- Part (b) statement -/
theorem octagon_no_isosceles_triangle_same_color :
  ∃ (p : regular_ngon 8), ¬∃ (v1 v2 v3 : ℕ), isosceles_triangle p v1 v2 v3 ∧ same_color p v1 v2 v3 := 
by
  sorry

/-- Part (c) statement -/
theorem general_ngon_isosceles_triangle_same_color :
  ∀ (n : ℕ), (n = 5 ∨ n = 7 ∨ n ≥ 9) → 
  ∀ (p : regular_ngon n), ∃ (v1 v2 v3 : ℕ), isosceles_triangle p v1 v2 v3 ∧ same_color p v1 v2 v3 := 
by
  sorry

end PolygonColoring

end heptagon_isosceles_triangle_same_color_octagon_no_isosceles_triangle_same_color_general_ngon_isosceles_triangle_same_color_l246_246022


namespace increasing_interval_of_f_l246_246709

noncomputable def f (x : ℝ) : ℝ := x^2 - 6 * x

theorem increasing_interval_of_f :
  ∀ x : ℝ, 3 ≤ x → ∀ y : ℝ, 3 ≤ y → x < y → f x < f y := 
sorry

end increasing_interval_of_f_l246_246709


namespace angie_age_l246_246004

variables (age : ℕ)

theorem angie_age (h : 2 * age + 4 = 20) : age = 8 :=
sorry

end angie_age_l246_246004


namespace unique_expression_values_count_l246_246263

def expression_values : Finset ℤ :=
  (Finset.univ.product Finset.univ).product (Finset.univ.product Finset.univ).product Finset.univ |
  let digits := [1, 2, 3, 4, 5].to_finset in
  digits.bind $ λ a, 
  digits.bind $ λ b, 
  digits.bind $ λ c, 
  digits.bind $ λ d, 
  digits.bind $ λ e,
  if (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
      b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
      c ≠ d ∧ c ≠ e ∧ 
      d ≠ e) then 
    {(a * b - c * d + e : Int)}
  else ∅

theorem unique_expression_values_count : 
  ∃ k : ℕ, k = expression_values.card → k = PICK_AMONG_CHOICES := 
sorry

end unique_expression_values_count_l246_246263


namespace proof_P_and_Q_l246_246711

/-!
Proposition P: The line y=2x is perpendicular to the line x+2y=0.
Proposition Q: The projections of skew lines in the same plane could be parallel lines.
Prove: P ∧ Q is true.
-/

def proposition_P : Prop := 
  let slope1 := 2
  let slope2 := -1 / 2
  slope1 * slope2 = -1

def proposition_Q : Prop :=
  ∃ (a b : ℝ), (∃ (p q r s : ℝ),
    (a * r + b * p = 0) ∧ (a * s + b * q = 0)) ∧
    (a ≠ 0 ∨ b ≠ 0)

theorem proof_P_and_Q : proposition_P ∧ proposition_Q :=
  by
  -- We need to prove the conjunction of both propositions is true.
  sorry

end proof_P_and_Q_l246_246711


namespace min_sum_complementary_events_l246_246373

theorem min_sum_complementary_events (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (hP : (1 / y) + (4 / x) = 1) : x + y ≥ 9 :=
sorry

end min_sum_complementary_events_l246_246373


namespace problem1_problem2_l246_246766

-- Problem 1
theorem problem1 (a b : ℝ) (h : 2 * (a + 1) * (b + 1) = (a + b) * (a + b + 2)) : a^2 + b^2 = 2 := sorry

-- Problem 2
theorem problem2 (a b c : ℝ) (h : a^2 + c^2 = 2 * b^2) : (a + b) * (a + c) + (c + a) * (c + b) = 2 * (b + a) * (b + c) := sorry

end problem1_problem2_l246_246766


namespace calc_length_RS_l246_246496

-- Define the trapezoid properties
def trapezoid (PQRS : Type) (PR QS : ℝ) (h A : ℝ) : Prop :=
  PR = 12 ∧ QS = 20 ∧ h = 10 ∧ A = 180

-- Define the length of the side RS
noncomputable def length_RS (PQRS : Type) (PR QS h A : ℝ) : ℝ :=
  18 - 0.5 * Real.sqrt 44 - 5 * Real.sqrt 3

-- Define the theorem statement
theorem calc_length_RS {PQRS : Type} (PR QS h A : ℝ) :
  trapezoid PQRS PR QS h A → length_RS PQRS PR QS h A = 18 - 0.5 * Real.sqrt 44 - 5 * Real.sqrt 3 :=
by
  intros
  exact Eq.refl (18 - 0.5 * Real.sqrt 44 - 5 * Real.sqrt 3)

end calc_length_RS_l246_246496


namespace opposite_of_negative_five_l246_246936

theorem opposite_of_negative_five : ∃ x : ℤ, -5 + x = 0 ∧ x = 5 :=
by
  use 5
  split
  {
    sorry,
  }
  {
    refl,
  }

end opposite_of_negative_five_l246_246936


namespace ratio_rect_prism_l246_246946

namespace ProofProblem

variables (w l h : ℕ)
def rect_prism (w l h : ℕ) : Prop := w * l * h = 128

theorem ratio_rect_prism (h1 : rect_prism w l h) :
  (w : ℕ) ≠ 0 ∧ (l : ℕ) ≠ 0 ∧ (h : ℕ) ≠ 0 ∧ 
  (∃ k, w = k ∧ l = k ∧ h = 2 * k) :=
sorry

end ProofProblem

end ratio_rect_prism_l246_246946


namespace exists_n_no_rational_solution_l246_246221

noncomputable def quadratic_polynomial (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem exists_n_no_rational_solution {a b c : ℝ} (h : ∃ n : ℕ, ¬∃ x : ℚ, quadratic_polynomial a b c x = (1 : ℝ) / n) :
  ∃ n : ℕ, ¬∃ x : ℚ, quadratic_polynomial a b c x = 1 / n :=
begin
  sorry
end

end exists_n_no_rational_solution_l246_246221


namespace combined_age_l246_246563

variable (m y o : ℕ)

noncomputable def younger_brother_age := 5

noncomputable def older_brother_age_based_on_younger := 3 * younger_brother_age

noncomputable def older_brother_age_based_on_michael (m : ℕ) := 1 + 2 * (m - 1)

theorem combined_age (m y o : ℕ) (h1 : y = younger_brother_age) (h2 : o = older_brother_age_based_on_younger) (h3 : o = older_brother_age_based_on_michael m) :
  y + o + m = 28 := by
  sorry

end combined_age_l246_246563


namespace profit_at_15_is_correct_l246_246969

noncomputable def profit (x : ℝ) : ℝ := (2 * x - 20) * (40 - x)

theorem profit_at_15_is_correct :
  profit 15 = 1250 := by
  sorry

end profit_at_15_is_correct_l246_246969


namespace min_value_of_function_l246_246631

theorem min_value_of_function : 
  ∃ (c : ℝ), (∀ x : ℝ, (x ∈ Set.Icc (Real.pi / 6) (5 * Real.pi / 6)) → (2 * (Real.sin x) ^ 2 + 2 * Real.sin x - 1 / 2) ≥ c) ∧
             (∀ x : ℝ, (x ∈ Set.Icc (Real.pi / 6) (5 * Real.pi / 6)) → (2 * (Real.sin x) ^ 2 + 2 * Real.sin x - 1 / 2 = c) → c = 1) := 
sorry

end min_value_of_function_l246_246631


namespace residue_at_zero_l246_246063

noncomputable def f (z : ℂ) : ℂ := Complex.exp (1 / z^2) * Complex.cos z

theorem residue_at_zero : Complex.residue f 0 = 0 := sorry

end residue_at_zero_l246_246063


namespace common_ratio_of_sequence_l246_246546

variable {a : ℕ → ℝ}
variable {d : ℝ}

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) (n1 n2 n3 : ℕ) : Prop :=
  a n2 = a n1 * r ∧ a n3 = a n1 * r^2

theorem common_ratio_of_sequence {a : ℕ → ℝ} {d : ℝ}
  (h_arith : arithmetic_sequence a d)
  (h_geom : geometric_sequence a ((a 2)/(a 1)) 2 3 6) :
  ((a 3) / (a 2)) = 3 ∨ ((a 3) / (a 2)) = 1 :=
sorry

end common_ratio_of_sequence_l246_246546


namespace triangle_area_eq_e_div_4_l246_246288

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

noncomputable def P : ℝ × ℝ := (1, Real.exp 1)

noncomputable def tangent_line (x : ℝ) : ℝ :=
  let k := (Real.exp 1) * (x + 1)
  k * (x - 1) + Real.exp 1

theorem triangle_area_eq_e_div_4 :
  let area := (1 / 2) * Real.exp 1 * (1 / 2)
  area = (Real.exp 1) / 4 :=
by
  sorry

end triangle_area_eq_e_div_4_l246_246288


namespace sequence_general_term_l246_246352

variable {a : ℕ → ℚ}
variable {S : ℕ → ℚ}

theorem sequence_general_term (h : ∀ n : ℕ, S n = 2 * n - a n) :
  ∀ n : ℕ, a n = (2^n - 1) / (2^(n-1)) :=
by
  sorry

end sequence_general_term_l246_246352


namespace integer_solution_existence_l246_246598

theorem integer_solution_existence : ∃ (x y : ℤ), 2 * x + y - 1 = 0 :=
by
  use 1
  use -1
  sorry

end integer_solution_existence_l246_246598


namespace andrea_needs_1500_sod_squares_l246_246786

-- Define the measurements of the yard sections
def section1_length : ℕ := 30
def section1_width : ℕ := 40
def section2_length : ℕ := 60
def section2_width : ℕ := 80

-- Define the measurements of the sod square
def sod_length : ℕ := 2
def sod_width : ℕ := 2

-- Compute the areas
def area_section1 : ℕ := section1_length * section1_width
def area_section2 : ℕ := section2_length * section2_width
def total_area : ℕ := area_section1 + area_section2

-- Compute the area of one sod square
def area_sod : ℕ := sod_length * sod_width

-- Compute the number of sod squares needed
def num_sod_squares : ℕ := total_area / area_sod

-- Theorem and proof placeholder
theorem andrea_needs_1500_sod_squares : num_sod_squares = 1500 :=
by {
  -- Place proof here
  sorry
}

end andrea_needs_1500_sod_squares_l246_246786


namespace function_domain_l246_246951

noncomputable def domain_function : Set ℝ :=
  {x : ℝ | x ≠ 8}

theorem function_domain :
  ∀ x, x ∈ domain_function ↔ x ∈ (Set.Iio 8 ∪ Set.Ioi 8) := by
  intro x
  sorry

end function_domain_l246_246951


namespace albums_created_l246_246408

def phone_pics : ℕ := 2
def camera_pics : ℕ := 4
def pics_per_album : ℕ := 2
def total_pics : ℕ := phone_pics + camera_pics

theorem albums_created : total_pics / pics_per_album = 3 := by
  sorry

end albums_created_l246_246408


namespace tucker_boxes_l246_246151

def tissues_per_box := 160
def used_tissues := 210
def left_tissues := 270

def total_tissues := used_tissues + left_tissues

theorem tucker_boxes : total_tissues = tissues_per_box * 3 :=
by
  sorry

end tucker_boxes_l246_246151


namespace expand_expression_l246_246504

theorem expand_expression (x : ℝ) : (2 * x - 3) * (2 * x + 3) * (4 * x ^ 2 + 9) = 4 * x ^ 4 - 81 := by
  sorry

end expand_expression_l246_246504


namespace remainder_q_x_plus_2_l246_246861

def q (x : ℝ) (D E F : ℝ) : ℝ := D * x ^ 6 + E * x ^ 4 + F * x ^ 2 + 5

theorem remainder_q_x_plus_2 (D E F : ℝ) (h : q 2 D E F = 13) : q (-2) D E F = 13 :=
by
  sorry

end remainder_q_x_plus_2_l246_246861


namespace isosceles_base_angle_l246_246253

theorem isosceles_base_angle (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A = B ∨ A = C) (h3 : A = 80 ∨ B = 80 ∨ C = 80) : (A = 80 ∧ B = 80) ∨ (A = 80 ∧ C = 80) ∨ (B = 80 ∧ C = 50) ∨ (C = 80 ∧ B = 50) :=
sorry

end isosceles_base_angle_l246_246253


namespace sixth_term_geometric_sequence_l246_246016

theorem sixth_term_geometric_sequence (a r : ℚ) (h_a : a = 16) (h_r : r = 1/2) : 
  a * r^(5) = 1/2 :=
by 
  rw [h_a, h_r]
  sorry

end sixth_term_geometric_sequence_l246_246016


namespace persimmons_count_l246_246245

theorem persimmons_count (x : ℕ) (h : x - 5 = 12) : x = 17 :=
by
  sorry

end persimmons_count_l246_246245


namespace opposite_of_neg_five_l246_246899

/-- Definition of the opposite of a number -/
def opposite (a : Int) : Int := -a

theorem opposite_of_neg_five : opposite (-5) = 5 := by
  sorry

end opposite_of_neg_five_l246_246899


namespace negation_proposition_l246_246576

theorem negation_proposition :
  ¬(∀ x : ℝ, x^2 > x) ↔ ∃ x : ℝ, x^2 ≤ x :=
sorry

end negation_proposition_l246_246576


namespace tan_alpha_eq_7_over_5_l246_246370

theorem tan_alpha_eq_7_over_5
  (α : ℝ)
  (h : Real.tan (α - π / 4) = 1 / 6) :
  Real.tan α = 7 / 5 :=
by
  sorry

end tan_alpha_eq_7_over_5_l246_246370


namespace line_parameterization_l246_246337

theorem line_parameterization (r k : ℝ) (t : ℝ) :
  (∀ x y : ℝ, (x, y) = (r + 3 * t, 2 + k * t) → (y = 2 * x - 5) ) ∧
  (t = 0 → r = 7 / 2) ∧
  (t = 1 → k = 6) :=
by
  sorry

end line_parameterization_l246_246337


namespace find_beta_l246_246085

variables {m n p : ℤ} -- defining variables m, n, p as integers
variables {α β : ℤ} -- defining roots α and β as integers

theorem find_beta (h1: α = 3)
  (h2: ∀ x, x^2 - (m+n)*x + (m*n - p) = 0) -- defining the quadratic equation
  (h3: α + β = m + n)
  (h4: α * β = m * n - p)
  (h5: m ≠ n) (h6: n ≠ p) (h7: m ≠ p) : -- ensuring m, n, and p are distinct
  β = m + n - 3 := sorry

end find_beta_l246_246085


namespace sin_2alpha_plus_sin_squared_l246_246075

theorem sin_2alpha_plus_sin_squared (α : ℝ) (h : Real.tan α = 1 / 2) : Real.sin (2 * α) + Real.sin α ^ 2 = 1 :=
sorry

end sin_2alpha_plus_sin_squared_l246_246075


namespace solve_equation_l246_246500

variable (a b c : ℝ)

theorem solve_equation (h : (a / Real.sqrt (18 * b)) * (c / Real.sqrt (72 * b)) = 1) : 
  a * c = 36 * b :=
by 
  -- Proof goes here
  sorry

end solve_equation_l246_246500


namespace opposite_of_neg_five_l246_246924

theorem opposite_of_neg_five : -(-5) = 5 := by
  sorry

end opposite_of_neg_five_l246_246924


namespace trajectory_of_P_l246_246073

-- Define points F1 and F2
def F1 : ℝ × ℝ := (-4, 0)
def F2 : ℝ × ℝ := (4, 0)

-- Define the condition |PF2| - |PF1| = 4 for a moving point P
def condition (P : ℝ × ℝ) : Prop :=
  let PF1 := Real.sqrt ((P.1 + 4)^2 + P.2^2)
  let PF2 := Real.sqrt ((P.1 - 4)^2 + P.2^2)
  abs (PF2 - PF1) = 4

-- The target equation of the trajectory
def target_eq (P : ℝ × ℝ) : Prop :=
  P.1 * P.1 / 4 - P.2 * P.2 / 12 = 1 ∧ P.1 ≤ -2

theorem trajectory_of_P :
  ∀ P : ℝ × ℝ, condition P → target_eq P := by
  sorry

end trajectory_of_P_l246_246073


namespace total_number_of_toys_l246_246203

theorem total_number_of_toys (average_cost_Dhoni_toys : ℕ) (number_Dhoni_toys : ℕ) 
    (price_David_toy : ℕ) (new_avg_cost : ℕ) 
    (h1 : average_cost_Dhoni_toys = 10) (h2 : number_Dhoni_toys = 5) 
    (h3 : price_David_toy = 16) (h4 : new_avg_cost = 11) : 
    (number_Dhoni_toys + 1) = 6 := 
by
  sorry

end total_number_of_toys_l246_246203


namespace joe_money_left_l246_246113

theorem joe_money_left (starting_amount : ℕ) (num_notebooks : ℕ) (cost_per_notebook : ℕ) (num_books : ℕ) (cost_per_book : ℕ)
  (h_starting_amount : starting_amount = 56)
  (h_num_notebooks : num_notebooks = 7)
  (h_cost_per_notebook : cost_per_notebook = 4)
  (h_num_books : num_books = 2)
  (h_cost_per_book : cost_per_book = 7) : 
  starting_amount - (num_notebooks * cost_per_notebook + num_books * cost_per_book) = 14 :=
by
  rw [h_starting_amount, h_num_notebooks, h_cost_per_notebook, h_num_books, h_cost_per_book]
  -- sorry lõpetab ajutiselt
  norm_num  
  -- sorry 

end joe_money_left_l246_246113


namespace domestic_probability_short_haul_probability_long_haul_probability_l246_246471

variable (P_internet_domestic P_snacks_domestic P_entertainment_domestic P_legroom_domestic : ℝ)
variable (P_internet_short_haul P_snacks_short_haul P_entertainment_short_haul P_legroom_short_haul : ℝ)
variable (P_internet_long_haul P_snacks_long_haul P_entertainment_long_haul P_legroom_long_haul : ℝ)

noncomputable def P_domestic :=
  P_internet_domestic * P_snacks_domestic * P_entertainment_domestic * P_legroom_domestic

theorem domestic_probability :
  P_domestic 0.40 0.60 0.70 0.50 = 0.084 := by
  sorry

noncomputable def P_short_haul :=
  P_internet_short_haul * P_snacks_short_haul * P_entertainment_short_haul * P_legroom_short_haul

theorem short_haul_probability :
  P_short_haul 0.50 0.75 0.55 0.60 = 0.12375 := by
  sorry

noncomputable def P_long_haul :=
  P_internet_long_haul * P_snacks_long_haul * P_entertainment_long_haul * P_legroom_long_haul

theorem long_haul_probability :
  P_long_haul 0.65 0.80 0.75 0.70 = 0.273 := by
  sorry

end domestic_probability_short_haul_probability_long_haul_probability_l246_246471


namespace valid_punching_settings_l246_246442

theorem valid_punching_settings :
  let total_patterns := 2^9
  let symmetric_patterns := 2^6
  total_patterns - symmetric_patterns = 448 :=
by
  sorry

end valid_punching_settings_l246_246442


namespace max_terms_in_arithmetic_seq_l246_246979

variable (a n : ℝ)

def arithmetic_seq_max_terms (a n : ℝ) : Prop :=
  let d := 4
  a^2 + (n - 1) * (a + d) + (n - 1) * n / 2 * d ≤ 100

theorem max_terms_in_arithmetic_seq (a n : ℝ) (h : arithmetic_seq_max_terms a n) : n ≤ 8 :=
sorry

end max_terms_in_arithmetic_seq_l246_246979


namespace range_of_a_l246_246362

def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 2

theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, -1 ≤ x → f a x ≥ a) ↔ (-3 ≤ a ∧ a ≤ 1) :=
by sorry

end range_of_a_l246_246362


namespace sequence_number_theorem_l246_246867

def seq_count (n k : ℕ) : ℕ :=
  -- Sequence count function definition given the conditions.
  sorry -- placeholder, as we are only defining the statement, not the function itself.

theorem sequence_number_theorem (n k : ℕ) : seq_count n k = Nat.choose (n-1) k :=
by
  -- This is where the proof would go, currently omitted.
  sorry

end sequence_number_theorem_l246_246867


namespace opposite_of_neg_five_l246_246902

/-- Definition of the opposite of a number -/
def opposite (a : Int) : Int := -a

theorem opposite_of_neg_five : opposite (-5) = 5 := by
  sorry

end opposite_of_neg_five_l246_246902


namespace lucky_sum_probability_eq_l246_246399

/--
Given that there are N balls numbered from 1 to N,
where 10 balls are selected in the main draw with their sum being 63,
and 8 balls are selected in the additional draw with their sum being 44,
we need to prove that N = 18 such that the events are equally likely.
-/
theorem lucky_sum_probability_eq (N : ℕ) (h1 : ∃ (S : Finset ℕ), S.card = 10 ∧ S.sum id = 63) 
    (h2 : ∃ (T : Finset ℕ), T.card = 8 ∧ T.sum id = 44) : N = 18 :=
sorry

end lucky_sum_probability_eq_l246_246399


namespace inscribed_sphere_radius_l246_246355

theorem inscribed_sphere_radius 
  (V : ℝ) (S1 S2 S3 S4 : ℝ) (R : ℝ) :
  (1 / 3) * R * (S1 + S2 + S3 + S4) = V ↔ R = (3 * V) / (S1 + S2 + S3 + S4) := 
by
  sorry

end inscribed_sphere_radius_l246_246355


namespace minimize_distances_is_k5_l246_246572

-- Define the coordinates of points A, B, and D
def A : ℝ × ℝ := (4, 3)
def B : ℝ × ℝ := (1, 2)
def D : ℝ × ℝ := (0, 5)

-- Define C as a point vertically below D, implying the x-coordinate is the same as that of D and y = k
def C (k : ℝ) : ℝ × ℝ := (0, k)

-- Prove that the value of k that minimizes the distances over AC and BC is k = 5
theorem minimize_distances_is_k5 : ∃ k : ℝ, (C k = (0, 5)) ∧ k = 5 :=
by {
  sorry
}

end minimize_distances_is_k5_l246_246572


namespace determine_g_2023_l246_246414

noncomputable def g (x : ℕ) : ℝ := sorry

axiom g_pos (x : ℕ) (hx : x > 0) : g x > 0

axiom g_property (x y : ℕ) (h1 : x > 2 * y) (h2 : 0 < y) : 
  g (x - y) = Real.sqrt (g (x / y) + 3)

theorem determine_g_2023 : g 2023 = (1 + Real.sqrt 13) / 2 :=
by
  sorry

end determine_g_2023_l246_246414


namespace point_B_between_A_and_C_l246_246501

theorem point_B_between_A_and_C (a b c : ℚ) (h_abc : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : |a - b| + |b - c| = |a - c|) : 
  (a < b ∧ b < c) ∨ (c < b ∧ b < a) :=
sorry

end point_B_between_A_and_C_l246_246501


namespace friend_spent_13_50_l246_246599

noncomputable def amount_you_spent : ℝ := 
  let x := (22 - 5) / 2
  x

noncomputable def amount_friend_spent (x : ℝ) : ℝ := 
  x + 5

theorem friend_spent_13_50 :
  ∃ x : ℝ, (x + (x + 5) = 22) ∧ (x + 5 = 13.5) :=
by
  sorry

end friend_spent_13_50_l246_246599


namespace min_value_of_a_plus_b_l246_246859

theorem min_value_of_a_plus_b (a b : ℤ) (h_ab : a * b = 72) (h_even : a % 2 = 0) : a + b ≥ -38 :=
sorry

end min_value_of_a_plus_b_l246_246859


namespace gcd_of_gx_x_is_one_l246_246358

noncomputable def gcd_gx_x (x : ℤ) : ℤ :=
  let g := (3 * x + 4) * (8 * x + 5) * (15 * x + 9) * (x + 15)
  gcd g x

theorem gcd_of_gx_x_is_one (x : ℤ) (h : ∃ k : ℤ, x = 37521 * k) : gcd_gx_x x = 1 :=
by
  sorry

end gcd_of_gx_x_is_one_l246_246358


namespace smallest_int_with_18_divisors_l246_246751

theorem smallest_int_with_18_divisors : ∃ n : ℕ, (∀ d : ℕ, 0 < d ∧ d ≤ n → d = 288) ∧ (∃ a1 a2 a3 : ℕ, a1 + 1 * a2 + 1 * a3 + 1 = 18) := 
by 
  sorry

end smallest_int_with_18_divisors_l246_246751


namespace three_digit_condition_l246_246343

-- Define the three-digit number and its rotated variants
def num (a b c : ℕ) := 100 * a + 10 * b + c
def num_bca (a b c : ℕ) := 100 * b + 10 * c + a
def num_cab (a b c : ℕ) := 100 * c + 10 * a + b

-- The main statement to prove
theorem three_digit_condition (a b c: ℕ) (h_a : 1 ≤ a ∧ a ≤ 9) (h_b : 0 ≤ b ∧ b ≤ 9) (h_c : 0 ≤ c ∧ c ≤ 9) :
  2 * num a b c = num_bca a b c + num_cab a b c ↔ 
  (num a b c = 111 ∨ num a b c = 222 ∨ 
  num a b c = 333 ∨ num a b c = 370 ∨ 
  num a b c = 407 ∨ num a b c = 444 ∨ 
  num a b c = 481 ∨ num a b c = 518 ∨ 
  num a b c = 555 ∨ num a b c = 592 ∨ 
  num a b c = 629 ∨ num a b c = 666 ∨ 
  num a b c = 777 ∨ num a b c = 888 ∨ 
  num a b c = 999) := by
  sorry

end three_digit_condition_l246_246343


namespace range_of_a_for_three_tangents_curve_through_point_l246_246256

noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := x^3 + 3 * x^2 + a * x + a - 2

noncomputable def curve_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 6 * x + a

theorem range_of_a_for_three_tangents_curve_through_point :
  ∀ (a : ℝ), (∀ x0 : ℝ, 2 * x0^3 + 3 * x0^2 + 4 - a = 0 → 
    ((2 * -1^3 + 3 * -1^2 + 4 - a > 0) ∧ (2 * 0^3 + 3 * 0^2 + 4 - a < 0))) ↔ (4 < a ∧ a < 5) :=
by
  sorry

end range_of_a_for_three_tangents_curve_through_point_l246_246256


namespace work_completion_l246_246757

theorem work_completion (a b : ℕ) (hab : a = 2 * b) (hwork_together : (1/a + 1/b) = 1/8) : b = 24 := by
  sorry

end work_completion_l246_246757


namespace negation_of_squared_inequality_l246_246379

theorem negation_of_squared_inequality (p : ∀ n : ℕ, n^2 ≤ 2*n + 5) : 
  ∃ n : ℕ, n^2 > 2*n + 5 :=
sorry

end negation_of_squared_inequality_l246_246379


namespace tina_work_time_l246_246297

theorem tina_work_time (T : ℕ) (h1 : ∀ Ann_hours, Ann_hours = 9)
                       (h2 : ∀ Tina_worked_hours, Tina_worked_hours = 8)
                       (h3 : ∀ Ann_worked_hours, Ann_worked_hours = 3)
                       (h4 : (8 : ℚ) / T + (1 : ℚ) / 3 = 1) : T = 12 :=
by
  sorry

end tina_work_time_l246_246297


namespace solve_for_x_l246_246284

theorem solve_for_x (x : ℤ) (h : 45 - (5 * 3) = x + 7) : x = 23 := 
by
  sorry

end solve_for_x_l246_246284


namespace pie_piece_cost_l246_246567

theorem pie_piece_cost (pieces_per_pie : ℕ) (pies_per_hour : ℕ) (total_earnings : ℝ) :
  pieces_per_pie = 3 → pies_per_hour = 12 → total_earnings = 138 →
  (total_earnings / (pieces_per_pie * pies_per_hour)) = 3.83 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end pie_piece_cost_l246_246567


namespace ratio_of_dimensions_128_l246_246945

noncomputable def volume128 (w l h : ℕ) : Prop := w * l * h = 128

theorem ratio_of_dimensions_128 (w l h : ℕ) (h_volume : volume128 w l h) : 
  ∃ wratio lratio, (w / l = wratio) ∧ (w / h = lratio) :=
sorry

end ratio_of_dimensions_128_l246_246945


namespace top_layer_lamps_l246_246678

theorem top_layer_lamps (a : ℕ) :
  (a + 2 * a + 4 * a + 8 * a + 16 * a + 32 * a + 64 * a = 381) → a = 3 := 
by
  intro h
  sorry

end top_layer_lamps_l246_246678


namespace log_ab_eq_l246_246885

-- Definition and conditions
variables (a b x : ℝ)
variables (ha : 0 < a) (hb : 0 < b) (hx : 0 < x)

-- The theorem to prove
theorem log_ab_eq (a b x : ℝ) (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) :
  Real.log (x) / Real.log (a * b) = (Real.log (x) / Real.log (a)) * (Real.log (x) / Real.log (b)) / ((Real.log (x) / Real.log (a)) + (Real.log (x) / Real.log (b))) :=
sorry

end log_ab_eq_l246_246885


namespace solve_inequality_a_eq_2_solve_inequality_a_in_R_l246_246986

theorem solve_inequality_a_eq_2 :
  {x : ℝ | x > 2 ∨ x < 1} = {x : ℝ | x^2 - 3*x + 2 > 0} :=
sorry

theorem solve_inequality_a_in_R (a : ℝ) :
  {x : ℝ | 
    (a > 1 ∧ (x > a ∨ x < 1)) ∨ 
    (a = 1 ∧ x ≠ 1) ∨ 
    (a < 1 ∧ (x > 1 ∨ x < a))
  } = 
  {x : ℝ | x^2 - (1 + a)*x + a > 0} :=
sorry

end solve_inequality_a_eq_2_solve_inequality_a_in_R_l246_246986


namespace min_value_x_squared_plus_y_squared_l246_246818

theorem min_value_x_squared_plus_y_squared {x y : ℝ} 
  (h : x^2 + y^2 - 4*x - 6*y + 12 = 0) : 
  ∃ m : ℝ, m = 14 - 2 * Real.sqrt 13 ∧ ∀ u v : ℝ, (u^2 + v^2 - 4*u - 6*v + 12 = 0) → (u^2 + v^2 ≥ m) :=
by
  sorry

end min_value_x_squared_plus_y_squared_l246_246818


namespace remainder_when_divided_by_8_l246_246459

theorem remainder_when_divided_by_8 (x : ℤ) (k : ℤ) (h : x = 72 * k + 19) : x % 8 = 3 :=
by sorry

end remainder_when_divided_by_8_l246_246459


namespace number_of_x_intercepts_l246_246532

def parabola (y : ℝ) : ℝ := -3 * y ^ 2 + 2 * y + 3

theorem number_of_x_intercepts : (∃ y : ℝ, parabola y = 0) ∧ (∃! x : ℝ, parabola x = 0) :=
by
  sorry

end number_of_x_intercepts_l246_246532


namespace complex_number_division_l246_246998

theorem complex_number_division (i : ℂ) (h_i : i^2 = -1) :
  2 / (i * (3 - i)) = (1 - 3 * i) / 5 :=
by
  sorry

end complex_number_division_l246_246998


namespace Mike_profit_l246_246273

def total_cost (acres : ℕ) (cost_per_acre : ℕ) : ℕ :=
  acres * cost_per_acre

def revenue (acres_sold : ℕ) (price_per_acre : ℕ) : ℕ :=
  acres_sold * price_per_acre

def profit (revenue : ℕ) (cost : ℕ) : ℕ :=
  revenue - cost

theorem Mike_profit :
  let acres := 200
  let cost_per_acre := 70
  let acres_sold := acres / 2
  let price_per_acre := 200
  let cost := total_cost acres cost_per_acre
  let rev := revenue acres_sold price_per_acre
  profit rev cost = 6000 :=
by
  sorry

end Mike_profit_l246_246273


namespace base_problem_l246_246441

theorem base_problem (c d : Nat) (pos_c : c > 0) (pos_d : d > 0) (h : 5 * c + 8 = 8 * d + 5) : c + d = 15 :=
sorry

end base_problem_l246_246441


namespace problem1_problem2_problem3_problem4_l246_246791

-- Problem (1)
theorem problem1 : 6 - -2 + -4 - 3 = 1 :=
by sorry

-- Problem (2)
theorem problem2 : 8 / -2 * (1 / 3 : ℝ) * (-(1 + 1/2: ℝ)) = 2 :=
by sorry

-- Problem (3)
theorem problem3 : (13 + (2 / 7 - 1 / 14) * 56) / (-1 / 4) = -100 :=
by sorry

-- Problem (4)
theorem problem4 : 
  |-(5 / 6 : ℝ)| / ((-(3 + 1 / 5: ℝ)) / (-4)^2 + (-7 / 4) * (4 / 7)) = -(25 / 36) :=
by sorry

end problem1_problem2_problem3_problem4_l246_246791


namespace smallest_integer_with_18_divisors_l246_246739

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, has_exactly_18_divisors n ∧ (∀ m : ℕ, has_exactly_18_divisors m → n ≤ m) ∧ n = 540 := sorry

end smallest_integer_with_18_divisors_l246_246739


namespace how_many_more_choc_chip_cookies_l246_246664

-- Define the given conditions
def choc_chip_cookies_yesterday := 19
def raisin_cookies_this_morning := 231
def choc_chip_cookies_this_morning := 237

-- Define the total chocolate chip cookies
def total_choc_chip_cookies : ℕ := choc_chip_cookies_this_morning + choc_chip_cookies_yesterday

-- Define the proof statement
theorem how_many_more_choc_chip_cookies :
  total_choc_chip_cookies - raisin_cookies_this_morning = 25 :=
by
  -- Proof will go here
  sorry

end how_many_more_choc_chip_cookies_l246_246664


namespace students_per_table_correct_l246_246475

-- Define the number of tables and students
def num_tables := 34
def num_students := 204

-- Define x as the number of students per table
def students_per_table := 6

-- State the theorem
theorem students_per_table_correct : num_students / num_tables = students_per_table :=
by
  sorry

end students_per_table_correct_l246_246475


namespace sum_of_numbers_l246_246150

variable (x y S : ℝ)
variable (H1 : x + y = S)
variable (H2 : x * y = 375)
variable (H3 : (1 / x) + (1 / y) = 0.10666666666666667)

theorem sum_of_numbers (H1 : x + y = S) (H2 : x * y = 375) (H3 : (1 / x) + (1 / y) = 0.10666666666666667) : S = 40 :=
by {
  sorry
}

end sum_of_numbers_l246_246150


namespace sum_of_digits_succ_2080_l246_246120

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_succ_2080 (m : ℕ) (h : sum_of_digits m = 2080) :
  sum_of_digits (m + 1) = 2081 ∨ sum_of_digits (m + 1) = 2090 :=
sorry

end sum_of_digits_succ_2080_l246_246120


namespace simplify_exponents_l246_246283

variable (x : ℝ)

theorem simplify_exponents (x : ℝ) : (x^5) * (x^2) = x^(7) :=
by
  sorry

end simplify_exponents_l246_246283


namespace polyhedron_euler_formula_l246_246829

variable (A F S : ℕ)
variable (closed_polyhedron : Prop)

theorem polyhedron_euler_formula (h : closed_polyhedron) : A + 2 = F + S := sorry

end polyhedron_euler_formula_l246_246829


namespace exists_nat_no_rational_solution_l246_246222

theorem exists_nat_no_rational_solution (p : ℝ → ℝ) (hp : ∃ a b c : ℝ, ∀ x, p x = a*x^2 + b*x + c) :
  ∃ n : ℕ, ∀ q : ℚ, p q ≠ 1 / (n : ℝ) :=
by
  sorry

end exists_nat_no_rational_solution_l246_246222


namespace range_of_a_l246_246651

-- Define the conditions and the problem
def neg_p (x : ℝ) : Prop := -3 < x ∧ x < 0
def neg_q (x : ℝ) (a : ℝ) : Prop := x > a
def p (x : ℝ) : Prop := x ≤ -3 ∨ x ≥ 0
def q (x : ℝ) (a : ℝ) : Prop := x ≤ a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, neg_p x → ¬ p x) ∧
  (∀ x : ℝ, neg_q x a → ¬ q x a) ∧
  (∀ x : ℝ, q x a → p x) ∧
  (∃ x : ℝ, ¬ (q x a → p x)) →
  a ≤ -3 :=
by
  sorry

end range_of_a_l246_246651


namespace arithmetic_sequence_6000th_term_l246_246707

theorem arithmetic_sequence_6000th_term :
  ∀ (p r : ℕ), 
  (2 * p) = 2 * p → 
  (2 * p + 2 * r = 14) → 
  (14 + 2 * r = 4 * p - r) → 
  (2 * p + (6000 - 1) * 4 = 24006) :=
by 
  intros p r h h1 h2
  sorry

end arithmetic_sequence_6000th_term_l246_246707


namespace biography_percentage_increase_l246_246270

variable {T : ℝ}
variable (hT : T > 0 ∧ T ≤ 10000)
variable (B : ℝ := 0.20 * T)
variable (B' : ℝ := 0.32 * T)
variable (percentage_increase : ℝ := ((B' - B) / B) * 100)

theorem biography_percentage_increase :
  percentage_increase = 60 :=
by
  sorry

end biography_percentage_increase_l246_246270


namespace part_one_part_two_l246_246354

-- Defining the sequence {a_n} with the sum of the first n terms.
def S (n : ℕ) : ℕ := 3 * n ^ 2 + 10 * n

-- Defining a_n in terms of the sum S_n
def a (n : ℕ) : ℕ :=
  if n = 1 then S 1 else S n - S (n - 1)

-- Defining the arithmetic sequence {b_n}
def b (n : ℕ) : ℕ := 3 * n + 2

-- Defining the sequence {c_n}
def c (n : ℕ) : ℕ := (a n + 1)^(n + 1) / (b n + 2)^n

-- Defining the sum of the first n terms of {c_n}
def T (n : ℕ) : ℕ :=
  (3 * n + 1) * 2^(n + 2) - 4

-- Theorem to prove general term formula for {b_n}
theorem part_one : ∀ n : ℕ, b n = 3 * n + 2 := 
by sorry

-- Theorem to prove the sum of the first n terms of {c_n}
theorem part_two (n : ℕ) : ∀ n : ℕ, T n = (3 * n + 1) * 2^(n + 2) - 4 :=
by sorry

end part_one_part_two_l246_246354


namespace initial_amount_proof_l246_246369

noncomputable def initial_amount (A B : ℝ) : ℝ :=
  A + B

theorem initial_amount_proof :
  ∃ (A B : ℝ), B = 4000.0000000000005 ∧ 
               (A * 0.15 * 2 = B * 0.18 * 2 + 360) ∧ 
               initial_amount A B = 10000.000000000002 :=
by
  sorry

end initial_amount_proof_l246_246369


namespace find_savings_l246_246439

theorem find_savings (I E : ℕ) (h1 : I = 21000) (h2 : I / E = 7 / 6) : I - E = 3000 := by
  sorry

end find_savings_l246_246439


namespace count_pairs_l246_246579

theorem count_pairs (m n : ℕ) (h1 : 1 ≤ m ∧ m ≤ 2012)
  (cond : ∀ n : ℕ, 5^n < 2^m ∧ 2^m < 2^(m + 2) ∧ 2^(m + 2) < 5^(n + 1)) :
  ∃ k : ℕ, k = 279 :=
begin
  sorry
end

end count_pairs_l246_246579


namespace point_C_values_l246_246869

variable (B C : ℝ)
variable (distance_BC : ℝ)
variable (hB : B = 3)
variable (hDistance : distance_BC = 2)

theorem point_C_values (hBC : abs (C - B) = distance_BC) : (C = 1 ∨ C = 5) := 
by
  sorry

end point_C_values_l246_246869


namespace part_a_l246_246028

theorem part_a (x y : ℝ) (hx : 1 > x ∧ x ≥ 0) (hy : 1 > y ∧ y ≥ 0) : 
  ⌊5 * x⌋ + ⌊5 * y⌋ ≥ ⌊3 * x + y⌋ + ⌊3 * y + x⌋ := sorry

end part_a_l246_246028


namespace original_weight_of_beef_l246_246166

theorem original_weight_of_beef (w_after : ℝ) (loss_percentage : ℝ) (w_before : ℝ) : 
  (w_after = 550) → (loss_percentage = 0.35) → (w_after = 550) → (w_before = 846.15) :=
by
  intros
  sorry

end original_weight_of_beef_l246_246166


namespace find_defective_keys_l246_246963

-- Definitions from the conditions
def ten_digit_sequence : Type := list ℕ
def registered_digits : Type := list ℕ

axiom typed_ten_digits (s : ten_digit_sequence) : s.length = 10
axiom only_seven_registered (t : registered_digits) : t.length = 7
axiom three_missing_digits (s : ten_digit_sequence) (t : registered_digits) : 
             s.length - t.length = 3

-- This indicates that it is the same type of digits just subsets of initial sequence
axiom all_digits_in_sequence (s : ten_digit_sequence) (t : registered_digits) : 
            ∀ (d : ℕ), d ∈ t → d ∈ s

axiom defective_key_condition (s : ten_digit_sequence) (t : registered_digits) : 
            ∃ d : ℕ, (d ∈ s ∧ d ∉ t) ∧ count s d >= 5 ∧ count t d = 2

axiom multiple_defective_keys_condition (s : ten_digit_sequence) (t : registered_digits): 
           ∃ d1 d2 : ℕ, d1 ≠ d2 ∧ (d1 ∈ s ∧ d2 ∈ s) ∧ (d1 ∉ t ∧ d2 ∉ t) ∧ 
           (count s d1 >= 3 ∧ count s d2 >= 3)

-- Proving the answer:
theorem find_defective_keys (s : ten_digit_sequence) (t : registered_digits) :
  typed_ten_digits s → only_seven_registered t → three_missing_digits s t → 
  all_digits_in_sequence s t → defective_key_condition s t → multiple_defective_keys_condition s t → 
  ∃ (keys : list ℕ), keys = [7, 9] :=
begin
  sorry
end

end find_defective_keys_l246_246963


namespace necessary_and_sufficient_condition_l246_246663

open Set

noncomputable def M : Set (ℝ × ℝ) := {p | p.2 ≥ p.1 ^ 2}

noncomputable def N (a : ℝ) : Set (ℝ × ℝ) := {p | p.1 ^ 2 + (p.2 - a) ^ 2 ≤ 1}

theorem necessary_and_sufficient_condition (a : ℝ) :
  N a ⊆ M ↔ a ≥ 5 / 4 := sorry

end necessary_and_sufficient_condition_l246_246663


namespace cars_and_tourists_l246_246767

theorem cars_and_tourists (n t : ℕ) (h : n * t = 737) : n = 11 ∧ t = 67 ∨ n = 67 ∧ t = 11 :=
by
  sorry

end cars_and_tourists_l246_246767


namespace positive_difference_l246_246593

-- Define the binomial coefficient
def binomial (n : ℕ) (k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

-- Define the probability of heads in a fair coin flip
def fair_coin_prob : ℚ := 1 / 2

-- Define the probability of exactly k heads out of n flips
def prob_heads (n k : ℕ) : ℚ :=
  binomial n k * (fair_coin_prob ^ k) * (fair_coin_prob ^ (n - k))

-- Define the probabilities for the given problem
def prob_3_heads_out_of_5 : ℚ := prob_heads 5 3
def prob_5_heads_out_of_5 : ℚ := prob_heads 5 5

-- Claim the positive difference
theorem positive_difference :
  prob_3_heads_out_of_5 - prob_5_heads_out_of_5 = 9 / 32 :=
by
  sorry

end positive_difference_l246_246593


namespace problem_1_problem_2_l246_246469

-- Problem (1)
theorem problem_1 (a c : ℝ) (h1 : ∀ x, -1/3 < x ∧ x < 1/2 → ax^2 + 2*x + c > 0) :
  ∃ s, s = { x | -2 < x ∧ x < 3 } ∧ (∀ x, x ∈ s → cx^2 - 2*x + a < 0) := 
sorry

-- Problem (2)
theorem problem_2 (m : ℝ) (h : ∀ x : ℝ, x > 0 → x^2 - m*x + 4 > 0) :
  m < 4 := 
sorry

end problem_1_problem_2_l246_246469


namespace opposite_of_negative_five_l246_246938

theorem opposite_of_negative_five : ∃ x : ℤ, -5 + x = 0 ∧ x = 5 :=
by
  use 5
  split
  {
    sorry,
  }
  {
    refl,
  }

end opposite_of_negative_five_l246_246938


namespace sum_of_tens_and_units_digit_l246_246798

theorem sum_of_tens_and_units_digit (n : ℕ) (h : n = 11^2004 - 5) : 
  (n % 100 / 10) + (n % 10) = 9 :=
by
  sorry

end sum_of_tens_and_units_digit_l246_246798


namespace five_people_six_chairs_l246_246836

/-- Number of ways to sit 5 people in 6 chairs -/
def ways_to_sit_in_chairs : ℕ :=
  6 * 5 * 4 * 3 * 2

theorem five_people_six_chairs : ways_to_sit_in_chairs = 720 := by
  -- placeholder for the proof
  sorry

end five_people_six_chairs_l246_246836


namespace sum_series_eq_l246_246503

theorem sum_series_eq : 
  (∑' k : ℕ, (k + 1) * (1/4)^(k + 1)) = 4 / 9 :=
by sorry

end sum_series_eq_l246_246503


namespace positive_square_factors_of_180_l246_246091

theorem positive_square_factors_of_180 :
  let prime_factorization_180 := (2^2, 3^2, 5^1)
  let num_even_exponents (exp : ℕ) := if exp = 1 then 1 else 2 in
  let factors := num_even_exponents 2 * num_even_exponents 2 * num_even_exponents 1 in
  factors = 4 :=
by
  let prime_factorization_180 := (2^2, 3^2, 5^1)
  let num_even_exponents (exp : ℕ) := if exp = 1 then 1 else 2 in
  let factors := num_even_exponents 2 * num_even_exponents 2 * num_even_exponents 1 in
  show factors = 4
  sorry

end positive_square_factors_of_180_l246_246091


namespace jake_weight_l246_246374

theorem jake_weight (J S B : ℝ) (h1 : J - 8 = 2 * S)
                            (h2 : B = 2 * J + 6)
                            (h3 : J + S + B = 480)
                            (h4 : B = 1.25 * S) :
  J = 230 :=
by
  sorry

end jake_weight_l246_246374


namespace tan_four_fifths_alpha_l246_246074

theorem tan_four_fifths_alpha 
  (α : ℝ) 
  (hα1 : 0 < α) 
  (hα2 : α < π / 2) 
  (h_eq : 2 * sqrt 3 * (cos α) ^ 2 - sin (2 * α) + 2 - sqrt 3 = 0) : 
  tan (4 / 5 * α) = sqrt 3 := 
sorry

end tan_four_fifths_alpha_l246_246074


namespace evaluate_g_3_times_l246_246416

def g (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 2 * n - 1
  else 2 * n + 3

theorem evaluate_g_3_times : g (g (g 3)) = 65 := by
  sorry

end evaluate_g_3_times_l246_246416


namespace glee_club_female_members_l246_246385

theorem glee_club_female_members (m f : ℕ) 
  (h1 : f = 2 * m) 
  (h2 : m + f = 18) : 
  f = 12 :=
by
  sorry

end glee_club_female_members_l246_246385


namespace factor_expression_l246_246435

theorem factor_expression (a b c : ℝ) :
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 =
  (a - b)^2 * (b - c)^2 * (c - a)^2 * (a + b + c) :=
sorry

end factor_expression_l246_246435


namespace sum_of_digits_floor_large_number_div_50_eq_457_l246_246996

-- Define a helper function to sum the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the large number as the sum of its components
def large_number : ℕ :=
  51 * 10^96 + 52 * 10^94 + 53 * 10^92 + 54 * 10^90 + 55 * 10^88 + 56 * 10^86 + 
  57 * 10^84 + 58 * 10^82 + 59 * 10^80 + 60 * 10^78 + 61 * 10^76 + 62 * 10^74 + 
  63 * 10^72 + 64 * 10^70 + 65 * 10^68 + 66 * 10^66 + 67 * 10^64 + 68 * 10^62 + 
  69 * 10^60 + 70 * 10^58 + 71 * 10^56 + 72 * 10^54 + 73 * 10^52 + 74 * 10^50 + 
  75 * 10^48 + 76 * 10^46 + 77 * 10^44 + 78 * 10^42 + 79 * 10^40 + 80 * 10^38 + 
  81 * 10^36 + 82 * 10^34 + 83 * 10^32 + 84 * 10^30 + 85 * 10^28 + 86 * 10^26 + 
  87 * 10^24 + 88 * 10^22 + 89 * 10^20 + 90 * 10^18 + 91 * 10^16 + 92 * 10^14 + 
  93 * 10^12 + 94 * 10^10 + 95 * 10^8 + 96 * 10^6 + 97 * 10^4 + 98 * 10^2 + 99

-- Define the main statement to be proven
theorem sum_of_digits_floor_large_number_div_50_eq_457 : 
    sum_of_digits (Nat.floor (large_number / 50)) = 457 :=
by
  sorry

end sum_of_digits_floor_large_number_div_50_eq_457_l246_246996


namespace value_at_4_value_of_x_when_y_is_0_l246_246660

-- Problem statement
def f (x : ℝ) : ℝ := 2 * x - 3

-- Proof statement 1: When x = 4, y = 5
theorem value_at_4 : f 4 = 5 := sorry

-- Proof statement 2: When y = 0, x = 3/2
theorem value_of_x_when_y_is_0 : (∃ x : ℝ, f x = 0) → (∃ x : ℝ, x = 3 / 2) := sorry

end value_at_4_value_of_x_when_y_is_0_l246_246660


namespace concave_number_probability_l246_246482

/-- Definition of a concave number -/
def is_concave_number (a b c : ℕ) : Prop :=
  a > b ∧ c > b

/-- Set of possible digits -/
def digits : Finset ℕ := {4, 5, 6, 7, 8}

 /-- Total number of distinct three-digit combinations -/
def total_combinations : ℕ := 60

 /-- Number of concave numbers -/
def concave_numbers : ℕ := 20

 /-- Probability that a randomly chosen three-digit number is a concave number -/
def probability_concave : ℚ := concave_numbers / total_combinations

theorem concave_number_probability :
  probability_concave = 1 / 3 :=
by
  sorry

end concave_number_probability_l246_246482


namespace nicky_run_time_l246_246276

-- Define the constants according to the conditions in the problem
def head_start : ℕ := 100 -- Nicky's head start (meters)
def cr_speed : ℕ := 8 -- Cristina's speed (meters per second)
def ni_speed : ℕ := 4 -- Nicky's speed (meters per second)

-- Define the event of Cristina catching up to Nicky
def meets_at_time (t : ℕ) : Prop :=
  cr_speed * t = head_start + ni_speed * t

-- The proof statement
theorem nicky_run_time : ∃ t : ℕ, meets_at_time t ∧ t = 25 :=
by
  sorry

end nicky_run_time_l246_246276


namespace sara_initial_quarters_l246_246282

theorem sara_initial_quarters (borrowed quarters_current : ℕ) (q_initial : ℕ) :
  quarters_current = 512 ∧ quarters_borrowed = 271 → q_initial = 783 :=
by
  sorry

end sara_initial_quarters_l246_246282


namespace p_sufficient_for_not_q_l246_246652

variable (x : ℝ)
def p : Prop := 0 < x ∧ x ≤ 1
def q : Prop := 1 / x < 1

theorem p_sufficient_for_not_q : p x → ¬q x :=
by
  sorry

end p_sufficient_for_not_q_l246_246652


namespace expected_interval_proof_l246_246870

noncomputable def expected_interval_between_trains : ℝ := 3

theorem expected_interval_proof
  (northern_route_time southern_route_time : ℝ)
  (counter_clockwise_delay : ℝ)
  (home_to_work_less_than_work_to_home : ℝ) :
  northern_route_time = 17 →
  southern_route_time = 11 →
  counter_clockwise_delay = 75 / 60 →
  home_to_work_less_than_work_to_home = 1 →
  expected_interval_between_trains = 3 :=
by
  intros
  sorry

end expected_interval_proof_l246_246870


namespace equal_likelihood_of_lucky_sums_solution_l246_246400

variables (N : ℕ)

def main_draw := 10
def additional_draw := 8

def lucky_sum_main (N : ℕ) (sum : ℕ) := ∃ (S : finset ℕ), S.card = main_draw ∧ S.sum = sum
def lucky_sum_additional (N : ℕ) (sum : ℕ) := ∃ (S : finset ℕ), S.card = additional_draw ∧ S.sum = sum

theorem equal_likelihood_of_lucky_sums :
  lucky_sum_main N 63 ↔ lucky_sum_additional N 44 :=
sorry

theorem solution :
  equal_likelihood_of_lucky_sums 18 :=
sorry

end equal_likelihood_of_lucky_sums_solution_l246_246400


namespace subtraction_of_decimals_l246_246048

theorem subtraction_of_decimals : (3.75 - 0.48) = 3.27 :=
by
  sorry

end subtraction_of_decimals_l246_246048


namespace book_arrangements_l246_246470

theorem book_arrangements (n : ℕ) (b1 b2 b3 b4 b5 : ℕ) (h_b123 : b1 < b2 ∧ b2 < b3):
  n = 20 := sorry

end book_arrangements_l246_246470


namespace problem_statement_l246_246720

-- Define the conditions as Lean predicates
def is_odd (n : ℕ) : Prop := n % 2 = 1
def between_400_and_600 (n : ℕ) : Prop := 400 < n ∧ n < 600
def divisible_by_55 (n : ℕ) : Prop := n % 55 = 0

-- Define a function to calculate the sum of the digits
def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + (n % 100 / 10) + (n % 10)

-- Main theorem to prove
theorem problem_statement (N : ℕ)
  (h_odd : is_odd N)
  (h_range : between_400_and_600 N)
  (h_divisible : divisible_by_55 N) :
  sum_of_digits N = 18 :=
sorry

end problem_statement_l246_246720


namespace coffee_expenses_l246_246301

-- Define amounts consumed and unit costs for French and Columbian roast
def ounces_per_donut_M := 2
def ounces_per_donut_D := 3
def ounces_per_donut_S := ounces_per_donut_D
def ounces_per_pot_F := 12
def ounces_per_pot_C := 15
def cost_per_pot_F := 3
def cost_per_pot_C := 4

-- Define number of donuts consumed
def donuts_M := 8
def donuts_D := 12
def donuts_S := 16

-- Calculate total ounces needed
def total_ounces_F := donuts_M * ounces_per_donut_M
def total_ounces_C := (donuts_D + donuts_S) * ounces_per_donut_D

-- Calculate pots needed, rounding up since partial pots are not allowed
def pots_needed_F := Nat.ceil (total_ounces_F / ounces_per_pot_F)
def pots_needed_C := Nat.ceil (total_ounces_C / ounces_per_pot_C)

-- Calculate total cost
def total_cost := (pots_needed_F * cost_per_pot_F) + (pots_needed_C * cost_per_pot_C)

-- Theorem statement to assert the proof
theorem coffee_expenses : total_cost = 30 := by
  sorry

end coffee_expenses_l246_246301


namespace correct_factorization_l246_246321

theorem correct_factorization :
  ∀ x : ℝ, x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end correct_factorization_l246_246321


namespace gcf_of_294_and_108_l246_246157

theorem gcf_of_294_and_108 : Nat.gcd 294 108 = 6 :=
by
  -- We are given numbers 294 and 108
  -- Their prime factorizations are 294 = 2 * 3 * 7^2 and 108 = 2^2 * 3^3
  -- The minimum power of the common prime factors are 2^1 and 3^1
  -- Thus, the GCF by multiplying these factors is 2^1 * 3^1 = 6
  sorry

end gcf_of_294_and_108_l246_246157


namespace number_of_divisors_l246_246665

theorem number_of_divisors (n : ℕ) (h1: n = 70) (h2: ∀ k, k ∣ n → k > 3 ↔ (k = 5 ∨ k = 7 ∨ k = 10 ∨ k = 14 ∨ k = 35 ∨ k = 70)) : 
  {k : ℕ | k ∣ n ∧ k > 3}.to_finset.card = 6 :=
by {
  simp [h1, h2],
  sorry
}

end number_of_divisors_l246_246665


namespace smallest_integer_with_18_divisors_l246_246738

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, has_exactly_18_divisors n ∧ (∀ m : ℕ, has_exactly_18_divisors m → n ≤ m) ∧ n = 540 := sorry

end smallest_integer_with_18_divisors_l246_246738


namespace lizette_overall_average_is_94_l246_246864

-- Defining the given conditions
def third_quiz_score : ℕ := 92
def first_two_quizzes_average : ℕ := 95
def total_quizzes : ℕ := 3

-- Calculating total points from the conditions
def total_points : ℕ := first_two_quizzes_average * 2 + third_quiz_score

-- Defining the overall average to prove
def overall_average : ℕ := total_points / total_quizzes

-- The theorem stating Lizette's overall average after taking the third quiz
theorem lizette_overall_average_is_94 : overall_average = 94 := by
  sorry

end lizette_overall_average_is_94_l246_246864


namespace farmer_total_profit_l246_246772

theorem farmer_total_profit :
  let group1_revenue := 3 * 375
  let group1_cost := (8 * 13 + 3 * 15) * 3
  let group1_profit := group1_revenue - group1_cost

  let group2_revenue := 4 * 425
  let group2_cost := (5 * 14 + 9 * 16) * 4
  let group2_profit := group2_revenue - group2_cost

  let group3_revenue := 2 * 475
  let group3_cost := (10 * 15 + 8 * 18) * 2
  let group3_profit := group3_revenue - group3_cost

  let group4_revenue := 1 * 550
  let group4_cost := 20 * 20 * 1
  let group4_profit := group4_revenue - group4_cost

  let total_profit := group1_profit + group2_profit + group3_profit + group4_profit
  total_profit = 2034 :=
by
  sorry

end farmer_total_profit_l246_246772


namespace geom_seq_42_l246_246070

variable {α : Type*} [Field α] [CharZero α]

noncomputable def a_n (n : ℕ) (a1 q : α) : α := a1 * q ^ n

theorem geom_seq_42 (a1 q : α) (h1 : a1 = 3) (h2 : a1 * (1 + q^2 + q^4) = 21) :
  a1 * (q^2 + q^4 + q^6) = 42 := 
by
  sorry

end geom_seq_42_l246_246070


namespace dave_paid_for_6_candy_bars_l246_246550

-- Given conditions
def number_of_candy_bars : ℕ := 20
def cost_per_candy_bar : ℝ := 1.50
def amount_paid_by_john : ℝ := 21

-- Correct answer
def number_of_candy_bars_paid_by_dave : ℝ := 6

-- The proof problem in Lean statement
theorem dave_paid_for_6_candy_bars (H : number_of_candy_bars * cost_per_candy_bar - amount_paid_by_john = 9) :
  number_of_candy_bars_paid_by_dave = 6 := by
sorry

end dave_paid_for_6_candy_bars_l246_246550


namespace flower_pots_count_l246_246700

noncomputable def total_flower_pots (x : ℕ) : ℕ :=
  if h : ((x / 2) + (x / 4) + (x / 7) ≤ x - 1) then x else 0

theorem flower_pots_count : total_flower_pots 28 = 28 :=
by
  sorry

end flower_pots_count_l246_246700


namespace negate_statement_l246_246821

variable (Students Teachers : Type)
variable (Patient : Students → Prop)
variable (PatientT : Teachers → Prop)
variable (a : ∀ t : Teachers, PatientT t)
variable (b : ∃ t : Teachers, PatientT t)
variable (c : ∀ s : Students, ¬ Patient s)
variable (d : ∀ s : Students, ¬ Patient s)
variable (e : ∃ s : Students, ¬ Patient s)
variable (f : ∀ s : Students, Patient s)

theorem negate_statement : (∃ s : Students, ¬ Patient s) ↔ ¬ (∀ s : Students, Patient s) :=
by sorry

end negate_statement_l246_246821


namespace intersection_A_B_l246_246087

def A : Set ℕ := {1, 2, 3}

def B : Set ℕ := {y | ∃ x ∈ A, y = 2 * x - 1}

theorem intersection_A_B : A ∩ B = {1, 3} := 
by 
  sorry

end intersection_A_B_l246_246087


namespace ten_years_less_than_average_age_l246_246261

theorem ten_years_less_than_average_age (L : ℕ) :
  (2 * L - 14) = 
    (2 * L - 4) - 10 :=
by {
  sorry
}

end ten_years_less_than_average_age_l246_246261


namespace range_of_a_l246_246690

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 2 then (x - a) ^ 2 + Real.exp 1 else x / Real.log x + a + 10

theorem range_of_a (a : ℝ) :
    (∀ x, f x a ≥ f 2 a) → (2 ≤ a ∧ a ≤ 6) :=
by
  sorry

end range_of_a_l246_246690


namespace find_sum_l246_246888

variable (a b : ℝ)

theorem find_sum (h1 : 2 = b - 1) (h2 : -1 = a + 3) : a + b = -1 :=
by
  sorry

end find_sum_l246_246888


namespace cyclist_speed_ratio_is_4_l246_246948

noncomputable def ratio_of_speeds (v_a v_b v_c : ℝ) : ℝ :=
  if v_a ≤ v_b ∧ v_b ≤ v_c then v_c / v_a else 0

theorem cyclist_speed_ratio_is_4
  (v_a v_b v_c : ℝ)
  (h1 : v_a + v_b = d / 5)
  (h2 : v_b + v_c = 15)
  (h3 : 15 = (45 - d) / 3)
  (d : ℝ) : 
  ratio_of_speeds v_a v_b v_c = 4 :=
by
  sorry

end cyclist_speed_ratio_is_4_l246_246948


namespace line_through_two_points_line_with_intercept_sum_l246_246210

theorem line_through_two_points (a b x1 y1 x2 y2: ℝ) : 
  (x1 = 2) → (y1 = 1) → (x2 = 0) → (y2 = -3) → (2 * x - y - 3 = 0) :=
by
                
  sorry

theorem line_with_intercept_sum (a b : ℝ) (x y : ℝ) :
  (x = 0) → (y = 5) → (a + b = 2) → (b = 5) → (5 * x - 3 * y + 15 = 0) :=
by
  sorry

end line_through_two_points_line_with_intercept_sum_l246_246210


namespace arithmetic_sequence_15th_term_l246_246595

theorem arithmetic_sequence_15th_term :
  let a1 := 3
  let d := 7
  let n := 15
  a1 + (n - 1) * d = 101 :=
by
  let a1 := 3
  let d := 7
  let n := 15
  sorry

end arithmetic_sequence_15th_term_l246_246595


namespace value_of_y_when_x_equals_8_l246_246447

variables (x y k : ℝ)

theorem value_of_y_when_x_equals_8 
  (hp : x * y = k)
  (h1 : x + y = 30)
  (h2 : x - y = 10)
  (hx8 : x = 8) :
  y = 25 :=
sorry

end value_of_y_when_x_equals_8_l246_246447


namespace line_through_point_perpendicular_l246_246993

theorem line_through_point_perpendicular :
  ∃ (a b : ℝ), ∀ (x : ℝ), y = - (3 / 2) * x + 8 ∧ y - 2 = - (3 / 2) * (x - 4) ∧ 2*x - 3*y = 6 → y = - (3 / 2) * x + 8 :=
by 
  sorry

end line_through_point_perpendicular_l246_246993


namespace identify_faulty_key_l246_246959

variable (digits : Finset ℕ)
variable (faulty : ℕ → Bool)

-- Conditions described in the problem statement
variable (attempted_sequence : List ℕ) (registered_sequence : List ℕ)
variable (sequence_length : Nat := 10)
variable (registered_count : Nat := 7)
variable (faulty_press_threshold : Nat := 5)

-- Let attempted_sequence be the sequence typed out and registered_sequence be what was actually registered.

theorem identify_faulty_key (h_len_attempted : attempted_sequence.length = sequence_length)
                            (h_len_registered : registered_sequence.length = registered_count)
                            (h_frequent_digits : ∃ d1 d2, d1 ≠ d2 ∧
                                                        attempted_sequence.count d1 ≥ 2 ∧
                                                        attempted_sequence.count d2 ≥ 2 ∧
                                                        (attempted_sequence.count d1 - registered_sequence.count d1 ≥ 1) ∧
                                                        (attempted_sequence.count d2 - registered_sequence.count d2 ≥ 1)) :
  ∃ d, faulty d ∧ (d = 7 ∨ d = 9) :=
sorry

end identify_faulty_key_l246_246959


namespace equation_of_parallel_line_through_point_l246_246345

theorem equation_of_parallel_line_through_point :
  ∃ m b, (∀ x y, y = m * x + b → (∃ k, k = 3 ^ 2 - 9 * 2 + 1)) ∧ 
         (∀ x y, y = 3 * x + b → y - 0 = 3 * (x - (-2))) :=
sorry

end equation_of_parallel_line_through_point_l246_246345


namespace inequality_for_positive_reals_l246_246278

theorem inequality_for_positive_reals
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a / (a^4 + b^2) + b / (a^2 + b^4) ≤ 1 / (a * b) := 
sorry

end inequality_for_positive_reals_l246_246278


namespace wheel_moves_distance_in_one_hour_l246_246759

-- Definition of the given conditions
def rotations_per_minute : ℕ := 10
def distance_per_rotation : ℕ := 20
def minutes_per_hour : ℕ := 60

-- Theorem statement to prove the wheel moves 12000 cm in one hour
theorem wheel_moves_distance_in_one_hour : 
  rotations_per_minute * minutes_per_hour * distance_per_rotation = 12000 := 
by
  sorry

end wheel_moves_distance_in_one_hour_l246_246759


namespace largest_even_number_l246_246255

theorem largest_even_number (x : ℕ) (h : x + (x+2) + (x+4) = 1194) : x + 4 = 400 :=
by
  have : 3*x + 6 = 1194 := by linarith
  have : 3*x = 1188 := by linarith
  have : x = 396 := by linarith
  linarith

end largest_even_number_l246_246255


namespace max_abs_sum_x_y_l246_246100

theorem max_abs_sum_x_y (x y : ℝ) (h : x^2 + y^2 = 4) : |x| + |y| ≤ 2 * real.sqrt 2 :=
sorry

end max_abs_sum_x_y_l246_246100


namespace curve_eq_circle_l246_246061

theorem curve_eq_circle (r θ : ℝ) : (∀ θ : ℝ, r = 3) ↔ ∃ c : ℝ, c = 0 ∧ ∀ z : ℝ, (r = real.sqrt ((3 - c)^2 + θ^2)) := sorry

end curve_eq_circle_l246_246061


namespace integral_quarter_circle_l246_246502

noncomputable def integral_problem : ℝ :=
  ∫ x in 0..1, Real.sqrt (x * (2 - x))

theorem integral_quarter_circle (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 
  (Real.sqrt (x * (2 - x))) = Real.sqrt (1 - (x - 1)^2)) :
  integral_problem = (1/4) * Real.pi :=
  sorry

end integral_quarter_circle_l246_246502


namespace furniture_definition_based_on_vocabulary_study_l246_246463

theorem furniture_definition_based_on_vocabulary_study (term : String) (h : term = "furniture") :
  term = "furniture" :=
by
  sorry

end furniture_definition_based_on_vocabulary_study_l246_246463


namespace time_to_finish_furniture_l246_246466

-- Define the problem's conditions
def chairs : ℕ := 7
def tables : ℕ := 3
def minutes_per_piece : ℕ := 4

-- Define total furniture
def total_furniture : ℕ := chairs + tables

-- Define the function to calculate total time
def total_time (pieces : ℕ) (time_per_piece: ℕ) : ℕ :=
  pieces * time_per_piece

-- Theorem statement to be proven
theorem time_to_finish_furniture : total_time total_furniture minutes_per_piece = 40 := 
by
  -- Provide a placeholder for the proof
  sorry

end time_to_finish_furniture_l246_246466


namespace points_earned_l246_246568

-- Define the number of pounds required to earn one point
def pounds_per_point : ℕ := 4

-- Define the number of pounds Paige recycled
def paige_recycled : ℕ := 14

-- Define the number of pounds Paige's friends recycled
def friends_recycled : ℕ := 2

-- Define the total number of pounds recycled
def total_recycled : ℕ := paige_recycled + friends_recycled

-- Define the total number of points earned
def total_points : ℕ := total_recycled / pounds_per_point

-- Theorem to prove the total points earned
theorem points_earned : total_points = 4 := by
  sorry

end points_earned_l246_246568


namespace product_equation_l246_246315

theorem product_equation (a b : ℝ) (h1 : ∀ (a b : ℝ), 0.2 * b = 0.9 * a - b) : 
  0.9 * a - b = 0.2 * b :=
by
  sorry

end product_equation_l246_246315


namespace positive_difference_l246_246594

-- Define the binomial coefficient
def binomial (n : ℕ) (k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

-- Define the probability of heads in a fair coin flip
def fair_coin_prob : ℚ := 1 / 2

-- Define the probability of exactly k heads out of n flips
def prob_heads (n k : ℕ) : ℚ :=
  binomial n k * (fair_coin_prob ^ k) * (fair_coin_prob ^ (n - k))

-- Define the probabilities for the given problem
def prob_3_heads_out_of_5 : ℚ := prob_heads 5 3
def prob_5_heads_out_of_5 : ℚ := prob_heads 5 5

-- Claim the positive difference
theorem positive_difference :
  prob_3_heads_out_of_5 - prob_5_heads_out_of_5 = 9 / 32 :=
by
  sorry

end positive_difference_l246_246594


namespace trig_identity_l246_246095

variable (α : Real)
variable (h : Real.tan α = 2)

theorem trig_identity :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3 / 4 := by
  sorry

end trig_identity_l246_246095


namespace find_a4_l246_246224
open Nat

def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = a n / (2 * a n + 3)

theorem find_a4 (a : ℕ → ℚ) (h : seq a) : a 4 = 1 / 53 :=
by
  obtain ⟨h1, h_rec⟩ := h
  have a2 := h_rec 1 (by decide)
  have a3 := h_rec 2 (by decide)
  have a4 := h_rec 3 (by decide)
  -- Proof steps would go here
  sorry

end find_a4_l246_246224


namespace range_of_a_l246_246081

noncomputable def f (x : ℝ) : ℝ :=
  Real.exp x - Real.exp (-x) + Real.log (x + Real.sqrt (x^2 + 1))

theorem range_of_a
  (h : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → f (x^2 + 2) + f (-2 * a * x) ≥ 0) :
  -3/2 ≤ a ∧ a ≤ Real.sqrt 2 :=
sorry

end range_of_a_l246_246081


namespace rowing_time_to_and_fro_l246_246778

noncomputable def rowing_time (distance rowing_speed current_speed : ℤ) : ℤ :=
  let speed_to_place := rowing_speed - current_speed
  let speed_back_place := rowing_speed + current_speed
  let time_to_place := distance / speed_to_place
  let time_back_place := distance / speed_back_place
  time_to_place + time_back_place

theorem rowing_time_to_and_fro (distance rowing_speed current_speed : ℤ) :
  distance = 72 → rowing_speed = 10 → current_speed = 2 → rowing_time distance rowing_speed current_speed = 15 := by
  intros h_dist h_row_speed h_curr_speed
  rw [h_dist, h_row_speed, h_curr_speed]
  sorry

end rowing_time_to_and_fro_l246_246778


namespace range_of_k_l246_246254

theorem range_of_k (k : ℝ) : (∀ x : ℝ, |x - 2| + |x - 3| > |k - 1|) → 0 < k ∧ k < 2 :=
by
  sorry

end range_of_k_l246_246254


namespace correct_statement_about_meiosis_and_fertilization_l246_246620

def statement_A : Prop := 
  ∃ oogonia spermatogonia zygotes : ℕ, 
    oogonia = 20 ∧ spermatogonia = 8 ∧ zygotes = 32 ∧ 
    (oogonia + spermatogonia = zygotes)

def statement_B : Prop := 
  ∀ zygote_dna mother_half father_half : ℕ,
    zygote_dna = mother_half + father_half ∧ 
    mother_half = father_half

def statement_C : Prop := 
  ∀ (meiosis stabilizes : Prop) (chromosome_count : ℕ),
    (meiosis → stabilizes) ∧ 
    (stabilizes → chromosome_count = (chromosome_count / 2 + chromosome_count / 2))

def statement_D : Prop := 
  ∀ (diversity : Prop) (gene_mutations chromosomal_variations : Prop),
    (diversity → ¬ (gene_mutations ∨ chromosomal_variations))

theorem correct_statement_about_meiosis_and_fertilization :
  ¬ statement_A ∧ ¬ statement_B ∧ statement_C ∧ ¬ statement_D :=
by
  sorry

end correct_statement_about_meiosis_and_fertilization_l246_246620


namespace total_votes_l246_246021

/-- Let V be the total number of votes. Define the votes received by the candidate and rival. -/
def votes_cast (V : ℕ) (votes_candidate : ℕ) (votes_rival : ℕ) : Prop :=
  votes_candidate = 40 * V / 100 ∧ votes_rival = votes_candidate + 2000 ∧ votes_candidate + votes_rival = V

/-- Prove that the total number of votes is 10000 given the conditions. -/
theorem total_votes (V : ℕ) (votes_candidate : ℕ) (votes_rival : ℕ) :
  votes_cast V votes_candidate votes_rival → V = 10000 :=
by
  sorry

end total_votes_l246_246021


namespace correct_calculation_for_b_l246_246302

theorem correct_calculation_for_b (a : ℝ) : (a^2)^3 = a^6 := 
by sorry

end correct_calculation_for_b_l246_246302


namespace price_increase_percentage_l246_246476

theorem price_increase_percentage (c : ℝ) (r : ℝ) (p : ℝ) 
  (h1 : r = 1.4 * c) 
  (h2 : p = 1.15 * r) : 
  (p - c) / c * 100 = 61 := 
sorry

end price_increase_percentage_l246_246476


namespace divisors_count_of_108n5_l246_246644

theorem divisors_count_of_108n5 {n : ℕ} (hn_pos : 0 < n) (h_divisors_150n3 : (150 * n^3).divisors.card = 150) : 
(108 * n^5).divisors.card = 432 :=
sorry

end divisors_count_of_108n5_l246_246644


namespace inequality_proof_l246_246520

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a^2 / (b + c) + b^2 / (c + a) + c^2 / (a + b)) ≥ 1 / 2 * (a + b + c) := 
by
  sorry

end inequality_proof_l246_246520


namespace proof_1_proof_2_proof_3_proof_4_proof_5_proof_6_l246_246984

noncomputable def problem_1 : Int :=
13 + (-5) - (-21) - 19

noncomputable def answer_1 : Int := 10

theorem proof_1 : problem_1 = answer_1 := 
by
  sorry

noncomputable def problem_2 : Rat :=
(0.125 : Rat) - (3 + 3 / 4 : Rat) + (-(3 + 1 / 8 : Rat)) - (-(10 + 2 / 3 : Rat)) - (1.25 : Rat)

noncomputable def answer_2 : Rat := 10 + 1 / 6

theorem proof_2 : problem_2 = answer_2 :=
by
  sorry

noncomputable def problem_3 : Rat :=
(36 : Int) / (-8) * (1 / 8 : Rat)

noncomputable def answer_3 : Rat := -9 / 16

theorem proof_3 : problem_3 = answer_3 :=
by
  sorry

noncomputable def problem_4 : Rat :=
((11 / 12 : Rat) - (7 / 6 : Rat) + (3 / 4 : Rat) - (13 / 24 : Rat)) * (-48)

noncomputable def answer_4 : Int := 2

theorem proof_4 : problem_4 = answer_4 :=
by
  sorry

noncomputable def problem_5 : Rat :=
(-(99 + 15 / 16 : Rat)) * 4

noncomputable def answer_5 : Rat := -(399 + 3 / 4 : Rat)

theorem proof_5 : problem_5 = answer_5 :=
by
  sorry

noncomputable def problem_6 : Rat :=
-(1 ^ 4 : Int) - ((1 - 0.5 : Rat) * (1 / 3 : Rat) * (2 - ((-3) ^ 2 : Int) : Int))

noncomputable def answer_6 : Rat := 1 / 6

theorem proof_6 : problem_6 = answer_6 :=
by
  sorry

end proof_1_proof_2_proof_3_proof_4_proof_5_proof_6_l246_246984


namespace algebraic_expression_analysis_l246_246847

theorem algebraic_expression_analysis :
  (∀ x y : ℝ, (x - 1/2 * y) * (x + 1/2 * y) = x^2 - (1/2 * y)^2) ∧
  (∀ a b c : ℝ, ¬ ((3 * a + b * c) * (-b * c - 3 * a) = (3 * a + b * c)^2)) ∧
  (∀ x y : ℝ, (3 - x + y) * (3 + x + y) = (3 + y)^2 - x^2) ∧
  ((100 + 1) * (100 - 1) = 100^2 - 1) :=
by
  intros
  repeat { split }; sorry

end algebraic_expression_analysis_l246_246847


namespace workshop_total_workers_l246_246108

theorem workshop_total_workers
  (avg_salary_per_head : ℕ)
  (num_technicians num_managers num_apprentices total_workers : ℕ)
  (avg_tech_salary avg_mgr_salary avg_appr_salary : ℕ) 
  (h1 : avg_salary_per_head = 700)
  (h2 : num_technicians = 5)
  (h3 : num_managers = 3)
  (h4 : avg_tech_salary = 800)
  (h5 : avg_mgr_salary = 1200)
  (h6 : avg_appr_salary = 650)
  (h7 : total_workers = num_technicians + num_managers + num_apprentices)
  : total_workers = 48 := 
sorry

end workshop_total_workers_l246_246108


namespace cody_paid_amount_l246_246495

/-- Cody buys $40 worth of stuff,
    the tax rate is 5%,
    he receives an $8 discount after taxes,
    and he and his friend split the final price equally.
    Prove that Cody paid $17. -/
theorem cody_paid_amount
  (initial_cost : ℝ)
  (tax_rate : ℝ)
  (discount : ℝ)
  (final_split : ℝ)
  (H1 : initial_cost = 40)
  (H2 : tax_rate = 0.05)
  (H3 : discount = 8)
  (H4 : final_split = 2) :
  (initial_cost * (1 + tax_rate) - discount) / final_split = 17 :=
by
  sorry

end cody_paid_amount_l246_246495


namespace no_positive_integer_solutions_l246_246423

theorem no_positive_integer_solutions (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  x^2017 - 1 ≠ (x - 1) * (y^2015 - 1) :=
by sorry

end no_positive_integer_solutions_l246_246423


namespace angie_age_l246_246005

variables (A : ℕ)

theorem angie_age (h : 2 * A + 4 = 20) : A = 8 :=
by {
  -- Proof will be provided in actual usage or practice
  sorry
}

end angie_age_l246_246005


namespace min_ab_l246_246647

variable (a b : ℝ)

theorem min_ab (h1 : a > 1) (h2 : b > 2) (h3 : a * b = 2 * a + b) : a + b ≥ 3 + 2 * Real.sqrt 2 := 
sorry

end min_ab_l246_246647


namespace arith_seq_sum_proof_l246_246444

open Function

variable (a : ℕ → ℕ) -- Define the arithmetic sequence
variables (S : ℕ → ℕ) -- Define the sum function of the sequence

-- Conditions: S_8 = 9 and S_5 = 6
axiom S8 : S 8 = 9
axiom S5 : S 5 = 6

-- Mathematical equivalence
theorem arith_seq_sum_proof : S 13 = 13 :=
sorry

end arith_seq_sum_proof_l246_246444


namespace minimum_value_expression_l246_246511

theorem minimum_value_expression (x : ℝ) : 
  (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 4 := 
by
  sorry

end minimum_value_expression_l246_246511


namespace equal_likelihood_of_lucky_sums_solution_l246_246401

variables (N : ℕ)

def main_draw := 10
def additional_draw := 8

def lucky_sum_main (N : ℕ) (sum : ℕ) := ∃ (S : finset ℕ), S.card = main_draw ∧ S.sum = sum
def lucky_sum_additional (N : ℕ) (sum : ℕ) := ∃ (S : finset ℕ), S.card = additional_draw ∧ S.sum = sum

theorem equal_likelihood_of_lucky_sums :
  lucky_sum_main N 63 ↔ lucky_sum_additional N 44 :=
sorry

theorem solution :
  equal_likelihood_of_lucky_sums 18 :=
sorry

end equal_likelihood_of_lucky_sums_solution_l246_246401


namespace quadratic_expression_evaluation_l246_246518

theorem quadratic_expression_evaluation (x y : ℝ) (h1 : 3 * x + y = 10) (h2 : x + 3 * y = 14) :
  10 * x^2 + 12 * x * y + 10 * y^2 = 296 :=
by
  -- Proof goes here
  sorry

end quadratic_expression_evaluation_l246_246518


namespace discriminant_of_quadratic_polynomial_l246_246805

theorem discriminant_of_quadratic_polynomial :
  let a := 5
  let b := (5 + 1/5 : ℚ)
  let c := (1/5 : ℚ) 
  let Δ := b^2 - 4 * a * c
  Δ = (576/25 : ℚ) :=
by
  sorry

end discriminant_of_quadratic_polynomial_l246_246805


namespace average_speed_is_42_l246_246968

theorem average_speed_is_42 (v t : ℝ) (h : t > 0)
  (h_eq : v * t = (v + 21) * (2/3) * t) : v = 42 :=
by
  sorry

end average_speed_is_42_l246_246968


namespace toys_sold_in_first_week_l246_246183

/-
  Problem statement:
  An online toy store stocked some toys. It sold some toys at the first week and 26 toys at the second week.
  If it had 19 toys left and there were 83 toys in stock at the beginning, how many toys were sold in the first week?
-/

theorem toys_sold_in_first_week (initial_stock toys_left toys_sold_second_week : ℕ) 
  (h_initial_stock : initial_stock = 83) 
  (h_toys_left : toys_left = 19) 
  (h_toys_sold_second_week : toys_sold_second_week = 26) : 
  (initial_stock - toys_left - toys_sold_second_week) = 38 :=
by
  -- Proof goes here
  sorry

end toys_sold_in_first_week_l246_246183


namespace exists_consecutive_divisible_by_cube_l246_246689

theorem exists_consecutive_divisible_by_cube (k : ℕ) (hk : 0 < k) : 
  ∃ n : ℕ, ∀ j : ℕ, j < k → ∃ m : ℕ, 1 < m ∧ (n + j) % (m^3) = 0 := 
sorry

end exists_consecutive_divisible_by_cube_l246_246689


namespace find_larger_number_l246_246760

theorem find_larger_number (L S : ℕ) (h1 : L - S = 1325) (h2 : L = 5 * S + 5) : L = 1655 :=
sorry

end find_larger_number_l246_246760


namespace problem_l246_246643

theorem problem (a b c d : ℤ) (ha : 0 ≤ a ∧ a ≤ 99) (hb : 0 ≤ b ∧ b ≤ 99) 
  (hc : 0 ≤ c ∧ c ≤ 99) (hd : 0 ≤ d ∧ d ≤ 99) :
  let n (x : ℤ) := 101 * x - 100 * 2 ^ x in
  n a + n b ≡ n c + n d [ZMOD 10100] →
  ({a, b} = {c, d} : multiset ℤ) := 
by sorry

end problem_l246_246643


namespace combined_cost_is_107_l246_246194

def wallet_cost : ℕ := 22
def purse_cost (wallet_price : ℕ) : ℕ := 4 * wallet_price - 3
def combined_cost (wallet_price : ℕ) (purse_price : ℕ) : ℕ := wallet_price + purse_price

theorem combined_cost_is_107 : combined_cost wallet_cost (purse_cost wallet_cost) = 107 := 
by 
  -- Proof
  sorry

end combined_cost_is_107_l246_246194


namespace evaluate_expression_l246_246339

variable (x y : ℝ)

theorem evaluate_expression :
  (1 + x^2 + y^3) * (1 - x^3 - y^3) = 1 + x^2 - x^3 - y^3 - x^5 - x^2 * y^3 - x^3 * y^3 - y^6 :=
by
  sorry

end evaluate_expression_l246_246339


namespace regular_hours_l246_246187

variable (R : ℕ)

theorem regular_hours (h1 : 5 * R + 6 * (44 - R) + 5 * R + 6 * (48 - R) = 472) : R = 40 :=
by
  sorry

end regular_hours_l246_246187


namespace number_of_pear_trees_l246_246848

theorem number_of_pear_trees (A P : ℕ) (h1 : A + P = 46)
  (h2 : ∀ (s : Finset (Fin 46)), s.card = 28 → ∃ (i : Fin 46), i ∈ s ∧ i < A)
  (h3 : ∀ (s : Finset (Fin 46)), s.card = 20 → ∃ (i : Fin 46), i ∈ s ∧ A ≤ i) :
  P = 27 :=
by
  sorry

end number_of_pear_trees_l246_246848


namespace area_PQRS_l246_246131

structure Rectangle :=
  (length : ℝ)
  (breadth : ℝ)

structure EquilateralTriangle :=
  (side_length : ℝ)

noncomputable def area_of_quadrilateral_PQRS (W X Y Z P Q R S : ℝ) 
  (rect : Rectangle) (eq_triangle_WXP eq_triangle_XQY eq_triangle_YRZ eq_triangle_ZSW : EquilateralTriangle)
  (h1: rect.length = 8) (h2: rect.breadth = 6)
  (h3: eq_triangle_WXP.side_length = 6) 
  (h4: eq_triangle_XQY.side_length = 8)
  (h5: eq_triangle_YRZ.side_length = 8)
  (h6: eq_triangle_ZSW.side_length = 6) : ℝ :=
  82 * Real.sqrt 3

theorem area_PQRS (W X Y Z P Q R S : ℝ)
  (rect : Rectangle) (eq_triangle_WXP eq_triangle_XQY eq_triangle_YRZ eq_triangle_ZSW : EquilateralTriangle)
  (h1: rect.length = 8) (h2: rect.breadth = 6)
  (h3: eq_triangle_WXP.side_length = 6) 
  (h4: eq_triangle_XQY.side_length = 8)
  (h5: eq_triangle_YRZ.side_length = 8)
  (h6: eq_triangle_ZSW.side_length = 6) :
  area_of_quadrilateral_PQRS W X Y Z P Q R S rect eq_triangle_WXP eq_triangle_XQY eq_triangle_YRZ eq_triangle_ZSW h1 h2 h3 h4 h5 h6 = 82 * Real.sqrt 3 :=
sorry

end area_PQRS_l246_246131


namespace find_birth_rate_l246_246540

noncomputable def average_birth_rate (B : ℕ) : Prop :=
  let death_rate := 3
  let net_increase_per_2_seconds := B - death_rate
  let seconds_per_hour := 3600
  let hours_per_day := 24
  let seconds_per_day := seconds_per_hour * hours_per_day
  let net_increase_times := seconds_per_day / 2
  let total_net_increase := net_increase_times * net_increase_per_2_seconds
  total_net_increase = 172800

theorem find_birth_rate (B : ℕ) (h : average_birth_rate B) : B = 7 :=
  sorry

end find_birth_rate_l246_246540


namespace find_k_l246_246349

def vector (α : Type) := (α × α)
def a : vector ℝ := (1, 3)
def b (k : ℝ) : vector ℝ := (-2, k)
def add (v1 v2 : vector ℝ) : vector ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def smul (c : ℝ) (v : vector ℝ) : vector ℝ := (c * v.1, c * v.2)
def cross_product (v1 v2 : vector ℝ) : ℝ := v1.1 * v2.2 - v1.2 * v2.1

theorem find_k (k : ℝ) (h : cross_product (add a (smul 2 (b k)))
                                          (add (smul 3 a) (smul (-1) (b k))) = 0) : k = -6 :=
sorry

end find_k_l246_246349


namespace jenny_kenny_reunion_time_l246_246681

/-- Define initial conditions given in the problem --/
def jenny_initial_pos : ℝ × ℝ := (-60, 100)
def kenny_initial_pos : ℝ × ℝ := (-60, -100)
def building_radius : ℝ := 60
def kenny_speed : ℝ := 4
def jenny_speed : ℝ := 2
def distance_apa : ℝ := 200
def initial_distance : ℝ := 200

theorem jenny_kenny_reunion_time : ∃ t : ℚ, 
  (t = (10 * (Real.sqrt 35)) / 7) ∧ 
  (17 = (10 + 7)) :=
by
  -- conditions to be used
  let jenny_pos (t : ℝ) := (-60 + 2 * t, 100)
  let kenny_pos (t : ℝ) := (-60 + 4 * t, -100)
  let circle_eq (x y : ℝ) := (x^2 + y^2 = building_radius^2)
  
  sorry

end jenny_kenny_reunion_time_l246_246681


namespace compare_points_on_line_l246_246653

theorem compare_points_on_line (m n : ℝ) 
  (hA : ∃ (x : ℝ), x = -3 ∧ m = -2 * x + 1) 
  (hB : ∃ (x : ℝ), x = 2 ∧ n = -2 * x + 1) : 
  m > n :=
by sorry

end compare_points_on_line_l246_246653


namespace candy_bars_to_buy_l246_246034

variable (x : ℕ)

theorem candy_bars_to_buy (h1 : 25 * x + 2 * 75 + 50 = 11 * 25) : x = 3 :=
by
  sorry

end candy_bars_to_buy_l246_246034


namespace find_function_expression_l246_246656

noncomputable def f (x : ℝ) : ℝ := x^2 - 5*x + 7

theorem find_function_expression (x : ℝ) :
  (∀ x : ℝ, f (x + 2) = x^2 - x + 1) →
  f x = x^2 - 5*x + 7 :=
by
  intro h
  sorry

end find_function_expression_l246_246656


namespace sum_difference_of_consecutive_integers_l246_246168

theorem sum_difference_of_consecutive_integers (n : ℤ) :
  let set1 := [(n-3), (n-2), (n-1), n, (n+1), (n+2), (n+3)]
  let set2 := [(n+1), (n+2), (n+3), (n+4), (n+5), (n+6), (n+7)]
  let S1 := set1.sum
  let S2 := set2.sum
  S2 - S1 = 28 :=
by
  let set1 := [(n-3), (n-2), (n-1), n, (n+1), (n+2), (n+3)]
  let set2 := [(n+1), (n+2), (n+3), (n+4), (n+5), (n+6), (n+7)]
  let S1 := set1.sum
  let S2 := set2.sum
  have hS1 : S1 = (n-3) + (n-2) + (n-1) + n + (n+1) + (n+2) + (n+3) := by sorry
  have hS2 : S2 = (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) + (n+7) := by sorry
  have h_diff : S2 - S1 = 28 := by sorry
  exact h_diff

end sum_difference_of_consecutive_integers_l246_246168


namespace find_y_value_l246_246997

theorem find_y_value (a y : ℕ) (h1 : (15^2) * y^3 / 256 = a) (h2 : a = 450) : y = 8 := 
by 
  sorry

end find_y_value_l246_246997


namespace opposite_of_negative_five_l246_246935

theorem opposite_of_negative_five : ∃ x : ℤ, -5 + x = 0 ∧ x = 5 :=
by
  use 5
  split
  {
    sorry,
  }
  {
    refl,
  }

end opposite_of_negative_five_l246_246935


namespace strictly_increasing_interval_l246_246335

noncomputable def f (x : ℝ) : ℝ :=
  Real.logb (1/3) (x^2 - 4 * x + 3)

theorem strictly_increasing_interval : ∀ x y : ℝ, x < 1 → y < 1 → x < y → f x < f y :=
by
  sorry

end strictly_increasing_interval_l246_246335


namespace average_speed_round_trip_l246_246307

theorem average_speed_round_trip (v1 v2 : ℝ) (h1 : v1 = 60) (h2 : v2 = 100) :
  (2 * v1 * v2) / (v1 + v2) = 75 :=
by
  sorry

end average_speed_round_trip_l246_246307


namespace erasers_pens_markers_cost_l246_246041

theorem erasers_pens_markers_cost 
  (E P M : ℝ)
  (h₁ : E + 3 * P + 2 * M = 240)
  (h₂ : 2 * E + 4 * M + 5 * P = 440) :
  3 * E + 4 * P + 6 * M = 520 :=
sorry

end erasers_pens_markers_cost_l246_246041


namespace expression_for_C_value_of_C_l246_246811

variables (x y : ℝ)

-- Definitions based on the given conditions
def A := x^2 - 2 * x * y + y^2
def B := x^2 + 2 * x * y + y^2

-- The algebraic expression for C
def C := - x^2 + 10 * x * y - y^2

-- Prove that the expression for C is correct
theorem expression_for_C (h : 3 * A x y - 2 * B x y + C x y = 0) : 
  C x y = - x^2 + 10 * x * y - y^2 := 
by {
  sorry
}

-- Prove the value of C when x = 1/2 and y = -2
theorem value_of_C : C (1/2) (-2) = -57/4 :=
by {
  sorry
}

end expression_for_C_value_of_C_l246_246811


namespace area_ratio_triangle_l246_246332

noncomputable def area_ratio (x y : ℝ) (n m : ℕ) : ℝ :=
(x * y) / (2 * n) / ((x * y) / (2 * m))

theorem area_ratio_triangle (x y : ℝ) (n m : ℕ) (hx : 0 ≤ x) (hy : 0 ≤ y) : 
  area_ratio x y n m = (m : ℝ) / (n : ℝ) := by
  sorry

end area_ratio_triangle_l246_246332


namespace tangent_line_parabola_l246_246498

theorem tangent_line_parabola (d : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) → d = 1 :=
by
  sorry

end tangent_line_parabola_l246_246498


namespace number_of_perfect_square_factors_of_180_l246_246089

theorem number_of_perfect_square_factors_of_180 :
  let prime_factors := (2, 2, 1); 
  let conditions (p1 p2 p3 : ℕ) := 
    p1 ∈ {0, 2} ∧ 
    p2 ∈ {0, 2} ∧ 
    p3 ∈ {0}
  in Σ' x, x ∈ { (p1, p2, p3) | conditions p1 p2 p3 } = 4 :=
begin
  sorry
end

end number_of_perfect_square_factors_of_180_l246_246089


namespace cube_truncation_edges_l246_246487

-- Define the initial condition: a cube
def initial_cube_edges : ℕ := 12

-- Define the condition of each corner being cut off
def corners_cut (corners : ℕ) (edges_added : ℕ) : ℕ :=
  corners * edges_added

-- Define the proof problem
theorem cube_truncation_edges : initial_cube_edges + corners_cut 8 3 = 36 := by
  sorry

end cube_truncation_edges_l246_246487


namespace inequality_proof_l246_246232

variable {a b c : ℝ}

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a * b * c = 1) : 
  (1 / (a^2 * (b + c))) + (1 / (b^2 * (c + a))) + (1 / (c^2 * (a + b))) ≥ 3 / 2 :=
sorry

end inequality_proof_l246_246232


namespace solve_inequality_l246_246881

theorem solve_inequality (a x : ℝ) : 
  (a < 0 → (x ≤ 3 / a ∨ x ≥ 1) ↔ ax^2 - (a + 3) * x + 3 ≤ 0) ∧
  (a = 0 → (x ≥ 1) ↔ ax^2 - (a + 3) * x + 3 ≤ 0) ∧
  (0 < a ∧ a < 3 → (1 ≤ x ∧ x ≤ 3 / a) ↔ ax^2 - (a + 3) * x + 3 ≤ 0) ∧
  (a = 3 → (x = 1) ↔ ax^2 - (a + 3) * x + 3 ≤ 0) ∧
  (a > 3 → (3 / a ≤ x ∧ x ≤ 1) ↔ ax^2 - (a + 3) * x + 3 ≤ 0) :=
  sorry

end solve_inequality_l246_246881


namespace equally_likely_events_A_and_B_l246_246402

-- Given conditions definitions
def number_of_balls (N : ℕ) := N
def main_draw_balls (N : ℕ) := (finset.range (N + 1)).powerset_len 10
def additional_draw_balls (N : ℕ) := (finset.range (N + 1)).powerset_len 8
def lucky_sum_main (s : finset ℕ) := s.sum = 63
def lucky_sum_additional (s : finset ℕ) := s.sum = 44

-- Prove the events A and B are equally likely if and only if N = 18
theorem equally_likely_events_A_and_B (N : ℕ) :
  (∃ s ∈ main_draw_balls N, lucky_sum_main s) ∧ (∃ t ∈ additional_draw_balls N, lucky_sum_additional t)
  → (N = 18) :=
sorry

end equally_likely_events_A_and_B_l246_246402


namespace number_of_slices_l246_246863

theorem number_of_slices 
  (pepperoni ham sausage total_meat pieces_per_slice : ℕ)
  (h1 : pepperoni = 30)
  (h2 : ham = 2 * pepperoni)
  (h3 : sausage = pepperoni + 12)
  (h4 : total_meat = pepperoni + ham + sausage)
  (h5 : pieces_per_slice = 22) :
  total_meat / pieces_per_slice = 6 :=
by
  sorry

end number_of_slices_l246_246863


namespace right_triangle_leg_length_l246_246390

theorem right_triangle_leg_length (a c b : ℝ) (h : a = 4) (h₁ : c = 5) (h₂ : a^2 + b^2 = c^2) : b = 3 := 
by
  -- by is used for the proof, which we are skipping using sorry.
  sorry

end right_triangle_leg_length_l246_246390


namespace find_sheets_used_l246_246136

variable (x y : ℕ) -- define variables for x and y
variable (h₁ : 82 - x = y) -- 82 - x = number of sheets left
variable (h₂ : y = x - 6) -- number of sheets left = number of sheets used - 6

theorem find_sheets_used (h₁ : 82 - x = x - 6) : x = 44 := 
by
  sorry

end find_sheets_used_l246_246136


namespace benny_cards_left_l246_246982

theorem benny_cards_left (n : ℕ) : ℕ :=
  (n + 4) / 2

end benny_cards_left_l246_246982


namespace max_sequence_length_l246_246206

def valid_sequence (s : ℕ → ℤ) : Prop :=
  (∀ n : ℕ, 0 ≤ n ∧ n < 7 → ∑ i in (Finset.range 7), s (n + i) > 0) ∧
  (∀ n : ℕ, 0 ≤ n ∧ n < 11 → ∑ i in (Finset.range 11), s (n + i) < 0)

theorem max_sequence_length {s : ℕ → ℤ} (h : valid_sequence s) : 
  ∃ (n : ℕ), n = 16 ∧ (¬ ∃ m > 16, valid_sequence (λ i, s (i % m))) :=
sorry

end max_sequence_length_l246_246206


namespace min_x_plus_y_l246_246220

theorem min_x_plus_y (x y : ℝ) (h1 : x * y = 2 * x + y + 2) (h2 : x > 1) :
  x + y ≥ 7 :=
sorry

end min_x_plus_y_l246_246220


namespace melted_mixture_weight_l246_246165

theorem melted_mixture_weight (Z C : ℝ) (ratio : 9 / 11 = Z / C) (zinc_weight : Z = 28.8) : Z + C = 64 :=
by
  sorry

end melted_mixture_weight_l246_246165


namespace remainder_of_prime_division_l246_246078

theorem remainder_of_prime_division
  (p : ℕ) (hp : Nat.Prime p)
  (r : ℕ) (hr : r = p % 210) 
  (hcomp : ¬ Nat.Prime r)
  (hsum : ∃ a b : ℕ, r = a^2 + b^2) : 
  r = 169 := 
sorry

end remainder_of_prime_division_l246_246078


namespace t_in_base_c_l246_246627

theorem t_in_base_c (c : ℕ) (t : ℕ) :
  ((c + 4) * (c + 7) * (c + 9) = 5 * c^3 + 2 * c^2 + 4 * c + 3) →
  t = (14 + 17 + 19 : ℕ) →
  c = 11 →
  (t : ℕ) = 49 :=
by
  intros h_product h_sum h_c
  rw [h_c, h_sum]
  sorry

end t_in_base_c_l246_246627


namespace rectangle_area_l246_246478

theorem rectangle_area (x : ℝ) (l : ℝ) (h1 : 3 * l = x^2 / 10) : 
  3 * l^2 = 3 * x^2 / 10 :=
by sorry

end rectangle_area_l246_246478


namespace total_cakes_served_today_l246_246316

def cakes_served_lunch : ℕ := 6
def cakes_served_dinner : ℕ := 9
def total_cakes_served (lunch cakes_served_dinner : ℕ) : ℕ :=
  lunch + cakes_served_dinner

theorem total_cakes_served_today : total_cakes_served cakes_served_lunch cakes_served_dinner = 15 := 
by
  sorry

end total_cakes_served_today_l246_246316


namespace words_on_each_page_l246_246474

/-- Given a book with 150 pages, where each page has between 50 and 150 words, 
    and the total number of words in the book is congruent to 217 modulo 221, 
    prove that each page has 135 words. -/
theorem words_on_each_page (p : ℕ) (h1 : 50 ≤ p) (h2 : p ≤ 150) (h3 : 150 * p ≡ 217 [MOD 221]) : 
  p = 135 :=
by
  sorry

end words_on_each_page_l246_246474


namespace floor_sqrt_245_l246_246632

theorem floor_sqrt_245 : (Int.floor (Real.sqrt 245)) = 15 :=
by
  sorry

end floor_sqrt_245_l246_246632


namespace operation_correct_l246_246055

def operation (a b : ℝ) : ℝ := 2 * a + 5 * b

theorem operation_correct :
  operation 4 2 = 18 :=
by
  show 2 * 4 + 5 * 2 = 18
  sorry

end operation_correct_l246_246055


namespace find_equation_of_line_l246_246346

variable (x y : ℝ)

def line_parallel (x y : ℝ) (m : ℝ) :=
  x - 2*y + m = 0

def line_through_point (x y : ℝ) (px py : ℝ) (m : ℝ) :=
  (px - 2 * py + m = 0)
  
theorem find_equation_of_line :
  let px := -1
  let py := 3
  ∃ m, line_parallel x y m ∧ line_through_point x y px py m ∧ m = 7 :=
by
  sorry

end find_equation_of_line_l246_246346


namespace milk_total_correct_l246_246214

def chocolate_milk : Nat := 2
def strawberry_milk : Nat := 15
def regular_milk : Nat := 3
def total_milk : Nat := chocolate_milk + strawberry_milk + regular_milk

theorem milk_total_correct : total_milk = 20 := by
  sorry

end milk_total_correct_l246_246214


namespace number_of_valid_pairs_l246_246079

theorem number_of_valid_pairs : 
  ∃ (n : ℕ), n = 1995003 ∧ (∃ b c : ℤ, c < 2000 ∧ b > 2 ∧ (∀ x : ℂ, x^2 - (b:ℝ) * x + (c:ℝ) = 0 → x.re > 1)) := 
sorry

end number_of_valid_pairs_l246_246079


namespace range_of_a_l246_246365

theorem range_of_a (a : ℝ) (h : 2 * a - 1 ≤ 11) : a < 6 :=
by
  sorry

end range_of_a_l246_246365


namespace number_added_is_10_l246_246586

-- Define the conditions.
def number_thought_of : ℕ := 55
def result : ℕ := 21

-- Define the statement of the problem.
theorem number_added_is_10 : ∃ (y : ℕ), (number_thought_of / 5 + y = result) ∧ (y = 10) := by
  sorry

end number_added_is_10_l246_246586


namespace largest_divisible_by_7_l246_246794

theorem largest_divisible_by_7 :
  ∃ (A B C : ℕ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  let n := 10000 * A + 1000 * B + 100 * B + 10 * C + A in
  (n ≡ 0 [MOD 7]) ∧ n = 98879 :=
sorry

end largest_divisible_by_7_l246_246794


namespace Chris_age_l246_246431

variable (a b c : ℕ)

theorem Chris_age : a + b + c = 36 ∧ b = 2*c + 9 ∧ b = a → c = 4 :=
by
  sorry

end Chris_age_l246_246431


namespace m_n_sum_l246_246103

theorem m_n_sum (m n : ℝ) (h : ∀ x : ℝ, x^2 + m * x + 6 = (x - 2) * (x - n)) : m + n = -2 :=
by
  sorry

end m_n_sum_l246_246103


namespace maria_zoo_ticket_discount_percentage_l246_246694

theorem maria_zoo_ticket_discount_percentage 
  (regular_price : ℝ) (paid_price : ℝ) (discount_percentage : ℝ)
  (h1 : regular_price = 15) (h2 : paid_price = 9) :
  discount_percentage = 40 :=
by
  sorry

end maria_zoo_ticket_discount_percentage_l246_246694


namespace find_omega_find_period_and_intervals_find_solution_set_l246_246822

noncomputable def omega_condition (ω : ℝ) :=
  0 < ω ∧ ω < 2

noncomputable def function_fx (ω : ℝ) (x : ℝ) := 
  3 * Real.sin (2 * ω * x + Real.pi / 3)

noncomputable def center_of_symmetry_condition (ω : ℝ) := 
  function_fx ω (-Real.pi / 6) = 0

noncomputable def period_condition (ω : ℝ) :=
  Real.pi / abs ω

noncomputable def intervals_of_increase (ω : ℝ) (x : ℝ) : Prop :=
  ∃ k : ℤ, ((Real.pi / 12 + k * Real.pi) ≤ x) ∧ (x < (5 * Real.pi / 12 + k * Real.pi))

noncomputable def solution_set_fx_ge_half (x : ℝ) : Prop :=
  ∃ k : ℤ, (Real.pi / 12 + k * Real.pi) ≤ x ∧ (x ≤ 5 * Real.pi / 12 + k * Real.pi)

theorem find_omega : ∀ ω : ℝ, omega_condition ω ∧ center_of_symmetry_condition ω → omega = 1 := sorry

theorem find_period_and_intervals : 
  ∀ ω : ℝ, omega_condition ω ∧ (ω = 1) → period_condition ω = Real.pi :=
sorry

theorem find_solution_set :
  ∀ ω : ℝ, omega_condition ω ∧ (ω = 1) → (∀ x, solution_set_fx_ge_half x) :=
sorry

end find_omega_find_period_and_intervals_find_solution_set_l246_246822


namespace opposite_of_neg_five_is_five_l246_246921

-- Define the condition for the opposite of a number
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- State the theorem that the opposite of -5 is 5
theorem opposite_of_neg_five_is_five : is_opposite (-5) 5 :=
by
  -- Insert the proof here
  sorry

end opposite_of_neg_five_is_five_l246_246921


namespace red_cookies_count_l246_246564

-- Definitions of the conditions
def total_cookies : ℕ := 86
def pink_cookies : ℕ := 50

-- The proof problem statement
theorem red_cookies_count : ∃ y : ℕ, y = total_cookies - pink_cookies := by
  use 36
  show 36 = total_cookies - pink_cookies
  sorry

end red_cookies_count_l246_246564


namespace base_length_of_parallelogram_l246_246804

theorem base_length_of_parallelogram (A : ℕ) (H : ℕ) (Base : ℕ) (hA : A = 576) (hH : H = 48) (hArea : A = Base * H) : 
  Base = 12 := 
by 
  -- We skip the proof steps since we only need to provide the Lean theorem statement.
  sorry

end base_length_of_parallelogram_l246_246804


namespace geometric_series_first_term_l246_246044

theorem geometric_series_first_term (r : ℚ) (S : ℚ) (a : ℚ) (h_r : r = 1 / 4) (h_S : S = 80) (h_sum : S = a / (1 - r)) : a = 60 :=
by {
  sorry
}

end geometric_series_first_term_l246_246044


namespace smallest_integer_with_18_divisors_l246_246742

theorem smallest_integer_with_18_divisors : 
  ∃ n : ℕ, (n > 0 ∧ (nat.divisors_count n = 18) ∧ (∀ m : ℕ, m > 0 ∧ (nat.divisors_count m = 18) → n ≤ m)) := 
sorry

end smallest_integer_with_18_divisors_l246_246742


namespace find_p_l246_246547

theorem find_p (p : ℝ) (h1 : (1/2) * 15 * (3 + 15) - ((1/2) * 3 * (15 - p) + (1/2) * 15 * p) = 40) : 
  p = 12.0833 :=
by sorry

end find_p_l246_246547


namespace first_term_of_geometric_sequence_l246_246436

theorem first_term_of_geometric_sequence (a r : ℕ) :
  (a * r ^ 3 = 54) ∧ (a * r ^ 4 = 162) → a = 2 :=
by
  -- Provided conditions and the goal
  sorry

end first_term_of_geometric_sequence_l246_246436


namespace sufficient_but_not_necessary_l246_246763

theorem sufficient_but_not_necessary (a : ℝ) :
  0 < a ∧ a < 1 → (∀ x : ℝ, ax^2 + 2 * a * x + 1 > 0) ∧ ¬ (∀ a, (∀ x : ℝ, ax^2 + 2 * a * x + 1 > 0) → 0 < a ∧ a < 1) :=
by
  sorry

end sufficient_but_not_necessary_l246_246763


namespace third_pipe_empty_time_l246_246011

theorem third_pipe_empty_time :
  let A_rate := 1/60
  let B_rate := 1/75
  let combined_rate := 1/50
  let third_pipe_rate := combined_rate - (A_rate + B_rate)
  let time_to_empty := 1 / third_pipe_rate
  time_to_empty = 100 :=
by
  sorry

end third_pipe_empty_time_l246_246011


namespace Alex_Hours_Upside_Down_Per_Month_l246_246319

-- Define constants and variables based on the conditions
def AlexCurrentHeight : ℝ := 48
def AlexRequiredHeight : ℝ := 54
def NormalGrowthPerMonth : ℝ := 1 / 3
def UpsideDownGrowthPerHour : ℝ := 1 / 12
def MonthsInYear : ℕ := 12

-- Compute total growth needed and additional required growth terms
def TotalGrowthNeeded : ℝ := AlexRequiredHeight - AlexCurrentHeight
def NormalGrowthInYear : ℝ := NormalGrowthPerMonth * MonthsInYear
def AdditionalGrowthNeeded : ℝ := TotalGrowthNeeded - NormalGrowthInYear
def TotalUpsideDownHours : ℝ := AdditionalGrowthNeeded * 12
def UpsideDownHoursPerMonth : ℝ := TotalUpsideDownHours / MonthsInYear

-- The statement to prove
theorem Alex_Hours_Upside_Down_Per_Month : UpsideDownHoursPerMonth = 2 := by
  sorry

end Alex_Hours_Upside_Down_Per_Month_l246_246319


namespace Tom_total_spend_l246_246801

theorem Tom_total_spend :
  let notebook_price := 2
  let notebook_discount := 0.75
  let notebook_count := 4
  let magazine_price := 5
  let magazine_count := 2
  let pen_price := 1.50
  let pen_discount := 0.75
  let pen_count := 3
  let book_price := 12
  let book_count := 1
  let discount_threshold := 30
  let coupon_discount := 10
  let total_cost :=
    (notebook_count * (notebook_price * notebook_discount)) +
    (magazine_count * magazine_price) +
    (pen_count * (pen_price * pen_discount)) +
    (book_count * book_price)
  let final_cost := if total_cost >= discount_threshold then total_cost - coupon_discount else total_cost
  final_cost = 21.375 :=
by
  sorry

end Tom_total_spend_l246_246801


namespace altitude_of_triangle_l246_246601

theorem altitude_of_triangle (x : ℝ) (h : ℝ) 
  (h1 : x^2 = (1/2) * x * h) : h = 2 * x :=
by
  sorry

end altitude_of_triangle_l246_246601


namespace sequence_property_l246_246353

theorem sequence_property (a : ℕ+ → ℤ) (h_add : ∀ p q : ℕ+, a (p + q) = a p + a q) (h_a2 : a 2 = -6) :
  a 10 = -30 := 
sorry

end sequence_property_l246_246353


namespace equilateral_triangle_ratio_correct_l246_246735

noncomputable def equilateral_triangle_ratio : ℝ :=
  let side_length := 12
  let perimeter := 3 * side_length
  let altitude := side_length * (Real.sqrt 3 / 2)
  let area := (side_length * altitude) / 2
  area / perimeter

theorem equilateral_triangle_ratio_correct :
  equilateral_triangle_ratio = Real.sqrt 3 :=
sorry

end equilateral_triangle_ratio_correct_l246_246735


namespace maximal_length_sequence_l246_246205

theorem maximal_length_sequence :
  ∃ (a : ℕ → ℤ) (n : ℕ), (∀ i, 1 ≤ i → i + 6 ≤ n → (a i + a (i + 1) + a (i + 2) + a (i + 3) + a (i + 4) + a (i + 5) + a (i + 6) > 0)) ∧ 
                          (∀ j, 1 ≤ j → j + 10 ≤ n → (a j + a (j + 1) + a (j + 2) + a (j + 3) + a (j + 4) + a (j + 5) + a (j + 6) + a (j + 7) + a (j + 8) + a (j + 9) + a (j + 10) < 0)) ∧ 
                          n = 16 :=
sorry

end maximal_length_sequence_l246_246205


namespace angle_F_measure_l246_246832

-- Define angle B
def angle_B := 120

-- Define angle C being supplementary to angle B on a straight line
def angle_C := 180 - angle_B

-- Define angle D
def angle_D := 45

-- Define angle E
def angle_E := 30

-- Define the vertically opposite angle F to angle C
def angle_F := angle_C

theorem angle_F_measure : angle_F = 60 :=
by
  -- Provide a proof by specifying sorry to indicate the proof is not complete
  sorry

end angle_F_measure_l246_246832


namespace semicircle_area_in_quarter_circle_l246_246415

theorem semicircle_area_in_quarter_circle (r : ℝ) (A : ℝ) (π : ℝ) (one : ℝ) :
    r = 1 / (Real.sqrt (2) + 1) →
    A = π * r^2 →
    120 * A / π = 20 :=
sorry

end semicircle_area_in_quarter_circle_l246_246415


namespace lawrence_worked_hours_l246_246853

-- Let h_M, h_T, h_F be the hours worked on Monday, Tuesday, and Friday respectively
-- Let h_W be the hours worked on Wednesday (h_W = 5.5)
-- Let h_R be the hours worked on Thursday (h_R = 5.5)
-- Let total hours worked in 5 days be 25
-- Prove that h_M + h_T + h_F = 14

theorem lawrence_worked_hours :
  ∀ (h_M h_T h_F : ℝ), h_W = 5.5 → h_R = 5.5 → (5 * 5 = 25) → 
  h_M + h_T + h_F + h_W + h_R = 25 → h_M + h_T + h_F = 14 :=
by
  intros h_M h_T h_F h_W h_R h_total h_sum
  sorry

end lawrence_worked_hours_l246_246853


namespace problem_solution_set_l246_246814

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^3 - 8 else (-x)^3 - 8

theorem problem_solution_set : 
  { x : ℝ | f (x-2) > 0 } = {x : ℝ | x < 0} ∪ {x : ℝ | x > 4} :=
by sorry

end problem_solution_set_l246_246814


namespace maximum_value_expression_l246_246588

-- Definitions
def f (x : ℝ) := -3 * x^2 + 18 * x - 1

-- Lean statement to prove that the maximum value of the function f is 26.
theorem maximum_value_expression : ∃ x : ℝ, f x = 26 :=
sorry

end maximum_value_expression_l246_246588


namespace g_is_odd_l246_246406

def g (x : ℝ) : ℝ := (1 / (3^x - 1)) + (1 / 3)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) :=
by
  sorry

end g_is_odd_l246_246406


namespace train_speed_in_km_per_hr_l246_246180

-- Define the conditions
def length_of_train : ℝ := 100 -- length in meters
def time_to_cross_pole : ℝ := 6 -- time in seconds

-- Define the conversion factor from meters/second to kilometers/hour
def conversion_factor : ℝ := 18 / 5

-- Define the formula for speed calculation
def speed_of_train := (length_of_train / time_to_cross_pole) * conversion_factor

-- The theorem to be proven
theorem train_speed_in_km_per_hr : speed_of_train = 50 := by
  sorry

end train_speed_in_km_per_hr_l246_246180


namespace total_cost_of_supplies_l246_246040

variable (E P M : ℝ)

open Real

theorem total_cost_of_supplies (h1 : E + 3 * P + 2 * M = 240)
                                (h2 : 2 * E + 4 * M + 5 * P = 440)
                                : 3 * E + 4 * P + 6 * M = 520 := 
sorry

end total_cost_of_supplies_l246_246040


namespace find_son_age_l246_246716

theorem find_son_age (F S : ℕ) (h1 : F + S = 55)
  (h2 : ∃ Y, S + Y = F ∧ (F + Y) + (S + Y) = 93)
  (h3 : F = 18 ∨ S = 18) : S = 18 :=
by
  sorry  -- Proof to be filled in

end find_son_age_l246_246716


namespace find_k_l246_246209

theorem find_k (k : ℝ) : -x^2 - (k + 10) * x - 8 = -(x - 2) * (x - 4) → k = -16 := by
  intro h
  sorry

end find_k_l246_246209


namespace total_suitcases_l246_246558

-- Definitions based on the conditions in a)
def siblings : ℕ := 4
def suitcases_per_sibling : ℕ := 2
def parents : ℕ := 2
def suitcases_per_parent : ℕ := 3
def suitcases_per_Lily : ℕ := 0

-- The statement to be proved
theorem total_suitcases : (siblings * suitcases_per_sibling) + (parents * suitcases_per_parent) + suitcases_per_Lily = 14 :=
by
  sorry

end total_suitcases_l246_246558


namespace number_of_tons_is_3_l246_246882

noncomputable def calculate_tons_of_mulch {total_cost price_per_pound pounds_per_ton : ℝ} 
  (h_total_cost : total_cost = 15000) 
  (h_price_per_pound : price_per_pound = 2.5) 
  (h_pounds_per_ton : pounds_per_ton = 2000) : ℝ := 
  total_cost / price_per_pound / pounds_per_ton

theorem number_of_tons_is_3 
  (total_cost price_per_pound pounds_per_ton : ℝ) 
  (h_total_cost : total_cost = 15000) 
  (h_price_per_pound : price_per_pound = 2.5) 
  (h_pounds_per_ton : pounds_per_ton = 2000) : 
  calculate_tons_of_mulch h_total_cost h_price_per_pound h_pounds_per_ton = 3 := 
by
  sorry

end number_of_tons_is_3_l246_246882


namespace simplify_sqrt_24_l246_246571

theorem simplify_sqrt_24 : Real.sqrt 24 = 2 * Real.sqrt 6 :=
sorry

end simplify_sqrt_24_l246_246571


namespace polar_to_cartesian_equiv_l246_246629

noncomputable def polar_to_cartesian (rho theta : ℝ) : Prop :=
  let x := rho * Real.cos theta
  let y := rho * Real.sin theta
  (Real.sqrt 3 * x + y = 2) ↔ (rho * Real.cos (theta - Real.pi / 6) = 1)

theorem polar_to_cartesian_equiv (rho theta : ℝ) : polar_to_cartesian rho theta :=
by
  sorry

end polar_to_cartesian_equiv_l246_246629


namespace radius_of_sphere_is_two_sqrt_46_l246_246975

theorem radius_of_sphere_is_two_sqrt_46
  (a b c : ℝ)
  (s : ℝ)
  (h1 : 4 * (a + b + c) = 160)
  (h2 : 2 * (a * b + b * c + c * a) = 864)
  (h3 : s = Real.sqrt ((a^2 + b^2 + c^2) / 4)) :
  s = 2 * Real.sqrt 46 :=
by
  -- proof placeholder
  sorry

end radius_of_sphere_is_two_sqrt_46_l246_246975


namespace round_to_nearest_whole_l246_246570

theorem round_to_nearest_whole (x : ℝ) (hx : x = 12345.49999) : round x = 12345 := by
  -- Proof omitted.
  sorry

end round_to_nearest_whole_l246_246570


namespace bob_homework_time_l246_246619

variable (T_Alice T_Bob : ℕ)

theorem bob_homework_time (h_Alice : T_Alice = 40) (h_Bob : T_Bob = (3 * T_Alice) / 8) : T_Bob = 15 :=
by
  rw [h_Alice] at h_Bob
  norm_num at h_Bob
  exact h_Bob

-- Assuming T_Alice represents the time taken by Alice to complete her homework
-- and T_Bob represents the time taken by Bob to complete his homework,
-- we prove that T_Bob is 15 minutes given the conditions.

end bob_homework_time_l246_246619


namespace sum_of_remainders_l246_246755

theorem sum_of_remainders (n : ℤ) (h : n % 18 = 11) : 
  (n % 3 + n % 6) = 7 :=
by
  sorry

end sum_of_remainders_l246_246755


namespace kids_bike_wheels_l246_246623

theorem kids_bike_wheels
  (x : ℕ) 
  (h1 : 7 * 2 + 11 * x = 58) :
  x = 4 :=
sorry

end kids_bike_wheels_l246_246623


namespace triangle_sides_angles_l246_246123

open Real

variables {a b c : ℝ} {α β γ : ℝ}

theorem triangle_sides_angles
  (triangle_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (triangle_angles : α > 0 ∧ β > 0 ∧ γ > 0)
  (angles_sum : α + β + γ = π)
  (condition : 3 * α + 2 * β = π) :
  a^2 + b * c - c^2 = 0 :=
sorry

end triangle_sides_angles_l246_246123


namespace quadrilateral_area_l246_246344

theorem quadrilateral_area (d h1 h2 : ℝ) (hd : d = 40) (hh1 : h1 = 9) (hh2 : h2 = 6) :
  1 / 2 * d * h1 + 1 / 2 * d * h2 = 300 := 
by
  sorry

end quadrilateral_area_l246_246344


namespace perpendicular_lines_b_value_l246_246438

theorem perpendicular_lines_b_value :
  ( ∀ x y : ℝ, 2 * x + 3 * y + 4 = 0)  →
  ( ∀ x y : ℝ, b * x + 3 * y - 1 = 0) →
  ( - (2 : ℝ) / (3 : ℝ) * - b / (3 : ℝ) = -1 ) →
  b = - (9 : ℝ) / (2 : ℝ) :=
by
  intros h1 h2 h3
  sorry

end perpendicular_lines_b_value_l246_246438


namespace sector_radius_l246_246287

theorem sector_radius (A L : ℝ) (hA : A = 240 * Real.pi) (hL : L = 20 * Real.pi) : 
  ∃ r : ℝ, r = 24 :=
by
  sorry

end sector_radius_l246_246287


namespace Andy_has_4_more_candies_than_Caleb_l246_246329

-- Define the initial candies each person has
def Billy_initial_candies : ℕ := 6
def Caleb_initial_candies : ℕ := 11
def Andy_initial_candies : ℕ := 9

-- Define the candies bought by the father and their distribution
def father_bought_candies : ℕ := 36
def Billy_received_from_father : ℕ := 8
def Caleb_received_from_father : ℕ := 11

-- Calculate the remaining candies for Andy after distribution
def Andy_received_from_father : ℕ := father_bought_candies - (Billy_received_from_father + Caleb_received_from_father)

-- Calculate the total candies each person has
def Billy_total_candies : ℕ := Billy_initial_candies + Billy_received_from_father
def Caleb_total_candies : ℕ := Caleb_initial_candies + Caleb_received_from_father
def Andy_total_candies : ℕ := Andy_initial_candies + Andy_received_from_father

-- Prove that Andy has 4 more candies than Caleb
theorem Andy_has_4_more_candies_than_Caleb :
  Andy_total_candies = Caleb_total_candies + 4 :=
by {
  -- Skipping the proof
  sorry
}

end Andy_has_4_more_candies_than_Caleb_l246_246329


namespace pizza_order_l246_246312

theorem pizza_order (couple_want: ℕ) (child_want: ℕ) (num_couples: ℕ) (num_children: ℕ) (slices_per_pizza: ℕ)
  (hcouple: couple_want = 3) (hchild: child_want = 1) (hnumc: num_couples = 1) (hnumch: num_children = 6) (hsp: slices_per_pizza = 4) :
  (couple_want * 2 * num_couples + child_want * num_children) / slices_per_pizza = 3 := 
by
  -- Proof here
  sorry

end pizza_order_l246_246312


namespace find_k_l246_246239

-- Define the problem parameters
variables {x y k : ℝ}

-- The conditions given in the problem
def system_of_equations (x y k : ℝ) : Prop :=
  (x + 2 * y = k - 1) ∧ (2 * x + y = 5 * k + 4)

def solution_condition (x y : ℝ) : Prop :=
  x + y = 5

-- The proof statement
theorem find_k (x y k : ℝ) (h1 : system_of_equations x y k) (h2 : solution_condition x y) :
  k = 2 :=
sorry

end find_k_l246_246239


namespace combined_cost_of_items_l246_246197

theorem combined_cost_of_items (wallet_cost : ℕ) 
  (purse_cost : ℕ) (combined_cost : ℕ) :
  wallet_cost = 22 →
  purse_cost = 4 * wallet_cost - 3 →
  combined_cost = wallet_cost + purse_cost →
  combined_cost = 107 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end combined_cost_of_items_l246_246197


namespace sqrt_four_eq_two_l246_246190

theorem sqrt_four_eq_two : Real.sqrt 4 = 2 :=
by
  sorry

end sqrt_four_eq_two_l246_246190


namespace not_all_inequalities_true_l246_246132

theorem not_all_inequalities_true (a b c : ℝ) (h₀ : 0 < a ∧ a < 1) (h₁ : 0 < b ∧ b < 1) (h₂ : 0 < c ∧ c < 1) :
  ¬(a * (1 - b) > 1 / 4 ∧ b * (1 - c) > 1 / 4 ∧ c * (1 - a) > 1 / 4) :=
  sorry

end not_all_inequalities_true_l246_246132


namespace difference_of_squares_divisible_by_9_l246_246630

theorem difference_of_squares_divisible_by_9 (a b : ℤ) : ∃ k : ℤ, (3 * a + 2)^2 - (3 * b + 2)^2 = 9 * k := by
  sorry

end difference_of_squares_divisible_by_9_l246_246630


namespace number_of_perfect_square_factors_of_180_l246_246092

-- Define the prime factorization of 180
def prime_factorization_180 : List (ℕ × ℕ) := [(2, 2), (3, 2), (5, 1)]

-- Define what it means to be a factor of 180
def is_factor_of_180 (n : ℕ) : Prop :=
  ∀ p k, (p, k) ∈ prime_factorization_180 → ∃ m, n = p ^ m ∧ m ≤ k

-- Define what it means to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∀ p m, ∃ k, n = p ^ m ∧ even m

-- Function to count factors of 180 that are perfect squares
def count_perfect_square_factors_of_180 : ℕ :=
  List.length [d | d ← List.range 181, is_factor_of_180 d ∧ is_perfect_square d]

-- Main theorem
theorem number_of_perfect_square_factors_of_180 : count_perfect_square_factors_of_180 = 4 := 
by sorry

end number_of_perfect_square_factors_of_180_l246_246092


namespace function_C_is_quadratic_l246_246161

def isQuadratic (f : ℝ → ℝ) :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, f x = a * x^2 + b * x + c

def function_C (x : ℝ) : ℝ := (x + 1)^2 - 5

theorem function_C_is_quadratic : isQuadratic function_C :=
by
  sorry

end function_C_is_quadratic_l246_246161


namespace greatest_n_4022_l246_246857

noncomputable def arithmetic_sequence_greatest_n 
  (a : ℕ → ℝ)
  (a1_pos : a 1 > 0)
  (cond1 : a 2011 + a 2012 > 0)
  (cond2 : a 2011 * a 2012 < 0) : ℕ :=
  4022

theorem greatest_n_4022 
  (a : ℕ → ℝ)
  (h_arith : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h1 : a 1 > 0)
  (h2 : a 2011 + a 2012 > 0)
  (h3 : a 2011 * a 2012 < 0):
  arithmetic_sequence_greatest_n a h1 h2 h3 = 4022 :=
sorry

end greatest_n_4022_l246_246857


namespace smallest_integer_with_18_divisors_l246_246748

def num_divisors (n : ℕ) : ℕ := 
  (factors n).freq.map (λ x => x.2 + 1).prod

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, num_divisors n = 18 ∧ ∀ m : ℕ, num_divisors m = 18 → n ≤ m := 
by
  let n := 180
  use n
  constructor
  ...
  sorry

end smallest_integer_with_18_divisors_l246_246748


namespace middle_number_is_five_l246_246587

theorem middle_number_is_five
  (a b c : ℕ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_sum : a + b + c = 20)
  (h_sorted : a < b ∧ b < c)
  (h_bella : ¬∀ x y z, x + y + z = 20 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x < y ∧ y < z → x = a → y = b ∧ z = c)
  (h_della : ¬∀ x y z, x + y + z = 20 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x < y ∧ y < z → y = b → x = a ∧ z = c)
  (h_nella : ¬∀ x y z, x + y + z = 20 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x < y ∧ y < z → z = c → x = a ∧ y = b) :
  b = 5 := sorry

end middle_number_is_five_l246_246587


namespace no_solution_for_s_l246_246347

theorem no_solution_for_s : ∀ s : ℝ,
  (s^2 - 6 * s + 8) / (s^2 - 9 * s + 20) ≠ (s^2 - 3 * s - 18) / (s^2 - 2 * s - 15) :=
by
  intros s
  sorry

end no_solution_for_s_l246_246347


namespace total_suitcases_correct_l246_246559

-- Conditions as definitions
def num_siblings : Nat := 4
def suitcases_per_sibling : Nat := 2
def num_parents : Nat := 2
def suitcases_per_parent : Nat := 3

-- Total suitcases calculation
def total_suitcases :=
  (num_siblings * suitcases_per_sibling) + (num_parents * suitcases_per_parent)

-- Statement to prove
theorem total_suitcases_correct : total_suitcases = 14 :=
by
  sorry

end total_suitcases_correct_l246_246559


namespace three_digit_numbers_with_repeated_digits_l246_246724

theorem three_digit_numbers_with_repeated_digits :
  let total_three_digit_numbers := 900
  let without_repeats := 9 * 9 * 8
  total_three_digit_numbers - without_repeats = 252 := by
{
  let total_three_digit_numbers := 900
  let without_repeats := 9 * 9 * 8
  show total_three_digit_numbers - without_repeats = 252
  sorry
}

end three_digit_numbers_with_repeated_digits_l246_246724


namespace student_tickets_sold_l246_246949

theorem student_tickets_sold (A S : ℝ) (h1 : A + S = 59) (h2 : 4 * A + 2.5 * S = 222.50) : S = 9 :=
by
  sorry

end student_tickets_sold_l246_246949


namespace marcus_brought_30_peanut_butter_cookies_l246_246111

/-- Jenny brought in 40 peanut butter cookies. -/
def jenny_peanut_butter_cookies := 40

/-- Jenny brought in 50 chocolate chip cookies. -/
def jenny_chocolate_chip_cookies := 50

/-- Marcus brought in 20 lemon cookies. -/
def marcus_lemon_cookies := 20

/-- The total number of non-peanut butter cookies is the sum of chocolate chip and lemon cookies. -/
def non_peanut_butter_cookies := jenny_chocolate_chip_cookies + marcus_lemon_cookies

/-- The total number of peanut butter cookies is Jenny's plus Marcus'. -/
def total_peanut_butter_cookies (marcus_peanut_butter_cookies : ℕ) := jenny_peanut_butter_cookies + marcus_peanut_butter_cookies

/-- If Renee has a 50% chance of picking a peanut butter cookie, the number of peanut butter cookies must equal the number of non-peanut butter cookies. -/
theorem marcus_brought_30_peanut_butter_cookies (x : ℕ) : total_peanut_butter_cookies x = non_peanut_butter_cookies → x = 30 :=
by
  sorry

end marcus_brought_30_peanut_butter_cookies_l246_246111


namespace game_is_unfair_swap_to_make_fair_l246_246543

-- Part 1: Prove the game is unfair
theorem game_is_unfair (y b r : ℕ) (hb : y = 5) (bb : b = 13) (rb : r = 22) :
  ¬((b : ℚ) / (y + b + r) = (y : ℚ) / (y + b + r)) :=
by
  -- The proof is omitted as per the instructions.
  sorry

-- Part 2: Prove that swapping 4 black balls with 4 yellow balls makes the game fair.
theorem swap_to_make_fair (y b r : ℕ) (hb : y = 5) (bb : b = 13) (rb : r = 22) (x: ℕ) :
  x = 4 →
  (b - x : ℚ) / (y + b + r) = (y + x : ℚ) / (y + b + r) :=
by
  -- The proof is omitted as per the instructions.
  sorry

end game_is_unfair_swap_to_make_fair_l246_246543


namespace initial_number_of_rabbits_is_50_l246_246542

-- Initial number of weasels
def initial_weasels := 100

-- Each fox catches 4 weasels and 2 rabbits per week
def weasels_caught_per_fox_per_week := 4
def rabbits_caught_per_fox_per_week := 2

-- There are 3 foxes
def num_foxes := 3

-- After 3 weeks, 96 weasels and rabbits are left
def weasels_and_rabbits_left := 96
def weeks := 3

theorem initial_number_of_rabbits_is_50 :
  (initial_weasels + (initial_weasels + weasels_and_rabbits_left)) - initial_weasels = 50 :=
by
  sorry

end initial_number_of_rabbits_is_50_l246_246542


namespace fixed_point_of_f_l246_246765

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + Real.logb a (|x + 1|)

theorem fixed_point_of_f (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) : f a 0 = 1 :=
by
  sorry

end fixed_point_of_f_l246_246765


namespace min_n_for_factorization_l246_246641

theorem min_n_for_factorization (n : ℤ) :
  (∃ A B : ℤ, 6 * A * B = 60 ∧ n = 6 * B + A) → n = 66 :=
sorry

end min_n_for_factorization_l246_246641


namespace curve_cartesian_eq_correct_intersection_distances_sum_l246_246849

noncomputable section

def curve_parametric_eqns (θ : ℝ) : ℝ × ℝ := 
  (1 + 3 * Real.cos θ, 3 + 3 * Real.sin θ)

def line_parametric_eqns (t : ℝ) : ℝ × ℝ := 
  (3 + (1/2) * t, 3 + (Real.sqrt 3 / 2) * t)

def curve_cartesian_eq (x y : ℝ) : Prop := 
  (x - 1)^2 + (y - 3)^2 = 9

def point_p : ℝ × ℝ := 
  (3, 3)

def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem curve_cartesian_eq_correct (θ : ℝ) : 
  curve_cartesian_eq (curve_parametric_eqns θ).1 (curve_parametric_eqns θ).2 := 
by 
  sorry

theorem intersection_distances_sum (t1 t2 : ℝ) 
  (h1 : curve_cartesian_eq (line_parametric_eqns t1).1 (line_parametric_eqns t1).2) 
  (h2 : curve_cartesian_eq (line_parametric_eqns t2).1 (line_parametric_eqns t2).2) : 
  distance point_p (line_parametric_eqns t1) + distance point_p (line_parametric_eqns t2) = 2 * Real.sqrt 3 := 
by 
  sorry

end curve_cartesian_eq_correct_intersection_distances_sum_l246_246849


namespace interest_rate_is_six_percent_l246_246639

noncomputable def amount : ℝ := 1120
noncomputable def principal : ℝ := 979.0209790209791
noncomputable def time_years : ℝ := 2 + 2 / 5

noncomputable def total_interest (A P: ℝ) : ℝ := A - P

noncomputable def interest_rate_per_annum (I P T: ℝ) : ℝ := I / (P * T) * 100

theorem interest_rate_is_six_percent :
  interest_rate_per_annum (total_interest amount principal) principal time_years = 6 := 
by
  sorry

end interest_rate_is_six_percent_l246_246639


namespace white_space_area_is_31_l246_246448

-- Definitions and conditions from the problem
def board_width : ℕ := 4
def board_length : ℕ := 18
def board_area : ℕ := board_width * board_length

def area_C : ℕ := 4 + 2 + 2
def area_O : ℕ := (4 * 3) - (2 * 1)
def area_D : ℕ := (4 * 3) - (2 * 1)
def area_E : ℕ := 4 + 3 + 3 + 3

def total_black_area : ℕ := area_C + area_O + area_D + area_E

def white_space_area : ℕ := board_area - total_black_area

-- Proof problem statement
theorem white_space_area_is_31 : white_space_area = 31 := by
  sorry

end white_space_area_is_31_l246_246448


namespace find_k_l246_246257

theorem find_k (x : ℝ) (a h k : ℝ) (h1 : 9 * x^2 - 12 * x = a * (x - h)^2 + k) : k = -4 := by
  sorry

end find_k_l246_246257


namespace number_of_ordered_triples_l246_246177

theorem number_of_ordered_triples :
  let b := 2023
  let n := (b ^ 2)
  ∀ (a c : ℕ), a * c = n ∧ a ≤ b ∧ b ≤ c → (∃ (k : ℕ), k = 7) :=
by
  sorry

end number_of_ordered_triples_l246_246177


namespace non_neg_int_solutions_l246_246636

theorem non_neg_int_solutions : 
  ∀ (x y : ℕ), 2 * x ^ 2 + 2 * x * y - x + y = 2020 → 
               (x = 0 ∧ y = 2020) ∨ (x = 1 ∧ y = 673) :=
by
  sorry

end non_neg_int_solutions_l246_246636


namespace opposite_of_neg5_is_pos5_l246_246911

theorem opposite_of_neg5_is_pos5 : -(-5) = 5 := 
by
  sorry

end opposite_of_neg5_is_pos5_l246_246911


namespace find_last_score_l246_246275

/-- The list of scores in ascending order -/
def scores : List ℕ := [60, 65, 70, 75, 80, 85, 95]

/--
  The problem states that the average score after each entry is an integer.
  Given the scores in ascending order, determine the last score entered.
-/
theorem find_last_score (h : ∀ (n : ℕ) (hn : n < scores.length),
    (scores.take (n + 1) |>.sum : ℤ) % (n + 1) = 0) :
  scores.last' = some 80 :=
sorry

end find_last_score_l246_246275


namespace sufficient_not_necessary_condition_l246_246467

theorem sufficient_not_necessary_condition (x : ℝ) : 
  (x = -1 → x^2 = 1) ∧ ¬(x^2 = 1 → x = -1) :=
by
  sorry

end sufficient_not_necessary_condition_l246_246467


namespace three_tenths_of_number_l246_246464

theorem three_tenths_of_number (x : ℝ) (h : (1/3) * (1/4) * x = 18) : (3/10) * x = 64.8 :=
sorry

end three_tenths_of_number_l246_246464


namespace identify_faulty_key_l246_246960

variable (digits : Finset ℕ)
variable (faulty : ℕ → Bool)

-- Conditions described in the problem statement
variable (attempted_sequence : List ℕ) (registered_sequence : List ℕ)
variable (sequence_length : Nat := 10)
variable (registered_count : Nat := 7)
variable (faulty_press_threshold : Nat := 5)

-- Let attempted_sequence be the sequence typed out and registered_sequence be what was actually registered.

theorem identify_faulty_key (h_len_attempted : attempted_sequence.length = sequence_length)
                            (h_len_registered : registered_sequence.length = registered_count)
                            (h_frequent_digits : ∃ d1 d2, d1 ≠ d2 ∧
                                                        attempted_sequence.count d1 ≥ 2 ∧
                                                        attempted_sequence.count d2 ≥ 2 ∧
                                                        (attempted_sequence.count d1 - registered_sequence.count d1 ≥ 1) ∧
                                                        (attempted_sequence.count d2 - registered_sequence.count d2 ≥ 1)) :
  ∃ d, faulty d ∧ (d = 7 ∨ d = 9) :=
sorry

end identify_faulty_key_l246_246960


namespace mike_picked_12_pears_l246_246549

theorem mike_picked_12_pears
  (jason_pears : ℕ)
  (keith_pears : ℕ)
  (total_pears : ℕ)
  (H1 : jason_pears = 46)
  (H2 : keith_pears = 47)
  (H3 : total_pears = 105) :
  (total_pears - (jason_pears + keith_pears)) = 12 :=
by
  sorry

end mike_picked_12_pears_l246_246549


namespace lowest_possible_students_l246_246462

-- Definitions based on conditions
def isDivisibleBy (n m : ℕ) : Prop := n % m = 0

def canBeDividedIntoTeams (num_students num_teams : ℕ) : Prop := isDivisibleBy num_students num_teams

-- Theorem statement for the lowest possible number of students
theorem lowest_possible_students (n : ℕ) : 
  (canBeDividedIntoTeams n 8) ∧ (canBeDividedIntoTeams n 12) → n = 24 := by
  sorry

end lowest_possible_students_l246_246462


namespace opposite_of_neg_five_l246_246922

theorem opposite_of_neg_five : -(-5) = 5 := by
  sorry

end opposite_of_neg_five_l246_246922


namespace smallest_integer_with_18_divisors_l246_246737

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, (∀ m : ℕ, (m > 0 → m ≤ 131072) → 
  (∃ d : ℕ, d = 18 → (associated (factors.count d) (num_factors m)) → m = 131072)) :=
sorry

end smallest_integer_with_18_divisors_l246_246737


namespace no_nat_n_exists_l246_246056

theorem no_nat_n_exists (n : ℕ) : ¬ ∃ n, ∃ k, n ^ 2012 - 1 = 2 ^ k := by
  sorry

end no_nat_n_exists_l246_246056


namespace lucky_sum_equal_prob_l246_246395

theorem lucky_sum_equal_prob (N : ℕ) (S63_main_draw : Finset ℕ) (S44_additional_draw : Finset ℕ)
  (hN_pos : 0 < N)
  (hS63 : S63_main_draw.card = 10)
  (hSum63 : S63_main_draw.sum id = 63)
  (hS44 : S44_additional_draw.card = 8)
  (hSum44 : S44_additional_draw.sum id = 44)
  (hS_distinct : ∀ (n ∈ S63_main_draw ∪ S44_additional_draw), n ∈ Finset.range (N + 1)) :
  N = 18 :=
by
  sorry

end lucky_sum_equal_prob_l246_246395


namespace equilateral_triangle_area_l246_246622

theorem equilateral_triangle_area (perimeter : ℝ) (h1 : perimeter = 120) :
  ∃ A : ℝ, A = 400 * Real.sqrt 3 ∧
    (∃ s : ℝ, s = perimeter / 3 ∧ A = (Real.sqrt 3 / 4) * (s ^ 2)) :=
by
  sorry

end equilateral_triangle_area_l246_246622


namespace ratio_area_perimeter_eq_sqrt3_l246_246729

theorem ratio_area_perimeter_eq_sqrt3 :
  let side_length := 12
  let altitude := side_length * (Real.sqrt 3) / 2
  let area := (1 / 2) * side_length * altitude
  let perimeter := 3 * side_length
  let ratio := area / perimeter
  ratio = Real.sqrt 3 := 
by
  sorry

end ratio_area_perimeter_eq_sqrt3_l246_246729


namespace positive_diff_probability_fair_coin_l246_246589

theorem positive_diff_probability_fair_coin :
  let p1 := (Nat.choose 5 3) * (1 / 2)^5
  let p2 := (1 / 2)^5
  p1 - p2 = 9 / 32 :=
by
  sorry

end positive_diff_probability_fair_coin_l246_246589


namespace cryptarithm_solution_l246_246548

theorem cryptarithm_solution (A B C D : ℕ) (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_adjacent : A = C + 1 ∨ A = C - 1)
  (h_diff : B = D + 2 ∨ B = D - 2) :
  1000 * A + 100 * B + 10 * C + D = 5240 :=
sorry

end cryptarithm_solution_l246_246548


namespace total_number_of_cats_l246_246246

def Cat := Type -- Define a type of Cat.

variable (A B C: Cat) -- Declaring three cats A, B, and C.

variable (kittens_A: Fin 4 → {gender : Bool // (2 : Fin 4).val = 2 ∧ (2 : Fin 4).val = 2}) -- 4 kittens: 2 males, 2 females.
variable (kittens_B: Fin 3 → {gender : Bool // (1 : Fin 3).val = 1 ∧ (2 : Fin 3).val = 2}) -- 3 kittens: 1 male, 2 females.
variable (kittens_C: Fin 5 → {gender : Bool // (3 : Fin 5).val = 3 ∧ (2 : Fin 5).val = 2}) -- 5 kittens: 3 males, 2 females.

variable (extra_kittens: Fin 2 → {gender : Bool // (1 : Fin 2).val = 1 ∧ (1 : Fin 2).val = 1}) -- 2 kittens of the additional female kitten of Cat A.

theorem total_number_of_cats : 
  3 + 4 + 2 + 3 + 5 = 17 :=
by
  sorry

end total_number_of_cats_l246_246246


namespace complex_equation_l246_246357

theorem complex_equation (m n : ℝ) (i : ℂ)
  (hi : i^2 = -1)
  (h1 : m * (1 + i) = 1 + n * i) :
  ( (m + n * i) / (m - n * i) )^2 = -1 :=
sorry

end complex_equation_l246_246357


namespace incorrect_statement_l246_246291

-- Define the general rules of program flowcharts
def isValidStart (box : String) : Prop := box = "start"
def isValidEnd (box : String) : Prop := box = "end"
def isInputBox (box : String) : Prop := box = "input"
def isOutputBox (box : String) : Prop := box = "output"

-- Define the statement to be proved incorrect
def statement (boxes : List String) : Prop :=
  ∀ xs ys, boxes = xs ++ ["start", "input"] ++ ys ->
           ∀ zs ws, boxes = zs ++ ["output", "end"] ++ ws

-- The target theorem stating that the statement is incorrect
theorem incorrect_statement (boxes : List String) :
  ¬ statement boxes :=
sorry

end incorrect_statement_l246_246291


namespace f_1982_l246_246144

-- Define the function f and the essential properties and conditions
def f : ℕ → ℕ := sorry

axiom f_nonneg (n : ℕ) : f n ≥ 0
axiom f_add_property (m n : ℕ) : f (m + n) - f m - f n = 0 ∨ f (m + n) - f m - f n = 1
axiom f_2 : f 2 = 0
axiom f_3_pos : f 3 > 0
axiom f_9999 : f 9999 = 3333

-- Statement of the theorem we want to prove
theorem f_1982 : f 1982 = 660 := 
  by sorry

end f_1982_l246_246144


namespace estimate_larger_than_difference_l246_246844

theorem estimate_larger_than_difference (x y z : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : z > 0) :
    (x + z) - (y - z) > x - y :=
    sorry

end estimate_larger_than_difference_l246_246844


namespace hyperbola_standard_equations_l246_246642

-- Definitions derived from conditions
def focal_distance (c : ℝ) : Prop := c = 8
def eccentricity (e : ℝ) : Prop := e = 4 / 3
def equilateral_focus (c : ℝ) : Prop := c^2 = 36

-- Theorem stating the standard equations given the conditions
noncomputable def hyperbola_equation1 (y2 : ℝ) (x2 : ℝ) : Prop :=
y2 / 36 - x2 / 28 = 1

noncomputable def hyperbola_equation2 (x2 : ℝ) (y2 : ℝ) : Prop :=
x2 / 18 - y2 / 18 = 1

theorem hyperbola_standard_equations
  (c y2 x2 : ℝ)
  (c_focus : focal_distance c)
  (e_value : eccentricity (4 / 3))
  (equi_focus : equilateral_focus c) :
  hyperbola_equation1 y2 x2 ∧ hyperbola_equation2 x2 y2 :=
by
  sorry

end hyperbola_standard_equations_l246_246642


namespace min_distance_eq_3_l246_246235

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (Real.pi / 3 * x + Real.pi / 4)

theorem min_distance_eq_3 (x₁ x₂ : ℝ) 
  (h₁ : f x₁ ≤ f x) (h₂ : f x ≤ f x₂) 
  (x : ℝ) :
  |x₁ - x₂| = 3 :=
by
  -- Sorry placeholder for proof.
  sorry

end min_distance_eq_3_l246_246235


namespace infinite_series_sum_l246_246505

theorem infinite_series_sum :
  (∑' n : ℕ, (3^n) / (1 + 3^n + 3^(n + 1) + 3^(2 * n + 1))) = 1 / 4 :=
by
  sorry

end infinite_series_sum_l246_246505


namespace unique_solution_single_element_l246_246360

theorem unique_solution_single_element (a : ℝ) 
  (h : ∀ x y : ℝ, (a * x^2 + a * x + 1 = 0) → (a * y^2 + a * y + 1 = 0) → x = y) : a = 4 := 
by
  sorry

end unique_solution_single_element_l246_246360


namespace problem1_problem2_l246_246083

noncomputable def f (a x : ℝ) := a * x - (Real.log x) / x - a

theorem problem1 (a : ℝ) (h : deriv (f a) 1 = 0) : a = 1 := by
  sorry

theorem problem2 (a : ℝ) (h : ∀ x, 1 < x ∧ x < Real.exp 1 → f a x ≤ 0) : a ≤ 1 / (Real.exp 1 * (Real.exp 1 - 1)) := by
  sorry

end problem1_problem2_l246_246083


namespace find_a5_l246_246069

-- Define the problem conditions within Lean
variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Conditions of the problem
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q
def positive_terms (a : ℕ → ℝ) := ∀ n, 0 < a n
def condition1 (a : ℕ → ℝ) := a 1 * a 3 = 4
def condition2 (a : ℕ → ℝ) := a 7 * a 9 = 25

-- Proposition to prove
theorem find_a5 :
  geometric_sequence a q →
  positive_terms a →
  condition1 a →
  condition2 a →
  a 5 = Real.sqrt 10 :=
by
  sorry

end find_a5_l246_246069


namespace min_trips_needed_l246_246943

noncomputable def min_trips (n : ℕ) (h : 2 ≤ n) : ℕ :=
  6

theorem min_trips_needed
  (n : ℕ) (h : 2 ≤ n) (students : Finset (Fin (2 * n)))
  (trip : ℕ → Finset (Fin (2 * n)))
  (trip_cond : ∀ i, (trip i).card = n)
  (pair_cond : ∀ (s t : Fin (2 * n)),
    s ≠ t → ∃ i, s ∈ trip i ∧ t ∈ trip i) :
  ∃ k, k = min_trips n h :=
by
  use 6
  sorry

end min_trips_needed_l246_246943


namespace number_of_boys_l246_246445

theorem number_of_boys (x g : ℕ) (h1 : x + g = 100) (h2 : g = x) : x = 50 := by
  sorry

end number_of_boys_l246_246445


namespace man_and_son_work_together_l246_246031

-- Define the rates at which the man and his son can complete the work
def man_work_rate := 1 / 5
def son_work_rate := 1 / 20

-- Define the combined work rate when they work together
def combined_work_rate := man_work_rate + son_work_rate

-- Define the total time taken to complete the work together
def days_to_complete_together := 1 / combined_work_rate

-- The theorem stating that they will complete the work in 4 days
theorem man_and_son_work_together : days_to_complete_together = 4 := by
  sorry

end man_and_son_work_together_l246_246031


namespace exp_division_rule_l246_246051

-- The theorem to prove the given problem
theorem exp_division_rule (x : ℝ) (hx : x ≠ 0) :
  x^10 / x^5 = x^5 :=
by sorry

end exp_division_rule_l246_246051


namespace snow_at_mrs_hilts_house_l246_246981

theorem snow_at_mrs_hilts_house
    (snow_at_school : ℕ)
    (extra_snow_at_house : ℕ) 
    (school_snow_amount : snow_at_school = 17) 
    (extra_snow_amount : extra_snow_at_house = 12) :
  snow_at_school + extra_snow_at_house = 29 := 
by
  sorry

end snow_at_mrs_hilts_house_l246_246981


namespace cube_root_equation_solutions_l246_246207

theorem cube_root_equation_solutions (x : ℝ) :
  (∃ (y : ℝ), y = real.cbrt x ∧ y = 15 / (8 - y)) ↔ (x = 27 ∨ x = 125) :=
by
  sorry

end cube_root_equation_solutions_l246_246207


namespace protein_percentage_in_mixture_l246_246883

theorem protein_percentage_in_mixture :
  let soybean_meal_weight := 240
  let cornmeal_weight := 40
  let mixture_weight := 280
  let soybean_protein_content := 0.14
  let cornmeal_protein_content := 0.07
  let total_protein := soybean_meal_weight * soybean_protein_content + cornmeal_weight * cornmeal_protein_content
  let protein_percentage := (total_protein / mixture_weight) * 100
  protein_percentage = 13 :=
by
  sorry

end protein_percentage_in_mixture_l246_246883


namespace podcast_ratio_l246_246880

theorem podcast_ratio
  (total_drive_time : ℕ)
  (first_podcast : ℕ)
  (third_podcast : ℕ)
  (fourth_podcast : ℕ)
  (next_podcast : ℕ)
  (second_podcast : ℕ) :
  total_drive_time = 360 →
  first_podcast = 45 →
  third_podcast = 105 →
  fourth_podcast = 60 →
  next_podcast = 60 →
  second_podcast = total_drive_time - (first_podcast + third_podcast + fourth_podcast + next_podcast) →
  second_podcast / first_podcast = 2 :=
by
  sorry

end podcast_ratio_l246_246880


namespace quadrilateral_EFGH_area_l246_246875

-- Definitions based on conditions
def quadrilateral_EFGH_right_angles (F H : ℝ) : Prop :=
  ∃ E G, E - F = 0 ∧ H - G = 0

def quadrilateral_length_hypotenuse (E G : ℝ) : Prop :=
  E - G = 5

def distinct_integer_lengths (EF FG EH HG : ℝ) : Prop :=
  EF ≠ FG ∧ EH ≠ HG ∧ ∃ a b : ℕ, EF = a ∧ FG = b ∧ EH = b ∧ HG = a ∧ a * a + b * b = 25

-- Proof statement
theorem quadrilateral_EFGH_area (F H : ℝ) 
  (EF FG EH HG E G : ℝ) 
  (h1 : quadrilateral_EFGH_right_angles F H) 
  (h2 : quadrilateral_length_hypotenuse E G)
  (h3 : distinct_integer_lengths EF FG EH HG) 
: 
  EF * FG / 2 + EH * HG / 2 = 12 := 
sorry

end quadrilateral_EFGH_area_l246_246875


namespace max_x2_y2_z4_l246_246556

theorem max_x2_y2_z4 (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hxyz : x + y + z = 1) :
  x^2 + y^2 + z^4 ≤ 1 :=
sorry

end max_x2_y2_z4_l246_246556


namespace total_amount_invested_l246_246182

-- Define the problem details: given conditions
def interest_rate_share1 : ℚ := 9 / 100
def interest_rate_share2 : ℚ := 11 / 100
def total_interest_rate : ℚ := 39 / 400
def amount_invested_share2 : ℚ := 3750

-- Define the total amount invested (A), the amount invested at the 9% share (x)
variable (A x : ℚ)

-- Conditions
axiom condition1 : x + amount_invested_share2 = A
axiom condition2 : interest_rate_share1 * x + interest_rate_share2 * amount_invested_share2 = total_interest_rate * A

-- Prove that the total amount invested in both types of shares is Rs. 10,000
theorem total_amount_invested : A = 10000 :=
by {
  -- proof goes here
  sorry
}

end total_amount_invested_l246_246182


namespace initial_earning_members_l246_246289

theorem initial_earning_members (n : ℕ) (h1 : (n * 735) - ((n - 1) * 650) = 905) : n = 3 := by
  sorry

end initial_earning_members_l246_246289


namespace money_left_eq_l246_246112

theorem money_left_eq :
  let initial_money := 56
  let notebooks := 7
  let cost_per_notebook := 4
  let books := 2
  let cost_per_book := 7
  let money_left := initial_money - (notebooks * cost_per_notebook + books * cost_per_book)
  money_left = 14 :=
by
  sorry

end money_left_eq_l246_246112


namespace coeff_sum_eq_twenty_l246_246094

theorem coeff_sum_eq_twenty 
  (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ)
  (h : ((2 * x - 3) ^ 5) = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5) :
  a₁ + 2 * a₂ + 3 * a₃ + 4 * a₄ + 5 * a₅ = 20 :=
by
  sorry

end coeff_sum_eq_twenty_l246_246094


namespace distance_between_Q_and_R_l246_246674

noncomputable def distance_QR : ℝ :=
  let DE : ℝ := 9
  let EF : ℝ := 12
  let DF : ℝ := 15
  let N : ℝ := 7.5
  let QF : ℝ := (N * DF) / EF
  let QD : ℝ := DF - QF
  let QR : ℝ := (QD * DF) / EF
  QR

theorem distance_between_Q_and_R 
  (DE EF DF N QF QD QR : ℝ )
  (h1 : DE = 9)
  (h2 : EF = 12)
  (h3 : DF = 15)
  (h4 : N = DF / 2)
  (h5 : QF = N * DF / EF)
  (h6 : QD = DF - QF)
  (h7 : QR = QD * DF / EF) :
  QR = 7.03125 :=
by
  sorry

end distance_between_Q_and_R_l246_246674


namespace correct_statements_count_l246_246485

theorem correct_statements_count :
  (∃ n : ℕ, odd_positive_integer = 4 * n + 1 ∨ odd_positive_integer = 4 * n + 3) ∧
  (∀ k : ℕ, k = 3 * m ∨ k = 3 * m + 1 ∨ k = 3 * m + 2) ∧
  (∀ s : ℕ, odd_positive_integer ^ 2 = 8 * p + 1) ∧
  (∀ t : ℕ, perfect_square = 3 * q ∨ perfect_square = 3 * q + 1) →
  num_correct_statements = 2 :=
by
  sorry

end correct_statements_count_l246_246485


namespace twice_x_plus_one_third_y_l246_246991

theorem twice_x_plus_one_third_y (x y : ℝ) : 2 * x + (1 / 3) * y = 2 * x + (1 / 3) * y := 
by 
  sorry

end twice_x_plus_one_third_y_l246_246991


namespace min_value_of_f_min_value_at_x_1_l246_246250

noncomputable def f (x : ℝ) : ℝ := 1 / (1 - 2 * x) + 1 / (2 - 3 * x)

theorem min_value_of_f :
  ∀ x : ℝ, x > 0 → f x ≥ 35 :=
by
  sorry

-- As an additional statement, we can check the specific case at x = 1
theorem min_value_at_x_1 :
  f 1 = 35 :=
by
  sorry

end min_value_of_f_min_value_at_x_1_l246_246250


namespace smallest_positive_integer_with_18_divisors_l246_246747

theorem smallest_positive_integer_with_18_divisors : ∃ n : ℕ, 0 < n ∧ (∀ d : ℕ, d ∣ n → 0 < d → d ≠ n → (∀ m : ℕ, m ∣ d → m = 1) → n = 180) :=
sorry

end smallest_positive_integer_with_18_divisors_l246_246747


namespace train_length_is_correct_l246_246484

noncomputable def speed_of_train_kmph : ℝ := 77.993280537557

noncomputable def speed_of_man_kmph : ℝ := 6

noncomputable def conversion_factor : ℝ := 5 / 18

noncomputable def speed_of_train_mps : ℝ := speed_of_train_kmph * conversion_factor

noncomputable def speed_of_man_mps : ℝ := speed_of_man_kmph * conversion_factor

noncomputable def relative_speed : ℝ := speed_of_train_mps + speed_of_man_mps

noncomputable def time_to_pass_man : ℝ := 6

noncomputable def length_of_train : ℝ := relative_speed * time_to_pass_man

theorem train_length_is_correct : length_of_train = 139.99 := by
  sorry

end train_length_is_correct_l246_246484


namespace find_longer_parallel_side_length_l246_246481

noncomputable def longer_parallel_side_length_of_trapezoid : ℝ :=
  let square_side_length : ℝ := 2
  let center_to_side_length : ℝ := square_side_length / 2
  let midline_length : ℝ := square_side_length / 2
  let equal_area : ℝ := (square_side_length^2) / 3
  let height_of_trapezoid : ℝ := center_to_side_length
  let shorter_parallel_side_length : ℝ := midline_length
  let longer_parallel_side_length := (2 * equal_area / height_of_trapezoid) - shorter_parallel_side_length
  longer_parallel_side_length

theorem find_longer_parallel_side_length : 
  longer_parallel_side_length_of_trapezoid = 5/3 := 
sorry

end find_longer_parallel_side_length_l246_246481


namespace cody_payment_l246_246494

def initial_cost : ℝ := 40
def tax_rate : ℝ := 0.05
def discount : ℝ := 8

def tax_amount := initial_cost * tax_rate
def total_with_tax := initial_cost + tax_amount
def final_price := total_with_tax - discount
def cody_share := final_price / 2

theorem cody_payment : cody_share = 17 := by
  sorry

end cody_payment_l246_246494


namespace stationery_box_cost_l246_246780

theorem stationery_box_cost (unit_price : ℕ) (quantity : ℕ) (total_cost : ℕ) :
  unit_price = 23 ∧ quantity = 3 ∧ total_cost = 3 * 23 → total_cost = 69 :=
by
  sorry

end stationery_box_cost_l246_246780


namespace right_triangle_legs_l246_246391

theorem right_triangle_legs (a b c : ℝ) 
  (h : ℝ) 
  (h_h : h = 12) 
  (h_perimeter : a + b + c = 60) 
  (h1 : a^2 + b^2 = c^2) 
  (h_altitude : h = a * b / c) :
  (a = 15 ∧ b = 20) ∨ (a = 20 ∧ b = 15) :=
by
  sorry

end right_triangle_legs_l246_246391


namespace arithmetic_sequence_a3_l246_246688

variable {a : ℕ → ℝ}  -- Define the sequence as a function from natural numbers to real numbers.

-- Definition that the sequence is arithmetic.
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m, a (n + 1) - a n = a (m + 1) - a m

-- The given condition in the problem
axiom h1 : a 1 + a 5 = 6

-- The statement to prove
theorem arithmetic_sequence_a3 (h : is_arithmetic_sequence a) : a 3 = 3 :=
by {
  -- The proof is omitted.
  sorry
}

end arithmetic_sequence_a3_l246_246688


namespace width_of_deck_l246_246865

noncomputable def length : ℝ := 30
noncomputable def cost_per_sqft_construction : ℝ := 3
noncomputable def cost_per_sqft_sealant : ℝ := 1
noncomputable def total_cost : ℝ := 4800
noncomputable def total_cost_per_sqft : ℝ := cost_per_sqft_construction + cost_per_sqft_sealant

theorem width_of_deck (w : ℝ) 
  (h1 : length * w * total_cost_per_sqft = total_cost) : 
  w = 40 := 
sorry

end width_of_deck_l246_246865


namespace abc_sum_eq_sixteen_l246_246686

theorem abc_sum_eq_sixteen (a b c : ℤ) (h1 : a ≠ b ∨ a ≠ c ∨ b ≠ c) (h2 : a ≥ 4 ∧ b ≥ 4 ∧ c ≥ 4) (h3 : 4 * a * b * c = (a + 3) * (b + 3) * (c + 3)) : a + b + c = 16 :=
by 
  sorry

end abc_sum_eq_sixteen_l246_246686


namespace simple_interest_rate_l246_246186

theorem simple_interest_rate (P A T : ℝ) (H1 : P = 1750) (H2 : A = 2000) (H3 : T = 4) :
  ∃ R : ℝ, R = 3.57 ∧ A = P * (1 + (R * T) / 100) :=
by
  sorry

end simple_interest_rate_l246_246186


namespace min_value_x_4_over_x_min_value_x_4_over_x_eq_l246_246667

theorem min_value_x_4_over_x (x : ℝ) (h : x > 0) : x + 4 / x ≥ 4 :=
sorry

theorem min_value_x_4_over_x_eq (x : ℝ) (h : x > 0) : (x + 4 / x = 4) ↔ (x = 2) :=
sorry

end min_value_x_4_over_x_min_value_x_4_over_x_eq_l246_246667


namespace logarithmic_inequality_l246_246873

theorem logarithmic_inequality (a : ℝ) (h : a > 1) : 
  1 / 2 + 1 / Real.log a ≥ 1 := 
sorry

end logarithmic_inequality_l246_246873


namespace solution_set_for_inequality_l246_246412

noncomputable def f : ℝ → ℝ := sorry

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem solution_set_for_inequality
  (h1 : is_odd f)
  (h2 : f 2 = 0)
  (h3 : ∀ x > 0, x * deriv f x - f x < 0) :
  {x : ℝ | f x / x > 0} = {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | 0 < x ∧ x < 2} :=
by sorry

end solution_set_for_inequality_l246_246412


namespace cell_value_l246_246428

variable (P Q R S : ℕ)

-- Condition definitions
def topLeftCell (P : ℕ) : ℕ := P
def topMiddleCell (P Q : ℕ) : ℕ := P + Q
def centerCell (P Q R S : ℕ) : ℕ := P + Q + R + S
def bottomLeftCell (S : ℕ) : ℕ := S

-- Given Conditions
axiom bottomLeftCell_value : bottomLeftCell S = 13
axiom topMiddleCell_value : topMiddleCell P Q = 18
axiom centerCell_value : centerCell P Q R S = 47

-- To prove: R = 16
theorem cell_value : R = 16 :=
by
  sorry

end cell_value_l246_246428


namespace incorrect_statements_l246_246072

/-- Definitions for lines and planes and their relations. -/
variables {a b : Set ℝ^3} {α β : Set ℝ^3}

def parallel (l₁ l₂ : Set ℝ^3) : Prop := ∀ p₁ ∈ l₁, ∀ p₂ ∈ l₂, ∃ v, p₁ + v = p₂

def contained_in (l : Set ℝ^3) (p : Set ℝ^3) : Prop := ∀ pt ∈ l, pt ∈ p

def skew (l₁ l₂ : Set ℝ^3) : Prop := ¬(parallel l₁ l₂) ∧ ∀ pt₁ ∈ l₁, ∀ pt₂ ∈ l₂, pt₁ ≠ pt₂

def conditions : Prop :=
  (parallel a b ∧ contained_in b α) ∧
  (parallel a α ∧ parallel b α) ∧
  (skew a b ∧ contained_in a α ∧ contained_in b β ∧ parallel a β ∧ parallel b α) ∧
  (parallel a α ∧ parallel a β)

/-- The theorem stating the incorrect statements are A, B, and D. -/
theorem incorrect_statements : conditions → ([false, false, true, false] = [A, B, C, D]) :=
by
  sorry

end incorrect_statements_l246_246072


namespace opposite_of_negative_five_l246_246908

theorem opposite_of_negative_five : ∀ x : Int, -5 + x = 0 → x = 5 :=
by
  intros x h
  sorry

end opposite_of_negative_five_l246_246908


namespace number_of_balls_in_last_box_l246_246872

noncomputable def box_question (b : ℕ → ℕ) : Prop :=
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ 2010 → b i + b (i + 1) = 14 + i) ∧
  (b 1 + b 2011 = 1023)

theorem number_of_balls_in_last_box (b : ℕ → ℕ) (h : box_question b) : b 2011 = 1014 :=
by
  sorry

end number_of_balls_in_last_box_l246_246872


namespace prob_green_second_given_first_green_l246_246769

def total_balls : Nat := 14
def green_balls : Nat := 8
def red_balls : Nat := 6

def prob_green_first_draw : ℚ := green_balls / total_balls

theorem prob_green_second_given_first_green :
  prob_green_first_draw = (8 / 14) → (green_balls / total_balls) = (4 / 7) :=
by
  sorry

end prob_green_second_given_first_green_l246_246769


namespace complement_of_A_in_U_is_2_l246_246692

open Set

def U : Set ℕ := { x | x ≥ 2 }
def A : Set ℕ := { x | x^2 ≥ 5 }

theorem complement_of_A_in_U_is_2 : compl A ∩ U = {2} :=
by
  sorry

end complement_of_A_in_U_is_2_l246_246692


namespace total_sum_lent_l246_246781

theorem total_sum_lent (x : ℚ) (second_part : ℚ) (total_sum : ℚ) (h : second_part = 1688) 
  (h_interest : x * 3/100 * 8 = second_part * 5/100 * 3) : total_sum = 2743 :=
by
  sorry

end total_sum_lent_l246_246781


namespace geometric_series_first_term_l246_246045

theorem geometric_series_first_term (r : ℚ) (S : ℚ) (a : ℚ) (h_r : r = 1 / 4) (h_S : S = 80) (h_sum : S = a / (1 - r)) : a = 60 :=
by 
  sorry

end geometric_series_first_term_l246_246045


namespace problem_l246_246233

def S (n : ℕ) : ℤ := n^2 - 4 * n + 1

def a (n : ℕ) : ℤ :=
  if n = 1 then S 1 else S n - S (n - 1)

def sum_abs_a_10 : ℤ :=
  (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8| + |a 9| + |a 10|)

theorem problem : sum_abs_a_10 = 67 := by
  sorry

end problem_l246_246233


namespace color_points_distance_d_l246_246855

open Real Rat Set

variable {d r s : ℝ} (rational_r : r ∈ ℚ) (rational_s : s ∈ ℚ)

theorem color_points_distance_d (h : d^2 = r^2 + s^2) :
  ∃ (coloring : ℚ × ℚ → ℕ), (∀ (a b : ℚ × ℚ), dist a b = d → coloring a ≠ coloring b) :=
sorry

end color_points_distance_d_l246_246855


namespace lines_intersect_and_sum_l246_246896

theorem lines_intersect_and_sum (a b : ℝ) :
  (∃ x y : ℝ, x = (1 / 3) * y + a ∧ y = (1 / 3) * x + b ∧ x = 3 ∧ y = 3) →
  a + b = 4 :=
by
  sorry

end lines_intersect_and_sum_l246_246896


namespace find_A_from_AB9_l246_246635

theorem find_A_from_AB9 (A B : ℕ) (h1 : 0 ≤ A ∧ A ≤ 9) (h2 : 0 ≤ B ∧ B ≤ 9) (h3 : 100 * A + 10 * B + 9 = 459) : A = 4 :=
sorry

end find_A_from_AB9_l246_246635


namespace total_balloons_l246_246452

-- Define the number of yellow balloons each person has
def tom_balloons : Nat := 18
def sara_balloons : Nat := 12
def alex_balloons : Nat := 7

-- Prove that the total number of balloons is 37
theorem total_balloons : tom_balloons + sara_balloons + alex_balloons = 37 := 
by 
  sorry

end total_balloons_l246_246452


namespace opposite_of_neg_five_is_five_l246_246919

-- Define the condition for the opposite of a number
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- State the theorem that the opposite of -5 is 5
theorem opposite_of_neg_five_is_five : is_opposite (-5) 5 :=
by
  -- Insert the proof here
  sorry

end opposite_of_neg_five_is_five_l246_246919


namespace range_a_for_inequality_l246_246213

theorem range_a_for_inequality (a : ℝ) :
  (∀ x : ℝ, (a-2) * x^2 - 2 * (a-2) * x - 4 < 0) ↔ (-2 < a ∧ a ≤ 2) :=
by
  sorry

end range_a_for_inequality_l246_246213


namespace largest_int_starting_with_8_l246_246509

theorem largest_int_starting_with_8 (n : ℕ) : 
  (n / 100 = 8) ∧ (n >= 800) ∧ (n < 900) ∧ ∀ (d : ℕ), (d ∣ n ∧ d ≠ 0 ∧ d ≠ 7) → d ∣ 864 → (n ≤ 864) :=
sorry

end largest_int_starting_with_8_l246_246509


namespace yoongi_hoseok_age_sum_l246_246429

-- Definitions of given conditions
def age_aunt : ℕ := 38
def diff_aunt_yoongi : ℕ := 23
def diff_yoongi_hoseok : ℕ := 4

-- Definitions related to ages of Yoongi and Hoseok derived from given conditions
def age_yoongi : ℕ := age_aunt - diff_aunt_yoongi
def age_hoseok : ℕ := age_yoongi - diff_yoongi_hoseok

-- The theorem we need to prove
theorem yoongi_hoseok_age_sum : age_yoongi + age_hoseok = 26 := by
  sorry

end yoongi_hoseok_age_sum_l246_246429


namespace restaurant_total_spent_l246_246424

theorem restaurant_total_spent (appetizer_cost : ℕ) (entree_cost : ℕ) (num_entrees : ℕ) (tip_rate : ℚ) 
  (H1 : appetizer_cost = 10) (H2 : entree_cost = 20) (H3 : num_entrees = 4) (H4 : tip_rate = 0.20) :
  appetizer_cost + num_entrees * entree_cost + (appetizer_cost + num_entrees * entree_cost) * tip_rate = 108 :=
by
  sorry

end restaurant_total_spent_l246_246424


namespace certain_number_proof_l246_246172

noncomputable def certain_number : ℝ := 30

theorem certain_number_proof (h1: 0.60 * 50 = 30) (h2: 30 = 0.40 * certain_number + 18) : 
  certain_number = 30 := 
sorry

end certain_number_proof_l246_246172


namespace Isabella_hair_length_l246_246851

-- Define the conditions using variables
variables (h_current h_cut_off h_initial : ℕ)

-- The proof problem statement
theorem Isabella_hair_length :
  h_current = 9 → h_cut_off = 9 → h_initial = h_current + h_cut_off → h_initial = 18 :=
by
  intros hc hc' hi
  rw [hc, hc'] at hi
  exact hi


end Isabella_hair_length_l246_246851


namespace smallest_integer_with_18_divisors_l246_246753

theorem smallest_integer_with_18_divisors :
  ∃ n : ℕ, (∀ m : ℕ, m > 0 → (∀ d, d ∣ n → d ∣ m) → n = 288) ∧ (nat.divisors_count 288 = 18) := by
  sorry

end smallest_integer_with_18_divisors_l246_246753


namespace sqrt_of_1024_l246_246830

theorem sqrt_of_1024 (x : ℝ) (h1 : x > 0) (h2 : x ^ 2 = 1024) : x = 32 :=
sorry

end sqrt_of_1024_l246_246830


namespace solve_for_constants_l246_246508

theorem solve_for_constants : 
  ∃ (t s : ℚ), (∀ x : ℚ, (3 * x^2 - 4 * x + 9) * (5 * x^2 + t * x + 12) = 15 * x^4 + s * x^3 + 33 * x^2 + 12 * x + 108) ∧ 
  t = 37 / 5 ∧ 
  s = 11 / 5 :=
by
  sorry

end solve_for_constants_l246_246508


namespace total_steps_l246_246566

def steps_on_feet (jason_steps : Nat) (nancy_ratio : Nat) : Nat :=
  jason_steps + (nancy_ratio * jason_steps)

theorem total_steps (jason_steps : Nat) (nancy_ratio : Nat) (h1 : jason_steps = 8) (h2 : nancy_ratio = 3) :
  steps_on_feet jason_steps nancy_ratio = 32 :=
by
  sorry

end total_steps_l246_246566


namespace joe_eggs_town_hall_l246_246268

-- Define the conditions.
def eggs_club_house : ℕ := 12
def eggs_park : ℕ := 5
def eggs_total : ℕ := 20

-- Define the desired result.
def eggs_town_hall : ℕ := eggs_total - eggs_club_house - eggs_park

-- The statement that needs to be proved.
theorem joe_eggs_town_hall : eggs_town_hall = 3 :=
by
  sorry

end joe_eggs_town_hall_l246_246268


namespace smallest_integer_with_18_divisors_l246_246744

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, (∃ (p q r : ℕ), n = p^5 * q^2 ∧ prime p ∧ prime q)
  ∨ (∃ (p q r : ℕ), n = p^2 * q^5 ∧ prime p ∧ prime q)
  ∨ (∃ (p q r : ℕ), n = p^2 * q^2 * r^2 ∧ prime p ∧ prime q ∧ prime r)
  ∨ (∃ (p q : ℕ), n = p^8 * q ∧ prime p ∧ prime q)
  ∨ (∃ (p q : ℕ), n = p^17 ∧ prime p)
  ∧ n = 288 := sorry

end smallest_integer_with_18_divisors_l246_246744


namespace profit_divided_equally_l246_246562

noncomputable def Mary_investment : ℝ := 800
noncomputable def Mike_investment : ℝ := 200
noncomputable def total_profit : ℝ := 2999.9999999999995
noncomputable def Mary_extra : ℝ := 1200

theorem profit_divided_equally (E : ℝ) : 
  (E / 2 + 4 / 5 * (total_profit - E)) - (E / 2 + 1 / 5 * (total_profit - E)) = Mary_extra →
  E = 1000 :=
  by sorry

end profit_divided_equally_l246_246562


namespace opposite_of_neg_five_is_five_l246_246917

-- Define the condition for the opposite of a number
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- State the theorem that the opposite of -5 is 5
theorem opposite_of_neg_five_is_five : is_opposite (-5) 5 :=
by
  -- Insert the proof here
  sorry

end opposite_of_neg_five_is_five_l246_246917


namespace find_sum_l246_246889

variable (a b : ℝ)

theorem find_sum (h1 : 2 = b - 1) (h2 : -1 = a + 3) : a + b = -1 :=
by
  sorry

end find_sum_l246_246889


namespace total_chocolates_l246_246583

-- Definitions based on conditions
def chocolates_per_bag := 156
def number_of_bags := 20

-- Statement to prove
theorem total_chocolates : chocolates_per_bag * number_of_bags = 3120 :=
by
  -- skip the proof
  sorry

end total_chocolates_l246_246583


namespace concurrency_of_tangent_lines_l246_246417

open EuclideanGeometry

-- Definitions from conditions
variable {A B C : Point}
variable {Γ : Circle}
variable {GammaA GammaB GammaC : Circle}
variable {A' B' C' : Point}

-- Conditions
def circle_tangent_to_sides (ΓA : Circle) (AB AC : Segment) (Γ : Circle) (A' : Point) :=
  (ΓA.tangentTo AB) ∧ (ΓA.tangentTo AC) ∧ (ΓA.tangentTo Γ) ∧ (ΓA.center ∈ ΓInside)

axiom circumscribed_circle (ABC : Triangle) : Γ
axiom tangent_circle_A (ABC : Triangle) (Γ : Circle) : ∃ ΓA A', circle_tangent_to_sides ΓA ABC.AB ABC.AC Γ A'
axiom tangent_circle_B (ABC : Triangle) (Γ : Circle) : ∃ ΓB B', circle_tangent_to_sides ΓB ABC.BC ABC.BA Γ B'
axiom tangent_circle_C (ABC : Triangle) (Γ : Circle) : ∃ ΓC C', circle_tangent_to_sides ΓC ABC.CA ABC.CB Γ C'

-- Proof problem statement
theorem concurrency_of_tangent_lines (ABC : Triangle) :
  let Γ := circumscribed_circle ABC in
  (∃ (ΓA : Circle) (A' : Point), circle_tangent_to_sides ΓA ABC.AB ABC.AC Γ A') →
  (∃ (ΓB : Circle) (B' : Point), circle_tangent_to_sides ΓB ABC.BC ABC.BA Γ B') →
  (∃ (ΓC : Circle) (C' : Point), circle_tangent_to_sides ΓC ABC.CA ABC.CB Γ C') →
  ∃ X : Point, collinear [ABC.A, A', X] ∧ collinear [ABC.B, B', X] ∧ collinear [ABC.C, C', X]
:= by sorry

end concurrency_of_tangent_lines_l246_246417


namespace mike_picked_peaches_l246_246126

def initial_peaches : ℕ := 34
def total_peaches : ℕ := 86

theorem mike_picked_peaches : total_peaches - initial_peaches = 52 :=
by
  sorry

end mike_picked_peaches_l246_246126


namespace find_teddy_dogs_l246_246139

-- Definitions from the conditions
def teddy_cats := 8
def ben_dogs (teddy_dogs : ℕ) := teddy_dogs + 9
def dave_cats (teddy_cats : ℕ) := teddy_cats + 13
def dave_dogs (teddy_dogs : ℕ) := teddy_dogs - 5
def total_pets (teddy_dogs teddy_cats : ℕ) := teddy_dogs + teddy_cats + (ben_dogs teddy_dogs) + (dave_dogs teddy_dogs) + (dave_cats teddy_cats)

-- Theorem statement
theorem find_teddy_dogs (teddy_dogs : ℕ) (teddy_cats : ℕ) (hd : total_pets teddy_dogs teddy_cats = 54) :
  teddy_dogs = 7 := sorry

end find_teddy_dogs_l246_246139


namespace quadrilateral_is_trapezoid_l246_246393

variables {V : Type*} [AddCommGroup V] [Module ℝ V] -- Define the type of vectors and vector space over the reals
variables (a b : V) -- Vectors a and b
variables (AB BC CD AD : V) -- Vectors representing sides of quadrilateral

-- Condition: vectors a and b are not collinear
def not_collinear (a b : V) : Prop := ∀ k : ℝ, k ≠ 0 → a ≠ k • b

-- Given Conditions
def conditions (a b AB BC CD : V) : Prop :=
  AB = a + 2 • b ∧
  BC = -4 • a - b ∧
  CD = -5 • a - 3 • b ∧
  not_collinear a b

-- The to-be-proven property
def is_trapezoid (AB BC CD AD : V) : Prop :=
  AD = 2 • BC

theorem quadrilateral_is_trapezoid 
  (a b AB BC CD : V) 
  (h : conditions a b AB BC CD)
  : is_trapezoid AB BC CD (AB + BC + CD) :=
sorry

end quadrilateral_is_trapezoid_l246_246393


namespace faulty_key_in_digits_l246_246961

-- Problem statement definitions
def is_faulty_key (digit_seq : list ℕ) (faulty_keys : set ℕ) : Prop :=
  ∃ (missing_digits faulty_occurrences : ℕ), 
    (∃ (attempted_seq : list ℕ), length digit_seq = 10 ∧ length attempted_seq = 7 ∧
    missing_digits = 10 - 7 ∧ length (digit_seq.filter (λ d, d ∈ faulty_keys)) ≥ 5 ∧
    length (attempted_seq.filter (λ d, d ∈ faulty_keys)) ≥ 2 ∧
    length (digit_seq.filter (λ d, d ∈ faulty_keys)) - length (attempted_seq.filter (λ d, d ∈ faulty_keys)) = 3)

-- Theorem: Proving which keys could be the faulty ones.
theorem faulty_key_in_digits (digit_seq : list ℕ) :
  is_faulty_key digit_seq {7, 9} :=
sorry

end faulty_key_in_digits_l246_246961


namespace train_speed_kph_l246_246178

-- Define conditions as inputs
def train_time_to_cross_pole : ℝ := 6 -- seconds
def train_length : ℝ := 100 -- meters

-- Conversion factor from meters per second to kilometers per hour
def mps_to_kph : ℝ := 3.6

-- Define and state the theorem to be proved
theorem train_speed_kph : (train_length / train_time_to_cross_pole) * mps_to_kph = 50 :=
by
  sorry

end train_speed_kph_l246_246178


namespace arrangement_ways_l246_246841

theorem arrangement_ways : 
  ∀ (persons : ℕ) (chairs : ℕ), 
  persons = 5 ∧ chairs = 6 → 
  (∏ i in finset.range persons, (chairs - i)) = 720 :=
begin
  intros persons chairs,
  rintros ⟨h1, h2⟩,
  subst h1,
  subst h2,
  simp only [finset.prod_range_succ, finset.prod_range_succ, nat.cast_sub, nat.cast_succ, nat.cast_bit0, nat.cast_bit1],
  norm_num
end

end arrangement_ways_l246_246841


namespace yura_picture_dimensions_l246_246164

theorem yura_picture_dimensions (l w : ℕ) (h_frame : (l + 2) * (w + 2) - l * w = l * w) :
    (l = 3 ∧ w = 10) ∨ (l = 4 ∧ w = 6) :=
by {
  sorry
}

end yura_picture_dimensions_l246_246164


namespace right_triangle_leg_square_l246_246541

theorem right_triangle_leg_square (a b c : ℝ) 
  (h1 : c = a + 2) 
  (h2 : a^2 + b^2 = c^2) : b^2 = 4 * a + 4 := 
by
  sorry

end right_triangle_leg_square_l246_246541


namespace cans_for_credit_l246_246702

theorem cans_for_credit (P C R : ℕ) : 
  (3 * P = 2 * C) → (C ≠ 0) → (R ≠ 0) → P * R / C = (P * R / C : ℕ) :=
by
  intros h1 h2 h3
  -- proof required here
  sorry

end cans_for_credit_l246_246702


namespace find_a_plus_b_l246_246247

noncomputable def real_part (z : ℂ) : ℝ := z.re
noncomputable def imag_part (z : ℂ) : ℝ := z.im

theorem find_a_plus_b (a b : ℝ) (i : ℂ) (h1 : i * i = -1) (h2 : (1 + i) * (2 + i) = a + b * i) : a + b = 4 :=
by sorry

end find_a_plus_b_l246_246247


namespace find_natural_number_l246_246058

theorem find_natural_number (n : ℕ) (k : ℤ) (h : 2^n + 3 = k^2) : n = 0 :=
sorry

end find_natural_number_l246_246058


namespace length_of_bridge_l246_246790

noncomputable def convert_speed (km_per_hour : ℝ) : ℝ := km_per_hour * (1000 / 3600)

theorem length_of_bridge 
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (passing_time : ℝ)
  (total_distance_covered : ℝ)
  (bridge_length : ℝ) :
  train_length = 120 →
  train_speed_kmh = 40 →
  passing_time = 25.2 →
  total_distance_covered = convert_speed train_speed_kmh * passing_time →
  bridge_length = total_distance_covered - train_length →
  bridge_length = 160 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end length_of_bridge_l246_246790


namespace sum_of_squares_l246_246657

theorem sum_of_squares (a b c : ℝ) (h₁ : a + b + c = 31) (h₂ : ab + bc + ca = 10) :
  a^2 + b^2 + c^2 = 941 :=
by
  sorry

end sum_of_squares_l246_246657


namespace angies_age_l246_246010

variable (A : ℝ)

theorem angies_age (h : 2 * A + 4 = 20) : A = 8 :=
sorry

end angies_age_l246_246010


namespace ratio_of_area_to_perimeter_of_equilateral_triangle_l246_246730

def equilateral_triangle (s : ℝ) : Prop :=
  s = 12

theorem ratio_of_area_to_perimeter_of_equilateral_triangle :
  ∀ (s : ℝ), equilateral_triangle s → (let P := 3 * s in
  let A := (sqrt 3 / 4) * s ^ 2 in
  A / P = sqrt 3) :=
by
  intro s hs
  rw [equilateral_triangle, hs]
  let P := 3 * 12
  let A := (sqrt 3 / 4) * 12 ^ 2
  have A_eq : A = 36 * sqrt 3 := by
    calc
      (sqrt 3 / 4) * 12 ^ 2 = (sqrt 3 / 4) * 144  : by norm_num
                      ... = (sqrt 3 * 144) / 4  : by rw div_mul_cancel
                      ... = 36 * sqrt 3         : by norm_num
  have P_eq : P = 36 := by norm_num
  rw [A_eq, P_eq]
  norm_num
  rfl

end ratio_of_area_to_perimeter_of_equilateral_triangle_l246_246730


namespace selling_price_41_l246_246606

-- Purchase price per item
def purchase_price : ℝ := 30

-- Government restriction on pice increase: selling price cannot be more than 40% increase of the purchase price
def price_increase_restriction (a : ℝ) : Prop :=
  a <= purchase_price * 1.4

-- Profit condition equation
def profit_condition (a : ℝ) : Prop :=
  (a - purchase_price) * (112 - 2 * a) = 330

-- The selling price of each item that satisfies all conditions is 41 yuan  
theorem selling_price_41 (a : ℝ) (h1 : profit_condition a) (h2 : price_increase_restriction a) :
  a = 41 := sorry

end selling_price_41_l246_246606


namespace Tobias_change_l246_246950

def cost_of_shoes := 95
def allowance_per_month := 5
def months_saving := 3
def charge_per_lawn := 15
def lawns_mowed := 4
def charge_per_driveway := 7
def driveways_shoveled := 5
def total_amount_saved : ℕ := (allowance_per_month * months_saving)
                          + (charge_per_lawn * lawns_mowed)
                          + (charge_per_driveway * driveways_shoveled)

theorem Tobias_change : total_amount_saved - cost_of_shoes = 15 := by
  sorry

end Tobias_change_l246_246950


namespace escher_prints_probability_l246_246565

theorem escher_prints_probability :
  let total_pieces := 12
  let escher_pieces := 4
  let total_permutations := Nat.factorial total_pieces
  let block_permutations := 9 * Nat.factorial (total_pieces - escher_pieces)
  let probability := block_permutations / (total_pieces * Nat.factorial (total_pieces - 1))
  probability = 1 / 1320 :=
by
  let total_pieces := 12
  let escher_pieces := 4
  let total_permutations := Nat.factorial total_pieces
  let block_permutations := 9 * Nat.factorial (total_pieces - escher_pieces)
  let probability := block_permutations / (total_pieces * Nat.factorial (total_pieces - 1))
  sorry

end escher_prints_probability_l246_246565


namespace sufficient_not_necessary_l246_246704

theorem sufficient_not_necessary (x : ℝ) (h1 : -1 < x) (h2 : x < 3) :
    x^2 - 2*x < 8 :=
by
    -- Proof to be filled in.
    sorry

end sufficient_not_necessary_l246_246704


namespace minimum_ladder_rungs_l246_246322

theorem minimum_ladder_rungs (a b : ℕ) (h_a_pos : 0 < a) (h_b_pos : 0 < b): ∃ n, n = a + b - 1 :=
by
    sorry

end minimum_ladder_rungs_l246_246322


namespace opposite_of_negative_five_l246_246932

theorem opposite_of_negative_five : (-(-5) = 5) :=
by
  sorry

end opposite_of_negative_five_l246_246932


namespace marbles_in_jar_is_144_l246_246030

noncomputable def marbleCount (M : ℕ) : Prop :=
  M / 16 - M / 18 = 1

theorem marbles_in_jar_is_144 : ∃ M : ℕ, marbleCount M ∧ M = 144 :=
by
  use 144
  unfold marbleCount
  sorry

end marbles_in_jar_is_144_l246_246030


namespace largest_divisor_of_consecutive_even_product_l246_246119

theorem largest_divisor_of_consecutive_even_product :
  ∀ (n : ℕ), ∃ k : ℤ, k = 24 ∧ 
  (2 * n) * (2 * n + 2) * (2 * n + 4) % k = 0 :=
by
  sorry

end largest_divisor_of_consecutive_even_product_l246_246119


namespace cuboid_third_edge_l246_246893

theorem cuboid_third_edge (a b V h : ℝ) (ha : a = 4) (hb : b = 4) (hV : V = 96) (volume_formula : V = a * b * h) : h = 6 :=
by
  sorry

end cuboid_third_edge_l246_246893


namespace perfect_square_factors_of_180_l246_246093

theorem perfect_square_factors_of_180 {n : ℕ} (h1 : n = 180) : 
  (∃ f : ℕ → ℕ, (∀ p, (p = 2 ∨ p = 3 ∨ p = 5 → 
    (f p = 0 ∨ (p = 2 ∨ p = 3) ∧ f p = 2 ∧ p ≠ 5))) ∧ 2^(f 2) * 3^(f 3) * 5^(f 5) = n ∧  ((f 2 + f 3 + f 5) % 2 = 0)) → 
  4 :=
begin
  sorry
end

end perfect_square_factors_of_180_l246_246093


namespace range_of_x_for_sqrt_l246_246380

theorem range_of_x_for_sqrt (x : ℝ) (hx : sqrt (1 / (x - 1)) = sqrt (1 / (x - 1))) : x > 1 :=
by
  sorry

end range_of_x_for_sqrt_l246_246380


namespace min_value_f_l246_246510

noncomputable def f (x : ℝ) : ℝ := (x^2 + 9) / real.sqrt (x^2 + 5)

theorem min_value_f : ∃ x : ℝ, ∀ y : ℝ, f(y) ≥ f(x) ∧ f(x) = 9 / real.sqrt 5 :=
by sorry

end min_value_f_l246_246510


namespace inequality_proof_l246_246552

theorem inequality_proof 
  (a b : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (h1 : a ≤ 2 * b) 
  (h2 : 2 * b ≤ 4 * a) :
  4 * a * b ≤ 2 * (a ^ 2 + b ^ 2) ∧ 2 * (a ^ 2 + b ^ 2) ≤ 5 * a * b := 
by
  sorry

end inequality_proof_l246_246552


namespace baseball_cards_initial_count_unkn_l246_246866

-- Definitions based on the conditions
def cardValue : ℕ := 6
def tradedCards : ℕ := 2
def receivedCardsValue : ℕ := (3 * 2) + 9   -- 3 cards worth $2 each and 1 card worth $9
def profit : ℕ := receivedCardsValue - (tradedCards * cardValue)

-- Lean 4 statement to represent the proof problem
theorem baseball_cards_initial_count_unkn (h_trade : tradedCards * cardValue = 12)
    (h_receive : receivedCardsValue = 15)
    (h_profit : profit = 3) : ∃ n : ℕ, n >= 2 ∧ n = 2 + (n - 2) :=
sorry

end baseball_cards_initial_count_unkn_l246_246866


namespace arrangement_ways_l246_246840

theorem arrangement_ways : 
  ∀ (persons : ℕ) (chairs : ℕ), 
  persons = 5 ∧ chairs = 6 → 
  (∏ i in finset.range persons, (chairs - i)) = 720 :=
begin
  intros persons chairs,
  rintros ⟨h1, h2⟩,
  subst h1,
  subst h2,
  simp only [finset.prod_range_succ, finset.prod_range_succ, nat.cast_sub, nat.cast_succ, nat.cast_bit0, nat.cast_bit1],
  norm_num
end

end arrangement_ways_l246_246840


namespace find_six_quotients_l246_246059

def is_5twos_3ones (n: ℕ) : Prop :=
  n.digits 10 = [2, 2, 2, 2, 2, 1, 1, 1]

def divides_by_7 (n: ℕ) : Prop :=
  n % 7 = 0

theorem find_six_quotients:
  ∃ n₁ n₂ n₃ n₄ n₅: ℕ, 
    n₁ ≠ n₂ ∧ n₁ ≠ n₃ ∧ n₂ ≠ n₃ ∧ n₁ ≠ n₄ ∧ n₂ ≠ n₄ ∧ n₃ ≠ n₄ ∧ n₁ ≠ n₅ ∧ n₂ ≠ n₅ ∧ n₃ ≠ n₅ ∧ n₄ ≠ n₅ ∧
    is_5twos_3ones n₁ ∧ is_5twos_3ones n₂ ∧ is_5twos_3ones n₃ ∧ is_5twos_3ones n₄ ∧ is_5twos_3ones n₅ ∧
    divides_by_7 n₁ ∧ divides_by_7 n₂ ∧ divides_by_7 n₃ ∧ divides_by_7 n₄ ∧ divides_by_7 n₅ ∧
    n₁ / 7 = 1744603 ∧ n₂ / 7 = 3031603 ∧ n₃ / 7 = 3160303 ∧ n₄ / 7 = 3017446 ∧ n₅ / 7 = 3030316 :=
sorry

end find_six_quotients_l246_246059


namespace oscar_marathon_training_l246_246871

theorem oscar_marathon_training :
  let initial_miles := 2
  let target_miles := 20
  let increment_per_week := (2 : ℝ) / 3
  ∃ weeks_required, target_miles - initial_miles = weeks_required * increment_per_week → weeks_required = 27 :=
by
  sorry

end oscar_marathon_training_l246_246871


namespace part_1_part_2_l246_246082

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x * Real.log x - m * x
noncomputable def f' (x : ℝ) (m : ℝ) : ℝ := Real.log x + 1 - m
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := x * Real.log x - x - a * x^3
noncomputable def r (x : ℝ) : ℝ := (Real.log x - 1) / x^2
noncomputable def r' (x : ℝ) : ℝ := (3 - 2 * Real.log x) / x^3

theorem part_1 (x : ℝ) (m : ℝ) (h1 : f x m = -1) (h2 : f' x m = 0) :
  m = 1 ∧ (∀ y, y > 0 → y < x → f' y 1 < 0) ∧ (∀ y, y > x → f' y 1 > 0) :=
sorry

theorem part_2 (a : ℝ) :
  (a > 1 / (2 * Real.exp 3) → ∀ x, h x a ≠ 0) ∧
  (a ≤ 0 ∨ a = 1 / (2 * Real.exp 3) → ∃ x, h x a = 0 ∧ ∀ y, h y a = 0 → y = x) ∧
  (0 < a ∧ a < 1 / (2 * Real.exp 3) → ∃ x1 x2, x1 ≠ x2 ∧ h x1 a = 0 ∧ h x2 a = 0) :=
sorry

end part_1_part_2_l246_246082


namespace probability_of_at_least_one_head_in_three_tosses_is_7_over_8_l246_246624

def probability_of_at_least_one_head (p : ℚ) (n : ℕ) : ℚ := 
  1 - (1 - p)^n

theorem probability_of_at_least_one_head_in_three_tosses_is_7_over_8 :
  probability_of_at_least_one_head (1/2) 3 = 7/8 :=
by 
  sorry

end probability_of_at_least_one_head_in_three_tosses_is_7_over_8_l246_246624


namespace tim_cantaloupes_l246_246066

theorem tim_cantaloupes (fred_cantaloupes : ℕ) (total_cantaloupes : ℕ) (h1 : fred_cantaloupes = 38) (h2 : total_cantaloupes = 82) : total_cantaloupes - fred_cantaloupes = 44 :=
by {
  -- proof steps go here
  sorry
}

end tim_cantaloupes_l246_246066


namespace total_rainfall_2004_l246_246672

theorem total_rainfall_2004 (average_rainfall_2003 : ℝ) (increase_percentage : ℝ) (months : ℝ) :
  average_rainfall_2003 = 36 →
  increase_percentage = 0.10 →
  months = 12 →
  (average_rainfall_2003 * (1 + increase_percentage) * months) = 475.2 :=
by
  -- The proof is left as an exercise
  sorry

end total_rainfall_2004_l246_246672


namespace smallest_integer_with_18_divisors_l246_246745

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, (∃ (p q r : ℕ), n = p^5 * q^2 ∧ prime p ∧ prime q)
  ∨ (∃ (p q r : ℕ), n = p^2 * q^5 ∧ prime p ∧ prime q)
  ∨ (∃ (p q r : ℕ), n = p^2 * q^2 * r^2 ∧ prime p ∧ prime q ∧ prime r)
  ∨ (∃ (p q : ℕ), n = p^8 * q ∧ prime p ∧ prime q)
  ∨ (∃ (p q : ℕ), n = p^17 ∧ prime p)
  ∧ n = 288 := sorry

end smallest_integer_with_18_divisors_l246_246745


namespace six_digit_divisible_by_72_l246_246036

theorem six_digit_divisible_by_72 (n m : ℕ) (h1 : n = 920160 ∨ n = 120168) :
  (∃ (x y : ℕ), 10 * x + y = 2016 ∧ (10^5 * x + n * 10 + m) % 72 = 0) :=
by
  sorry

end six_digit_divisible_by_72_l246_246036


namespace smallest_positive_integer_with_18_divisors_l246_246746

theorem smallest_positive_integer_with_18_divisors : ∃ n : ℕ, 0 < n ∧ (∀ d : ℕ, d ∣ n → 0 < d → d ≠ n → (∀ m : ℕ, m ∣ d → m = 1) → n = 180) :=
sorry

end smallest_positive_integer_with_18_divisors_l246_246746


namespace combined_cost_of_items_l246_246198

theorem combined_cost_of_items (wallet_cost : ℕ) 
  (purse_cost : ℕ) (combined_cost : ℕ) :
  wallet_cost = 22 →
  purse_cost = 4 * wallet_cost - 3 →
  combined_cost = wallet_cost + purse_cost →
  combined_cost = 107 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end combined_cost_of_items_l246_246198


namespace faulty_key_in_digits_l246_246962

-- Problem statement definitions
def is_faulty_key (digit_seq : list ℕ) (faulty_keys : set ℕ) : Prop :=
  ∃ (missing_digits faulty_occurrences : ℕ), 
    (∃ (attempted_seq : list ℕ), length digit_seq = 10 ∧ length attempted_seq = 7 ∧
    missing_digits = 10 - 7 ∧ length (digit_seq.filter (λ d, d ∈ faulty_keys)) ≥ 5 ∧
    length (attempted_seq.filter (λ d, d ∈ faulty_keys)) ≥ 2 ∧
    length (digit_seq.filter (λ d, d ∈ faulty_keys)) - length (attempted_seq.filter (λ d, d ∈ faulty_keys)) = 3)

-- Theorem: Proving which keys could be the faulty ones.
theorem faulty_key_in_digits (digit_seq : list ℕ) :
  is_faulty_key digit_seq {7, 9} :=
sorry

end faulty_key_in_digits_l246_246962


namespace minimum_value_f_l246_246363

open Real

noncomputable def f (m : ℝ) (x : ℝ) := x^4 * cos x + m * x^2 + 2 * x

theorem minimum_value_f'' (m : ℝ) :
  (∀ x ∈ Icc (-4 : ℝ) 4, has_deriv_at (deriv (deriv (f m))) x (16)) →
  ∃ c ∈ Icc (-4 : ℝ) 4, deriv (deriv (f m)) c = -12 := 
sorry

end minimum_value_f_l246_246363


namespace limit_problem_l246_246492

open Real

theorem limit_problem :
  tendsto (fun x => (root 3 (1 + arctan (4 * x)) - root 3 (1 - arctan (4 * x))) /
                   (sqrt (1 - asin (3 * x)) - sqrt (1 + arctan (3 * x)))) (𝓝 0) (𝓝 (-8 / 9)) :=
begin
  -- sorry is a placeholder for the proof
  sorry
end

end limit_problem_l246_246492


namespace probability_of_part_selected_l246_246383

open Rational

theorem probability_of_part_selected (total_parts : ℕ) (selected_parts : ℕ) 
  (H_total_parts : total_parts = 120) (H_selected_parts : selected_parts = 20):
  selected_parts / total_parts = 1 / 6 := by
  sorry

end probability_of_part_selected_l246_246383


namespace smallest_positive_root_l246_246995

noncomputable def alpha : ℝ := Real.arctan (2 / 9)
noncomputable def beta : ℝ := Real.arctan (6 / 7)

theorem smallest_positive_root :
  ∃ x > 0, (2 * Real.sin (6 * x) + 9 * Real.cos (6 * x) = 6 * Real.sin (2 * x) + 7 * Real.cos (2 * x))
    ∧ x = (alpha + beta) / 8 := sorry

end smallest_positive_root_l246_246995


namespace reaction_rate_reduction_l246_246306

theorem reaction_rate_reduction (k : ℝ) (NH3 Br2 NH3_new : ℝ) (v1 v2 : ℝ):
  (v1 = k * NH3^8 * Br2) →
  (v2 = k * NH3_new^8 * Br2) →
  (v2 / v1 = 60) →
  NH3_new = 60 ^ (1 / 8) :=
by
  intro hv1 hv2 hratio
  sorry

end reaction_rate_reduction_l246_246306


namespace equilateral_triangle_ratio_l246_246732

theorem equilateral_triangle_ratio (s : ℝ) (h_s : s = 12) : 
  (let A := (√3 * s^2) / 4 in let P := 3 * s in A / P = √3) :=
by
  sorry

end equilateral_triangle_ratio_l246_246732


namespace find_faulty_keys_l246_246956

-- Define the conditions given in the problem
def total_digits : ℕ := 10
def registered_digits : ℕ := 7
def missing_digits : ℕ := 3
def defective_key_min_presses : ℕ := 5
def defective_key_successful_presses : ℕ := 2

-- Define that we need to find which keys could be faulty
def possible_faulty_keys : List ℤ := [7, 9]

-- The main theorem statement
theorem find_faulty_keys (total_digits = 10) (registered_digits = 7) (missing_digits = 3)
  (defective_key_min_presses = 5) (defective_key_successful_presses ≥ 2) :
  possible_faulty_keys = [7, 9] :=
by
  sorry

end find_faulty_keys_l246_246956


namespace coordinates_equality_l246_246890

theorem coordinates_equality (a b : ℤ) 
  (h1 : b - 1 = 2) 
  (h2 : a + 3 = -1) : a + b = -1 :=
by 
  sorry

end coordinates_equality_l246_246890


namespace opposite_of_negative_five_l246_246937

theorem opposite_of_negative_five : ∃ x : ℤ, -5 + x = 0 ∧ x = 5 :=
by
  use 5
  split
  {
    sorry,
  }
  {
    refl,
  }

end opposite_of_negative_five_l246_246937


namespace total_cost_l246_246940

-- Definition: Cost of first 100 notebooks
def cost_first_100_notebooks : ℕ := 230

-- Definition: Cost per notebook beyond the first 100 notebooks
def cost_additional_notebooks (n : ℕ) : ℕ := n * 2

-- Theorem: Total cost given a > 100 notebooks
theorem total_cost (a : ℕ) (h : a > 100) : (cost_first_100_notebooks + cost_additional_notebooks (a - 100) = 2 * a + 30) := by
  sorry

end total_cost_l246_246940


namespace man_wage_l246_246472

variable (m w b : ℝ) -- wages of man, woman, boy respectively
variable (W : ℝ) -- number of women equivalent to 5 men and 8 boys

-- Conditions given in the problem
axiom condition1 : 5 * m = W * w
axiom condition2 : W * w = 8 * b
axiom condition3 : 5 * m + 8 * b + 8 * b = 90

-- Prove the wage of one man
theorem man_wage : m = 6 := 
by
  -- proof steps would be here, but skipped as per instructions
  sorry

end man_wage_l246_246472


namespace tom_to_luke_ratio_l246_246001

theorem tom_to_luke_ratio (Tom Luke Anthony : ℕ) 
  (hAnthony : Anthony = 44) 
  (hTom : Tom = 33) 
  (hLuke : Luke = Anthony / 4) : 
  Tom / Nat.gcd Tom Luke = 3 ∧ Luke / Nat.gcd Tom Luke = 1 := 
by
  sorry

end tom_to_luke_ratio_l246_246001


namespace trajectory_is_one_branch_of_hyperbola_l246_246812

open Real

-- Condition 1: Given points F1 and F2
def F1 : ℝ × ℝ := (-3, 0)
def F2 : ℝ × ℝ := (3, 0)

-- Condition 2: Moving point P such that |PF1| - |PF2| = 4
def satisfies_condition (P : ℝ × ℝ) : Prop :=
  abs (dist P F1) - abs (dist P F2) = 4

-- Prove the trajectory of point P is one branch of a hyperbola
theorem trajectory_is_one_branch_of_hyperbola (P : ℝ × ℝ) (h : satisfies_condition P) : 
  (∃ a b : ℝ, ∀ x y: ℝ, satisfies_condition (x, y) → (((x^2 / a^2) - (y^2 / b^2) = 1) ∨ ((x^2 / a^2) - (y^2 / b^2) = -1))) :=
sorry

end trajectory_is_one_branch_of_hyperbola_l246_246812


namespace remainingAreaCalculation_l246_246272

noncomputable def totalArea : ℝ := 9500.0
noncomputable def lizzieGroupArea : ℝ := 2534.1
noncomputable def hilltownTeamArea : ℝ := 2675.95
noncomputable def greenValleyCrewArea : ℝ := 1847.57

theorem remainingAreaCalculation :
  (totalArea - (lizzieGroupArea + hilltownTeamArea + greenValleyCrewArea) = 2442.38) :=
by
  sorry

end remainingAreaCalculation_l246_246272


namespace linear_equation_check_l246_246160

theorem linear_equation_check : 
  (∃ a b : ℝ, a ≠ 0 ∧ (∀ x : ℝ, a * x + b = 1)) ∧ 
  ¬ (∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (∀ x y : ℝ, a * x + b * y = 3)) ∧ 
  ¬ (∀ x : ℝ, x^2 - 2 * x = 0) ∧ 
  ¬ (∀ x : ℝ, x - 1 / x = 0) := 
sorry

end linear_equation_check_l246_246160


namespace terminating_decimal_l246_246340

theorem terminating_decimal : (47 : ℚ) / (2 * 5^4) = 376 / 10^4 :=
by sorry

end terminating_decimal_l246_246340


namespace min_value_expression_l246_246816

theorem min_value_expression (a b : ℝ) (h1 : 2 * a + b = 1) (h2 : a > 0) (h3 : b > 0) :
  (1 / a) + ((1 - b) / b) = 2 + 2 * Real.sqrt 2 :=
by
  sorry

end min_value_expression_l246_246816


namespace unique_ordered_triples_count_l246_246649

theorem unique_ordered_triples_count :
  ∃ (n : ℕ), n = 1 ∧ ∀ (a b c : ℕ), 1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧
  abc = 4 * (ab + bc + ca) ∧ a = c / 4 -> False :=
sorry

end unique_ordered_triples_count_l246_246649


namespace max_abs_sum_x_y_l246_246099

theorem max_abs_sum_x_y (x y : ℝ) (h : x^2 + y^2 = 4) : |x| + |y| ≤ 2 * real.sqrt 2 :=
sorry

end max_abs_sum_x_y_l246_246099


namespace curve_is_circle_l246_246060

theorem curve_is_circle (r : ℝ) (θ : ℝ) (h : r = 3) : 
  ∃ (c : ℝ) (p : ℝ × ℝ), c = 3 ∧ p = (3 * Real.cos θ, 3 * Real.sin θ) := 
sorry

end curve_is_circle_l246_246060


namespace find_d_l246_246443

open Real

theorem find_d (a b c d : ℝ) 
  (h : a^3 + b^3 + c^3 + a^2 + b^2 + 1 = d^2 + d + sqrt (a + b + c - 2 * d)) : 
  d = 1 ∨ d = -(4 / 3) :=
sorry

end find_d_l246_246443


namespace winning_percentage_l246_246392

theorem winning_percentage (total_votes majority : ℕ) (h1 : total_votes = 455) (h2 : majority = 182) :
  ∃ P : ℕ, P = 70 ∧ (P / 100) * total_votes - ((100 - P) / 100) * total_votes = majority := 
sorry

end winning_percentage_l246_246392


namespace sarah_ellie_total_reflections_l246_246799

def sarah_tall_reflections : ℕ := 10
def sarah_wide_reflections : ℕ := 5
def sarah_narrow_reflections : ℕ := 8

def ellie_tall_reflections : ℕ := 6
def ellie_wide_reflections : ℕ := 3
def ellie_narrow_reflections : ℕ := 4

def tall_mirror_passages : ℕ := 3
def wide_mirror_passages : ℕ := 5
def narrow_mirror_passages : ℕ := 4

def total_reflections (sarah_tall sarah_wide sarah_narrow ellie_tall ellie_wide ellie_narrow
    tall_passages wide_passages narrow_passages : ℕ) : ℕ :=
  (sarah_tall * tall_passages + sarah_wide * wide_passages + sarah_narrow * narrow_passages) +
  (ellie_tall * tall_passages + ellie_wide * wide_passages + ellie_narrow * narrow_passages)

theorem sarah_ellie_total_reflections :
  total_reflections sarah_tall_reflections sarah_wide_reflections sarah_narrow_reflections
  ellie_tall_reflections ellie_wide_reflections ellie_narrow_reflections
  tall_mirror_passages wide_mirror_passages narrow_mirror_passages = 136 :=
by
  sorry

end sarah_ellie_total_reflections_l246_246799


namespace min_circles_l246_246675

noncomputable def segments_intersecting_circles (N : ℕ) : Prop :=
  ∀ seg : (ℝ × ℝ) × ℝ, (seg.fst.fst ≥ 0 ∧ seg.fst.fst + seg.snd ≤ 100 ∧ seg.fst.snd ≥ 0 ∧ seg.fst.snd ≤ 100 ∧ seg.snd = 10) →
    ∃ c : ℝ × ℝ, (dist c seg.fst < 1 ∧ c.fst ≥ 0 ∧ c.fst ≤ 100 ∧ c.snd ≥ 0 ∧ c.snd ≤ 100) 

theorem min_circles (N : ℕ) (h : segments_intersecting_circles N) : N ≥ 400 :=
sorry

end min_circles_l246_246675


namespace find_k_l246_246350

def vector (α : Type*) := α × α

def a : vector ℝ := (1, 3)
def b (k : ℝ) : vector ℝ := (-2, k)

def vector_parallel (u v : vector ℝ) : Prop :=
  ∃ (λ:ℝ), λ • u = v

theorem find_k (k : ℝ) : 
  vector_parallel ((a.1 + 2 * b(k).1, a.2 + 2 * b(k).2))
                  ((3 * a.1 - b(k).1, 3 * a.2 - b(k).2))
  → k = -6 := 
by 
  sorry

end find_k_l246_246350


namespace basketball_substitution_mod_1000_l246_246770

def basketball_substitution_count_mod (n_playing n_substitutes max_subs : ℕ) : ℕ :=
  let no_subs := 1
  let one_sub := n_playing * n_substitutes
  let two_subs := n_playing * (n_playing - 1) * (n_substitutes * (n_substitutes - 1)) / 2
  let three_subs := n_playing * (n_playing - 1) * (n_playing - 2) *
                    (n_substitutes * (n_substitutes - 1) * (n_substitutes - 2)) / 6
  no_subs + one_sub + two_subs + three_subs 

theorem basketball_substitution_mod_1000 :
  basketball_substitution_count_mod 9 9 3 % 1000 = 10 :=
  by 
    -- Here the proof would be implemented
    sorry

end basketball_substitution_mod_1000_l246_246770


namespace bicycle_final_price_l246_246971

-- Define initial conditions
def original_price : ℝ := 200
def wednesday_discount : ℝ := 0.40
def friday_increase : ℝ := 0.20
def saturday_discount : ℝ := 0.25

-- Statement to prove that the final price, after all discounts and increases, is $108
theorem bicycle_final_price :
  (original_price * (1 - wednesday_discount) * (1 + friday_increase) * (1 - saturday_discount)) = 108 := by
  sorry

end bicycle_final_price_l246_246971


namespace find_z_plus_one_over_y_l246_246886

theorem find_z_plus_one_over_y 
  (x y z : ℝ) 
  (h1 : 0 < x)
  (h2 : 0 < y)
  (h3 : 0 < z)
  (h4 : x * y * z = 1)
  (h5 : x + 1/z = 4)
  (h6 : y + 1/x = 20) :
  z + 1/y = 26 / 79 :=
by
  sorry

end find_z_plus_one_over_y_l246_246886


namespace convert_base_5_to_base_10_l246_246201

theorem convert_base_5_to_base_10 :
  let a3 := 2 * 5^3
  let a2 := 2 * 5^2
  let a1 := 0 * 5^1
  let a0 := 2 * 5^0
  a3 + a2 + a1 + a0 = 302 := by
  let a3 := 2 * 5^3
  let a2 := 2 * 5^2
  let a1 := 0 * 5^1
  let a0 := 2 * 5^0
  show a3 + a2 + a1 + a0 = 302
  sorry

end convert_base_5_to_base_10_l246_246201


namespace num_perfect_square_factors_of_180_l246_246090

theorem num_perfect_square_factors_of_180 (n : ℕ) (h : n = 180) :
  ∃ k : ℕ, k = 4 ∧ ∀ d : ℕ, d ∣ n → ∃ a b c : ℕ, d = 2^a * 3^b * 5^c ∧ a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 :=
by
  use 4
  sorry

end num_perfect_square_factors_of_180_l246_246090


namespace probability_of_selection_l246_246646

-- Problem setup
def number_of_students : ℕ := 54
def number_of_students_eliminated : ℕ := 4
def number_of_remaining_students : ℕ := number_of_students - number_of_students_eliminated
def number_of_students_selected : ℕ := 5

-- Statement to be proved
theorem probability_of_selection :
  (number_of_students_selected : ℚ) / (number_of_students : ℚ) = 5 / 54 :=
sorry

end probability_of_selection_l246_246646


namespace events_equally_likely_iff_N_eq_18_l246_246397

variable (N : ℕ)

-- Define the number of combinations in the draws
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the sums of selecting balls
noncomputable def S_63_10 (N: ℕ) : ℕ := sorry -- placeholder definition
noncomputable def S_44_8 (N: ℕ) : ℕ := sorry 

-- Condition for the events being equally likely
theorem events_equally_likely_iff_N_eq_18 : 
  (S_63_10 N * C N 8 = S_44_8 N * C N 10) ↔ N = 18 :=
sorry

end events_equally_likely_iff_N_eq_18_l246_246397


namespace angie_age_l246_246006

variables (A : ℕ)

theorem angie_age (h : 2 * A + 4 = 20) : A = 8 :=
by {
  -- Proof will be provided in actual usage or practice
  sorry
}

end angie_age_l246_246006


namespace logical_equivalence_l246_246756

theorem logical_equivalence (P Q R : Prop) :
  ((¬ P ∧ ¬ Q) → ¬ R) ↔ (R → (P ∨ Q)) :=
by sorry

end logical_equivalence_l246_246756


namespace number_of_intersections_l246_246497

def line₁ (x y : ℝ) := 2 * x - 3 * y + 6 = 0
def line₂ (x y : ℝ) := 5 * x + 2 * y - 10 = 0
def line₃ (x y : ℝ) := x - 2 * y + 1 = 0
def line₄ (x y : ℝ) := 3 * x - 4 * y + 8 = 0

theorem number_of_intersections : 
  ∃! (p₁ p₂ p₃ : ℝ × ℝ),
    (line₁ p₁.1 p₁.2 ∨ line₂ p₁.1 p₁.2) ∧ (line₃ p₁.1 p₁.2 ∨ line₄ p₁.1 p₁.2) ∧
    (line₁ p₂.1 p₂.2 ∨ line₂ p₂.1 p₂.2) ∧ (line₃ p₂.1 p₂.2 ∨ line₄ p₂.1 p₂.2) ∧
    (line₁ p₃.1 p₃.2 ∨ line₂ p₃.1 p₃.2) ∧ (line₃ p₃.1 p₃.2 ∨ line₄ p₃.1 p₃.2) ∧
    p₁ ≠ p₂ ∧ p₂ ≠ p₃ ∧ p₁ ≠ p₃ := 
sorry

end number_of_intersections_l246_246497


namespace Newville_Academy_fraction_l246_246185

theorem Newville_Academy_fraction :
  let total_students := 100
  let enjoy_sports := 0.7 * total_students
  let not_enjoy_sports := 0.3 * total_students
  let say_enjoy_right := 0.75 * enjoy_sports
  let say_not_enjoy_wrong := 0.25 * enjoy_sports
  let say_not_enjoy_right := 0.85 * not_enjoy_sports
  let say_enjoy_wrong := 0.15 * not_enjoy_sports
  let say_not_enjoy_total := say_not_enjoy_wrong + say_not_enjoy_right
  let say_not_enjoy_but_enjoy := say_not_enjoy_wrong
  (say_not_enjoy_but_enjoy / say_not_enjoy_total) = (7 / 17) := by
  sorry

end Newville_Academy_fraction_l246_246185


namespace problem_solution_l246_246460

def expr := 1 + 1 / (1 + 1 / (1 + 1))
def answer : ℚ := 5 / 3

theorem problem_solution : expr = answer :=
by
  sorry

end problem_solution_l246_246460


namespace x_is_4286_percent_less_than_y_l246_246671

theorem x_is_4286_percent_less_than_y (x y : ℝ) (h : y = 1.75 * x) : 
  ((y - x) / y) * 100 = 42.86 :=
by
  sorry

end x_is_4286_percent_less_than_y_l246_246671


namespace point_quadrant_l246_246536

theorem point_quadrant (x y : ℝ) (h : (x + y)^2 = x^2 + y^2 - 2) : 
  ((x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)) :=
by
  sorry

end point_quadrant_l246_246536


namespace strawberries_final_count_l246_246493

def initial_strawberries := 300
def buckets := 5
def strawberries_per_bucket := initial_strawberries / buckets
def strawberries_removed_per_bucket := 20
def redistributed_in_first_two := 15
def redistributed_in_third := 25

-- Defining the final counts after redistribution
def final_strawberries_first := strawberries_per_bucket - strawberries_removed_per_bucket + redistributed_in_first_two
def final_strawberries_second := strawberries_per_bucket - strawberries_removed_per_bucket + redistributed_in_first_two
def final_strawberries_third := strawberries_per_bucket - strawberries_removed_per_bucket + redistributed_in_third
def final_strawberries_fourth := strawberries_per_bucket - strawberries_removed_per_bucket
def final_strawberries_fifth := strawberries_per_bucket - strawberries_removed_per_bucket

theorem strawberries_final_count :
  final_strawberries_first = 55 ∧
  final_strawberries_second = 55 ∧
  final_strawberries_third = 65 ∧
  final_strawberries_fourth = 40 ∧
  final_strawberries_fifth = 40 := by
  sorry

end strawberries_final_count_l246_246493


namespace solve_CD_l246_246992

noncomputable def find_CD : Prop :=
  ∃ C D : ℝ, (C = 11 ∧ D = 0) ∧ (∀ x : ℝ, x ≠ -4 ∧ x ≠ 12 → 
    (7 * x - 3) / ((x + 4) * (x - 12)) = C / (x + 4) + D / (x - 12))

theorem solve_CD : find_CD :=
sorry

end solve_CD_l246_246992


namespace find_r_divisibility_l246_246645

theorem find_r_divisibility :
  ∃ r : ℝ, (10 * r ^ 2 - 4 * r - 26 = 0 ∧ (r = (19 / 10) ∨ r = (-3 / 2))) ∧ (r = -3 / 2) ∧ (10 * r ^ 3 - 5 * r ^ 2 - 52 * r + 60 = 0) :=
by
  sorry

end find_r_divisibility_l246_246645


namespace dividend_correct_l246_246169

-- Given constants for the problem
def divisor := 19
def quotient := 7
def remainder := 6

-- Dividend formula
def dividend := (divisor * quotient) + remainder

-- The proof problem statement
theorem dividend_correct : dividend = 139 := by
  sorry

end dividend_correct_l246_246169


namespace number_of_questionnaires_from_unit_D_l246_246782

theorem number_of_questionnaires_from_unit_D 
  (a d : ℕ) 
  (total : ℕ) 
  (samples : ℕ → ℕ) 
  (h_seq : samples 0 = a ∧ samples 1 = a + d ∧ samples 2 = a + 2 * d ∧ samples 3 = a + 3 * d)
  (h_total : samples 0 + samples 1 + samples 2 + samples 3 = total)
  (h_stratified : ∀ (i : ℕ), i < 4 → samples i * 100 / total = 20 → i = 1) 
  : samples 3 = 40 := sorry

end number_of_questionnaires_from_unit_D_l246_246782


namespace largest_value_of_x_l246_246952

theorem largest_value_of_x (x : ℝ) (hx : x / 3 + 1 / (7 * x) = 1 / 2) : 
  x = (21 + Real.sqrt 105) / 28 := 
sorry

end largest_value_of_x_l246_246952


namespace possible_faulty_keys_l246_246958

theorem possible_faulty_keys (d : ℕ) (digits : list ℕ) (len_d : digits.length = 10) 
  (registered : list ℕ) (len_r : registered.length = 7) :
  (∃ d ∈ digits, d = 7 ∨ d = 9) :=
by
  sorry

end possible_faulty_keys_l246_246958


namespace odd_function_value_at_2_l246_246513

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x : ℝ, f (-x) = -f x)

theorem odd_function_value_at_2 : f (-2) + f (2) = 0 :=
by
  sorry

end odd_function_value_at_2_l246_246513


namespace problem_1_problem_2_l246_246813

theorem problem_1 (a b c d : ℝ) (h : d > 0) (h_sum : a + b + c + d = 3) :
  (a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c)) ≥ 3 / 4 := 
sorry

theorem problem_2 (a b c d : ℝ) (h : d > 0) (h_sum : a + b + c + d = 3) :
  (a / (b + 2 * c + 3 * d) + b / (c + 2 * d + 3 * a) + c / (d + 2 * a + 3 * b) + d / (a + 2 * b + 3 * c)) ≥ 2 / 3 :=
sorry

end problem_1_problem_2_l246_246813


namespace opposite_of_neg_five_l246_246923

theorem opposite_of_neg_five : -(-5) = 5 := by
  sorry

end opposite_of_neg_five_l246_246923


namespace find_f1_and_f_prime1_l246_246437

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Conditions
axiom differentiable_f : Differentiable ℝ f
axiom f_def : ∀ x : ℝ, f x = 2 * x^2 - f' 1 * x - 3

-- Proof using conditions
theorem find_f1_and_f_prime1 : f 1 + (f' 1) = -1 :=
sorry

end find_f1_and_f_prime1_l246_246437


namespace opposite_of_neg5_is_pos5_l246_246912

theorem opposite_of_neg5_is_pos5 : -(-5) = 5 := 
by
  sorry

end opposite_of_neg5_is_pos5_l246_246912


namespace angle_quadrant_l246_246666

theorem angle_quadrant 
  (θ : Real) 
  (h1 : Real.cos θ > 0) 
  (h2 : Real.sin (2 * θ) < 0) : 
  3 * π / 2 < θ ∧ θ < 2 * π := 
by
  sorry

end angle_quadrant_l246_246666


namespace no_uniformly_colored_rectangle_l246_246795

open Int

def point := (ℤ × ℤ)

def is_green (P : point) : Prop :=
  3 ∣ (P.1 + P.2)

def is_red (P : point) : Prop :=
  ¬ is_green P

def is_rectangle (A B C D : point) : Prop :=
  A.1 = D.1 ∧ B.1 = C.1 ∧ A.2 = B.2 ∧ D.2 = C.2

def rectangle_area (A B : point) : ℤ :=
  abs (B.1 - A.1) * abs (B.2 - A.2)

theorem no_uniformly_colored_rectangle :
  ∀ (A B C D : point) (k : ℕ), 
  is_rectangle A B C D →
  rectangle_area A C = 2^k →
  ¬ (is_green A ∧ is_green B ∧ is_green C ∧ is_green D) ∧
  ¬ (is_red A ∧ is_red B ∧ is_red C ∧ is_red D) :=
by sorry

end no_uniformly_colored_rectangle_l246_246795


namespace opposite_of_neg5_is_pos5_l246_246915

theorem opposite_of_neg5_is_pos5 : -(-5) = 5 := 
by
  sorry

end opposite_of_neg5_is_pos5_l246_246915


namespace inequality_proof_l246_246465

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 3) :
  (1 / (x + y + z^2)) + (1 / (y + z + x^2)) + (1 / (z + x + y^2)) ≤ 1 :=
by
  sorry

end inequality_proof_l246_246465


namespace length_of_yellow_line_l246_246477

theorem length_of_yellow_line
  (w1 w2 w3 w4 : ℝ) (path_width : ℝ) (middle_line_dist : ℝ)
  (h1 : w1 = 40) (h2 : w2 = 10) (h3 : w3 = 20) (h4 : w4 = 30) (h5 : path_width = 5) (h6 : middle_line_dist = 2.5) :
  w1 - path_width * middle_line_dist/2 + w2 + w3 + w4 - path_width * middle_line_dist/2 = 95 :=
by sorry

end length_of_yellow_line_l246_246477


namespace cook_remaining_potatoes_l246_246607

def total_time_to_cook_remaining_potatoes (total_potatoes cooked_potatoes time_per_potato : ℕ) : ℕ :=
  (total_potatoes - cooked_potatoes) * time_per_potato

theorem cook_remaining_potatoes 
  (total_potatoes cooked_potatoes time_per_potato : ℕ) 
  (h_total_potatoes : total_potatoes = 13)
  (h_cooked_potatoes : cooked_potatoes = 5)
  (h_time_per_potato : time_per_potato = 6) : 
  total_time_to_cook_remaining_potatoes total_potatoes cooked_potatoes time_per_potato = 48 :=
by
  -- Proof not required
  sorry

end cook_remaining_potatoes_l246_246607


namespace exists_disjoint_nonempty_subsets_with_equal_sum_l246_246156

theorem exists_disjoint_nonempty_subsets_with_equal_sum :
  ∀ (A : Finset ℕ), (A.card = 11) → (∀ a ∈ A, 1 ≤ a ∧ a ≤ 100) →
  ∃ (B C : Finset ℕ), B ≠ ∅ ∧ C ≠ ∅ ∧ B ∩ C = ∅ ∧ (B ∪ C ⊆ A) ∧ (B.sum id = C.sum id) :=
by
  sorry

end exists_disjoint_nonempty_subsets_with_equal_sum_l246_246156


namespace quadratic_real_roots_range_l246_246808

theorem quadratic_real_roots_range (k : ℝ) : 
  (∃ x : ℝ, k * x^2 + 3 * x - 1 = 0) ↔ (k ≥ -9 / 4 ∧ k ≠ 0) :=
by
  sorry

end quadratic_real_roots_range_l246_246808


namespace product_of_roots_l246_246229

theorem product_of_roots (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) (h : x + 3 / x = y + 3 / y) : x * y = 3 :=
sorry

end product_of_roots_l246_246229


namespace coexistent_pair_example_coexistent_pair_neg_coexistent_pair_find_a_l246_246129

section coexistent_rational_number_pairs

-- Definitions based on the problem conditions:
def coexistent_pair (a b : ℚ) : Prop := a - b = a * b + 1

-- Proof problem 1
theorem coexistent_pair_example : coexistent_pair 3 (1/2) :=
sorry

-- Proof problem 2
theorem coexistent_pair_neg (m n : ℚ) (h : coexistent_pair m n) :
  coexistent_pair (-n) (-m) :=
sorry

-- Proof problem 3
example : ∃ (p q : ℚ), coexistent_pair p q ∧ (p, q) ≠ (2, 1/3) ∧ (p, q) ≠ (5, 2/3) ∧ (p, q) ≠ (3, 1/2) :=
sorry

-- Proof problem 4
theorem coexistent_pair_find_a (a : ℚ) (h : coexistent_pair a 3) :
  a = -2 :=
sorry

end coexistent_rational_number_pairs

end coexistent_pair_example_coexistent_pair_neg_coexistent_pair_find_a_l246_246129


namespace part_time_employees_l246_246775

theorem part_time_employees (total_employees : ℕ) (full_time_employees : ℕ) (h1 : total_employees = 65134) (h2 : full_time_employees = 63093) :
  total_employees - full_time_employees = 2041 :=
by
  -- Suppose that total_employees - full_time_employees = 2041
  sorry

end part_time_employees_l246_246775


namespace opposite_of_neg_five_is_five_l246_246916

-- Define the condition for the opposite of a number
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- State the theorem that the opposite of -5 is 5
theorem opposite_of_neg_five_is_five : is_opposite (-5) 5 :=
by
  -- Insert the proof here
  sorry

end opposite_of_neg_five_is_five_l246_246916


namespace log_expression_evaluation_l246_246189

open Real

theorem log_expression_evaluation : log 5 * log 20 + (log 2) ^ 2 = 1 := 
sorry

end log_expression_evaluation_l246_246189


namespace sum_of_cubes_divisible_l246_246135

theorem sum_of_cubes_divisible (a b c : ℤ) (h : (a + b + c) % 3 = 0) : 
  (a^3 + b^3 + c^3) % 3 = 0 := 
by sorry

end sum_of_cubes_divisible_l246_246135


namespace x_add_inv_ge_two_l246_246133

theorem x_add_inv_ge_two (x : ℝ) (hx : x > 0) : x + (1 / x) ≥ 2 :=
by
  sorry

end x_add_inv_ge_two_l246_246133


namespace coloring_satisfies_conditions_l246_246418

-- Definitions of point colors
inductive Color
| Red
| White
| Black

def color_point (x y : ℤ) : Color :=
  if (x + y) % 2 = 1 then Color.Red
  else if (x % 2 = 1 ∧ y % 2 = 0) then Color.White
  else Color.Black

-- Problem statement
theorem coloring_satisfies_conditions :
  (∀ y : ℤ, ∃ x1 x2 x3 : ℤ, 
    color_point x1 y = Color.Red ∧ 
    color_point x2 y = Color.White ∧
    color_point x3 y = Color.Black)
  ∧ 
  (∀ (x1 y1 x2 y2 x3 y3 : ℤ),
    color_point x1 y1 = Color.White →
    color_point x2 y2 = Color.Red →
    color_point x3 y3 = Color.Black →
    ∃ x4 y4, 
      color_point x4 y4 = Color.Red ∧ 
      x4 = x3 + (x1 - x2) ∧ 
      y4 = y3 + (y1 - y2)) :=
by
  sorry

end coloring_satisfies_conditions_l246_246418


namespace perpendicular_bisector_c_value_l246_246290

theorem perpendicular_bisector_c_value :
  (∃ c : ℝ, ∀ x y : ℝ, 
    2 * x - y = c ↔ x = 5 ∧ y = 8) → c = 2 := 
by
  sorry

end perpendicular_bisector_c_value_l246_246290


namespace ellipse_properties_l246_246806

theorem ellipse_properties :
  ∀ {x y : ℝ}, 4 * x^2 + 2 * y^2 = 16 →
    (∃ a b e c, a = 2 * Real.sqrt 2 ∧ b = 2 ∧ e = Real.sqrt 2 / 2 ∧ c = 2) ∧
    (∃ f1 f2, f1 = (0, 2) ∧ f2 = (0, -2)) ∧
    (∃ v1 v2 v3 v4, v1 = (0, 2 * Real.sqrt 2) ∧ v2 = (0, -2 * Real.sqrt 2) ∧ v3 = (2, 0) ∧ v4 = (-2, 0)) :=
by
  sorry

end ellipse_properties_l246_246806


namespace euler_polyhedron_problem_l246_246800

theorem euler_polyhedron_problem : 
  ( ∀ (V E F T S : ℕ), F = 42 → (T = 2 ∧ S = 3) → V - E + F = 2 → 100 * S + 10 * T + V = 337 ) := 
by sorry

end euler_polyhedron_problem_l246_246800


namespace andy_more_candies_than_caleb_l246_246327

theorem andy_more_candies_than_caleb :
  let billy_initial := 6
  let caleb_initial := 11
  let andy_initial := 9
  let father_packet := 36
  let billy_additional := 8
  let caleb_additional := 11
  let billy_total := billy_initial + billy_additional
  let caleb_total := caleb_initial + caleb_additional
  let total_given := billy_additional + caleb_additional
  let andy_additional := father_packet - total_given
  let andy_total := andy_initial + andy_additional
  andy_total - caleb_total = 4 :=
by {
  sorry
}

end andy_more_candies_than_caleb_l246_246327


namespace union_of_A_and_B_l246_246517

-- Definition of the sets A and B
def A : Set ℕ := {1, 2}
def B : Set ℕ := ∅

-- The theorem to prove
theorem union_of_A_and_B : A ∪ B = {1, 2} := 
by sorry

end union_of_A_and_B_l246_246517


namespace polynomial_relation_l246_246084

def M (m : ℚ) : ℚ := 5 * m^2 - 8 * m + 1
def N (m : ℚ) : ℚ := 4 * m^2 - 8 * m - 1

theorem polynomial_relation (m : ℚ) : M m > N m := by
  sorry

end polynomial_relation_l246_246084


namespace solution_set_quadratic_inequality_l246_246580

def quadraticInequalitySolutionSet 
  (x : ℝ) : Prop := 
  3 + 5 * x - 2 * x^2 > 0

theorem solution_set_quadratic_inequality :
  { x : ℝ | quadraticInequalitySolutionSet x } = 
  { x : ℝ | - (1:ℝ) / 2 < x ∧ x < 3 } :=
by {
  sorry
}

end solution_set_quadratic_inequality_l246_246580


namespace part_a_l246_246773

theorem part_a (b c: ℤ) : ∃ (n : ℕ) (a : ℕ → ℤ), 
  (a 0 = b) ∧ (a n = c) ∧ (∀ i, 1 ≤ i ∧ i ≤ n → |a i - a (i - 1)| = i^2) :=
sorry

end part_a_l246_246773


namespace count_original_scissors_l246_246152

def originalScissors (addedScissors totalScissors : ℕ) : ℕ := totalScissors - addedScissors

theorem count_original_scissors :
  ∃ (originalScissorsCount : ℕ), originalScissorsCount = originalScissors 13 52 := 
  sorry

end count_original_scissors_l246_246152


namespace geometry_problem_z_eq_87_deg_l246_246626

noncomputable def measure_angle_z (ABC ABD ADB : Real) : Real :=
  43 -- \angle ADB

theorem geometry_problem_z_eq_87_deg
  (ABC : Real)
  (h1 : ABC = 130)
  (ABD : Real)
  (h2 : ABD = 50)
  (ADB : Real)
  (h3 : ADB = 43) :
  measure_angle_z ABC ABD ADB = 87 :=
by
  unfold measure_angle_z
  sorry

end geometry_problem_z_eq_87_deg_l246_246626


namespace quadrant_of_half_angle_in_second_quadrant_l246_246359

theorem quadrant_of_half_angle_in_second_quadrant (θ : ℝ) (h : π / 2 < θ ∧ θ < π) :
  (0 < θ / 2 ∧ θ / 2 < π / 2) ∨ (π < θ / 2 ∧ θ / 2 < 3 * π / 2) :=
by
  sorry

end quadrant_of_half_angle_in_second_quadrant_l246_246359


namespace find_a3_l246_246796

theorem find_a3 (a : ℕ → ℕ) (h₁ : a 1 = 2)
  (h₂ : ∀ n, (1 + 2 * a (n + 1)) = (1 + 2 * a n) + 1) : a 3 = 3 :=
by
  -- This is where the proof would go, but we'll leave it as sorry for now.
  sorry

end find_a3_l246_246796


namespace not_sophomores_percentage_l246_246260

theorem not_sophomores_percentage (total_students : ℕ)
    (juniors_percentage : ℚ) (juniors : ℕ)
    (seniors : ℕ) (freshmen sophomores : ℕ)
    (h1 : total_students = 800)
    (h2 : juniors_percentage = 0.22)
    (h3 : juniors = juniors_percentage * total_students)
    (h4 : seniors = 160)
    (h5 : freshmen = sophomores + 48)
    (h6 : freshmen + sophomores + juniors + seniors = total_students) :
    ((total_students - sophomores : ℚ) / total_students) * 100 = 74 := by
  sorry

end not_sophomores_percentage_l246_246260


namespace choose_elements_sum_eq_binom_l246_246138

open Finset
open Fintype

theorem choose_elements_sum_eq_binom {n : ℕ} 
  (S : Fin n → Finset (Fin (2 * n))) 
  (hS_nonempty : ∀ i, (S i).Nonempty)
  (hS_sum : ∑ i in univ, ∑ x in S i, (x : ℕ) = (2 * n + 1) * n / 2) :
  ∃ f : Fin n → Fin (2 * n), (∀ i, f i ∈ S i) ∧ ∑ i in univ, (f i : ℕ) = (2 * n - 1) * n / 2 :=
  sorry

end choose_elements_sum_eq_binom_l246_246138


namespace problem_l246_246654

def A : Set ℝ := {x | x^2 + x - 6 > 0}
def B : Set ℝ := {x | 0 < x ∧ x < 6}
def C := (Aᶜ) ∩ B

theorem problem : C = {x | 0 < x ∧ x ≤ 2} :=
by
  sorry

end problem_l246_246654


namespace exterior_angle_regular_octagon_l246_246850

theorem exterior_angle_regular_octagon : 
  (∃ n : ℕ, n = 8 ∧ ∀ (i : ℕ), i < n → true) → 
  ∃ θ : ℝ, θ = 45 := by
  sorry

end exterior_angle_regular_octagon_l246_246850


namespace inequality_log_l246_246068

variable (a b c : ℝ)
variable (h1 : 1 < a)
variable (h2 : 1 < b)
variable (h3 : 1 < c)

theorem inequality_log (a b c : ℝ) (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 < c) : 
  2 * ( (Real.log a / Real.log b) / (a + b) + (Real.log b / Real.log c) / (b + c) + (Real.log c / Real.log a) / (c + a) ) 
  ≥ 9 / (a + b + c) := 
sorry

end inequality_log_l246_246068


namespace ethan_days_worked_per_week_l246_246057

-- Define the conditions
def hourly_wage : ℕ := 18
def hours_per_day : ℕ := 8
def total_earnings : ℕ := 3600
def weeks_worked : ℕ := 5

-- Compute derived values
def daily_earnings : ℕ := hourly_wage * hours_per_day
def weekly_earnings : ℕ := total_earnings / weeks_worked

-- Define the proposition to be proved
theorem ethan_days_worked_per_week : ∃ d: ℕ, d * daily_earnings = weekly_earnings ∧ d = 5 :=
by
  use 5
  simp [daily_earnings, weekly_earnings]
  sorry

end ethan_days_worked_per_week_l246_246057


namespace tan_sum_eq_two_l246_246411

theorem tan_sum_eq_two (a b c : ℝ) (A B C : ℝ) (h1 : a / b + b / a = 4 * Real.cos C)
  (h2 : a = b * Real.sin A / Real.sin B) (h3 : c = (a^2 + b^2 - 2 * a * b * Real.cos C) / a)
  (h4 : C = Real.acos (a^2 + b^2 - c^2) / (2 * a * b)) :
  Real.tan C / Real.tan A + Real.tan C / Real.tan B = 2 :=
by
  sorry

end tan_sum_eq_two_l246_246411


namespace simplify_f_of_alpha_value_of_f_given_cos_l246_246519

variable (α : Real) (f : Real → Real)

def third_quadrant (α : Real) : Prop :=
  3 * Real.pi / 2 < α ∧ α < 2 * Real.pi

noncomputable def f_def : Real → Real := 
  λ α => (Real.sin (α - Real.pi / 2) * 
           Real.cos (3 * Real.pi / 2 + α) * 
           Real.tan (Real.pi - α)) / 
           (Real.tan (-α - Real.pi) * 
           Real.sin (-Real.pi - α))

theorem simplify_f_of_alpha (h : third_quadrant α) :
  f α = -Real.cos α := sorry

theorem value_of_f_given_cos 
  (h : third_quadrant α) 
  (cos_h : Real.cos (α - 3 / 2 * Real.pi) = 1 / 5) :
  f α = 2 * Real.sqrt 6 / 5 := sorry

end simplify_f_of_alpha_value_of_f_given_cos_l246_246519


namespace five_people_six_chairs_l246_246838

theorem five_people_six_chairs : 
  ∃ (f : Fin 6 → Bool), (∑ i, if f i then 1 else 0) = 5 ∧ 
  (∃ (g : Fin 5 → Fin 6), ∀ i j : Fin 5, i ≠ j → g i ≠ g j) →
  (5!) * (choose 6 5) = 720 :=
by
  sorry

end five_people_six_chairs_l246_246838


namespace find_principal_l246_246758

theorem find_principal 
  (SI : ℝ) 
  (R : ℝ) 
  (T : ℝ) 
  (h_SI : SI = 4052.25) 
  (h_R : R = 9) 
  (h_T : T = 5) : 
  (SI * 100) / (R * T) = 9005 := 
by 
  rw [h_SI, h_R, h_T]
  sorry

end find_principal_l246_246758


namespace cranberry_parts_l246_246685

theorem cranberry_parts (L C : ℕ) :
  L = 3 →
  L + C = 72 →
  C = L + 18 →
  C = 21 :=
by
  intros hL hSum hDiff
  sorry

end cranberry_parts_l246_246685


namespace average_percentage_for_all_students_l246_246761

-- Definitions of the variables
def students1 : Nat := 15
def average1 : Nat := 75
def students2 : Nat := 10
def average2 : Nat := 90
def total_students : Nat := students1 + students2
def total_percentage1 : Nat := students1 * average1
def total_percentage2 : Nat := students2 * average2
def total_percentage : Nat := total_percentage1 + total_percentage2

-- Main theorem stating the average percentage for all students.
theorem average_percentage_for_all_students :
  total_percentage / total_students = 81 := by
  sorry

end average_percentage_for_all_students_l246_246761


namespace diagonal_of_square_l246_246895

theorem diagonal_of_square (s d : ℝ) (h_perimeter : 4 * s = 40) : d = 10 * Real.sqrt 2 :=
by
  sorry

end diagonal_of_square_l246_246895


namespace opposite_of_neg_five_l246_246901

/-- Definition of the opposite of a number -/
def opposite (a : Int) : Int := -a

theorem opposite_of_neg_five : opposite (-5) = 5 := by
  sorry

end opposite_of_neg_five_l246_246901


namespace strips_area_coverage_l246_246807

-- Define paper strips and their properties
def length_strip : ℕ := 8
def width_strip : ℕ := 2
def number_of_strips : ℕ := 5

-- Total area without considering overlaps
def area_one_strip : ℕ := length_strip * width_strip
def total_area_without_overlap : ℕ := number_of_strips * area_one_strip

-- Overlapping areas
def area_center_overlap : ℕ := 4 * (2 * 2)
def area_additional_overlap : ℕ := 2 * (2 * 2)
def total_overlap_area : ℕ := area_center_overlap + area_additional_overlap

-- Actual area covered
def actual_area_covered : ℕ := total_area_without_overlap - total_overlap_area

-- Theorem stating the required proof
theorem strips_area_coverage : actual_area_covered = 56 :=
by sorry

end strips_area_coverage_l246_246807


namespace minutes_to_seconds_l246_246244

theorem minutes_to_seconds (m : ℝ) (hm : m = 6.5) : m * 60 = 390 := by
  sorry

end minutes_to_seconds_l246_246244


namespace glee_club_female_members_l246_246386

theorem glee_club_female_members (m f : ℕ) 
  (h1 : f = 2 * m) 
  (h2 : m + f = 18) : 
  f = 12 :=
by
  sorry

end glee_club_female_members_l246_246386


namespace opposite_of_negative_five_l246_246939

theorem opposite_of_negative_five : ∃ x : ℤ, -5 + x = 0 ∧ x = 5 :=
by
  use 5
  split
  {
    sorry,
  }
  {
    refl,
  }

end opposite_of_negative_five_l246_246939


namespace restaurant_total_spent_l246_246425

theorem restaurant_total_spent (appetizer_cost : ℕ) (entree_cost : ℕ) (num_entrees : ℕ) (tip_rate : ℚ) 
  (H1 : appetizer_cost = 10) (H2 : entree_cost = 20) (H3 : num_entrees = 4) (H4 : tip_rate = 0.20) :
  appetizer_cost + num_entrees * entree_cost + (appetizer_cost + num_entrees * entree_cost) * tip_rate = 108 :=
by
  sorry

end restaurant_total_spent_l246_246425


namespace parallel_lines_slope_l246_246378

theorem parallel_lines_slope (m : ℝ) (h : (x + (1 + m) * y + m - 2 = 0) ∧ (m * x + 2 * y + 6 = 0)) :
  m = 1 ∨ m = -2 :=
  sorry

end parallel_lines_slope_l246_246378


namespace four_friends_same_group_prob_l246_246887

-- Definitions of conditions
def students : Nat := 900
def groups : Nat := 3
def group_size : Nat := students / groups

-- The probability of assigning one student to a specific group
def prob_assign_to_same_group : ℚ := 1 / groups

-- Lean statement for the proof problem
theorem four_friends_same_group_prob :
  (prob_assign_to_same_group ^ 3) = 1 / 27 :=
by
  -- Proof to be filled in
  sorry

end four_friends_same_group_prob_l246_246887


namespace tomato_plants_per_row_l246_246490

-- Definitions based on given conditions.
variables (T C P : ℕ)

-- Condition 1: For each row of tomato plants, she is planting 2 rows of cucumbers
def cucumber_rows (T : ℕ) := 2 * T

-- Condition 2: She has enough room for 15 rows of plants in total
def total_rows (T : ℕ) (C : ℕ) := T + C = 15

-- Condition 3: If each plant produces 3 tomatoes, she will have 120 tomatoes in total
def total_tomatoes (P : ℕ) := 5 * P * 3 = 120

-- The task is to prove that P = 8
theorem tomato_plants_per_row : 
  ∀ T C P : ℕ, cucumber_rows T = C → total_rows T C → total_tomatoes P → P = 8 :=
by
  -- The actual proof will go here.
  sorry

end tomato_plants_per_row_l246_246490


namespace num_children_with_dogs_only_l246_246538

-- Defining the given values and constants
def total_children : ℕ := 30
def children_with_cats : ℕ := 12
def children_with_dogs_and_cats : ℕ := 6

-- Define the required proof statement
theorem num_children_with_dogs_only : 
  ∃ (D : ℕ), D + children_with_dogs_and_cats + (children_with_cats - children_with_dogs_and_cats) = total_children ∧ D = 18 :=
by
  sorry

end num_children_with_dogs_only_l246_246538


namespace find_y_l246_246426

theorem find_y (t : ℝ) (x : ℝ) (y : ℝ) (h1 : x = 3 - t) (h2 : y = 3 * t + 6) (h3 : x = -6) : y = 33 := by
  sorry

end find_y_l246_246426


namespace shortest_distance_between_circles_l246_246457

-- Conditions
def first_circle (x y : ℝ) : Prop := x^2 - 10 * x + y^2 - 4 * y - 7 = 0
def second_circle (x y : ℝ) : Prop := x^2 + 14 * x + y^2 + 6 * y + 49 = 0

-- Goal: Prove the shortest distance between the two circles is 4
theorem shortest_distance_between_circles : 
  -- Given conditions about the equations of the circles
  (∀ x y : ℝ, first_circle x y ↔ (x - 5)^2 + (y - 2)^2 = 36) ∧ 
  (∀ x y : ℝ, second_circle x y ↔ (x + 7)^2 + (y + 3)^2 = 9) →
  -- Assert the shortest distance between the two circles is 4
  13 - (6 + 3) = 4 :=
by
  sorry

end shortest_distance_between_circles_l246_246457


namespace exists_identical_coordinates_l246_246834

theorem exists_identical_coordinates
  (O O' : ℝ × ℝ)
  (Ox Oy O'x' O'y' : ℝ → ℝ)
  (units_different : ∃ u v : ℝ, u ≠ v)
  (O_ne_O' : O ≠ O')
  (Ox_not_parallel_O'x' : ∃ θ : ℝ, θ ≠ 0 ∧ θ ≠ π) :
  ∃ S : ℝ × ℝ, (S.1 = Ox S.1 ∧ S.2 = Oy S.2) ∧ (S.1 = O'x' S.1 ∧ S.2 = O'y' S.2) :=
sorry

end exists_identical_coordinates_l246_246834


namespace num_distinguishable_octahedrons_l246_246338

-- Define the given conditions
def num_faces : ℕ := 8
def num_colors : ℕ := 8
def total_permutations : ℕ := Nat.factorial num_colors
def distinct_orientations : ℕ := 24

-- Prove the main statement
theorem num_distinguishable_octahedrons : total_permutations / distinct_orientations = 1680 :=
by
  sorry

end num_distinguishable_octahedrons_l246_246338


namespace three_digit_largest_fill_four_digit_smallest_fill_l246_246722

theorem three_digit_largest_fill (n : ℕ) (h1 : n * 1000 + 28 * 4 < 1000) : n ≤ 2 := sorry

theorem four_digit_smallest_fill (n : ℕ) (h2 : n * 1000 + 28 * 4 ≥ 1000) : 3 ≤ n := sorry

end three_digit_largest_fill_four_digit_smallest_fill_l246_246722


namespace irrational_roots_of_odd_quadratic_l246_246134

theorem irrational_roots_of_odd_quadratic (a b c : ℤ) (ha : Odd a) (hb : Odd b) (hc : Odd c) :
  ¬ ∃ p q : ℤ, q ≠ 0 ∧ gcd p q = 1 ∧ p * p = a * (p / q) * (p / q) + b * (p / q) + c := sorry

end irrational_roots_of_odd_quadratic_l246_246134


namespace five_student_committees_from_ten_select_two_committees_with_three_overlap_l246_246809

-- Lean statement for the first part: number of different five-student committees from ten students.
theorem five_student_committees_from_ten : 
  (Nat.choose 10 5) = 252 := 
by
  sorry

-- Lean statement for the second part: number of ways to choose two five-student committees with exactly three overlapping members.
theorem select_two_committees_with_three_overlap :
  ( (Nat.choose 10 5) * ( (Nat.choose 5 3) * (Nat.choose 5 2) ) ) / 2 = 12600 := 
by
  sorry

end five_student_committees_from_ten_select_two_committees_with_three_overlap_l246_246809


namespace arithmetic_sequence_S9_l246_246361

-- Define the sum of an arithmetic sequence: S_n
def arithmetic_sequence_sum (a d : ℕ → ℕ) (n : ℕ) : ℕ := (n * (2 * a (0) + (n - 1) * d (0))) / 2

-- Conditions
variable (a d : ℕ → ℕ)
variable (S_n : ℕ → ℕ)
variable (h1 : S_n 3 = 9)
variable (h2 : S_n 6 = 27)

-- Question: Prove that S_9 = 54
theorem arithmetic_sequence_S9 : S_n 9 = 54 := by
    sorry

end arithmetic_sequence_S9_l246_246361


namespace unique_x_intersect_l246_246219

theorem unique_x_intersect (m : ℝ) (h : ∀ x : ℝ, (m - 4) * x^2 - 2 * m * x - m - 6 = 0 → ∀ y : ℝ, (m - 4) * y^2 - 2 * m * y - m - 6 = 0 → x = y) :
  m = -4 ∨ m = 3 ∨ m = 4 :=
sorry

end unique_x_intersect_l246_246219


namespace max_abs_sum_sqrt2_l246_246098

theorem max_abs_sum_sqrt2 (x y : ℝ) (h : x^2 + y^2 = 4) : 
  ∃ (a : ℝ), (a = |x| + |y| ∧ a ≤ 2 * Real.sqrt 2) ∧ 
             ∀ (x y : ℝ), x^2 + y^2 = 4 → (|x| + |y|) ≤ 2 * Real.sqrt 2 :=
sorry

end max_abs_sum_sqrt2_l246_246098


namespace gray_region_area_l246_246053

theorem gray_region_area
  (center_C : ℝ × ℝ) (r_C : ℝ)
  (center_D : ℝ × ℝ) (r_D : ℝ)
  (C_center : center_C = (3, 5)) (C_radius : r_C = 5)
  (D_center : center_D = (13, 5)) (D_radius : r_D = 5) :
  let rect_area := 10 * 5
  let semi_circle_area := 12.5 * π
  rect_area - 2 * semi_circle_area = 50 - 25 * π := 
by 
  sorry

end gray_region_area_l246_246053


namespace number_of_people_going_on_trip_l246_246585

theorem number_of_people_going_on_trip
  (bags_per_person : ℕ)
  (weight_per_bag : ℕ)
  (total_luggage_capacity : ℕ)
  (additional_capacity : ℕ)
  (bags_per_additional_capacity : ℕ)
  (h1 : bags_per_person = 5)
  (h2 : weight_per_bag = 50)
  (h3 : total_luggage_capacity = 6000)
  (h4 : additional_capacity = 90) :
  (total_luggage_capacity + (bags_per_additional_capacity * weight_per_bag)) / (weight_per_bag * bags_per_person) = 42 := 
by
  simp [h1, h2, h3, h4]
  repeat { sorry }

end number_of_people_going_on_trip_l246_246585


namespace proof_problem_l246_246121

-- Definitions
def U : Set ℕ := {x | x < 7 ∧ x > 0}
def A : Set ℕ := {1, 2, 5}
def B : Set ℕ := {2, 3, 4, 5}

-- The equality proof statement
theorem proof_problem :
  (A ∩ B = {2, 5}) ∧
  ({x | x ∈ U ∧ ¬ (x ∈ A)} = {3, 4, 6}) ∧
  (A ∪ {x | x ∈ U ∧ ¬ (x ∈ B)} = {1, 2, 5, 6}) :=
by
  sorry

end proof_problem_l246_246121


namespace perpendicular_bisector_eq_l246_246894

theorem perpendicular_bisector_eq (x y : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 2 * x - 5 = 0 ∧ x^2 + y^2 + 2 * x - 4 * y - 4 = 0) →
  x + y - 1 = 0 :=
by
  sorry

end perpendicular_bisector_eq_l246_246894


namespace number_of_elements_in_M_intersect_N_l246_246127

noncomputable theory

def M : Set (ℝ × ℝ) := {p | ∃ k k' : ℤ, p = (k, k') ∧ tan (π * k') = 0 ∧ sin (π * k) = 0}
def N : Set (ℝ × ℝ) := {p | p.fst^2 + p.snd^2 ≤ 2}

theorem number_of_elements_in_M_intersect_N : 
  ∃ n : ℕ, n = 9 ∧ ∀ p : ℝ × ℝ, p ∈ M ∩ N → n = 9 :=
sorry

end number_of_elements_in_M_intersect_N_l246_246127


namespace problem_l246_246286

theorem problem (p q r : ℝ)
    (h1 : p * 1^2 + q * 1 + r = 5)
    (h2 : p * 2^2 + q * 2 + r = 3) :
  p + q + 2 * r = 10 := 
sorry

end problem_l246_246286


namespace bryan_total_after_discount_l246_246789

theorem bryan_total_after_discount 
  (n : ℕ) (p : ℝ) (d : ℝ) (h_n : n = 8) (h_p : p = 1785) (h_d : d = 0.12) :
  (n * p - (n * p * d) = 12566.4) :=
by
  sorry

end bryan_total_after_discount_l246_246789


namespace river_flow_rate_l246_246479

variables (d w : ℝ) (V : ℝ)

theorem river_flow_rate (h₁ : d = 4) (h₂ : w = 40) (h₃ : V = 10666.666666666666) :
  ((V / 60) / (d * w) * 3.6) = 4 :=
by sorry

end river_flow_rate_l246_246479


namespace max_value_expression_l246_246336

theorem max_value_expression (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1 / Real.sqrt 3) :
  27 * a * b * c + a * Real.sqrt (a^2 + 2 * b * c) + b * Real.sqrt (b^2 + 2 * c * a) + c * Real.sqrt (c^2 + 2 * a * b) ≤ 2 / (3 * Real.sqrt 3) :=
sorry

end max_value_expression_l246_246336


namespace percentage_cleared_land_l246_246972

theorem percentage_cleared_land (T C : ℝ) (hT : T = 6999.999999999999) (hC : 0.20 * C + 0.70 * C + 630 = C) :
  (C / T) * 100 = 90 :=
by {
  sorry
}

end percentage_cleared_land_l246_246972


namespace not_all_less_than_two_l246_246076

theorem not_all_less_than_two {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ¬(a + 1/b < 2 ∧ b + 1/c < 2 ∧ c + 1/a < 2) :=
sorry

end not_all_less_than_two_l246_246076


namespace factorial_base_a5_l246_246507

theorem factorial_base_a5 (b : ℕ → ℕ) (n : ℕ) (h_repr : 1034 = ∑ k in finset.range (n + 1), b (k + 1) * (k + 1)!)
    (h_bound : ∀ k, k < n + 1 → b (k + 1) ≤ k + 1) : b 5 = 5 :=
sorry

end factorial_base_a5_l246_246507


namespace find_staff_age_l246_246025

theorem find_staff_age (n_students : ℕ) (avg_age_students : ℕ) (avg_age_with_staff : ℕ) (total_students : ℕ) :
  n_students = 32 →
  avg_age_students = 16 →
  avg_age_with_staff = 17 →
  total_students = 33 →
  (33 * 17 - 32 * 16) = 49 :=
by
  intros
  sorry

end find_staff_age_l246_246025


namespace proof_problem_l246_246227

-- Declare x, y as real numbers
variables (x y : ℝ)

-- Define the condition given in the problem
def condition (k : ℝ) : Prop :=
  (x^2 + y^2) / (x^2 - y^2) + (x^2 - y^2) / (x^2 + y^2) = k

-- The main conclusion we need to prove given the condition
theorem proof_problem (k : ℝ) (h : condition x y k) :
  (x^8 + y^8) / (x^8 - y^8) + (x^8 - y^8) / (x^8 + y^8) = (k^4 + 24 * k^2 + 16) / (4 * k^3 + 16 * k) :=
sorry

end proof_problem_l246_246227


namespace andy_more_candies_than_caleb_l246_246326

theorem andy_more_candies_than_caleb :
  let billy_initial := 6
  let caleb_initial := 11
  let andy_initial := 9
  let father_packet := 36
  let billy_additional := 8
  let caleb_additional := 11
  let billy_total := billy_initial + billy_additional
  let caleb_total := caleb_initial + caleb_additional
  let total_given := billy_additional + caleb_additional
  let andy_additional := father_packet - total_given
  let andy_total := andy_initial + andy_additional
  andy_total - caleb_total = 4 :=
by {
  sorry
}

end andy_more_candies_than_caleb_l246_246326


namespace division_result_l246_246554

def m : ℕ := 16 ^ 2024

theorem division_result : m / 8 = 8 * 16 ^ 2020 :=
by
  -- sorry for the actual proof
  sorry

end division_result_l246_246554


namespace remainder_7_pow_93_mod_12_l246_246015

theorem remainder_7_pow_93_mod_12 : 7 ^ 93 % 12 = 7 := 
by
  -- the sequence repeats every two terms: 7, 1, 7, 1, ...
  sorry

end remainder_7_pow_93_mod_12_l246_246015


namespace calc_expression_l246_246324

theorem calc_expression : 3 * 3^4 - 9^32 / 9^30 = 162 := by
  -- We would provide the proof here, but skipping with sorry
  sorry

end calc_expression_l246_246324


namespace smallest_divisible_by_2022_l246_246018

theorem smallest_divisible_by_2022 (n : ℕ) (N : ℕ) :
  (N = 20230110) ∧ (∃ k : ℕ, N = 2023 * 10^n + k) ∧ N % 2022 = 0 → 
  ∀ M: ℕ, (∃ m : ℕ, M = 2023 * 10^n + m) ∧ M % 2022 = 0 → N ≤ M :=
sorry

end smallest_divisible_by_2022_l246_246018


namespace time_difference_between_car_and_minivan_arrival_l246_246605

variable (car_speed : ℝ := 40)
variable (minivan_speed : ℝ := 50)
variable (pass_time : ℝ := 1 / 6) -- in hours

theorem time_difference_between_car_and_minivan_arrival :
  (60 * (1 / 6 - (20 / 3 / 50))) = 2 := sorry

end time_difference_between_car_and_minivan_arrival_l246_246605


namespace opposite_of_neg_five_is_five_l246_246920

-- Define the condition for the opposite of a number
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- State the theorem that the opposite of -5 is 5
theorem opposite_of_neg_five_is_five : is_opposite (-5) 5 :=
by
  -- Insert the proof here
  sorry

end opposite_of_neg_five_is_five_l246_246920


namespace total_travel_time_correct_l246_246117

-- Define the conditions
def highway_distance : ℕ := 100 -- miles
def mountain_distance : ℕ := 15 -- miles
def break_time : ℕ := 30 -- minutes
def time_on_mountain_road : ℕ := 45 -- minutes
def speed_ratio : ℕ := 5

-- Define the speeds using the given conditions.
def mountain_speed := mountain_distance / time_on_mountain_road -- miles per minute
def highway_speed := speed_ratio * mountain_speed -- miles per minute

-- Prove that total trip time equals 240 minutes
def total_trip_time : ℕ := 2 * (time_on_mountain_road + (highway_distance / highway_speed)) + break_time

theorem total_travel_time_correct : total_trip_time = 240 := 
by
  -- to be proved
  sorry

end total_travel_time_correct_l246_246117


namespace liquid_left_after_evaporation_l246_246422

-- Definitions
def solution_y (total_mass : ℝ) : ℝ × ℝ :=
  (0.30 * total_mass, 0.70 * total_mass) -- liquid_x, water

def evaporate_water (initial_water : ℝ) (evaporated_mass : ℝ) : ℝ :=
  initial_water - evaporated_mass

-- Condition that new solution is 45% liquid x
theorem liquid_left_after_evaporation 
  (initial_mass : ℝ) 
  (evaporated_mass : ℝ)
  (added_mass : ℝ)
  (new_percentage_liquid_x : ℝ) :
  initial_mass = 8 → 
  evaporated_mass = 4 → 
  added_mass = 4 →
  new_percentage_liquid_x = 0.45 →
  solution_y initial_mass = (2.4, 5.6) →
  evaporate_water 5.6 evaporated_mass = 1.6 →
  solution_y added_mass = (1.2, 2.8) →
  2.4 + 1.2 = 3.6 →
  1.6 + 2.8 = 4.4 →
  0.45 * (3.6 + 4.4) = 3.6 →
  4 = 2.4 + 1.6 := sorry

end liquid_left_after_evaporation_l246_246422


namespace calories_burned_per_week_l246_246266

-- Definitions of the conditions
def classes_per_week : ℕ := 3
def hours_per_class : ℝ := 1.5
def calories_per_min : ℝ := 7
def minutes_per_hour : ℝ := 60

-- Theorem stating the proof problem
theorem calories_burned_per_week : 
  (classes_per_week * (hours_per_class * minutes_per_hour) * calories_per_min) = 1890 := by
  sorry

end calories_burned_per_week_l246_246266


namespace value_of_y_l246_246104

theorem value_of_y (x y : ℝ) (h1 : x = 2) (h2 : x^(3 * y) = 8) : y = 1 :=
by
  sorry

end value_of_y_l246_246104


namespace division_of_15_by_neg_5_l246_246049

theorem division_of_15_by_neg_5 : 15 / (-5) = -3 :=
by
  sorry

end division_of_15_by_neg_5_l246_246049


namespace amoeba_count_after_week_l246_246978

-- Definition of the initial conditions
def amoeba_splits_daily (n : ℕ) : ℕ := 2^n

-- Theorem statement translating the problem to Lean
theorem amoeba_count_after_week : amoeba_splits_daily 7 = 128 :=
by
  sorry

end amoeba_count_after_week_l246_246978


namespace opposite_of_neg_five_l246_246903

/-- Definition of the opposite of a number -/
def opposite (a : Int) : Int := -a

theorem opposite_of_neg_five : opposite (-5) = 5 := by
  sorry

end opposite_of_neg_five_l246_246903


namespace carol_points_loss_l246_246052

theorem carol_points_loss 
  (first_round_points : ℕ) (second_round_points : ℕ) (end_game_points : ℕ) 
  (h1 : first_round_points = 17) 
  (h2 : second_round_points = 6) 
  (h3 : end_game_points = 7) : 
  (first_round_points + second_round_points - end_game_points = 16) :=
by 
  sorry

end carol_points_loss_l246_246052


namespace Joe_team_wins_eq_1_l246_246267

-- Definition for the points a team gets for winning a game.
def points_per_win := 3
-- Definition for the points a team gets for a tie game.
def points_per_tie := 1

-- Given conditions
def Joe_team_draws := 3
def first_place_wins := 2
def first_place_ties := 2
def points_difference := 2

def first_place_points := (first_place_wins * points_per_win) + (first_place_ties * points_per_tie)

def Joe_team_total_points := first_place_points - points_difference
def Joe_team_points_from_ties := Joe_team_draws * points_per_tie
def Joe_team_points_from_wins := Joe_team_total_points - Joe_team_points_from_ties

-- To prove: number of games Joe's team won
theorem Joe_team_wins_eq_1 : (Joe_team_points_from_wins / points_per_win) = 1 :=
by
  sorry

end Joe_team_wins_eq_1_l246_246267


namespace base_angle_isosceles_triangle_l246_246252

theorem base_angle_isosceles_triangle (x : ℝ) :
    (∀ (Δ : Triangle) (a b c : ℝ),
        Δ.Angles = [a, b, c] ∧ a = 80 ∧ IsIsosceles Δ → 
        (b = 80 ∨ b = 50)) :=
by
    -- Assuming AngleSum and IsIsosceles are defined appropriately in the context
    sorry

end base_angle_isosceles_triangle_l246_246252


namespace six_digit_start_5_no_12_digit_perfect_square_l246_246023

theorem six_digit_start_5_no_12_digit_perfect_square :
  ∀ (n : ℕ), (500000 ≤ n ∧ n < 600000) → 
  (∀ (m : ℕ), n * 10^6 + m ≠ k^2) :=
by
  sorry

end six_digit_start_5_no_12_digit_perfect_square_l246_246023


namespace parabola_x_intercepts_l246_246533

theorem parabola_x_intercepts : 
  ∃! x : ℝ, ∃ y : ℝ, y = 0 ∧ x = -3 * y ^ 2 + 2 * y + 3 :=
by 
  sorry

end parabola_x_intercepts_l246_246533


namespace min_max_value_in_interval_l246_246348

theorem min_max_value_in_interval : ∀ (x : ℝ),
  -2 < x ∧ x < 5 →
  ∃ (y : ℝ), (y = -1.5 ∨ y = 1.5) ∧ y = (x^2 - 4 * x + 6) / (2 * x - 4) := 
by sorry

end min_max_value_in_interval_l246_246348


namespace simplified_equation_has_solution_l246_246574

theorem simplified_equation_has_solution (n : ℤ) :
  (∃ x y z : ℤ, x^2 + y^2 + z^2 - x * y - y * z - z * x = n) →
  (∃ x y : ℤ, x^2 + y^2 - x * y = n) :=
by
  intros h
  exact sorry

end simplified_equation_has_solution_l246_246574


namespace clothing_loss_l246_246035

theorem clothing_loss
  (a : ℝ)
  (h1 : ∃ x y : ℝ, x * 1.25 = a ∧ y * 0.75 = a ∧ x + y - 2 * a = -8) :
  a = 60 :=
sorry

end clothing_loss_l246_246035


namespace ratio_male_whales_l246_246409

def num_whales_first_trip_males : ℕ := 28
def num_whales_first_trip_females : ℕ := 56
def num_whales_second_trip_babies : ℕ := 8
def num_whales_second_trip_parents_males : ℕ := 8
def num_whales_second_trip_parents_females : ℕ := 8
def num_whales_third_trip_females : ℕ := 56
def total_whales : ℕ := 178

theorem ratio_male_whales (M : ℕ) (ratio : ℕ × ℕ) 
  (h_total_whales : num_whales_first_trip_males + num_whales_first_trip_females 
    + num_whales_second_trip_babies + num_whales_second_trip_parents_males 
    + num_whales_second_trip_parents_females + M + num_whales_third_trip_females = total_whales) 
  (h_ratio : ratio = ((M : ℕ) / Nat.gcd M num_whales_first_trip_males, 
                       num_whales_first_trip_males / Nat.gcd M num_whales_first_trip_males)) 
  : ratio = (1, 2) :=
by
  sorry

end ratio_male_whales_l246_246409


namespace ratio_area_perimeter_eq_sqrt3_l246_246728

theorem ratio_area_perimeter_eq_sqrt3 :
  let side_length := 12
  let altitude := side_length * (Real.sqrt 3) / 2
  let area := (1 / 2) * side_length * altitude
  let perimeter := 3 * side_length
  let ratio := area / perimeter
  ratio = Real.sqrt 3 := 
by
  sorry

end ratio_area_perimeter_eq_sqrt3_l246_246728


namespace oil_bill_for_january_l246_246835

-- Definitions and conditions
def ratio_F_J (F J : ℝ) : Prop := F / J = 3 / 2
def ratio_F_M (F M : ℝ) : Prop := F / M = 4 / 5
def ratio_F_J_modified (F J : ℝ) : Prop := (F + 20) / J = 5 / 3
def ratio_F_M_modified (F M : ℝ) : Prop := (F + 20) / M = 2 / 3

-- The main statement to prove
theorem oil_bill_for_january (J F M : ℝ) 
  (h1 : ratio_F_J F J)
  (h2 : ratio_F_M F M)
  (h3 : ratio_F_J_modified F J)
  (h4 : ratio_F_M_modified F M) :
  J = 120 :=
sorry

end oil_bill_for_january_l246_246835


namespace probability_three_cards_same_l246_246154

-- Let A, B, C each send a card to D or E, with each choice being equally likely.
noncomputable def probability_all_send_same (p : ℕ) : ℚ :=
  if h : p = 1 then 1 else sorry

theorem probability_three_cards_same :
  probability_all_send_same 2 = 1 / 4 :=
by
  sorry

end probability_three_cards_same_l246_246154


namespace smallest_int_square_eq_3x_plus_72_l246_246017

theorem smallest_int_square_eq_3x_plus_72 :
  ∃ x : ℤ, x^2 = 3 * x + 72 ∧ (∀ y : ℤ, y^2 = 3 * y + 72 → x ≤ y) :=
sorry

end smallest_int_square_eq_3x_plus_72_l246_246017


namespace molecular_weight_of_compound_is_correct_l246_246159

noncomputable def molecular_weight (nC nH nN nO : ℕ) (wC wH wN wO : ℝ) :=
  nC * wC + nH * wH + nN * wN + nO * wO

theorem molecular_weight_of_compound_is_correct :
  molecular_weight 8 18 2 4 12.01 1.008 14.01 16.00 = 206.244 :=
by
  sorry

end molecular_weight_of_compound_is_correct_l246_246159


namespace fixed_point_inequality_l246_246648

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 3 * a^((x + 1) / 2) - 4

theorem fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a (-1) = -1 :=
sorry

theorem inequality (a : ℝ) (x : ℝ) (h : a > 1) :
  f a (x - 3 / 4) ≥ 3 / (a^(x^2 / 2)) - 4 :=
sorry

end fixed_point_inequality_l246_246648


namespace alison_money_l246_246783

theorem alison_money (k b br bt al : ℝ) 
  (h1 : al = 1/2 * bt) 
  (h2 : bt = 4 * br) 
  (h3 : br = 2 * k) 
  (h4 : k = 1000) : 
  al = 4000 := 
by 
  sorry

end alison_money_l246_246783


namespace sum_of_fractions_l246_246342

theorem sum_of_fractions : (1 / 3 : ℚ) + (2 / 7) = 13 / 21 :=
by
  sorry

end sum_of_fractions_l246_246342


namespace smallest_integer_with_18_divisors_l246_246749

def num_divisors (n : ℕ) : ℕ := 
  (factors n).freq.map (λ x => x.2 + 1).prod

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, num_divisors n = 18 ∧ ∀ m : ℕ, num_divisors m = 18 → n ≤ m := 
by
  let n := 180
  use n
  constructor
  ...
  sorry

end smallest_integer_with_18_divisors_l246_246749


namespace exist_identical_2x2_squares_l246_246473

theorem exist_identical_2x2_squares : 
  ∃ sq1 sq2 : Finset (Fin 5 × Fin 5), 
    sq1.card = 4 ∧ sq2.card = 4 ∧ 
    (∀ (i : Fin 5) (j : Fin 5), 
      (i = 0 ∧ j = 0) ∨ (i = 4 ∧ j = 4) → 
      (i, j) ∈ sq1 ∧ (i, j) ∈ sq2 ∧ 
      (sq1 ≠ sq2 → ∃ p ∈ sq1, p ∉ sq2)) :=
sorry

end exist_identical_2x2_squares_l246_246473


namespace nap_duration_is_two_hours_l246_246269

-- Conditions as definitions in Lean
def naps_per_week : ℕ := 3
def days : ℕ := 70
def total_nap_hours : ℕ := 60

-- Calculate the duration of each nap
theorem nap_duration_is_two_hours :
  ∃ (nap_duration : ℕ), nap_duration = 2 ∧
  (days / 7) * naps_per_week * nap_duration = total_nap_hours :=
by
  sorry

end nap_duration_is_two_hours_l246_246269


namespace Alice_has_3_more_dimes_than_quarters_l246_246617

-- Definitions of the conditions given in the problem
variable (n d : ℕ) -- number of 5-cent and 10-cent coins
def q : ℕ := 10
def total_coins : ℕ := 30
def total_value : ℕ := 435
def extra_dimes : ℕ := 6

-- Conditions translated to Lean
axiom total_coin_count : n + d + q = total_coins
axiom total_value_count : 5 * n + 10 * d + 25 * q = total_value
axiom dime_difference : d = n + extra_dimes

-- The theorem that needs to be proven: Alice has 3 more 10-cent coins than 25-cent coins.
theorem Alice_has_3_more_dimes_than_quarters :
  d - q = 3 :=
sorry

end Alice_has_3_more_dimes_than_quarters_l246_246617


namespace probability_even_toys_l246_246600

theorem probability_even_toys:
  let total_toys := 21
  let even_toys := 10
  let probability_first_even := (even_toys : ℚ) / total_toys
  let probability_second_even := (even_toys - 1 : ℚ) / (total_toys - 1)
  let probability_both_even := probability_first_even * probability_second_even
  probability_both_even = 3 / 14 :=
by
  sorry

end probability_even_toys_l246_246600


namespace greatest_value_of_x_l246_246158

theorem greatest_value_of_x : ∀ x : ℝ, 4*x^2 + 6*x + 3 = 5 → x ≤ 1/2 :=
by
  intro x
  intro h
  sorry

end greatest_value_of_x_l246_246158


namespace complement_intersection_complement_l246_246856

-- Define the sets U, A, and B
def U : Set ℕ := {1, 2, 3, 4, 6}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Define the statement of the proof problem
theorem complement_intersection_complement:
  (U \ (A ∩ B)) = {1, 4, 6} := by
  sorry

end complement_intersection_complement_l246_246856


namespace largest_value_fraction_l246_246701

noncomputable def largest_value (x y : ℝ) : ℝ := (x + y) / x

theorem largest_value_fraction
  (x y : ℝ)
  (hx1 : -5 ≤ x)
  (hx2 : x ≤ -3)
  (hy1 : 3 ≤ y)
  (hy2 : y ≤ 5)
  (hy_odd : ∃ k : ℤ, y = 2 * k + 1) :
  largest_value x y = 0.4 :=
sorry

end largest_value_fraction_l246_246701


namespace actual_distance_between_towns_l246_246573

theorem actual_distance_between_towns
  (d_map : ℕ) (scale1 : ℕ) (scale2 : ℕ) (distance1 : ℕ) (distance2 : ℕ) (remaining_distance : ℕ) :
  d_map = 9 →
  scale1 = 10 →
  distance1 = 5 →
  scale2 = 8 →
  remaining_distance = d_map - distance1 →
  d_map = distance1 + remaining_distance →
  (distance1 * scale1 + remaining_distance * scale2 = 82) := by
  intros h1 h2 h3 h4 h5 h6
  sorry

end actual_distance_between_towns_l246_246573


namespace ratio_of_work_capacity_l246_246141

theorem ratio_of_work_capacity (work_rate_A work_rate_B : ℝ)
  (hA : work_rate_A = 1 / 45)
  (hAB : work_rate_A + work_rate_B = 1 / 18) :
  work_rate_A⁻¹ / work_rate_B⁻¹ = 3 / 2 :=
by
  sorry

end ratio_of_work_capacity_l246_246141


namespace jacob_fifth_test_score_l246_246680

theorem jacob_fifth_test_score (s1 s2 s3 s4 s5 : ℕ) :
  s1 = 85 ∧ s2 = 79 ∧ s3 = 92 ∧ s4 = 84 ∧ ((s1 + s2 + s3 + s4 + s5) / 5 = 85) →
  s5 = 85 :=
sorry

end jacob_fifth_test_score_l246_246680


namespace cryptarithmetic_proof_l246_246110

theorem cryptarithmetic_proof (A B C D : ℕ) 
  (h1 : A * B = 6) 
  (h2 : C = 2) 
  (h3 : A + B + D = 13) 
  (h4 : A + B + C = D) : 
  D = 6 :=
by
  sorry

end cryptarithmetic_proof_l246_246110


namespace total_suitcases_correct_l246_246560

-- Conditions as definitions
def num_siblings : Nat := 4
def suitcases_per_sibling : Nat := 2
def num_parents : Nat := 2
def suitcases_per_parent : Nat := 3

-- Total suitcases calculation
def total_suitcases :=
  (num_siblings * suitcases_per_sibling) + (num_parents * suitcases_per_parent)

-- Statement to prove
theorem total_suitcases_correct : total_suitcases = 14 :=
by
  sorry

end total_suitcases_correct_l246_246560


namespace seating_arrangements_l246_246842

/--
Prove that the number of ways to seat five people in a row of six chairs is 720.
-/
theorem seating_arrangements (people : ℕ) (chairs : ℕ) (h_people : people = 5) (h_chairs : chairs = 6) :
  ∃ (n : ℕ), n = 720 ∧ n = (6 * 5 * 4 * 3 * 2) :=
by
  sorry

end seating_arrangements_l246_246842


namespace rate_of_stream_l246_246176

-- Definitions from problem conditions
def rowing_speed_still_water : ℕ := 24

-- Assume v is the rate of the stream
variable (v : ℕ)

-- Time taken to row up is three times the time taken to row down
def rowing_time_condition : Prop :=
  1 / (rowing_speed_still_water - v) = 3 * (1 / (rowing_speed_still_water + v))

-- The rate of the stream (v) should be 12 kmph
theorem rate_of_stream (h : rowing_time_condition v) : v = 12 :=
  sorry

end rate_of_stream_l246_246176


namespace seating_arrangements_l246_246843

/--
Prove that the number of ways to seat five people in a row of six chairs is 720.
-/
theorem seating_arrangements (people : ℕ) (chairs : ℕ) (h_people : people = 5) (h_chairs : chairs = 6) :
  ∃ (n : ℕ), n = 720 ∧ n = (6 * 5 * 4 * 3 * 2) :=
by
  sorry

end seating_arrangements_l246_246843


namespace runners_align_same_point_first_time_l246_246274

def lap_time_stein := 6
def lap_time_rose := 10
def lap_time_schwartz := 18

theorem runners_align_same_point_first_time : Nat.lcm (Nat.lcm lap_time_stein lap_time_rose) lap_time_schwartz = 90 :=
by
  sorry

end runners_align_same_point_first_time_l246_246274


namespace second_container_sand_capacity_l246_246309

def volume (h: ℕ) (w: ℕ) (l: ℕ) : ℕ := h * w * l

def sand_capacity (v1: ℕ) (s1: ℕ) (v2: ℕ) : ℕ := (s1 * v2) / v1

theorem second_container_sand_capacity:
  let h1 := 3
  let w1 := 4
  let l1 := 6
  let s1 := 72
  let h2 := 3 * h1
  let w2 := 2 * w1
  let l2 := l1
  let v1 := volume h1 w1 l1
  let v2 := volume h2 w2 l2
  sand_capacity v1 s1 v2 = 432 :=
by {
  sorry
}

end second_container_sand_capacity_l246_246309


namespace sample_older_employees_count_l246_246308

-- Definitions of known quantities
def N := 400
def N_older := 160
def N_no_older := 240
def n := 50

-- The proof statement showing that the number of employees older than 45 in the sample equals 20
theorem sample_older_employees_count : 
  let proportion_older := (N_older:ℝ) / (N:ℝ)
  let n_older := proportion_older * (n:ℝ)
  n_older = 20 := by
  sorry

end sample_older_employees_count_l246_246308


namespace n_leq_1972_l246_246535

theorem n_leq_1972 (n : ℕ) (h1 : 4 ^ 27 + 4 ^ 1000 + 4 ^ n = k ^ 2) : n ≤ 1972 :=
by
  sorry

end n_leq_1972_l246_246535


namespace equally_likely_events_A_and_B_l246_246403

-- Given conditions definitions
def number_of_balls (N : ℕ) := N
def main_draw_balls (N : ℕ) := (finset.range (N + 1)).powerset_len 10
def additional_draw_balls (N : ℕ) := (finset.range (N + 1)).powerset_len 8
def lucky_sum_main (s : finset ℕ) := s.sum = 63
def lucky_sum_additional (s : finset ℕ) := s.sum = 44

-- Prove the events A and B are equally likely if and only if N = 18
theorem equally_likely_events_A_and_B (N : ℕ) :
  (∃ s ∈ main_draw_balls N, lucky_sum_main s) ∧ (∃ t ∈ additional_draw_balls N, lucky_sum_additional t)
  → (N = 18) :=
sorry

end equally_likely_events_A_and_B_l246_246403


namespace simplify_expression_l246_246421

theorem simplify_expression :
  20 * (14 / 15) * (2 / 18) * (5 / 4) = 70 / 27 :=
by sorry

end simplify_expression_l246_246421


namespace opposite_of_negative_five_l246_246906

theorem opposite_of_negative_five : ∀ x : Int, -5 + x = 0 → x = 5 :=
by
  intros x h
  sorry

end opposite_of_negative_five_l246_246906


namespace total_cost_after_discount_l246_246311

def num_children : Nat := 6
def num_adults : Nat := 10
def num_seniors : Nat := 4

def child_ticket_price : Real := 12
def adult_ticket_price : Real := 20
def senior_ticket_price : Real := 15

def group_discount_rate : Real := 0.15

theorem total_cost_after_discount :
  let total_cost_before_discount :=
    num_children * child_ticket_price +
    num_adults * adult_ticket_price +
    num_seniors * senior_ticket_price
  let discount := group_discount_rate * total_cost_before_discount
  let total_cost := total_cost_before_discount - discount
  total_cost = 282.20 := by
  sorry

end total_cost_after_discount_l246_246311


namespace probability_of_green_or_yellow_marble_l246_246726

theorem probability_of_green_or_yellow_marble :
  let total_marbles := 4 + 3 + 6 in
  let favorable_marbles := 4 + 3 in
  favorable_marbles / total_marbles = 7 / 13 :=
by
  sorry

end probability_of_green_or_yellow_marble_l246_246726


namespace dennis_rocks_left_l246_246987

theorem dennis_rocks_left : 
  ∀ (initial_rocks : ℕ) (fish_ate_fraction : ℕ) (fish_spit_out : ℕ),
    initial_rocks = 10 →
    fish_ate_fraction = 2 →
    fish_spit_out = 2 →
    initial_rocks - (initial_rocks / fish_ate_fraction) + fish_spit_out = 7 :=
by
  intros initial_rocks fish_ate_fraction fish_spit_out h_initial_rocks h_fish_ate_fraction h_fish_spit_out
  rw [h_initial_rocks, h_fish_ate_fraction, h_fish_spit_out]
  sorry

end dennis_rocks_left_l246_246987


namespace necessarily_positive_y_plus_z_l246_246878

theorem necessarily_positive_y_plus_z
  (x y z : ℝ)
  (hx : 0 < x ∧ x < 0.5) 
  (hy : -0.5 < y ∧ y < 0) 
  (hz : 0.5 < z ∧ z < 1) :
  y + z > 0 := 
by
  sorry

end necessarily_positive_y_plus_z_l246_246878


namespace totalPieces_l246_246419

   -- Definitions given by the conditions
   def packagesGum := 21
   def packagesCandy := 45
   def packagesMints := 30
   def piecesPerGumPackage := 9
   def piecesPerCandyPackage := 12
   def piecesPerMintPackage := 8

   -- Define the total pieces of gum, candy, and mints
   def totalPiecesGum := packagesGum * piecesPerGumPackage
   def totalPiecesCandy := packagesCandy * piecesPerCandyPackage
   def totalPiecesMints := packagesMints * piecesPerMintPackage

   -- The mathematical statement to prove
   theorem totalPieces :
     totalPiecesGum + totalPiecesCandy + totalPiecesMints = 969 :=
   by
     -- Proof is skipped
     sorry
   
end totalPieces_l246_246419


namespace degree_of_minus_5x4y_l246_246705

def degree_of_monomial (coeff : Int) (x_exp y_exp : Nat) : Nat :=
  x_exp + y_exp

theorem degree_of_minus_5x4y : degree_of_monomial (-5) 4 1 = 5 :=
by
  sorry

end degree_of_minus_5x4y_l246_246705


namespace sum_of_a_l246_246124

theorem sum_of_a (a_1 a_2 a_3 a_4 : ℚ) 
  (h : {a_1 * a_2, a_1 * a_3, a_1 * a_4, a_2 * a_3, a_2 * a_4, a_3 * a_4} = 
       {-24, -2, -3/2, -1/8, 1, 3}) :
  a_1 + a_2 + a_3 + a_4 = 9/4 ∨ a_1 + a_2 + a_3 + a_4 = -(9/4) :=
sorry

end sum_of_a_l246_246124


namespace opposite_of_neg5_is_pos5_l246_246913

theorem opposite_of_neg5_is_pos5 : -(-5) = 5 := 
by
  sorry

end opposite_of_neg5_is_pos5_l246_246913


namespace average_weight_l246_246544

theorem average_weight (w_girls w_boys : ℕ) (avg_girls avg_boys : ℕ) (n : ℕ) : 
  n = 5 → avg_girls = 45 → avg_boys = 55 → 
  w_girls = n * avg_girls → w_boys = n * avg_boys →
  ∀ total_weight, total_weight = w_girls + w_boys →
  ∀ avg_weight, avg_weight = total_weight / (2 * n) →
  avg_weight = 50 :=
by
  intros h_n h_avg_girls h_avg_boys h_w_girls h_w_boys h_total_weight h_avg_weight
  -- here you would start the proof, but it is omitted as per the instructions
  sorry

end average_weight_l246_246544


namespace find_varphi_l246_246377

theorem find_varphi
  (ϕ : ℝ)
  (h : ∃ k : ℤ, ϕ = (π / 8) + (k * π / 2)) :
  ϕ = π / 8 :=
by
  sorry

end find_varphi_l246_246377


namespace age_difference_constant_l246_246310

theorem age_difference_constant (seokjin_age_mother_age_diff : ∀ (t : ℕ), 33 - 7 = 26) : 
  ∀ (n : ℕ), 33 + n - (7 + n) = 26 := 
by
  sorry

end age_difference_constant_l246_246310


namespace operation_X_value_l246_246670

def operation_X (a b : ℤ) : ℤ := b + 7 * a - a^3 + 2 * b

theorem operation_X_value : operation_X 4 3 = -27 := by
  sorry

end operation_X_value_l246_246670


namespace intersection_nonempty_range_b_l246_246817

noncomputable def A : Set ℝ := {x | (x - 1) / (x + 1) < 0}
noncomputable def B (b : ℝ) (a : ℝ) : Set ℝ := {x | (x - b)^2 < a}

theorem intersection_nonempty_range_b (b : ℝ) : 
  A ∩ B b 1 ≠ ∅ ↔ -2 < b ∧ b < 2 := 
by
  sorry

end intersection_nonempty_range_b_l246_246817


namespace park_cycling_time_l246_246941

def length_breadth_ratio (L B : ℕ) : Prop := L / B = 1 / 3
def area_of_park (L B : ℕ) : Prop := L * B = 120000
def speed_of_cyclist : ℕ := 200 -- meters per minute
def perimeter (L B : ℕ) : ℕ := 2 * L + 2 * B
def time_to_complete_round (P v : ℕ) : ℕ := P / v

theorem park_cycling_time
  (L B : ℕ)
  (h_ratio : length_breadth_ratio L B)
  (h_area : area_of_park L B)
  : time_to_complete_round (perimeter L B) speed_of_cyclist = 8 :=
by
  sorry

end park_cycling_time_l246_246941


namespace find_number_l246_246171

theorem find_number (x : ℕ) (h : x + 18 = 44) : x = 26 :=
by
  sorry

end find_number_l246_246171


namespace avogadro_constant_problem_l246_246295

theorem avogadro_constant_problem 
  (N_A : ℝ) -- Avogadro's constant
  (mass1 : ℝ := 18) (molar_mass1 : ℝ := 20) (moles1 : ℝ := mass1 / molar_mass1) 
  (atoms_D2O_molecules : ℝ := 2) (atoms_D2O : ℝ := moles1 * atoms_D2O_molecules * N_A)
  (mass2 : ℝ := 14) (molar_mass_N2CO : ℝ := 28) (moles2 : ℝ := mass2 / molar_mass_N2CO)
  (electrons_per_molecule : ℝ := 14) (total_electrons_mixture : ℝ := moles2 * electrons_per_molecule * N_A)
  (volume3 : ℝ := 2.24) (temp_unk : Prop := true) -- unknown temperature
  (pressure_unk : Prop := true) -- unknown pressure
  (carbonate_molarity : ℝ := 0.1) (volume_solution : ℝ := 1) (moles_carbonate : ℝ := carbonate_molarity * volume_solution) 
  (anions_carbonate_solution : ℝ := moles_carbonate * N_A) :
  (atoms_D2O ≠ 2 * N_A) ∧ (anions_carbonate_solution > 0.1 * N_A) ∧ (total_electrons_mixture = 7 * N_A) -> 
  True := sorry

end avogadro_constant_problem_l246_246295


namespace S_11_l246_246226

variable (a : ℕ → ℕ) (S : ℕ → ℕ)

-- Conditions
-- Define that {a_n} is an arithmetic sequence.
def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) := ∀ n, a (n + 1) = a n + d

-- Sum of the first n terms of the arithmetic sequence
def sum_arithmetic_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) := ∀ n, S n = n * (a 1 + a n) / 2

-- Given condition: a_5 + a_7 = 14
def sum_condition (a : ℕ → ℕ) := a 5 + a 7 = 14

-- Prove S_{11} = 77
theorem S_11 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (d : ℕ)
  (h1 : arithmetic_sequence a d)
  (h2 : sum_arithmetic_sequence S a)
  (h3 : sum_condition a) :
  S 11 = 77 := by
  -- The proof steps would follow here.
  sorry

end S_11_l246_246226


namespace motorcycle_price_l246_246175

variable (x : ℝ) -- selling price of each motorcycle
variable (car_cost material_car material_motorcycle : ℝ)

theorem motorcycle_price
  (h1 : car_cost = 100)
  (h2 : material_car = 4 * 50)
  (h3 : material_motorcycle = 250)
  (h4 : 8 * x - material_motorcycle = material_car - car_cost + 50)
  : x = 50 := 
sorry

end motorcycle_price_l246_246175


namespace intersection_complement_A_B_subset_A_C_l246_246527

-- Definition of sets A, B, and complements in terms of conditions
def setA : Set ℝ := { x | 3 ≤ x ∧ x < 7 }
def setB : Set ℝ := { x | 2 < x ∧ x < 10 }
def complement_A : Set ℝ := { x | x < 3 ∨ x ≥ 7 }

-- Proof Problem (1)
theorem intersection_complement_A_B :
  ((complement_A) ∩ setB) = { x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10) } := 
  sorry

-- Definition of set C 
def setC (a : ℝ) : Set ℝ := { x | x < a }
-- Proof Problem (2)
theorem subset_A_C {a : ℝ} (h : setA ⊆ setC a) : a ≥ 7 :=
  sorry

end intersection_complement_A_B_subset_A_C_l246_246527


namespace upper_limit_of_people_l246_246802

theorem upper_limit_of_people (P : ℕ) (h1 : 36 = (3 / 8) * P) (h2 : P > 50) (h3 : (5 / 12) * P = 40) : P = 96 :=
by
  sorry

end upper_limit_of_people_l246_246802


namespace angie_age_l246_246007

variables (A : ℕ)

theorem angie_age (h : 2 * A + 4 = 20) : A = 8 :=
by {
  -- Proof will be provided in actual usage or practice
  sorry
}

end angie_age_l246_246007


namespace max_value_expr_l246_246077

variable (x y z : ℝ)

theorem max_value_expr (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) :
  (∃ a, ∀ x y z, (a = (x*y + y*z) / (x^2 + y^2 + z^2)) ∧ a ≤ (Real.sqrt 2) / 2) ∧
  (∃ x' y' z', (x' > 0) ∧ (y' > 0) ∧ (z' > 0) ∧ ((x'*y' + y'*z') / (x'^2 + y'^2 + z'^2) = (Real.sqrt 2) / 2)) :=
by
  sorry

end max_value_expr_l246_246077


namespace opposite_of_negative_five_l246_246929

theorem opposite_of_negative_five : (-(-5) = 5) :=
by
  sorry

end opposite_of_negative_five_l246_246929


namespace probability_below_x_axis_l246_246669

open Real

theorem probability_below_x_axis :
  let center : ℝ × ℝ := (2, sqrt 3),
      radius : ℝ := 2,
      circle_area : ℝ := π * radius ^ 2,
      sector_area : ℝ := 1 / 6 * circle_area,
      triangle_area : ℝ := 2 * sqrt 3,
      segment_area : ℝ := sector_area - triangle_area,
      probability : ℝ := segment_area / circle_area
  in probability = (1 / 6 - sqrt 3 / (2 * π)) := by
  sorry

end probability_below_x_axis_l246_246669


namespace five_people_six_chairs_l246_246837

/-- Number of ways to sit 5 people in 6 chairs -/
def ways_to_sit_in_chairs : ℕ :=
  6 * 5 * 4 * 3 * 2

theorem five_people_six_chairs : ways_to_sit_in_chairs = 720 := by
  -- placeholder for the proof
  sorry

end five_people_six_chairs_l246_246837


namespace almond_butter_ratio_l246_246128

theorem almond_butter_ratio
  (peanut_cost almond_cost batch_extra almond_per_batch : ℝ)
  (h1 : almond_cost = 3 * peanut_cost)
  (h2 : peanut_cost = 3)
  (h3 : almond_per_batch = batch_extra)
  (h4 : batch_extra = 3) :
  almond_per_batch / almond_cost = 1 / 3 := sorry

end almond_butter_ratio_l246_246128


namespace outer_boundary_diameter_l246_246776

theorem outer_boundary_diameter (fountain_diameter garden_width path_width : ℝ) 
(h1 : fountain_diameter = 12) 
(h2 : garden_width = 10) 
(h3 : path_width = 6) : 
2 * ((fountain_diameter / 2) + garden_width + path_width) = 44 :=
by
  -- Sorry, proof not needed for this statement
  sorry

end outer_boundary_diameter_l246_246776


namespace alison_money_l246_246784

theorem alison_money (k b br bt al : ℝ) 
  (h1 : al = 1/2 * bt) 
  (h2 : bt = 4 * br) 
  (h3 : br = 2 * k) 
  (h4 : k = 1000) : 
  al = 4000 := 
by 
  sorry

end alison_money_l246_246784


namespace necessary_and_sufficient_conditions_l246_246228

-- Define the function f
def f (a x : ℝ) : ℝ := a * x - x^2

-- Define the domain of x
def dom_x (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 1

theorem necessary_and_sufficient_conditions {a : ℝ} (ha : a > 0) :
  (∀ x : ℝ, dom_x x → f a x ≤ 1) ↔ (0 < a ∧ a ≤ 2) :=
sorry

end necessary_and_sufficient_conditions_l246_246228


namespace parabola_x_intercepts_l246_246534

theorem parabola_x_intercepts : 
  ∃! x : ℝ, ∃ y : ℝ, y = 0 ∧ x = -3 * y ^ 2 + 2 * y + 3 :=
by 
  sorry

end parabola_x_intercepts_l246_246534


namespace inscribed_circle_radius_l246_246486

noncomputable def side1 := 13
noncomputable def side2 := 13
noncomputable def side3 := 10
noncomputable def s := (side1 + side2 + side3) / 2
noncomputable def area := Real.sqrt (s * (s - side1) * (s - side2) * (s - side3))
noncomputable def inradius := area / s

theorem inscribed_circle_radius :
  inradius = 10 / 3 :=
by
  sorry

end inscribed_circle_radius_l246_246486


namespace problem_sequence_k_term_l246_246225

theorem problem_sequence_k_term (a : ℕ → ℤ) (S : ℕ → ℤ) (h₀ : ∀ n, S n = n^2 - 9 * n)
    (h₁ : ∀ n, a n = S n - S (n - 1)) (h₂ : 5 < a 8 ∧ a 8 < 8) : 8 = 8 :=
sorry

end problem_sequence_k_term_l246_246225


namespace math_problem_equivalence_l246_246050

theorem math_problem_equivalence :
  (-3 : ℚ) / (-1 - 3 / 4) * (3 / 4) / (3 / 7) = 3 := 
by 
  sorry

end math_problem_equivalence_l246_246050


namespace arithmetic_seq_a7_value_l246_246356

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ): Prop := 
  ∀ n : ℕ, a (n+1) = a n + d

theorem arithmetic_seq_a7_value
  (a : ℕ → ℝ) (d : ℝ)
  (h1 : arithmetic_sequence a d)
  (h2 : a 4 = 4)
  (h3 : a 3 + a 8 = 5) :
  a 7 = 1 := 
sorry

end arithmetic_seq_a7_value_l246_246356

import Mathlib

namespace Sarah_substitution_l270_27072

theorem Sarah_substitution :
  ∀ (f g h i j : ℤ), 
    f = 2 → g = 4 → h = 5 → i = 10 →
    (f - (g - (h * (i - j))) = 48 - 5 * j) →
    (f - g - h * i - j = -52 - j) →
    j = 25 :=
by
  intros f g h i j hfg hi hhi hmf hCm hRn
  sorry

end Sarah_substitution_l270_27072


namespace marbles_left_l270_27046

-- Definitions and conditions
def marbles_initial : ℕ := 38
def marbles_lost : ℕ := 15

-- Statement of the problem
theorem marbles_left : marbles_initial - marbles_lost = 23 := by
  sorry

end marbles_left_l270_27046


namespace trig_identity_l270_27021

theorem trig_identity (α : ℝ) (h : Real.sin (Real.pi / 8 + α) = 3 / 4) : 
  Real.cos (3 * Real.pi / 8 - α) = 3 / 4 := 
by 
  sorry

end trig_identity_l270_27021


namespace problem_solution_l270_27033

variable (f : ℝ → ℝ)

noncomputable def solution_set (x : ℝ) : Prop :=
  (0 < x ∧ x < 1/2) ∨ (2 < x)

theorem problem_solution
  (hf_even : ∀ x, f x = f (-x))
  (hf_decreasing : ∀ x y, x < y ∧ y ≤ 0 → f x > f y)
  (hf_at_1 : f 1 = 2) :
  ∀ x, f (Real.log x / Real.log 2) > 2 ↔ solution_set x :=
by
  sorry

end problem_solution_l270_27033


namespace max_largest_integer_l270_27060

theorem max_largest_integer (A B C D E : ℕ) 
  (h1 : A ≤ B) 
  (h2 : B ≤ C) 
  (h3 : C ≤ D) 
  (h4 : D ≤ E)
  (h5 : (A + B + C + D + E) / 5 = 60) 
  (h6 : E - A = 10) : 
  E ≤ 290 :=
sorry

end max_largest_integer_l270_27060


namespace cody_paid_17_l270_27043

-- Definitions for the conditions
def initial_cost : ℝ := 40
def tax_rate : ℝ := 0.05
def discount : ℝ := 8
def final_price_after_discount : ℝ := initial_cost * (1 + tax_rate) - discount
def cody_payment : ℝ := 17

-- The proof statement
theorem cody_paid_17 :
  cody_payment = (final_price_after_discount / 2) :=
by
  -- Proof steps, which we omit by using sorry
  sorry

end cody_paid_17_l270_27043


namespace largest_of_consecutive_non_prime_integers_l270_27020

-- Definition of a prime number
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m:ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of consecutive non-prime sequence condition
def consecutive_non_prime_sequence (start : ℕ) : Prop :=
  ∀ i : ℕ, 0 ≤ i → i < 10 → ¬ is_prime (start + i)

theorem largest_of_consecutive_non_prime_integers :
  (∃ start, start + 9 < 50 ∧ consecutive_non_prime_sequence start) →
  (∃ start, start + 9 = 47) :=
by
  sorry

end largest_of_consecutive_non_prime_integers_l270_27020


namespace problem_1_l270_27099

theorem problem_1 : -9 + 5 - (-12) + (-3) = 5 :=
by {
  -- Proof goes here
  sorry
}

end problem_1_l270_27099


namespace six_digit_number_under_5_lakh_with_digit_sum_43_l270_27002

def is_6_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n < 1000000
def under_500000 (n : ℕ) : Prop := n < 500000
def digit_sum (n : ℕ) : ℕ := (n / 100000) + (n / 10000 % 10) + (n / 1000 % 10) + (n / 100 % 10) + (n / 10 % 10) + (n % 10)

theorem six_digit_number_under_5_lakh_with_digit_sum_43 :
  is_6_digit 499993 ∧ under_500000 499993 ∧ digit_sum 499993 = 43 :=
by 
  sorry

end six_digit_number_under_5_lakh_with_digit_sum_43_l270_27002


namespace term_in_sequence_l270_27078

   theorem term_in_sequence (n : ℕ) (h1 : 1 ≤ n) (h2 : 6 * n + 1 = 2005) : n = 334 :=
   by
     sorry
   
end term_in_sequence_l270_27078


namespace find_principal_sum_l270_27086

theorem find_principal_sum (SI : ℝ) (R : ℝ) (T : ℕ) (P : ℝ) 
  (hSI : SI = 4016.25) (hR : R = 9) (hT : T = 5) : P = 8925 := 
by
  sorry

end find_principal_sum_l270_27086


namespace radius_formula_l270_27071

noncomputable def radius_of_circumscribed_sphere (a : ℝ) : ℝ :=
  let angle := 42 * Real.pi / 180 -- converting 42 degrees to radians
  let R := a / (Real.sqrt 3)
  let h := R * Real.tan angle
  Real.sqrt ((R * R) + (h * h))

theorem radius_formula (a : ℝ) : radius_of_circumscribed_sphere a = (a * Real.sqrt 3) / 3 :=
by
  sorry

end radius_formula_l270_27071


namespace subtract_two_percent_is_multiplying_l270_27079

theorem subtract_two_percent_is_multiplying (a : ℝ) : (a - 0.02 * a) = 0.98 * a := by
  sorry

end subtract_two_percent_is_multiplying_l270_27079


namespace marble_ratio_l270_27073

theorem marble_ratio 
  (K A M : ℕ) 
  (M_has_5_times_as_many_as_K : M = 5 * K)
  (M_has_85_marbles : M = 85)
  (M_has_63_more_than_A : M = A + 63)
  (A_needs_12_more : A + 12 = 34) :
  34 / 17 = 2 := 
by 
  sorry

end marble_ratio_l270_27073


namespace nurses_count_l270_27097

theorem nurses_count (total : ℕ) (ratio_doc : ℕ) (ratio_nurse : ℕ) (nurses : ℕ) : 
  total = 200 → 
  ratio_doc = 4 → 
  ratio_nurse = 6 → 
  nurses = (ratio_nurse * total / (ratio_doc + ratio_nurse)) → 
  nurses = 120 := 
by 
  intros h_total h_ratio_doc h_ratio_nurse h_calc
  rw [h_total, h_ratio_doc, h_ratio_nurse] at h_calc
  simp at h_calc
  exact h_calc

end nurses_count_l270_27097


namespace parabola_y_intercepts_l270_27065

theorem parabola_y_intercepts : 
  (∃ y1 y2 : ℝ, 3 * y1^2 - 4 * y1 + 1 = 0 ∧ 3 * y2^2 - 4 * y2 + 1 = 0 ∧ y1 ≠ y2) :=
by
  sorry

end parabola_y_intercepts_l270_27065


namespace real_solutions_unique_l270_27031

theorem real_solutions_unique (a b c : ℝ) :
  (2 * a - b = a^2 * b ∧ 2 * b - c = b^2 * c ∧ 2 * c - a = c^2 * a) →
  (a, b, c) = (-1, -1, -1) ∨ (a, b, c) = (0, 0, 0) ∨ (a, b, c) = (1, 1, 1) :=
by
  sorry

end real_solutions_unique_l270_27031


namespace intersection_M_N_l270_27062

-- Definitions:
def M := {x : ℝ | 0 ≤ x}
def N := {y : ℝ | -2 ≤ y}

-- The theorem statement:
theorem intersection_M_N : M ∩ N = {z : ℝ | 0 ≤ z} := sorry

end intersection_M_N_l270_27062


namespace percent_difference_l270_27044

variable (w e y z : ℝ)

-- Definitions based on the given conditions
def condition1 : Prop := w = 0.60 * e
def condition2 : Prop := e = 0.60 * y
def condition3 : Prop := z = 0.54 * y

-- Statement of the theorem to prove
theorem percent_difference (h1 : condition1 w e) (h2 : condition2 e y) (h3 : condition3 z y) : 
  (z - w) / w * 100 = 50 := 
by
  sorry

end percent_difference_l270_27044


namespace sin_alpha_second_quadrant_l270_27040

/-- Given angle α in the second quadrant such that tan(π - α) = 3/4, we need to prove that sin α = 3/5. -/
theorem sin_alpha_second_quadrant (α : ℝ) (hα1 : π / 2 < α ∧ α < π) (hα2 : Real.tan (π - α) = 3 / 4) : Real.sin α = 3 / 5 := by
  sorry

end sin_alpha_second_quadrant_l270_27040


namespace evaluate_expression_l270_27005

theorem evaluate_expression :
  (3^5 + 3^5 + 3^5 + 3^5 + 3^5 + 3^5 + 3^5 + 3^5 + 3^7) = (6^5 + 3^7) :=
sorry

end evaluate_expression_l270_27005


namespace part_one_part_two_l270_27001

def f (x a : ℝ) : ℝ := |x - a| + |x + 2|

-- Question (1)
theorem part_one (x : ℝ) (a : ℝ) (h : a = 1) : f x a ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2 := 
sorry

-- Question (2)
theorem part_two (a : ℝ) (h : a ≥ 1) : ∀ (y : ℝ), (∃ x : ℝ, f x a = y) ↔ (∃ b : ℝ, y = b + 2 ∧ b ≥ a) := 
sorry

end part_one_part_two_l270_27001


namespace compute_sin_product_l270_27095

theorem compute_sin_product : 
  (1 - Real.sin (Real.pi / 12)) *
  (1 - Real.sin (5 * Real.pi / 12)) *
  (1 - Real.sin (7 * Real.pi / 12)) *
  (1 - Real.sin (11 * Real.pi / 12)) = 
  (1 / 16) :=
by
  sorry

end compute_sin_product_l270_27095


namespace find_ordered_pair_l270_27058

theorem find_ordered_pair (x y : ℝ) (h : (x - 2 * y)^2 + (y - 1)^2 = 0) : x = 2 ∧ y = 1 :=
by
  sorry

end find_ordered_pair_l270_27058


namespace remainder_of_2345678_div_5_l270_27029

theorem remainder_of_2345678_div_5 : (2345678 % 5) = 3 :=
by
  sorry

end remainder_of_2345678_div_5_l270_27029


namespace calculation_result_l270_27061

theorem calculation_result : 2014 * (1/19 - 1/53) = 68 := by
  sorry

end calculation_result_l270_27061


namespace general_term_formula_l270_27092

theorem general_term_formula (a : ℕ → ℝ) (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, 0 < n → (n+1) * a (n+1) - n * a n^2 + (n+1) * a n * a (n+1) - n * a n = 0) :
  ∀ n : ℕ, 0 < n → a n = 1 / n :=
by
  sorry

end general_term_formula_l270_27092


namespace no_valid_prime_l270_27022

open Nat

def base_p_polynomial (p : ℕ) (coeffs : List ℕ) : ℕ → ℕ :=
  fun (n : ℕ) => coeffs.foldl (λ sum coef => sum * p + coef) 0

def num_1013 (p : ℕ) := base_p_polynomial p [1, 0, 1, 3]
def num_207 (p : ℕ) := base_p_polynomial p [2, 0, 7]
def num_214 (p : ℕ) := base_p_polynomial p [2, 1, 4]
def num_100 (p : ℕ) := base_p_polynomial p [1, 0, 0]
def num_10 (p : ℕ) := base_p_polynomial p [1, 0]

def num_321 (p : ℕ) := base_p_polynomial p [3, 2, 1]
def num_403 (p : ℕ) := base_p_polynomial p [4, 0, 3]
def num_210 (p : ℕ) := base_p_polynomial p [2, 1, 0]

theorem no_valid_prime (p : ℕ) [Fact (Nat.Prime p)] :
  num_1013 p + num_207 p + num_214 p + num_100 p + num_10 p ≠
  num_321 p + num_403 p + num_210 p := by
  sorry

end no_valid_prime_l270_27022


namespace intersecting_lines_l270_27045

def diamondsuit (a b : ℝ) : ℝ := a^2 + a * b - b^2

theorem intersecting_lines (x y : ℝ) : 
  (diamondsuit x y = diamondsuit y x) ↔ (y = x ∨ y = -x) := by
  sorry

end intersecting_lines_l270_27045


namespace electricity_price_per_kWh_l270_27054

theorem electricity_price_per_kWh (consumption_rate : ℝ) (hours_used : ℝ) (total_cost : ℝ) :
  consumption_rate = 2.4 → hours_used = 25 → total_cost = 6 →
  total_cost / (consumption_rate * hours_used) = 0.10 :=
by
  intros hc hh ht
  have h_energy : consumption_rate * hours_used = 60 :=
    by rw [hc, hh]; norm_num
  rw [ht, h_energy]
  norm_num

end electricity_price_per_kWh_l270_27054


namespace treasure_in_heaviest_bag_l270_27055

theorem treasure_in_heaviest_bag (A B C D : ℝ) (h1 : A + B < C)
                                        (h2 : A + C = D)
                                        (h3 : A + D > B + C) : D > A ∧ D > B ∧ D > C :=
by 
  sorry

end treasure_in_heaviest_bag_l270_27055


namespace complement_of_M_l270_27026

open Set

def U : Set ℝ := univ
def M : Set ℝ := { x | x^2 - 2 * x > 0 }
def comp_M_Real := compl M

theorem complement_of_M :
  comp_M_Real = { x : ℝ | 0 ≤ x ∧ x ≤ 2 } :=
sorry

end complement_of_M_l270_27026


namespace trapezoid_area_pqrs_l270_27039

theorem trapezoid_area_pqrs :
  let P := (1, 1)
  let Q := (1, 4)
  let R := (6, 4)
  let S := (7, 1)
  let parallelogram := true -- indicates that PQ and RS are parallel
  let PQ := abs (Q.2 - P.2)
  let RS := abs (S.1 - R.1)
  let height := abs (R.1 - P.1)
  (1 / 2 : ℚ) * (PQ + RS) * height = 10 := by
  sorry

end trapezoid_area_pqrs_l270_27039


namespace triangle_ABC_is_isosceles_roots_of_quadratic_for_equilateral_l270_27011

variable {a b c x : ℝ}

-- Part (1)
theorem triangle_ABC_is_isosceles (h : (a + b) * 1 ^ 2 - 2 * c * 1 + (a - b) = 0) : a = c :=
by 
  -- Proof omitted
  sorry

-- Part (2)
theorem roots_of_quadratic_for_equilateral (h_eq : a = b ∧ b = c ∧ c = a) : 
  (∀ x : ℝ, (a + a) * x ^ 2 - 2 * a * x + (a - a) = 0 → (x = 0 ∨ x = 1)) :=
by 
  -- Proof omitted
  sorry

end triangle_ABC_is_isosceles_roots_of_quadratic_for_equilateral_l270_27011


namespace range_of_function_l270_27010

theorem range_of_function (x : ℝ) : x ≠ 2 ↔ ∃ y, y = x / (x - 2) :=
sorry

end range_of_function_l270_27010


namespace largest_integer_n_apples_l270_27007

theorem largest_integer_n_apples (t : ℕ) (a : ℕ → ℕ) (h1 : t = 150) 
    (h2 : ∀ i : ℕ, 100 ≤ a i ∧ a i ≤ 130) :
  ∃ n : ℕ, n = 5 ∧ (∀ i j : ℕ, a i = a j → i = j → 5 ≤ i ∧ 5 ≤ j) :=
by
  sorry

end largest_integer_n_apples_l270_27007


namespace n_power_four_plus_sixtyfour_power_n_composite_l270_27066

theorem n_power_four_plus_sixtyfour_power_n_composite (n : ℕ) : ∃ m k, m * k = n^4 + 64^n ∧ m > 1 ∧ k > 1 :=
by
  sorry

end n_power_four_plus_sixtyfour_power_n_composite_l270_27066


namespace chicken_farm_l270_27038

def total_chickens (roosters hens : ℕ) : ℕ := roosters + hens

theorem chicken_farm (roosters hens : ℕ) (h1 : 2 * hens = roosters) (h2 : roosters = 6000) : 
  total_chickens roosters hens = 9000 :=
by
  sorry

end chicken_farm_l270_27038


namespace num_counting_numbers_dividing_52_leaving_remainder_7_l270_27048

def divides (a b : ℕ) : Prop := ∃ k, b = k * a

theorem num_counting_numbers_dividing_52_leaving_remainder_7 (n : ℕ) :
  (∃ n : ℕ, 59 ≡ 7 [MOD n]) → (n > 7 ∧ divides n 52) → n = 3 := 
sorry

end num_counting_numbers_dividing_52_leaving_remainder_7_l270_27048


namespace final_price_chocolate_l270_27052

-- Conditions
def original_cost : ℝ := 2.00
def discount : ℝ := 0.57

-- Question and answer
theorem final_price_chocolate : original_cost - discount = 1.43 :=
by
  sorry

end final_price_chocolate_l270_27052


namespace third_grade_contribution_fourth_grade_contribution_l270_27047

def first_grade := 20
def second_grade := 45
def third_grade := first_grade + second_grade - 17
def fourth_grade := 2 * third_grade - 36

theorem third_grade_contribution : third_grade = 48 := by
  sorry

theorem fourth_grade_contribution : fourth_grade = 60 := by
  sorry

end third_grade_contribution_fourth_grade_contribution_l270_27047


namespace problem_statement_l270_27082

theorem problem_statement (m n : ℤ) (h : |m - 2| + (n + 1)^2 = 0) : m + n = 1 :=
by sorry

end problem_statement_l270_27082


namespace coordinates_of_point_l270_27081

theorem coordinates_of_point (x y : ℝ) (h : (x, y) = (-2, 3)) : (x, y) = (-2, 3) :=
by
  exact h

end coordinates_of_point_l270_27081


namespace eight_pow_15_div_sixtyfour_pow_6_l270_27085

theorem eight_pow_15_div_sixtyfour_pow_6 :
  8^15 / 64^6 = 512 := by
  sorry

end eight_pow_15_div_sixtyfour_pow_6_l270_27085


namespace find_linear_function_passing_A_B_l270_27019

-- Conditions
def line_function (k b x : ℝ) : ℝ := k * x + b

theorem find_linear_function_passing_A_B :
  (∃ k b : ℝ, k ≠ 0 ∧ line_function k b 1 = 3 ∧ line_function k b 0 = -2) → 
  ∃ k b : ℝ, k = 5 ∧ b = -2 ∧ ∀ x : ℝ, line_function k b x = 5 * x - 2 :=
by
  -- Proof will be added here
  sorry

end find_linear_function_passing_A_B_l270_27019


namespace simplify_expression_l270_27076

theorem simplify_expression : 
  ((Real.sqrt 5 + Real.sqrt 2) * (Real.sqrt 5 - Real.sqrt 2)) - 
  (Real.sqrt 3 * (Real.sqrt 3 + Real.sqrt (2 / 3))) = -Real.sqrt 2 :=
by
  sorry

end simplify_expression_l270_27076


namespace cards_difference_l270_27096

theorem cards_difference
  (H : ℕ)
  (F : ℕ)
  (B : ℕ)
  (hH : H = 200)
  (hF : F = 4 * H)
  (hTotal : B + F + H = 1750) :
  F - B = 50 :=
by
  sorry

end cards_difference_l270_27096


namespace find_angle_A_find_side_a_l270_27063

-- Define the triangle ABC with sides a, b, c opposite to angles A, B, C respectively
variables {a b c A B C : ℝ}
-- Assumption conditions in the problem
variables (h₁ : a * sin B = sqrt 3 * b * cos A)
variables (hb : b = 3)
variables (hc : c = 2)

-- Prove that A = π / 3 given the first condition
theorem find_angle_A : h₁ → A = π / 3 := by
  -- Proof is omitted
  sorry

-- Prove that a = sqrt 7 given b = 3, c = 2, and A = π / 3
theorem find_side_a : h₁ → hb → hc → a = sqrt 7 := by
  -- Proof is omitted
  sorry

end find_angle_A_find_side_a_l270_27063


namespace calc_sub_neg_eq_add_problem_0_sub_neg_3_l270_27012

theorem calc_sub_neg_eq_add (a b : Int) : a - (-b) = a + b := by
  sorry

theorem problem_0_sub_neg_3 : 0 - (-3) = 3 := by
  exact calc_sub_neg_eq_add 0 3

end calc_sub_neg_eq_add_problem_0_sub_neg_3_l270_27012


namespace baker_cakes_total_l270_27037

def initial_cakes : ℕ := 110
def cakes_sold : ℕ := 75
def additional_cakes : ℕ := 76

theorem baker_cakes_total : 
  (initial_cakes - cakes_sold) + additional_cakes = 111 := by
  sorry

end baker_cakes_total_l270_27037


namespace displacement_correct_l270_27023

-- Define the initial conditions of the problem
def init_north := 50
def init_east := 70
def init_south := 20
def init_west := 30

-- Define the net movements
def net_north := init_north - init_south
def net_east := init_east - init_west

-- Define the straight-line distance using the Pythagorean theorem
def displacement_AC := (net_north ^ 2 + net_east ^ 2).sqrt

theorem displacement_correct : displacement_AC = 50 := 
by sorry

end displacement_correct_l270_27023


namespace inv_mod_997_l270_27056

theorem inv_mod_997 : ∃ x : ℤ, 0 ≤ x ∧ x < 997 ∧ (10 * x) % 997 = 1 := 
sorry

end inv_mod_997_l270_27056


namespace Debby_spent_on_yoyo_l270_27098

theorem Debby_spent_on_yoyo 
  (hat_tickets stuffed_animal_tickets total_tickets : ℕ) 
  (h1 : hat_tickets = 2) 
  (h2 : stuffed_animal_tickets = 10) 
  (h3 : total_tickets = 14) 
  : ∃ yoyo_tickets : ℕ, hat_tickets + stuffed_animal_tickets + yoyo_tickets = total_tickets ∧ yoyo_tickets = 2 := 
by 
  sorry

end Debby_spent_on_yoyo_l270_27098


namespace find_a_l270_27036

theorem find_a (f : ℝ → ℝ) (a x : ℝ) 
  (h1 : ∀ x, f (1/2 * x - 1) = 2 * x - 5) 
  (h2 : f a = 6) : a = 7 / 4 := 
by
  sorry

end find_a_l270_27036


namespace equation_of_tangent_circle_l270_27077

/-- Lean Statement for the circle problem -/
theorem equation_of_tangent_circle (center_C : ℝ × ℝ)
    (h1 : ∃ x, center_C = (x, 0) ∧ x - 0 + 1 = 0)
    (circle_tangent : ∃ r, ((2 - (center_C.1))^2 + (3 - (center_C.2))^2 = (2 * Real.sqrt 2) + r)) :
    ∃ r, (x + 1)^2 + y^2 = r^2 := 
sorry

end equation_of_tangent_circle_l270_27077


namespace rain_probability_at_most_3_days_l270_27090

open BigOperators

def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def rain_probability := (1:ℝ)/5
noncomputable def no_rain_probability := (4:ℝ)/5

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binom n k) * (p^k) * ((1-p)^(n-k))

theorem rain_probability_at_most_3_days :
  ∑ k in Finset.range 4, binomial_probability 31 k rain_probability = 0.544 :=
by
  sorry

end rain_probability_at_most_3_days_l270_27090


namespace task_pages_l270_27094

theorem task_pages (A B T : ℕ) (hB : B = A + 5) (hTogether : (A + B) * 18 = T)
  (hAlone : A * 60 = T) : T = 225 :=
by
  sorry

end task_pages_l270_27094


namespace circle_iff_m_gt_neg_1_over_2_l270_27075

noncomputable def represents_circle (m: ℝ) : Prop :=
  ∀ (x y : ℝ), (x^2 + y^2 + x + y - m = 0) → m > -1/2

theorem circle_iff_m_gt_neg_1_over_2 (m : ℝ) : represents_circle m ↔ m > -1/2 := by
  sorry

end circle_iff_m_gt_neg_1_over_2_l270_27075


namespace kolya_made_mistake_l270_27069

theorem kolya_made_mistake (ab cd effe : ℕ)
  (h_eq : ab * cd = effe)
  (h_eff_div_11 : effe % 11 = 0)
  (h_ab_cd_not_div_11 : ab % 11 ≠ 0 ∧ cd % 11 ≠ 0) :
  false :=
by
  -- Note: This is where the proof would go, but we are illustrating the statement only.
  sorry

end kolya_made_mistake_l270_27069


namespace solve_inequalities_l270_27087

theorem solve_inequalities :
  {x : ℝ // 3 * x - 2 < (x + 2) ^ 2 ∧ (x + 2) ^ 2 < 9 * x - 8} = {x : ℝ // 3 < x ∧ x < 4} :=
sorry

end solve_inequalities_l270_27087


namespace min_value_x_3y_l270_27025

noncomputable def min_value (x y : ℝ) : ℝ := x + 3 * y

theorem min_value_x_3y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) :
  ∃ (x y : ℝ), min_value x y = 18 + 21 * Real.sqrt 3 :=
sorry

end min_value_x_3y_l270_27025


namespace divides_2_pow_26k_plus_2_plus_3_by_19_l270_27030

theorem divides_2_pow_26k_plus_2_plus_3_by_19 (k : ℕ) : 19 ∣ (2^(26*k+2) + 3) := 
by
  sorry

end divides_2_pow_26k_plus_2_plus_3_by_19_l270_27030


namespace problem_statement_l270_27016

noncomputable def middle_of_three_consecutive (x : ℕ) : ℕ :=
  let y := x + 1
  let z := x + 2
  y

theorem problem_statement :
  ∃ x : ℕ, 
    (x + (x + 1) = 18) ∧ 
    (x + (x + 2) = 20) ∧ 
    ((x + 1) + (x + 2) = 23) ∧ 
    (middle_of_three_consecutive x = 7) :=
by
  sorry

end problem_statement_l270_27016


namespace total_lives_l270_27053

def initial_players := 25
def additional_players := 10
def lives_per_player := 15

theorem total_lives :
  (initial_players + additional_players) * lives_per_player = 525 := by
  sorry

end total_lives_l270_27053


namespace total_animals_l270_27070

theorem total_animals (H C2 C1 : ℕ) (humps_eq : 2 * C2 + C1 = 200) (horses_eq : H = C2) :
  H + C2 + C1 = 200 :=
by
  /- Proof steps are not required -/
  sorry

end total_animals_l270_27070


namespace domain_of_f_l270_27051

theorem domain_of_f :
  {x : ℝ | x > -1 ∧ x ≠ 0 ∧ x ≤ 2} = {x : ℝ | (-1 < x ∧ x < 0) ∨ (0 < x ∧ x ≤ 2)} :=
by
  sorry

end domain_of_f_l270_27051


namespace percent_more_than_l270_27057

-- Definitions and conditions
variables (x y p : ℝ)

-- Condition: x is p percent more than y
def x_is_p_percent_more_than_y (x y p : ℝ) : Prop :=
  x = y + (p / 100) * y

-- The theorem to prove
theorem percent_more_than (h : x_is_p_percent_more_than_y x y p) :
  p = 100 * (x / y - 1) :=
sorry

end percent_more_than_l270_27057


namespace inequality_proof_l270_27041

def f (x : ℝ) (m : ℕ) : ℝ := |x - m| + |x|

theorem inequality_proof (α β : ℝ) (m : ℕ) (h1 : 1 < α) (h2 : 1 < β) (h3 : m = 1) 
  (h4 : f α m + f β m = 2) : (4 / α) + (1 / β) ≥ 9 / 2 := by
  sorry

end inequality_proof_l270_27041


namespace loan_payment_difference_l270_27089

noncomputable def compounded_amount (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def simple_interest_amount (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P + P * r * t

noncomputable def loan1_payment (P : ℝ) (r : ℝ) (n : ℝ) (t1 : ℝ) (t2 : ℝ) : ℝ :=
  let A1 := compounded_amount P r n t1
  let one_third_payment := A1 / 3
  let remaining := A1 - one_third_payment
  one_third_payment + compounded_amount remaining r n t2

noncomputable def loan2_payment (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  simple_interest_amount P r t

noncomputable def positive_difference (x y : ℝ) : ℝ :=
  if x > y then x - y else y - x

theorem loan_payment_difference: 
  ∀ P : ℝ, ∀ r1 r2 : ℝ, ∀ n : ℝ, ∀ t1 t2 : ℝ,
  P = 12000 → r1 = 0.08 → r2 = 0.09 → n = 12 → t1 = 7 → t2 = 8 →
  positive_difference 
    (loan2_payment P r2 (t1 + t2)) 
    (loan1_payment P r1 n t1 t2) = 2335 := 
by
  intros
  sorry

end loan_payment_difference_l270_27089


namespace range_of_a_l270_27014

def A : Set ℝ := { x | x^2 - x - 2 > 0 }
def B (a : ℝ) : Set ℝ := { x | abs (x - a) < 3 }

theorem range_of_a (a : ℝ) :
  (A ∪ B a = Set.univ) → a ∈ Set.Ioo (-1 : ℝ) 2 :=
by
  sorry

end range_of_a_l270_27014


namespace greatest_sum_of_visible_numbers_l270_27059

/-- Definition of a cube with numbered faces -/
structure Cube where
  face1 : ℕ
  face2 : ℕ
  face3 : ℕ
  face4 : ℕ
  face5 : ℕ
  face6 : ℕ

/-- The cubes face numbers -/
def cube_numbers : List ℕ := [1, 2, 4, 8, 16, 32]

/-- Stacked cubes with maximized visible numbers sum -/
def maximize_visible_sum :=
  let cube1 := Cube.mk 1 2 4 8 16 32
  let cube2 := Cube.mk 1 2 4 8 16 32
  let cube3 := Cube.mk 1 2 4 8 16 32
  let cube4 := Cube.mk 1 2 4 8 16 32
  244

theorem greatest_sum_of_visible_numbers : maximize_visible_sum = 244 := 
  by
    sorry -- Proof to be done

end greatest_sum_of_visible_numbers_l270_27059


namespace direct_proportion_graph_is_straight_line_l270_27018

-- Defining the direct proportion function
def direct_proportion_function (k x : ℝ) : ℝ := k * x

-- Theorem statement
theorem direct_proportion_graph_is_straight_line (k : ℝ) :
  ∀ x : ℝ, ∃ y : ℝ, y = direct_proportion_function k x ∧ 
    ∀ (x1 x2 : ℝ), 
    ∃ a b : ℝ, b ≠ 0 ∧ 
    (a * x1 + b * (direct_proportion_function k x1)) = (a * x2 + b * (direct_proportion_function k x2)) :=
by
  sorry

end direct_proportion_graph_is_straight_line_l270_27018


namespace inequality_equality_condition_l270_27068

theorem inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + b + c + 2 = a * b * c) :
  (a + 1) * (b + 1) * (c + 1) ≥ 27 :=
sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + b + c + 2 = a * b * c) :
  (a + 1) * (b + 1) * (c + 1) = 27 ↔ a = 2 ∧ b = 2 ∧ c = 2 :=
sorry

end inequality_equality_condition_l270_27068


namespace mark_wait_time_l270_27035

theorem mark_wait_time (t1 t2 T : ℕ) (h1 : t1 = 4) (h2 : t2 = 20) (hT : T = 38) : 
  T - (t1 + t2) = 14 :=
by sorry

end mark_wait_time_l270_27035


namespace total_runs_opponents_correct_l270_27088

-- Define the scoring conditions
def team_scores : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
def lost_games_scores : List ℕ := [3, 5, 7, 9, 11, 13]
def won_games_scores : List ℕ := [2, 4, 6, 8, 10, 12]

-- Define the total runs scored by opponents in lost games
def total_runs_lost_games : ℕ := (lost_games_scores.map (λ x => x + 1)).sum

-- Define the total runs scored by opponents in won games
def total_runs_won_games : ℕ := (won_games_scores.map (λ x => x / 2)).sum

-- Total runs scored by opponents (given)
def total_runs_opponents : ℕ := total_runs_lost_games + total_runs_won_games

-- The theorem to prove
theorem total_runs_opponents_correct : total_runs_opponents = 75 := by
  -- Proof goes here
  sorry

end total_runs_opponents_correct_l270_27088


namespace number_of_squares_l270_27006

-- Define the conditions and the goal
theorem number_of_squares {x : ℤ} (hx0 : 0 ≤ x) (hx6 : x ≤ 6) {y : ℤ} (hy0 : -1 ≤ y) (hy : y ≤ 3 * x) :
  ∃ (n : ℕ), n = 123 :=
by 
  sorry

end number_of_squares_l270_27006


namespace even_function_value_at_three_l270_27013

variable (f : ℝ → ℝ)
variable (x : ℝ)

-- f is an even function
axiom h_even : ∀ x, f x = f (-x)

-- f(x) is defined as x^2 + x when x < 0
axiom h_neg_def : ∀ x, x < 0 → f x = x^2 + x

theorem even_function_value_at_three : f 3 = 6 :=
by {
  -- To be proved
  sorry
}

end even_function_value_at_three_l270_27013


namespace greatest_integer_gcd_18_is_6_l270_27004

theorem greatest_integer_gcd_18_is_6 (n : ℕ) (h1 : n < 150) (h2 : Nat.gcd n 18 = 6) : n = 138 := 
sorry

end greatest_integer_gcd_18_is_6_l270_27004


namespace total_applicants_is_40_l270_27009

def total_applicants (PS GPA_high Not_PS_GPA_low both : ℕ) : ℕ :=
  let PS_or_GPA_high := PS + GPA_high - both 
  PS_or_GPA_high + Not_PS_GPA_low

theorem total_applicants_is_40 :
  total_applicants 15 20 10 5 = 40 :=
by
  sorry

end total_applicants_is_40_l270_27009


namespace minimum_vertical_distance_l270_27042

noncomputable def absolute_value (x : ℝ) : ℝ := abs x

noncomputable def quadratic_function (x : ℝ) : ℝ := -x^2 - 3 * x - 5

theorem minimum_vertical_distance :
  ∃ x : ℝ, (∀ y : ℝ, |absolute_value y - quadratic_function y| ≥ 4) ∧ (|absolute_value x - quadratic_function x| = 4) := 
sorry

end minimum_vertical_distance_l270_27042


namespace find_value_l270_27003

theorem find_value (a b : ℝ) (h : a + b + 1 = -2) : (a + b - 1) * (1 - a - b) = -16 := by
  sorry

end find_value_l270_27003


namespace find_x_satisfying_inequality_l270_27074

open Real

theorem find_x_satisfying_inequality :
  ∀ x : ℝ, 0 < x → (x * sqrt (16 - x) + sqrt (16 * x - x^3) ≥ 16 ↔ x = 4) :=
by
  sorry

end find_x_satisfying_inequality_l270_27074


namespace Lauryn_employees_l270_27050

variables (M W : ℕ)

theorem Lauryn_employees (h1 : M = W - 20) (h2 : M + W = 180) : M = 80 :=
by {
    sorry
}

end Lauryn_employees_l270_27050


namespace smallestBeta_satisfies_l270_27008

noncomputable def validAlphaBeta (alpha beta : ℕ) : Prop :=
  16 / 37 < (alpha : ℚ) / beta ∧ (alpha : ℚ) / beta < 7 / 16

def smallestBeta : ℕ := 23

theorem smallestBeta_satisfies :
  (∀ (alpha beta : ℕ), validAlphaBeta alpha beta → beta ≥ 23) ∧
  (∃ (alpha : ℕ), validAlphaBeta alpha 23) :=
by sorry

end smallestBeta_satisfies_l270_27008


namespace find_m_l270_27091

def vector (α : Type*) := α × α

def a : vector ℤ := (1, -2)
def b : vector ℤ := (3, 0)

def two_a_plus_b (a b : vector ℤ) : vector ℤ := (2 * a.1 + b.1, 2 * a.2 + b.2)
def m_a_minus_b (m : ℤ) (a b : vector ℤ) : vector ℤ := (m * a.1 - b.1, m * a.2 - b.2)

def parallel (v w : vector ℤ) : Prop := v.1 * w.2 = v.2 * w.1

theorem find_m : parallel (two_a_plus_b a b) (m_a_minus_b (-2) a b) :=
by
  sorry -- proof placeholder

end find_m_l270_27091


namespace calc_result_l270_27080

theorem calc_result : 75 * 1313 - 25 * 1313 = 65750 := 
by 
  sorry

end calc_result_l270_27080


namespace houses_with_two_car_garage_l270_27064

theorem houses_with_two_car_garage
  (T P GP N G : ℕ)
  (hT : T = 90)
  (hP : P = 40)
  (hGP : GP = 35)
  (hN : N = 35)
  (hFormula : G + P - GP = T - N) :
  G = 50 :=
by
  rw [hT, hP, hGP, hN] at hFormula
  simp at hFormula
  exact hFormula

end houses_with_two_car_garage_l270_27064


namespace log_eqn_l270_27083

theorem log_eqn (a b : ℝ) (h1 : a = (Real.log 400 / Real.log 16))
                          (h2 : b = Real.log 20 / Real.log 2) : a = (1/2) * b :=
sorry

end log_eqn_l270_27083


namespace largest_angle_of_convex_hexagon_l270_27049

theorem largest_angle_of_convex_hexagon 
  (x : ℝ) 
  (hx : (x + 2) + (2 * x - 1) + (3 * x + 1) + (4 * x - 2) + (5 * x + 3) + (6 * x - 4) = 720) :
  6 * x - 4 = 202 :=
sorry

end largest_angle_of_convex_hexagon_l270_27049


namespace solve_w_from_system_of_equations_l270_27084

open Real

variables (w x y z : ℝ)

theorem solve_w_from_system_of_equations
  (h1 : 2 * w + x + y + z = 1)
  (h2 : w + 2 * x + y + z = 2)
  (h3 : w + x + 2 * y + z = 2)
  (h4 : w + x + y + 2 * z = 1) :
  w = -1 / 5 :=
by
  sorry

end solve_w_from_system_of_equations_l270_27084


namespace functional_equation_implies_identity_l270_27093

theorem functional_equation_implies_identity 
  (f : ℝ → ℝ) 
  (hf : ∀ x y : ℝ, 0 < x → 0 < y → 
    f ((x + y) / 2) + f ((2 * x * y) / (x + y)) = f x + f y) 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  : 2 * f (Real.sqrt (x * y)) = f x + f y := sorry

end functional_equation_implies_identity_l270_27093


namespace sum_of_coordinates_l270_27017

theorem sum_of_coordinates :
  let in_distance_from_line (p : (ℝ × ℝ)) (d : ℝ) (line_y : ℝ) : Prop := abs (p.2 - line_y) = d
  let in_distance_from_point (p1 p2 : (ℝ × ℝ)) (d : ℝ) : Prop := (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 = d^2
  ∃ (P1 P2 P3 P4 : ℝ × ℝ),
  in_distance_from_line P1 4 13 ∧ in_distance_from_point P1 (7, 13) 10 ∧
  in_distance_from_line P2 4 13 ∧ in_distance_from_point P2 (7, 13) 10 ∧
  in_distance_from_line P3 4 13 ∧ in_distance_from_point P3 (7, 13) 10 ∧
  in_distance_from_line P4 4 13 ∧ in_distance_from_point P4 (7, 13) 10 ∧
  (P1.1 + P2.1 + P3.1 + P4.1) + (P1.2 + P2.2 + P3.2 + P4.2) = 80 :=
sorry

end sum_of_coordinates_l270_27017


namespace linear_equation_in_two_vars_example_l270_27032

def is_linear_equation_in_two_vars (eq : String) : Prop :=
  eq = "x + 4y = 6"

theorem linear_equation_in_two_vars_example :
  is_linear_equation_in_two_vars "x + 4y = 6" :=
by
  sorry

end linear_equation_in_two_vars_example_l270_27032


namespace sum_of_roots_eq_p_l270_27067

variable (p q : ℝ)
variable (hq : q = p^2 - 1)

theorem sum_of_roots_eq_p (h : q = p^2 - 1) : 
  let r1 := p
  let r2 := q
  r1 + r2 = p := 
sorry

end sum_of_roots_eq_p_l270_27067


namespace part1_part2_l270_27024

-- Define the main condition of the farthest distance formula
def distance_formula (S h : ℝ) : Prop := S^2 = 1.7 * h

-- Define part 1: Given h = 1.7, prove S = 1.7
theorem part1
  (h : ℝ)
  (hyp : h = 1.7)
  : ∃ S : ℝ, distance_formula S h ∧ S = 1.7 :=
by
  sorry
  
-- Define part 2: Given S = 6.8 and height of eyes to ground 1.5, prove the height of tower = 25.7
theorem part2
  (S : ℝ)
  (h1 : ℝ)
  (height_eyes_to_ground : ℝ)
  (hypS : S = 6.8)
  (height_eyes_to_ground_eq : height_eyes_to_ground = 1.5)
  : ∃ h : ℝ, distance_formula S h ∧ (h - height_eyes_to_ground) = 25.7 :=
by
  sorry

end part1_part2_l270_27024


namespace decimal_equivalent_one_quarter_power_one_l270_27000

theorem decimal_equivalent_one_quarter_power_one : (1 / 4 : ℝ) ^ 1 = 0.25 := by
  sorry

end decimal_equivalent_one_quarter_power_one_l270_27000


namespace only_odd_digit_square_l270_27027

def odd_digits (n : ℕ) : Prop :=
  ∀ d ∈ (n.digits 10), d % 2 = 1

theorem only_odd_digit_square (n : ℕ) : n^2 = n → odd_digits n → n = 1 ∨ n = 9 :=
by
  intros
  sorry

end only_odd_digit_square_l270_27027


namespace stamp_collection_l270_27015

theorem stamp_collection (x : ℕ) :
  (5 * x + 3 * (x + 20) = 300) → (x = 30) ∧ (x + 20 = 50) :=
by
  sorry

end stamp_collection_l270_27015


namespace find_line_l270_27028

def point_on_line (P : ℝ × ℝ) (m b : ℝ) : Prop :=
  P.2 = m * P.1 + b

def intersection_points_distance (k m b : ℝ) : Prop :=
  |(k^2 - 4*k + 4) - (m*k + b)| = 6

noncomputable def desired_line (m b : ℝ) : Prop :=
  point_on_line (2, 3) m b ∧ ∀ (k : ℝ), intersection_points_distance k m b

theorem find_line : desired_line (-6) 15 := sorry

end find_line_l270_27028


namespace find_200_digit_number_l270_27034

noncomputable def original_number_condition (N : ℕ) (c : ℕ) (k : ℕ) : Prop :=
  let m := 0
  let a := 2 * c
  let b := 3 * c
  k = 197 ∧ (c = 1 ∨ c = 2 ∨ c = 3) ∧ N = 132 * c * 10^197

theorem find_200_digit_number :
  ∃ N c, original_number_condition N c 197 :=
by
  sorry

end find_200_digit_number_l270_27034

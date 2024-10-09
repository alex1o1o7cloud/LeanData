import Mathlib

namespace problem1_solution_problem2_solution_l2096_209656

-- Problem 1
theorem problem1_solution (x y : ℝ) (h1 : 2 * x + 3 * y = 8) (h2 : x = y - 1) : x = 1 ∧ y = 2 := by
  sorry

-- Problem 2
theorem problem2_solution (x y : ℝ) (h1 : 2 * x - y = -1) (h2 : x + 3 * y = 17) : x = 2 ∧ y = 5 := by
  sorry

end problem1_solution_problem2_solution_l2096_209656


namespace go_total_pieces_l2096_209617

theorem go_total_pieces (T : ℕ) (h : T > 0) (prob_black : T = (3 : ℕ) * 4) : T = 12 := by
  sorry

end go_total_pieces_l2096_209617


namespace james_huskies_count_l2096_209668

theorem james_huskies_count 
  (H : ℕ) 
  (pitbulls : ℕ := 2) 
  (golden_retrievers : ℕ := 4) 
  (husky_pups_per_husky : ℕ := 3) 
  (pitbull_pups_per_pitbull : ℕ := 3) 
  (extra_pups_per_golden_retriever : ℕ := 2) 
  (pup_difference : ℕ := 30) :
  H + pitbulls + golden_retrievers + pup_difference = 3 * H + pitbulls * pitbull_pups_per_pitbull + golden_retrievers * (husky_pups_per_husky + extra_pups_per_golden_retriever) :=
sorry

end james_huskies_count_l2096_209668


namespace lucy_lovely_age_ratio_l2096_209676

theorem lucy_lovely_age_ratio (L l : ℕ) (x : ℕ) (h1 : L = 50) (h2 : 45 = x * (l - 5)) (h3 : 60 = 2 * (l + 10)) :
  (45 / (l - 5)) = 3 :=
by
  sorry

end lucy_lovely_age_ratio_l2096_209676


namespace number_of_people_l2096_209643

def avg_weight_increase : ℝ := 2.5
def old_person_weight : ℝ := 45
def new_person_weight : ℝ := 65

theorem number_of_people (n : ℕ) 
  (h1 : avg_weight_increase = 2.5) 
  (h2 : old_person_weight = 45) 
  (h3 : new_person_weight = 65) :
  n = 8 :=
  sorry

end number_of_people_l2096_209643


namespace tan_diff_identity_l2096_209696

theorem tan_diff_identity (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β + π / 4) = 1 / 4) :
  Real.tan (α - π / 4) = 3 / 22 :=
sorry

end tan_diff_identity_l2096_209696


namespace yu_chan_walked_distance_l2096_209600

def step_length : ℝ := 0.75
def walking_time : ℝ := 13
def steps_per_minute : ℝ := 70

theorem yu_chan_walked_distance : step_length * steps_per_minute * walking_time = 682.5 :=
by
  sorry

end yu_chan_walked_distance_l2096_209600


namespace board_train_immediately_probability_l2096_209611

-- Define conditions
def total_time : ℝ := 10
def favorable_time : ℝ := 1

-- Define the probability P(A) as favorable_time / total_time
noncomputable def probability_A : ℝ := favorable_time / total_time

-- State the proposition to prove that the probability is 1/10
theorem board_train_immediately_probability : probability_A = 1 / 10 :=
by sorry

end board_train_immediately_probability_l2096_209611


namespace shortest_player_height_l2096_209625

theorem shortest_player_height :
  ∀ (tallest_height difference : ℝ), 
    tallest_height = 77.75 ∧ difference = 9.5 → 
    tallest_height - difference = 68.25 :=
by
  intros tallest_height difference h
  cases h
  sorry

end shortest_player_height_l2096_209625


namespace find_b_l2096_209677

-- Define functions p and q
def p (x : ℝ) : ℝ := 3 * x - 5
def q (x : ℝ) (b : ℝ) : ℝ := 4 * x - b

-- Set the target value for p(q(3))
def target_val : ℝ := 9

-- Prove that b = 22/3
theorem find_b (b : ℝ) : p (q 3 b) = target_val → b = 22 / 3 := by
  intro h
  sorry

end find_b_l2096_209677


namespace intersection_point_exists_l2096_209638

noncomputable def line1 (t : ℝ) : ℝ × ℝ := (1 - 2 * t, 2 + 6 * t)
noncomputable def line2 (u : ℝ) : ℝ × ℝ := (3 + u, 8 + 3 * u)

theorem intersection_point_exists :
  ∃ t u : ℝ, line1 t = (1, 2) ∧ line2 u = (1, 2) := 
by
  sorry

end intersection_point_exists_l2096_209638


namespace equation_of_line_l2096_209633

noncomputable def vector := (Real × Real)
noncomputable def point := (Real × Real)

def line_equation (x y : Real) : Prop := 
  let v1 : vector := (-1, 2)
  let p : point := (3, -4)
  let lhs := (v1.1 * (x - p.1) + v1.2 * (y - p.2)) = 0
  lhs

theorem equation_of_line (x y : Real) :
  line_equation x y ↔ y = (1/2) * x - (11/2) := 
  sorry

end equation_of_line_l2096_209633


namespace larger_segment_length_l2096_209609

theorem larger_segment_length 
  (x y : ℝ)
  (h1 : 40^2 = x^2 + y^2)
  (h2 : 90^2 = (110 - x)^2 + y^2) :
  110 - x = 84.55 :=
by
  sorry

end larger_segment_length_l2096_209609


namespace extreme_value_and_tangent_line_l2096_209608

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * x^2 - 3 * x

theorem extreme_value_and_tangent_line (a b : ℝ) (h1 : f a b 1 = 0) (h2 : f a b (-1) = 0) :
  (f 1 0 (-1) = 2) ∧ (f 1 0 1 = -2) ∧ (∀ x : ℝ, x = -2 → (9 * x - (x^3 - 3 * x) + 16 = 0)) :=
by
  sorry

end extreme_value_and_tangent_line_l2096_209608


namespace bulb_standard_probability_l2096_209689

noncomputable def prob_A 
  (P_H1 : ℝ) (P_H2 : ℝ) (P_A_given_H1 : ℝ) (P_A_given_H2 : ℝ) :=
  P_A_given_H1 * P_H1 + P_A_given_H2 * P_H2

theorem bulb_standard_probability 
  (P_H1 : ℝ := 0.6) (P_H2 : ℝ := 0.4) 
  (P_A_given_H1 : ℝ := 0.95) (P_A_given_H2 : ℝ := 0.85) :
  prob_A P_H1 P_H2 P_A_given_H1 P_A_given_H2 = 0.91 :=
by
  sorry

end bulb_standard_probability_l2096_209689


namespace sum_moments_equal_l2096_209667

theorem sum_moments_equal
  (x1 x2 x3 y1 y2 : ℝ)
  (m1 m2 m3 n1 n2 : ℝ) :
  n1 * y1 + n2 * y2 = m1 * x1 + m2 * x2 + m3 * x3 :=
sorry

end sum_moments_equal_l2096_209667


namespace square_side_length_l2096_209672

theorem square_side_length (x : ℝ) (h : x^2 = (1/2) * x * 2) : x = 1 := by
  sorry

end square_side_length_l2096_209672


namespace power_of_fraction_l2096_209623

theorem power_of_fraction :
  ( (2 / 5: ℝ) ^ 7 = 128 / 78125) :=
by
  sorry

end power_of_fraction_l2096_209623


namespace inv_sum_eq_six_l2096_209671

theorem inv_sum_eq_six (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a + b = 6 * (a * b)) : 1 / a + 1 / b = 6 := 
by 
  sorry

end inv_sum_eq_six_l2096_209671


namespace condition_A_sufficient_not_necessary_condition_B_l2096_209673

theorem condition_A_sufficient_not_necessary_condition_B {a b : ℝ} (hA : a > 1 ∧ b > 1) : 
  (a + b > 2 ∧ ab > 1) ∧ ¬∀ a b, (a + b > 2 ∧ ab > 1) → (a > 1 ∧ b > 1) :=
by
  sorry

end condition_A_sufficient_not_necessary_condition_B_l2096_209673


namespace nell_has_cards_left_l2096_209660

def initial_cards : ℕ := 242
def cards_given_away : ℕ := 136

theorem nell_has_cards_left :
  initial_cards - cards_given_away = 106 :=
by
  sorry

end nell_has_cards_left_l2096_209660


namespace number_of_envelopes_l2096_209664

theorem number_of_envelopes (total_weight_grams : ℕ) (weight_per_envelope_grams : ℕ) (n : ℕ) :
  total_weight_grams = 7480 ∧ weight_per_envelope_grams = 8500 ∧ n = 880 → total_weight_grams = n * weight_per_envelope_grams := 
sorry

end number_of_envelopes_l2096_209664


namespace B_time_to_finish_race_l2096_209678

theorem B_time_to_finish_race (t : ℝ) 
  (race_distance : ℝ := 130)
  (A_time : ℝ := 36)
  (A_beats_B_by : ℝ := 26)
  (A_speed : ℝ := race_distance / A_time) 
  (B_distance_when_A_finishes : ℝ := race_distance - A_beats_B_by) 
  (B_speed := B_distance_when_A_finishes / t) :
  B_speed * (t - A_time) = A_beats_B_by → t = 48 := 
by
  intros h
  sorry

end B_time_to_finish_race_l2096_209678


namespace slope_intercept_form_correct_l2096_209686

theorem slope_intercept_form_correct:
  ∀ (x y : ℝ), (2 * (x - 3) - 1 * (y + 4) = 0) → (∃ m b, y = m * x + b ∧ m = 2 ∧ b = -10) :=
by
  intro x y h
  use 2, -10
  sorry

end slope_intercept_form_correct_l2096_209686


namespace polynomial_divisibility_l2096_209636

theorem polynomial_divisibility (m : ℕ) (h_pos : 0 < m) : 
  ∀ x : ℝ, x * (x + 1) * (2 * x + 1) ∣ (x + 1)^(2 * m) - x^(2 * m) - 2 * x - 1 :=
sorry

end polynomial_divisibility_l2096_209636


namespace possible_analytical_expression_for_f_l2096_209683

noncomputable def f (x : ℝ) : ℝ := (Real.sin (2 * x)) + (Real.cos (2 * x))

theorem possible_analytical_expression_for_f :
  (∀ x : ℝ, f (x + π) = f x) ∧
  (∀ x : ℝ, f (x - π/4) = f (-x)) ∧
  (∀ x : ℝ, π/8 < x ∧ x < π/2 → f x < f (x - 1)) :=
by
  sorry

end possible_analytical_expression_for_f_l2096_209683


namespace suitable_comprehensive_survey_l2096_209639

theorem suitable_comprehensive_survey :
  ¬(A = "comprehensive") ∧ ¬(B = "comprehensive") ∧ (C = "comprehensive") ∧ ¬(D = "comprehensive") → 
  suitable_survey = "C" :=
by
  sorry

end suitable_comprehensive_survey_l2096_209639


namespace ratio_correct_l2096_209669

def cost_of_flasks := 150
def remaining_budget := 25
def total_budget := 325
def spent_budget := total_budget - remaining_budget
def cost_of_test_tubes := 100
def cost_of_safety_gear := cost_of_test_tubes / 2
def ratio_test_tubes_flasks := cost_of_test_tubes / cost_of_flasks

theorem ratio_correct :
  spent_budget = cost_of_flasks + cost_of_test_tubes + cost_of_safety_gear → 
  ratio_test_tubes_flasks = 2 / 3 :=
by
  sorry

end ratio_correct_l2096_209669


namespace ROI_diff_after_2_years_is_10_l2096_209616

variables (investment_Emma : ℝ) (investment_Briana : ℝ)
variables (yield_Emma : ℝ) (yield_Briana : ℝ)
variables (years : ℝ)

def annual_ROI_Emma (investment_Emma yield_Emma : ℝ) : ℝ :=
  yield_Emma * investment_Emma

def annual_ROI_Briana (investment_Briana yield_Briana : ℝ) : ℝ :=
  yield_Briana * investment_Briana

def total_ROI_Emma (investment_Emma yield_Emma years : ℝ) : ℝ :=
  annual_ROI_Emma investment_Emma yield_Emma * years

def total_ROI_Briana (investment_Briana yield_Briana years : ℝ) : ℝ :=
  annual_ROI_Briana investment_Briana yield_Briana * years

def ROI_difference (investment_Emma investment_Briana yield_Emma yield_Briana years : ℝ) : ℝ :=
  total_ROI_Briana investment_Briana yield_Briana years - total_ROI_Emma investment_Emma yield_Emma years

theorem ROI_diff_after_2_years_is_10 :
  ROI_difference 300 500 0.15 0.10 2 = 10 :=
by
  sorry

end ROI_diff_after_2_years_is_10_l2096_209616


namespace expected_value_of_win_is_162_l2096_209652

noncomputable def expected_value_of_win : ℝ :=
  (1/8) * (1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 + 7^3 + 8^3)

theorem expected_value_of_win_is_162 : expected_value_of_win = 162 := 
by 
  sorry

end expected_value_of_win_is_162_l2096_209652


namespace find_root_interval_l2096_209692

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem find_root_interval : ∃ k : ℕ, (f 1 < 0 ∧ f 2 > 0) → k = 1 :=
by
  sorry

end find_root_interval_l2096_209692


namespace pencils_more_than_pens_l2096_209687

theorem pencils_more_than_pens (pencils pens : ℕ) (h_ratio : 5 * pencils = 6 * pens) (h_pencils : pencils = 48) : 
  pencils - pens = 8 :=
by
  sorry

end pencils_more_than_pens_l2096_209687


namespace dog_food_weighs_more_l2096_209607

def weight_in_ounces (weight_in_pounds: ℕ) := weight_in_pounds * 16
def total_food_weight (cat_food_bags dog_food_bags: ℕ) (cat_food_pounds dog_food_pounds: ℕ) :=
  (cat_food_bags * weight_in_ounces cat_food_pounds) + (dog_food_bags * weight_in_ounces dog_food_pounds)

theorem dog_food_weighs_more
  (cat_food_bags: ℕ) (cat_food_pounds: ℕ) (dog_food_bags: ℕ) (total_weight_ounces: ℕ) (ounces_in_pound: ℕ)
  (H1: cat_food_bags * weight_in_ounces cat_food_pounds = 96)
  (H2: total_food_weight cat_food_bags dog_food_bags cat_food_pounds dog_food_pounds = total_weight_ounces)
  (H3: ounces_in_pound = 16) :
  dog_food_pounds - cat_food_pounds = 2 := 
by sorry

end dog_food_weighs_more_l2096_209607


namespace pencils_multiple_of_30_l2096_209697

-- Defines the conditions of the problem
def num_pens : ℕ := 2010
def max_students : ℕ := 30
def equal_pens_per_student := num_pens % max_students = 0

-- Proves that the number of pencils must be a multiple of 30
theorem pencils_multiple_of_30 (P : ℕ) (h1 : equal_pens_per_student) (h2 : ∀ n, n ≤ max_students → ∃ m, n * m = num_pens) : ∃ k : ℕ, P = max_students * k :=
sorry

end pencils_multiple_of_30_l2096_209697


namespace sin_alpha_plus_3pi_div_2_l2096_209602

theorem sin_alpha_plus_3pi_div_2 (α : ℝ) (h : Real.cos α = 1 / 3) : 
  Real.sin (α + 3 * Real.pi / 2) = -1 / 3 :=
by
  sorry

end sin_alpha_plus_3pi_div_2_l2096_209602


namespace vivians_mail_in_august_l2096_209657

-- Definitions based on the conditions provided
def mail_july : ℕ := 40
def business_days_august : ℕ := 22
def weekend_days_august : ℕ := 9

-- Lean 4 statement to prove the equivalent proof problem
theorem vivians_mail_in_august :
  let mail_business_days := 2 * mail_july
  let total_mail_business_days := business_days_august * mail_business_days
  let mail_weekend_days := mail_july / 2
  let total_mail_weekend_days := weekend_days_august * mail_weekend_days
  total_mail_business_days + total_mail_weekend_days = 1940 := by
  sorry

end vivians_mail_in_august_l2096_209657


namespace mandy_chocolate_pieces_l2096_209631

def chocolate_pieces_total : ℕ := 60
def half (n : ℕ) : ℕ := n / 2

def michael_taken : ℕ := half chocolate_pieces_total
def paige_taken : ℕ := half (chocolate_pieces_total - michael_taken)
def ben_taken : ℕ := half (chocolate_pieces_total - michael_taken - paige_taken)
def mandy_left : ℕ := chocolate_pieces_total - michael_taken - paige_taken - ben_taken

theorem mandy_chocolate_pieces : mandy_left = 8 :=
  by
  -- proof to be provided here
  sorry

end mandy_chocolate_pieces_l2096_209631


namespace exists_a_div_by_3_l2096_209646

theorem exists_a_div_by_3 (a : ℝ) (h : ∀ n : ℕ, ∃ k : ℤ, a * n * (n + 2) * (n + 4) = k) :
  ∃ k : ℤ, a = k / 3 :=
by
  sorry

end exists_a_div_by_3_l2096_209646


namespace coeff_a_zero_l2096_209614

-- Define the problem in Lean 4

theorem coeff_a_zero (a b c : ℝ) (h : ∀ p : ℝ, 0 < p → ∀ x, a * x^2 + b * x + c + p = 0 → 0 < x) :
  a = 0 :=
sorry

end coeff_a_zero_l2096_209614


namespace find_solutions_l2096_209654

noncomputable def cuberoot (x : ℝ) : ℝ := x^(1/3)

theorem find_solutions :
    {x : ℝ | cuberoot x = 15 / (8 - cuberoot x)} = {125, 27} :=
by
  sorry

end find_solutions_l2096_209654


namespace fraction_expression_of_repeating_decimal_l2096_209622

theorem fraction_expression_of_repeating_decimal :
  ∃ (x : ℕ), x = 79061333 ∧ (∀ y : ℚ, y = 0.71 + 264 * (1/999900) → x / 999900 = y) :=
by
  sorry

end fraction_expression_of_repeating_decimal_l2096_209622


namespace solve_3x_plus_7y_eq_23_l2096_209665

theorem solve_3x_plus_7y_eq_23 :
  ∃ (x y : ℕ), 3 * x + 7 * y = 23 ∧ x = 3 ∧ y = 2 := by
sorry

end solve_3x_plus_7y_eq_23_l2096_209665


namespace Joey_swimming_days_l2096_209663

-- Define the conditions and required proof statement
theorem Joey_swimming_days (E : ℕ) (h1 : 3 * E / 4 = 9) : E / 2 = 6 :=
by
  sorry

end Joey_swimming_days_l2096_209663


namespace maximum_profit_l2096_209658

/-- 
Given:
- The fixed cost is 3000 (in thousand yuan).
- The revenue per hundred vehicles is 500 (in thousand yuan).
- The additional cost y is defined as follows:
  - y = 10*x^2 + 100*x for 0 < x < 40
  - y = 501*x + 10000/x - 4500 for x ≥ 40
  
Prove:
1. The profit S(x) (in thousand yuan) in 2020 is:
   - S(x) = -10*x^2 + 400*x - 3000 for 0 < x < 40
   - S(x) = 1500 - x - 10000/x for x ≥ 40
2. The production volume x (in hundreds of vehicles) to achieve the maximum profit is 100,
   and the maximum profit is 1300 (in thousand yuan).
-/
noncomputable def profit_function (x : ℝ) : ℝ :=
  if (0 < x ∧ x < 40) then
    -10 * x^2 + 400 * x - 3000
  else if (x ≥ 40) then
    1500 - x - 10000 / x
  else
    0 -- Undefined for other values, though our x will always be positive in our case

theorem maximum_profit : ∃ x : ℝ, 0 < x ∧ 
  (profit_function x = 1300 ∧ x = 100) ∧
  ∀ y, 0 < y → profit_function y ≤ 1300 :=
sorry

end maximum_profit_l2096_209658


namespace twenty_percent_greater_l2096_209613

theorem twenty_percent_greater (x : ℝ) (h : x = 52 + 0.2 * 52) : x = 62.4 :=
by {
  sorry
}

end twenty_percent_greater_l2096_209613


namespace reflection_point_sum_l2096_209670

theorem reflection_point_sum (m b : ℝ) (H : ∀ x y : ℝ, (1, 2) = (x, y) ∨ (7, 6) = (x, y) → 
    y = m * x + b) : m + b = 8.5 := by
  sorry

end reflection_point_sum_l2096_209670


namespace x_div_11p_is_integer_l2096_209645

theorem x_div_11p_is_integer (x p : ℕ) (h1 : x > 0) (h2 : Prime p) (h3 : x = 66) : ∃ k : ℤ, x / (11 * p) = k := by
  sorry

end x_div_11p_is_integer_l2096_209645


namespace tyson_age_l2096_209627

noncomputable def age_proof : Prop :=
  let k := 25      -- Kyle's age
  let j := k - 5   -- Julian's age
  let f := j + 20  -- Frederick's age
  let t := f / 2   -- Tyson's age
  t = 20           -- Statement that needs to be proved

theorem tyson_age : age_proof :=
by
  let k := 25      -- Kyle's age
  let j := k - 5   -- Julian's age
  let f := j + 20  -- Frederick's age
  let t := f / 2   -- Tyson's age
  show t = 20
  sorry

end tyson_age_l2096_209627


namespace inequality_proof_l2096_209628

theorem inequality_proof (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) :
  (1 / (1 - x^2)) + (1 / (1 - y^2)) ≥ (2 / (1 - x * y)) :=
by sorry

end inequality_proof_l2096_209628


namespace shortest_hypotenuse_max_inscribed_circle_radius_l2096_209635

variable {a b c r : ℝ}

-- Condition 1: The perimeter of the right-angled triangle is 1 meter.
def perimeter_condition (a b : ℝ) : Prop :=
  a + b + Real.sqrt (a^2 + b^2) = 1

-- Problem 1: Prove the shortest length of the hypotenuse is √2 - 1.
theorem shortest_hypotenuse (a b : ℝ) (h : perimeter_condition a b) :
  Real.sqrt (a^2 + b^2) = Real.sqrt 2 - 1 :=
sorry

-- Problem 2: Prove the maximum value of the inscribed circle radius is 3/2 - √2.
theorem max_inscribed_circle_radius (a b r : ℝ) (h : perimeter_condition a b) :
  (a * b = r) → r = 3/2 - Real.sqrt 2 :=
sorry

end shortest_hypotenuse_max_inscribed_circle_radius_l2096_209635


namespace find_rate_of_grapes_l2096_209655

def rate_per_kg_of_grapes (G : ℝ) : Prop :=
  let cost_of_grapes := 8 * G
  let cost_of_mangoes := 10 * 55
  let total_paid := 1110
  cost_of_grapes + cost_of_mangoes = total_paid

theorem find_rate_of_grapes : rate_per_kg_of_grapes 70 :=
by
  unfold rate_per_kg_of_grapes
  sorry

end find_rate_of_grapes_l2096_209655


namespace integer_solutions_count_eq_11_l2096_209651

theorem integer_solutions_count_eq_11 :
  ∃ (count : ℕ), (∀ n : ℤ, (n + 2) * (n - 5) + n ≤ 10 ↔ (n ≥ -4 ∧ n ≤ 6)) ∧ count = 11 :=
by
  sorry

end integer_solutions_count_eq_11_l2096_209651


namespace slices_with_only_mushrooms_l2096_209680

theorem slices_with_only_mushrooms :
  ∀ (T P M n : ℕ),
    T = 16 →
    P = 9 →
    M = 12 →
    (9 - n) + (12 - n) + n = 16 →
    M - n = 7 :=
by
  intros T P M n hT hP hM h_eq
  sorry

end slices_with_only_mushrooms_l2096_209680


namespace no_natural_numbers_satisfy_equation_l2096_209640

theorem no_natural_numbers_satisfy_equation :
  ¬ ∃ (x y : ℕ), Nat.gcd x y + Nat.lcm x y + x + y = 2019 :=
by
  sorry

end no_natural_numbers_satisfy_equation_l2096_209640


namespace algebra_expression_value_l2096_209653

theorem algebra_expression_value (a : ℤ) (h : (2023 - a) ^ 2 + (a - 2022) ^ 2 = 7) :
  (2023 - a) * (a - 2022) = -3 := 
sorry

end algebra_expression_value_l2096_209653


namespace num_prime_divisors_50_factorial_eq_15_l2096_209601

theorem num_prime_divisors_50_factorial_eq_15 :
  (Finset.filter Nat.Prime (Finset.range 51)).card = 15 :=
by
  sorry

end num_prime_divisors_50_factorial_eq_15_l2096_209601


namespace correct_calculation_l2096_209661

theorem correct_calculation :
  (-2 * a * b^2)^3 = -8 * a^3 * b^6 :=
by sorry

end correct_calculation_l2096_209661


namespace solve_cubic_eq_l2096_209688

theorem solve_cubic_eq (x : ℝ) (h1 : (x + 1)^3 = x^3) (h2 : 0 ≤ x) (h3 : x < 1) : x = 0 :=
by
  sorry

end solve_cubic_eq_l2096_209688


namespace arithmetic_seq_a7_l2096_209685

theorem arithmetic_seq_a7 (a : ℕ → ℤ) (d : ℤ) (h1 : ∀ (n m : ℕ), a (n + m) = a n + m * d)
  (h2 : a 4 + a 9 = 24) (h3 : a 6 = 11) :
  a 7 = 13 :=
sorry

end arithmetic_seq_a7_l2096_209685


namespace peter_change_left_l2096_209675

theorem peter_change_left
  (cost_small : ℕ := 3)
  (cost_large : ℕ := 5)
  (total_money : ℕ := 50)
  (num_small : ℕ := 8)
  (num_large : ℕ := 5) :
  total_money - (num_small * cost_small + num_large * cost_large) = 1 :=
by
  sorry

end peter_change_left_l2096_209675


namespace same_leading_digit_l2096_209679

theorem same_leading_digit (n : ℕ) (hn : 0 < n) : 
  (∀ a k l : ℕ, (a * 10^k < 2^n ∧ 2^n < (a+1) * 10^k) ∧ (a * 10^l < 5^n ∧ 5^n < (a+1) * 10^l) → a = 3) := 
sorry

end same_leading_digit_l2096_209679


namespace alice_bob_meet_l2096_209624

theorem alice_bob_meet (n : ℕ) (h_n : n = 18) (alice_move : ℕ) (bob_move : ℕ)
  (h_alice : alice_move = 7) (h_bob : bob_move = 13) :
  ∃ k : ℕ, alice_move * k % n = (n - bob_move) * k % n :=
by
  sorry

end alice_bob_meet_l2096_209624


namespace rectangular_prism_volume_l2096_209674

theorem rectangular_prism_volume
  (L W h : ℝ)
  (h1 : L - W = 23)
  (h2 : 2 * L + 2 * W = 166) :
  L * W * h = 1590 * h :=
by
  sorry

end rectangular_prism_volume_l2096_209674


namespace steven_more_peaches_than_apples_l2096_209606

def steven_peaches : Nat := 17
def steven_apples : Nat := 16

theorem steven_more_peaches_than_apples : steven_peaches - steven_apples = 1 := by
  sorry

end steven_more_peaches_than_apples_l2096_209606


namespace greatest_divisor_l2096_209693

theorem greatest_divisor (d : ℕ) (h₀ : 1657 % d = 6) (h₁ : 2037 % d = 5) : d = 127 :=
by
  -- Proof skipped
  sorry

end greatest_divisor_l2096_209693


namespace net_change_in_collection_is_94_l2096_209604

-- Definitions for the given conditions
def thrown_away_caps : Nat := 6
def initially_found_caps : Nat := 50
def additionally_found_caps : Nat := 44 + thrown_away_caps

-- Definition of the total found bottle caps
def total_found_caps : Nat := initially_found_caps + additionally_found_caps

-- Net change in Bottle Cap collection
def net_change_in_collection : Nat := total_found_caps - thrown_away_caps

-- Proof statement
theorem net_change_in_collection_is_94 : net_change_in_collection = 94 :=
by
  -- skipped proof
  sorry

end net_change_in_collection_is_94_l2096_209604


namespace find_cost_price_l2096_209637

theorem find_cost_price (C : ℝ) (h1 : 0.88 * C + 1500 = 1.12 * C) : C = 6250 := 
by
  sorry

end find_cost_price_l2096_209637


namespace ratio_future_age_l2096_209626

variables (S : ℕ) (M : ℕ) (S_future : ℕ) (M_future : ℕ)

def son_age := 44
def man_age := son_age + 46
def son_age_future := son_age + 2
def man_age_future := man_age + 2

theorem ratio_future_age : man_age_future / son_age_future = 2 := by
  -- You can add the proof here if you want
  sorry

end ratio_future_age_l2096_209626


namespace contrapositive_of_x_squared_lt_one_is_true_l2096_209649

variable {x : ℝ}

theorem contrapositive_of_x_squared_lt_one_is_true
  (h : ∀ x : ℝ, x^2 < 1 → -1 < x ∧ x < 1) :
  ∀ x : ℝ, x ≤ -1 ∨ x ≥ 1 → x^2 ≥ 1 :=
by
  sorry

end contrapositive_of_x_squared_lt_one_is_true_l2096_209649


namespace line_circle_intersection_common_points_l2096_209620

noncomputable def radius (d : ℝ) := d / 2

theorem line_circle_intersection_common_points 
  (diameter : ℝ) (distance_from_center_to_line : ℝ) 
  (h_dlt_r : distance_from_center_to_line < radius diameter) :
  ∃ common_points : ℕ, common_points = 2 :=
by
  sorry

end line_circle_intersection_common_points_l2096_209620


namespace arithmetic_sequence_fifth_term_l2096_209612

theorem arithmetic_sequence_fifth_term 
  (a : ℕ → ℤ) 
  (S : ℕ → ℤ) 
  (h_arith_seq : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (a6 : a 6 = -3) 
  (S6 : S 6 = 12)
  (h_sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 1 - a 0)) / 2)
  : a 5 = -1 :=
sorry

end arithmetic_sequence_fifth_term_l2096_209612


namespace total_number_of_pieces_paper_l2096_209699

-- Define the number of pieces of paper each person picked up
def olivia_pieces : ℝ := 127.5
def edward_pieces : ℝ := 345.25
def sam_pieces : ℝ := 518.75

-- Define the total number of pieces of paper picked up
def total_pieces : ℝ := olivia_pieces + edward_pieces + sam_pieces

-- The theorem to be proven
theorem total_number_of_pieces_paper :
  total_pieces = 991.5 :=
by
  -- Sorry is used as we are not required to provide a proof here
  sorry

end total_number_of_pieces_paper_l2096_209699


namespace rectangle_perimeter_l2096_209681

theorem rectangle_perimeter (a b : ℝ) (h1 : (a + 3) * (b + 3) = a * b + 48) : 
  2 * (a + 3 + b + 3) = 38 :=
by
  sorry

end rectangle_perimeter_l2096_209681


namespace wholesale_cost_l2096_209648

theorem wholesale_cost (W R : ℝ) (h1 : R = 1.20 * W) (h2 : 0.70 * R = 168) : W = 200 :=
by
  sorry

end wholesale_cost_l2096_209648


namespace probability_two_girls_from_twelve_l2096_209698

theorem probability_two_girls_from_twelve : 
  let total_members := 12
  let boys := 4
  let girls := 8
  let choose_two_total := Nat.choose total_members 2
  let choose_two_girls := Nat.choose girls 2
  let probability := (choose_two_girls : ℚ) / (choose_two_total : ℚ)
  probability = (14 / 33) := by
  -- Proof goes here
  sorry

end probability_two_girls_from_twelve_l2096_209698


namespace least_x_y_z_value_l2096_209632

theorem least_x_y_z_value :
  ∃ (x y z : ℕ), (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (3 * x = 4 * y) ∧ (4 * y = 7 * z) ∧ (3 * x = 7 * z) ∧ (x - y + z = 19) :=
by
  sorry

end least_x_y_z_value_l2096_209632


namespace water_filter_capacity_l2096_209629

theorem water_filter_capacity (x : ℝ) (h : 0.30 * x = 36) : x = 120 :=
sorry

end water_filter_capacity_l2096_209629


namespace price_of_child_ticket_l2096_209691

theorem price_of_child_ticket (total_seats : ℕ) (adult_ticket_price : ℕ) (total_revenue : ℕ)
  (child_tickets_sold : ℕ) (child_ticket_price : ℕ) :
  total_seats = 80 →
  adult_ticket_price = 12 →
  total_revenue = 519 →
  child_tickets_sold = 63 →
  (17 * adult_ticket_price) + (child_tickets_sold * child_ticket_price) = total_revenue →
  child_ticket_price = 5 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end price_of_child_ticket_l2096_209691


namespace perpendicular_pair_is_14_l2096_209610

variable (x y : ℝ)

def equation1 := 4 * y - 3 * x = 16
def equation2 := -3 * x - 4 * y = 15
def equation3 := 4 * y + 3 * x = 16
def equation4 := 3 * y + 4 * x = 15

theorem perpendicular_pair_is_14 : (∃ y1 y2 x1 x2 : ℝ,
  4 * y1 - 3 * x1 = 16 ∧ 3 * y2 + 4 * x2 = 15 ∧ (3 / 4) * (-4 / 3) = -1) :=
sorry

end perpendicular_pair_is_14_l2096_209610


namespace expected_winnings_correct_l2096_209644

def winnings (roll : ℕ) : ℚ :=
  if roll % 2 = 1 then 0
  else if roll % 4 = 0 then 2 * roll
  else roll

def expected_winnings : ℚ :=
  (winnings 1) / 8 + (winnings 2) / 8 +
  (winnings 3) / 8 + (winnings 4) / 8 +
  (winnings 5) / 8 + (winnings 6) / 8 +
  (winnings 7) / 8 + (winnings 8) / 8

theorem expected_winnings_correct : expected_winnings = 3.75 := by 
  sorry

end expected_winnings_correct_l2096_209644


namespace pure_ghee_percentage_l2096_209618

theorem pure_ghee_percentage (Q : ℝ) (vanaspati_percentage : ℝ:= 0.40) (additional_pure_ghee : ℝ := 10) (new_vanaspati_percentage : ℝ := 0.20) (original_quantity : ℝ := 10) :
  (Q = original_quantity) ∧ (vanaspati_percentage = 0.40) ∧ (additional_pure_ghee = 10) ∧ (new_vanaspati_percentage = 0.20) →
  (100 - (vanaspati_percentage * 100)) = 60 :=
by
  sorry

end pure_ghee_percentage_l2096_209618


namespace intersection_is_open_interval_l2096_209619

open Set
open Real

noncomputable def M : Set ℝ := {x | x < 1}
noncomputable def N : Set ℝ := {x | 0 < x ∧ x < 2}

theorem intersection_is_open_interval :
  M ∩ N = { x | 0 < x ∧ x < 1 } := by
  sorry

end intersection_is_open_interval_l2096_209619


namespace plane_overtake_time_is_80_minutes_l2096_209634

noncomputable def plane_overtake_time 
  (speed_a speed_b : ℝ)
  (head_start : ℝ) 
  (t : ℝ) : Prop :=
  speed_a * (t + head_start) = speed_b * t

theorem plane_overtake_time_is_80_minutes :
  plane_overtake_time 200 300 (2/3) (80 / 60)
:=
  sorry

end plane_overtake_time_is_80_minutes_l2096_209634


namespace positive_solution_x_l2096_209603

theorem positive_solution_x (x y z : ℝ) (h1 : x * y = 10 - 3 * x - 2 * y) 
(h2 : y * z = 10 - 5 * y - 3 * z) 
(h3 : x * z = 40 - 5 * x - 2 * z) 
(h_pos : x > 0) : 
  x = 8 :=
sorry

end positive_solution_x_l2096_209603


namespace upper_limit_of_range_l2096_209605

theorem upper_limit_of_range (N : ℕ) :
  (∀ n : ℕ, (20 + n * 10 ≤ N) = (n < 198)) → N = 1990 :=
by
  sorry

end upper_limit_of_range_l2096_209605


namespace intersection_of_M_and_N_l2096_209641

noncomputable def M : Set ℝ := {x | x - 2 > 0}
noncomputable def N : Set ℝ := {y | ∃ x, y = Real.sqrt (x^2 + 1)}

theorem intersection_of_M_and_N :
  M ∩ N = {x | x > 2} :=
sorry

end intersection_of_M_and_N_l2096_209641


namespace light_coloured_blocks_in_tower_l2096_209630

theorem light_coloured_blocks_in_tower :
  let central_blocks := 4
  let outer_columns := 8
  let height_per_outer_column := 2
  let total_light_coloured_blocks := central_blocks + outer_columns * height_per_outer_column
  total_light_coloured_blocks = 20 :=
by
  let central_blocks := 4
  let outer_columns := 8
  let height_per_outer_column := 2
  let total_light_coloured_blocks := central_blocks + outer_columns * height_per_outer_column
  show total_light_coloured_blocks = 20
  sorry

end light_coloured_blocks_in_tower_l2096_209630


namespace mean_of_two_equals_mean_of_three_l2096_209684

theorem mean_of_two_equals_mean_of_three (z : ℚ) : 
  (5 + 10 + 20) / 3 = (15 + z) / 2 → 
  z = 25 / 3 := 
by 
  sorry

end mean_of_two_equals_mean_of_three_l2096_209684


namespace nails_needed_l2096_209650

-- Define the number of nails needed for each plank
def nails_per_plank : ℕ := 2

-- Define the number of planks used by John
def planks_used : ℕ := 16

-- The total number of nails needed.
theorem nails_needed : (nails_per_plank * planks_used) = 32 :=
by
  -- Our goal is to prove that nails_per_plank * planks_used = 32
  sorry

end nails_needed_l2096_209650


namespace shortest_distance_dasha_vasya_l2096_209666

variables (dasha galia asya borya vasya : Type)
variables (dist : ∀ (a b : Type), ℕ)
variables (dist_dasha_galia : dist dasha galia = 15)
variables (dist_vasya_galia : dist vasya galia = 17)
variables (dist_asya_galia : dist asya galia = 12)
variables (dist_galia_borya : dist galia borya = 10)
variables (dist_asya_borya : dist asya borya = 8)

theorem shortest_distance_dasha_vasya : dist dasha vasya = 18 :=
by sorry

end shortest_distance_dasha_vasya_l2096_209666


namespace cube_decomposition_smallest_number_91_l2096_209642

theorem cube_decomposition_smallest_number_91 (m : ℕ) (h1 : 0 < m) (h2 : (91 - 1) / 2 + 2 = m * m - m + 1) : m = 10 := by {
  sorry
}

end cube_decomposition_smallest_number_91_l2096_209642


namespace least_number_of_apples_l2096_209659

theorem least_number_of_apples (b : ℕ) : (b % 3 = 2) → (b % 4 = 3) → (b % 5 = 1) → b = 11 :=
by
  intros h1 h2 h3
  sorry

end least_number_of_apples_l2096_209659


namespace matrix_power_B_l2096_209621

def B : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![0, 1, 0],
  ![0, 0, 1],
  ![1, 0, 0]
]

theorem matrix_power_B :
  B ^ 150 = 1 :=
by sorry

end matrix_power_B_l2096_209621


namespace Marcus_fit_pies_l2096_209682

theorem Marcus_fit_pies (x : ℕ) 
(h1 : ∀ b, (7 * b - 8) = 27) : x = 5 := by
  sorry

end Marcus_fit_pies_l2096_209682


namespace roots_of_quadratic_eval_l2096_209647

theorem roots_of_quadratic_eval :
  ∀ x₁ x₂ : ℝ, (x₁^2 + 4 * x₁ + 2 = 0) ∧ (x₂^2 + 4 * x₂ + 2 = 0) ∧ (x₁ + x₂ = -4) ∧ (x₁ * x₂ = 2) →
    x₁^3 + 14 * x₂ + 55 = 7 :=
by
  sorry

end roots_of_quadratic_eval_l2096_209647


namespace leak_empties_tank_in_18_hours_l2096_209615

theorem leak_empties_tank_in_18_hours :
  let A : ℚ := 1 / 6
  let L : ℚ := 1 / 6 - 1 / 9
  (1 / L) = 18 := by
    sorry

end leak_empties_tank_in_18_hours_l2096_209615


namespace leif_has_more_oranges_than_apples_l2096_209662

-- We are given that Leif has 14 apples and 24 oranges.
def number_of_apples : ℕ := 14
def number_of_oranges : ℕ := 24

-- We need to show how many more oranges he has than apples.
theorem leif_has_more_oranges_than_apples :
  number_of_oranges - number_of_apples = 10 :=
by
  -- The proof would go here, but we are skipping it.
  sorry

end leif_has_more_oranges_than_apples_l2096_209662


namespace fraction_of_sides_area_of_triangle_l2096_209690

-- Part (1)
theorem fraction_of_sides (A B C : ℝ) (a b c : ℝ) (h_triangle : A + B + C = π)
  (h_sines : 2 * (Real.tan A + Real.tan B) = (Real.tan A / Real.cos B) + (Real.tan B / Real.cos A))
  (h_sine_law : c = 2) : (a + b) / c = 2 :=
sorry

-- Part (2)
theorem area_of_triangle (A B C : ℝ) (a b c : ℝ) (h_triangle : A + B + C = π)
  (h_sines : 2 * (Real.tan A + Real.tan B) = (Real.tan A / Real.cos B) + (Real.tan B / Real.cos A))
  (h_sine_law : c = 2) (h_C : C = π / 3) : (1 / 2) * a * b * Real.sin C = Real.sqrt 3 :=
sorry

end fraction_of_sides_area_of_triangle_l2096_209690


namespace probability_different_colors_l2096_209694

theorem probability_different_colors
  (red_chips green_chips : ℕ)
  (total_chips : red_chips + green_chips = 10)
  (prob_red : ℚ := red_chips / 10)
  (prob_green : ℚ := green_chips / 10) :
  ((prob_red * prob_green) + (prob_green * prob_red) = 12 / 25) := by
sorry

end probability_different_colors_l2096_209694


namespace bananas_count_l2096_209695

theorem bananas_count
    (total_fruit : ℕ)
    (apples_ratio : ℕ)
    (persimmons_ratio : ℕ)
    (apples_and_persimmons : apples_ratio * bananas + persimmons_ratio * bananas = total_fruit)
    (apples_ratio_val : apples_ratio = 4)
    (persimmons_ratio_val : persimmons_ratio = 3)
    (total_fruit_value : total_fruit = 210) :
    bananas = 30 :=
by
  sorry

end bananas_count_l2096_209695

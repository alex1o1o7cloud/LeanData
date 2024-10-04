import Mathlib

namespace original_price_of_shoes_l103_103651

theorem original_price_of_shoes (P : ℝ) (h1 : 2 * 0.60 * P + 0.80 * 100 = 140) : P = 50 :=
by
  sorry

end original_price_of_shoes_l103_103651


namespace find_x_l103_103755

theorem find_x (x : ℝ) (h : 2500 - 1002 / x = 2450) : x = 20.04 :=
by 
  sorry

end find_x_l103_103755


namespace angle_negative_225_in_second_quadrant_l103_103879

def inSecondQuadrant (angle : Int) : Prop :=
  angle % 360 > -270 ∧ angle % 360 <= -180

theorem angle_negative_225_in_second_quadrant :
  inSecondQuadrant (-225) :=
by
  sorry

end angle_negative_225_in_second_quadrant_l103_103879


namespace combined_earnings_l103_103515

theorem combined_earnings (dwayne_earnings brady_earnings : ℕ) (h1 : dwayne_earnings = 1500) (h2 : brady_earnings = dwayne_earnings + 450) : 
  dwayne_earnings + brady_earnings = 3450 :=
by 
  rw [h1, h2]
  sorry

end combined_earnings_l103_103515


namespace p_sufficient_not_necessary_for_q_l103_103411

variable (a : ℝ)

def p : Prop := a > 0
def q : Prop := a^2 + a ≥ 0

theorem p_sufficient_not_necessary_for_q : (p a → q a) ∧ ¬ (q a → p a) := by
  sorry

end p_sufficient_not_necessary_for_q_l103_103411


namespace find_a_if_circle_l103_103080

noncomputable def curve_eq (a x y : ℝ) : ℝ :=
  a^2 * x^2 + (a + 2) * y^2 + 2 * a * x + a

def is_circle_condition (a : ℝ) : Prop :=
  ∀ x y : ℝ, curve_eq a x y = 0 → (∃ k : ℝ, curve_eq a x y = k * (x^2 + y^2))

theorem find_a_if_circle :
  (∀ a : ℝ, is_circle_condition a → a = -1) :=
by
  sorry

end find_a_if_circle_l103_103080


namespace inequality_for_pos_reals_l103_103797

open Real Nat

theorem inequality_for_pos_reals
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (h : 1/a + 1/b = 1)
  (n : ℕ) :
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n + 1) :=
by 
  sorry

end inequality_for_pos_reals_l103_103797


namespace quadratic_complete_square_l103_103397

theorem quadratic_complete_square : ∀ x : ℝ, (x^2 - 8*x - 1) = (x - 4)^2 - 17 :=
by sorry

end quadratic_complete_square_l103_103397


namespace completing_the_square_solution_correct_l103_103174

theorem completing_the_square_solution_correct (x : ℝ) :
  (x^2 + 8 * x + 9 = 0) ↔ ((x + 4)^2 = 7) :=
by
  sorry

end completing_the_square_solution_correct_l103_103174


namespace product_of_5_consecutive_numbers_not_square_l103_103297

-- Define what it means for a product to be a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- The main theorem stating the problem
theorem product_of_5_consecutive_numbers_not_square :
  ∀ (a : ℕ), 0 < a → ¬ is_perfect_square (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
by 
  sorry

end product_of_5_consecutive_numbers_not_square_l103_103297


namespace probability_none_solve_l103_103890

theorem probability_none_solve (a b c : ℕ) 
    (ha : a > 0) (hb : b > 0) (hc : c > 0)
    (h_prob : ((1 - (1/a)) * (1 - (1/b)) * (1 - (1/c)) = 8/15)) : 
  (1 - (1/a)) * (1 - (1/b)) * (1 - (1/c)) = 8/15 := 
by 
  sorry

end probability_none_solve_l103_103890


namespace constant_S13_l103_103150

theorem constant_S13 (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) (h_seq : ∀ n, a n = a 1 + (n - 1) * d) 
(h_sum : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2)
(h_constant : ∀ a1 d, (a 2 + a 8 + a 11 = 3 * a1 + 18 * d)) : (S 13 = 91 * d) :=
by
  sorry

end constant_S13_l103_103150


namespace jack_bill_age_difference_l103_103153

def jack_bill_ages_and_difference (a b : ℕ) :=
  let jack_age := 10 * a + b
  let bill_age := 10 * b + a
  (a + b = 2) ∧ (7 * a - 29 * b = 14) → jack_age - bill_age = 18

theorem jack_bill_age_difference (a b : ℕ) (h₀ : a + b = 2) (h₁ : 7 * a - 29 * b = 14) : 
  let jack_age := 10 * a + b
  let bill_age := 10 * b + a
  jack_age - bill_age = 18 :=
by {
  sorry
}

end jack_bill_age_difference_l103_103153


namespace systematic_sampling_student_l103_103392

theorem systematic_sampling_student (total_students sample_size : ℕ) 
  (h_total_students : total_students = 56)
  (h_sample_size : sample_size = 4)
  (student1 student2 student3 student4 : ℕ)
  (h_student1 : student1 = 6)
  (h_student2 : student2 = 34)
  (h_student3 : student3 = 48) :
  student4 = 20 :=
sorry

end systematic_sampling_student_l103_103392


namespace tractors_planting_rate_l103_103789

theorem tractors_planting_rate (total_acres : ℕ) (total_days : ℕ) 
    (tractors_first_team : ℕ) (days_first_team : ℕ)
    (tractors_second_team : ℕ) (days_second_team : ℕ)
    (total_tractor_days : ℕ) :
    total_acres = 1700 →
    total_days = 5 →
    tractors_first_team = 2 →
    days_first_team = 2 →
    tractors_second_team = 7 →
    days_second_team = 3 →
    total_tractor_days = (tractors_first_team * days_first_team) + (tractors_second_team * days_second_team) →
    total_acres / total_tractor_days = 68 :=
by
  -- proof can be filled in later
  intros
  sorry

end tractors_planting_rate_l103_103789


namespace find_tan_of_cos_in_4th_quadrant_l103_103251

-- Given conditions
variable (α : ℝ) (h1 : Real.cos α = 3/5) (h2 : α > 3*Real.pi/2 ∧ α < 2*Real.pi)

-- Lean statement to prove the question
theorem find_tan_of_cos_in_4th_quadrant : Real.tan α = - (4 / 3) := 
by
  sorry

end find_tan_of_cos_in_4th_quadrant_l103_103251


namespace find_s_and_x_l103_103666

theorem find_s_and_x (s x t : ℝ) (h1 : t = 15 * s^2) (h2 : t = 3.75) :
  s = 0.5 ∧ x = s / 2 → x = 0.25 :=
by
  sorry

end find_s_and_x_l103_103666


namespace a_range_l103_103069

noncomputable def f (a x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else a ^ x - a

theorem a_range (a : ℝ) : (∀ x y : ℝ, x < y → f a x ≥ f a y) ↔ a ∈ Set.Ico (1/7 : ℝ) (1/3 : ℝ) :=
by
  sorry

end a_range_l103_103069


namespace value_of_m_if_f_is_power_function_l103_103425

theorem value_of_m_if_f_is_power_function (m : ℤ) :
  (2 * m + 3 = 1) → m = -1 :=
by
  sorry

end value_of_m_if_f_is_power_function_l103_103425


namespace cost_price_l103_103222

theorem cost_price (MP SP C : ℝ) (h1 : MP = 74.21875)
  (h2 : SP = MP - 0.20 * MP)
  (h3 : SP = 1.25 * C) : C = 47.5 :=
by
  sorry

end cost_price_l103_103222


namespace perimeter_of_rhombus_l103_103310

theorem perimeter_of_rhombus (d1 d2 : ℝ) (hd1 : d1 = 8) (hd2 : d2 = 30) :
  (perimeter : ℝ) = 4 * Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) :=
by
  simp [hd1, hd2]
  sorry

end perimeter_of_rhombus_l103_103310


namespace max_distance_l103_103543

-- Define the curve C
def curve_C (x y : ℝ) : Prop := (x^2 / 9) + y^2 = 1

-- Define the Cartesian equation of line l
def line_l (x y : ℝ) : Prop := (x - y + 2) = 0

-- Prove the maximum distance from any point P on curve C to line l
theorem max_distance (P : ℝ × ℝ) (hP : curve_C P.1 P.2) : 
  ∃ d, d = sqrt 5 + sqrt 2 ∧ ∀ x y, curve_C x y → abs (x - y + 2) / sqrt 2 ≤ d :=
sorry

end max_distance_l103_103543


namespace positive_number_is_nine_l103_103258

theorem positive_number_is_nine (x : ℝ) (n : ℝ) (hx : x > 0) (hn : n > 0)
  (sqrt1 : x^2 = n) (sqrt2 : (x - 6)^2 = n) : 
  n = 9 :=
by
  sorry

end positive_number_is_nine_l103_103258


namespace product_of_5_consecutive_numbers_not_square_l103_103299

-- Define what it means for a product to be a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- The main theorem stating the problem
theorem product_of_5_consecutive_numbers_not_square :
  ∀ (a : ℕ), 0 < a → ¬ is_perfect_square (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
by 
  sorry

end product_of_5_consecutive_numbers_not_square_l103_103299


namespace roots_reciprocal_l103_103722

theorem roots_reciprocal (a b c r s : ℝ) (h_eqn : a ≠ 0) (h_roots : a * r^2 + b * r + c = 0 ∧ a * s^2 + b * s + c = 0) (h_cond : b^2 = 4 * a * c) : r * s = 1 :=
by
  -- Proof goes here
  sorry

end roots_reciprocal_l103_103722


namespace find_x_y_n_l103_103282

def is_reverse_digit (x y : ℕ) : Prop := 
  x / 10 = y % 10 ∧ x % 10 = y / 10

def is_two_digit_nonzero (z : ℕ) : Prop := 
  10 ≤ z ∧ z < 100

theorem find_x_y_n : 
  ∃ (x y n : ℕ), is_two_digit_nonzero x ∧ is_two_digit_nonzero y ∧ is_reverse_digit x y ∧ (x^2 - y^2 = 44 * n) ∧ (x + y + n = 93) :=
sorry

end find_x_y_n_l103_103282


namespace cost_of_old_car_l103_103773

theorem cost_of_old_car (C_old C_new : ℝ): 
  C_new = 2 * C_old → 
  1800 + 2000 = C_new → 
  C_old = 1900 :=
by
  intros H1 H2
  sorry

end cost_of_old_car_l103_103773


namespace largest_A_l103_103219

theorem largest_A (A B C : ℕ) (h1 : A = 7 * B + C) (h2 : B = C) : A ≤ 48 :=
  sorry

end largest_A_l103_103219


namespace min_value_expression_l103_103555

theorem min_value_expression (x : ℝ) (h : x ≠ -7) : 
  ∃ y, y = 1 ∧ ∀ z, z = (2 * x ^ 2 + 98) / ((x + 7) ^ 2) → y ≤ z := 
sorry

end min_value_expression_l103_103555


namespace deductive_reasoning_not_always_correct_l103_103110

theorem deductive_reasoning_not_always_correct (P: Prop) (Q: Prop) 
    (h1: (P → Q) → (P → Q)) :
    (¬ (∀ P Q : Prop, (P → Q) → Q → Q)) :=
sorry

end deductive_reasoning_not_always_correct_l103_103110


namespace combined_height_l103_103845

/-- Given that Mr. Martinez is two feet taller than Chiquita and Chiquita is 5 feet tall, prove that their combined height is 12 feet. -/
theorem combined_height (h_chiquita : ℕ) (h_martinez : ℕ) 
  (h1 : h_chiquita = 5) (h2 : h_martinez = h_chiquita + 2) : 
  h_chiquita + h_martinez = 12 :=
by sorry

end combined_height_l103_103845


namespace find_a_b_l103_103546

noncomputable def f (a b c x : ℝ) : ℝ := a * x^3 + b * x + c

theorem find_a_b (a b c : ℝ) (h1 : (12 * a + b = 0)) (h2 : (4 * a + b = -3)) :
  a = 3 / 8 ∧ b = -9 / 2 := by
  sorry

end find_a_b_l103_103546


namespace complete_square_l103_103187

theorem complete_square (x : ℝ) (h : x^2 + 8 * x + 9 = 0) : (x + 4)^2 = 7 := by
  sorry

end complete_square_l103_103187


namespace final_amounts_calculation_l103_103709

noncomputable def article_A_original_cost : ℚ := 200
noncomputable def article_B_original_cost : ℚ := 300
noncomputable def article_C_original_cost : ℚ := 400
noncomputable def exchange_rate_euro_to_usd : ℚ := 1.10
noncomputable def exchange_rate_gbp_to_usd : ℚ := 1.30
noncomputable def discount_A : ℚ := 0.50
noncomputable def discount_B : ℚ := 0.30
noncomputable def discount_C : ℚ := 0.40
noncomputable def sales_tax_rate : ℚ := 0.05
noncomputable def reward_points : ℚ := 100
noncomputable def reward_point_value : ℚ := 0.05

theorem final_amounts_calculation :
  let discounted_A := article_A_original_cost * discount_A
  let final_A := (article_A_original_cost - discounted_A) * exchange_rate_euro_to_usd
  let discounted_B := article_B_original_cost * discount_B
  let final_B := (article_B_original_cost - discounted_B) * exchange_rate_gbp_to_usd
  let discounted_C := article_C_original_cost * discount_C
  let final_C := article_C_original_cost - discounted_C
  let total_discounted_cost_usd := final_A + final_B + final_C
  let sales_tax := total_discounted_cost_usd * sales_tax_rate
  let reward := reward_points * reward_point_value
  let final_amount_usd := total_discounted_cost_usd + sales_tax - reward
  let final_amount_euro := final_amount_usd / exchange_rate_euro_to_usd
  final_amount_usd = 649.15 ∧ final_amount_euro = 590.14 :=
by
  sorry

end final_amounts_calculation_l103_103709


namespace cody_paid_17_l103_103519

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

end cody_paid_17_l103_103519


namespace wood_not_heavier_than_brick_l103_103947

-- Define the weights of the wood and the brick
def block_weight_kg : ℝ := 8
def brick_weight_g : ℝ := 8000

-- Conversion function from kg to g
def kg_to_g (kg : ℝ) : ℝ := kg * 1000

-- State the proof problem
theorem wood_not_heavier_than_brick : ¬ (kg_to_g block_weight_kg > brick_weight_g) :=
by
  -- Begin the proof
  sorry

end wood_not_heavier_than_brick_l103_103947


namespace net_effect_sale_value_net_effect_sale_value_percentage_increase_l103_103697

def sale_value (P Q : ℝ) : ℝ := P * Q

theorem net_effect_sale_value (P Q : ℝ) :
  sale_value (0.8 * P) (1.8 * Q) = 1.44 * sale_value P Q :=
by
  sorry

theorem net_effect_sale_value_percentage_increase (P Q : ℝ) :
  (sale_value (0.8 * P) (1.8 * Q) - sale_value P Q) / sale_value P Q = 0.44 :=
by
  sorry

end net_effect_sale_value_net_effect_sale_value_percentage_increase_l103_103697


namespace find_ab_l103_103385

theorem find_ab (a b : ℝ) 
  (H_period : (1 : ℝ) * (π / b) = π / 2)
  (H_point : a * Real.tan (b * (π / 8)) = 4) :
  a * b = 8 :=
sorry

end find_ab_l103_103385


namespace lowest_price_for_butter_l103_103378

def cost_single_package : ℝ := 7.0
def cost_8oz_package : ℝ := 4.0
def cost_4oz_package : ℝ := 2.0
def discount : ℝ := 0.5

theorem lowest_price_for_butter : 
  min cost_single_package (cost_8oz_package + 2 * (cost_4oz_package * discount)) = 6.0 :=
by
  sorry

end lowest_price_for_butter_l103_103378


namespace value_at_neg_9_over_2_l103_103408

def f : ℝ → ℝ := sorry 

axiom odd_function (x : ℝ) : f (-x) + f x = 0

axiom symmetric_y_axis (x : ℝ) : f (1 + x) = f (1 - x)

axiom functional_eq (x k : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hk : 0 ≤ k ∧ k ≤ 1) : f (k * x) + 1 = (f x + 1) ^ k

axiom f_at_1 : f 1 = - (1 / 2)

theorem value_at_neg_9_over_2 : f (- (9 / 2)) = 1 - (Real.sqrt 2) / 2 := 
sorry

end value_at_neg_9_over_2_l103_103408


namespace inequality_solution_l103_103894

-- Define the variable x as a real number
variable (x : ℝ)

-- Define the given condition that x is positive
def is_positive (x : ℝ) := x > 0

-- Define the condition that x satisfies the inequality sqrt(9x) < 3x^2
def satisfies_inequality (x : ℝ) := Real.sqrt (9 * x) < 3 * x^2

-- The statement we need to prove
theorem inequality_solution (x : ℝ) (h : is_positive x) : satisfies_inequality x ↔ x > 1 :=
sorry

end inequality_solution_l103_103894


namespace arithmetic_sequence_general_formula_l103_103108

noncomputable def arithmetic_sequence (a : ℕ → ℤ) :=
  ∀ n, a n = 2 * n - 3

theorem arithmetic_sequence_general_formula
    (a : ℕ → ℤ)
    (h1 : (a 2 + a 6) / 2 = 5)
    (h2 : (a 3 + a 7) / 2 = 7) :
  arithmetic_sequence a :=
by
  sorry

end arithmetic_sequence_general_formula_l103_103108


namespace minimum_value_of_expression_l103_103073

noncomputable def min_value_expression (x y : ℝ) : ℝ := 
  (x + 1)^2 / (x + 2) + 3 / (x + 2) + y^2 / (y + 1)

theorem minimum_value_of_expression :
  ∀ (x y : ℝ), x ≥ 0 → y ≥ 0 → x + y = 2 → min_value_expression x y = 14 / 5 :=
by
  sorry

end minimum_value_of_expression_l103_103073


namespace complete_square_q_value_l103_103949

theorem complete_square_q_value :
  ∃ p q, (16 * x^2 - 32 * x - 512 = 0) ∧ ((x + p)^2 = q) → q = 33 := by
  sorry

end complete_square_q_value_l103_103949


namespace orange_ratio_l103_103505

theorem orange_ratio (total_oranges alice_oranges : ℕ) (h_total : total_oranges = 180) (h_alice : alice_oranges = 120) :
  alice_oranges / (total_oranges - alice_oranges) = 2 :=
by
  sorry

end orange_ratio_l103_103505


namespace lowest_price_is_six_l103_103376

def single_package_cost : ℝ := 7
def eight_oz_package_cost : ℝ := 4
def four_oz_package_original_cost : ℝ := 2
def discount_rate : ℝ := 0.5

theorem lowest_price_is_six
  (cost_single : single_package_cost = 7)
  (cost_eight : eight_oz_package_cost = 4)
  (cost_four : four_oz_package_original_cost = 2)
  (discount : discount_rate = 0.5) :
  min single_package_cost (eight_oz_package_cost + 2 * (four_oz_package_original_cost * discount_rate)) = 6 := by
  sorry

end lowest_price_is_six_l103_103376


namespace probability_forming_more_from_remont_probability_forming_papa_from_papaha_l103_103750

-- Definition for part (a)
theorem probability_forming_more_from_remont : 
  (6 * 5 * 4 * 3 = 360) ∧ (1 / 360 = 0.00278) :=
by
  sorry

-- Definition for part (b)
theorem probability_forming_papa_from_papaha : 
  (6 * 5 * 4 * 3 = 360) ∧ (12 / 360 = 0.03333) :=
by
  sorry

end probability_forming_more_from_remont_probability_forming_papa_from_papaha_l103_103750


namespace cube_painted_four_faces_l103_103765

theorem cube_painted_four_faces (n : ℕ) (hn : n ≠ 0) (h : (4 * n^2) / (6 * n^3) = 1 / 3) : n = 2 :=
by
  have : 4 * n^2 = 4 * n^2 := by rfl
  sorry

end cube_painted_four_faces_l103_103765


namespace david_is_30_l103_103328

-- Definitions representing the conditions
def uncleBobAge : ℕ := 60
def emilyAge : ℕ := (2 * uncleBobAge) / 3
def davidAge : ℕ := emilyAge - 10

-- Statement that represents the equivalence to be proven
theorem david_is_30 : davidAge = 30 :=
by
  sorry

end david_is_30_l103_103328


namespace negation_exists_positive_real_square_plus_one_l103_103537

def exists_positive_real_square_plus_one : Prop :=
  ∃ (x : ℝ), x^2 + 1 > 0

def forall_non_positive_real_square_plus_one : Prop :=
  ∀ (x : ℝ), x^2 + 1 ≤ 0

theorem negation_exists_positive_real_square_plus_one :
  ¬ exists_positive_real_square_plus_one ↔ forall_non_positive_real_square_plus_one :=
by
  sorry

end negation_exists_positive_real_square_plus_one_l103_103537


namespace volume_in_cubic_yards_l103_103355

-- Define the conditions given in the problem
def volume_in_cubic_feet : ℕ := 216
def cubic_feet_per_cubic_yard : ℕ := 27

-- Define the theorem that needs to be proven
theorem volume_in_cubic_yards :
  volume_in_cubic_feet / cubic_feet_per_cubic_yard = 8 :=
by
  sorry

end volume_in_cubic_yards_l103_103355


namespace problem_statement_l103_103091

theorem problem_statement (a x : ℝ) (h_linear_eq : (a + 4) * x ^ |a + 3| + 8 = 0) : a^2 + a - 1 = 1 :=
sorry

end problem_statement_l103_103091


namespace volume_in_cubic_yards_l103_103361

theorem volume_in_cubic_yards (V : ℝ) (conversion_factor : ℝ) (hV : V = 216) (hcf : conversion_factor = 27) :
  V / conversion_factor = 8 := by
  sorry

end volume_in_cubic_yards_l103_103361


namespace number_of_members_l103_103500

theorem number_of_members (n : ℕ) (h1 : ∀ m : ℕ, m = n → m * m = 1936) : n = 44 :=
by
  -- Proof omitted
  sorry

end number_of_members_l103_103500


namespace arithmetic_expression_eval_l103_103393

theorem arithmetic_expression_eval : 2 + 8 * 3 - 4 + 10 * 2 / 5 = 26 := by
  sorry

end arithmetic_expression_eval_l103_103393


namespace ratio_sum_eq_seven_eight_l103_103933

theorem ratio_sum_eq_seven_eight 
  (a b c x y z : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h1 : a^2 + b^2 + c^2 = 49)
  (h2 : x^2 + y^2 + z^2 = 64)
  (h3 : a * x + b * y + c * z = 56) :
  (a + b + c) / (x + y + z) = 7/8 :=
by
  sorry

end ratio_sum_eq_seven_eight_l103_103933


namespace min_value_u_l103_103072

theorem min_value_u (x y : ℝ) (h₀ : x ≥ 0) (h₁ : y ≥ 0)
  (h₂ : 2 * x + y = 6) : 
  ∀u, u = 4 * x ^ 2 + 3 * x * y + y ^ 2 - 6 * x - 3 * y -> 
  u ≥ 27 / 2 := sorry

end min_value_u_l103_103072


namespace sqrt_x_minus_2_domain_l103_103834

theorem sqrt_x_minus_2_domain {x : ℝ} : (∃y : ℝ, y = Real.sqrt (x - 2)) ↔ x ≥ 2 :=
by sorry

end sqrt_x_minus_2_domain_l103_103834


namespace segments_after_cuts_l103_103729

-- Definitions from the conditions
def cuts : ℕ := 10

-- Mathematically equivalent proof statement
theorem segments_after_cuts : (cuts + 1 = 11) :=
by sorry

end segments_after_cuts_l103_103729


namespace magic_8_ball_probability_l103_103707

noncomputable def probability_exactly_4_positive (p q : ℚ) (n k : ℕ) : ℚ :=
  (Nat.choose n k) * (p ^ k) * (q ^ (n - k))

open Probability

theorem magic_8_ball_probability :
  probability_exactly_4_positive (3 / 7) (4 / 7) 7 4 = 181440 / 823543 :=
by
  sorry

end magic_8_ball_probability_l103_103707


namespace correct_calculation_l103_103999

-- Define the conditions of the problem
variable (x : ℕ)
variable (h : x + 5 = 43)

-- The theorem we want to prove
theorem correct_calculation : 5 * x = 190 :=
by
  -- Since Lean requires a proof and we're skipping it, we use 'sorry'
  sorry

end correct_calculation_l103_103999


namespace remaining_water_l103_103711

theorem remaining_water (initial_water : ℚ) (used_water : ℚ) (remaining_water : ℚ) 
  (h1 : initial_water = 3) (h2 : used_water = 5/4) : remaining_water = 7/4 :=
by
  -- The proof would go here, but we are skipping it as per the instructions.
  sorry

end remaining_water_l103_103711


namespace evaluate_powers_of_i_l103_103787

noncomputable def imag_unit := Complex.I

theorem evaluate_powers_of_i :
  (imag_unit^11 + imag_unit^16 + imag_unit^21 + imag_unit^26 + imag_unit^31) = -imag_unit :=
by
  sorry

end evaluate_powers_of_i_l103_103787


namespace volume_in_cubic_yards_l103_103354

-- Define the conditions given in the problem
def volume_in_cubic_feet : ℕ := 216
def cubic_feet_per_cubic_yard : ℕ := 27

-- Define the theorem that needs to be proven
theorem volume_in_cubic_yards :
  volume_in_cubic_feet / cubic_feet_per_cubic_yard = 8 :=
by
  sorry

end volume_in_cubic_yards_l103_103354


namespace xy_product_l103_103875

theorem xy_product (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : x * y = 21 :=
sorry

end xy_product_l103_103875


namespace angle_in_triangle_l103_103706

theorem angle_in_triangle
  (A B C : Type)
  (a b c : ℝ)
  (angle_ABC : ℝ)
  (h1 : a = 15)
  (h2 : angle_ABC = π/3 ∨ angle_ABC = 2 * π / 3)
  : angle_ABC = π/3 ∨ angle_ABC = 2 * π / 3 := 
  sorry

end angle_in_triangle_l103_103706


namespace traffic_safety_team_eq_twice_fire_l103_103832

-- Define initial members in the teams
def t0 : ℕ := 8
def f0 : ℕ := 7

-- Define the main theorem
theorem traffic_safety_team_eq_twice_fire (x : ℕ) : t0 + x = 2 * (f0 - x) :=
by sorry

end traffic_safety_team_eq_twice_fire_l103_103832


namespace total_amount_paid_l103_103276

-- Definitions based on the conditions.
def cost_per_pizza : ℝ := 12
def delivery_charge : ℝ := 2
def distance_threshold : ℝ := 1000 -- distance in meters
def park_distance : ℝ := 100
def building_distance : ℝ := 2000

def pizzas_at_park : ℕ := 3
def pizzas_at_building : ℕ := 2

-- The proof problem stating the total amount paid to Jimmy.
theorem total_amount_paid :
  let total_pizzas := pizzas_at_park + pizzas_at_building
  let cost_without_delivery := total_pizzas * cost_per_pizza
  let park_charge := if park_distance > distance_threshold then pizzas_at_park * delivery_charge else 0
  let building_charge := if building_distance > distance_threshold then pizzas_at_building * delivery_charge else 0
  let total_cost := cost_without_delivery + park_charge + building_charge
  total_cost = 64 :=
by
  sorry

end total_amount_paid_l103_103276


namespace simplify_polynomial_l103_103138

theorem simplify_polynomial : 
  (12 * X^10 - 3 * X^9 + 8 * X^8 - 5 * X^7) - (2 * X^10 + 2 * X^9 - X^8 + X^7 + 4 * X^4 + 6 * X^2 + 9) 
  = 10 * X^10 - 5 * X^9 + 9 * X^8 - 6 * X^7 - 4 * X^4 - 6 * X^2 - 9 :=
by
  sorry

end simplify_polynomial_l103_103138


namespace problem1_problem2_l103_103205

-- Problem 1
theorem problem1 (m n : ℝ)
    (H : ∀ x : ℝ, mx^2 + 3 * x - 2 > 0 ↔ n < x ∧ x < 2) :
  m = -1 ∧ n = 1 :=
by sorry

-- Problem 2
theorem problem2 (a : ℝ)
    (H : ∀ x : ℝ, x^2 + (a - 1) * x - a > 0 ↔ 
        (a < -1 ∧ (x > 1 ∨ x < -a)) ∨ 
        (a = -1 ∧ x ≠ 1)) :
  (a < -1 → ∀ x : ℝ, x^2 + (a - 1) * x - a > 0 ↔ x > 1 ∨ x < -a) ∧
  (a = -1 → ∀ x : ℝ, x^2 + (a - 1) * x - a > 0 ↔ x ≠ 1) :=
by sorry

end problem1_problem2_l103_103205


namespace man_l103_103210

-- Define the man's rowing speed in still water, the speed of the current, the downstream speed and headwind reduction.
def v : Real := 17.5
def speed_current : Real := 4.5
def speed_downstream : Real := 22
def headwind_reduction : Real := 1.5

-- Define the man's speed against the current and headwind.
def speed_against_current_headwind := v - speed_current - headwind_reduction

-- The statement to prove. 
theorem man's_speed_against_current_and_headwind :
  speed_against_current_headwind = 11.5 := by
  -- Using the conditions (which are already defined in lean expressions above), we can end the proof here.
  sorry

end man_l103_103210


namespace inequality_relation_l103_103674

open Real

theorem inequality_relation (x : ℝ) :
  ¬ ((∀ x, (x - 1) * (x + 3) < 0 → (x + 1) * (x - 3) < 0) ∧
     (∀ x, (x + 1) * (x - 3) < 0 → (x - 1) * (x + 3) < 0)) := 
by
  sorry

end inequality_relation_l103_103674


namespace stadium_length_in_feet_l103_103157

-- Assume the length of the stadium is 80 yards
def stadium_length_yards := 80

-- Assume the conversion factor is 3 feet per yard
def conversion_factor := 3

-- The length in feet is the product of the length in yards and the conversion factor
def length_in_feet := stadium_length_yards * conversion_factor

-- We want to prove that this length in feet is 240 feet
theorem stadium_length_in_feet : length_in_feet = 240 := by
  -- Definitions and conditions are directly restated here; the proof is sketched as 'sorry'
  sorry

end stadium_length_in_feet_l103_103157


namespace root_inverse_cubes_l103_103934

theorem root_inverse_cubes (a b c r s : ℝ) (h1 : a ≠ 0)
  (h2 : ∀ x, a * x^2 + b * x + c = 0 ↔ x = r ∨ x = s) :
  (1 / r^3) + (1 / s^3) = (-b^3 + 3 * a * b * c) / c^3 :=
by
  sorry

end root_inverse_cubes_l103_103934


namespace inequality_proof_l103_103937

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (a * b * (b + 1) * (c + 1))) + 
  (1 / (b * c * (c + 1) * (a + 1))) + 
  (1 / (c * a * (a + 1) * (b + 1))) ≥ 
  (3 / (1 + a * b * c)^2) :=
sorry

end inequality_proof_l103_103937


namespace reciprocal_of_one_is_one_l103_103867

def is_reciprocal (x y : ℝ) : Prop := x * y = 1

theorem reciprocal_of_one_is_one : is_reciprocal 1 1 := 
by
  sorry

end reciprocal_of_one_is_one_l103_103867


namespace repeating_decimal_as_fraction_l103_103622

theorem repeating_decimal_as_fraction :
  (0.58207 : ℝ) = 523864865 / 999900 := sorry

end repeating_decimal_as_fraction_l103_103622


namespace ticket_price_values_l103_103039

theorem ticket_price_values : 
  ∃ (x_values : Finset ℕ), 
    (∀ x ∈ x_values, x ∣ 60 ∧ x ∣ 80) ∧ 
    x_values.card = 6 :=
by
  sorry

end ticket_price_values_l103_103039


namespace certain_positive_integer_value_l103_103700

-- Define factorial
def fact : ℕ → ℕ 
| 0     => 1
| (n+1) => (n+1) * fact n

-- Statement of the problem
theorem certain_positive_integer_value (i k m a : ℕ) :
  (fact 8 = 2^i * 3^k * 5^m * 7^a) ∧ (i + k + m + a = 11) → a = 1 :=
by 
  sorry

end certain_positive_integer_value_l103_103700


namespace necessary_and_sufficient_condition_l103_103684

def line1 (a : ℝ) (x y : ℝ) := 2 * x - a * y + 1 = 0
def line2 (a : ℝ) (x y : ℝ) := (a - 1) * x - y + a = 0
def parallel (a : ℝ) : Prop := ∀ x y : ℝ, line1 a x y = line2 a x y

theorem necessary_and_sufficient_condition (a : ℝ) : 
  (a = 2 ↔ parallel a) :=
sorry

end necessary_and_sufficient_condition_l103_103684


namespace cost_price_for_A_l103_103214

variable (A B C : Type) [Field A] [Field B] [Field C]

noncomputable def cost_price (CP_A : A) := 
  let CP_B := 1.20 * CP_A
  let CP_C := 1.25 * CP_B
  CP_C = (225 : A)

theorem cost_price_for_A (CP_A : A) : cost_price CP_A -> CP_A = 150 := by
  sorry

end cost_price_for_A_l103_103214


namespace find_unknown_number_l103_103935

theorem find_unknown_number (x : ℕ) (hx1 : 100 % x = 16) (hx2 : 200 % x = 4) : x = 28 :=
by 
  sorry

end find_unknown_number_l103_103935


namespace no_integer_n_gt_1_satisfies_inequality_l103_103049

open Int

theorem no_integer_n_gt_1_satisfies_inequality :
  ∀ (n : ℤ), n > 1 → ¬ (⌊(Real.sqrt (↑n - 2) + 2 * Real.sqrt (↑n + 2))⌋ < ⌊Real.sqrt (9 * (↑n : ℝ) + 6)⌋) :=
by
  intros n hn
  sorry

end no_integer_n_gt_1_satisfies_inequality_l103_103049


namespace common_number_is_eleven_l103_103872

theorem common_number_is_eleven 
  (a b c d e f g h i : ℝ)
  (H1 : (a + b + c + d + e) / 5 = 7)
  (H2 : (e + f + g + h + i) / 5 = 10)
  (H3 : (a + b + c + d + e + f + g + h + i) / 9 = 74 / 9) :
  e = 11 := 
sorry

end common_number_is_eleven_l103_103872


namespace expression_value_l103_103161

theorem expression_value : (5 - 2) / (2 + 1) = 1 := by
  sorry

end expression_value_l103_103161


namespace neg_existential_proposition_l103_103685

open Nat

theorem neg_existential_proposition :
  (¬ (∃ n : ℕ, n + 10 / n < 4)) ↔ (∀ n : ℕ, n + 10 / n ≥ 4) :=
by
  sorry

end neg_existential_proposition_l103_103685


namespace range_of_a_l103_103427

open Real

theorem range_of_a (a : ℝ) 
  (h : ¬ ∃ x₀ : ℝ, 2 ^ x₀ - 2 ≤ a ^ 2 - 3 * a) : 1 ≤ a ∧ a ≤ 2 := 
by
  sorry

end range_of_a_l103_103427


namespace number_of_valid_sets_l103_103261

universe u

def U : Set ℕ := {1,2,3,4,5,6,7,8,9,10}
def valid_set (A : Set ℕ) : Prop :=
  ∃ a1 a2 a3, A = {a1, a2, a3} ∧ a3 ∈ U ∧ a2 ∈ U ∧ a1 ∈ U ∧ a3 ≥ a2 + 1 ∧ a2 ≥ a1 + 4

theorem number_of_valid_sets : ∃ (n : ℕ), n = 56 ∧ ∃ S : Finset (Set ℕ), (∀ A ∈ S, valid_set A) ∧ S.card = n := by
  sorry

end number_of_valid_sets_l103_103261


namespace custom_op_2006_l103_103906

def custom_op (n : ℕ) : ℕ := 
  match n with 
  | 0 => 1
  | (n+1) => 2 + custom_op n

theorem custom_op_2006 : custom_op 2005 = 4011 :=
by {
  sorry
}

end custom_op_2006_l103_103906


namespace find_a_l103_103253

-- Define the conditions and the proof goal
theorem find_a (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h_eq : a + a⁻¹ = 5/2) :
  a = 1/2 :=
by
  sorry

end find_a_l103_103253


namespace height_of_block_l103_103212

theorem height_of_block (h : ℝ) : 
  ((∃ (side : ℝ), ∃ (n : ℕ), side = 15 ∧ n = 10 ∧ 15 * 30 * h = n * side^3) → h = 75) := 
by
  intros
  sorry

end height_of_block_l103_103212


namespace mod_exponent_problem_l103_103193

theorem mod_exponent_problem : (11 ^ 2023) % 100 = 31 := by
  sorry

end mod_exponent_problem_l103_103193


namespace problem_l103_103270

-- Conditions
variables (x y : ℚ)
def condition1 := 3 * x + 5 = 12
def condition2 := 10 * y - 2 = 5

-- Theorem to prove
theorem problem (h1 : condition1 x) (h2 : condition2 y) : x + y = 91 / 30 := sorry

end problem_l103_103270


namespace min_distance_curve_C_to_line_l_l103_103942

noncomputable section

open Real

def curve_C (θ : ℝ) : ℝ × ℝ :=
  (4 * cos θ, 3 * sin θ)

def line_l (t : ℝ) : ℝ × ℝ :=
  (3 + (sqrt 2 / 2) * t, -3 + (sqrt 2 / 2) * t)

def distance_point_to_line (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  abs (x - y - 6) / sqrt 2

theorem min_distance_curve_C_to_line_l :
  ∃ θ : ℝ, distance_point_to_line (curve_C θ) = sqrt 2 / 2 :=
sorry

end min_distance_curve_C_to_line_l_l103_103942


namespace population_net_increase_l103_103941

-- Define conditions
def birth_rate : ℚ := 5 / 2    -- 5 people every 2 seconds
def death_rate : ℚ := 3 / 2    -- 3 people every 2 seconds
def one_day_in_seconds : ℕ := 86400   -- Number of seconds in one day

-- Define the net increase per second
def net_increase_per_second := birth_rate - death_rate

-- Prove that the net increase in one day is 86400 people given the conditions
theorem population_net_increase :
  net_increase_per_second * one_day_in_seconds = 86400 :=
sorry

end population_net_increase_l103_103941


namespace squares_in_ap_l103_103476

theorem squares_in_ap (a b c : ℝ) (h : (1 / (a + b) + 1 / (b + c)) / 2 = 1 / (a + c)) : 
  a^2 + c^2 = 2 * b^2 :=
by
  sorry

end squares_in_ap_l103_103476


namespace combined_tickets_l103_103649

-- Definitions from the conditions
def dave_spent : Nat := 43
def dave_left : Nat := 55
def alex_spent : Nat := 65
def alex_left : Nat := 42

-- Theorem to prove that the combined starting tickets of Dave and Alex is 205
theorem combined_tickets : dave_spent + dave_left + alex_spent + alex_left = 205 := 
by
  sorry

end combined_tickets_l103_103649


namespace complement_A_complement_U_range_of_a_empty_intersection_l103_103581

open Set Real

noncomputable def complement_A_in_U := { x : ℝ | ¬ (x < -1 ∨ x > 3) }

theorem complement_A_complement_U
  {A : Set ℝ} (hA : A = {x | x^2 - 2 * x - 3 > 0}) :
  (complement_A_in_U = (Icc (-1) 3)) :=
by sorry

theorem range_of_a_empty_intersection
  {B : Set ℝ} {a : ℝ}
  (hB : B = {x | abs (x - a) > 3})
  (h_empty : (Icc (-1) 3) ∩ B = ∅) :
  (0 ≤ a ∧ a ≤ 2) :=
by sorry

end complement_A_complement_U_range_of_a_empty_intersection_l103_103581


namespace problem_4_5_part_I_problem_4_5_part_II_l103_103461

-- Definitions used in Lean 4 are directly from the conditions in a)
def f (x m : ℝ) : ℝ := abs (x - m) - abs (x + 3 * m)

-- Definition of the first proof
theorem problem_4_5_part_I (x : ℝ) : f x 1 ≥ 1 → x ≤ -3 / 2 := 
by
  sorry

-- Definition of the second proof
theorem problem_4_5_part_II (x t m : ℝ) (h : 0 < m) : (∀ x t : ℝ, f x m < abs (2 + t) + abs (t - 1)) → m < 3 / 4 := 
by
  sorry

end problem_4_5_part_I_problem_4_5_part_II_l103_103461


namespace road_completion_l103_103469

/- 
  The company "Roga and Kopyta" undertook a project to build a road 100 km long. 
  The construction plan is: 
  - In the first month, 1 km of the road will be built.
  - Subsequently, if by the beginning of some month A km is already completed, then during that month an additional 1 / A^10 km of road will be constructed.
  Prove that the road will be completed within 100^11 months.
-/

theorem road_completion (L : ℕ → ℝ) (h1 : L 1 = 1)
  (h2 : ∀ n ≥ 1, L (n + 1) = L n + 1 / (L n) ^ 10) :
  ∃ m ≤ 100 ^ 11, L m ≥ 100 := 
  sorry

end road_completion_l103_103469


namespace sequence_limit_l103_103776

open Filter Real

noncomputable def sequence_term (n : ℕ) : ℝ :=
  sqrt (n * (n + 1) * (n + 2)) * (sqrt (n^3 - 3) - sqrt (n^3 - 2))

theorem sequence_limit :
  tendsto (λ n : ℕ, sequence_term n) at_top (𝓝 (-1/2)) :=
sorry

end sequence_limit_l103_103776


namespace oprah_years_to_reduce_collection_l103_103849

theorem oprah_years_to_reduce_collection (initial_cars final_cars average_cars_per_year : ℕ) (h1 : initial_cars = 3500) (h2 : final_cars = 500) (h3 : average_cars_per_year = 50) : 
  (initial_cars - final_cars) / average_cars_per_year = 60 := 
by
  rw [h1, h2, h3]
  sorry

end oprah_years_to_reduce_collection_l103_103849


namespace complete_the_square_l103_103176

theorem complete_the_square (x : ℝ) :
  (x^2 + 8*x + 9 = 0) ↔ ((x + 4)^2 = 7) :=
sorry

end complete_the_square_l103_103176


namespace xy_product_l103_103295

theorem xy_product (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y - 44) : x * y = -24 := 
by {
  sorry
}

end xy_product_l103_103295


namespace purchase_total_cost_l103_103584

theorem purchase_total_cost :
  (1 * 16) + (3 * 2) + (6 * 1) = 28 :=
sorry

end purchase_total_cost_l103_103584


namespace fish_buckets_last_l103_103770

theorem fish_buckets_last (buckets_sharks : ℕ) (buckets_total : ℕ) 
  (h1 : buckets_sharks = 4)
  (h2 : ∀ (buckets_dolphins : ℕ), buckets_dolphins = buckets_sharks / 2)
  (h3 : ∀ (buckets_other : ℕ), buckets_other = 5 * buckets_sharks)
  (h4 : buckets_total = 546)
  : 546 / ((buckets_sharks + (buckets_sharks / 2) + (5 * buckets_sharks)) * 7) = 3 :=
by
  -- Calculation steps skipped for brevity
  sorry

end fish_buckets_last_l103_103770


namespace combined_annual_income_l103_103977

-- Define the given conditions and verify the combined annual income
def A_ratio : ℤ := 5
def B_ratio : ℤ := 2
def C_ratio : ℤ := 3
def D_ratio : ℤ := 4

def C_income : ℤ := 15000
def B_income : ℤ := 16800
def A_income : ℤ := 25000
def D_income : ℤ := 21250

theorem combined_annual_income :
  (A_income + B_income + C_income + D_income) * 12 = 936600 :=
by
  sorry

end combined_annual_income_l103_103977


namespace final_image_of_F_is_correct_l103_103976

-- Define the initial F position as a struct
structure Position where
  base : (ℝ × ℝ)
  stem : (ℝ × ℝ)

-- Function to rotate a point 90 degrees counterclockwise around the origin
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, p.1)

-- Function to reflect a point in the x-axis
def reflectX (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Function to rotate a point by 180 degrees around the origin (half turn)
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

-- Define the initial state of F
def initialFPosition : Position := {
  base := (-1, 0),  -- Base along the negative x-axis
  stem := (0, -1)   -- Stem along the negative y-axis
}

-- Perform all transformations on the Position of F
def transformFPosition (pos : Position) : Position :=
  let afterRotation90 := Position.mk (rotate90 pos.base) (rotate90 pos.stem)
  let afterReflectionX := Position.mk (reflectX afterRotation90.base) (reflectX afterRotation90.stem)
  let finalPosition := Position.mk (rotate180 afterReflectionX.base) (rotate180 afterReflectionX.stem)
  finalPosition

-- Define the target final position we expect
def finalFPosition : Position := {
  base := (0, 1),   -- Base along the positive y-axis
  stem := (1, 0)    -- Stem along the positive x-axis
}

-- The theorem statement: After the transformations, the position of F
-- should match the final expected position
theorem final_image_of_F_is_correct :
  transformFPosition initialFPosition = finalFPosition := by
  sorry

end final_image_of_F_is_correct_l103_103976


namespace sum_of_three_squares_l103_103768

theorem sum_of_three_squares (a b : ℝ)
  (h1 : 3 * a + 2 * b = 18)
  (h2 : 2 * a + 3 * b = 22) :
  3 * b = 18 :=
sorry

end sum_of_three_squares_l103_103768


namespace scallops_cost_l103_103340

-- define the conditions
def scallops_per_pound : ℝ := 8
def cost_per_pound : ℝ := 24
def scallops_per_person : ℝ := 2
def number_of_people : ℝ := 8

-- the question
theorem scallops_cost : (scallops_per_person * number_of_people / scallops_per_pound) * cost_per_pound = 48 := by 
  sorry

end scallops_cost_l103_103340


namespace ratio_cost_to_marked_l103_103896

variable (m : ℝ)

def marked_price (m : ℝ) := m

def selling_price (m : ℝ) : ℝ := 0.75 * m

def cost_price (m : ℝ) : ℝ := 0.60 * selling_price m

theorem ratio_cost_to_marked (m : ℝ) : 
  cost_price m / marked_price m = 0.45 := 
by
  sorry

end ratio_cost_to_marked_l103_103896


namespace complement_union_l103_103264

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem complement_union :
  U \ (M ∪ N) = {4} :=
by
  sorry

end complement_union_l103_103264


namespace steve_average_speed_l103_103971

-- Define the conditions as constants
def hours1 := 5
def speed1 := 40
def hours2 := 3
def speed2 := 80
def hours3 := 2
def speed3 := 60

-- Define a theorem that calculates average speed and proves the result is 56
theorem steve_average_speed :
  (hours1 * speed1 + hours2 * speed2 + hours3 * speed3) / (hours1 + hours2 + hours3) = 56 := by
  sorry

end steve_average_speed_l103_103971


namespace parabola_intersections_l103_103619

-- Define the first parabola
def parabola1 (x : ℝ) : ℝ :=
  2 * x^2 - 10 * x - 10

-- Define the second parabola
def parabola2 (x : ℝ) : ℝ :=
  x^2 - 4 * x + 6

-- Define the theorem stating the points of intersection
theorem parabola_intersections :
  ∀ (p : ℝ × ℝ), (parabola1 p.1 = p.2) ∧ (parabola2 p.1 = p.2) ↔ (p = (-2, 18) ∨ p = (8, 38)) :=
by
  sorry

end parabola_intersections_l103_103619


namespace factor_x_squared_minus_sixtyfour_l103_103054

theorem factor_x_squared_minus_sixtyfour (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) :=
by sorry

end factor_x_squared_minus_sixtyfour_l103_103054


namespace isosceles_triangle_height_ratio_l103_103203

theorem isosceles_triangle_height_ratio (b1 h1 b2 h2 : ℝ) 
  (A1 : ℝ := 1/2 * b1 * h1) (A2 : ℝ := 1/2 * b2 * h2)
  (area_ratio : A1 / A2 = 16 / 49)
  (similar : b1 / b2 = h1 / h2) : 
  h1 / h2 = 4 / 7 := 
by {
  sorry
}

end isosceles_triangle_height_ratio_l103_103203


namespace S_is_multiples_of_six_l103_103409

-- Defining the problem.
def S : Set ℝ :=
  { t | ∃ n : ℤ, t = 6 * n }

-- We are given that S is non-empty
axiom S_non_empty : ∃ x, x ∈ S

-- Condition: For any x, y ∈ S, both x + y ∈ S and x - y ∈ S.
axiom S_closed_add_sub : ∀ x y, x ∈ S → y ∈ S → (x + y ∈ S ∧ x - y ∈ S)

-- The smallest positive number in S is 6.
axiom S_smallest : ∀ ε, ε > 0 → ∃ x, x ∈ S ∧ x = 6

-- The goal is to prove that S is exactly the set of all multiples of 6.
theorem S_is_multiples_of_six : ∀ t, t ∈ S ↔ ∃ n : ℤ, t = 6 * n :=
by
  sorry

end S_is_multiples_of_six_l103_103409


namespace volume_conversion_l103_103368

theorem volume_conversion (V_ft : ℕ) (h_V : V_ft = 216) (conversion_factor : ℕ) (h_cf : conversion_factor = 27) :
  V_ft / conversion_factor = 8 :=
by
  sorry

end volume_conversion_l103_103368


namespace sphere_radius_l103_103807

theorem sphere_radius 
  (r h1 h2 : ℝ)
  (A1_eq : 5 * π = π * (r^2 - h1^2))
  (A2_eq : 8 * π = π * (r^2 - h2^2))
  (h1_h2_eq : h1 - h2 = 1) : r = 3 :=
by
  sorry

end sphere_radius_l103_103807


namespace altitude_on_hypotenuse_l103_103106

theorem altitude_on_hypotenuse (a b : ℝ) (h₁ : a = 5) (h₂ : b = 12) (c : ℝ) (h₃ : c = Real.sqrt (a^2 + b^2)) :
  ∃ h : ℝ, h = (a * b) / c ∧ h = 60 / 13 :=
by
  use (5 * 12) / 13
  -- proof that (60 / 13) is indeed the altitude will be done by verifying calculations
  sorry

end altitude_on_hypotenuse_l103_103106


namespace complete_square_l103_103186

theorem complete_square (x : ℝ) (h : x^2 + 8 * x + 9 = 0) : (x + 4)^2 = 7 := by
  sorry

end complete_square_l103_103186


namespace circle_radius_of_equal_area_l103_103273

theorem circle_radius_of_equal_area (A B C D : Type) (r : ℝ) (π : ℝ) 
  (h_rect_area : 8 * 9 = 72)
  (h_circle_area : π * r ^ 2 = 36) :
  r = 6 / Real.sqrt π :=
by
  sorry

end circle_radius_of_equal_area_l103_103273


namespace find_x_l103_103692

theorem find_x (x y : ℚ) (h1 : 3 * x - 2 * y = 7) (h2 : x + 3 * y = 8) : x = 37 / 11 := by
  sorry

end find_x_l103_103692


namespace remainder_when_divided_by_296_and_37_l103_103990

theorem remainder_when_divided_by_296_and_37 (N : ℤ) (k : ℤ)
  (h : N = 296 * k + 75) : N % 37 = 1 :=
by
  sorry

end remainder_when_divided_by_296_and_37_l103_103990


namespace prop_C_prop_D_l103_103334

theorem prop_C (a b : ℝ) (h : a > b) : a^3 > b^3 := sorry

theorem prop_D (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c := sorry

end prop_C_prop_D_l103_103334


namespace completing_the_square_l103_103168

theorem completing_the_square (x : ℝ) : x^2 + 8 * x + 9 = 0 → (x + 4)^2 = 7 :=
by 
  intro h
  sorry

end completing_the_square_l103_103168


namespace A_and_C_complete_remaining_work_in_2_point_4_days_l103_103631

def work_rate_A : ℚ := 1 / 12
def work_rate_B : ℚ := 1 / 15
def work_rate_C : ℚ := 1 / 18
def work_completed_B_in_10_days : ℚ := (10 : ℚ) * work_rate_B
def remaining_work : ℚ := 1 - work_completed_B_in_10_days
def combined_work_rate_AC : ℚ := work_rate_A + work_rate_C
def time_to_complete_remaining_work : ℚ := remaining_work / combined_work_rate_AC

theorem A_and_C_complete_remaining_work_in_2_point_4_days :
  time_to_complete_remaining_work = 2.4 := 
sorry

end A_and_C_complete_remaining_work_in_2_point_4_days_l103_103631


namespace roots_inverse_sum_eq_two_thirds_l103_103695

theorem roots_inverse_sum_eq_two_thirds {x₁ x₂ : ℝ} (h1 : x₁ ^ 2 + 2 * x₁ - 3 = 0) (h2 : x₂ ^ 2 + 2 * x₂ - 3 = 0) : 
  (1 / x₁) + (1 / x₂) = 2 / 3 :=
sorry

end roots_inverse_sum_eq_two_thirds_l103_103695


namespace find_k_l103_103921

variable (a b : EuclideanSpace ℝ (Fin 3)) (k : ℝ)

-- Non-collinear unit vectors
axiom h1 : ∥a∥ = 1
axiom h2 : ∥b∥ = 1
axiom h3 : a ≠ b
axiom h4 : a ≠ -b

-- Perpendicular condition
axiom h5 : InnerProductSpace.inner (a + b) (k • a - b) = 0

theorem find_k : k = 1 := by
  sorry

end find_k_l103_103921


namespace sum_proper_divisors_243_l103_103018

theorem sum_proper_divisors_243 : 
  let proper_divisors_243 := [1, 3, 9, 27, 81] in
  proper_divisors_243.sum = 121 := 
by
  sorry

end sum_proper_divisors_243_l103_103018


namespace sum_first_seven_terms_l103_103100

variable (a : ℕ → ℝ)
variable (d : ℝ)
variable (a1 a2 a3 a4 a5 a6 a7 : ℝ)

-- Assuming a is an arithmetic sequence with common difference d
axiom (h_arith_seq : ∀ n : ℕ, a (n + 1) = a n + d)

-- Given conditions
axiom (h_cond : a 3 + a 4 + a 5 = 12)

open Nat

theorem sum_first_seven_terms : a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 := by
  sorry

end sum_first_seven_terms_l103_103100


namespace mike_unbroken_seashells_l103_103127

-- Define the conditions from the problem
def totalSeashells : ℕ := 6
def brokenSeashells : ℕ := 4
def unbrokenSeashells : ℕ := totalSeashells - brokenSeashells

-- Statement to prove
theorem mike_unbroken_seashells : unbrokenSeashells = 2 := by
  sorry

end mike_unbroken_seashells_l103_103127


namespace find_new_length_l103_103853

def initial_length_cm : ℕ := 100
def erased_length_cm : ℕ := 24
def final_length_cm : ℕ := 76

theorem find_new_length : initial_length_cm - erased_length_cm = final_length_cm := by
  sorry

end find_new_length_l103_103853


namespace fixed_constant_t_l103_103324

-- Representation of point on the Cartesian plane
structure Point where
  x : ℝ
  y : ℝ

-- Definition of the parabola y = 4x^2
def parabola (p : Point) : Prop := p.y = 4 * p.x^2

-- Definition of distance squared between two points
def distance_squared (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Main theorem statement
theorem fixed_constant_t :
  ∃ (c : ℝ) (C : Point), c = 1/8 ∧ C = ⟨1, c⟩ ∧ 
  (∀ (A B : Point), parabola A ∧ parabola B ∧ 
  (∃ m k : ℝ, A.y = m * A.x + k ∧ B.y = m * B.x + k ∧ k = c - m) → 
  (1 / distance_squared A C + 1 / distance_squared B C = 16)) :=
by {
  -- Proof omitted
  sorry
}

end fixed_constant_t_l103_103324


namespace cost_of_each_toy_car_l103_103598

theorem cost_of_each_toy_car (S M C A B : ℕ) (hS : S = 53) (hM : M = 7) (hA : A = 10) (hB : B = 14) 
(hTotalSpent : S - M = C + A + B) (hTotalCars : 2 * C / 2 = 11) : 
C / 2 = 11 :=
by
  rw [hS, hM, hA, hB] at hTotalSpent
  sorry

end cost_of_each_toy_car_l103_103598


namespace completing_the_square_solution_l103_103183

theorem completing_the_square_solution : ∀ x : ℝ, x^2 + 8 * x + 9 = 0 ↔ (x + 4)^2 = 7 := by
  sorry

end completing_the_square_solution_l103_103183


namespace cos_double_angle_l103_103932

theorem cos_double_angle (α : ℝ) (h : Real.sin (Real.pi / 2 + α) = 3 / 5) : Real.cos (2 * α) = -7 / 25 :=
sorry

end cos_double_angle_l103_103932


namespace min_value_of_reciprocals_l103_103283

theorem min_value_of_reciprocals (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 3) (hz1 : z = 1) :
    1/x + 1/y + 1/z ≥ 3 :=
by
  have hz_pos : 0 < z := hz
  have hz_eq_one : z = 1 := hz1
  rw [hz_eq_one, one_div_one] at *
  sorry

end min_value_of_reciprocals_l103_103283


namespace exponent_comparison_l103_103911

theorem exponent_comparison : 1.7 ^ 0.3 > 0.9 ^ 11 := 
by sorry

end exponent_comparison_l103_103911


namespace triangle_sin_a_triangle_area_l103_103680

theorem triangle_sin_a (B : ℝ) (a b c : ℝ) (hB : B = π / 4)
  (h_bc : b = Real.sqrt 5 ∧ c = Real.sqrt 2 ∨ a = 3 ∧ c = Real.sqrt 2) :
  Real.sin A = (3 * Real.sqrt 10) / 10 :=
sorry

theorem triangle_area (B a b c : ℝ) (hB : B = π / 4) (hb : b = Real.sqrt 5)
  (h_ac : a + c = 3) : 1 / 2 * a * c * Real.sin B = Real.sqrt 2 - 1 :=
sorry

end triangle_sin_a_triangle_area_l103_103680


namespace cakes_difference_l103_103900

theorem cakes_difference (cakes_made : ℕ) (cakes_sold : ℕ) (cakes_bought : ℕ) 
  (h1 : cakes_made = 648) (h2 : cakes_sold = 467) (h3 : cakes_bought = 193) :
  (cakes_sold - cakes_bought = 274) :=
by
  sorry

end cakes_difference_l103_103900


namespace minimum_value_of_y_at_l103_103881

noncomputable def y (x : ℝ) : ℝ := x * 2^x

theorem minimum_value_of_y_at :
  ∃ x : ℝ, (∀ x' : ℝ, y x ≤ y x') ∧ x = -1 / Real.log 2 :=
by
  sorry

end minimum_value_of_y_at_l103_103881


namespace journey_distance_l103_103025

theorem journey_distance :
  ∃ D : ℝ, ((D / 2) / 21 + (D / 2) / 24 = 10) ∧ D = 224 :=
by
  use 224
  sorry

end journey_distance_l103_103025


namespace graph_of_equation_l103_103748

theorem graph_of_equation (x y : ℝ) : (x - y)^2 = x^2 + y^2 ↔ (x = 0 ∨ y = 0) :=
by
  sorry

end graph_of_equation_l103_103748


namespace b_20_value_l103_103472

-- Definitions based on conditions
def a (n : ℕ) : ℕ := 2 * n - 1

def b (n : ℕ) : ℕ := a n  -- Given that \( b_n = a_n \)

-- The theorem stating that \( b_{20} = 39 \)
theorem b_20_value : b 20 = 39 :=
by
  -- Skipping the proof
  sorry

end b_20_value_l103_103472


namespace sum_of_prime_factors_172944_l103_103878

theorem sum_of_prime_factors_172944 : 
  (∃ (a b c : ℕ), 2^a * 3^b * 1201^c = 172944 ∧ a = 4 ∧ b = 2 ∧ c = 1) → 2 + 3 + 1201 = 1206 := 
by 
  intros h 
  exact sorry

end sum_of_prime_factors_172944_l103_103878


namespace arrange_number_of_ways_l103_103213

-- Define the problem setup
def num_schools : ℕ := 10
def total_days : ℕ := 30
def larger_school_days : ℕ := 2
def other_schools : ℕ := 9

-- Calculate the number of ways to arrange the visits
noncomputable def number_of_arrangements : ℕ :=
  (29.choose 1) * (28.perm 9)

-- Statement to prove
theorem arrange_number_of_ways:
  number_of_arrangements = 
  (29.choose 1) * (28.perm 9) :=
sorry

end arrange_number_of_ways_l103_103213


namespace remainder_of_N_eq_4101_l103_103838

noncomputable def N : ℕ :=
  20 + 3^(3^(3+1) - 13)

theorem remainder_of_N_eq_4101 : N % 10000 = 4101 := by
  sorry

end remainder_of_N_eq_4101_l103_103838


namespace total_paint_is_correct_l103_103510

/-- Given conditions -/
def paint_per_large_canvas := 3
def paint_per_small_canvas := 2
def large_paintings := 3
def small_paintings := 4

/-- Define total paint used using the given conditions -/
noncomputable def total_paint_used : ℕ := 
  (paint_per_large_canvas * large_paintings) + (paint_per_small_canvas * small_paintings)

/-- Theorem statement to show the total paint used equals 17 ounces -/
theorem total_paint_is_correct : total_paint_used = 17 := by
  sorry

end total_paint_is_correct_l103_103510


namespace maximum_value_l103_103545

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem maximum_value (a b c : ℝ) (h_a : 1 ≤ a ∧ a ≤ 2)
  (h_f1 : f a b c 1 ≤ 1) (h_f2 : f a b c 2 ≤ 1) :
  7 * b + 5 * c ≤ -6 :=
sorry

end maximum_value_l103_103545


namespace complete_square_l103_103185

theorem complete_square (x : ℝ) (h : x^2 + 8 * x + 9 = 0) : (x + 4)^2 = 7 := by
  sorry

end complete_square_l103_103185


namespace rook_reaches_right_total_rook_reaches_right_seven_moves_l103_103035

-- Definition of the conditions for the problem
def rook_ways_total (n : Nat) :=
  2 ^ (n - 2)

def rook_ways_in_moves (n k : Nat) :=
  Nat.choose (n - 2) (k - 1)

-- Proof problem statements
theorem rook_reaches_right_total : rook_ways_total 30 = 2 ^ 28 := 
by sorry

theorem rook_reaches_right_seven_moves : rook_ways_in_moves 30 7 = Nat.choose 28 6 := 
by sorry

end rook_reaches_right_total_rook_reaches_right_seven_moves_l103_103035


namespace a_200_correct_l103_103868

def a_seq : ℕ → ℕ
| 0     := 1
| (n+1) := a_seq n + 2 * a_seq n / n

theorem a_200_correct : a_seq 200 = 20100 := 
by sorry

end a_200_correct_l103_103868


namespace negation_of_proposition_l103_103317

-- Definitions based on given conditions
def is_not_divisible_by_2 (n : ℤ) := n % 2 ≠ 0
def is_odd (n : ℤ) := n % 2 = 1

-- The negation proposition to be proved
theorem negation_of_proposition : ∃ n : ℤ, is_not_divisible_by_2 n ∧ ¬ is_odd n := 
sorry

end negation_of_proposition_l103_103317


namespace chef_meals_prepared_l103_103702

theorem chef_meals_prepared (S D_added D_total L R : ℕ)
  (hS : S = 12)
  (hD_added : D_added = 5)
  (hD_total : D_total = 10)
  (hR : R + D_added = D_total)
  (hL : L = S + R) : L = 17 :=
by
  sorry

end chef_meals_prepared_l103_103702


namespace range_of_a_l103_103548

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x → x - Real.log x - a > 0) → a < 1 :=
sorry

end range_of_a_l103_103548


namespace age_of_15th_student_l103_103201

theorem age_of_15th_student (avg_age_15 avg_age_3 avg_age_11 : ℕ) 
  (h_avg_15 : avg_age_15 = 15) 
  (h_avg_3 : avg_age_3 = 14) 
  (h_avg_11 : avg_age_11 = 16) : 
  ∃ x : ℕ, x = 7 := 
by
  sorry

end age_of_15th_student_l103_103201


namespace fraction_problem_l103_103423

-- Definitions given in the conditions
variables {p q r s : ℚ}
variables (h₁ : p / q = 8)
variables (h₂ : r / q = 5)
variables (h₃ : r / s = 3 / 4)

-- Statement to prove
theorem fraction_problem : s / p = 5 / 6 :=
by
  sorry

end fraction_problem_l103_103423


namespace find_Y_value_l103_103795

theorem find_Y_value : ∃ Y : ℤ, 80 - (Y - (6 + 2 * (7 - 8 - 5))) = 89 ∧ Y = -15 := by
  sorry

end find_Y_value_l103_103795


namespace staffing_battle_station_l103_103226

-- Define the qualifications
def num_assistant_engineer := 3
def num_maintenance_1 := 4
def num_maintenance_2 := 4
def num_field_technician := 5
def num_radio_specialist := 5

-- Prove the total number of ways to fill the positions
theorem staffing_battle_station : 
  num_assistant_engineer * num_maintenance_1 * num_maintenance_2 * num_field_technician * num_radio_specialist = 960 := by
  sorry

end staffing_battle_station_l103_103226


namespace down_payment_amount_l103_103275

-- Define the monthly savings per person
def monthly_savings_per_person : ℤ := 1500

-- Define the number of people
def number_of_people : ℤ := 2

-- Define the total monthly savings
def total_monthly_savings : ℤ := monthly_savings_per_person * number_of_people

-- Define the number of years they will save
def years_saving : ℤ := 3

-- Define the number of months in a year
def months_in_year : ℤ := 12

-- Define the total number of months
def total_months : ℤ := years_saving * months_in_year

-- Define the total savings needed for the down payment
def total_savings_needed : ℤ := total_monthly_savings * total_months

-- Prove that the total amount needed for the down payment is $108,000
theorem down_payment_amount : total_savings_needed = 108000 := by
  -- This part requires a proof, which we skip with sorry
  sorry

end down_payment_amount_l103_103275


namespace completing_the_square_solution_correct_l103_103172

theorem completing_the_square_solution_correct (x : ℝ) :
  (x^2 + 8 * x + 9 = 0) ↔ ((x + 4)^2 = 7) :=
by
  sorry

end completing_the_square_solution_correct_l103_103172


namespace dodgeballs_purchasable_l103_103480

-- Definitions for the given conditions
def original_budget (B : ℝ) := B
def new_budget (B : ℝ) := 1.2 * B
def cost_per_dodgeball : ℝ := 5
def cost_per_softball : ℝ := 9
def softballs_purchased (B : ℝ) := 10

-- Theorem statement
theorem dodgeballs_purchasable {B : ℝ} (h : new_budget B = 90) : original_budget B / cost_per_dodgeball = 15 := 
by 
  sorry

end dodgeballs_purchasable_l103_103480


namespace find_prism_height_l103_103837

variables (base_side_length : ℝ) (density : ℝ) (weight : ℝ) (height : ℝ)

-- Assume the base_side_length is 2 meters, density is 2700 kg/m³, and weight is 86400 kg
def given_conditions := (base_side_length = 2) ∧ (density = 2700) ∧ (weight = 86400)

-- Define the volume based on weight and density
noncomputable def volume (density weight : ℝ) : ℝ := weight / density

-- Define the area of the base
def base_area (side_length : ℝ) : ℝ := side_length * side_length

-- Define the height of the prism
noncomputable def prism_height (volume base_area : ℝ) : ℝ := volume / base_area

-- The proof statement
theorem find_prism_height (h : ℝ) : given_conditions base_side_length density weight → prism_height (volume density weight) (base_area base_side_length) = h :=
by
  intros h_cond
  sorry

end find_prism_height_l103_103837


namespace A_inter_complement_B_eq_set_minus_one_to_zero_l103_103688

open Set

theorem A_inter_complement_B_eq_set_minus_one_to_zero :
  let U := @univ ℝ
  let A := {x : ℝ | x < 0}
  let B := {x : ℝ | x ≤ -1}
  A ∩ (U \ B) = {x : ℝ | -1 < x ∧ x < 0} := 
by
  sorry

end A_inter_complement_B_eq_set_minus_one_to_zero_l103_103688


namespace right_triangle_tangent_length_l103_103831

theorem right_triangle_tangent_length (DE DF : ℝ) (h1 : DE = 7) (h2 : DF = Real.sqrt 85)
  (h3 : ∀ (EF : ℝ), DE^2 + EF^2 = DF^2 → EF = 6): FQ = 6 :=
by
  sorry

end right_triangle_tangent_length_l103_103831


namespace gain_percent_is_40_l103_103028

-- Define the conditions
def purchase_price : ℕ := 800
def repair_costs : ℕ := 200
def selling_price : ℕ := 1400

-- Define the total cost
def total_cost : ℕ := purchase_price + repair_costs

-- Define the gain
def gain : ℕ := selling_price - total_cost

-- Define the gain percent
def gain_percent : ℕ := (gain * 100) / total_cost

theorem gain_percent_is_40 : gain_percent = 40 := by
  -- Placeholder for the proof
  sorry

end gain_percent_is_40_l103_103028


namespace sarah_toads_l103_103005

theorem sarah_toads (tim_toads : ℕ) (jim_toads : ℕ) (sarah_toads : ℕ)
  (h1 : tim_toads = 30)
  (h2 : jim_toads = tim_toads + 20)
  (h3 : sarah_toads = 2 * jim_toads) :
  sarah_toads = 100 :=
by
  sorry

end sarah_toads_l103_103005


namespace range_of_m_l103_103406

variable (m : ℝ)

def p : Prop := ∀ x : ℝ, 2 * x > m * (x^2 + 1)
def q : Prop := ∃ x0 : ℝ, x0^2 + 2 * x0 - m - 1 = 0

theorem range_of_m (hp : p m) (hq : q m) : -2 ≤ m ∧ m < -1 :=
sorry

end range_of_m_l103_103406


namespace yz_zx_xy_minus_2xyz_leq_7_27_l103_103255

theorem yz_zx_xy_minus_2xyz_leq_7_27 (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) (h₃ : x + y + z = 1) :
  (y * z + z * x + x * y - 2 * x * y * z) ≤ 7 / 27 := 
by 
  sorry

end yz_zx_xy_minus_2xyz_leq_7_27_l103_103255


namespace scientific_notation_150_billion_l103_103594

theorem scientific_notation_150_billion : 150000000000 = 1.5 * 10^11 :=
sorry

end scientific_notation_150_billion_l103_103594


namespace find_smallest_n_modulo_l103_103796

theorem find_smallest_n_modulo :
  ∃ n : ℕ, n > 0 ∧ (2007 * n) % 1000 = 837 ∧ n = 691 :=
by
  sorry

end find_smallest_n_modulo_l103_103796


namespace last_donation_on_saturday_l103_103767

def total_amount : ℕ := 2010
def daily_donation : ℕ := 10
def first_day_donation : ℕ := 0 -- where 0 represents Monday, 6 represents Sunday

def total_days : ℕ := total_amount / daily_donation

def last_donation_day_of_week : ℕ := (total_days % 7 + first_day_donation) % 7

theorem last_donation_on_saturday : last_donation_day_of_week = 5 := by
  -- Prove it by calculation
  sorry

end last_donation_on_saturday_l103_103767


namespace isabella_euros_l103_103114

theorem isabella_euros (d : ℝ) : 
  (5 / 8) * d - 80 = 2 * d → d = 58 :=
by
  sorry

end isabella_euros_l103_103114


namespace expand_expression_l103_103052

variable (x y z : ℝ)

theorem expand_expression :
  (x + 8) * (3 * y + 12) * (2 * z + 4) =
  6 * x * y * z + 12 * x * z + 24 * y * z + 12 * x * y + 48 * x + 96 * y + 96 * z + 384 := 
  sorry

end expand_expression_l103_103052


namespace vectors_are_perpendicular_l103_103419

def vector_a : ℝ × ℝ := (-5, 6)
def vector_b : ℝ × ℝ := (6, 5)

theorem vectors_are_perpendicular :
  (vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2) = 0 :=
by
  sorry

end vectors_are_perpendicular_l103_103419


namespace no_solution_l103_103658

theorem no_solution (x : ℝ) : ¬ (3 * x - 2 < (x + 2)^2 ∧ (x + 2)^2 < 9 * x - 5) :=
by
  sorry

end no_solution_l103_103658


namespace instantaneous_velocity_at_2_l103_103892

def displacement (t : ℝ) : ℝ := 2 * t^3

theorem instantaneous_velocity_at_2 :
  let velocity := deriv displacement
  velocity 2 = 24 :=
by
  sorry

end instantaneous_velocity_at_2_l103_103892


namespace most_appropriate_sampling_l103_103036

def total_students := 126 + 280 + 95
def adjusted_total_students := 126 - 1 + 280 + 95
def required_sample_size := 100

def elementary_proportion (total : Nat) (sample : Nat) : Nat := (sample * 126) / total
def middle_proportion (total : Nat) (sample : Nat) : Nat := (sample * 280) / total
def high_proportion (total : Nat) (sample : Nat) : Nat := (sample * 95) / total

theorem most_appropriate_sampling :
  required_sample_size = elementary_proportion adjusted_total_students required_sample_size + 
                         middle_proportion adjusted_total_students required_sample_size + 
                         high_proportion adjusted_total_students required_sample_size :=
by
  sorry

end most_appropriate_sampling_l103_103036


namespace birds_flew_up_l103_103998

theorem birds_flew_up (original_birds total_birds birds_flew_up : ℕ) 
  (h1 : original_birds = 14)
  (h2 : total_birds = 35)
  (h3 : total_birds = original_birds + birds_flew_up) :
  birds_flew_up = 21 :=
by
  rw [h1, h2] at h3
  linarith

end birds_flew_up_l103_103998


namespace total_number_of_workers_l103_103857

theorem total_number_of_workers 
  (W : ℕ) 
  (avg_all : ℕ) 
  (n_technicians : ℕ) 
  (avg_technicians : ℕ) 
  (avg_non_technicians : ℕ) :
  avg_all * W = avg_technicians * n_technicians + avg_non_technicians * (W - n_technicians) →
  avg_all = 8000 →
  n_technicians = 7 →
  avg_technicians = 12000 →
  avg_non_technicians = 6000 →
  W = 21 :=
by 
  intro h1 h2 h3 h4 h5
  sorry

end total_number_of_workers_l103_103857


namespace teacher_age_is_94_5_l103_103337

noncomputable def avg_age_students : ℝ := 18
noncomputable def num_students : ℝ := 50
noncomputable def avg_age_class_with_teacher : ℝ := 19.5
noncomputable def num_total : ℝ := 51

noncomputable def total_age_students : ℝ := num_students * avg_age_students
noncomputable def total_age_class_with_teacher : ℝ := num_total * avg_age_class_with_teacher

theorem teacher_age_is_94_5 : ∃ T : ℝ, total_age_students + T = total_age_class_with_teacher ∧ T = 94.5 := by
  sorry

end teacher_age_is_94_5_l103_103337


namespace division_sequence_l103_103984

theorem division_sequence : (120 / 5) / 2 / 3 = 4 := by
  sorry

end division_sequence_l103_103984


namespace distance_between_foci_of_ellipse_l103_103403

theorem distance_between_foci_of_ellipse :
  ∀ x y : ℝ,
  9 * x^2 - 36 * x + 4 * y^2 + 16 * y + 16 = 0 →
  2 * Real.sqrt (9 - 4) = 2 * Real.sqrt 5 :=
by 
  sorry

end distance_between_foci_of_ellipse_l103_103403


namespace find_y_squared_l103_103751

theorem find_y_squared (x y : ℤ) (h1 : 4 * x + y = 34) (h2 : 2 * x - y = 20) : y ^ 2 = 4 := 
sorry

end find_y_squared_l103_103751


namespace kilometers_to_chains_l103_103252

theorem kilometers_to_chains :
  (1 * 10 * 50 = 500) :=
by
  sorry

end kilometers_to_chains_l103_103252


namespace probability_same_color_opposite_feet_l103_103441

/-- Define the initial conditions: number of pairs of each color. -/
def num_black_pairs : ℕ := 8
def num_brown_pairs : ℕ := 4
def num_gray_pairs : ℕ := 3
def num_red_pairs : ℕ := 1

/-- The total number of shoes. -/
def total_shoes : ℕ := 2 * (num_black_pairs + num_brown_pairs + num_gray_pairs + num_red_pairs)

theorem probability_same_color_opposite_feet :
  ((num_black_pairs * (num_black_pairs - 1)) + 
   (num_brown_pairs * (num_brown_pairs - 1)) + 
   (num_gray_pairs * (num_gray_pairs - 1)) + 
   (num_red_pairs * (num_red_pairs - 1))) * 2 / (total_shoes * (total_shoes - 1)) = 45 / 248 :=
by sorry

end probability_same_color_opposite_feet_l103_103441


namespace inequality_positive_numbers_l103_103132

theorem inequality_positive_numbers (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) : 
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 := 
sorry

end inequality_positive_numbers_l103_103132


namespace average_chemistry_mathematics_l103_103160

noncomputable def marks (P C M B : ℝ) : Prop := 
  P + C + M + B = (P + B) + 180 ∧ P = 1.20 * B

theorem average_chemistry_mathematics 
  (P C M B : ℝ) (h : marks P C M B) : (C + M) / 2 = 90 :=
by
  sorry

end average_chemistry_mathematics_l103_103160


namespace range_of_m_l103_103103

theorem range_of_m (x y m : ℝ) 
  (h1 : x - 2 * y = 1) 
  (h2 : 2 * x + y = 4 * m) 
  (h3 : x + 3 * y < 6) : 
  m < 7 / 4 := 
sorry

end range_of_m_l103_103103


namespace area_per_cabbage_is_one_l103_103346

noncomputable def area_per_cabbage (x y : ℕ) : ℕ :=
  let num_cabbages_this_year : ℕ := 10000
  let increase_in_cabbages : ℕ := 199
  let area_this_year : ℕ := y^2
  let area_last_year : ℕ := x^2
  let area_per_cabbage : ℕ := area_this_year / num_cabbages_this_year
  area_per_cabbage

theorem area_per_cabbage_is_one (x y : ℕ) (hx : y^2 = 10000) (hy : y^2 = x^2 + 199) : area_per_cabbage x y = 1 :=
by 
  sorry

end area_per_cabbage_is_one_l103_103346


namespace buildingC_floors_if_five_times_l103_103903

-- Defining the number of floors in Building B
def floorsBuildingB : ℕ := 13

-- Theorem to prove the number of floors in Building C if it had five times as many floors as Building B
theorem buildingC_floors_if_five_times (FB : ℕ) (h : FB = floorsBuildingB) : (5 * FB) = 65 :=
by
  rw [h]
  exact rfl

end buildingC_floors_if_five_times_l103_103903


namespace volume_conversion_l103_103366

theorem volume_conversion (V_ft : ℕ) (h_V : V_ft = 216) (conversion_factor : ℕ) (h_cf : conversion_factor = 27) :
  V_ft / conversion_factor = 8 :=
by
  sorry

end volume_conversion_l103_103366


namespace net_amount_spent_correct_l103_103843

def trumpet_cost : ℝ := 145.16
def song_book_revenue : ℝ := 5.84
def net_amount_spent : ℝ := 139.32

theorem net_amount_spent_correct : trumpet_cost - song_book_revenue = net_amount_spent :=
by
  sorry

end net_amount_spent_correct_l103_103843


namespace largest_three_digit_number_l103_103331

theorem largest_three_digit_number (a b c : ℕ) (h1 : a = 8) (h2 : b = 0) (h3 : c = 7) :
  ∃ (n : ℕ), ∀ (x : ℕ), (x = a * 100 + b * 10 + c) → x = 870 :=
by
  sorry

end largest_three_digit_number_l103_103331


namespace parity_of_f_min_value_of_f_min_value_of_f_l103_103839

open Real

def f (a x : ℝ) := x^2 + abs (x - a) + 1

theorem parity_of_f (a : ℝ) :
  (∀ x : ℝ, f 0 x = f 0 (-x)) ∧ (∀ x : ℝ, f a x ≠ f a (-x) ∧ f a x ≠ -f a x) ↔ a = 0 :=
by sorry

theorem min_value_of_f (a : ℝ) (h : a ≤ -1/2) : 
  ∀ x : ℝ, x ≥ a → f a x ≥ f a (-1/2) :=
by sorry

theorem min_value_of_f' (a : ℝ) (h : -1/2 < a) : 
  ∀ x : ℝ, x ≥ a → f a x ≥ f a a :=
by sorry

end parity_of_f_min_value_of_f_min_value_of_f_l103_103839


namespace largest_sum_of_digits_l103_103556

theorem largest_sum_of_digits (a b c : ℕ) (y : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : 0 ≤ c ∧ c ≤ 9) (h4 : 1 ≤ y ∧ y ≤ 10) (h5 : (1000 * (a * 100 + b * 10 + c)) = 1000) : 
  a + b + c = 8 :=
sorry

end largest_sum_of_digits_l103_103556


namespace sin_beta_l103_103536

variable (α β : ℝ)
variable (hα1 : 0 < α) (hα2 : α < Real.pi / 2)
variable (hβ1 : 0 < β) (hβ2: β < Real.pi / 2)
variable (h1 : Real.cos α = 5 / 13)
variable (h2 : Real.sin (α - β) = 4 / 5)

theorem sin_beta (α β : ℝ) (hα1 : 0 < α) (hα2 : α < Real.pi / 2) 
  (hβ1 : 0 < β) (hβ2 : β < Real.pi / 2) 
  (h1 : Real.cos α = 5 / 13) 
  (h2 : Real.sin (α - β) = 4 / 5) : 
  Real.sin β = 16 / 65 := 
by 
  sorry

end sin_beta_l103_103536


namespace eggs_left_in_box_l103_103000

theorem eggs_left_in_box (initial_eggs : ℕ) (taken_eggs : ℕ) (remaining_eggs : ℕ) : 
  initial_eggs = 47 → taken_eggs = 5 → remaining_eggs = initial_eggs - taken_eggs → remaining_eggs = 42 :=
by
  sorry

end eggs_left_in_box_l103_103000


namespace vincent_total_cost_l103_103335

theorem vincent_total_cost :
  let day1_packs := 15
  let day1_pack_cost := 2.50
  let discount_percent := 0.10
  let day2_packs := 25
  let day2_pack_cost := 3.00
  let tax_percent := 0.05
  let day1_total_cost_before_discount := day1_packs * day1_pack_cost
  let day1_discount_amount := discount_percent * day1_total_cost_before_discount
  let day1_total_cost_after_discount := day1_total_cost_before_discount - day1_discount_amount
  let day2_total_cost_before_tax := day2_packs * day2_pack_cost
  let day2_tax_amount := tax_percent * day2_total_cost_before_tax
  let day2_total_cost_after_tax := day2_total_cost_before_tax + day2_tax_amount
  let total_cost := day1_total_cost_after_discount + day2_total_cost_after_tax
  total_cost = 112.50 :=
by 
  -- Mathlib can be used for floating point calculations, if needed
  -- For the purposes of this example, we assume calculations are correct.
  sorry

end vincent_total_cost_l103_103335


namespace possible_values_of_m_l103_103924

def f (x a m : ℝ) := abs (x - a) + m * abs (x + a)

theorem possible_values_of_m {a m : ℝ} (h1 : 0 < m) (h2 : m < 1) 
  (h3 : ∀ x : ℝ, f x a m ≥ 2)
  (h4 : a ≤ -5 ∨ a ≥ 5) : m = 1 / 5 :=
by 
  sorry

end possible_values_of_m_l103_103924


namespace exists_separating_line_l103_103964

noncomputable def separate_convex_figures (Φ₁ Φ₂ : Set ℝ²) : Prop :=
  Bounded Φ₁ ∧ Bounded Φ₂ ∧ Convex ℝ Φ₁ ∧ Convex ℝ Φ₂ ∧
  Disjoint Φ₁ Φ₂ → 
  ∃ l : AffineSubspace ℝ ℝ², (∃ (H₁ : Φ₁ ⊆ HalfSpace l ∧ Φ₂ ⊆ -HalfSpace l))

theorem exists_separating_line (Φ₁ Φ₂ : Set ℝ²) :
  separate_convex_figures Φ₁ Φ₂ :=
sorry

end exists_separating_line_l103_103964


namespace money_distribution_l103_103495

theorem money_distribution (a b c : ℝ) (h1 : 4 * (a - b - c) = 16)
                           (h2 : 6 * b - 2 * a - 2 * c = 16)
                           (h3 : 7 * c - a - b = 16) :
  a = 29 := 
by 
  sorry

end money_distribution_l103_103495


namespace rebate_percentage_l103_103710

theorem rebate_percentage (r : ℝ) (h1 : 0 ≤ r) (h2 : r ≤ 1) 
(h3 : (6650 - 6650 * r) * 1.10 = 6876.1) : r = 0.06 :=
sorry

end rebate_percentage_l103_103710


namespace min_value_of_f_min_value_achieved_min_value_f_l103_103407

noncomputable def f (x : ℝ) := x + 2 / (2 * x + 1) - 1

theorem min_value_of_f : ∀ x : ℝ, x > 0 → f x ≥ 1/2 := 
by sorry

theorem min_value_achieved : f (1/2) = 1/2 := 
by sorry

theorem min_value_f : ∃ x : ℝ, x > 0 ∧ f x = 1/2 := 
⟨1/2, by norm_num, by sorry⟩

end min_value_of_f_min_value_achieved_min_value_f_l103_103407


namespace gcd_f_of_x_and_x_l103_103413

theorem gcd_f_of_x_and_x (x : ℕ) (hx : 7200 ∣ x) :
  Nat.gcd ((5 * x + 6) * (8 * x + 3) * (11 * x + 9) * (4 * x + 12)) x = 72 :=
sorry

end gcd_f_of_x_and_x_l103_103413


namespace equal_number_of_boys_and_girls_l103_103107

theorem equal_number_of_boys_and_girls
  (m d M D : ℝ)
  (hm : m ≠ 0)
  (hd : d ≠ 0)
  (avg1 : M / m ≠ D / d)
  (avg2 : (M / m + D / d) / 2 = (M + D) / (m + d)) :
  m = d :=
by
  sorry

end equal_number_of_boys_and_girls_l103_103107


namespace expression_is_product_l103_103314

def not_sum (a x : Int) : Prop :=
  ¬(a + x = -7 * x)

def not_difference (a x : Int) : Prop :=
  ¬(a - x = -7 * x)

def not_quotient (a x : Int) : Prop :=
  ¬(a / x = -7 * x)

theorem expression_is_product (x : Int) : 
  not_sum (-7) x ∧ not_difference (-7) x ∧ not_quotient (-7) x → (-7 * x = -7 * x) :=
by sorry

end expression_is_product_l103_103314


namespace total_plates_used_l103_103874

-- Definitions from the conditions
def number_of_people := 6
def meals_per_day_per_person := 3
def plates_per_meal_per_person := 2
def number_of_days := 4

-- Statement of the theorem
theorem total_plates_used : number_of_people * meals_per_day_per_person * plates_per_meal_per_person * number_of_days = 144 := 
by
  sorry

end total_plates_used_l103_103874


namespace complete_the_square_l103_103179

theorem complete_the_square (x : ℝ) :
  (x^2 + 8*x + 9 = 0) ↔ ((x + 4)^2 = 7) :=
sorry

end complete_the_square_l103_103179


namespace system_of_equations_abs_diff_l103_103687

theorem system_of_equations_abs_diff 
  (x y m n : ℝ) 
  (h₁ : 2 * x - y = m)
  (h₂ : x + m * y = n)
  (hx : x = 2)
  (hy : y = 1) : 
  |m - n| = 2 :=
by
  sorry

end system_of_equations_abs_diff_l103_103687


namespace volume_conversion_l103_103367

theorem volume_conversion (V_ft : ℕ) (h_V : V_ft = 216) (conversion_factor : ℕ) (h_cf : conversion_factor = 27) :
  V_ft / conversion_factor = 8 :=
by
  sorry

end volume_conversion_l103_103367


namespace cargo_arrival_day_l103_103632

-- Definitions based on conditions
def navigation_days : Nat := 21
def customs_days : Nat := 4
def warehouse_days_from_today : Nat := 2
def departure_days_ago : Nat := 30

-- Definition represents the total transit time
def total_transit_days : Nat := navigation_days + customs_days + warehouse_days_from_today

-- Theorem to prove the cargo always arrives at the rural warehouse 1 day after leaving the port in Vancouver
theorem cargo_arrival_day : 
  (departure_days_ago - total_transit_days + warehouse_days_from_today = 1) :=
by
  -- Placeholder for the proof
  sorry

end cargo_arrival_day_l103_103632


namespace turner_total_tickets_l103_103617

-- Definition of conditions
def days := 3
def rollercoaster_rides_per_day := 3
def catapult_rides_per_day := 2
def ferris_wheel_rides_per_day := 1

def rollercoaster_ticket_cost := 4
def catapult_ticket_cost := 4
def ferris_wheel_ticket_cost := 1

-- Proof statement
theorem turner_total_tickets : 
  days * (rollercoaster_rides_per_day * rollercoaster_ticket_cost 
  + catapult_rides_per_day * catapult_ticket_cost 
  + ferris_wheel_rides_per_day * ferris_wheel_ticket_cost) 
  = 63 := 
by
  sorry

end turner_total_tickets_l103_103617


namespace new_rate_ratio_l103_103512

/--
Hephaestus charged 3 golden apples for the first six months and raised his rate halfway through the year.
Apollo paid 54 golden apples in total for the entire year.
The ratio of the new rate to the old rate is 2.
-/
theorem new_rate_ratio
  (old_rate new_rate : ℕ)
  (total_payment : ℕ)
  (H1 : old_rate = 3)
  (H2 : total_payment = 54)
  (H3 : ∀ R : ℕ, new_rate = R * old_rate ∧ total_payment = 18 + 18 * R) :
  ∃ (R : ℕ), R = 2 :=
by {
  sorry
}

end new_rate_ratio_l103_103512


namespace blue_marble_difference_l103_103737

theorem blue_marble_difference :
  ∃ a b : ℕ, (10 * a = 10 * b) ∧ (3 * a + b = 80) ∧ (7 * a - 9 * b = 40) := by
  sorry

end blue_marble_difference_l103_103737


namespace absolute_value_is_four_l103_103101

-- Given condition: the absolute value of a number equals 4
theorem absolute_value_is_four (x : ℝ) : abs x = 4 → (x = 4 ∨ x = -4) :=
by
  sorry

end absolute_value_is_four_l103_103101


namespace time_until_next_consecutive_increasing_time_l103_103208

def is_valid_consecutive_increasing_time (h : ℕ) (m : ℕ) : Prop :=
  let digits := List.map (fin_to_int) ([h / 10, h % 10, m / 10, m % 10] : List (Fin 10))
  List.sorted Nat.lt digits ∧ List.pairwise Nat.succ digits

theorem time_until_next_consecutive_increasing_time :
  let current_hour := 4
  let current_minute := 56
  let next_valid_hour := 12
  let next_valid_minute := 34
  (next_valid_hour * 60 + next_valid_minute) - (current_hour * 60 + current_minute) = 458 :=
by sorry

end time_until_next_consecutive_increasing_time_l103_103208


namespace number_of_ways_to_choose_bases_l103_103671

-- Definitions of the conditions
def num_students : Nat := 4
def num_bases : Nat := 3

-- The main statement that we need to prove
theorem number_of_ways_to_choose_bases : (num_bases ^ num_students) = 81 := by
  sorry

end number_of_ways_to_choose_bases_l103_103671


namespace cost_price_as_percentage_l103_103151

theorem cost_price_as_percentage (SP CP : ℝ) 
  (profit_percentage : ℝ := 4.166666666666666) 
  (P : ℝ := SP - CP)
  (profit_eq : P = (profit_percentage / 100) * SP) :
  CP = (95.83333333333334 / 100) * SP := 
by
  sorry

end cost_price_as_percentage_l103_103151


namespace sufficiency_but_not_necessary_l103_103087

theorem sufficiency_but_not_necessary (x y : ℝ) : |x| + |y| ≤ 1 → x^2 + y^2 ≤ 1 ∧ ¬(x^2 + y^2 ≤ 1 → |x| + |y| ≤ 1) :=
by
  sorry

end sufficiency_but_not_necessary_l103_103087


namespace abc_inequality_l103_103955

theorem abc_inequality 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) 
  (h_abc : a * b * c = 1) : 
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := 
by
  sorry

end abc_inequality_l103_103955


namespace muffin_sum_l103_103116

theorem muffin_sum (N : ℕ) : 
  (N % 13 = 3) → 
  (N % 8 = 5) → 
  (N < 120) → 
  (N = 16 ∨ N = 81 ∨ N = 107) → 
  (16 + 81 + 107 = 204) := 
by sorry

end muffin_sum_l103_103116


namespace student_distribution_l103_103048

open Finset Nat

theorem student_distribution :
  let n := 7
  count_combine (choose n 2 + choose n 3) 2 = 112 :=
by
  sorry

end student_distribution_l103_103048


namespace incorrect_option_l103_103799

-- Definitions and conditions from the problem
def p (x : ℝ) : Prop := (x - 2) * Real.sqrt (x^2 - 3*x + 2) ≥ 0
def q (k : ℝ) : Prop := ∀ x : ℝ, k * x^2 - k * x - 1 < 0

-- The Lean 4 statement to verify the problem
theorem incorrect_option :
  (¬ ∃ x, p x) ∧ (∃ k, q k) ∧
  (∀ k, -4 < k ∧ k ≤ 0 → q k) →
  (∃ x, ¬p x) :=
  by
  sorry

end incorrect_option_l103_103799


namespace selling_price_l103_103217

theorem selling_price (cost_price profit_percentage selling_price : ℝ) (h1 : cost_price = 86.95652173913044)
  (h2 : profit_percentage = 0.15) : 
  selling_price = 100 :=
by
  sorry

end selling_price_l103_103217


namespace tan_theta_minus_pi_over_4_l103_103414

theorem tan_theta_minus_pi_over_4 
  (θ : Real) (h1 : π / 2 < θ ∧ θ < 2 * π)
  (h2 : Real.sin (θ + π / 4) = -3 / 5) :
  Real.tan (θ - π / 4) = 4 / 3 := 
  sorry

end tan_theta_minus_pi_over_4_l103_103414


namespace tan_theta_eq_one_half_l103_103714

theorem tan_theta_eq_one_half
  (θ : ℝ)
  (h1 : 0 < θ ∧ θ < real.pi / 2)
  (a : ℝ × ℝ)
  (ha : a = (real.cos θ, 2))
  (b : ℝ × ℝ)
  (hb : b = (-1, real.sin θ))
  (h2 : a.1 * b.1 + a.2 * b.2 = 0) :
  real.tan θ = 1 / 2 := 
sorry

end tan_theta_eq_one_half_l103_103714


namespace kem_hourly_wage_l103_103136

theorem kem_hourly_wage (shem_total_earnings: ℝ) (shem_hours_worked: ℝ) (ratio: ℝ)
  (h1: shem_total_earnings = 80)
  (h2: shem_hours_worked = 8)
  (h3: ratio = 2.5) :
  (shem_total_earnings / shem_hours_worked) / ratio = 4 :=
by 
  sorry

end kem_hourly_wage_l103_103136


namespace smallest_positive_solution_l103_103621

theorem smallest_positive_solution :
  ∃ x : ℝ, x > 0 ∧ (x ^ 4 - 50 * x ^ 2 + 576 = 0) ∧ (∀ y : ℝ, y > 0 ∧ y ^ 4 - 50 * y ^ 2 + 576 = 0 → x ≤ y) ∧ x = 3 * Real.sqrt 2 :=
sorry

end smallest_positive_solution_l103_103621


namespace distribute_students_l103_103045

theorem distribute_students:
  (∃ f : Fin 7 → Fin 2, (∀ i, f i = 0 ∨ f i = 1) ∧ 
  (∑ i, if f i = 0 then 1 else 0) ≥ 2 ∧ 
  (∑ i, if f i = 1 then 1 else 0) ≥ 2) → 
  (∃ d_choices: ℕ, d_choices = (Nat.choose 7 2 + Nat.choose 7 3 + Nat.choose 7 4 + Nat.choose 7 5) ∧ d_choices = 112) := 
by
  sorry

end distribute_students_l103_103045


namespace probability_of_top_card_heart_l103_103520

-- Define the total number of cards in the deck.
def total_cards : ℕ := 39

-- Define the number of hearts in the deck.
def hearts : ℕ := 13

-- Define the probability that the top card is a heart.
def probability_top_card_heart : ℚ := hearts / total_cards

-- State the theorem to prove.
theorem probability_of_top_card_heart : probability_top_card_heart = 1 / 3 :=
by
  sorry

end probability_of_top_card_heart_l103_103520


namespace range_of_m_l103_103428

-- Defining the quadratic function with the given condition
def quadratic (m : ℝ) (x : ℝ) : ℝ := (m-1)*x^2 + (m-1)*x + 2

-- Stating the problem
theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, quadratic m x > 0) ↔ 1 ≤ m ∧ m < 9 :=
by
  sorry

end range_of_m_l103_103428


namespace sum_proper_divisors_243_l103_103013

theorem sum_proper_divisors_243 : (1 + 3 + 9 + 27 + 81) = 121 := by
  sorry

end sum_proper_divisors_243_l103_103013


namespace product_of_five_consecutive_numbers_not_square_l103_103303

theorem product_of_five_consecutive_numbers_not_square (a b c d e : ℕ)
  (ha : a > 0) (hb : b = a + 1) (hc : c = b + 1) (hd : d = c + 1) (he : e = d + 1) :
  ¬ ∃ k : ℕ, a * b * c * d * e = k^2 := by
  sorry

end product_of_five_consecutive_numbers_not_square_l103_103303


namespace relatively_prime_pair_count_l103_103401

theorem relatively_prime_pair_count :
  (∃ m n : ℕ, m > 0 ∧ n > 0 ∧ m + n = 190 ∧ Nat.gcd m n = 1) →
  (∃! k : ℕ, k = 26) :=
by
  sorry

end relatively_prime_pair_count_l103_103401


namespace solve_for_a_l103_103092

theorem solve_for_a (a : ℝ) (h : 2 * a + (1 - 4 * a) = 0) : a = 1 / 2 :=
sorry

end solve_for_a_l103_103092


namespace roots_difference_is_one_l103_103727

noncomputable def quadratic_eq (p : ℝ) :=
  ∃ (α β : ℝ), (α ≠ β) ∧ (α - β = 1) ∧ (α ^ 2 - p * α + (p ^ 2 - 1) / 4 = 0) ∧ (β ^ 2 - p * β + (p ^ 2 - 1) / 4 = 0)

theorem roots_difference_is_one (p : ℝ) : quadratic_eq p :=
  sorry

end roots_difference_is_one_l103_103727


namespace jerky_fulfillment_time_l103_103009

-- Defining the conditions
def bags_per_batch : ℕ := 10
def order_quantity : ℕ := 60
def existing_bags : ℕ := 20
def batch_time_in_nights : ℕ := 1

-- Define the proposition to be proved
theorem jerky_fulfillment_time :
  let additional_bags := order_quantity - existing_bags in
  let batches_needed := additional_bags / bags_per_batch in
  let days_needed := batches_needed * batch_time_in_nights in
  days_needed = 4 :=
begin
  sorry
end

end jerky_fulfillment_time_l103_103009


namespace area_percentage_l103_103102

theorem area_percentage (D_S D_R : ℝ) (h : D_R = 0.8 * D_S) : 
  let R_S := D_S / 2
  let R_R := D_R / 2
  let A_S := π * R_S^2
  let A_R := π * R_R^2
  (A_R / A_S) * 100 = 64 := 
by
  sorry

end area_percentage_l103_103102


namespace min_sum_of_segments_l103_103939

theorem min_sum_of_segments 
  (X Y Z P Q: Point) 
  (angle_XYZ : ∠XYZ = 50) 
  (XY_eq : dist X Y = 8) 
  (XZ_eq : dist X Z = 10)
  (P_on_XY : PointOnSegment P X Y)
  (Q_on_XZ : PointOnSegment Q X Z): 
  ∃ P Q, min (dist Y P + dist P Q + dist Q Z) = √(164 + 80 * √3) := 
by 
  sorry

end min_sum_of_segments_l103_103939


namespace distribute_students_l103_103046

theorem distribute_students:
  (∃ f : Fin 7 → Fin 2, (∀ i, f i = 0 ∨ f i = 1) ∧ 
  (∑ i, if f i = 0 then 1 else 0) ≥ 2 ∧ 
  (∑ i, if f i = 1 then 1 else 0) ≥ 2) → 
  (∃ d_choices: ℕ, d_choices = (Nat.choose 7 2 + Nat.choose 7 3 + Nat.choose 7 4 + Nat.choose 7 5) ∧ d_choices = 112) := 
by
  sorry

end distribute_students_l103_103046


namespace fraction_simplification_l103_103085

theorem fraction_simplification 
  (d e f : ℝ) 
  (h : d + e + f ≠ 0) : 
  (d^2 + e^2 - f^2 + 2 * d * e) / (d^2 + f^2 - e^2 + 3 * d * f) = (d + e - f) / (d + f - e) :=
sorry

end fraction_simplification_l103_103085


namespace subtract_045_from_3425_l103_103190

theorem subtract_045_from_3425 : 34.25 - 0.45 = 33.8 :=
by sorry

end subtract_045_from_3425_l103_103190


namespace rectangle_area_ratio_k_l103_103866

theorem rectangle_area_ratio_k (d : ℝ) (l w : ℝ) (h1 : l / w = 5 / 2) (h2 : d^2 = l^2 + w^2) :
  ∃ k : ℝ, k = 10 / 29 ∧ (l * w = k * d^2) :=
by {
  -- proof steps will go here
  sorry
}

end rectangle_area_ratio_k_l103_103866


namespace selling_price_when_profit_equals_loss_l103_103479

theorem selling_price_when_profit_equals_loss (CP SP Rs_57 : ℕ) (h1: CP = 50) (h2: Rs_57 = 57) (h3: Rs_57 - CP = CP - SP) : 
  SP = 43 := by
  sorry

end selling_price_when_profit_equals_loss_l103_103479


namespace combined_work_time_l103_103627

def ajay_completion_time : ℕ := 8
def vijay_completion_time : ℕ := 24

theorem combined_work_time (T_A T_V : ℕ) (h1 : T_A = ajay_completion_time) (h2 : T_V = vijay_completion_time) :
  1 / (1 / (T_A : ℝ) + 1 / (T_V : ℝ)) = 6 :=
by
  rw [h1, h2]
  sorry

end combined_work_time_l103_103627


namespace largest_prime_m_satisfying_quadratic_inequality_l103_103794

theorem largest_prime_m_satisfying_quadratic_inequality :
  ∃ (m : ℕ), m = 5 ∧ m^2 - 11 * m + 28 < 0 ∧ Prime m :=
by sorry

end largest_prime_m_satisfying_quadratic_inequality_l103_103794


namespace moli_initial_payment_l103_103589

variable (R C S M : ℕ)

-- Conditions
def condition1 : Prop := 3 * R + 7 * C + 1 * S = M
def condition2 : Prop := 4 * R + 10 * C + 1 * S = 164
def condition3 : Prop := 1 * R + 1 * C + 1 * S = 32

theorem moli_initial_payment : condition1 R C S M ∧ condition2 R C S ∧ condition3 R C S → M = 120 := by
  sorry

end moli_initial_payment_l103_103589


namespace fraction_power_zero_l103_103876

variable (a b : ℤ)
variable (h_a : a ≠ 0) (h_b : b ≠ 0)

theorem fraction_power_zero : (a / b)^0 = 1 := by
  sorry

end fraction_power_zero_l103_103876


namespace max_magnitude_value_is_4_l103_103928

noncomputable def max_value_vector_magnitude (θ : ℝ) : ℝ :=
  let a := (Real.cos θ, Real.sin θ)
  let b := (Real.sqrt 3, -1)
  let vector := (2 * a.1 - b.1, 2 * a.2 + 1)
  Real.sqrt (vector.1 ^ 2 + vector.2 ^ 2)

theorem max_magnitude_value_is_4 (θ : ℝ) : 
  ∃ θ : ℝ, max_value_vector_magnitude θ = 4 :=
sorry

end max_magnitude_value_is_4_l103_103928


namespace total_surface_area_correct_l103_103141

def six_cubes_surface_area : ℕ :=
  let cube_edge := 1
  let cubes := 6
  let initial_surface_area := 6 * cubes -- six faces per cube, total initial surface area
  let hidden_faces := 10 -- determined by counting connections
  initial_surface_area - hidden_faces

theorem total_surface_area_correct : six_cubes_surface_area = 26 := by
  sorry

end total_surface_area_correct_l103_103141


namespace expression_divisible_by_19_l103_103131

theorem expression_divisible_by_19 (n : ℕ) (h : n > 0) : 
  19 ∣ (5^(2*n - 1) + 3^(n - 2) * 2^(n - 1)) := 
by 
  sorry

end expression_divisible_by_19_l103_103131


namespace hexagonal_H5_find_a_find_t_find_m_l103_103778

section problem1

-- Define the hexagonal number formula
def hexagonal_number (n : ℕ) : ℕ :=
  2 * n^2 - n

-- Define that H_5 should equal 45
theorem hexagonal_H5 : hexagonal_number 5 = 45 := sorry

end problem1

section problem2

variables (a b c : ℕ)

-- Given hexagonal number equations
def H1 := a + b + c
def H2 := 4 * a + 2 * b + c
def H3 := 9 * a + 3 * b + c

-- Conditions given in problem
axiom H1_def : H1 = 1
axiom H2_def : H2 = 7
axiom H3_def : H3 = 19

-- Prove that a = 3
theorem find_a : a = 3 := sorry

end problem2

section problem3

variables (p q r t : ℕ)

-- Given ratios in problem
axiom ratio1 : p * 3 = 2 * q
axiom ratio2 : q * 5 = 4 * r

-- Prove that t = 12
theorem find_t : t = 12 := sorry

end problem3

section problem4

variables (x y m : ℕ)

-- Given proportional conditions
axiom ratio3 : x * 3 = y * 4
axiom ratio4 : (x + y) * 3 = x * m

-- Prove that m = 7
theorem find_m : m = 7 := sorry

end problem4

end hexagonal_H5_find_a_find_t_find_m_l103_103778


namespace rons_siblings_product_l103_103553

theorem rons_siblings_product
  (H_sisters : ℕ)
  (H_brothers : ℕ)
  (Ha_sisters : ℕ)
  (Ha_brothers : ℕ)
  (R_sisters : ℕ)
  (R_brothers : ℕ)
  (Harry_cond : H_sisters = 4 ∧ H_brothers = 6)
  (Harriet_cond : Ha_sisters = 4 ∧ Ha_brothers = 6)
  (Ron_cond_sisters : R_sisters = Ha_sisters)
  (Ron_cond_brothers : R_brothers = Ha_brothers + 2)
  : R_sisters * R_brothers = 32 := by
  sorry

end rons_siblings_product_l103_103553


namespace monomial_coeff_degree_product_l103_103681

theorem monomial_coeff_degree_product (m n : ℚ) (h₁ : m = -3/4) (h₂ : n = 4) : m * n = -3 := 
by
  sorry

end monomial_coeff_degree_product_l103_103681


namespace a_works_less_than_b_l103_103641

theorem a_works_less_than_b (A B : ℝ) (x y : ℝ)
  (h1 : A = 3 * B)
  (h2 : (A + B) * 22.5 = A * x)
  (h3 : y = 3 * x) :
  y - x = 60 :=
by sorry

end a_works_less_than_b_l103_103641


namespace find_a200_l103_103869

def seq (a : ℕ → ℕ) : Prop :=
a 1 = 1 ∧ ∀ n ≥ 1, a (n + 1) = a n + 2 * a n / n

theorem find_a200 (a : ℕ → ℕ) (h : seq a) : a 200 = 20100 :=
sorry

end find_a200_l103_103869


namespace triangle_perfect_square_l103_103373

theorem triangle_perfect_square (a b c : ℤ) (h : ∃ h₁ h₂ h₃ : ℤ, (1/2) * a * h₁ = (1/2) * b * h₂ ∧ (1/2) * b * h₂ = (1/2) * c * h₃ ∧ (h₁ = h₂ + h₃)) :
  ∃ k : ℤ, a^2 + b^2 + c^2 = k^2 :=
by
  sorry

end triangle_perfect_square_l103_103373


namespace minimum_a_condition_l103_103696

-- Define the quadratic function
def f (a x : ℝ) := x^2 + a * x + 1

-- Define the condition that the function remains non-negative in the open interval (0, 1/2)
def f_non_negative_in_interval (a : ℝ) : Prop :=
  ∀ (x : ℝ), 0 < x ∧ x < 1 / 2 → f a x ≥ 0

-- State the theorem that the minimum value for a with the given condition is -5/2
theorem minimum_a_condition : ∀ (a : ℝ), f_non_negative_in_interval a → a ≥ -5 / 2 :=
by sorry

end minimum_a_condition_l103_103696


namespace tony_pool_filling_time_l103_103115

theorem tony_pool_filling_time
  (J S T : ℝ)
  (hJ : J = 1 / 30)
  (hS : S = 1 / 45)
  (hCombined : J + S + T = 1 / 15) :
  T = 1 / 90 :=
by
  -- the setup for proof would be here
  sorry

end tony_pool_filling_time_l103_103115


namespace probability_of_friends_in_same_lunch_group_l103_103309

theorem probability_of_friends_in_same_lunch_group :
  let groups := 4
  let students := 720
  let group_size := students / groups
  let probability := (1 / groups) * (1 / groups) * (1 / groups)
  students % groups = 0 ->  -- Students can be evenly divided into groups
  groups > 0 ->             -- There is at least one group
  probability = (1 : ℝ) / 64 :=
by
  intros
  sorry

end probability_of_friends_in_same_lunch_group_l103_103309


namespace age_of_hospital_l103_103563

theorem age_of_hospital (grant_current_age : ℕ) (future_ratio : ℚ)
                        (grant_future_age : grant_current_age + 5 = 30)
                        (hospital_age_ratio : future_ratio = 2 / 3) :
                        (grant_current_age = 25) → 
                        (grant_current_age + 5 = future_ratio * (grant_current_age + 5 + 5)) →
                        (grant_current_age + 5 + 5 - 5 = 40) :=
by
  sorry

end age_of_hospital_l103_103563


namespace min_positive_period_pi_not_center_of_symmetry_not_axis_of_symmetry_monotonicity_interval_l103_103260
   
noncomputable def f (x : ℝ) := 2 * sin x * cos x + 2 * sqrt 3 * (sin x)^2

theorem min_positive_period_pi : ∃ T > 0, (∀ x : ℝ, f (x + T) = f x) ∧ T = π := sorry

theorem not_center_of_symmetry : ¬ (∃ (c : ℝ) (d : ℝ), c = π / 6 ∧ d = 0 ∧ ∀ x : ℝ, f (2 * c - x) = 2 * d - f x) := sorry

theorem not_axis_of_symmetry : ¬ ∃ (c : ℝ), c = π / 12 ∧ ∀ x : ℝ, f (2 * c - x) = f x := sorry

theorem monotonicity_interval : ∀ x, (π / 6) < x ∧ x < (5 * π / 12) → f' x > 0 := sorry

end min_positive_period_pi_not_center_of_symmetry_not_axis_of_symmetry_monotonicity_interval_l103_103260


namespace annual_interest_rate_continuous_compounding_l103_103620

noncomputable def continuous_compounding_rate (A P : ℝ) (t : ℝ) : ℝ :=
  (Real.log (A / P)) / t

theorem annual_interest_rate_continuous_compounding :
  continuous_compounding_rate 8500 5000 10 = (Real.log (1.7)) / 10 :=
by
  sorry

end annual_interest_rate_continuous_compounding_l103_103620


namespace lcm_16_35_l103_103064

theorem lcm_16_35 : Nat.lcm 16 35 = 560 := by
  sorry

end lcm_16_35_l103_103064


namespace power_of_seven_l103_103020

theorem power_of_seven : 
  (7 : ℝ) ^ (1 / 4) / (7 ^ (1 / 7)) = (7 ^ (3 / 28)) :=
by
  sorry

end power_of_seven_l103_103020


namespace fraction_squares_sum_l103_103554

theorem fraction_squares_sum (x y z a b c : ℝ)
  (h1 : x / a + y / b + z / c = 3)
  (h2 : a / x + b / y + c / z = 0) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 9 := 
sorry

end fraction_squares_sum_l103_103554


namespace father_l103_103733

variable (S F : ℕ)

theorem father's_age (h1 : F = 3 * S + 3) (h2 : F + 3 = 2 * (S + 3) + 10) : F = 33 := by
  sorry

end father_l103_103733


namespace min_bounces_l103_103885

theorem min_bounces
  (h₀ : ℝ := 160)  -- initial height
  (r : ℝ := 3/4)  -- bounce ratio
  (final_h : ℝ := 20)  -- desired height
  (b : ℕ)  -- number of bounces
  : ∃ b, (h₀ * (r ^ b) < final_h ∧ ∀ b', b' < b → ¬(h₀ * (r ^ b') < final_h)) :=
sorry

end min_bounces_l103_103885


namespace time_to_finish_work_with_both_tractors_l103_103893

-- Definitions of given conditions
def work_rate_A : ℚ := 1 / 20
def work_rate_B : ℚ := 1 / 15
def time_A_worked : ℚ := 13
def remaining_work : ℚ := 1 - (work_rate_A * time_A_worked)
def combined_work_rate : ℚ := work_rate_A + work_rate_B

-- Statement that needs to be proven
theorem time_to_finish_work_with_both_tractors : 
  remaining_work / combined_work_rate = 3 :=
by
  sorry

end time_to_finish_work_with_both_tractors_l103_103893


namespace convert_volume_cubic_feet_to_cubic_yards_l103_103364

theorem convert_volume_cubic_feet_to_cubic_yards (V : ℤ) (V_ft³ : V = 216) : 
  V / 27 = 8 := 
by {
  sorry
}

end convert_volume_cubic_feet_to_cubic_yards_l103_103364


namespace completing_the_square_solution_correct_l103_103170

theorem completing_the_square_solution_correct (x : ℝ) :
  (x^2 + 8 * x + 9 = 0) ↔ ((x + 4)^2 = 7) :=
by
  sorry

end completing_the_square_solution_correct_l103_103170


namespace golden_state_total_points_l103_103569

theorem golden_state_total_points :
  let draymond_points := 12
  let curry_points := 2 * draymond_points
  let kelly_points := 9
  let durant_points := 2 * kelly_points
  let klay_points := draymond_points / 2
  draymond_points + curry_points + kelly_points + durant_points + klay_points = 69 :=
by
  let draymond_points := 12
  let curry_points := 2 * draymond_points
  let kelly_points := 9
  let durant_points := 2 * kelly_points
  let klay_points := draymond_points / 2
  calc
    draymond_points + curry_points + kelly_points + durant_points + klay_points
    = 12 + (2 * 12) + 9 + (2 * 9) + (12 / 2) : by sorry
    = 69 : by sorry

end golden_state_total_points_l103_103569


namespace parabola_c_value_l103_103764

theorem parabola_c_value (b c : ℝ) 
  (h1 : 2 * b + c = 6) 
  (h2 : -2 * b + c = 2)
  (vertex_cond : ∃ x y : ℝ, y = x^2 + b * x + c ∧ y = -x + 4) : 
  c = 4 :=
sorry

end parabola_c_value_l103_103764


namespace point_on_inverse_proportion_l103_103809

theorem point_on_inverse_proportion :
  ∀ (k x y : ℝ), 
    (∀ (x y: ℝ), (x = -2 ∧ y = 6) → y = k / x) →
    k = -12 →
    y = k / x →
    (x = 1 ∧ y = -12) :=
by
  sorry

end point_on_inverse_proportion_l103_103809


namespace range_of_m_l103_103828

theorem range_of_m (m : ℝ) : (∃ x : ℝ, |x - 1| + |x + m| ≤ 4) → -5 ≤ m ∧ m ≤ 3 :=
by
  intro h
  sorry

end range_of_m_l103_103828


namespace factorize_expression_l103_103062

variable (a : ℝ)

theorem factorize_expression : a^3 - 2 * a^2 = a^2 * (a - 2) :=
by
  sorry

end factorize_expression_l103_103062


namespace simplify_fraction_l103_103531

theorem simplify_fraction :
  ((1 / 4) + (1 / 6)) / ((3 / 8) - (1 / 3)) = 10 := by
  sorry

end simplify_fraction_l103_103531


namespace candy_problem_minimum_candies_l103_103871

theorem candy_problem_minimum_candies : ∃ (N : ℕ), N > 1 ∧ N % 2 = 1 ∧ N % 3 = 1 ∧ N % 5 = 1 ∧ N = 31 :=
by
  sorry

end candy_problem_minimum_candies_l103_103871


namespace sum_of_six_smallest_multiples_of_12_l103_103743

-- Define the six smallest distinct positive integer multiples of 12
def multiples_of_12 : List ℕ := [12, 24, 36, 48, 60, 72]

-- Define their sum
def sum_of_multiples : ℕ := multiples_of_12.sum

-- The proof statement
theorem sum_of_six_smallest_multiples_of_12 : sum_of_multiples = 252 := 
by
  sorry

end sum_of_six_smallest_multiples_of_12_l103_103743


namespace average_of_21_numbers_l103_103148

theorem average_of_21_numbers (n₁ n₂ : ℕ) (a b c : ℕ)
  (h₁ : n₁ = 11 * 48) -- Sum of the first 11 numbers
  (h₂ : n₂ = 11 * 41) -- Sum of the last 11 numbers
  (h₃ : c = 55) -- The 11th number
  : (n₁ + n₂ - c) / 21 = 44 := -- Average of all 21 numbers
by
  sorry

end average_of_21_numbers_l103_103148


namespace simplify_expression_l103_103967

theorem simplify_expression (a b : ℚ) (ha : a = -1) (hb : b = 1/4) : 
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by 
  sorry

end simplify_expression_l103_103967


namespace forty_percent_of_number_is_240_l103_103718

-- Define the conditions as assumptions in Lean
variable (N : ℝ)
variable (h1 : (1/4) * (1/3) * (2/5) * N = 20)

-- Prove that 40% of the number N is 240
theorem forty_percent_of_number_is_240 (h1: (1/4) * (1/3) * (2/5) * N = 20) : 0.40 * N = 240 :=
  sorry

end forty_percent_of_number_is_240_l103_103718


namespace rewrite_expression_and_compute_l103_103597

noncomputable def c : ℚ := 8
noncomputable def p : ℚ := -3 / 8
noncomputable def q : ℚ := 119 / 8

theorem rewrite_expression_and_compute :
  (∃ (c p q : ℚ), 8 * j ^ 2 - 6 * j + 16 = c * (j + p) ^ 2 + q) →
  q / p = -119 / 3 :=
by
  sorry

end rewrite_expression_and_compute_l103_103597


namespace remi_spilled_second_time_l103_103459

-- Defining the conditions from the problem
def bottle_capacity : ℕ := 20
def daily_refills : ℕ := 3
def total_days : ℕ := 7
def total_water_consumed : ℕ := 407
def first_spill : ℕ := 5

-- Using the conditions to define the total amount of water that Remi would have drunk without spilling.
def no_spill_total : ℕ := bottle_capacity * daily_refills * total_days

-- Defining the second spill
def second_spill : ℕ := no_spill_total - first_spill - total_water_consumed

-- Stating the theorem that we need to prove
theorem remi_spilled_second_time : second_spill = 8 :=
by
  sorry

end remi_spilled_second_time_l103_103459


namespace greatest_integer_solution_l103_103400

theorem greatest_integer_solution :
  ∃ x : ℤ, (∀ y : ℤ, (6 * (y : ℝ)^2 + 5 * (y : ℝ) - 8) < (3 * (y : ℝ)^2 - 4 * (y : ℝ) + 1) → y ≤ x) 
  ∧ (6 * (x : ℝ)^2 + 5 * (x : ℝ) - 8) < (3 * (x : ℝ)^2 - 4 * (x : ℝ) + 1) ∧ x = 0 :=
by
  sorry

end greatest_integer_solution_l103_103400


namespace min_value_of_F_on_negative_half_l103_103082

variable (f g : ℝ → ℝ)
variable (a b : ℝ)

def F (x : ℝ) := a * f x + b * g x + 2

def is_odd (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = -h x

theorem min_value_of_F_on_negative_half
  (h_f : is_odd f) (h_g : is_odd g)
  (max_F_positive_half : ∃ x, x > 0 ∧ F f g a b x = 5) :
  ∃ x, x < 0 ∧ F f g a b x = -3 :=
by {
  sorry
}

end min_value_of_F_on_negative_half_l103_103082


namespace son_l103_103846

noncomputable def my_age_in_years : ℕ := 84
noncomputable def total_age_in_years : ℕ := 140
noncomputable def months_in_a_year : ℕ := 12
noncomputable def weeks_in_a_year : ℕ := 52

theorem son's_age_in_weeks (G_d S_m G_m S_y : ℕ) (G_y : ℚ) :
  G_d = S_m →
  G_m = my_age_in_years * months_in_a_year →
  G_y = (G_m : ℚ) / months_in_a_year →
  G_y + S_y + my_age_in_years = total_age_in_years →
  S_y * weeks_in_a_year = 2548 :=
by
  intros h1 h2 h3 h4
  sorry

end son_l103_103846


namespace number_of_roses_two_days_ago_l103_103007

-- Define the conditions
variables (R : ℕ) 
-- Condition 1: Variable R is the number of roses planted two days ago.
-- Condition 2: The number of roses planted yesterday is R + 20.
-- Condition 3: The number of roses planted today is 2R.
-- Condition 4: The total number of roses planted over three days is 220.
axiom condition_1 : 0 ≤ R
axiom condition_2 : (R + (R + 20) + (2 * R)) = 220

-- Proof goal: Prove that R = 50 
theorem number_of_roses_two_days_ago : R = 50 :=
by sorry

end number_of_roses_two_days_ago_l103_103007


namespace x_value_l103_103993

theorem x_value :
  ∀ (x y : ℝ), x = y - 0.1 * y ∧ y = 125 + 0.1 * 125 → x = 123.75 :=
by
  intros x y h
  sorry

end x_value_l103_103993


namespace shortest_side_of_right_triangle_l103_103897

theorem shortest_side_of_right_triangle 
  (a b : ℕ) (ha : a = 7) (hb : b = 10) (c : ℝ) (hright : a^2 + b^2 = c^2) :
  min a b = 7 :=
by
  sorry

end shortest_side_of_right_triangle_l103_103897


namespace next_term_geometric_sequence_l103_103192

theorem next_term_geometric_sequence (x : ℝ) (r : ℝ) (a₀ a₃ next_term : ℝ)
    (h1 : a₀ = 2)
    (h2 : r = 3 * x)
    (h3 : a₃ = 54 * x^3)
    (h4 : next_term = a₃ * r) :
    next_term = 162 * x^4 := by
  sorry

end next_term_geometric_sequence_l103_103192


namespace triangle_angle_sum_acute_l103_103155

theorem triangle_angle_sum_acute (x : ℝ) (h1 : 60 + 70 + x = 180) (h2 : x ≠ 60 ∧ x ≠ 70) :
  x = 50 ∧ (60 < 90 ∧ 70 < 90 ∧ x < 90) := by
  sorry

end triangle_angle_sum_acute_l103_103155


namespace ratio_of_chords_l103_103325

theorem ratio_of_chords 
  (E F G H Q : Type)
  (EQ GQ FQ HQ : ℝ)
  (h1 : EQ = 4)
  (h2 : GQ = 10)
  (h3 : EQ * FQ = GQ * HQ) : 
  FQ / HQ = 5 / 2 := 
by 
  sorry

end ratio_of_chords_l103_103325


namespace find_larger_integer_l103_103995

theorem find_larger_integer (x : ℕ) (hx₁ : 4 * x > 0) (hx₂ : (x + 6) * 3 = 4 * x) : 4 * x = 72 :=
by
  sorry

end find_larger_integer_l103_103995


namespace total_paint_is_correct_l103_103509

/-- Given conditions -/
def paint_per_large_canvas := 3
def paint_per_small_canvas := 2
def large_paintings := 3
def small_paintings := 4

/-- Define total paint used using the given conditions -/
noncomputable def total_paint_used : ℕ := 
  (paint_per_large_canvas * large_paintings) + (paint_per_small_canvas * small_paintings)

/-- Theorem statement to show the total paint used equals 17 ounces -/
theorem total_paint_is_correct : total_paint_used = 17 := by
  sorry

end total_paint_is_correct_l103_103509


namespace simplify_fraction_subtraction_l103_103462

theorem simplify_fraction_subtraction : (1 / 210) - (17 / 35) = -101 / 210 := by
  sorry

end simplify_fraction_subtraction_l103_103462


namespace max_ab_bc_cd_l103_103713

theorem max_ab_bc_cd (a b c d : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d) (h_sum : a + b + c + d = 200) : 
    ab + bc + cd ≤ 10000 := by
  sorry

end max_ab_bc_cd_l103_103713


namespace inheritance_amount_l103_103279

theorem inheritance_amount
  (x : ℝ)
  (H1 : 0.25 * x + 0.15 * (x - 0.25 * x) = 15000) : x = 41379 := 
sorry

end inheritance_amount_l103_103279


namespace marks_age_more_than_thrice_aarons_l103_103162

theorem marks_age_more_than_thrice_aarons :
  ∃ (A : ℕ)(X : ℕ), 28 = A + 17 ∧ 25 = 3 * (A - 3) + X ∧ 32 = 2 * (A + 4) + 2 ∧ X = 1 :=
by
  sorry

end marks_age_more_than_thrice_aarons_l103_103162


namespace MNPQ_cyclic_l103_103443

open EuclideanGeometry

variable {A B C M N P Q : Point}
variable {angleA angleB angleC : Angle}
variable {ABC : Triangle}

-- given conditions
axiom angle_condition : angleA < angleB ∧ angleB ≤ angleC
axiom M_midpoint_CA : Midpoint M C A
axiom N_midpoint_AB : Midpoint N A B
axiom P_projection_B_CN : Projection P B (Median A B C N)
axiom Q_projection_C_BM : Projection Q C (Median A C B M)

theorem MNPQ_cyclic (h1 : angleA < angleB) (h2 : angleB ≤ angleC)
    (h3 : Midpoint M C A) (h4 : Midpoint N A B)
    (h5 : Projection P B (Median A B C N)) (h6 : Projection Q C (Median A C B M)) :
    CyclicQuadrilateral M N P Q :=
sorry

end MNPQ_cyclic_l103_103443


namespace number_of_four_digit_numbers_with_two_identical_digits_l103_103158

-- Define the conditions
def starts_with_nine (n : ℕ) : Prop := n / 1000 = 9
def has_exactly_two_identical_digits (n : ℕ) : Prop := 
  (∃ d1 d2, d1 ≠ d2 ∧ (n % 1000) / 100 = d1 ∧ (n % 100) / 10 = d1 ∧ n % 10 = d2) ∨
  (∃ d1 d2, d1 ≠ d2 ∧ (n % 1000) / 100 = d2 ∧ (n % 100) / 10 = d1 ∧ n % 10 = d1) ∨
  (∃ d1 d2, d1 ≠ d2 ∧ (n % 1000) / 100 = d1 ∧ (n % 100) / 10 = d2 ∧ n % 10 = d1)

-- Define the proof problem
theorem number_of_four_digit_numbers_with_two_identical_digits : 
  ∃ n, starts_with_nine n ∧ has_exactly_two_identical_digits n ∧ n = 432 := 
sorry

end number_of_four_digit_numbers_with_two_identical_digits_l103_103158


namespace arithmetic_sequence_common_difference_l103_103416

   variable (a_n : ℕ → ℝ)
   variable (a_5 : ℝ := 13)
   variable (S_5 : ℝ := 35)
   variable (d : ℝ)

   theorem arithmetic_sequence_common_difference {a_1 : ℝ} :
     (a_1 + 4 * d = a_5) ∧ (5 * a_1 + 10 * d = S_5) → d = 3 :=
   by
     sorry
   
end arithmetic_sequence_common_difference_l103_103416


namespace factor_difference_of_squares_l103_103059

-- Given: x is a real number.
-- Prove: x^2 - 64 = (x - 8) * (x + 8).
theorem factor_difference_of_squares (x : ℝ) : 
  x^2 - 64 = (x - 8) * (x + 8) :=
by
  sorry

end factor_difference_of_squares_l103_103059


namespace production_today_l103_103669

def average_production (P : ℕ) (n : ℕ) := P / n

theorem production_today :
  ∀ (T P n : ℕ), n = 9 → average_production P n = 50 → average_production (P + T) (n + 1) = 54 → T = 90 :=
by
  intros T P n h1 h2 h3
  sorry

end production_today_l103_103669


namespace product_of_five_consecutive_numbers_not_square_l103_103305

theorem product_of_five_consecutive_numbers_not_square (a b c d e : ℕ)
  (ha : a > 0) (hb : b = a + 1) (hc : c = b + 1) (hd : d = c + 1) (he : e = d + 1) :
  ¬ ∃ k : ℕ, a * b * c * d * e = k^2 := by
  sorry

end product_of_five_consecutive_numbers_not_square_l103_103305


namespace min_value_expression_l103_103920

theorem min_value_expression (x : ℝ) (hx : x > 3) : x + 4 / (x - 3) ≥ 7 :=
sorry

end min_value_expression_l103_103920


namespace find_x_l103_103936

-- Given condition: 144 / x = 14.4 / 0.0144
theorem find_x (x : ℝ) (h : 144 / x = 14.4 / 0.0144) : x = 0.144 := by
  sorry

end find_x_l103_103936


namespace no_integer_solution_for_conditions_l103_103095

theorem no_integer_solution_for_conditions :
  ¬∃ (x : ℤ), 
    (18 + x = 2 * (5 + x)) ∧
    (18 + x = 3 * (2 + x)) ∧
    ((18 + x) + (5 + x) + (2 + x) = 50) :=
by
  sorry

end no_integer_solution_for_conditions_l103_103095


namespace monotonic_intervals_max_min_values_l103_103683

def f (x : ℝ) := x^3 - 3*x
def f_prime (x : ℝ) := 3*(x-1)*(x+1)

theorem monotonic_intervals :
  (∀ x : ℝ, x < -1 → 0 < f_prime x) ∧ (∀ x : ℝ, -1 < x ∧ x < 1 → f_prime x < 0) ∧ (∀ x : ℝ, x > 1 → 0 < f_prime x) :=
  by
  sorry

theorem max_min_values :
  ∀ x ∈ Set.Icc (-1 : ℝ) 3, f x ≤ 18 ∧ f x ≥ -2 ∧ 
  (f 1 = -2) ∧
  (f 3 = 18) :=
  by
  sorry

end monotonic_intervals_max_min_values_l103_103683


namespace internship_choices_l103_103670

theorem internship_choices :
  let choices := 3
  let students := 4
  (choices ^ students) = 81 := 
by
  intros
  calc
    3 ^ 4 = 81 : by norm_num

end internship_choices_l103_103670


namespace multiplication_of_935421_and_625_l103_103496

theorem multiplication_of_935421_and_625 :
  935421 * 625 = 584638125 :=
by sorry

end multiplication_of_935421_and_625_l103_103496


namespace single_rooms_booked_l103_103224

noncomputable def hotel_problem (S D : ℕ) : Prop :=
  S + D = 260 ∧ 35 * S + 60 * D = 14000

theorem single_rooms_booked (S D : ℕ) (h : hotel_problem S D) : S = 64 :=
by
  sorry

end single_rooms_booked_l103_103224


namespace regression_equation_is_correct_l103_103250

theorem regression_equation_is_correct 
  (linear_corr : ∃ (f : ℝ → ℝ), ∀ (x : ℝ), ∃ (y : ℝ), y = f x)
  (mean_b : ℝ)
  (mean_x : ℝ)
  (mean_y : ℝ)
  (mean_b_eq : mean_b = 0.51)
  (mean_x_eq : mean_x = 61.75)
  (mean_y_eq : mean_y = 38.14) : 
  mean_y = mean_b * mean_x + 6.65 :=
sorry

end regression_equation_is_correct_l103_103250


namespace plan_b_cheaper_than_plan_a_l103_103387

theorem plan_b_cheaper_than_plan_a (x : ℕ) (h : 401 ≤ x) :
  2000 + 5 * x < 10 * x :=
by
  sorry

end plan_b_cheaper_than_plan_a_l103_103387


namespace fraction_of_constants_l103_103460

theorem fraction_of_constants :
  ∃ a b c : ℤ, (4 : ℤ) * a * (k + b)^2 + c = 4 * k^2 - 8 * k + 16 ∧
             4 * -1 * (k + (-1))^2 + 12 = 4 * k^2 - 8 * k + 16 ∧
             a = 4 ∧ b = -1 ∧ c = 12 ∧ c / b = -12 :=
by
  sorry

end fraction_of_constants_l103_103460


namespace mike_scored_212_l103_103128

variable {M : ℕ}

def passing_marks (max_marks : ℕ) : ℕ := (30 * max_marks) / 100

def mike_marks (passing_marks shortfall : ℕ) : ℕ := passing_marks - shortfall

theorem mike_scored_212 (max_marks : ℕ) (shortfall : ℕ)
  (h1 : max_marks = 790)
  (h2 : shortfall = 25)
  (h3 : M = mike_marks (passing_marks max_marks) shortfall) : 
  M = 212 := 
by 
  sorry

end mike_scored_212_l103_103128


namespace decompose_two_over_eleven_decompose_two_over_n_l103_103271

-- Problem 1: Decompose 2/11
theorem decompose_two_over_eleven : (2 : ℚ) / 11 = (1 / 6) + (1 / 66) :=
  sorry

-- Problem 2: General form for 2/n for odd n >= 5
theorem decompose_two_over_n (n : ℕ) (hn : n ≥ 5) (odd_n : n % 2 = 1) :
  (2 : ℚ) / n = (1 / ((n + 1) / 2)) + (1 / (n * (n + 1) / 2)) :=
  sorry

end decompose_two_over_eleven_decompose_two_over_n_l103_103271


namespace range_of_a_l103_103544

theorem range_of_a (a : ℝ) (h : ∃ x1 x2, x1 ≠ x2 ∧ 3 * x1^2 + a = 0 ∧ 3 * x2^2 + a = 0) : a < 0 :=
sorry

end range_of_a_l103_103544


namespace arithmetic_sequence_sum_l103_103099

variable {a : ℕ → ℤ} 
variable {a_3 a_4 a_5 : ℤ}

-- Hypothesis: arithmetic sequence and given condition
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n, a (n+1) - a n = a 2 - a 1

theorem arithmetic_sequence_sum (h : is_arithmetic_sequence a) (h_sum : a_3 + a_4 + a_5 = 12) : 
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 := 
by 
  sorry

end arithmetic_sequence_sum_l103_103099


namespace planet_X_periods_l103_103033

theorem planet_X_periods (N : ℕ) (h1 : N = 100000) :
  ∃ n m : ℕ, n * m = N ∧ 1 ≤ n ∧ 1 ≤ m ∧ ∀ a b : ℕ, a * b = N → ((a = n ∧ b = m) ∨ (a = m ∧ b = n)) :=
sorry

end planet_X_periods_l103_103033


namespace remainder_div_2468135790_101_l103_103332

theorem remainder_div_2468135790_101 : 2468135790 % 101 = 50 :=
by
  sorry

end remainder_div_2468135790_101_l103_103332


namespace total_food_per_day_l103_103608

theorem total_food_per_day 
  (first_soldiers : ℕ)
  (second_soldiers : ℕ)
  (food_first_side_per_soldier : ℕ)
  (food_second_side_per_soldier : ℕ) :
  first_soldiers = 4000 →
  second_soldiers = first_soldiers - 500 →
  food_first_side_per_soldier = 10 →
  food_second_side_per_soldier = food_first_side_per_soldier - 2 →
  (first_soldiers * food_first_side_per_soldier + second_soldiers * food_second_side_per_soldier = 68000) :=
by
  intros h1 h2 h3 h4
  sorry

end total_food_per_day_l103_103608


namespace a_values_l103_103445

def A (a : ℤ) : Set ℤ := {2, a^2 - a + 2, 1 - a}

theorem a_values (a : ℤ) (h : 4 ∈ A a) : a = 2 ∨ a = -3 :=
sorry

end a_values_l103_103445


namespace root_of_equation_l103_103395

theorem root_of_equation (x : ℝ) :
  (∃ u : ℝ, u = Real.sqrt (x + 15) ∧ u - 7 / u = 6) → x = 34 :=
by
  sorry

end root_of_equation_l103_103395


namespace new_customers_needed_l103_103503

theorem new_customers_needed 
  (initial_customers : ℕ)
  (customers_after_some_left : ℕ)
  (first_group_left : ℕ)
  (second_group_left : ℕ)
  (new_customers : ℕ)
  (h1 : initial_customers = 13)
  (h2 : customers_after_some_left = 9)
  (h3 : first_group_left = initial_customers - customers_after_some_left)
  (h4 : second_group_left = 8)
  (h5 : new_customers = first_group_left + second_group_left) :
  new_customers = 12 :=
by
  sorry

end new_customers_needed_l103_103503


namespace incorrect_inequality_l103_103093

theorem incorrect_inequality (x y : ℝ) (h : x > y) : ¬ (-3 * x > -3 * y) :=
by
  sorry

end incorrect_inequality_l103_103093


namespace find_y_l103_103752

theorem find_y (x y : ℤ) (h1 : x + y = 260) (h2 : x - y = 200) : y = 30 :=
sorry

end find_y_l103_103752


namespace x_zero_necessary_but_not_sufficient_l103_103339

-- Definitions based on conditions
def x_eq_zero (x : ℝ) := x = 0
def xsq_plus_ysq_eq_zero (x y : ℝ) := x^2 + y^2 = 0

-- Statement that x = 0 is a necessary but not sufficient condition for x^2 + y^2 = 0
theorem x_zero_necessary_but_not_sufficient (x y : ℝ) : (x = 0 ↔ x^2 + y^2 = 0) → False :=
by sorry

end x_zero_necessary_but_not_sufficient_l103_103339


namespace find_ab_l103_103121

noncomputable def poly (x a b : ℝ) := x^4 + a * x^3 - 5 * x^2 + b * x - 6

theorem find_ab (a b : ℝ) (h : poly 2 a b = 0) : (a = 0 ∧ b = 4) :=
by
  sorry

end find_ab_l103_103121


namespace translate_point_A_l103_103703

theorem translate_point_A :
  let A : ℝ × ℝ := (-1, 2)
  let x_translation : ℝ := 4
  let y_translation : ℝ := -2
  let A1 : ℝ × ℝ := (A.1 + x_translation, A.2 + y_translation)
  A1 = (3, 0) :=
by
  let A : ℝ × ℝ := (-1, 2)
  let x_translation : ℝ := 4
  let y_translation : ℝ := -2
  let A1 : ℝ × ℝ := (A.1 + x_translation, A.2 + y_translation)
  show A1 = (3, 0)
  sorry

end translate_point_A_l103_103703


namespace eq_satisfies_exactly_four_points_l103_103859

theorem eq_satisfies_exactly_four_points : ∀ (x y : ℝ), 
  (x^2 - 4)^2 + (y^2 - 4)^2 = 0 ↔ 
  (x = 2 ∧ y = 2) ∨ (x = -2 ∧ y = 2) ∨ (x = 2 ∧ y = -2) ∨ (x = -2 ∧ y = -2) := 
by
  sorry

end eq_satisfies_exactly_four_points_l103_103859


namespace area_of_union_of_triangles_l103_103218

structure Point :=
  (x : ℝ)
  (y : ℝ)

noncomputable def triangle_area (p1 p2 p3 : Point) : ℝ :=
  0.5 * |(p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))|

def reflected_point (p : Point) (y_reflect_line : ℝ) : Point :=
  { x := p.x, y := 2 * y_reflect_line - p.y }

def triangle_union_area (p1 p2 p3 rp1 rp2 rp3 : Point) : ℝ :=
  if triangle_area p1 p2 p3 > triangle_area rp1 rp2 rp3 then
    triangle_area p1 p2 p3
  else
    triangle_area rp1 rp2 rp3

theorem area_of_union_of_triangles :
  let A := Point.mk 6 5,
      B := Point.mk 8 3,
      C := Point.mk 9 1,
      y_line := 1,
      A' := reflected_point A y_line,
      B' := reflected_point B y_line,
      C' := reflected_point C y_line
  in triangle_union_area A B C A' B' C' = 4 := by
  sorry

end area_of_union_of_triangles_l103_103218


namespace prime_sum_diff_l103_103234

open Nat

-- Define what it means for a number to be prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- The problem statement
theorem prime_sum_diff (p : ℕ) (q s r t : ℕ) :
  is_prime p → is_prime q → is_prime s → is_prime r → is_prime t →
  p = q + s → p = r - t → p = 5 :=
by
  sorry

end prime_sum_diff_l103_103234


namespace height_inradius_ratio_is_7_l103_103540

-- Definitions of geometric entities and given conditions.
variable (h r : ℝ)
variable (cos_theta : ℝ)
variable (cos_theta_eq : cos_theta = 1 / 6)

-- Theorem statement: Ratio of height to inradius is 7 given the cosine condition.
theorem height_inradius_ratio_is_7
  (h r : ℝ)
  (cos_theta : ℝ)
  (cos_theta_eq : cos_theta = 1 / 6)
  (prism_def : true) -- Added to mark the geometric nature properly
: h / r = 7 :=
sorry  -- Placeholder for the actual proof.

end height_inradius_ratio_is_7_l103_103540


namespace maximum_pairwise_sum_is_maximal_l103_103216

noncomputable def maximum_pairwise_sum (set_sums : List ℝ) (x y z w : ℝ) : Prop :=
  ∃ (a b c d e : ℝ), set_sums = [400, 500, 600, 700, 800, 900, x, y, z, w] ∧  
  ((2 / 5) * (400 + 500 + 600 + 700 + 800 + 900 + x + y + z + w)) = 
    (a + b + c + d + e) ∧ 
  5 * (a + b + c + d + e) - (400 + 500 + 600 + 700 + 800 + 900) = 1966.67

theorem maximum_pairwise_sum_is_maximal :
  maximum_pairwise_sum [400, 500, 600, 700, 800, 900] 1966.67 (1966.67 / 4) 
(1966.67 / 3) (1966.67 / 2) :=
sorry

end maximum_pairwise_sum_is_maximal_l103_103216


namespace division_into_rectangles_l103_103567

theorem division_into_rectangles (figure : Type) (valid_division : figure → Prop) : (∃ ways, ways = 8) :=
by {
  -- assume given conditions related to valid_division using "figure"
  sorry
}

end division_into_rectangles_l103_103567


namespace length_of_second_train_l103_103372

theorem length_of_second_train 
  (length_first_train : ℝ) 
  (speed_first_train_kmph : ℝ) 
  (speed_second_train_kmph : ℝ) 
  (time_to_cross : ℝ) 
  (h1 : length_first_train = 400)
  (h2 : speed_first_train_kmph = 72)
  (h3 : speed_second_train_kmph = 36)
  (h4 : time_to_cross = 69.99440044796417) :
  let speed_first_train := speed_first_train_kmph * (1000 / 3600)
  let speed_second_train := speed_second_train_kmph * (1000 / 3600)
  let relative_speed := speed_first_train - speed_second_train
  let distance := relative_speed * time_to_cross
  let length_second_train := distance - length_first_train
  length_second_train = 299.9440044796417 :=
  by
    sorry

end length_of_second_train_l103_103372


namespace jog_time_each_morning_is_1_5_hours_l103_103960

-- Define the total time Mr. John spent jogging
def total_time_spent_jogging : ℝ := 21

-- Define the number of days Mr. John jogged
def number_of_days_jogged : ℕ := 14

-- Define the time Mr. John jogs each morning
noncomputable def time_jogged_each_morning : ℝ := total_time_spent_jogging / number_of_days_jogged

-- State the theorem that the time jogged each morning is 1.5 hours
theorem jog_time_each_morning_is_1_5_hours : time_jogged_each_morning = 1.5 := by
  sorry

end jog_time_each_morning_is_1_5_hours_l103_103960


namespace max_value_of_f_l103_103065

open Real

noncomputable def f (x : ℝ) : ℝ := (log x) / x

theorem max_value_of_f :
  ∃ x, (f x = 1 / exp 1) ∧ (∀ y, f y ≤ f x) :=
by
  sorry

end max_value_of_f_l103_103065


namespace maximum_median_soda_shop_l103_103648

noncomputable def soda_shop_median (total_cans : ℕ) (total_customers : ℕ) (min_cans_per_customer : ℕ) : ℝ :=
  if total_cans = 300 ∧ total_customers = 120 ∧ min_cans_per_customer = 1 then 3.5 else sorry

theorem maximum_median_soda_shop : soda_shop_median 300 120 1 = 3.5 :=
by
  sorry

end maximum_median_soda_shop_l103_103648


namespace factorize_problem1_factorize_problem2_l103_103233

-- Problem 1: Factorization of 4x^2 - 16
theorem factorize_problem1 (x : ℝ) : 4 * x^2 - 16 = 4 * (x - 2) * (x + 2) :=
by
  sorry

-- Problem 2: Factorization of a^2b - 4ab + 4b
theorem factorize_problem2 (a b : ℝ) : a^2 * b - 4 * a * b + 4 * b = b * (a - 2) ^ 2 :=
by
  sorry

end factorize_problem1_factorize_problem2_l103_103233


namespace simplify_expression_l103_103970

section
variable (a b : ℚ) (h_a : a = -1) (h_b : b = 1/4)

theorem simplify_expression : 
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by
  sorry
end

end simplify_expression_l103_103970


namespace average_speed_correct_l103_103611

-- Define the conditions
def distance_first_hour := 90 -- in km
def distance_second_hour := 30 -- in km
def time_first_hour := 1 -- in hours
def time_second_hour := 1 -- in hours

-- Define the total distance and total time
def total_distance := distance_first_hour + distance_second_hour
def total_time := time_first_hour + time_second_hour

-- Define the average speed
def avg_speed := total_distance / total_time

-- State the theorem to prove the average speed is 60
theorem average_speed_correct :
  avg_speed = 60 := 
by 
  -- Placeholder for the actual proof
  sorry

end average_speed_correct_l103_103611


namespace balloon_height_l103_103819

theorem balloon_height :
  let initial_money : ℝ := 200
  let cost_sheet : ℝ := 42
  let cost_rope : ℝ := 18
  let cost_tank_and_burner : ℝ := 14
  let helium_price_per_ounce : ℝ := 1.5
  let lift_per_ounce : ℝ := 113
  let remaining_money := initial_money - cost_sheet - cost_rope - cost_tank_and_burner
  let ounces_of_helium := remaining_money / helium_price_per_ounce
  let height := ounces_of_helium * lift_per_ounce
  height = 9492 :=
by
  sorry

end balloon_height_l103_103819


namespace initial_crayons_count_l103_103483

theorem initial_crayons_count (C : ℕ) :
  (3 / 8) * C = 18 → C = 48 :=
by
  sorry

end initial_crayons_count_l103_103483


namespace mildred_weight_l103_103450

theorem mildred_weight (carol_weight mildred_is_heavier : ℕ) (h1 : carol_weight = 9) (h2 : mildred_is_heavier = 50) :
  carol_weight + mildred_is_heavier = 59 :=
by
  sorry

end mildred_weight_l103_103450


namespace area_OMVK_l103_103660

theorem area_OMVK :
  ∀ (S_OKCL S_ONAM S_ONBM S_ABCD S_OMVK : ℝ),
    S_OKCL = 6 →
    S_ONAM = 12 →
    S_ONBM = 24 →
    S_ABCD = 4 * (S_OKCL + S_ONAM) →
    S_OMVK = S_ABCD - S_OKCL - S_ONAM - S_ONBM →
    S_OMVK = 30 :=
by
  intros S_OKCL S_ONAM S_ONBM S_ABCD S_OMVK h_OKCL h_ONAM h_ONBM h_ABCD h_OMVK
  rw [h_OKCL, h_ONAM, h_ONBM] at *
  sorry

end area_OMVK_l103_103660


namespace solution_l103_103814

theorem solution (x : ℝ) : (x = -2/5) → (x < x^3 ∧ x^3 < x^2) :=
by
  intro h
  rw [h]
  -- sorry to skip the proof
  sorry

end solution_l103_103814


namespace sarahs_score_l103_103599

theorem sarahs_score (g s : ℕ) (h₁ : s = g + 30) (h₂ : (s + g) / 2 = 95) : s = 110 := by
  sorry

end sarahs_score_l103_103599


namespace tan_theta_eq_neg3_then_expr_eq_5_div_2_l103_103070

theorem tan_theta_eq_neg3_then_expr_eq_5_div_2
  (θ : ℝ) (h : Real.tan θ = -3) :
  (Real.sin θ - 2 * Real.cos θ) / (Real.cos θ + Real.sin θ) = 5 / 2 := 
sorry

end tan_theta_eq_neg3_then_expr_eq_5_div_2_l103_103070


namespace not_valid_mapping_circle_triangle_l103_103021

inductive Point
| mk : ℝ → ℝ → Point

inductive Circle
| mk : ℝ → ℝ → ℝ → Circle

inductive Triangle
| mk : Point → Point → Point → Triangle

open Point (mk)
open Circle (mk)
open Triangle (mk)

def valid_mapping (A B : Type) (f : A → B) := ∀ a₁ a₂ : A, f a₁ = f a₂ → a₁ = a₂

def inscribed_triangle_mapping (c : Circle) : Triangle := sorry -- map a circle to one of its inscribed triangles

theorem not_valid_mapping_circle_triangle :
  ¬ valid_mapping Circle Triangle inscribed_triangle_mapping :=
sorry

end not_valid_mapping_circle_triangle_l103_103021


namespace region_area_l103_103011

-- Let x and y be real numbers
variables (x y : ℝ)

-- Define the inequality condition
def region_condition (x y : ℝ) : Prop := abs (4 * x - 20) + abs (3 * y + 9) ≤ 6

-- The statement that needs to be proved
theorem region_area : (∃ x y : ℝ, region_condition x y) → ∃ A : ℝ, A = 6 :=
by
  sorry

end region_area_l103_103011


namespace number_of_workers_l103_103856

open Real

theorem number_of_workers (W : ℝ) 
    (average_salary_workers average_salary_technicians average_salary_non_technicians : ℝ)
    (h1 : average_salary_workers = 8000)
    (h2 : average_salary_technicians = 12000)
    (h3 : average_salary_non_technicians = 6000)
    (h4 : average_salary_workers * W = average_salary_technicians * 7 + average_salary_non_technicians * (W - 7)) :
    W = 21 :=
by
  rw [h1, h2, h3] at h4
  simp at h4
  linarith

end number_of_workers_l103_103856


namespace people_later_than_yoongi_l103_103754

variable (total_students : ℕ) (people_before_yoongi : ℕ)

theorem people_later_than_yoongi
    (h1 : total_students = 20)
    (h2 : people_before_yoongi = 11) :
    total_students - (people_before_yoongi + 1) = 8 := 
sorry

end people_later_than_yoongi_l103_103754


namespace matrix_equation_l103_103575

open Matrix

-- Define matrix B
def B : Matrix (Fin 2) (Fin 2) (ℤ) :=
  ![![1, -2], 
    ![-3, 5]]

-- The proof problem statement in Lean 4
theorem matrix_equation (r s : ℤ) (I : Matrix (Fin 2) (Fin 2) (ℤ))  [DecidableEq (ℤ)] [Fintype (Fin 2)] : 
  I = 1 ∧ B ^ 6 = r • B + s • I ↔ r = 2999 ∧ s = 2520 := by {
    sorry
}

end matrix_equation_l103_103575


namespace incorrect_observation_l103_103730

theorem incorrect_observation (n : ℕ) (mean_original mean_corrected correct_obs incorrect_obs : ℝ)
  (h1 : n = 40) 
  (h2 : mean_original = 36) 
  (h3 : mean_corrected = 36.45) 
  (h4 : correct_obs = 34) 
  (h5 : n * mean_original = 1440) 
  (h6 : n * mean_corrected = 1458) 
  (h_diff : 1458 - 1440 = 18) :
  incorrect_obs = 52 :=
by
  sorry

end incorrect_observation_l103_103730


namespace inequality_proof_l103_103097

theorem inequality_proof (a b c : ℝ) (h1 : b < c) (h2 : 1 < a) (h3 : a < b + c) (h4 : b + c < a + 1) : b < a :=
by
  sorry

end inequality_proof_l103_103097


namespace convert_volume_cubic_feet_to_cubic_yards_l103_103365

theorem convert_volume_cubic_feet_to_cubic_yards (V : ℤ) (V_ft³ : V = 216) : 
  V / 27 = 8 := 
by {
  sorry
}

end convert_volume_cubic_feet_to_cubic_yards_l103_103365


namespace folded_strip_fit_l103_103349

open Classical

noncomputable def canFitAfterFolding (r : ℝ) (strip : Set (ℝ × ℝ)) (folded_strip : Set (ℝ × ℝ)) : Prop :=
  ∀ p : ℝ × ℝ, p ∈ folded_strip → (p.1^2 + p.2^2 ≤ r^2)

theorem folded_strip_fit {r : ℝ} {strip folded_strip : Set (ℝ × ℝ)} :
  (∀ p : ℝ × ℝ, p ∈ strip → (p.1^2 + p.2^2 ≤ r^2)) →
  (∀ q : ℝ × ℝ, q ∈ folded_strip → (∃ p : ℝ × ℝ, p ∈ strip ∧ q = p)) →
  canFitAfterFolding r strip folded_strip :=
by
  intros hs hf
  sorry

end folded_strip_fit_l103_103349


namespace minimum_value_of_f_l103_103418

noncomputable def f (x a : ℝ) := (1/3) * x^3 + (a-1) * x^2 - 4 * a * x + a

theorem minimum_value_of_f (a : ℝ) (h : a < -1) :
  (if -3/2 < a then ∀ (x : ℝ), 2 ≤ x ∧ x ≤ 3 → f x a ≥ f (-2*a) a
   else ∀ (x : ℝ), 2 ≤ x ∧ x ≤ 3 → f x a ≥ f 3 a) :=
sorry

end minimum_value_of_f_l103_103418


namespace amy_money_left_l103_103221

def amount_left (initial_amount doll_price board_game_price comic_book_price doll_qty board_game_qty comic_book_qty board_game_discount sales_tax_rate : ℝ) :
    ℝ :=
  let cost_dolls := doll_qty * doll_price
  let cost_board_games := board_game_qty * board_game_price
  let cost_comic_books := comic_book_qty * comic_book_price
  let discounted_cost_board_games := cost_board_games * (1 - board_game_discount)
  let total_cost_before_tax := cost_dolls + discounted_cost_board_games + cost_comic_books
  let sales_tax := total_cost_before_tax * sales_tax_rate
  let total_cost_after_tax := total_cost_before_tax + sales_tax
  initial_amount - total_cost_after_tax

theorem amy_money_left :
  amount_left 100 1.25 12.75 3.50 3 2 4 0.10 0.08 = 56.04 :=
by
  sorry

end amy_money_left_l103_103221


namespace m_1_sufficient_but_not_necessary_l103_103798

def lines_parallel (m : ℝ) : Prop :=
  let l1_slope := -m
  let l2_slope := (2 - 3 * m) / m
  l1_slope = l2_slope

theorem m_1_sufficient_but_not_necessary (m : ℝ) (h₁ : lines_parallel m) : 
  (m = 1) → (∃ m': ℝ, lines_parallel m' ∧ m' ≠ 1) :=
sorry

end m_1_sufficient_but_not_necessary_l103_103798


namespace age_of_oldest_sibling_l103_103278

theorem age_of_oldest_sibling (Kay_siblings : ℕ) (Kay_age : ℕ) (youngest_sibling_age : ℕ) (oldest_sibling_age : ℕ) 
  (h1 : Kay_siblings = 14) (h2 : Kay_age = 32) (h3 : youngest_sibling_age = Kay_age / 2 - 5) 
  (h4 : oldest_sibling_age = 4 * youngest_sibling_age) : oldest_sibling_age = 44 := 
sorry

end age_of_oldest_sibling_l103_103278


namespace general_form_identity_expression_simplification_l103_103847

section
variable (a b x y : ℝ)

theorem general_form_identity : (a + b) * (a^2 - a * b + b^2) = a^3 + b^3 :=
by
  sorry

theorem expression_simplification : (x + y) * (x^2 - x * y + y^2) - (x - y) * (x^2 + x * y + y^2) = 2 * y^3 :=
by
  sorry
end

end general_form_identity_expression_simplification_l103_103847


namespace find_smallest_x_l103_103664

-- Definition of the conditions
def cong1 (x : ℤ) : Prop := x % 5 = 4
def cong2 (x : ℤ) : Prop := x % 7 = 6
def cong3 (x : ℤ) : Prop := x % 8 = 7

-- Statement of the problem
theorem find_smallest_x :
  ∃ (x : ℕ), x > 0 ∧ cong1 x ∧ cong2 x ∧ cong3 x ∧ x = 279 :=
by
  sorry

end find_smallest_x_l103_103664


namespace odd_function_m_value_l103_103925

noncomputable def f (x : ℝ) : ℝ := 2 - 3 / x
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := f x - m

theorem odd_function_m_value :
  ∃ m : ℝ, (∀ (x : ℝ), g (-x) m + g x m = 0) ∧ m = 2 :=
by
  sorry

end odd_function_m_value_l103_103925


namespace cone_volume_l103_103104

theorem cone_volume (r l h V: ℝ) (h1: 15 * Real.pi = Real.pi * r^2 + Real.pi * r * l)
  (h2: 2 * Real.pi * r = (1 / 3) * Real.pi * l) :
  (V = (1 / 3) * Real.pi * r^2 * h) → h = Real.sqrt (l^2 - r^2) → l = 6 * r → r = Real.sqrt (15 / 7) → 
  V = (25 * Real.sqrt 3 / 7) * Real.pi :=
sorry

end cone_volume_l103_103104


namespace speed_of_second_car_l103_103204

theorem speed_of_second_car (s1 s2 s : ℕ) (v1 : ℝ) (h_s1 : s1 = 500) (h_s2 : s2 = 700) 
  (h_s : s = 100) (h_v1 : v1 = 10) : 
  (∃ v2 : ℝ, v2 = 12 ∨ v2 = 16) :=
by 
  sorry

end speed_of_second_car_l103_103204


namespace monotonically_increasing_power_function_l103_103926

theorem monotonically_increasing_power_function (m : ℝ) :
  (∀ x : ℝ, 0 < x → (m ^ 2 - 2 * m - 2) * x ^ (m - 2) > 0 → (m ^ 2 - 2 * m - 2) > 0 ∧ (m - 2) > 0) ↔ m = 3 := 
sorry

end monotonically_increasing_power_function_l103_103926


namespace range_of_a_l103_103088

theorem range_of_a (a : ℝ) :
  (a + 1)^2 > (3 - 2 * a)^2 ↔ (2 / 3) < a ∧ a < 4 :=
sorry

end range_of_a_l103_103088


namespace geometric_seq_b6_l103_103945

variable {b : ℕ → ℝ}

theorem geometric_seq_b6 (h1 : b 3 * b 9 = 9) (h2 : ∃ r, ∀ n, b (n + 1) = r * b n) : b 6 = 3 ∨ b 6 = -3 :=
by
  sorry

end geometric_seq_b6_l103_103945


namespace line_AB_eq_line_circle_intersection_l103_103433

-- Definitions based on the conditions
def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

def circle_parametric_eq (θ : ℝ) : ℝ × ℝ :=
  (1 + 2 * Real.cos θ, 2 * Real.sin θ)

def line_eq (A B : ℝ × ℝ) : ℝ → ℝ := λ x, -x + 2

-- Points A and B in Cartesian coordinates
def point_A : ℝ × ℝ := polar_to_cartesian 2 (Real.pi / 2)
def point_B : ℝ × ℝ := polar_to_cartesian (Real.sqrt 2) (Real.pi / 4)

-- Line AB in Cartesian coordinates
def line_AB (x : ℝ) : ℝ := line_eq point_A point_B x

-- The standard form of line AB
theorem line_AB_eq : ∀ x y, x + y - 2 = 0 ↔ y = line_AB x := sorry

-- Determining the intersection of line AB and circle C
def center_C : ℝ × ℝ := (1, 0)
def radius_C : ℝ := 2

theorem line_circle_intersection : 
  let line_distance := λ (C : ℝ × ℝ), (|C.1 - 2| / Real.sqrt 2)
  line_distance center_C < radius_C ↔ line_AB 1 = 1 := sorry

end line_AB_eq_line_circle_intersection_l103_103433


namespace max_possible_cables_l103_103223

theorem max_possible_cables (num_employees : ℕ) (num_brand_X : ℕ) (num_brand_Y : ℕ) 
  (max_connections : ℕ) (num_cables : ℕ) :
  num_employees = 40 →
  num_brand_X = 25 →
  num_brand_Y = 15 →
  max_connections = 3 →
  (∀ x : ℕ, x < max_connections → num_cables ≤ 3 * num_brand_Y) →
  num_cables = 45 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end max_possible_cables_l103_103223


namespace playground_area_l103_103318

theorem playground_area (w l : ℕ) (h1 : 2 * l + 2 * w = 72) (h2 : l = 3 * w) : l * w = 243 := by
  sorry

end playground_area_l103_103318


namespace f_plus_g_eq_l103_103081

variables {R : Type*} [CommRing R]

-- Define the odd function f
def f (x : R) : R := sorry

-- Define the even function g
def g (x : R) : R := sorry

-- Define that f is odd and g is even
axiom f_odd (x : R) : f (-x) = -f x
axiom g_even (x : R) : g (-x) = g x

-- Define the given equation
axiom f_minus_g_eq (x : R) : f x - g x = x ^ 2 + 9 * x + 12

-- Statement of the goal
theorem f_plus_g_eq (x : R) : f x + g x = -x ^ 2 + 9 * x - 12 := by
  sorry

end f_plus_g_eq_l103_103081


namespace number_of_friends_l103_103118

-- Define the conditions
def kendra_packs : ℕ := 7
def tony_packs : ℕ := 5
def pens_per_kendra_pack : ℕ := 4
def pens_per_tony_pack : ℕ := 6
def pens_kendra_keep : ℕ := 3
def pens_tony_keep : ℕ := 3

-- Define the theorem to be proved
theorem number_of_friends 
  (packs_k : ℕ := kendra_packs)
  (packs_t : ℕ := tony_packs)
  (pens_per_pack_k : ℕ := pens_per_kendra_pack)
  (pens_per_pack_t : ℕ := pens_per_tony_pack)
  (kept_k : ℕ := pens_kendra_keep)
  (kept_t : ℕ := pens_tony_keep) :
  packs_k * pens_per_pack_k + packs_t * pens_per_pack_t - (kept_k + kept_t) = 52 :=
by
  sorry

end number_of_friends_l103_103118


namespace exists_odd_midpoint_l103_103842

open Real

variable {p : Fin 1993 → (ℤ × ℤ)}

-- Condition 1: There are 1993 distinct points with integer coordinates
def points_distinct (p : Fin 1993 → (ℤ × ℤ)) : Prop :=
  ∀ i j, i ≠ j → p i ≠ p j

-- Condition 2: Each segment does not contain other lattice points
def no_lattice_points (p : Fin 1993 → (ℤ × ℤ)) : Prop :=
  ∀ i, i < 1992 → ∀ k, k ≠ 0 → k ≠ 1 → (k : ℝ) * (fst (p ⟨i, sorry⟩) - fst (p ⟨i+1, sorry⟩), snd (p ⟨i, sorry⟩) - snd (p ⟨i+1, sorry⟩)) ∉ ℤ × ℤ

-- Proof Goal: There exists at least one segment containing Q satisfying the given conditions
theorem exists_odd_midpoint (p : Fin 1993 → (ℤ × ℤ)) 
  (h1 : points_distinct p) 
  (h2 : no_lattice_points p) : 
  ∃ i, i < 1992 ∧ 
    let m := (fst (p ⟨i, sorry⟩) + fst (p ⟨i+1, sorry⟩)) / 2,
        n := (snd (p ⟨i, sorry⟩) + snd (p ⟨i+1, sorry⟩)) / 2 in 
    odd (2 * m) ∧ odd (2 * n) := 
sorry

end exists_odd_midpoint_l103_103842


namespace total_travel_time_l103_103390

/-
Define the conditions:
1. Distance_1 is 150 miles,
2. Speed_1 is 50 mph,
3. Stop_time is 0.5 hours,
4. Distance_2 is 200 miles,
5. Speed_2 is 75 mph.

and prove that the total time equals 6.17 hours.
-/

theorem total_travel_time :
  let distance1 := 150
  let speed1 := 50
  let stop_time := 0.5
  let distance2 := 200
  let speed2 := 75
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  time1 + stop_time + time2 = 6.17 :=
by {
  -- sorry to skip the proof part
  sorry
}

end total_travel_time_l103_103390


namespace complex_expression_l103_103675

theorem complex_expression (z : ℂ) (h : z = (i + 1) / (i - 1)) : z^2 + z + 1 = -i := 
by 
  sorry

end complex_expression_l103_103675


namespace twelve_pow_mn_eq_P_pow_2n_Q_pow_m_l103_103146

variable {m n : ℕ}

def P (m : ℕ) : ℕ := 2^m
def Q (n : ℕ) : ℕ := 3^n

theorem twelve_pow_mn_eq_P_pow_2n_Q_pow_m (m n : ℕ) : 12^(m * n) = (P m)^(2 * n) * (Q n)^m := 
sorry

end twelve_pow_mn_eq_P_pow_2n_Q_pow_m_l103_103146


namespace find_a_l103_103078

noncomputable def f (x : Real) (a : Real) : Real :=
if h : 0 < x ∧ x < 2 then (Real.log x - a * x) 
else 
if h' : -2 < x ∧ x < 0 then sorry
else 
   sorry

theorem find_a (a : Real) : (∀ x : Real, f x a = - f (-x) a) → (∀ x: Real, (0 < x ∧ x < 2) → f x a = Real.log x - a * x) → a > (1 / 2) → (∀ x: Real, (-2 < x ∧ x < 0) → f x a ≥ 1) → a = 1 := 
sorry

end find_a_l103_103078


namespace cookies_left_l103_103240

theorem cookies_left (days_baking : ℕ) (trays_per_day : ℕ) (cookies_per_tray : ℕ) (frank_eats_per_day : ℕ) (ted_eats_on_sixth_day : ℕ) :
  trays_per_day * cookies_per_tray * days_baking - frank_eats_per_day * days_baking - ted_eats_on_sixth_day = 134 :=
by
  have days_baking := 6
  have trays_per_day := 2
  have cookies_per_tray := 12
  have frank_eats_per_day := 1
  have ted_eats_on_sixth_day := 4
  sorry

end cookies_left_l103_103240


namespace lucas_earnings_l103_103448

-- Declare constants and definitions given in the problem
def dollars_per_window : ℕ := 3
def windows_per_floor : ℕ := 5
def floors : ℕ := 4
def penalty_amount : ℕ := 2
def days_per_period : ℕ := 4
def total_days : ℕ := 12

-- Definition of the number of total windows
def total_windows : ℕ := windows_per_floor * floors

-- Initial earnings before penalties
def initial_earnings : ℕ := total_windows * dollars_per_window

-- Number of penalty periods
def penalty_periods : ℕ := total_days / days_per_period

-- Total penalty amount
def total_penalty : ℕ := penalty_periods * penalty_amount

-- Final earnings after penalties
def final_earnings : ℕ := initial_earnings - total_penalty

-- Proof problem: correct amount Lucas' father will pay
theorem lucas_earnings : final_earnings = 54 :=
by
  sorry

end lucas_earnings_l103_103448


namespace expression_is_product_l103_103313

def not_sum (a x : Int) : Prop :=
  ¬(a + x = -7 * x)

def not_difference (a x : Int) : Prop :=
  ¬(a - x = -7 * x)

def not_quotient (a x : Int) : Prop :=
  ¬(a / x = -7 * x)

theorem expression_is_product (x : Int) : 
  not_sum (-7) x ∧ not_difference (-7) x ∧ not_quotient (-7) x → (-7 * x = -7 * x) :=
by sorry

end expression_is_product_l103_103313


namespace judy_hits_percentage_l103_103231

theorem judy_hits_percentage 
  (total_hits : ℕ)
  (home_runs : ℕ)
  (triples : ℕ)
  (doubles : ℕ)
  (single_hits_percentage : ℚ) :
  total_hits = 35 →
  home_runs = 1 →
  triples = 1 →
  doubles = 5 →
  single_hits_percentage = (total_hits - (home_runs + triples + doubles)) / total_hits * 100 →
  single_hits_percentage = 80 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end judy_hits_percentage_l103_103231


namespace allergic_reaction_probability_is_50_percent_l103_103437

def can_have_allergic_reaction (choice : String) : Prop :=
  choice = "peanut_butter"

def percentage_of_allergic_reaction :=
  let total_peanut_butter := 40 + 30
  let total_cookies := 40 + 50 + 30 + 20
  (total_peanut_butter : Float) / (total_cookies : Float) * 100

theorem allergic_reaction_probability_is_50_percent :
  percentage_of_allergic_reaction = 50 := sorry

end allergic_reaction_probability_is_50_percent_l103_103437


namespace choose_three_of_nine_l103_103272

def combination (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem choose_three_of_nine : combination 9 3 = 84 :=
by 
  sorry

end choose_three_of_nine_l103_103272


namespace small_pizza_slices_correct_l103_103514

-- Defining the total number of people involved
def people_count : ℕ := 3

-- Defining the number of slices each person can eat
def slices_per_person : ℕ := 12

-- Calculating the total number of slices needed based on the number of people and slices per person
def total_slices_needed : ℕ := people_count * slices_per_person

-- Defining the number of slices in a large pizza
def large_pizza_slices : ℕ := 14

-- Defining the number of large pizzas ordered
def large_pizzas_count : ℕ := 2

-- Calculating the total number of slices provided by the large pizzas
def total_large_pizza_slices : ℕ := large_pizza_slices * large_pizzas_count

-- Defining the number of slices in a small pizza
def small_pizza_slices : ℕ := 8

-- Total number of slices provided needs to be at least the total slices needed
theorem small_pizza_slices_correct :
  total_slices_needed ≤ total_large_pizza_slices + small_pizza_slices := by
  sorry

end small_pizza_slices_correct_l103_103514


namespace combined_function_is_linear_l103_103211

def original_parabola (x : ℝ) : ℝ := 3 * x^2 + 4 * x - 5

def reflected_parabola (x : ℝ) : ℝ := -original_parabola x

def translated_original_parabola (x : ℝ) : ℝ := 3 * (x - 4)^2 + 4 * (x - 4) - 5

def translated_reflected_parabola (x : ℝ) : ℝ := -3 * (x + 6)^2 - 4 * (x + 6) + 5

def combined_function (x : ℝ) : ℝ := translated_original_parabola x + translated_reflected_parabola x

theorem combined_function_is_linear : ∃ (a b : ℝ), ∀ x : ℝ, combined_function x = a * x + b := by
  sorry

end combined_function_is_linear_l103_103211


namespace chord_of_ellipse_bisected_by_point_l103_103811

theorem chord_of_ellipse_bisected_by_point :
  ∀ (x y : ℝ),
  (∃ (x₁ x₂ y₁ y₂ : ℝ), 
    ( (x₁ + x₂) / 2 = 4 ∧ (y₁ + y₂) / 2 = 2) ∧ 
    (x₁^2 / 36 + y₁^2 / 9 = 1) ∧ 
    (x₂^2 / 36 + y₂^2 / 9 = 1)) →
  (x + 2 * y = 8) :=
by
  sorry

end chord_of_ellipse_bisected_by_point_l103_103811


namespace completing_the_square_solution_l103_103181

theorem completing_the_square_solution : ∀ x : ℝ, x^2 + 8 * x + 9 = 0 ↔ (x + 4)^2 = 7 := by
  sorry

end completing_the_square_solution_l103_103181


namespace complex_multiply_cis_l103_103389

open Complex

theorem complex_multiply_cis :
  (4 * (cos (25 * Real.pi / 180) + sin (25 * Real.pi / 180) * I)) *
  (-3 * (cos (48 * Real.pi / 180) + sin (48 * Real.pi / 180) * I)) =
  12 * (cos (253 * Real.pi / 180) + sin (253 * Real.pi / 180) * I) :=
sorry

end complex_multiply_cis_l103_103389


namespace max_ounces_among_items_l103_103506

theorem max_ounces_among_items
  (budget : ℝ)
  (candy_cost : ℝ)
  (candy_ounces : ℝ)
  (candy_stock : ℕ)
  (chips_cost : ℝ)
  (chips_ounces : ℝ)
  (chips_stock : ℕ)
  : budget = 7 → candy_cost = 1.25 → candy_ounces = 12 →
    candy_stock = 5 → chips_cost = 1.40 → chips_ounces = 17 → chips_stock = 4 →
    max (min ((budget / candy_cost) * candy_ounces) (candy_stock * candy_ounces))
        (min ((budget / chips_cost) * chips_ounces) (chips_stock * chips_ounces)) = 68 := 
by
  intros h_budget h_candy_cost h_candy_ounces h_candy_stock h_chips_cost h_chips_ounces h_chips_stock
  sorry

end max_ounces_among_items_l103_103506


namespace P_on_QR_l103_103848

theorem P_on_QR (P Q R : ℝ × ℝ) 
  (h : ∀ X : ℝ × ℝ, dist P X < dist Q X ∨ dist P X < dist R X) : 
  ∃ t ∈ Icc (0:ℝ) 1, P = t • Q + (1 - t) • R :=
by
  -- Proof
  sorry  -- Proof omitted

end P_on_QR_l103_103848


namespace area_of_square_adjacent_vertices_l103_103477

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem area_of_square_adjacent_vertices : 
  distance 1 3 5 6 ^ 2 = 25 :=
by
  let side_length := distance 1 3 5 6
  show side_length ^ 2 = 25
  sorry

end area_of_square_adjacent_vertices_l103_103477


namespace expression_evaluation_l103_103487

theorem expression_evaluation : 4 * (9 - 6) / 2 - 3 = 3 := 
by
  sorry

end expression_evaluation_l103_103487


namespace roses_picked_second_time_l103_103499

-- Define the initial conditions
def initial_roses : ℝ := 37.0
def first_pick : ℝ := 16.0
def total_roses_after_second_picking : ℝ := 72.0

-- Define the calculation after the first picking
def roses_after_first_picking : ℝ := initial_roses + first_pick

-- The Lean statement to prove the number of roses picked the second time
theorem roses_picked_second_time : total_roses_after_second_picking - roses_after_first_picking = 19.0 := 
by
  -- Use the facts stated in the conditions
  sorry

end roses_picked_second_time_l103_103499


namespace eggs_left_after_taking_l103_103002

def eggs_in_box_initial : Nat := 47
def eggs_taken_by_Harry : Nat := 5
theorem eggs_left_after_taking : eggs_in_box_initial - eggs_taken_by_Harry = 42 := 
by
  -- Proof placeholder
  sorry

end eggs_left_after_taking_l103_103002


namespace M_inter_N_eq_l103_103580

def M : Set ℝ := {x | -1/2 < x ∧ x < 1/2}
def N : Set ℝ := {x | x^2 ≤ x}

theorem M_inter_N_eq : (M ∩ N) = Set.Ico 0 (1/2) := 
by
  sorry

end M_inter_N_eq_l103_103580


namespace alphabet_letters_l103_103994

theorem alphabet_letters (DS S_only Total D_only : ℕ) 
  (h_DS : DS = 9) 
  (h_S_only : S_only = 24) 
  (h_Total : Total = 40) 
  (h_eq : Total = D_only + S_only + DS) 
  : D_only = 7 := 
by
  sorry

end alphabet_letters_l103_103994


namespace length_of_BC_l103_103672

theorem length_of_BC (x : ℝ) (h1 : (20 * x^2) / 3 - (400 * x) / 3 = 140) :
  ∃ (BC : ℝ), BC = 29 := 
by
  sorry

end length_of_BC_l103_103672


namespace kite_area_overlap_l103_103008

theorem kite_area_overlap (beta : Real) (h_beta : beta ≠ 0 ∧ beta ≠ π) : 
  ∃ (A : Real), A = 1 / Real.sin beta := by
  sorry

end kite_area_overlap_l103_103008


namespace find_x_l103_103557

theorem find_x
  (x : ℝ)
  (h : 5^29 * x^15 = 2 * 10^29) :
  x = 4 :=
by
  sorry

end find_x_l103_103557


namespace oranges_ratio_l103_103602

theorem oranges_ratio (initial_oranges_kgs : ℕ) (additional_oranges_kgs : ℕ) (total_oranges_three_weeks : ℕ) :
  initial_oranges_kgs = 10 →
  additional_oranges_kgs = 5 →
  total_oranges_three_weeks = 75 →
  (2 * (total_oranges_three_weeks - (initial_oranges_kgs + additional_oranges_kgs)) / 2) / (initial_oranges_kgs + additional_oranges_kgs) = 2 :=
by
  intros h_initial h_additional h_total
  sorry

end oranges_ratio_l103_103602


namespace tiles_needed_l103_103435

def hallway_length : ℕ := 14
def hallway_width : ℕ := 20
def border_tile_side : ℕ := 2
def interior_tile_side : ℕ := 3

theorem tiles_needed :
  let border_length_tiles := ((hallway_length - 2 * border_tile_side) / border_tile_side) * 2
  let border_width_tiles := ((hallway_width - 2 * border_tile_side) / border_tile_side) * 2
  let corner_tiles := 4
  let total_border_tiles := border_length_tiles + border_width_tiles + corner_tiles
  let interior_length := hallway_length - 2 * border_tile_side
  let interior_width := hallway_width - 2 * border_tile_side
  let interior_area := interior_length * interior_width
  let interior_tiles_needed := (interior_area + interior_tile_side * interior_tile_side - 1) / (interior_tile_side * interior_tile_side)
  total_border_tiles + interior_tiles_needed = 48 := 
by {
  sorry
}

end tiles_needed_l103_103435


namespace percent_not_filler_l103_103756

theorem percent_not_filler (total_weight filler_weight : ℕ) (h1 : total_weight = 180) (h2 : filler_weight = 45) : 
  ((total_weight - filler_weight) * 100 / total_weight = 75) :=
by 
  sorry

end percent_not_filler_l103_103756


namespace correct_sampling_method_is_D_l103_103197

def is_simple_random_sample (method : String) : Prop :=
  method = "drawing lots method to select 3 out of 10 products for quality inspection"

theorem correct_sampling_method_is_D : 
  is_simple_random_sample "drawing lots method to select 3 out of 10 products for quality inspection" :=
sorry

end correct_sampling_method_is_D_l103_103197


namespace fraction_simplification_l103_103396

theorem fraction_simplification (x y z : ℝ) (h : x + y + z ≠ 0) :
  (x^2 + y^2 - z^2 + 2 * x * y) / (x^2 + z^2 - y^2 + 2 * x * z) = (x + y - z) / (x + z - y) :=
by
  sorry

end fraction_simplification_l103_103396


namespace mean_age_is_10_l103_103147

def ages : List ℤ := [7, 7, 7, 14, 15]

theorem mean_age_is_10 : (List.sum ages : ℤ) / (ages.length : ℤ) = 10 := by
-- sorry placeholder for the actual proof
sorry

end mean_age_is_10_l103_103147


namespace probability_diff_grades_l103_103542

section

variable (n10 n11 n12 : ℕ)
variable (total_students selected_students : ℕ)
variable (p_same_grade : ℚ)

-- Conditions
def students_in_grade10 : ℕ := 180
def students_in_grade11 : ℕ := 180
def students_in_grade12 : ℕ := 90
def total_students : ℕ := students_in_grade10 + students_in_grade11 + students_in_grade12
def selected_students : ℕ := 5

-- Number of students selected from each grade
def selected_from_grade10 : ℕ := selected_students * students_in_grade10 / total_students
def selected_from_grade11 : ℕ := selected_students * students_in_grade11 / total_students
def selected_from_grade12 : ℕ := selected_students * students_in_grade12 / total_students

-- Probability that both students are from the same grade
def P_same_grade : ℚ :=
  (selected_from_grade10 * (selected_from_grade10 - 1)) / (selected_students * (selected_students - 1))
  + (selected_from_grade11 * (selected_from_grade11 - 1)) / (selected_students * (selected_students - 1))

-- Probability that the 2 selected students are from different grades
def P_diff_grades : ℚ := 1 - P_same_grade

-- Theorem statement
theorem probability_diff_grades : P_diff_grades = 4 / 5 := by
  -- This is where the proof would go
  sorry

end

end probability_diff_grades_l103_103542


namespace matrix_exp_1000_l103_103904

-- Define the matrix as a constant
noncomputable def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0], ![2, 1]]

-- The property of matrix exponentiation
theorem matrix_exp_1000 :
  A^1000 = ![![1, 0], ![2000, 1]] :=
by
  sorry

end matrix_exp_1000_l103_103904


namespace average_of_numbers_between_6_and_36_divisible_by_7_l103_103200

noncomputable def average_of_divisibles_by_seven : ℕ :=
  let numbers := [7, 14, 21, 28, 35]
  let sum := numbers.sum
  let count := numbers.length
  sum / count

theorem average_of_numbers_between_6_and_36_divisible_by_7 : average_of_divisibles_by_seven = 21 :=
by
  sorry

end average_of_numbers_between_6_and_36_divisible_by_7_l103_103200


namespace horizon_distance_ratio_l103_103630

def R : ℝ := 6000000
def h1 : ℝ := 1
def h2 : ℝ := 2

noncomputable def distance_to_horizon (R h : ℝ) : ℝ :=
  Real.sqrt (2 * R * h)

noncomputable def d1 : ℝ := distance_to_horizon R h1
noncomputable def d2 : ℝ := distance_to_horizon R h2

theorem horizon_distance_ratio : d2 / d1 = Real.sqrt 2 :=
  sorry

end horizon_distance_ratio_l103_103630


namespace inequality_holds_equality_cases_l103_103721

noncomputable def posReal : Type := { x : ℝ // 0 < x }

variables (a b c d : posReal)

theorem inequality_holds (a b c d : posReal) :
  (a.1 - b.1) * (a.1 - c.1) / (a.1 + b.1 + c.1) +
  (b.1 - c.1) * (b.1 - d.1) / (b.1 + c.1 + d.1) +
  (c.1 - d.1) * (c.1 - a.1) / (c.1 + d.1 + a.1) +
  (d.1 - a.1) * (d.1 - b.1) / (d.1 + a.1 + b.1) ≥ 0 :=
sorry

theorem equality_cases (a b c d : posReal) :
  (a.1 - b.1) * (a.1 - c.1) / (a.1 + b.1 + c.1) +
  (b.1 - c.1) * (b.1 - d.1) / (b.1 + c.1 + d.1) +
  (c.1 - d.1) * (c.1 - a.1) / (c.1 + d.1 + a.1) +
  (d.1 - a.1) * (d.1 - b.1) / (d.1 + a.1 + b.1) = 0 ↔
  (a.1 = c.1 ∧ b.1 = d.1) :=
sorry

end inequality_holds_equality_cases_l103_103721


namespace recurring_subtraction_l103_103657

theorem recurring_subtraction (x y : ℚ) (h1 : x = 35 / 99) (h2 : y = 7 / 9) : x - y = -14 / 33 := by
  sorry

end recurring_subtraction_l103_103657


namespace cube_triangulation_impossible_l103_103518

theorem cube_triangulation_impossible (vertex_sum : ℝ) (triangle_inter_sum : ℝ) (triangle_sum : ℝ) :
  vertex_sum = 270 ∧ triangle_inter_sum = 360 ∧ triangle_sum = 180 → ∃ (n : ℕ), n = 3 ∧ ∀ (m : ℕ), m ≠ 3 → false :=
by
  sorry

end cube_triangulation_impossible_l103_103518


namespace complete_square_l103_103188

theorem complete_square (x : ℝ) (h : x^2 + 8 * x + 9 = 0) : (x + 4)^2 = 7 := by
  sorry

end complete_square_l103_103188


namespace total_cost_correct_l103_103587

-- Define the cost of each category of items
def cost_of_book : ℕ := 16
def cost_of_binders : ℕ := 3 * 2
def cost_of_notebooks : ℕ := 6 * 1

-- Define the total cost calculation
def total_cost : ℕ := cost_of_book + cost_of_binders + cost_of_notebooks

-- Prove that the total cost of Léa's purchases is 28
theorem total_cost_correct : total_cost = 28 :=
by {
  -- This is where the proof would go, but it's omitted for now.
  sorry
}

end total_cost_correct_l103_103587


namespace bens_old_car_cost_l103_103772

theorem bens_old_car_cost :
  ∃ (O N : ℕ), N = 2 * O ∧ O = 1800 ∧ N = 1800 + 2000 ∧ O = 1900 :=
by 
  sorry

end bens_old_car_cost_l103_103772


namespace root_in_interval_l103_103975

noncomputable def f (x : ℝ) : ℝ := Real.log x - x + 2

theorem root_in_interval : ∃ x ∈ Set.Ioo (3 : ℝ) (4 : ℝ), f x = 0 := sorry

end root_in_interval_l103_103975


namespace communication_scenarios_l103_103051

theorem communication_scenarios
  (nA : ℕ) (nB : ℕ) (hA : nA = 10) (hB : nB = 20) : 
  (∃ scenarios : ℕ, scenarios = 2 ^ (nA * nB)) :=
by
  use 2 ^ (10 * 20)
  sorry

end communication_scenarios_l103_103051


namespace apples_found_l103_103708

theorem apples_found (start_apples : ℕ) (end_apples : ℕ) (h_start : start_apples = 7) (h_end : end_apples = 81) : 
  end_apples - start_apples = 74 := 
by 
  sorry

end apples_found_l103_103708


namespace complete_the_square_l103_103177

theorem complete_the_square (x : ℝ) :
  (x^2 + 8*x + 9 = 0) ↔ ((x + 4)^2 = 7) :=
sorry

end complete_the_square_l103_103177


namespace smallest_constant_for_triangle_sides_l103_103066

theorem smallest_constant_for_triangle_sides (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (triangle_condition : a + b > c ∧ b + c > a ∧ c + a > b) :
  ∃ N, (∀ a b c, (a + b > c ∧ b + c > a ∧ c + a > b) → (a^2 + b^2) / (a * b) < N) ∧ N = 2 := by
  sorry

end smallest_constant_for_triangle_sides_l103_103066


namespace correct_growth_rate_equation_l103_103763

-- Define the conditions
def packages_first_day := 200
def packages_third_day := 242

-- Define the average daily growth rate
variable (x : ℝ)

-- State the theorem to prove
theorem correct_growth_rate_equation :
  packages_first_day * (1 + x)^2 = packages_third_day :=
by
  sorry

end correct_growth_rate_equation_l103_103763


namespace volume_conversion_l103_103369

theorem volume_conversion (V_ft : ℕ) (h_V : V_ft = 216) (conversion_factor : ℕ) (h_cf : conversion_factor = 27) :
  V_ft / conversion_factor = 8 :=
by
  sorry

end volume_conversion_l103_103369


namespace price_of_soda_l103_103610

theorem price_of_soda (regular_price_per_can : ℝ) (case_discount : ℝ) (bulk_discount : ℝ) (num_cases : ℕ) (num_cans : ℕ) :
  regular_price_per_can = 0.15 →
  case_discount = 0.12 →
  bulk_discount = 0.05 →
  num_cases = 3 →
  num_cans = 75 →
  (num_cans * ((regular_price_per_can * (1 - case_discount)) * (1 - bulk_discount))) = 9.405 :=
by
  intros h1 h2 h3 h4 h5
  -- normal price per can
  have hp1 : ℝ := regular_price_per_can
  -- price after case discount
  have hp2 : ℝ := hp1 * (1 - case_discount)
  -- price after bulk discount
  have hp3 : ℝ := hp2 * (1 - bulk_discount)
  -- total price
  have total_price : ℝ := num_cans * hp3
  -- goal
  sorry -- skip the proof, as only the statement is needed.

end price_of_soda_l103_103610


namespace graph_of_equation_l103_103747

theorem graph_of_equation (x y : ℝ) : (x - y)^2 = x^2 + y^2 ↔ (x = 0 ∨ y = 0) :=
by
  sorry

end graph_of_equation_l103_103747


namespace polynomial_expansion_correct_l103_103676

theorem polynomial_expansion_correct :
  let p := (x - 2)^8 in
  p.coeff 0 = 256 ∧
  p.coeff 8 = 1 ∧
  (∑ i in Finset.range 8, p.coeff (i + 1)) = -255 ∧
  (Finset.sum (Finset.range 9) (λ i, p.coeff i * if i % 2 = 0 then 1 else -1)) = 6561 :=
by
  sorry

end polynomial_expansion_correct_l103_103676


namespace solve_for_b_l103_103974

theorem solve_for_b (b : ℝ) : (∃ y x : ℝ, 4 * y - 2 * x - 6 = 0 ∧ 5 * y + b * x + 1 = 0) → b = 10 :=
by sorry

end solve_for_b_l103_103974


namespace min_generic_tees_per_package_l103_103901

def total_golf_tees_needed (n : ℕ) : ℕ := 80
def max_generic_packages_used : ℕ := 2
def tees_per_aero_flight_package : ℕ := 2
def aero_flight_packages_needed : ℕ := 28
def total_tees_from_aero_flight_packages (n : ℕ) : ℕ := aero_flight_packages_needed * tees_per_aero_flight_package

theorem min_generic_tees_per_package (G : ℕ) :
  (total_golf_tees_needed 4) - (total_tees_from_aero_flight_packages aero_flight_packages_needed) ≤ max_generic_packages_used * G → G ≥ 12 :=
by
  sorry

end min_generic_tees_per_package_l103_103901


namespace pete_should_leave_by_0730_l103_103720

def walking_time : ℕ := 10
def train_time : ℕ := 80
def latest_arrival_time : String := "0900"
def departure_time : String := "0730"

theorem pete_should_leave_by_0730 :
  (latest_arrival_time = "0900" → walking_time = 10 ∧ train_time = 80 → departure_time = "0730") := by
  sorry

end pete_should_leave_by_0730_l103_103720


namespace remaining_stock_weighs_120_l103_103582

noncomputable def total_remaining_weight (green_beans_weight rice_weight sugar_weight : ℕ) :=
  let remaining_rice := rice_weight - (rice_weight / 3)
  let remaining_sugar := sugar_weight - (sugar_weight / 5)
  let remaining_stock := remaining_rice + remaining_sugar + green_beans_weight
  remaining_stock

theorem remaining_stock_weighs_120 : total_remaining_weight 60 30 50 = 120 :=
by
  have h1: 60 - 30 = 30 := by norm_num
  have h2: 60 - 10 = 50 := by norm_num
  have h3: 30 - (30 / 3) = 20 := by norm_num
  have h4: 50 - (50 / 5) = 40 := by norm_num
  have h5: 20 + 40 + 60 = 120 := by norm_num
  exact h5

end remaining_stock_weighs_120_l103_103582


namespace graph_of_equation_l103_103746

theorem graph_of_equation (x y : ℝ) : 
  (x - y)^2 = x^2 + y^2 ↔ (x = 0 ∨ y = 0) := 
by 
  sorry

end graph_of_equation_l103_103746


namespace factorize_x4_minus_16y4_l103_103525

theorem factorize_x4_minus_16y4 (x y : ℚ) : 
  x^4 - 16 * y^4 = (x^2 + 4 * y^2) * (x + 2 * y) * (x - 2 * y) := 
by 
  sorry

end factorize_x4_minus_16y4_l103_103525


namespace relationship_among_sets_l103_103840

-- Definitions based on the conditions
def RegularQuadrilateralPrism (x : Type) : Prop := -- prisms with a square base and perpendicular lateral edges
  sorry

def RectangularPrism (x : Type) : Prop := -- prisms with a rectangular base and perpendicular lateral edges
  sorry

def RightQuadrilateralPrism (x : Type) : Prop := -- prisms whose lateral edges are perpendicular to the base, and the base can be any quadrilateral
  sorry

def RightParallelepiped (x : Type) : Prop := -- prisms with lateral edges perpendicular to the base
  sorry

-- Sets
def M : Set Type := { x | RegularQuadrilateralPrism x }
def P : Set Type := { x | RectangularPrism x }
def N : Set Type := { x | RightQuadrilateralPrism x }
def Q : Set Type := { x | RightParallelepiped x }

-- Proof problem statement
theorem relationship_among_sets : M ⊂ P ∧ P ⊂ Q ∧ Q ⊂ N := 
  by
    sorry

end relationship_among_sets_l103_103840


namespace board_divisible_into_hexominos_l103_103498

theorem board_divisible_into_hexominos {m n : ℕ} (h_m_gt_5 : m > 5) (h_n_gt_5 : n > 5) 
  (h_m_div_by_3 : m % 3 = 0) (h_n_div_by_4 : n % 4 = 0) : 
  (m * n) % 6 = 0 :=
by
  sorry

end board_divisible_into_hexominos_l103_103498


namespace exterior_angle_measure_l103_103083

theorem exterior_angle_measure (sum_interior_angles : ℝ) (h : sum_interior_angles = 1260) :
  ∃ (n : ℕ) (d : ℝ), (n - 2) * 180 = sum_interior_angles ∧ d = 360 / n ∧ d = 40 := 
by
  sorry

end exterior_angle_measure_l103_103083


namespace total_pounds_of_peppers_l103_103905

-- Definitions and conditions
def green_peppers : ℝ := 2.8333333333333335
def red_peppers : ℝ := 2.8333333333333335

-- Theorem statement
theorem total_pounds_of_peppers : green_peppers + red_peppers = 5.666666666666667 :=
by
  sorry

end total_pounds_of_peppers_l103_103905


namespace count_valid_three_digit_numbers_l103_103422

theorem count_valid_three_digit_numbers : 
  let total_three_digit_numbers := 900 
  let invalid_AAB_or_ABA := 81 + 81
  total_three_digit_numbers - invalid_AAB_or_ABA = 738 := 
by 
  let total_three_digit_numbers := 900
  let invalid_AAB_or_ABA := 81 + 81
  show total_three_digit_numbers - invalid_AAB_or_ABA = 738 
  sorry

end count_valid_three_digit_numbers_l103_103422


namespace smallest_positive_period_f_intervals_monotonically_increasing_f_l103_103812

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x) * (Real.sin x + Real.cos x)

-- 1. Proving the smallest positive period is π
theorem smallest_positive_period_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = Real.pi := 
sorry

-- 2. Proving the intervals where the function is monotonically increasing
theorem intervals_monotonically_increasing_f : 
  ∀ k : ℤ, ∀ x : ℝ, x ∈ Set.Icc (k * Real.pi - (3 * Real.pi / 8)) (k * Real.pi + (Real.pi / 8)) → 
    0 < deriv f x :=
sorry

end smallest_positive_period_f_intervals_monotonically_increasing_f_l103_103812


namespace triangle_inequality_l103_103841

variables {l_a l_b l_c m_a m_b m_c h_n m_n h_h_n m_m_p : ℝ}

-- Assuming some basic properties for the variables involved (all are positive in their respective triangle context)
axiom pos_l_a : 0 < l_a
axiom pos_l_b : 0 < l_b
axiom pos_l_c : 0 < l_c
axiom pos_m_a : 0 < m_a
axiom pos_m_b : 0 < m_b
axiom pos_m_c : 0 < m_c
axiom pos_h_n : 0 < h_n
axiom pos_m_n : 0 < m_n
axiom pos_h_h_n : 0 < h_h_n
axiom pos_m_m_p : 0 < m_m_p

theorem triangle_inequality :
  (h_n / m_n) + (h_n / h_h_n) + (l_c / m_m_p) > 1 :=
sorry

end triangle_inequality_l103_103841


namespace aviana_brought_pieces_l103_103701

variable (total_people : ℕ) (fraction_eat_pizza : ℚ) (pieces_per_person : ℕ) (remaining_pieces : ℕ)

theorem aviana_brought_pieces (h1 : total_people = 15) 
                             (h2 : fraction_eat_pizza = 3 / 5) 
                             (h3 : pieces_per_person = 4) 
                             (h4 : remaining_pieces = 14) :
                             ∃ (brought_pieces : ℕ), brought_pieces = 50 :=
by sorry

end aviana_brought_pieces_l103_103701


namespace erika_sum_prob_l103_103656

-- Define the problem conditions and required types.
def age := 16
def coin_outcome := {10, 25}
def die_outcome := {1, 2, 3, 4, 5, 6}
def fair_coin_prob := (1 : ℚ) / 2
def die_prob := (1 : ℚ) / 6

-- The main theorem to prove the stated probability.
theorem erika_sum_prob : (∑ (coin : ℕ) in coin_outcome, 
                          if coin = 10 then fair_coin_prob * die_prob else 0) = (1 : ℚ) / 12 := 
by sorry

end erika_sum_prob_l103_103656


namespace fish_disappeared_l103_103290

theorem fish_disappeared (g : ℕ) (c : ℕ) (left : ℕ) (disappeared : ℕ) (h₁ : g = 7) (h₂ : c = 12) (h₃ : left = 15) (h₄ : g + c - left = disappeared) : disappeared = 4 :=
by
  sorry

end fish_disappeared_l103_103290


namespace lowest_price_for_16_oz_butter_l103_103383

-- Define the constants
def price_single_16_oz_package : ℝ := 7
def price_8_oz_package : ℝ := 4
def price_4_oz_package : ℝ := 2
def discount_4_oz_package : ℝ := 0.5

-- Calculate the discounted price for a 4 oz package
def discounted_price_4_oz_package : ℝ := price_4_oz_package * discount_4_oz_package

-- Calculate the total price for two discounted 4 oz packages
def total_price_two_discounted_4_oz_packages : ℝ := 2 * discounted_price_4_oz_package

-- Calculate the total price using the 8 oz package and two discounted 4 oz packages
def total_price_using_coupon : ℝ := price_8_oz_package + total_price_two_discounted_4_oz_packages

-- State the property to prove
theorem lowest_price_for_16_oz_butter :
  min price_single_16_oz_package total_price_using_coupon = 6 :=
sorry

end lowest_price_for_16_oz_butter_l103_103383


namespace graphs_symmetric_y_axis_l103_103489

theorem graphs_symmetric_y_axis : ∀ (x : ℝ), (-x) ∈ { y | y = 3^(-x) } ↔ x ∈ { y | y = 3^x } :=
by
  intro x
  sorry

end graphs_symmetric_y_axis_l103_103489


namespace equal_utilities_l103_103442

-- Conditions
def utility (juggling coding : ℕ) : ℕ := juggling * coding

def wednesday_utility (s : ℕ) : ℕ := utility s (12 - s)
def thursday_utility (s : ℕ) : ℕ := utility (6 - s) (s + 4)

-- Theorem
theorem equal_utilities (s : ℕ) (h : wednesday_utility s = thursday_utility s) : s = 12 / 5 := 
by sorry

end equal_utilities_l103_103442


namespace minimize_std_deviation_l103_103740

theorem minimize_std_deviation (m n : ℝ) (h1 : m + n = 32) 
    (h2 : 11 ≤ 12 ∧ 12 ≤ m ∧ m ≤ n ∧ n ≤ 20 ∧ 20 ≤ 27) : 
    m = 16 :=
by {
  -- No proof required, only the theorem statement as per instructions
  sorry
}

end minimize_std_deviation_l103_103740


namespace isosceles_triangle_base_l103_103041

theorem isosceles_triangle_base (h_perimeter : 2 * 1.5 + x = 3.74) : x = 0.74 :=
by
  sorry

end isosceles_triangle_base_l103_103041


namespace rectangle_relationships_l103_103353

theorem rectangle_relationships (x y S : ℝ) (h1 : 2 * x + 2 * y = 10) (h2 : S = x * y) :
  y = 5 - x ∧ S = 5 * x - x ^ 2 :=
by
  sorry

end rectangle_relationships_l103_103353


namespace jill_more_than_jake_l103_103950

-- Definitions from conditions
def jill_peaches := 12
def steven_peaches := jill_peaches + 15
def jake_peaches := steven_peaches - 16

-- Theorem to prove the question == answer given conditions
theorem jill_more_than_jake : jill_peaches - jake_peaches = 1 :=
by
  -- Proof steps would be here, but for the statement requirement we put sorry
  sorry

end jill_more_than_jake_l103_103950


namespace constant_term_in_binomial_expansion_l103_103112

theorem constant_term_in_binomial_expansion 
  (a b : ℕ) (n : ℕ)
  (sum_of_coefficients : (1 + 1)^n = 4)
  (A B : ℕ)
  (sum_A_B : A + B = 72) 
  (A_value : A = 4) :
  (b^2 = 9) :=
by sorry

end constant_term_in_binomial_expansion_l103_103112


namespace Sam_balloon_count_l103_103533

theorem Sam_balloon_count:
  ∀ (F M S : ℕ), F = 5 → M = 7 → (F + M + S = 18) → S = 6 :=
by
  intros F M S hF hM hTotal
  rw [hF, hM] at hTotal
  linarith

end Sam_balloon_count_l103_103533


namespace proof1_proof2_monotonically_increasing_interval_l103_103090

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (1, Real.sin x)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (Real.cos (2 * x + Real.pi / 3), Real.sin x)
noncomputable def f (x : ℝ) : ℝ := (vector_a x).fst * (vector_b x).fst + (vector_a x).snd * (vector_b x).snd - 0.5 * Real.cos (2 * x)

theorem proof1 : ∀ x : ℝ, f x = -Real.sin (2 * x + Real.pi / 6) + 0.5 :=
sorry

theorem proof2 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 3 → -0.5 ≤ f x ∧ f x ≤ 0 :=
sorry

theorem monotonically_increasing_interval (k : ℤ) : 
∃ lb ub : ℝ, lb = Real.pi / 6 + k * Real.pi ∧ ub = 2 * Real.pi / 3 + k * Real.pi ∧ ∀ x : ℝ, lb ≤ x ∧ x ≤ ub → f x = -Real.sin (2 * x + Real.pi / 6) + 0.5 :=
sorry

end proof1_proof2_monotonically_increasing_interval_l103_103090


namespace area_BCD_l103_103705

open Real EuclideanGeometry

noncomputable def point := (ℝ × ℝ)
noncomputable def A : point := (0, 0)
noncomputable def B : point := (10, 24)
noncomputable def C : point := (30, 0)
noncomputable def D : point := (40, 0)

def area_triangle (p1 p2 p3 : point) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  0.5 * |x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)|

theorem area_BCD : area_triangle B C D = 12 := sorry

end area_BCD_l103_103705


namespace gcd_16016_20020_l103_103661

theorem gcd_16016_20020 : Int.gcd 16016 20020 = 4004 :=
by
  sorry

end gcd_16016_20020_l103_103661


namespace prove_geomSeqSumFirst3_l103_103280

noncomputable def geomSeqSumFirst3 {a₁ a₆ : ℕ} (h₁ : a₁ = 1) (h₂ : a₆ = 32) : ℕ :=
  let r := 2 -- since r^5 = 32 which means r = 2
  let S3 := a₁ * (1 - r^3) / (1 - r)
  S3

theorem prove_geomSeqSumFirst3 : 
  geomSeqSumFirst3 (h₁ : 1 = 1) (h₂ : 32 = 32) = 7 := by
  sorry

end prove_geomSeqSumFirst3_l103_103280


namespace ratio_of_percent_changes_l103_103989

noncomputable def price_decrease_ratio (original_price : ℝ) (new_price : ℝ) : ℝ :=
(original_price - new_price) / original_price * 100

noncomputable def units_increase_ratio (original_units : ℝ) (new_units : ℝ) : ℝ :=
(new_units - original_units) / original_units * 100

theorem ratio_of_percent_changes 
  (original_price new_price original_units new_units : ℝ)
  (h1 : new_price = 0.7 * original_price)
  (h2 : original_price * original_units = new_price * new_units)
  : (units_increase_ratio original_units new_units) / (price_decrease_ratio original_price new_price) = 1.4285714285714286 :=
by
  sorry

end ratio_of_percent_changes_l103_103989


namespace total_apartments_in_building_l103_103457

theorem total_apartments_in_building (A k m n : ℕ)
  (cond1 : 5 = A)
  (cond2 : 636 = (m-1) * k + n)
  (cond3 : 242 = (A-m) * k + n) :
  A * k = 985 :=
by
  sorry

end total_apartments_in_building_l103_103457


namespace geom_seq_sum_l103_103415

variable (a : ℕ → ℝ) (r : ℝ) (a1 a4 : ℝ)

theorem geom_seq_sum :
  (∀ n : ℕ, a (n + 1) = a n * r) → r = 2 → a 2 + a 3 = 4 → a 1 + a 4 = 6 :=
by
  sorry

end geom_seq_sum_l103_103415


namespace sum_of_numbers_with_six_zeros_and_56_divisors_l103_103326

theorem sum_of_numbers_with_six_zeros_and_56_divisors :
  ∃ N1 N2 : ℕ, (N1 % 10^6 = 0) ∧ (N2 % 10^6 = 0) ∧ (N1_divisors = 56) ∧ (N2_divisors = 56) ∧ (N1 + N2 = 7000000) :=
by
  sorry

end sum_of_numbers_with_six_zeros_and_56_divisors_l103_103326


namespace parallel_lines_regular_ngon_l103_103777

def closed_n_hop_path (n : ℕ) (a : Fin (n + 1) → Fin n) : Prop :=
∀ i j : Fin n, a (i + 1) + a i = a (j + 1) + a j → i = j

theorem parallel_lines_regular_ngon (n : ℕ) (a : Fin (n + 1) → Fin n):
  (Even n → ∃ i j : Fin n, i ≠ j ∧ a (i + 1) + a i = a (j + 1) + a j) ∧
  (Odd n → ¬(∃ i j : Fin n, i ≠ j ∧ a (i + 1) + a i = a (j + 1) + a j ∧ ∀ k l : Fin n, k ≠ l → a (k + 1) + k ≠ a (l + 1) + l)) :=
by
  sorry

end parallel_lines_regular_ngon_l103_103777


namespace factor_x_squared_minus_sixtyfour_l103_103055

theorem factor_x_squared_minus_sixtyfour (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) :=
by sorry

end factor_x_squared_minus_sixtyfour_l103_103055


namespace sum_six_smallest_multiples_of_12_is_252_l103_103741

-- Define the six smallest positive distinct multiples of 12
def six_smallest_multiples_of_12 := [12, 24, 36, 48, 60, 72]

-- Define the sum problem
def sum_of_six_smallest_multiples_of_12 : Nat :=
  six_smallest_multiples_of_12.foldr (· + ·) 0

-- Main proof statement
theorem sum_six_smallest_multiples_of_12_is_252 :
  sum_of_six_smallest_multiples_of_12 = 252 :=
by
  sorry

end sum_six_smallest_multiples_of_12_is_252_l103_103741


namespace find_perfect_matching_l103_103246

-- Define the boys and girls
inductive Boy | B1 | B2 | B3
inductive Girl | G1 | G2 | G3

-- Define the knowledge relationship
def knows : Boy → Girl → Prop
| Boy.B1, Girl.G1 => true
| Boy.B1, Girl.G2 => true
| Boy.B2, Girl.G1 => true
| Boy.B2, Girl.G3 => true
| Boy.B3, Girl.G2 => true
| Boy.B3, Girl.G3 => true
| _, _ => false

-- Proposition to prove
theorem find_perfect_matching :
  ∃ (pairing : Boy → Girl), 
    (∀ b : Boy, knows b (pairing b)) ∧ 
    (∀ g : Girl, ∃ b : Boy, pairing b = g) :=
by
  sorry

end find_perfect_matching_l103_103246


namespace proof_problem_l103_103195

def x : ℝ := 0.80 * 1750
def y : ℝ := 0.35 * 3000
def z : ℝ := 0.60 * 4500
def w : ℝ := 0.40 * 2800
def a : ℝ := z * w
def b : ℝ := x + y

theorem proof_problem : a - b = 3021550 := by
  sorry

end proof_problem_l103_103195


namespace Mikaela_initially_planned_walls_l103_103959

/-- 
Mikaela bought 16 containers of paint to cover a certain number of equally-sized walls in her bathroom.
At the last minute, she decided to put tile on one wall and paint flowers on the ceiling with one 
container of paint instead. She had 3 containers of paint left over. 
Prove she initially planned to paint 13 walls.
-/
theorem Mikaela_initially_planned_walls
  (PaintContainers : ℕ)
  (CeilingPaint : ℕ)
  (LeftOverPaint : ℕ)
  (TiledWalls : ℕ) : PaintContainers = 16 → CeilingPaint = 1 → LeftOverPaint = 3 → TiledWalls = 1 → 
    (PaintContainers - CeilingPaint - LeftOverPaint + TiledWalls = 13) :=
by
  -- Given conditions:
  intros h1 h2 h3 h4
  -- Proof goes here.
  sorry

end Mikaela_initially_planned_walls_l103_103959


namespace find_x_l103_103691

theorem find_x (x y : ℚ) (h1 : 3 * x - 2 * y = 7) (h2 : x + 3 * y = 8) : x = 37 / 11 := by
  sorry

end find_x_l103_103691


namespace polyhedra_impossible_l103_103940

noncomputable def impossible_polyhedra_projections (p1_outer : List (ℝ × ℝ)) (p1_inner : List (ℝ × ℝ))
                                                  (p2_outer : List (ℝ × ℝ)) (p2_inner : List (ℝ × ℝ)) : Prop :=
  -- Add definitions for the vertices labeling here 
  let vertices_outer := ["A", "B", "C", "D"]
  let vertices_inner := ["A1", "B1", "C1", "D1"]
  -- Add the conditions for projection (a) and (b) 
  p1_outer = [(0,0), (1,0), (1,1), (0,1)] ∧
  p1_inner = [(0.25,0.25), (0.75,0.25), (0.75,0.75), (0.25,0.75)] ∧
  p2_outer = [(0,0), (1,0), (1,1), (0,1)] ∧
  p2_inner = [(0.25,0.25), (0.75,0.25), (0.75,0.75), (0.25,0.75)] →
  -- Prove that the polyhedra corresponding to these projections are impossible.
  false

-- Now let's state the theorem
theorem polyhedra_impossible : impossible_polyhedra_projections [(0,0), (1,0), (1,1), (0,1)] 
                                                                [(0.25,0.25), (0.75,0.25), (0.75,0.75), (0.25,0.75)]
                                                                [(0,0), (1,0), (1,1), (0,1)]
                                                                [(0.25,0.25), (0.75,0.25), (0.75,0.75), (0.25,0.75)] := 
by {
  sorry
}

end polyhedra_impossible_l103_103940


namespace total_earnings_correct_l103_103493

noncomputable def total_earnings (a_days b_days c_days b_share : ℝ) : ℝ :=
  let a_work_per_day := 1 / a_days
  let b_work_per_day := 1 / b_days
  let c_work_per_day := 1 / c_days
  let combined_work_per_day := a_work_per_day + b_work_per_day + c_work_per_day
  let b_fraction_of_total_work := b_work_per_day / combined_work_per_day
  let total_earnings := b_share / b_fraction_of_total_work
  total_earnings

theorem total_earnings_correct :
  total_earnings 6 8 12 780.0000000000001 = 2340 :=
by
  sorry

end total_earnings_correct_l103_103493


namespace slope_of_parallel_line_l103_103889

/-- A line is described by the equation 3x - 6y = 12. The slope of a line 
    parallel to this line is 1/2. -/
theorem slope_of_parallel_line (x y : ℝ) (h : 3 * x - 6 * y = 12) : 
  ∃ m : ℝ, m = 1/2 := by
  sorry

end slope_of_parallel_line_l103_103889


namespace complete_square_l103_103189

theorem complete_square (x : ℝ) (h : x^2 + 8 * x + 9 = 0) : (x + 4)^2 = 7 := by
  sorry

end complete_square_l103_103189


namespace transform_M_eq_l103_103549

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![0, 1/3], ![1, -2/3]]

def M : Fin 2 → ℚ :=
  ![-1, 1]

theorem transform_M_eq :
  A⁻¹.mulVec M = ![-1, -3] :=
by
  sorry

end transform_M_eq_l103_103549


namespace maria_initial_carrots_l103_103126

theorem maria_initial_carrots (C : ℕ) (h : C - 11 + 15 = 52) : C = 48 :=
by
  sorry

end maria_initial_carrots_l103_103126


namespace largest_number_A_l103_103982

theorem largest_number_A (A B C : ℕ) (h1: A = 7 * B + C) (h2: B = C) 
  : A ≤ 48 :=
sorry

end largest_number_A_l103_103982


namespace number_of_triangles_l103_103929

theorem number_of_triangles (n : ℕ) : 
  ∃ k : ℕ, k = ⌊((n + 1) * (n + 3) * (2 * n + 1) : ℝ) / 24⌋ := sorry

end number_of_triangles_l103_103929


namespace flagpole_break_height_l103_103034

theorem flagpole_break_height (total_height break_point distance_from_base : ℝ) 
(h_total : total_height = 6) 
(h_distance : distance_from_base = 2) 
(h_equation : (distance_from_base^2 + (total_height - break_point)^2) = break_point^2) :
  break_point = 3 := 
sorry

end flagpole_break_height_l103_103034


namespace projection_multiplier_l103_103530

noncomputable def a : ℝ × ℝ := (3, 6)
noncomputable def b : ℝ × ℝ := (-1, 0)

theorem projection_multiplier :
  let dot_product := a.1 * b.1 + a.2 * b.2
  let b_norm_sq := b.1 * b.1 + b.2 * b.2
  let proj := (dot_product / b_norm_sq) * 2
  (proj * b.1, proj * b.2) = (6, 0) :=
by 
  sorry

end projection_multiplier_l103_103530


namespace card_deal_probability_l103_103245

-- Definition of the problem conditions
def num_cards_in_deck := 52
def num_hearts := 13
def num_clubs := 13
def num_face_cards := 12
def probability_heart := num_hearts / num_cards_in_deck
def probability_club := num_clubs / (num_cards_in_deck - 1)
def probability_face_card := num_face_cards / (num_cards_in_deck - 2)

-- Combined probability calculation
def combined_probability := probability_heart * probability_club * probability_face_card

-- Statement of the theorem in Lean
theorem card_deal_probability :
  combined_probability = 39 / 2550 :=
by
  -- Definitions directly come from conditions
  have p_heart := probability_heart
  have p_club := probability_club
  have p_face_card := probability_face_card
  -- Probability calculation
  have combined := combined_probability
  -- Required equality for the theorem
  sorry

end card_deal_probability_l103_103245


namespace cos_half_angle_neg_sqrt_l103_103265

theorem cos_half_angle_neg_sqrt (theta m : ℝ) 
  (h1 : (5 / 2) * Real.pi < theta ∧ theta < 3 * Real.pi)
  (h2 : |Real.cos theta| = m) : 
  Real.cos (theta / 2) = -Real.sqrt ((1 - m) / 2) :=
sorry

end cos_half_angle_neg_sqrt_l103_103265


namespace solve_abs_eq_l103_103306

theorem solve_abs_eq (x : ℝ) : 
  (|x - 4| + 3 * x = 12) ↔ (x = 4) :=
by
  sorry

end solve_abs_eq_l103_103306


namespace peter_contains_five_l103_103850

theorem peter_contains_five (N : ℕ) (hN : N > 0) :
  ∃ k : ℕ, ∀ m : ℕ, m ≥ k → ∃ i : ℕ, 5 ≤ 10^i * (N * 5^m / 10^i) % 10 :=
sorry

end peter_contains_five_l103_103850


namespace geometric_sequence_sum_l103_103802

theorem geometric_sequence_sum :
  ∃ (a : ℕ → ℚ),
  (∀ n, 3 * a (n + 1) + a n = 0) ∧
  a 2 = -2/3 ∧
  (a 0 + a 1 + a 2 + a 3 + a 4) = 122/81 :=
sorry

end geometric_sequence_sum_l103_103802


namespace Roe_total_savings_l103_103852

-- Define savings amounts per period
def savings_Jan_to_Jul : Int := 7 * 10
def savings_Aug_to_Nov : Int := 4 * 15
def savings_Dec : Int := 20

-- Define total savings for the year
def total_savings : Int := savings_Jan_to_Jul + savings_Aug_to_Nov + savings_Dec

-- Prove that Roe's total savings for the year is $150
theorem Roe_total_savings : total_savings = 150 := by
  -- Proof goes here
  sorry

end Roe_total_savings_l103_103852


namespace solve_quadratic_l103_103307

theorem solve_quadratic (x : ℝ) :
  25 * x^2 - 10 * x - 1000 = 0 → ∃ r s, (x + r)^2 = s ∧ s = 40.04 :=
by
  intro h
  sorry

end solve_quadratic_l103_103307


namespace factor_difference_of_squares_l103_103057

theorem factor_difference_of_squares (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := 
by 
  sorry

end factor_difference_of_squares_l103_103057


namespace calculate_Al2O3_weight_and_H2_volume_l103_103517

noncomputable def weight_of_Al2O3 (moles : ℕ) : ℝ :=
  moles * ((2 * 26.98) + (3 * 16.00))

noncomputable def volume_of_H2_at_STP (moles_of_Al2O3 : ℕ) : ℝ :=
  (moles_of_Al2O3 * 3) * 22.4

theorem calculate_Al2O3_weight_and_H2_volume :
  weight_of_Al2O3 6 = 611.76 ∧ volume_of_H2_at_STP 6 = 403.2 :=
by
  sorry

end calculate_Al2O3_weight_and_H2_volume_l103_103517


namespace lowest_price_for_butter_l103_103380

def cost_single_package : ℝ := 7.0
def cost_8oz_package : ℝ := 4.0
def cost_4oz_package : ℝ := 2.0
def discount : ℝ := 0.5

theorem lowest_price_for_butter : 
  min cost_single_package (cost_8oz_package + 2 * (cost_4oz_package * discount)) = 6.0 :=
by
  sorry

end lowest_price_for_butter_l103_103380


namespace range_of_a_zeros_of_g_l103_103813

-- Definitions for the original functions f and g and their corresponding conditions
noncomputable def f (x a : ℝ) : ℝ := x * Real.log x - (a / 2) * x^2

noncomputable def g (x x2 a : ℝ) : ℝ := f x a - (x2 / 2)

-- Proving the range of a
theorem range_of_a (h : ∃ x1 x2 : ℝ, x1 < x2 ∧ x1 * Real.log x1 - (a / 2) * x1^2 = 0 ∧ x2 * Real.log x2 - (a / 2) * x2^2 = 0) :
  0 < a ∧ a < 1 := 
sorry

-- Proving the number of zeros of g based on the value of a
theorem zeros_of_g (a : ℝ) (x1 x2 : ℝ) (h : x1 < x2 ∧ x1 * Real.log x1 - (a / 2) * x1^2 = 0 ∧ x2 * Real.log x2 - (a / 2) * x2^2 = 0) :
  (0 < a ∧ a < 3 / Real.exp 2 → ∃ x3 x4, x3 ≠ x4 ∧ g x3 x2 a = 0 ∧ g x4 x2 a = 0) ∧
  (a = 3 / Real.exp 2 → ∃ x3, g x3 x2 a = 0) ∧
  (3 / Real.exp 2 < a ∧ a < 1 → ∀ x, g x x2 a ≠ 0) :=
sorry

end range_of_a_zeros_of_g_l103_103813


namespace bens_car_costs_l103_103386

theorem bens_car_costs :
  (∃ C_old C_2nd : ℕ,
    (2 * C_old = 4 * C_2nd) ∧
    (C_old = 1800) ∧
    (C_2nd = 900) ∧
    (2 * C_old = 3600) ∧
    (4 * C_2nd = 3600) ∧
    (1800 + 900 = 2700) ∧
    (3600 - 2700 = 900) ∧
    (2000 - 900 = 1100) ∧
    (900 * 0.05 = 45) ∧
    (45 * 2 = 90))
  :=
sorry

end bens_car_costs_l103_103386


namespace volume_of_box_l103_103652

-- Define the dimensions of the box
variables (L W H : ℝ)

-- Define the conditions as hypotheses
def side_face_area : Prop := H * W = 288
def top_face_area : Prop := L * W = 1.5 * 288
def front_face_area : Prop := L * H = 0.5 * (L * W)

-- Define the volume of the box
def box_volume : ℝ := L * W * H

-- The proof statement
theorem volume_of_box (h1 : side_face_area H W) (h2 : top_face_area L W) (h3 : front_face_area L H W) : box_volume L W H = 5184 :=
by
  sorry

end volume_of_box_l103_103652


namespace chord_line_equation_l103_103268

theorem chord_line_equation (x y : ℝ) 
  (ellipse : ∀ (x y : ℝ), x^2 / 36 + y^2 / 9 = 1)
  (bisect_point : x / 2 = 4 ∧ y / 2 = 2) : 
  x + 2 * y - 8 = 0 :=
sorry

end chord_line_equation_l103_103268


namespace dog_food_weight_l103_103345

/-- 
 Mike has 2 dogs, each dog eats 6 cups of dog food twice a day.
 Mike buys 9 bags of 20-pound dog food a month.
 Prove that a cup of dog food weighs 0.25 pounds.
-/
theorem dog_food_weight :
  let dogs := 2
  let cups_per_meal := 6
  let meals_per_day := 2
  let bags_per_month := 9
  let weight_per_bag := 20
  let days_per_month := 30
  let total_cups_per_day := cups_per_meal * meals_per_day * dogs
  let total_cups_per_month := total_cups_per_day * days_per_month
  let total_weight_per_month := bags_per_month * weight_per_bag
  (total_weight_per_month / total_cups_per_month : ℝ) = 0.25 :=
by
  sorry

end dog_food_weight_l103_103345


namespace ribbon_arrangement_count_correct_l103_103561

-- Definitions for the problem conditions
inductive Color
| red
| yellow
| blue

-- The color sequence from top to bottom
def color_sequence : List Color := [Color.red, Color.blue, Color.yellow, Color.yellow]

-- A function to count the valid arrangements
def count_valid_arrangements (sequence : List Color) : Nat :=
  -- Since we need to prove, we're bypassing the actual implementation with sorry
  sorry

-- The proof statement
theorem ribbon_arrangement_count_correct : count_valid_arrangements color_sequence = 12 :=
by
  sorry

end ribbon_arrangement_count_correct_l103_103561


namespace initial_pencils_l103_103289

theorem initial_pencils (P : ℕ) (h1 : 84 = P - (P - 15) / 4 + 16 - 12 + 23) : P = 71 :=
by
  sorry

end initial_pencils_l103_103289


namespace simplify_expr1_simplify_expr2_l103_103139

theorem simplify_expr1 (a b : ℤ) : 2 * a - (4 * a + 5 * b) + 2 * (3 * a - 4 * b) = 4 * a - 13 * b :=
by sorry

theorem simplify_expr2 (x y : ℤ) : 5 * x^2 - 2 * (3 * y^2 - 5 * x^2) + (-4 * y^2 + 7 * x * y) = 15 * x^2 - 10 * y^2 + 7 * x * y :=
by sorry

end simplify_expr1_simplify_expr2_l103_103139


namespace work_completion_time_l103_103492

theorem work_completion_time (B_rate A_rate Combined_rate : ℝ) (B_time : ℝ) :
  (B_rate = 1 / 60) →
  (A_rate = 4 * B_rate) →
  (Combined_rate = A_rate + B_rate) →
  (B_time = 1 / Combined_rate) →
  B_time = 12 :=
by sorry

end work_completion_time_l103_103492


namespace permutation_of_digits_l103_103262

-- Definition of factorial
def fact : ℕ → ℕ
| 0     => 1
| (n+1) => (n+1) * fact n

-- Given conditions
def n := 8
def n1 := 3
def n2 := 2
def n3 := 1
def n4 := 2

-- Statement
theorem permutation_of_digits :
  fact n / (fact n1 * fact n2 * fact n3 * fact n4) = 1680 :=
by
  sorry

end permutation_of_digits_l103_103262


namespace cost_of_ox_and_sheep_l103_103883

variable (x y : ℚ)

theorem cost_of_ox_and_sheep :
  (5 * x + 2 * y = 10) ∧ (2 * x + 8 * y = 8) → (x = 16 / 9 ∧ y = 5 / 9) :=
by
  sorry

end cost_of_ox_and_sheep_l103_103883


namespace base_8_addition_l103_103790

theorem base_8_addition (X Y : ℕ) (h1 : Y + 2 % 8 = X % 8) (h2 : X + 3 % 8 = 2 % 8) : X + Y = 12 := by
  sorry

end base_8_addition_l103_103790


namespace difference_in_zits_l103_103861

variable (avgZitsSwanson : ℕ := 5)
variable (avgZitsJones : ℕ := 6)
variable (numKidsSwanson : ℕ := 25)
variable (numKidsJones : ℕ := 32)
variable (totalZitsSwanson : ℕ := avgZitsSwanson * numKidsSwanson)
variable (totalZitsJones : ℕ := avgZitsJones * numKidsJones)

theorem difference_in_zits :
  totalZitsJones - totalZitsSwanson = 67 := by
  sorry

end difference_in_zits_l103_103861


namespace proof_problem_l103_103538

variables (p q : Prop)

theorem proof_problem (h₁ : p) (h₂ : ¬ q) : ¬ p ∨ ¬ q :=
by
  sorry

end proof_problem_l103_103538


namespace evaluate_expression_l103_103523

theorem evaluate_expression : (3200 - 3131) ^ 2 / 121 = 36 :=
by
  sorry

end evaluate_expression_l103_103523


namespace f_neg_1_l103_103254

-- Define the functions
variable (f : ℝ → ℝ) -- f is a real-valued function
variable (g : ℝ → ℝ) -- g is a real-valued function

-- Given conditions
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_def : ∀ x, g x = f x + 4
axiom g_at_1 : g 1 = 2

-- Define the theorem to prove
theorem f_neg_1 : f (-1) = 2 :=
by
  -- Proof goes here
  sorry

end f_neg_1_l103_103254


namespace triangle_inequality_side_len_l103_103075

theorem triangle_inequality_side_len (x : ℝ) : x = 8 → ¬ (2 < x ∧ x < 8) :=
by
  intro h
  rw h
  exact not_and_of_not_right _ (lt_irrefl 8)

#eval triangle_inequality_side_len 8 rfl

end triangle_inequality_side_len_l103_103075


namespace total_cost_correct_l103_103588

-- Define the cost of each category of items
def cost_of_book : ℕ := 16
def cost_of_binders : ℕ := 3 * 2
def cost_of_notebooks : ℕ := 6 * 1

-- Define the total cost calculation
def total_cost : ℕ := cost_of_book + cost_of_binders + cost_of_notebooks

-- Prove that the total cost of Léa's purchases is 28
theorem total_cost_correct : total_cost = 28 :=
by {
  -- This is where the proof would go, but it's omitted for now.
  sorry
}

end total_cost_correct_l103_103588


namespace inscribed_circle_radius_square_l103_103633

theorem inscribed_circle_radius_square (ER RF GS SH : ℕ) (r : ℕ)
  (hER : ER = 24) (hRF : RF = 31) (hGS : GS = 40) (hSH : SH = 29)
  (htangent_eq: 
    arctan (24 / r) + arctan (31 / r) + arctan (40 / r) + arctan (29 / r) = 180) :
  r^2 = 945 :=
by { sorry }

end inscribed_circle_radius_square_l103_103633


namespace find_a_extreme_value_l103_103417

theorem find_a_extreme_value (a : ℝ) :
  (f : ℝ → ℝ := λ x => x^3 + a*x^2 + 3*x - 9) →
  (f' : ℝ → ℝ := λ x => 3*x^2 + 2*a*x + 3) →
  f' (-3) = 0 →
  a = 5 :=
by
  sorry

end find_a_extreme_value_l103_103417


namespace total_food_per_day_l103_103609

theorem total_food_per_day 
  (first_soldiers : ℕ)
  (second_soldiers : ℕ)
  (food_first_side_per_soldier : ℕ)
  (food_second_side_per_soldier : ℕ) :
  first_soldiers = 4000 →
  second_soldiers = first_soldiers - 500 →
  food_first_side_per_soldier = 10 →
  food_second_side_per_soldier = food_first_side_per_soldier - 2 →
  (first_soldiers * food_first_side_per_soldier + second_soldiers * food_second_side_per_soldier = 68000) :=
by
  intros h1 h2 h3 h4
  sorry

end total_food_per_day_l103_103609


namespace computer_selling_price_l103_103502

variable (C SP : ℝ)

theorem computer_selling_price
  (h1 : 1.5 * C = 2678.57)
  (h2 : SP = 1.4 * C) :
  SP = 2500 :=
by
  sorry

end computer_selling_price_l103_103502


namespace words_per_page_l103_103887

theorem words_per_page (p : ℕ) (h1 : p ≤ 120) (h2 : 154 * p % 221 = 207) : p = 100 :=
sorry

end words_per_page_l103_103887


namespace remaining_customers_l103_103504

theorem remaining_customers (initial: ℕ) (left: ℕ) (remaining: ℕ) 
  (h1: initial = 14) (h2: left = 11) : remaining = initial - left → remaining = 3 :=
by {
  sorry
}

end remaining_customers_l103_103504


namespace truth_prob_l103_103558

-- Define the probabilities
def prob_A := 0.80
def prob_B := 0.60
def prob_C := 0.75

-- The problem statement
theorem truth_prob :
  prob_A * prob_B * prob_C = 0.27 :=
by
  -- Proof would go here
  sorry

end truth_prob_l103_103558


namespace tailor_time_calculation_l103_103640

-- Define the basic quantities and their relationships
def time_ratio_shirt : ℕ := 1
def time_ratio_pants : ℕ := 2
def time_ratio_jacket : ℕ := 3

-- Given conditions
def shirts_made := 2
def pants_made := 3
def jackets_made := 4
def total_time_initial : ℝ := 10

-- Unknown time per shirt
noncomputable def time_per_shirt := total_time_initial / (shirts_made * time_ratio_shirt 
  + pants_made * time_ratio_pants 
  + jackets_made * time_ratio_jacket)

-- Future quantities
def future_shirts := 14
def future_pants := 10
def future_jackets := 2

-- Calculate the future total time required
noncomputable def future_time_required := (future_shirts * time_ratio_shirt 
  + future_pants * time_ratio_pants 
  + future_jackets * time_ratio_jacket) * time_per_shirt

-- State the theorem to prove
theorem tailor_time_calculation : future_time_required = 20 := by
  sorry

end tailor_time_calculation_l103_103640


namespace rose_bought_flowers_l103_103134

theorem rose_bought_flowers (F : ℕ) (h1 : ∃ (daisies tulips sunflowers : ℕ), daisies = 2 ∧ sunflowers = 4 ∧ 
  tulips = (3 / 5) * (F - 2) ∧ sunflowers = (2 / 5) * (F - 2)) : F = 12 :=
sorry

end rose_bought_flowers_l103_103134


namespace distance_midpoint_to_origin_l103_103551

variables {a b c d m k l n : ℝ}

theorem distance_midpoint_to_origin (h1 : b = m * a + k) (h2 : d = m * c + k) (h3 : n = -1 / m) :
  dist (0, 0) ( ((a + c) / 2), ((m * (a + c) + 2 * k) / 2) ) = (1 / 2) * Real.sqrt ((1 + m^2) * (a + c)^2 + 4 * k^2 + 4 * m * (a + c) * k) :=
by
  sorry

end distance_midpoint_to_origin_l103_103551


namespace intersection_vertices_of_regular_octagon_l103_103089

noncomputable def set_A (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.1| + |p.2| = a ∧ a > 0}

def set_B : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.1 * p.2| + 1 = |p.1| + |p.2|}

theorem intersection_vertices_of_regular_octagon (a : ℝ) :
  (∃ (p : ℝ × ℝ), p ∈ set_A a ∧ p ∈ set_B) ↔ (a = Real.sqrt 2 ∨ a = 2 + Real.sqrt 2) :=
  sorry

end intersection_vertices_of_regular_octagon_l103_103089


namespace factor_difference_of_squares_l103_103056

theorem factor_difference_of_squares (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := 
by 
  sorry

end factor_difference_of_squares_l103_103056


namespace abs_difference_l103_103954

theorem abs_difference (a b : ℝ) (h₁ : a * b = 9) (h₂ : a + b = 10) : |a - b| = 8 :=
sorry

end abs_difference_l103_103954


namespace find_r_l103_103760

theorem find_r (r : ℝ) (AB AD BD : ℝ) (circle_radius : ℝ) (main_circle_radius : ℝ) :
  main_circle_radius = 2 →
  circle_radius = r →
  AB = 2 * r →
  AD = 2 * r →
  BD = 4 + 2 * r →
  (2 * r)^2 + (2 * r)^2 = (4 + 2 * r)^2 →
  r = 4 :=
by 
  intros h_main_radius h_circle_radius h_AB h_AD h_BD h_pythagorean
  sorry

end find_r_l103_103760


namespace positive_root_of_cubic_eq_l103_103913

theorem positive_root_of_cubic_eq : ∃ (x : ℝ), x > 0 ∧ x^3 - 3 * x^2 - x - Real.sqrt 2 = 0 ∧ x = 2 + Real.sqrt 2 := by
  sorry

end positive_root_of_cubic_eq_l103_103913


namespace age_of_hospital_l103_103564

theorem age_of_hospital (grant_current_age : ℕ) (future_ratio : ℚ)
                        (grant_future_age : grant_current_age + 5 = 30)
                        (hospital_age_ratio : future_ratio = 2 / 3) :
                        (grant_current_age = 25) → 
                        (grant_current_age + 5 = future_ratio * (grant_current_age + 5 + 5)) →
                        (grant_current_age + 5 + 5 - 5 = 40) :=
by
  sorry

end age_of_hospital_l103_103564


namespace next_term_of_geometric_sequence_l103_103191

theorem next_term_of_geometric_sequence (x : ℝ) (hx : x ≠ 0) : 
  let a₁ := 2
  let a₂ := 6 * x
  let a₃ := 18 * x^2
  let a₄ := 54 * x^3
  let r := a₂ / a₁
  let next_term := a₄ * r
  in next_term = 162 * x^4 :=
by
  -- Proof goes here
  sorry

end next_term_of_geometric_sequence_l103_103191


namespace list_price_is_40_l103_103898

open Real

def list_price (x : ℝ) : Prop :=
  0.15 * (x - 15) = 0.25 * (x - 25)

theorem list_price_is_40 : list_price 40 :=
by
  unfold list_price
  sorry

end list_price_is_40_l103_103898


namespace smallest_positive_value_of_expression_l103_103914

theorem smallest_positive_value_of_expression :
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ (a^3 + b^3 + c^3 - 3 * a * b * c = 4) :=
by
  sorry

end smallest_positive_value_of_expression_l103_103914


namespace paula_paint_cans_l103_103963

variables (rooms_per_can total_rooms_lost initial_rooms final_rooms cans_lost : ℕ)

theorem paula_paint_cans
  (h1 : initial_rooms = 50)
  (h2 : cans_lost = 2)
  (h3 : final_rooms = 42)
  (h4 : total_rooms_lost = initial_rooms - final_rooms)
  (h5 : rooms_per_can = total_rooms_lost / cans_lost) :
  final_rooms / rooms_per_can = 11 :=
by sorry

end paula_paint_cans_l103_103963


namespace equivalent_lengthEF_l103_103943

namespace GeometryProof

noncomputable def lengthEF 
  (AB CD EF : ℝ) 
  (h_AB_parallel_CD : true) 
  (h_lengthAB : AB = 200) 
  (h_lengthCD : CD = 50) 
  (h_angleEF : true) 
  : ℝ := 
  50

theorem equivalent_lengthEF
  (AB CD EF : ℝ) 
  (h_AB_parallel_CD : true) 
  (h_lengthAB : AB = 200) 
  (h_lengthCD : CD = 50) 
  (h_angleEF : true) 
  : lengthEF AB CD EF h_AB_parallel_CD h_lengthAB h_lengthCD h_angleEF = 50 :=
by
  sorry

end GeometryProof

end equivalent_lengthEF_l103_103943


namespace square_area_l103_103605

theorem square_area (perimeter : ℝ) (h : perimeter = 32) : 
  ∃ (side area : ℝ), side = perimeter / 4 ∧ area = side * side ∧ area = 64 := 
by
  sorry

end square_area_l103_103605


namespace position_of_point_l103_103426

theorem position_of_point (a b : ℝ) (h_tangent: (a ≠ 0 ∨ b ≠ 0) ∧ (a^2 + b^2 = 1)) : a^2 + b^2 = 1 :=
by
  sorry

end position_of_point_l103_103426


namespace problem_statement_l103_103541

variable (f : ℝ → ℝ)

theorem problem_statement (h : ∀ x : ℝ, 2 * (f x) + x * (deriv f x) > x^2) :
  ∀ x : ℝ, x^2 * f x ≥ 0 :=
by
  sorry

end problem_statement_l103_103541


namespace lcm_sum_div_lcm_even_l103_103624

open Nat

theorem lcm_sum_div_lcm_even (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
    2 ∣ (lcm x y + lcm y z) / lcm x z :=
sorry

end lcm_sum_div_lcm_even_l103_103624


namespace angle_variance_less_than_bound_l103_103293

noncomputable def angle_variance (α β γ : ℝ) : ℝ :=
  (1/3) * ((α - (2 * Real.pi / 3))^2 + (β - (2 * Real.pi / 3))^2 + (γ - (2 * Real.pi / 3))^2)

theorem angle_variance_less_than_bound (O A B C : ℝ → ℝ) :
  ∀ α β γ : ℝ, α + β + γ = 2 * Real.pi ∧ α ≥ β ∧ β ≥ γ → angle_variance α β γ < 2 * Real.pi^2 / 9 :=
by
  sorry

end angle_variance_less_than_bound_l103_103293


namespace oldest_sister_clothing_l103_103453

-- Define the initial conditions
def Nicole_initial := 10
def First_sister := Nicole_initial / 2
def Next_sister := Nicole_initial + 2
def Nicole_end := 36

-- Define the proof statement
theorem oldest_sister_clothing : 
    (First_sister + Next_sister + Nicole_initial + x = Nicole_end) → x = 9 :=
by
  sorry

end oldest_sister_clothing_l103_103453


namespace square_of_radius_l103_103634

-- Definitions based on conditions
def ER := 24
def RF := 31
def GS := 40
def SH := 29

-- The goal is to find square of radius r such that r^2 = 841
theorem square_of_radius (r : ℝ) :
  let R := ER
  let F := RF
  let G := GS
  let S := SH
  (∀ r : ℝ, (R + F) * (G + S) = r^2) → r^2 = 841 :=
sorry

end square_of_radius_l103_103634


namespace joe_initial_cars_l103_103439

theorem joe_initial_cars (x : ℕ) (h : x + 12 = 62) : x = 50 :=
by {
  sorry
}

end joe_initial_cars_l103_103439


namespace volume_in_cubic_yards_l103_103357

-- Define the conditions given in the problem
def volume_in_cubic_feet : ℕ := 216
def cubic_feet_per_cubic_yard : ℕ := 27

-- Define the theorem that needs to be proven
theorem volume_in_cubic_yards :
  volume_in_cubic_feet / cubic_feet_per_cubic_yard = 8 :=
by
  sorry

end volume_in_cubic_yards_l103_103357


namespace slope_of_perpendicular_line_l103_103237

-- Define what it means to be the slope of a line in a certain form
def slope_of_line (a b c : ℝ) (m : ℝ) : Prop :=
  b ≠ 0 ∧ m = -a / b

-- Define what it means for two slopes to be perpendicular
def are_perpendicular_slopes (m1 m2 : ℝ) : Prop :=
  m1 * m2 = -1

-- Given conditions
def given_line : Prop := slope_of_line 4 5 20 (-4 / 5)

-- The theorem to be proved
theorem slope_of_perpendicular_line : ∃ m : ℝ, given_line ∧ are_perpendicular_slopes (-4 / 5) m ∧ m = 5 / 4 :=
  sorry

end slope_of_perpendicular_line_l103_103237


namespace max_points_on_circle_l103_103593

noncomputable def circleMaxPoints (P C : ℝ × ℝ) (r1 r2 d : ℝ) : ℕ :=
  if d = r1 + r2 ∨ d = abs (r1 - r2) then 1 else 
  if d < r1 + r2 ∧ d > abs (r1 - r2) then 2 else 0

theorem max_points_on_circle (P : ℝ × ℝ) (C : ℝ × ℝ) :
  let rC := 5
  let distPC := 9
  let rP := 4
  circleMaxPoints P C rC rP distPC = 1 :=
by sorry

end max_points_on_circle_l103_103593


namespace interest_equality_l103_103829

-- Definitions based on the conditions
def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ := P * r * t

-- Constants for the problem
def P1 : ℝ := 200 -- 200 Rs is the principal of the first case
def r1 : ℝ := 0.1 -- 10% converted to a decimal
def t1 : ℝ := 12 -- 12 years

def P2 : ℝ := 1000 -- Correct answer for the other amount
def r2 : ℝ := 0.12 -- 12% converted to a decimal
def t2 : ℝ := 2 -- 2 years

-- Theorem stating that the interest generated is the same
theorem interest_equality : 
  simple_interest P1 r1 t1 = simple_interest P2 r2 t2 :=
by 
  -- Skip the proof since it is not required
  sorry

end interest_equality_l103_103829


namespace purchase_total_cost_l103_103583

theorem purchase_total_cost :
  (1 * 16) + (3 * 2) + (6 * 1) = 28 :=
sorry

end purchase_total_cost_l103_103583


namespace more_math_than_reading_l103_103851

def pages_reading := 4
def pages_math := 7

theorem more_math_than_reading : pages_math - pages_reading = 3 :=
by
  sorry

end more_math_than_reading_l103_103851


namespace average_speed_is_20_mph_l103_103717

-- Defining the conditions
def distance1 := 40 -- miles
def speed1 := 20 -- miles per hour
def distance2 := 20 -- miles
def speed2 := 40 -- miles per hour
def distance3 := 30 -- miles
def speed3 := 15 -- miles per hour

-- Calculating total distance and total time
def total_distance := distance1 + distance2 + distance3
def time1 := distance1 / speed1 -- hours
def time2 := distance2 / speed2 -- hours
def time3 := distance3 / speed3 -- hours
def total_time := time1 + time2 + time3

-- Theorem statement
theorem average_speed_is_20_mph : (total_distance / total_time) = 20 := by
  sorry

end average_speed_is_20_mph_l103_103717


namespace compute_fraction_mul_l103_103042

theorem compute_fraction_mul :
  (1 / 3) ^ 2 * (1 / 8) = 1 / 72 :=
by
  sorry

end compute_fraction_mul_l103_103042


namespace determine_m_range_l103_103800

theorem determine_m_range (m : ℝ) (h : (∃ (x y : ℝ), x^2 + y^2 + 2 * m * x + 2 = 0) ∧ 
                                    (∃ (r : ℝ) (h_r : r^2 = m^2 - 2), π * r^2 ≥ 4 * π)) :
  (m ≤ -Real.sqrt 6 ∨ m ≥ Real.sqrt 6) :=
by
  sorry

end determine_m_range_l103_103800


namespace triangle_height_l103_103571

theorem triangle_height (b h : ℕ) (A : ℕ) (hA : A = 50) (hb : b = 10) :
  A = (1 / 2 : ℝ) * b * h → h = 10 := 
by
  sorry

end triangle_height_l103_103571


namespace find_digit_B_l103_103019

theorem find_digit_B (A B : ℕ) (h1 : 100 * A + 78 - (210 + B) = 364) : B = 4 :=
by sorry

end find_digit_B_l103_103019


namespace solve_equation_l103_103463

theorem solve_equation (x y z : ℕ) : (3 ^ x + 5 ^ y + 14 = z!) ↔ ((x = 4 ∧ y = 2 ∧ z = 5) ∨ (x = 4 ∧ y = 4 ∧ z = 6)) :=
by
  sorry

end solve_equation_l103_103463


namespace sumOfTrianglesIs34_l103_103308

def triangleOp (a b c : ℕ) : ℕ := a * b - c

theorem sumOfTrianglesIs34 : 
  triangleOp 3 5 2 + triangleOp 4 6 3 = 34 := 
by
  sorry

end sumOfTrianglesIs34_l103_103308


namespace gear_C_rotation_direction_gear_C_rotation_count_l103_103513

/-- Definition of the radii of the gears -/
def radius_A : ℝ := 15
def radius_B : ℝ := 10 
def radius_C : ℝ := 5

/-- Gear \( A \) drives gear \( B \) and gear \( B \) drives gear \( C \) -/
def drives (x y : ℝ) := x * y

/-- Direction of rotation of gear \( C \) when gear \( A \) rotates clockwise -/
theorem gear_C_rotation_direction : drives radius_A radius_B = drives radius_C radius_B → drives radius_A radius_B > 0 → drives radius_C radius_B > 0 := by
  sorry

/-- Number of rotations of gear \( C \) when gear \( A \) makes one complete turn -/
theorem gear_C_rotation_count : ∀ n : ℝ, drives radius_A radius_B = drives radius_C radius_B → (n * radius_A)*(radius_B / radius_C) = 3 * n := by
  sorry

end gear_C_rotation_direction_gear_C_rotation_count_l103_103513


namespace largest_consecutive_odd_nat_divisible_by_3_sum_72_l103_103979

theorem largest_consecutive_odd_nat_divisible_by_3_sum_72
  (a : ℕ)
  (h₁ : a % 3 = 0)
  (h₂ : (a + 6) % 3 = 0)
  (h₃ : (a + 12) % 3 = 0)
  (h₄ : a % 2 = 1)
  (h₅ : (a + 6) % 2 = 1)
  (h₆ : (a + 12) % 2 = 1)
  (h₇ : a + (a + 6) + (a + 12) = 72) :
  a + 12 = 30 :=
by
  sorry

end largest_consecutive_odd_nat_divisible_by_3_sum_72_l103_103979


namespace bus_problem_l103_103031

theorem bus_problem : ∀ before_stop after_stop : ℕ, before_stop = 41 → after_stop = 18 → before_stop - after_stop = 23 :=
by
  intros before_stop after_stop h_before h_after
  sorry

end bus_problem_l103_103031


namespace angelas_insects_l103_103643

variable (DeanInsects : ℕ) (JacobInsects : ℕ) (AngelaInsects : ℕ)

theorem angelas_insects
  (h1 : DeanInsects = 30)
  (h2 : JacobInsects = 5 * DeanInsects)
  (h3 : AngelaInsects = JacobInsects / 2):
  AngelaInsects = 75 := 
by
  sorry

end angelas_insects_l103_103643


namespace min_value_of_fraction_l103_103956

theorem min_value_of_fraction (n : ℕ) (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 2) :
  (1 / (1 + a^n) + 1 / (1 + b^n)) = 1 :=
sorry

end min_value_of_fraction_l103_103956


namespace lea_total_cost_l103_103586

theorem lea_total_cost :
  let book_cost := 16 in
  let binders_count := 3 in
  let binder_cost := 2 in
  let notebooks_count := 6 in
  let notebook_cost := 1 in
  book_cost + (binders_count * binder_cost) + (notebooks_count * notebook_cost) = 28 :=
by
  sorry

end lea_total_cost_l103_103586


namespace arithmetic_sequence_values_l103_103931

theorem arithmetic_sequence_values (a b c : ℤ) 
  (h1 : 2 * b = a + c)
  (h2 : 2 * a = b + 1)
  (h3 : 2 * c = b + 9) 
  (h4 : a + b + c = -15) :
  b = -5 ∧ a * c = 21 :=
by
  sorry

end arithmetic_sequence_values_l103_103931


namespace circumference_of_circle_l103_103618

def speed_cyclist1 : ℝ := 7
def speed_cyclist2 : ℝ := 8
def meeting_time : ℝ := 42
def circumference : ℝ := 630

theorem circumference_of_circle :
  (speed_cyclist1 * meeting_time + speed_cyclist2 * meeting_time = circumference) :=
by
  sorry

end circumference_of_circle_l103_103618


namespace solve_ordered_pair_l103_103662

theorem solve_ordered_pair (x y : ℝ) (h1 : x + y = (5 - x) + (5 - y)) (h2 : x - y = (x - 1) + (y - 1)) : (x, y) = (4, 1) :=
by
  sorry

end solve_ordered_pair_l103_103662


namespace parabola_vertex_parabola_point_condition_l103_103835

-- Define the parabola function 
def parabola (x m : ℝ) : ℝ := x^2 - 2*m*x + m^2 - 1

-- 1. Prove the vertex of the parabola
theorem parabola_vertex (m : ℝ) : ∃ x y, (∀ x m, parabola x m = (x - m)^2 - 1) ∧ (x = m ∧ y = -1) :=
by
  sorry

-- 2. Prove the range of values for m given the conditions on points A and B
theorem parabola_point_condition (m : ℝ) (y1 y2 : ℝ) :
  (y1 > y2) ∧ 
  (parabola (1 - 2*m) m = y1) ∧ 
  (parabola (m + 1) m = y2) → m < 0 ∨ m > 2/3 :=
by
  sorry

end parabola_vertex_parabola_point_condition_l103_103835


namespace find_d_square_plus_5d_l103_103043

theorem find_d_square_plus_5d (a b c d : ℤ) (h₁: a^2 + 2 * a = 65) (h₂: b^2 + 3 * b = 125) (h₃: c^2 + 4 * c = 205) (h₄: d = 5 + 6) :
  d^2 + 5 * d = 176 :=
by
  rw [h₄]
  sorry

end find_d_square_plus_5d_l103_103043


namespace value_of_f1_l103_103577

variable (f : ℝ → ℝ)
open Function

theorem value_of_f1
  (h : ∀ x y : ℝ, f (f (x - y)) = f x * f y - f x + f y - 2 * x * y + 2 * x - 2 * y) :
  f 1 = -1 :=
sorry

end value_of_f1_l103_103577


namespace intersection_P_Q_l103_103404

def P : Set ℤ := { x | -4 ≤ x ∧ x ≤ 2 }

def Q : Set ℤ := { x | -3 < x ∧ x < 1 }

theorem intersection_P_Q : P ∩ Q = {-2, -1, 0} :=
by
  sorry

end intersection_P_Q_l103_103404


namespace largest_angle_in_triangle_l103_103430

open Real

theorem largest_angle_in_triangle
  (A B C : ℝ)
  (h : sin A / sin B / sin C = 1 / sqrt 2 / sqrt 5) :
  A ≤ B ∧ B ≤ C → C = 3 * π / 4 :=
by
  sorry

end largest_angle_in_triangle_l103_103430


namespace angelas_insects_l103_103644

variable (DeanInsects : ℕ) (JacobInsects : ℕ) (AngelaInsects : ℕ)

theorem angelas_insects
  (h1 : DeanInsects = 30)
  (h2 : JacobInsects = 5 * DeanInsects)
  (h3 : AngelaInsects = JacobInsects / 2):
  AngelaInsects = 75 := 
by
  sorry

end angelas_insects_l103_103644


namespace factor_difference_of_squares_l103_103060

-- Given: x is a real number.
-- Prove: x^2 - 64 = (x - 8) * (x + 8).
theorem factor_difference_of_squares (x : ℝ) : 
  x^2 - 64 = (x - 8) * (x + 8) :=
by
  sorry

end factor_difference_of_squares_l103_103060


namespace inequality_x2_y2_l103_103805

theorem inequality_x2_y2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) : 
  |x^2 + y^2| / (x + y) < |x^2 - y^2| / (x - y) :=
sorry

end inequality_x2_y2_l103_103805


namespace bacteria_reaches_final_in_24_hours_l103_103149

-- Define the initial number of bacteria
def initial_bacteria : ℕ := 200

-- Define the final number of bacteria
def final_bacteria : ℕ := 16200

-- Define the tripling period in hours
def tripling_period : ℕ := 6

-- Define the tripling factor
def tripling_factor : ℕ := 3

-- Define the number of hours needed to reach final number of bacteria
def hours_to_reach_final_bacteria : ℕ := 24

-- Define a function that models the number of bacteria after t hours
def bacteria_after (t : ℕ) : ℕ :=
  initial_bacteria * tripling_factor^((t / tripling_period))

-- Main statement of the problem: prove that the number of bacteria is 16200 after 24 hours
theorem bacteria_reaches_final_in_24_hours :
  bacteria_after hours_to_reach_final_bacteria = final_bacteria :=
sorry

end bacteria_reaches_final_in_24_hours_l103_103149


namespace B_alone_can_do_work_in_9_days_l103_103491

-- Define the conditions
def A_completes_work_in : ℕ := 15
def A_completes_portion_in (days : ℕ) : ℚ := days / 15
def portion_of_work_left (days : ℕ) : ℚ := 1 - A_completes_portion_in days
def B_completes_remaining_work_in_left_days (days_left : ℕ) : ℕ := 6
def B_completes_work_in (days_left : ℕ) : ℚ := B_completes_remaining_work_in_left_days days_left / (portion_of_work_left 5)

-- Define the theorem to be proven
theorem B_alone_can_do_work_in_9_days (days_left : ℕ) : B_completes_work_in days_left = 9 := by
  sorry

end B_alone_can_do_work_in_9_days_l103_103491


namespace number_of_meetings_l103_103327

-- Define the data for the problem
def pool_length : ℕ := 120
def swimmer_A_speed : ℕ := 4
def swimmer_B_speed : ℕ := 3
def total_time_seconds : ℕ := 15 * 60
def swimmer_A_turn_break_seconds : ℕ := 2
def swimmer_B_turn_break_seconds : ℕ := 0

-- Define the round trip time for each swimmer
def swimmer_A_round_trip_time : ℕ := 2 * (pool_length / swimmer_A_speed) + 2 * swimmer_A_turn_break_seconds
def swimmer_B_round_trip_time : ℕ := 2 * (pool_length / swimmer_B_speed) + 2 * swimmer_B_turn_break_seconds

-- Define the least common multiple of the round trip times
def lcm_round_trip_time : ℕ := Nat.lcm swimmer_A_round_trip_time swimmer_B_round_trip_time

-- Define the statement to prove
theorem number_of_meetings (lcm_round_trip_time : ℕ) : 
  (24 * (total_time_seconds / lcm_round_trip_time) + ((total_time_seconds % lcm_round_trip_time) / (pool_length / (swimmer_A_speed + swimmer_B_speed)))) = 51 := 
sorry

end number_of_meetings_l103_103327


namespace total_amount_paid_l103_103277

theorem total_amount_paid : 
  (∀ (p cost_per_pizza : ℕ), cost_per_pizza = 12 → p = 3 → cost_per_pizza * p = 36) →
  (∀ (d cost_per_pizza : ℕ), cost_per_pizza = 12 → d = 2 → cost_per_pizza * d + 2 = 26) →
  36 + 26 = 62 :=
by {
  intro h1 h2,
  sorry
}

end total_amount_paid_l103_103277


namespace set_interval_representation_l103_103022

open Set

theorem set_interval_representation :
  {x : ℝ | (0 ≤ x ∧ x < 5) ∨ x > 10} = Ico 0 5 ∪ Ioi 10 :=
by
  sorry

end set_interval_representation_l103_103022


namespace factory_workers_l103_103732

-- Define parameters based on given conditions
def sewing_factory_x : ℤ := 1995
def shoe_factory_y : ℤ := 1575

-- Conditions based on the problem setup
def shoe_factory_of_sewing_factory := (15 * sewing_factory_x) / 19 = shoe_factory_y
def shoe_factory_plan_exceed := (3 * shoe_factory_y) / 7 < 1000
def sewing_factory_plan_exceed := (3 * sewing_factory_x) / 5 > 1000

-- Theorem stating the problem's assertion
theorem factory_workers (x y : ℤ) 
  (h1 : (15 * x) / 19 = y)
  (h2 : (4 * y) / 7 < 1000)
  (h3 : (3 * x) / 5 > 1000) : 
  x = 1995 ∧ y = 1575 :=
sorry

end factory_workers_l103_103732


namespace friend_P_distance_l103_103163

theorem friend_P_distance (v t : ℝ) (hv : v > 0)
  (distance_trail : 22 = (1.20 * v * t) + (v * t))
  (h_t : t = 22 / (2.20 * v)) : 
  (1.20 * v * t = 12) :=
by
  sorry

end friend_P_distance_l103_103163


namespace artist_used_17_ounces_of_paint_l103_103507

def ounces_used_per_large_canvas : ℕ := 3
def ounces_used_per_small_canvas : ℕ := 2
def large_paintings_completed : ℕ := 3
def small_paintings_completed : ℕ := 4

theorem artist_used_17_ounces_of_paint :
  (ounces_used_per_large_canvas * large_paintings_completed + ounces_used_per_small_canvas * small_paintings_completed = 17) :=
by
  sorry

end artist_used_17_ounces_of_paint_l103_103507


namespace volume_in_cubic_yards_l103_103356

-- Define the conditions given in the problem
def volume_in_cubic_feet : ℕ := 216
def cubic_feet_per_cubic_yard : ℕ := 27

-- Define the theorem that needs to be proven
theorem volume_in_cubic_yards :
  volume_in_cubic_feet / cubic_feet_per_cubic_yard = 8 :=
by
  sorry

end volume_in_cubic_yards_l103_103356


namespace valid_N_eq_1_2_3_l103_103399

def is_valid_config (N : ℕ) (grid : set (ℕ × ℕ)) : Prop :=
  ∀ S : set (ℕ × ℕ), equilateral_triangle S → (grid ∩ S).card = N - 1

def infinite_triangular_grid (N : ℕ) (grid : set (ℕ × ℕ)) : Prop :=
  ∀ n, (grid ∩ {x | x.1 < n}).infinite

noncomputable def valid_N_values := { N | ∀ grid, infinite_triangular_grid N grid → is_valid_config N grid }

theorem valid_N_eq_1_2_3 : valid_N_values = {1, 2, 3} := sorry

end valid_N_eq_1_2_3_l103_103399


namespace erica_blank_question_count_l103_103613

variable {C W B : ℕ}

theorem erica_blank_question_count
  (h1 : C + W + B = 20)
  (h2 : 7 * C - 4 * W = 100) :
  B = 1 :=
by
  sorry

end erica_blank_question_count_l103_103613


namespace product_of_m_and_u_l103_103122

noncomputable def g : ℝ → ℝ := sorry

axiom g_conditions : (∀ x y : ℝ, g (x^2 - y^2) = (x - y) * ((g x) ^ 3 + (g y) ^ 3)) ∧ (g 1 = 1)

def m : ℕ := sorry
def u : ℝ := sorry

theorem product_of_m_and_u : m * u = 3 :=
by 
  -- all conditions about 'g' are assumed as axioms and not directly included in the proof steps
  exact sorry

end product_of_m_and_u_l103_103122


namespace product_of_five_consecutive_integers_not_square_l103_103300

theorem product_of_five_consecutive_integers_not_square (a : ℕ) :
  ¬ ∃ b c d e : ℕ, b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ e = a + 4 ∧ ∃ k : ℕ, (a * b * c * d * e) = k^2 :=
by
  sorry

end product_of_five_consecutive_integers_not_square_l103_103300


namespace probability_of_region_l103_103291

theorem probability_of_region :
  let area_rect := (1000: ℝ) * 1500
  let area_polygon := 500000
  let prob := area_polygon / area_rect
  prob = (1 / 3) := sorry

end probability_of_region_l103_103291


namespace z_investment_correct_l103_103753

noncomputable def z_investment 
    (x_investment : ℕ) 
    (y_investment : ℕ) 
    (z_profit : ℕ) 
    (total_profit : ℕ)
    (profit_z : ℕ) : ℕ := 
  let x_time := 12
  let y_time := 12
  let z_time := 8
  let x_share := x_investment * x_time
  let y_share := y_investment * y_time
  let profit_ratio := total_profit - profit_z
  (x_share + y_share) * z_time / profit_ratio

theorem z_investment_correct : 
  z_investment 36000 42000 4032 13860 4032 = 52000 :=
by sorry

end z_investment_correct_l103_103753


namespace constant_term_g_eq_l103_103281

noncomputable def f : Polynomial ℝ := sorry
noncomputable def g : Polynomial ℝ := sorry
noncomputable def h : Polynomial ℝ := f * g

theorem constant_term_g_eq : 
  (h.coeff 0 = 2) ∧ (f.coeff 0 = -6) →  g.coeff 0 = -1/3 := by
  sorry

end constant_term_g_eq_l103_103281


namespace find_first_term_l103_103481

def geom_seq (a r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

theorem find_first_term (a r : ℝ) (h1 : r = 2/3) (h2 : geom_seq a r 3 = 18) (h3 : geom_seq a r 4 = 12) : a = 40.5 := 
by sorry

end find_first_term_l103_103481


namespace part_a_part_b_l103_103712

variable (f : ℝ → ℝ) [LipschitzWith 1 (λ x, max (f x) 0) id] -- Ensure f is (1-)Lipschitz and non-negative

-- Part (a)
theorem part_a (hf : ∀ x, filter.tendsto (λ n, f (x + n)) filter.at_top filter.at_top) :
  filter.tendsto f filter.at_top filter.at_top :=
by 
  sorry

-- Part (b)
theorem part_b (α : ℝ) (hα : 0 ≤ α) (hf : ∀ x, filter.tendsto (λ n, f (x + n)) filter.at_top (nhds α)) :
  filter.tendsto f filter.at_top (nhds α) :=
by 
  sorry

end part_a_part_b_l103_103712


namespace find_z_l103_103912

theorem find_z (x y : ℤ) (h1 : x * y + x + y = 106) (h2 : x^2 * y + x * y^2 = 1320) :
  x^2 + y^2 = 748 ∨ x^2 + y^2 = 5716 :=
sorry

end find_z_l103_103912


namespace P_union_Q_eq_Q_l103_103447

noncomputable def P : Set ℝ := {x : ℝ | x > 1}
noncomputable def Q : Set ℝ := {x : ℝ | x^2 - x > 0}

theorem P_union_Q_eq_Q : P ∪ Q = Q := by
  sorry

end P_union_Q_eq_Q_l103_103447


namespace complete_the_square_l103_103175

theorem complete_the_square (x : ℝ) :
  (x^2 + 8*x + 9 = 0) ↔ ((x + 4)^2 = 7) :=
sorry

end complete_the_square_l103_103175


namespace swimming_pool_water_remaining_l103_103895

theorem swimming_pool_water_remaining :
  let initial_water := 500 -- initial water in gallons
  let evaporation_rate := 1.5 -- water loss due to evaporation in gallons/day
  let leak_rate := 0.8 -- water loss due to leak in gallons/day
  let total_days := 20 -- total number of days

  let total_daily_loss := evaporation_rate + leak_rate -- total daily loss in gallons/day
  let total_loss := total_daily_loss * total_days -- total loss over the period in gallons
  let remaining_water := initial_water - total_loss -- remaining water after 20 days in gallons

  remaining_water = 454 :=
by
  sorry

end swimming_pool_water_remaining_l103_103895


namespace find_a7_over_b7_l103_103734

-- Definitions of the sequences and the arithmetic properties
variable {a b: ℕ → ℕ}  -- sequences a_n and b_n
variable {S T: ℕ → ℕ}  -- sums of the first n terms

-- Problem conditions
def is_arithmetic_sequence (seq: ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, seq (n + 1) - seq n = d

def sum_of_first_n_terms (seq: ℕ → ℕ) (sum_fn: ℕ → ℕ) : Prop :=
  ∀ n, sum_fn n = n * (seq 1 + seq n) / 2

-- Given conditions
axiom h1: is_arithmetic_sequence a
axiom h2: is_arithmetic_sequence b
axiom h3: sum_of_first_n_terms a S
axiom h4: sum_of_first_n_terms b T
axiom h5: ∀ n, S n / T n = (3 * n + 2) / (2 * n)

-- Main theorem to prove
theorem find_a7_over_b7 : (a 7) / (b 7) = (41 / 26) :=
sorry

end find_a7_over_b7_l103_103734


namespace sum_of_decimals_l103_103736

theorem sum_of_decimals : 1.000 + 0.101 + 0.011 + 0.001 = 1.113 :=
by
  sorry

end sum_of_decimals_l103_103736


namespace tangent_line_at_one_unique_zero_of_f_exists_lower_bound_of_f_l103_103801

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + x * Real.exp x - Real.exp 1

-- Part (Ⅰ)
theorem tangent_line_at_one (h_a : a = 0) : ∃ m b : ℝ, ∀ x : ℝ, 2 * Real.exp 1 * x - y - 2 * Real.exp 1 = 0 := sorry

-- Part (Ⅱ)
theorem unique_zero_of_f (h_a : a > 0) : ∃! t : ℝ, f a t = 0 := sorry

-- Part (Ⅲ)
theorem exists_lower_bound_of_f (h_a : a < 0) : ∃ m : ℝ, ∀ x : ℝ, f a x ≥ m := sorry

end tangent_line_at_one_unique_zero_of_f_exists_lower_bound_of_f_l103_103801


namespace contestant_wins_probability_l103_103352

-- Define the basic parameters: number of questions and number of choices
def num_questions : ℕ := 4
def num_choices : ℕ := 3

-- Define the probability of getting a single question right
def prob_right : ℚ := 1 / num_choices

-- Define the probability of guessing all questions right
def prob_all_right : ℚ := prob_right ^ num_questions

-- Define the probability of guessing exactly three questions right (one wrong)
def prob_one_wrong : ℚ := (prob_right ^ 3) * (2 / num_choices)

-- Calculate the total probability of winning
def total_prob_winning : ℚ := prob_all_right + 4 * prob_one_wrong

-- The final statement to prove
theorem contestant_wins_probability :
  total_prob_winning = 1 / 9 := 
sorry

end contestant_wins_probability_l103_103352


namespace eggs_left_after_taking_l103_103001

def eggs_in_box_initial : Nat := 47
def eggs_taken_by_Harry : Nat := 5
theorem eggs_left_after_taking : eggs_in_box_initial - eggs_taken_by_Harry = 42 := 
by
  -- Proof placeholder
  sorry

end eggs_left_after_taking_l103_103001


namespace biology_collections_count_l103_103288

/-- 
Proof Problem: Given the letters in "BIOLOGY", with O's and G's being indistinguishable, 
the number of distinct possible collections of 3 vowels and 2 consonants is 12.
-/
theorem biology_collections_count : 
  let vowels := ['I', 'O', 'O', 'Y'],
      consonants := ['B', 'G', 'G'],
      num_possible_collections := 4 * 3
  in num_possible_collections = 12 := 
by 
  let vowels := ['I', 'O', 'O', 'Y']
  let consonants := ['B', 'G', 'G']
  let num_vowel_groups := 4 -- from the breakdown in the solution
  let num_consonant_groups := 3 -- from the breakdown in the solution
  have distinct_collections : 4 * 3 = 12, by norm_num
  exact distinct_collections

end biology_collections_count_l103_103288


namespace no_rational_roots_l103_103899

theorem no_rational_roots {p q : ℤ} (hp : p % 2 = 1) (hq : q % 2 = 1) :
  ¬ ∃ x : ℚ, x^2 + (2 * p) * x + (2 * q) = 0 :=
by
  -- proof using contradiction technique
  sorry

end no_rational_roots_l103_103899


namespace subtraction_example_l103_103225

theorem subtraction_example : 2 - 3 = -1 := 
by {
  -- We need to prove that 2 - 3 = -1
  -- The proof is to be filled here
  sorry
}

end subtraction_example_l103_103225


namespace product_of_five_consecutive_integers_not_square_l103_103301

theorem product_of_five_consecutive_integers_not_square (a : ℕ) :
  ¬ ∃ b c d e : ℕ, b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ e = a + 4 ∧ ∃ k : ℕ, (a * b * c * d * e) = k^2 :=
by
  sorry

end product_of_five_consecutive_integers_not_square_l103_103301


namespace problem_l103_103810

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

theorem problem
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_geom : geometric_sequence a q)
  (h1 : a 0 + a 1 = 4 / 9)
  (h2 : a 2 + a 3 + a 4 + a 5 = 40) :
  (a 6 + a 7 + a 8) / 9 = 117 :=
sorry

end problem_l103_103810


namespace quadratic_no_real_roots_l103_103269

theorem quadratic_no_real_roots (c : ℝ) : 
  (∀ x : ℝ, ¬(x^2 + x - c = 0)) ↔ c < -1/4 := 
sorry

end quadratic_no_real_roots_l103_103269


namespace number_of_ways_to_divide_l103_103565

-- Define the given shape
structure Shape :=
  (sides : Nat) -- Number of 3x1 stripes along the sides
  (centre : Nat) -- Size of the central square (3x3)

-- Define the specific problem shape
def problem_shape : Shape :=
  { sides := 4, centre := 9 } -- 3x1 stripes on all sides and a 3x3 centre

-- Theorem stating the number of ways to divide the shape into 1x3 rectangles
theorem number_of_ways_to_divide (s : Shape) (h1 : s.sides = 4) (h2 : s.centre = 9) : 
  ∃ ways, ways = 2 :=
by
  -- The proof is skipped
  sorry

end number_of_ways_to_divide_l103_103565


namespace average_speed_for_trip_l103_103440

theorem average_speed_for_trip :
  ∀ (walk_dist bike_dist drive_dist tot_dist walk_speed bike_speed drive_speed : ℝ)
  (h1 : walk_dist = 5) (h2 : bike_dist = 35) (h3 : drive_dist = 80)
  (h4 : tot_dist = 120) (h5 : walk_speed = 5) (h6 : bike_speed = 15)
  (h7 : drive_speed = 120),
  (tot_dist / (walk_dist / walk_speed + bike_dist / bike_speed + drive_dist / drive_speed)) = 30 :=
by
  intros
  sorry

end average_speed_for_trip_l103_103440


namespace general_term_min_value_S_n_l103_103918

-- Definitions and conditions according to the problem statement
variable (d : ℤ) (a₁ : ℤ) (n : ℕ)

def a_n (n : ℕ) : ℤ := a₁ + (n - 1) * d
def S_n (n : ℕ) : ℤ := n * (2 * a₁ + (n - 1) * d) / 2

-- Given conditions
axiom positive_common_difference : 0 < d
axiom a3_a4_product : a_n 3 * a_n 4 = 117
axiom a2_a5_sum : a_n 2 + a_n 5 = -22

-- Proof 1: General term of the arithmetic sequence
theorem general_term : a_n n = 4 * (n : ℤ) - 25 :=
  by sorry

-- Proof 2: Minimum value of the sum of the first n terms
theorem min_value_S_n : S_n 6 = -66 :=
  by sorry

end general_term_min_value_S_n_l103_103918


namespace lines_through_P_and_form_area_l103_103209

-- Definition of the problem conditions
def passes_through_P (k b : ℝ) : Prop :=
  b = 2 - k

def forms_area_with_axes (k b : ℝ) : Prop :=
  b^2 = 8 * |k|

-- Theorem statement
theorem lines_through_P_and_form_area :
  ∃ (k1 k2 k3 b1 b2 b3 : ℝ),
    passes_through_P k1 b1 ∧ forms_area_with_axes k1 b1 ∧
    passes_through_P k2 b2 ∧ forms_area_with_axes k2 b2 ∧
    passes_through_P k3 b3 ∧ forms_area_with_axes k3 b3 ∧
    k1 ≠ k2 ∧ k2 ≠ k3 ∧ k1 ≠ k3 :=
sorry

end lines_through_P_and_form_area_l103_103209


namespace zits_difference_l103_103860

variable (avg_zits_swanson : ℕ)
variable (num_students_swanson : ℕ)
variable (avg_zits_jones : ℕ)
variable (num_students_jones : ℕ)

-- Conditions
def total_zits_swanson := avg_zits_swanson * num_students_swanson
def total_zits_jones := avg_zits_jones * num_students_jones

-- Theorem to prove the difference in total zits
theorem zits_difference : 
  avg_zits_swanson = 5 → 
  num_students_swanson = 25 → 
  avg_zits_jones = 6 → 
  num_students_jones = 32 → 
  total_zits_jones avg_zits_jones num_students_jones - total_zits_swanson avg_zits_swanson num_students_swanson = 67 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  show 6 * 32 - 5 * 25 = 67
  norm_num
  sorry

end zits_difference_l103_103860


namespace store_hours_open_per_day_l103_103436

theorem store_hours_open_per_day
  (rent_per_week : ℝ)
  (utility_percentage : ℝ)
  (employees_per_shift : ℕ)
  (hourly_wage : ℝ)
  (days_per_week_open : ℕ)
  (weekly_expenses : ℝ)
  (H_rent : rent_per_week = 1200)
  (H_utility_percentage : utility_percentage = 0.20)
  (H_employees_per_shift : employees_per_shift = 2)
  (H_hourly_wage : hourly_wage = 12.50)
  (H_days_open : days_per_week_open = 5)
  (H_weekly_expenses : weekly_expenses = 3440) :
  (16 : ℝ) = weekly_expenses / ((rent_per_week * (1 + utility_percentage)) + (employees_per_shift * hourly_wage * days_per_week_open)) :=
by
  sorry

end store_hours_open_per_day_l103_103436


namespace area_triangle_BMN_squared_l103_103292

theorem area_triangle_BMN_squared :
  let A := (0, 0)
  let C := (20, 0)
  let B := (16, 0)
  (∃ D E M N : prod real real,
    equilateral_triangle D A B ∧
    equilateral_triangle E B C ∧
    M = midpoint A E ∧
    N = midpoint C D ∧
    4563 = (triangle_area_squared B M N)
  ) := sorry

end area_triangle_BMN_squared_l103_103292


namespace discount_is_10_percent_l103_103891

variable (C : ℝ)  -- Cost of the item
variable (S S' : ℝ)  -- Selling prices with and without discount

-- Conditions
def condition1 : Prop := S = 1.20 * C
def condition2 : Prop := S' = 1.30 * C

-- The proposition to prove
theorem discount_is_10_percent (h1 : condition1 C S) (h2 : condition2 C S') : S' - S = 0.10 * C := by
  sorry

end discount_is_10_percent_l103_103891


namespace smallest_number_divisible_remainders_l103_103348

theorem smallest_number_divisible_remainders :
  ∃ n : ℕ,
    (n % 10 = 9) ∧
    (n % 9 = 8) ∧
    (n % 8 = 7) ∧
    (n % 7 = 6) ∧
    (n % 6 = 5) ∧
    (n % 5 = 4) ∧
    (n % 4 = 3) ∧
    (n % 3 = 2) ∧
    (n % 2 = 1) ∧
    n = 2519 :=
sorry

end smallest_number_divisible_remainders_l103_103348


namespace cos_double_angle_of_tan_l103_103823

theorem cos_double_angle_of_tan (θ : ℝ) (h : Real.tan θ = -1 / 3) : Real.cos (2 * θ) = 4 / 5 :=
sorry

end cos_double_angle_of_tan_l103_103823


namespace find_b_plus_m_l103_103086

noncomputable def f (a b x : ℝ) : ℝ := Real.log (x + 1) / Real.log a + b 

variable (a b m : ℝ)
-- Conditions
axiom h1 : a > 0
axiom h2 : a ≠ 1
axiom h3 : f a b m = 3

theorem find_b_plus_m : b + m = 3 :=
sorry

end find_b_plus_m_l103_103086


namespace altered_prism_edges_l103_103654

theorem altered_prism_edges :
  let original_edges := 12
  let vertices := 8
  let edges_per_vertex := 3
  let faces := 6
  let edges_per_face := 1
  let total_edges := original_edges + edges_per_vertex * vertices + edges_per_face * faces
  total_edges = 42 :=
by
  let original_edges := 12
  let vertices := 8
  let edges_per_vertex := 3
  let faces := 6
  let edges_per_face := 1
  let total_edges := original_edges + edges_per_vertex * vertices + edges_per_face * faces
  show total_edges = 42
  sorry

end altered_prism_edges_l103_103654


namespace junk_mail_each_house_l103_103347

def blocks : ℕ := 16
def houses_per_block : ℕ := 17
def total_junk_mail : ℕ := 1088
def total_houses : ℕ := blocks * houses_per_block
def junk_mail_per_house : ℕ := total_junk_mail / total_houses

theorem junk_mail_each_house :
  junk_mail_per_house = 4 :=
by
  sorry

end junk_mail_each_house_l103_103347


namespace fraction_a_over_d_l103_103822

-- Defining the given conditions as hypotheses
variables (a b c d : ℚ)

-- Conditions
axiom h1 : a / b = 20
axiom h2 : c / b = 5
axiom h3 : c / d = 1 / 15

-- Goal to prove
theorem fraction_a_over_d : a / d = 4 / 15 :=
by
  sorry

end fraction_a_over_d_l103_103822


namespace area_between_sqrt_and_x_l103_103468

noncomputable def area_enclosed_by_sqrt_x_and_x : ℝ :=
∫ x in 0..1, real.sqrt x - x

theorem area_between_sqrt_and_x :
  area_enclosed_by_sqrt_x_and_x = 1 / 6 :=
by
  sorry

end area_between_sqrt_and_x_l103_103468


namespace trig_problem_l103_103247

variable (α : ℝ)

theorem trig_problem
  (h1 : Real.sin (Real.pi + α) = -1 / 3) :
  Real.cos (α - 3 * Real.pi / 2) = -1 / 3 ∧
  (Real.sin (Real.pi / 2 + α) = 2 * Real.sqrt 2 / 3 ∨ Real.sin (Real.pi / 2 + α) = -2 * Real.sqrt 2 / 3) ∧
  (Real.tan (5 * Real.pi - α) = -Real.sqrt 2 / 4 ∨ Real.tan (5 * Real.pi - α) = Real.sqrt 2 / 4) :=
sorry

end trig_problem_l103_103247


namespace sine_gamma_half_leq_c_over_a_plus_b_l103_103294

variable (a b c : ℝ) (γ : ℝ)

-- Consider a triangle with sides a, b, c, and angle γ opposite to side c.
-- We need to prove that sin(γ / 2) ≤ c / (a + b).
theorem sine_gamma_half_leq_c_over_a_plus_b (h_c_pos : 0 < c) 
  (h_g_angle : 0 < γ ∧ γ < 2 * π) : 
  Real.sin (γ / 2) ≤ c / (a + b) := 
  sorry

end sine_gamma_half_leq_c_over_a_plus_b_l103_103294


namespace probability_correct_l103_103783

-- Define the problem conditions.
def num_balls : ℕ := 8
def possible_colors : ℕ := 2

-- Probability calculation for a specific arrangement (either configuration of colors).
def probability_per_arrangement : ℚ := (1/2) ^ num_balls

-- Number of favorable arrangements with 4 black and 4 white balls.
def favorable_arrangements : ℕ := Nat.choose num_balls 4

-- The required probability for the solution.
def desired_probability : ℚ := favorable_arrangements * probability_per_arrangement

-- The proof statement to be provided.
theorem probability_correct :
  desired_probability = 35 / 128 := 
by
  sorry

end probability_correct_l103_103783


namespace matches_for_ladder_l103_103739

theorem matches_for_ladder (n : ℕ) (h : n = 25) : 
  (6 + 6 * (n - 1) = 150) :=
by
  sorry

end matches_for_ladder_l103_103739


namespace lowest_price_for_16_oz_butter_l103_103382

-- Define the constants
def price_single_16_oz_package : ℝ := 7
def price_8_oz_package : ℝ := 4
def price_4_oz_package : ℝ := 2
def discount_4_oz_package : ℝ := 0.5

-- Calculate the discounted price for a 4 oz package
def discounted_price_4_oz_package : ℝ := price_4_oz_package * discount_4_oz_package

-- Calculate the total price for two discounted 4 oz packages
def total_price_two_discounted_4_oz_packages : ℝ := 2 * discounted_price_4_oz_package

-- Calculate the total price using the 8 oz package and two discounted 4 oz packages
def total_price_using_coupon : ℝ := price_8_oz_package + total_price_two_discounted_4_oz_packages

-- State the property to prove
theorem lowest_price_for_16_oz_butter :
  min price_single_16_oz_package total_price_using_coupon = 6 :=
sorry

end lowest_price_for_16_oz_butter_l103_103382


namespace lemonade_quarts_l103_103930

theorem lemonade_quarts (total_parts water_parts lemon_juice_parts : ℕ) (total_gallons gallons_to_quarts : ℚ) 
  (h_ratio : water_parts = 4) (h_ratio_lemon : lemon_juice_parts = 1) (h_total_parts : total_parts = water_parts + lemon_juice_parts)
  (h_total_gallons : total_gallons = 1) (h_gallons_to_quarts : gallons_to_quarts = 4) :
  let volume_per_part := total_gallons / total_parts
  let volume_per_part_quarts := volume_per_part * gallons_to_quarts
  let water_volume := water_parts * volume_per_part_quarts
  water_volume = 16 / 5 :=
by
  sorry

end lemonade_quarts_l103_103930


namespace sand_problem_l103_103374

-- Definitions based on conditions
def initial_sand := 1050
def sand_lost_first := 32
def sand_lost_second := 67
def sand_lost_third := 45
def sand_lost_fourth := 54

-- Total sand lost
def total_sand_lost := sand_lost_first + sand_lost_second + sand_lost_third + sand_lost_fourth

-- Sand remaining
def sand_remaining := initial_sand - total_sand_lost

-- Theorem stating the proof problem
theorem sand_problem : sand_remaining = 852 :=
by
-- Skipping proof as per instructions
sorry

end sand_problem_l103_103374


namespace fred_allowance_is_16_l103_103244

def fred_weekly_allowance (A : ℕ) : Prop :=
  (A / 2) + 6 = 14

theorem fred_allowance_is_16 : ∃ A : ℕ, fred_weekly_allowance A ∧ A = 16 := 
by
  -- Proof can be filled here
  sorry

end fred_allowance_is_16_l103_103244


namespace magnitude_of_a_plus_b_l103_103579

open Real

noncomputable def magnitude (x y : ℝ) : ℝ :=
  sqrt (x^2 + y^2)

theorem magnitude_of_a_plus_b (m : ℝ) (a b : ℝ × ℝ)
  (h₁ : a = (m+2, 1))
  (h₂ : b = (1, -2*m))
  (h₃ : (a.1 * b.1 + a.2 * b.2 = 0)) :
  magnitude (a.1 + b.1) (a.2 + b.2) = sqrt 34 :=
by
  sorry

end magnitude_of_a_plus_b_l103_103579


namespace max_tan_alpha_l103_103679

theorem max_tan_alpha (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
    (h : Real.tan (α + β) = 9 * Real.tan β) : Real.tan α ≤ 4 / 3 :=
by
  sorry

end max_tan_alpha_l103_103679


namespace inequality_pos_reals_l103_103600

theorem inequality_pos_reals (x y z : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) : 
  (x^2 + 2) * (y^2 + 2) * (z^2 + 2) ≥ 9 * (x * y + y * z + z * x) :=
by
  sorry

end inequality_pos_reals_l103_103600


namespace hyperbola_asymptote_slope_l103_103319

theorem hyperbola_asymptote_slope (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) 
    (h_hyperbola : ∀ x y : ℝ, ∀ x² / a² - y² / b² = 1) 
    (A₁ : Prod (-a) 0) (A₂ : Prod a 0) (B : Prod c (b² / a)) (C : Prod c (-b² / a))
    (h_perpendicular : ((b² / a) / (c + a)) * (-(b² / a) / (c - a)) = -1) : 
    slope_asymptote = ±1 := by
  sorry

end hyperbola_asymptote_slope_l103_103319


namespace five_student_committee_l103_103534

theorem five_student_committee : ∀ (students : Finset ℕ) (alice bob : ℕ), 
  alice ∈ students → bob ∈ students → students.card = 8 → ∃ (committees : Finset (Finset ℕ)),
  (∀ committee ∈ committees, alice ∈ committee ∧ bob ∈ committee) ∧
  ∀ committee ∈ committees, committee.card = 5 ∧ committees.card = 20 :=
by
  sorry

end five_student_committee_l103_103534


namespace electrical_bill_undetermined_l103_103726

theorem electrical_bill_undetermined
    (gas_bill : ℝ)
    (gas_paid_fraction : ℝ)
    (additional_gas_payment : ℝ)
    (water_bill : ℝ)
    (water_paid_fraction : ℝ)
    (internet_bill : ℝ)
    (internet_payments : ℝ)
    (payment_amounts: ℝ)
    (total_remaining : ℝ) :
    gas_bill = 40 →
    gas_paid_fraction = 3 / 4 →
    additional_gas_payment = 5 →
    water_bill = 40 →
    water_paid_fraction = 1 / 2 →
    internet_bill = 25 →
    internet_payments = 4 * 5 →
    total_remaining = 30 →
    (∃ electricity_bill : ℝ, true) -> 
    false := by
  intro gas_bill_eq gas_paid_fraction_eq additional_gas_payment_eq
  intro water_bill_eq water_paid_fraction_eq
  intro internet_bill_eq internet_payments_eq 
  intro total_remaining_eq 
  intro exists_electricity_bill 
  sorry -- Proof that the electricity bill cannot be determined

end electrical_bill_undetermined_l103_103726


namespace range_of_a_l103_103248

-- Definitions of the conditions
def p (x : ℝ) : Prop := x^2 - 8 * x - 20 < 0
def q (x : ℝ) (a : ℝ) : Prop := x^2 - 2 * x + 1 - a^2 ≤ 0 ∧ a > 0

-- Statement of the theorem that proves the range of a
theorem range_of_a (x : ℝ) (a : ℝ) :
  (¬ (p x) → ¬ (q x a)) ∧ (¬ (q x a) → ¬ (p x)) → (a ≥ 9) :=
by
  sorry

end range_of_a_l103_103248


namespace dave_guitar_strings_l103_103779

theorem dave_guitar_strings (strings_per_night : ℕ) (shows_per_week : ℕ) (weeks : ℕ)
  (h1 : strings_per_night = 4)
  (h2 : shows_per_week = 6)
  (h3 : weeks = 24) : 
  strings_per_night * shows_per_week * weeks = 576 :=
by
  sorry

end dave_guitar_strings_l103_103779


namespace geometric_sum_formula_l103_103074

noncomputable def geometric_sequence_sum (n : ℕ) : ℕ :=
  sorry

theorem geometric_sum_formula (a : ℕ → ℕ)
  (h_geom : ∀ n, a (n + 1) = 2 * a n)
  (h_a1_a2 : a 0 + a 1 = 3)
  (h_a1_a2_a3 : a 0 * a 1 * a 2 = 8) :
  geometric_sequence_sum n = 2^n - 1 :=
sorry

end geometric_sum_formula_l103_103074


namespace master_codes_count_l103_103944

def num_colors : ℕ := 7
def num_slots : ℕ := 5

theorem master_codes_count : num_colors ^ num_slots = 16807 := by
  sorry

end master_codes_count_l103_103944


namespace boxes_of_nuts_purchased_l103_103668

theorem boxes_of_nuts_purchased (b : ℕ) (n : ℕ) (bolts_used : ℕ := 7 * 11 - 3) 
    (nuts_used : ℕ := 113 - bolts_used) (total_nuts : ℕ := nuts_used + 6) 
    (nuts_per_box : ℕ := 15) (h_bolts_boxes : b = 7) 
    (h_bolts_per_box : ∀ x, b * x = 77) 
    (h_nuts_boxes : ∃ x, n = x * nuts_per_box)
    : ∃ k, n = k * 15 ∧ k = 3 :=
by
  sorry

end boxes_of_nuts_purchased_l103_103668


namespace find_n_l103_103287

-- Define the arithmetic sequence
def arithmetic_sequence (n : ℕ) : ℕ :=
  11 + (n - 1) * 6

-- State the problem
theorem find_n (n : ℕ) : 
  (∀ m : ℕ, m ≥ n → arithmetic_sequence m > 2017) ↔ n = 336 :=
by
  sorry

end find_n_l103_103287


namespace earnings_per_widget_l103_103336

-- Defining the conditions as constants
def hours_per_week : ℝ := 40
def hourly_wage : ℝ := 12.50
def total_weekly_earnings : ℝ := 700
def widgets_produced : ℝ := 1250

-- We need to prove earnings per widget
theorem earnings_per_widget :
  (total_weekly_earnings - (hours_per_week * hourly_wage)) / widgets_produced = 0.16 := by
  sorry

end earnings_per_widget_l103_103336


namespace smallest_n_cube_ends_with_2016_l103_103098

theorem smallest_n_cube_ends_with_2016 : ∃ n : ℕ, (n^3 % 10000 = 2016) ∧ (∀ m : ℕ, (m^3 % 10000 = 2016) → n ≤ m) :=
sorry

end smallest_n_cube_ends_with_2016_l103_103098


namespace multiplication_identity_l103_103388

theorem multiplication_identity (x y z w : ℝ) (h1 : x = 2000) (h2 : y = 2992) (h3 : z = 0.2992) (h4 : w = 20) : 
  x * y * z * w = 4 * y^2 :=
by
  sorry

end multiplication_identity_l103_103388


namespace simplify_expression_l103_103133

theorem simplify_expression (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a ≠ b) (h4 : a ≠ -b) : 
  ((a^3 - a^2 * b) / (a^2 * b) - (a^2 * b - b^3) / (a * b - b^2) - (a * b) / (a^2 - b^2)) = 
  (-3 * a) / (a^2 - b^2) := 
by
  sorry

end simplify_expression_l103_103133


namespace guo_can_pay_exact_amount_l103_103552

-- Define the denominations and total amount Guo has
def note_denominations := [1, 10, 20, 50]
def total_amount := 20000
def cost_computer := 10000

-- The main theorem stating that Guo can pay exactly 10000 yuan
theorem guo_can_pay_exact_amount : ∃ bills : List ℕ, ∀ (b : ℕ), b ∈ bills → b ∈ note_denominations ∧
  bills.sum = cost_computer :=
sorry

end guo_can_pay_exact_amount_l103_103552


namespace mosquito_shadow_speed_l103_103762

theorem mosquito_shadow_speed
  (v : ℝ) (t : ℝ) (h : ℝ) (cos_theta : ℝ) (v_shadow : ℝ)
  (hv : v = 0.5) (ht : t = 20) (hh : h = 6) (hcos_theta : cos_theta = 0.6) :
  v_shadow = 0 ∨ v_shadow = 0.8 :=
  sorry

end mosquito_shadow_speed_l103_103762


namespace girls_at_picnic_l103_103420

variables (g b : ℕ)

-- Conditions
axiom total_students : g + b = 1500
axiom students_at_picnic : (3/4) * g + (2/3) * b = 900

-- Goal: Prove number of girls who attended the picnic
theorem girls_at_picnic (hg : (3/4 : ℚ) * 1200 = 900) : (3/4 : ℚ) * 1200 = 900 :=
by sorry

end girls_at_picnic_l103_103420


namespace gina_can_paint_6_rose_cups_an_hour_l103_103535

def number_of_rose_cups_painted_in_an_hour 
  (R : ℕ) (lily_rate : ℕ) (rose_order : ℕ) (lily_order : ℕ) (total_payment : ℕ) (hourly_rate : ℕ)
  (lily_hours : ℕ) (total_hours : ℕ) (rose_hours : ℕ) : Prop :=
  (lily_rate = 7) ∧
  (rose_order = 6) ∧
  (lily_order = 14) ∧
  (total_payment = 90) ∧
  (hourly_rate = 30) ∧
  (lily_hours = lily_order / lily_rate) ∧
  (total_hours = total_payment / hourly_rate) ∧
  (rose_hours = total_hours - lily_hours) ∧
  (rose_order = R * rose_hours)

theorem gina_can_paint_6_rose_cups_an_hour :
  ∃ R, number_of_rose_cups_painted_in_an_hour 
    R 7 6 14 90 30 (14 / 7) (90 / 30)  (90 / 30 - 14 / 7) ∧ R = 6 :=
by
  -- proof is left out intentionally
  sorry

end gina_can_paint_6_rose_cups_an_hour_l103_103535


namespace minimum_value_of_x_plus_y_l103_103827

theorem minimum_value_of_x_plus_y
  (x y : ℝ)
  (h1 : x > y)
  (h2 : y > 0)
  (h3 : 1 / (x - y) + 8 / (x + 2 * y) = 1) :
  x + y = 25 / 3 :=
sorry

end minimum_value_of_x_plus_y_l103_103827


namespace gcd_square_product_l103_103957

theorem gcd_square_product (x y z : ℕ) (h : 1 / (x : ℝ) - 1 / (y : ℝ) = 1 / (z : ℝ)) : 
    ∃ n : ℕ, gcd x (gcd y z) * x * y * z = n * n := 
sorry

end gcd_square_product_l103_103957


namespace markers_per_box_l103_103207

theorem markers_per_box
  (students : ℕ) (boxes : ℕ) (group1_students : ℕ) (group1_markers : ℕ)
  (group2_students : ℕ) (group2_markers : ℕ) (last_group_markers : ℕ)
  (h_students : students = 30)
  (h_boxes : boxes = 22)
  (h_group1_students : group1_students = 10)
  (h_group1_markers : group1_markers = 2)
  (h_group2_students : group2_students = 15)
  (h_group2_markers : group2_markers = 4)
  (h_last_group_markers : last_group_markers = 6) :
  (110 = students * ((group1_students * group1_markers + group2_students * group2_markers + (students - group1_students - group2_students) * last_group_markers)) / boxes) :=
by
  sorry

end markers_per_box_l103_103207


namespace sum_first_8_even_numbers_is_72_l103_103004

theorem sum_first_8_even_numbers_is_72 : (2 + 4 + 6 + 8 + 10 + 12 + 14 + 16) = 72 :=
by
  sorry

end sum_first_8_even_numbers_is_72_l103_103004


namespace problem_statement_l103_103596

variable (q p : ℚ)
#check λ (q p : ℚ), q / p

theorem problem_statement :
  let q := (119 : ℚ) / 8
  let p := -(3 : ℚ) / 8
  q / p = -(119 : ℚ) / 3 :=
by
  let q := (119 : ℚ) / 8
  let p := -(3 : ℚ) / 8
  sorry

end problem_statement_l103_103596


namespace subset_P_Q_l103_103815

def P := {x : ℝ | x > 1}
def Q := {x : ℝ | x^2 - x > 0}

theorem subset_P_Q : P ⊆ Q :=
by
  sorry

end subset_P_Q_l103_103815


namespace perpendicular_lines_and_slope_l103_103154

theorem perpendicular_lines_and_slope (b : ℝ) : (x + 3 * y + 4 = 0) ∧ (b * x + 3 * y + 6 = 0) → b = -9 :=
by
  sorry

end perpendicular_lines_and_slope_l103_103154


namespace chose_number_l103_103821

theorem chose_number (x : ℝ) (h : (x / 12)^2 - 240 = 8) : x = 24 * Real.sqrt 62 :=
sorry

end chose_number_l103_103821


namespace cube_root_of_4913_has_unit_digit_7_cube_root_of_50653_is_37_cube_root_of_110592_is_48_l103_103948

theorem cube_root_of_4913_has_unit_digit_7 :
  (∃ (y : ℕ), y^3 = 4913 ∧ y % 10 = 7) :=
sorry

theorem cube_root_of_50653_is_37 :
  (∃ (y : ℕ), y = 37 ∧ y^3 = 50653) :=
sorry

theorem cube_root_of_110592_is_48 :
  (∃ (y : ℕ), y = 48 ∧ y^3 = 110592) :=
sorry

end cube_root_of_4913_has_unit_digit_7_cube_root_of_50653_is_37_cube_root_of_110592_is_48_l103_103948


namespace seating_probability_l103_103482

/-- There are 9 representatives from 3 different countries, 
    each country having 3 representatives. They are seated randomly 
    around a round table with 9 chairs. The probability that each 
    representative has at least one representative from another country 
    sitting next to them is 41/56. -/
theorem seating_probability (total_representatives : ℕ) (countries : ℕ) (reps_per_country : ℕ) :
  total_representatives = 9 → 
  countries = 3 → 
  reps_per_country = 3 → 
  let total_arrangements := (Nat.factorial total_representatives) / (Nat.factorial reps_per_country * Nat.factorial reps_per_country * Nat.factorial reps_per_country) in
  let favorable_arrangements := total_arrangements - 450 in
  (favorable_arrangements / total_arrangements : ℚ) = 41 / 56 :=
by
  intros htotal hcountries hreps
  simp [total_arrangements, favorable_arrangements, Nat.factorial]
  sorry

end seating_probability_l103_103482


namespace factor_difference_of_squares_l103_103061

-- Given: x is a real number.
-- Prove: x^2 - 64 = (x - 8) * (x + 8).
theorem factor_difference_of_squares (x : ℝ) : 
  x^2 - 64 = (x - 8) * (x + 8) :=
by
  sorry

end factor_difference_of_squares_l103_103061


namespace lottery_probability_l103_103559

theorem lottery_probability (p: ℝ) :
  (∀ n, 1 ≤ n ∧ n ≤ 15 → p = 2/3) →
  (true → p = 0.6666666666666666) →
  p = 2/3 :=
by
  intros h h'
  sorry

end lottery_probability_l103_103559


namespace mildred_weight_is_correct_l103_103451

noncomputable def carol_weight := 9
noncomputable def mildred_weight := carol_weight + 50

theorem mildred_weight_is_correct : mildred_weight = 59 :=
by 
  -- the proof is omitted
  sorry

end mildred_weight_is_correct_l103_103451


namespace find_g_inv_f_3_l103_103266

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def g_inv : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry

axiom f_inv_g_eq : ∀ x : ℝ, f_inv (g x) = x^4 - x + 2
axiom g_has_inverse : ∀ y : ℝ, g (g_inv y) = y 

theorem find_g_inv_f_3 :
  ∃ α : ℝ, (α^4 - α - 1 = 0) ∧ g_inv (f 3) = α :=
sorry

end find_g_inv_f_3_l103_103266


namespace quadratic_common_root_inverse_other_roots_l103_103120

variables (p q r s : ℝ)
variables (hq : q ≠ -1) (hs : s ≠ -1)

theorem quadratic_common_root_inverse_other_roots :
  (∃ a b : ℝ, (a ≠ b) ∧ (a^2 + p * a + q = 0) ∧ (a * b = 1) ∧ (b^2 + r * b + s = 0)) ↔ 
  (p * r = (q + 1) * (s + 1) ∧ p * (q + 1) * s = r * (s + 1) * q) :=
sorry

end quadratic_common_root_inverse_other_roots_l103_103120


namespace angle_triple_of_supplement_l103_103330

theorem angle_triple_of_supplement (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end angle_triple_of_supplement_l103_103330


namespace find_sample_size_l103_103037

def sports_team (total: Nat) (soccer: Nat) (basketball: Nat) (table_tennis: Nat) : Prop :=
  total = soccer + basketball + table_tennis

def valid_sample_size (total: Nat) (n: Nat) :=
  (n > 0) ∧ (total % n == 0) ∧ (n % 6 == 0)

def systematic_sampling_interval (total: Nat) (n: Nat): Nat :=
  total / n

theorem find_sample_size :
  ∀ (total soccer basketball table_tennis: Nat),
  sports_team total soccer basketball table_tennis →
  total = 36 →
  soccer = 18 →
  basketball = 12 →
  table_tennis = 6 →
  (∃ n, valid_sample_size total n ∧ valid_sample_size (total - 1) (n + 1)) →
  ∃ n, n = 6 := by
  sorry

end find_sample_size_l103_103037


namespace largest_and_next_largest_difference_l103_103983

theorem largest_and_next_largest_difference (a b c : ℕ) (h1: a = 10) (h2: b = 11) (h3: c = 12) : 
  let largest := max a (max b c)
  let next_largest := min (max a b) (max (min a b) c)
  largest - next_largest = 1 :=
by
  -- Proof to be filled in for verification
  sorry

end largest_and_next_largest_difference_l103_103983


namespace min_cut_length_l103_103485

theorem min_cut_length (x : ℝ) (h_longer : 23 - x ≥ 0) (h_shorter : 15 - x ≥ 0) :
  23 - x ≥ 2 * (15 - x) → x ≥ 7 :=
by
  sorry

end min_cut_length_l103_103485


namespace Noah_age_in_10_years_is_22_l103_103454

def Joe_age : Nat := 6
def Noah_age := 2 * Joe_age
def Noah_age_after_10_years := Noah_age + 10

theorem Noah_age_in_10_years_is_22 : Noah_age_after_10_years = 22 := by
  sorry

end Noah_age_in_10_years_is_22_l103_103454


namespace find_x_for_g_l103_103694

noncomputable def g (x : ℝ) : ℝ := (↑((x + 5) / 3) : ℝ)^(1/3 : ℝ)

theorem find_x_for_g :
  ∃ x : ℝ, g (3 * x) = 3 * g x ↔ x = -65 / 12 :=
by
  sorry

end find_x_for_g_l103_103694


namespace unique_two_digit_integer_l103_103527

theorem unique_two_digit_integer (s : ℕ) (hs : s > 9 ∧ s < 100) (h : 13 * s ≡ 42 [MOD 100]) : s = 34 :=
by sorry

end unique_two_digit_integer_l103_103527


namespace lowest_price_is_six_l103_103377

def single_package_cost : ℝ := 7
def eight_oz_package_cost : ℝ := 4
def four_oz_package_original_cost : ℝ := 2
def discount_rate : ℝ := 0.5

theorem lowest_price_is_six
  (cost_single : single_package_cost = 7)
  (cost_eight : eight_oz_package_cost = 4)
  (cost_four : four_oz_package_original_cost = 2)
  (discount : discount_rate = 0.5) :
  min single_package_cost (eight_oz_package_cost + 2 * (four_oz_package_original_cost * discount_rate)) = 6 := by
  sorry

end lowest_price_is_six_l103_103377


namespace height_of_E_l103_103220

variable {h_E h_F h_G h_H : ℝ}

theorem height_of_E (h1 : h_E + h_F + h_G + h_H = 2 * (h_E + h_F))
                    (h2 : (h_E + h_F) / 2 = (h_E + h_G) / 2 - 4)
                    (h3 : h_H = h_E - 10)
                    (h4 : h_F + h_G = 288) :
  h_E = 139 :=
by
  sorry

end height_of_E_l103_103220


namespace minimum_value_expression_l103_103826

theorem minimum_value_expression (a b c : ℤ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
    3 * a^2 + 2 * b^2 + 4 * c^2 - a * b - 3 * b * c - 5 * c * a ≥ 6 :=
sorry

end minimum_value_expression_l103_103826


namespace sum_is_correct_l103_103576

-- Define the variables and conditions
variables (a b c d : ℝ)
variable (x : ℝ)

-- Define the condition
def condition : Prop :=
  a + 1 = x ∧
  b + 2 = x ∧
  c + 3 = x ∧
  d + 4 = x ∧
  a + b + c + d + 5 = x

-- The theorem we need to prove
theorem sum_is_correct (h : condition a b c d x) : a + b + c + d = -10 / 3 :=
  sorry

end sum_is_correct_l103_103576


namespace portrait_in_silver_box_l103_103067

-- Definitions for the first trial
def gold_box_1 : Prop := false
def gold_box_2 : Prop := true
def silver_box_1 : Prop := true
def silver_box_2 : Prop := false
def lead_box_1 : Prop := false
def lead_box_2 : Prop := true

-- Definitions for the second trial
def gold_box_3 : Prop := false
def gold_box_4 : Prop := true
def silver_box_3 : Prop := true
def silver_box_4 : Prop := false
def lead_box_3 : Prop := false
def lead_box_4 : Prop := true

-- The main theorem statement
theorem portrait_in_silver_box
  (gold_b1 : gold_box_1 = false)
  (gold_b2 : gold_box_2 = true)
  (silver_b1 : silver_box_1 = true)
  (silver_b2 : silver_box_2 = false)
  (lead_b1 : lead_box_1 = false)
  (lead_b2 : lead_box_2 = true)
  (gold_b3 : gold_box_3 = false)
  (gold_b4 : gold_box_4 = true)
  (silver_b3 : silver_box_3 = true)
  (silver_b4 : silver_box_4 = false)
  (lead_b3 : lead_box_3 = false)
  (lead_b4 : lead_box_4 = true) : 
  (silver_box_1 ∧ ¬lead_box_2) ∧ (silver_box_3 ∧ ¬lead_box_4) :=
sorry

end portrait_in_silver_box_l103_103067


namespace elsie_money_l103_103880

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

theorem elsie_money : 
  compound_interest 2500 0.04 20 = 5477.81 :=
by 
  sorry

end elsie_money_l103_103880


namespace equal_12_mn_P_2n_Q_m_l103_103144

-- Define P and Q based on given conditions
def P (m : ℕ) : ℕ := 2 ^ m
def Q (n : ℕ) : ℕ := 3 ^ n

-- The theorem to prove
theorem equal_12_mn_P_2n_Q_m (m n : ℕ) : (12 ^ (m * n)) = (P m ^ (2 * n)) * (Q n ^ m) :=
by
  -- Proof goes here
  sorry

end equal_12_mn_P_2n_Q_m_l103_103144


namespace product_of_roots_l103_103094

theorem product_of_roots (x1 x2 : ℝ) (h1 : x1 ^ 2 - 2 * x1 = 2) (h2 : x2 ^ 2 - 2 * x2 = 2) (hne : x1 ≠ x2) :
  x1 * x2 = -2 := 
sorry

end product_of_roots_l103_103094


namespace artist_used_17_ounces_of_paint_l103_103508

def ounces_used_per_large_canvas : ℕ := 3
def ounces_used_per_small_canvas : ℕ := 2
def large_paintings_completed : ℕ := 3
def small_paintings_completed : ℕ := 4

theorem artist_used_17_ounces_of_paint :
  (ounces_used_per_large_canvas * large_paintings_completed + ounces_used_per_small_canvas * small_paintings_completed = 17) :=
by
  sorry

end artist_used_17_ounces_of_paint_l103_103508


namespace simplify_expression_l103_103969

section
variable (a b : ℚ) (h_a : a = -1) (h_b : b = 1/4)

theorem simplify_expression : 
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by
  sorry
end

end simplify_expression_l103_103969


namespace smallest_k_value_eq_sqrt475_div_12_l103_103351

theorem smallest_k_value_eq_sqrt475_div_12 :
  ∀ (k : ℝ), (dist (⟨5 * Real.sqrt 3, k - 2⟩ : ℝ × ℝ) ⟨0, 0⟩ = 5 * k) →
  k = (1 + Real.sqrt 475) / 12 := 
by
  intro k
  sorry

end smallest_k_value_eq_sqrt475_div_12_l103_103351


namespace average_age_students_l103_103560

theorem average_age_students 
  (total_students : ℕ)
  (group1 : ℕ)
  (group1_avg_age : ℕ)
  (group2 : ℕ)
  (group2_avg_age : ℕ)
  (student15_age : ℕ)
  (avg_age : ℕ) 
  (h1 : total_students = 15)
  (h2 : group1_avg_age = 14)
  (h3 : group2 = 8)
  (h4 : group2_avg_age = 16)
  (h5 : student15_age = 13)
  (h6 : avg_age = (84 + 128 + 13) / 15)
  (h7 : avg_age = 15) :
  group1 = 6 :=
by sorry

end average_age_students_l103_103560


namespace mad_hatter_must_secure_at_least_70_percent_l103_103833

theorem mad_hatter_must_secure_at_least_70_percent :
  ∀ (N : ℕ) (uM uH uD : ℝ) (α : ℝ),
    uM = 0.2 ∧ uH = 0.25 ∧ uD = 0.3 → 
    uM + α * 0.25 ≥ 0.25 + (1 - α) * 0.25 ∧
    uM + α * 0.25 ≥ 0.3 + (1 - α) * 0.25 →
    α ≥ 0.7 :=
by
  intros N uM uH uD α h hx
  sorry 

end mad_hatter_must_secure_at_least_70_percent_l103_103833


namespace function_satisfies_equation_l103_103653

noncomputable def f (x : ℝ) : ℝ := x + 1 / x + 1 / (x - 1)

theorem function_satisfies_equation (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1) :
  f ((x - 1) / x) + f (1 / (1 - x)) = 2 - 2 * x := by
  sorry

end function_satisfies_equation_l103_103653


namespace car_A_overtakes_car_B_l103_103003

theorem car_A_overtakes_car_B (z : ℕ) :
  let y := (5 * z) / 4
  let x := (13 * z) / 10
  10 * y / (x - y) = 250 := 
by
  sorry

end car_A_overtakes_car_B_l103_103003


namespace intersection_complement_l103_103550

-- Definitions of the sets
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2, 5}

-- Complement of B with respect to U
def comp_B : Set ℕ := U \ B

-- Statement to be proven
theorem intersection_complement : A ∩ comp_B = {1, 3} :=
by 
  sorry

end intersection_complement_l103_103550


namespace percentage_of_liquid_X_in_solution_A_l103_103124

theorem percentage_of_liquid_X_in_solution_A (P : ℝ) :
  (0.018 * 700 / 1200 + P * 500 / 1200) = 0.0166 → P = 0.01464 :=
by 
  sorry

end percentage_of_liquid_X_in_solution_A_l103_103124


namespace concentration_after_5500_evaporates_l103_103038

noncomputable def concentration_after_evaporation 
  (V₀ Vₑ : ℝ) (C₀ : ℝ) : ℝ := 
  let sodium_chloride := C₀ * V₀
  let remaining_volume := V₀ - Vₑ
  100 * sodium_chloride / remaining_volume

theorem concentration_after_5500_evaporates 
  : concentration_after_evaporation 10000 5500 0.05 = 11.11 := 
by
  -- Formalize the calculations as we have derived
  -- sorry is used to skip the proof
  sorry

end concentration_after_5500_evaporates_l103_103038


namespace minimum_value_expression_l103_103123

theorem minimum_value_expression (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 27) :
  (x^2 + 6 * x * y + 9 * y^2 + 3/2 * z^2) ≥ 102 :=
sorry

end minimum_value_expression_l103_103123


namespace solve_system_and_find_6a_plus_b_l103_103686

theorem solve_system_and_find_6a_plus_b (x y a b : ℝ)
  (h1 : 3 * x - 2 * y + 20 = 0)
  (h2 : 2 * x + 15 * y - 3 = 0)
  (h3 : a * x - b * y = 3) :
  6 * a + b = -3 := by
  sorry

end solve_system_and_find_6a_plus_b_l103_103686


namespace fraction_identity_l103_103030

theorem fraction_identity (N F : ℝ) (hN : N = 8) (h : 0.5 * N = F * N + 2) : F = 1 / 4 :=
by {
  -- proof will go here
  sorry
}

end fraction_identity_l103_103030


namespace birds_not_hawks_warbler_kingfisher_l103_103027

variables (B : ℝ)
variables (hawks paddyfield_warblers kingfishers : ℝ)

-- Conditions
def condition1 := hawks = 0.30 * B
def condition2 := paddyfield_warblers = 0.40 * (B - hawks)
def condition3 := kingfishers = 0.25 * paddyfield_warblers

-- Question: Prove the percentage of birds that are not hawks, paddyfield-warblers, or kingfishers is 35%
theorem birds_not_hawks_warbler_kingfisher (B hawks paddyfield_warblers kingfishers : ℝ) 
 (h1 : hawks = 0.30 * B) 
 (h2 : paddyfield_warblers = 0.40 * (B - hawks)) 
 (h3 : kingfishers = 0.25 * paddyfield_warblers) : 
 (1 - (hawks + paddyfield_warblers + kingfishers) / B) * 100 = 35 :=
by
  sorry

end birds_not_hawks_warbler_kingfisher_l103_103027


namespace balloon_height_l103_103820

theorem balloon_height :
  let initial_money : ℝ := 200
  let cost_sheet : ℝ := 42
  let cost_rope : ℝ := 18
  let cost_tank_and_burner : ℝ := 14
  let helium_price_per_ounce : ℝ := 1.5
  let lift_per_ounce : ℝ := 113
  let remaining_money := initial_money - cost_sheet - cost_rope - cost_tank_and_burner
  let ounces_of_helium := remaining_money / helium_price_per_ounce
  let height := ounces_of_helium * lift_per_ounce
  height = 9492 :=
by
  sorry

end balloon_height_l103_103820


namespace num_tables_l103_103761

/-- Given conditions related to tables, stools, and benches, we want to prove the number of tables -/
theorem num_tables 
  (t s b : ℕ) 
  (h1 : s = 8 * t)
  (h2 : b = 2 * t)
  (h3 : 3 * s + 6 * b + 4 * t = 816) : 
  t = 20 := 
sorry

end num_tables_l103_103761


namespace fraction_meaningful_l103_103484

theorem fraction_meaningful (x : ℝ) : (x ≠ 2) ↔ (x - 2 ≠ 0) :=
by
  sorry

end fraction_meaningful_l103_103484


namespace possible_values_of_quadratic_l103_103521

theorem possible_values_of_quadratic (x : ℝ) (hx : x^2 - 7 * x + 12 < 0) :
  1.75 ≤ x^2 - 7 * x + 14 ∧ x^2 - 7 * x + 14 ≤ 2 := by
  sorry

end possible_values_of_quadratic_l103_103521


namespace bens_old_car_cost_l103_103771

theorem bens_old_car_cost :
  ∃ (O N : ℕ), N = 2 * O ∧ O = 1800 ∧ N = 1800 + 2000 ∧ O = 1900 :=
by 
  sorry

end bens_old_car_cost_l103_103771


namespace min_sum_l103_103405

variable {a b : ℝ}

theorem min_sum (h1 : a > 0) (h2 : b > 0) (h3 : a * b ^ 2 = 4) : a + b ≥ 3 := 
sorry

end min_sum_l103_103405


namespace gamin_difference_calculation_l103_103915

def largest_number : ℕ := 532
def smallest_number : ℕ := 406
def difference : ℕ := 126

theorem gamin_difference_calculation : largest_number - smallest_number = difference :=
by
  -- The solution proves that the difference between the largest and smallest numbers is 126.
  sorry

end gamin_difference_calculation_l103_103915


namespace smallest_solution_of_quadratic_l103_103194

theorem smallest_solution_of_quadratic :
  ∃ x : ℝ, 6 * x^2 - 29 * x + 35 = 0 ∧ x = 7 / 3 :=
sorry

end smallest_solution_of_quadratic_l103_103194


namespace graph_symmetry_l103_103532

theorem graph_symmetry (f : ℝ → ℝ) : 
  ∀ x : ℝ, f (x - 1) = f (-(x - 1)) ↔ x = 1 :=
by 
  sorry

end graph_symmetry_l103_103532


namespace small_boxes_in_big_box_l103_103758

theorem small_boxes_in_big_box (total_candles : ℕ) (candles_per_small : ℕ) (total_big_boxes : ℕ) 
  (h1 : total_candles = 8000) 
  (h2 : candles_per_small = 40) 
  (h3 : total_big_boxes = 50) :
  (total_candles / candles_per_small) / total_big_boxes = 4 :=
by
  sorry

end small_boxes_in_big_box_l103_103758


namespace prism_faces_l103_103263

theorem prism_faces (E V F n : ℕ) (h1 : E + V = 30) (h2 : F + V = E + 2) (h3 : E = 3 * n) : F = 8 :=
by
  -- Actual proof omitted
  sorry

end prism_faces_l103_103263


namespace evaluate_expression_l103_103986

theorem evaluate_expression : 
  ∀ (x y z : ℝ), 
  x = 2 → 
  y = -3 → 
  z = 1 → 
  x^2 + y^2 + z^2 + 2 * x * y - z^3 = 1 := by
  intros x y z hx hy hz
  rw [hx, hy, hz]
  sorry

end evaluate_expression_l103_103986


namespace sum_of_first_20_terms_l103_103410

variable {a : ℕ → ℕ}

-- Conditions given in the problem
axiom seq_property : ∀ n, a n + 2 * a (n + 1) = 3 * n + 2
axiom arithmetic_sequence : ∀ n m, a (n + 1) - a n = a (m + 1) - a m

-- Theorem to be proved
theorem sum_of_first_20_terms (a : ℕ → ℕ) (S20 := (Finset.range 20).sum a) :
  S20 = 210 :=
  sorry

end sum_of_first_20_terms_l103_103410


namespace missing_number_is_twelve_l103_103663

theorem missing_number_is_twelve
  (x : ℤ)
  (h : 10010 - x * 3 * 2 = 9938) :
  x = 12 :=
sorry

end missing_number_is_twelve_l103_103663


namespace fraction_equation_l103_103470

theorem fraction_equation (a : ℕ) (h : a > 0) (eq : (a : ℚ) / (a + 35) = 0.875) : a = 245 :=
by
  sorry

end fraction_equation_l103_103470


namespace renee_allergic_probability_l103_103438

theorem renee_allergic_probability :
  let peanut_butter_from_jenny := 40
  let chocolate_chip_from_jenny := 50
  let peanut_butter_from_marcus := 30
  let lemon_from_marcus := 20
  let total_cookies := peanut_butter_from_jenny + chocolate_chip_from_jenny + peanut_butter_from_marcus + lemon_from_marcus
  let total_peanut_butter := peanut_butter_from_jenny + peanut_butter_from_marcus
  let p := (total_peanut_butter : ℝ) / (total_cookies : ℝ) * 100
  in p = 50 := by sorry

end renee_allergic_probability_l103_103438


namespace mushrooms_collected_l103_103591

variable (P V : ℕ)

theorem mushrooms_collected (h1 : P = (V * 100) / (P + V)) (h2 : V % 2 = 1) :
  P + V = 25 ∨ P + V = 300 ∨ P + V = 525 ∨ P + V = 1900 ∨ P + V = 9900 := by
  sorry

end mushrooms_collected_l103_103591


namespace reported_length_correct_l103_103475

def length_in_yards := 80
def conversion_factor := 3 -- 1 yard is 3 feet
def length_in_feet := 240

theorem reported_length_correct :
  length_in_feet = length_in_yards * conversion_factor :=
by rfl

end reported_length_correct_l103_103475


namespace cost_of_old_car_l103_103774

theorem cost_of_old_car (C_old C_new : ℝ): 
  C_new = 2 * C_old → 
  1800 + 2000 = C_new → 
  C_old = 1900 :=
by
  intros H1 H2
  sorry

end cost_of_old_car_l103_103774


namespace evaluate_expression_l103_103909

theorem evaluate_expression (a b : ℕ) (h1 : a = 3) (h2 : b = 2) : (a^3 + b)^2 - (a^3 - b)^2 = 216 :=
by
  -- Proof is not required, add sorry to skip the proof
  sorry

end evaluate_expression_l103_103909


namespace count_n_satisfies_conditions_l103_103421

theorem count_n_satisfies_conditions :
  ∃ (count : ℕ), count = 36 ∧ ∀ (n : ℕ), 
    0 < n ∧ n < 150 →
    ∃ (k : ℕ), 
    (n = 2*k + 2) ∧ 
    (k*(k + 2) % 4 = 0) :=
by
  sorry

end count_n_satisfies_conditions_l103_103421


namespace helium_balloon_height_l103_103817

theorem helium_balloon_height :
    let total_budget := 200
    let cost_sheet := 42
    let cost_rope := 18
    let cost_propane := 14
    let cost_per_ounce_helium := 1.5
    let height_per_ounce := 113
    let amount_spent := cost_sheet + cost_rope + cost_propane
    let money_left_for_helium := total_budget - amount_spent
    let ounces_helium := money_left_for_helium / cost_per_ounce_helium
    let total_height := height_per_ounce * ounces_helium
    total_height = 9492 := sorry

end helium_balloon_height_l103_103817


namespace area_of_triangle_LEF_l103_103946

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

end area_of_triangle_LEF_l103_103946


namespace find_perimeter_square3_l103_103202

-- Define the conditions: perimeter of first and second square
def perimeter_square1 := 60
def perimeter_square2 := 48

-- Calculate side lengths based on the perimeter
def side_length_square1 := perimeter_square1 / 4
def side_length_square2 := perimeter_square2 / 4

-- Calculate areas of the two squares
def area_square1 := side_length_square1 * side_length_square1
def area_square2 := side_length_square2 * side_length_square2

-- Calculate the area of the third square
def area_square3 := area_square1 - area_square2

-- Calculate the side length of the third square
def side_length_square3 := Nat.sqrt area_square3

-- Define the perimeter of the third square
def perimeter_square3 := 4 * side_length_square3

/-- Theorem: The perimeter of the third square is 36 cm -/
theorem find_perimeter_square3 : perimeter_square3 = 36 := by
  sorry

end find_perimeter_square3_l103_103202


namespace solve_equation_l103_103464

theorem solve_equation (x y z : ℕ) : (3 ^ x + 5 ^ y + 14 = z!) ↔ ((x = 4 ∧ y = 2 ∧ z = 5) ∨ (x = 4 ∧ y = 4 ∧ z = 6)) :=
by
  sorry

end solve_equation_l103_103464


namespace sqrt_27_eq_3_sqrt_3_l103_103137

theorem sqrt_27_eq_3_sqrt_3 : Real.sqrt 27 = 3 * Real.sqrt 3 :=
by
  sorry

end sqrt_27_eq_3_sqrt_3_l103_103137


namespace AB_squared_eq_AB_AB_minus_BA_cubed_eq_zero_l103_103578

variables {n : ℕ} (A B : Matrix (Fin n) (Fin n) ℂ)

-- Condition
def AB2A_eq_AB := A * B ^ 2 * A = A * B * A

-- Part (a): Prove that (AB)^2 = AB
theorem AB_squared_eq_AB (h : AB2A_eq_AB A B) : (A * B) ^ 2 = A * B :=
sorry

-- Part (b): Prove that (AB - BA)^3 = 0
theorem AB_minus_BA_cubed_eq_zero (h : AB2A_eq_AB A B) : (A * B - B * A) ^ 3 = 0 :=
sorry

end AB_squared_eq_AB_AB_minus_BA_cubed_eq_zero_l103_103578


namespace Noah_age_in_10_years_is_22_l103_103455

def Joe_age : Nat := 6
def Noah_age := 2 * Joe_age
def Noah_age_after_10_years := Noah_age + 10

theorem Noah_age_in_10_years_is_22 : Noah_age_after_10_years = 22 := by
  sorry

end Noah_age_in_10_years_is_22_l103_103455


namespace sum_of_D_coordinates_l103_103592

-- Define points as tuples for coordinates (x, y)
structure Point :=
  (x : ℤ)
  (y : ℤ)

def midpoint (A B : Point) : Point :=
  ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩

noncomputable def pointD : Point :=
  let C := Point.mk 11 5
  let N := Point.mk 5 9
  let x := 2 * N.x - C.x
  let y := 2 * N.y - C.y
  Point.mk x y

theorem sum_of_D_coordinates : 
  let D := pointD
  D.x + D.y = 12 := by
  sorry

end sum_of_D_coordinates_l103_103592


namespace factor_difference_of_squares_l103_103058

theorem factor_difference_of_squares (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := 
by 
  sorry

end factor_difference_of_squares_l103_103058


namespace sum_proper_divisors_243_l103_103016

theorem sum_proper_divisors_243 : (1 + 3 + 9 + 27 + 81) = 121 :=
by
  sorry

end sum_proper_divisors_243_l103_103016


namespace number_of_customers_l103_103117

theorem number_of_customers 
  (total_cartons : ℕ) 
  (damaged_cartons : ℕ) 
  (accepted_cartons : ℕ) 
  (customers : ℕ) 
  (h1 : total_cartons = 400)
  (h2 : damaged_cartons = 60)
  (h3 : accepted_cartons = 160)
  (h_eq_per_customer : (total_cartons / customers) - damaged_cartons = accepted_cartons / customers) :
  customers = 4 :=
sorry

end number_of_customers_l103_103117


namespace january_first_is_tuesday_l103_103329

-- Define the days of the week for convenience
inductive Weekday
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define the problem conditions
def daysInJanuary : Nat := 31
def weeksInJanuary : Nat := daysInJanuary / 7   -- This is 4 weeks
def extraDays : Nat := daysInJanuary % 7         -- This leaves 3 extra days

-- Define the problem as proving January 1st is a Tuesday
theorem january_first_is_tuesday (fridaysInJanuary : Nat) (mondaysInJanuary : Nat)
    (h_friday : fridaysInJanuary = 4) (h_monday: mondaysInJanuary = 4) : Weekday :=
  -- Avoid specific proof steps from the solution; assume conditions and directly prove the result
  sorry

end january_first_is_tuesday_l103_103329


namespace product_of_five_consecutive_integers_not_square_l103_103302

theorem product_of_five_consecutive_integers_not_square (a : ℕ) :
  ¬ ∃ b c d e : ℕ, b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ e = a + 4 ∧ ∃ k : ℕ, (a * b * c * d * e) = k^2 :=
by
  sorry

end product_of_five_consecutive_integers_not_square_l103_103302


namespace factorization_of_expression_l103_103788

-- Define variables
variables {a x y : ℝ}

-- State the problem
theorem factorization_of_expression : a * x^2 - 4 * a * y^2 = a * (x + 2 * y) * (x - 2 * y) :=
  sorry

end factorization_of_expression_l103_103788


namespace sin_pi_over_6_plus_α_cos_pi_over_3_plus_2α_l103_103079

variable (α : ℝ)

-- Given conditions
def α_condition (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = 3 / 5) : Prop := 
  true

-- Prove the first part: sin(π / 6 + α) = (3 + 4 * real.sqrt 3) / 10
theorem sin_pi_over_6_plus_α (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = 3 / 5) :
  Real.sin (π / 6 + α) = (3 + 4 * Real.sqrt 3) / 10 :=
by
  sorry

-- Prove the second part: cos(π / 3 + 2 * α) = -(7 + 24 * real.sqrt 3) / 50
theorem cos_pi_over_3_plus_2α (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = 3 / 5) :
  Real.cos (π / 3 + 2 * α) = -(7 + 24 * Real.sqrt 3) / 50 :=
by
  sorry

end sin_pi_over_6_plus_α_cos_pi_over_3_plus_2α_l103_103079


namespace simplify_expr1_simplify_expr2_l103_103140

theorem simplify_expr1 (a b : ℤ) : 2 * a - (4 * a + 5 * b) + 2 * (3 * a - 4 * b) = 4 * a - 13 * b :=
by sorry

theorem simplify_expr2 (x y : ℤ) : 5 * x^2 - 2 * (3 * y^2 - 5 * x^2) + (-4 * y^2 + 7 * x * y) = 15 * x^2 - 10 * y^2 + 7 * x * y :=
by sorry

end simplify_expr1_simplify_expr2_l103_103140


namespace three_digit_number_uniq_l103_103196

theorem three_digit_number_uniq (n : ℕ) (h : 100 ≤ n ∧ n < 1000)
  (hundreds_digit : n / 100 = 5) (units_digit : n % 10 = 3)
  (div_by_9 : n % 9 = 0) : n = 513 :=
sorry

end three_digit_number_uniq_l103_103196


namespace num_toys_purchased_min_selling_price_l103_103135

variable (x m : ℕ)

-- Given conditions
axiom cond1 : 1500 / x + 5 = 3500 / (2 * x)
axiom cond2 : 150 * m - 5000 >= 1150

-- Required proof
theorem num_toys_purchased : x = 50 :=
by
  sorry

theorem min_selling_price : m >= 41 :=
by
  sorry

end num_toys_purchased_min_selling_price_l103_103135


namespace sum_six_smallest_multiples_of_12_is_252_l103_103742

-- Define the six smallest positive distinct multiples of 12
def six_smallest_multiples_of_12 := [12, 24, 36, 48, 60, 72]

-- Define the sum problem
def sum_of_six_smallest_multiples_of_12 : Nat :=
  six_smallest_multiples_of_12.foldr (· + ·) 0

-- Main proof statement
theorem sum_six_smallest_multiples_of_12_is_252 :
  sum_of_six_smallest_multiples_of_12 = 252 :=
by
  sorry

end sum_six_smallest_multiples_of_12_is_252_l103_103742


namespace children_eating_porridge_today_l103_103830

theorem children_eating_porridge_today
  (eat_every_day : ℕ)
  (eat_every_other_day : ℕ)
  (ate_yesterday : ℕ) :
  eat_every_day = 5 →
  eat_every_other_day = 7 →
  ate_yesterday = 9 →
  (eat_every_day + (eat_every_other_day - (ate_yesterday - eat_every_day)) = 8) :=
by
  intros h1 h2 h3
  sorry

end children_eating_porridge_today_l103_103830


namespace paul_initial_crayons_l103_103719

-- Define the variables for the crayons given away, lost, and left
def crayons_given_away : ℕ := 563
def crayons_lost : ℕ := 558
def crayons_left : ℕ := 332

-- Define the total number of crayons Paul got for his birthday
def initial_crayons : ℕ := 1453

-- The proof statement
theorem paul_initial_crayons :
  initial_crayons = crayons_given_away + crayons_lost + crayons_left :=
sorry

end paul_initial_crayons_l103_103719


namespace peter_change_left_l103_103590

theorem peter_change_left
  (cost_small : ℕ := 3)
  (cost_large : ℕ := 5)
  (total_money : ℕ := 50)
  (num_small : ℕ := 8)
  (num_large : ℕ := 5) :
  total_money - (num_small * cost_small + num_large * cost_large) = 1 :=
by
  sorry

end peter_change_left_l103_103590


namespace man_upstream_rate_l103_103636

theorem man_upstream_rate (rate_downstream : ℝ) (rate_still_water : ℝ) (rate_current : ℝ) 
    (h1 : rate_downstream = 32) (h2 : rate_still_water = 24.5) (h3 : rate_current = 7.5) : 
    rate_still_water - rate_current = 17 := 
by 
  sorry

end man_upstream_rate_l103_103636


namespace calc_3_op_2_op_4_op_1_l103_103650

def op (a b : ℕ) : ℕ :=
match a, b with
| 1, 1 => 2 | 1, 2 => 3 | 1, 3 => 4 | 1, 4 => 1
| 2, 1 => 3 | 2, 2 => 1 | 2, 3 => 2 | 2, 4 => 4
| 3, 1 => 4 | 3, 2 => 2 | 3, 3 => 1 | 3, 4 => 3
| 4, 1 => 1 | 4, 2 => 4 | 4, 3 => 3 | 4, 4 => 2
| _, _  => 0 -- default case, though won't be used

theorem calc_3_op_2_op_4_op_1 : op (op 3 2) (op 4 1) = 3 :=
by
  sorry

end calc_3_op_2_op_4_op_1_l103_103650


namespace total_food_consumed_l103_103606

theorem total_food_consumed (n1 n2 f1 f2 : ℕ) (h1 : n1 = 4000) (h2 : n2 = n1 - 500) (h3 : f1 = 10) (h4 : f2 = f1 - 2) : 
    n1 * f1 + n2 * f2 = 68000 := by 
  sorry

end total_food_consumed_l103_103606


namespace ice_cream_scoops_l103_103766

theorem ice_cream_scoops (total_money : ℝ) (spent_on_restaurant : ℝ) (remaining_money : ℝ) 
  (cost_per_scoop_after_discount : ℝ) (remaining_each : ℝ) 
  (initial_savings : ℝ) (service_charge_percent : ℝ) (restaurant_percent : ℝ) 
  (ice_cream_discount_percent : ℝ) (money_each : ℝ) :
  total_money = 400 ∧
  spent_on_restaurant = 320 ∧
  remaining_money = 80 ∧
  cost_per_scoop_after_discount = 5 ∧
  remaining_each = 8 ∧
  initial_savings = 200 ∧
  service_charge_percent = 0.20 ∧
  restaurant_percent = 0.80 ∧
  ice_cream_discount_percent = 0.10 ∧
  money_each = 5 → 
  ∃ (scoops_per_person : ℕ), scoops_per_person = 5 :=
by
  sorry

end ice_cream_scoops_l103_103766


namespace product_of_5_consecutive_numbers_not_square_l103_103298

-- Define what it means for a product to be a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- The main theorem stating the problem
theorem product_of_5_consecutive_numbers_not_square :
  ∀ (a : ℕ), 0 < a → ¬ is_perfect_square (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
by 
  sorry

end product_of_5_consecutive_numbers_not_square_l103_103298


namespace main_theorem_l103_103922

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

theorem main_theorem :
  (∀ x : ℝ, f (x + 5/2) + f x = 2) ∧
  (∀ x : ℝ, f (1 + 2*x) = f (1 - 2*x)) ∧
  (∀ x : ℝ, g (x + 2) = g (x - 2)) ∧
  (∀ x : ℝ, g (-x + 1) - 1 = -g (x + 1) + 1) ∧
  (∀ x : ℝ, 0 ≤ x → x ≤ 2 → f x + g x = 3^x + x^3) →
  f 2022 * g 2022 = 72 :=
sorry

end main_theorem_l103_103922


namespace division_into_rectangles_l103_103568

theorem division_into_rectangles (figure : Type) (valid_division : figure → Prop) : (∃ ways, ways = 8) :=
by {
  -- assume given conditions related to valid_division using "figure"
  sorry
}

end division_into_rectangles_l103_103568


namespace sanjay_homework_fraction_l103_103626

theorem sanjay_homework_fraction :
  let original := 1
  let done_on_monday := 3 / 5
  let remaining_after_monday := original - done_on_monday
  let done_on_tuesday := 1 / 3 * remaining_after_monday
  let remaining_after_tuesday := remaining_after_monday - done_on_tuesday
  remaining_after_tuesday = 4 / 15 :=
by
  -- original := 1
  -- done_on_monday := 3 / 5
  -- remaining_after_monday := 1 - 3 / 5
  -- done_on_tuesday := 1 / 3 * (1 - 3 / 5)
  -- remaining_after_tuesday := (1 - 3 / 5) - (1 / 3 * (1 - 3 / 5))
  sorry

end sanjay_homework_fraction_l103_103626


namespace amount_of_H2O_formed_l103_103659

-- Define the balanced chemical equation as a relation
def balanced_equation : Prop :=
  ∀ (naoh hcl nacl h2o : ℕ), 
    (naoh + hcl = nacl + h2o)

-- Define the reaction of 2 moles of NaOH and 2 moles of HCl
def reaction (naoh hcl : ℕ) : ℕ :=
  if (naoh = 2) ∧ (hcl = 2) then 2 else 0

theorem amount_of_H2O_formed :
  balanced_equation →
  reaction 2 2 = 2 :=
by
  sorry

end amount_of_H2O_formed_l103_103659


namespace area_of_circle_l103_103780

theorem area_of_circle (x y : ℝ) :
  (x^2 + y^2 - 8*x - 6*y = -9) → 
  (∃ (R : ℝ), (x - 4)^2 + (y - 3)^2 = R^2 ∧ π * R^2 = 16 * π) :=
by
  sorry

end area_of_circle_l103_103780


namespace square_area_example_l103_103478

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

noncomputable def square_area (x1 y1 x2 y2 : ℝ) : ℝ :=
  (distance x1 y1 x2 y2)^2

theorem square_area_example : square_area 1 3 5 6 = 25 :=
by
  sorry

end square_area_example_l103_103478


namespace max_lateral_surface_area_of_tetrahedron_l103_103256

open Real

theorem max_lateral_surface_area_of_tetrahedron :
  ∀ (PA PB PC : ℝ), (PA^2 + PB^2 + PC^2 = 36) → (PA * PB + PB * PC + PA * PC ≤ 36) →
  (1/2 * (PA * PB + PB * PC + PA * PC) ≤ 18) :=
by
  intro PA PB PC hsum hineq
  sorry

end max_lateral_surface_area_of_tetrahedron_l103_103256


namespace convert_volume_cubic_feet_to_cubic_yards_l103_103362

theorem convert_volume_cubic_feet_to_cubic_yards (V : ℤ) (V_ft³ : V = 216) : 
  V / 27 = 8 := 
by {
  sorry
}

end convert_volume_cubic_feet_to_cubic_yards_l103_103362


namespace operation_1_2010_l103_103785

def operation (m n : ℕ) : ℕ := sorry

axiom operation_initial : operation 1 1 = 2
axiom operation_step (m n : ℕ) : operation m (n + 1) = operation m n + 3

theorem operation_1_2010 : operation 1 2010 = 6029 := sorry

end operation_1_2010_l103_103785


namespace Rachel_drinks_correct_glasses_l103_103458

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

end Rachel_drinks_correct_glasses_l103_103458


namespace first_term_geometric_sequence_l103_103667

theorem first_term_geometric_sequence :
  ∀ (a b c : ℕ), 
    let r := 243 / 81 in 
    b = a * r ∧ 
    c = b * r ∧ 
    81 = c * r ∧ 
    243 = 81 * r → 
    a = 3 :=
by
  intros
  let r : ℕ := 243 / 81
  sorry

end first_term_geometric_sequence_l103_103667


namespace milk_for_18_cookies_l103_103616

def milk_needed_to_bake_cookies (cookies : ℕ) (milk_per_24_cookies : ℚ) (quarts_to_pints : ℚ) : ℚ :=
  (milk_per_24_cookies * quarts_to_pints) * (cookies / 24)

theorem milk_for_18_cookies :
  milk_needed_to_bake_cookies 18 4.5 2 = 6.75 :=
by
  sorry

end milk_for_18_cookies_l103_103616


namespace Angela_insect_count_l103_103646

variables (Angela Jacob Dean : ℕ)
-- Conditions
def condition1 : Prop := Angela = Jacob / 2
def condition2 : Prop := Jacob = 5 * Dean
def condition3 : Prop := Dean = 30

-- Theorem statement proving Angela's insect count
theorem Angela_insect_count (h1 : condition1 Angela Jacob) (h2 : condition2 Jacob Dean) (h3 : condition3 Dean) : Angela = 75 :=
by
  sorry

end Angela_insect_count_l103_103646


namespace first_place_team_wins_l103_103951

-- Define the conditions in Lean 4
variable (joe_won : ℕ := 1) (joe_draw : ℕ := 3) (fp_draw : ℕ := 2) (joe_points : ℕ := 3 * joe_won + joe_draw)
variable (fp_points : ℕ := joe_points + 2)

 -- Define the proof problem
theorem first_place_team_wins : 3 * (fp_points - fp_draw) / 3 = 2 := by
  sorry

end first_place_team_wins_l103_103951


namespace line_through_intersection_points_of_circles_l103_103728

theorem line_through_intersection_points_of_circles :
  (∀ x y : ℝ, x^2 + y^2 = 9 ∧ (x + 4)^2 + (y + 3)^2 = 8 → 4 * x + 3 * y + 13 = 0) :=
by
  sorry

end line_through_intersection_points_of_circles_l103_103728


namespace cookies_left_after_ted_leaves_l103_103243

theorem cookies_left_after_ted_leaves :
  let f : Nat := 2 -- trays per day
  let d : Nat := 6 -- days
  let e_f : Nat := 1 -- cookies eaten per day by Frank
  let t : Nat := 4 -- cookies eaten by Ted
  let c : Nat := 12 -- cookies per tray
  let total_cookies := f * c * d -- total cookies baked
  let cookies_eaten_by_frank := e_f * d -- total cookies eaten by Frank
  let cookies_before_ted := total_cookies - cookies_eaten_by_frank -- cookies before Ted
  let total_cookies_left := cookies_before_ted - t -- cookies left after Ted
  total_cookies_left = 134
:= by
  sorry

end cookies_left_after_ted_leaves_l103_103243


namespace initial_invited_people_l103_103343

theorem initial_invited_people (not_showed_up : ℕ) (table_capacity : ℕ) (tables_needed : ℕ) 
  (H1 : not_showed_up = 12) (H2 : table_capacity = 3) (H3 : tables_needed = 2) :
  not_showed_up + (table_capacity * tables_needed) = 18 :=
by
  sorry

end initial_invited_people_l103_103343


namespace angle_equiv_terminal_side_l103_103749

theorem angle_equiv_terminal_side (θ : ℤ) : 
  let θ_deg := (750 : ℕ)
  let reduced_angle := θ_deg % 360
  0 ≤ reduced_angle ∧ reduced_angle < 360 ∧ reduced_angle = 30:=
by
  sorry

end angle_equiv_terminal_side_l103_103749


namespace helium_balloon_height_l103_103818

theorem helium_balloon_height :
    let total_budget := 200
    let cost_sheet := 42
    let cost_rope := 18
    let cost_propane := 14
    let cost_per_ounce_helium := 1.5
    let height_per_ounce := 113
    let amount_spent := cost_sheet + cost_rope + cost_propane
    let money_left_for_helium := total_budget - amount_spent
    let ounces_helium := money_left_for_helium / cost_per_ounce_helium
    let total_height := height_per_ounce * ounces_helium
    total_height = 9492 := sorry

end helium_balloon_height_l103_103818


namespace rainfall_second_week_l103_103882

theorem rainfall_second_week (r1 r2 : ℝ) (h1 : r1 + r2 = 40) (h2 : r2 = 1.5 * r1) : r2 = 24 :=
by
  sorry

end rainfall_second_week_l103_103882


namespace prime_numbers_r_s_sum_l103_103603

theorem prime_numbers_r_s_sum (p q r s : ℕ) (hp : Fact (Nat.Prime p)) (hq : Fact (Nat.Prime q)) 
  (hr : Fact (Nat.Prime r)) (hs : Fact (Nat.Prime s)) (h1 : p < q) (h2 : q < r) (h3 : r < s) 
  (eqn : p * q * r * s + 1 = 4^(p + q)) : r + s = 274 :=
by
  sorry

end prime_numbers_r_s_sum_l103_103603


namespace election_percentage_l103_103431

-- Define the total number of votes (V), winner's votes, and the vote difference
def total_votes (V : ℕ) : Prop := V = 1944 + (1944 - 288)

-- Define the percentage calculation from the problem
def percentage_of_votes (votes_received total_votes : ℕ) : ℕ := (votes_received * 100) / total_votes

-- State the core theorem to prove the winner received 54 percent of the total votes
theorem election_percentage (V : ℕ) (h : total_votes V) : percentage_of_votes 1944 V = 54 := by
  sorry

end election_percentage_l103_103431


namespace adam_earnings_correct_l103_103040

def total_earnings (lawns_mowed lawns_to_mow : ℕ) (lawn_pay : ℕ)
                   (cars_washed cars_to_wash : ℕ) (car_pay_euros : ℕ) (euro_to_dollar : ℝ)
                   (dogs_walked dogs_to_walk : ℕ) (dog_pay_pesos : ℕ) (peso_to_dollar : ℝ) : ℝ :=
  let lawn_earnings := lawns_mowed * lawn_pay
  let car_earnings := (cars_washed * car_pay_euros : ℝ) * euro_to_dollar
  let dog_earnings := (dogs_walked * dog_pay_pesos : ℝ) * peso_to_dollar
  lawn_earnings + car_earnings + dog_earnings

theorem adam_earnings_correct :
  total_earnings 4 12 9 4 6 10 1.1 3 4 50 0.05 = 87.5 :=
by
  sorry

end adam_earnings_correct_l103_103040


namespace gray_percentage_correct_l103_103371

-- Define the conditions
def total_squares := 25
def type_I_triangle_equivalent_squares := 8 * (1 / 2)
def type_II_triangle_equivalent_squares := 8 * (1 / 4)
def full_gray_squares := 4

-- Calculate the gray component
def gray_squares := type_I_triangle_equivalent_squares + type_II_triangle_equivalent_squares + full_gray_squares

-- Fraction representing the gray part of the quilt
def gray_fraction := gray_squares / total_squares

-- Translate fraction to percentage
def gray_percentage := gray_fraction * 100

theorem gray_percentage_correct : gray_percentage = 40 := by
  simp [total_squares, type_I_triangle_equivalent_squares, type_II_triangle_equivalent_squares, full_gray_squares, gray_squares, gray_fraction, gray_percentage]
  sorry -- You could expand this to a detailed proof if needed.

end gray_percentage_correct_l103_103371


namespace algebraic_expression_is_product_l103_103315

def algebraicExpressionMeaning (x : ℝ) : Prop :=
  -7 * x = -7 * x

theorem algebraic_expression_is_product (x : ℝ) :
  algebraicExpressionMeaning x :=
by
  sorry

end algebraic_expression_is_product_l103_103315


namespace percentage_of_gold_coins_is_35_percent_l103_103511

-- Definitions of conditions
def percentage_of_objects_that_are_beads : ℝ := 0.30
def percentage_of_coins_that_are_silver : ℝ := 0.25
def percentage_of_coins_that_are_gold : ℝ := 0.50

-- Problem Statement
theorem percentage_of_gold_coins_is_35_percent 
  (h_beads : percentage_of_objects_that_are_beads = 0.30) 
  (h_silver_coins : percentage_of_coins_that_are_silver = 0.25) 
  (h_gold_coins : percentage_of_coins_that_are_gold = 0.50) :
  0.35 = 0.35 := 
sorry

end percentage_of_gold_coins_is_35_percent_l103_103511


namespace cookies_left_l103_103241

theorem cookies_left (days_baking : ℕ) (trays_per_day : ℕ) (cookies_per_tray : ℕ) (frank_eats_per_day : ℕ) (ted_eats_on_sixth_day : ℕ) :
  trays_per_day * cookies_per_tray * days_baking - frank_eats_per_day * days_baking - ted_eats_on_sixth_day = 134 :=
by
  have days_baking := 6
  have trays_per_day := 2
  have cookies_per_tray := 12
  have frank_eats_per_day := 1
  have ted_eats_on_sixth_day := 4
  sorry

end cookies_left_l103_103241


namespace at_least_two_even_l103_103938

theorem at_least_two_even (x y z : ℤ) (u : ℤ)
  (h : x^2 + y^2 + z^2 = u^2) : (↑x % 2 = 0) ∨ (↑y % 2 = 0) → (↑x % 2 = 0) ∨ (↑z % 2 = 0) ∨ (↑y % 2 = 0) := 
by
  sorry

end at_least_two_even_l103_103938


namespace percent_difference_l103_103690

theorem percent_difference (a b : ℝ) : 
  a = 67.5 * 250 / 100 → 
  b = 52.3 * 180 / 100 → 
  (a - b) = 74.61 :=
by
  intros ha hb
  rw [ha, hb]
  -- omitted proof
  sorry

end percent_difference_l103_103690


namespace expand_polynomial_l103_103524

theorem expand_polynomial (t : ℝ) :
  (3 * t^3 - 4 * t^2 + 5 * t - 3) * (4 * t^2 - 2 * t + 1) = 12 * t^5 - 22 * t^4 + 31 * t^3 - 26 * t^2 + 11 * t - 3 := by
  sorry

end expand_polynomial_l103_103524


namespace find_principal_amount_l103_103311

theorem find_principal_amount
  (r : ℝ := 0.05)  -- Interest rate (5% per annum)
  (t : ℕ := 2)    -- Time period (2 years)
  (diff : ℝ := 20) -- Given difference between CI and SI
  (P : ℝ := 8000) -- Principal amount to prove
  : P * (1 + r) ^ t - P - P * r * t = diff :=
by
  sorry

end find_principal_amount_l103_103311


namespace volume_in_cubic_yards_l103_103360

theorem volume_in_cubic_yards (V : ℝ) (conversion_factor : ℝ) (hV : V = 216) (hcf : conversion_factor = 27) :
  V / conversion_factor = 8 := by
  sorry

end volume_in_cubic_yards_l103_103360


namespace x_is_half_l103_103424

theorem x_is_half (w x y : ℝ) 
  (h1 : 6 / w + 6 / x = 6 / y) 
  (h2 : w * x = y) 
  (h3 : (w + x) / 2 = 0.5) : x = 0.5 :=
sorry

end x_is_half_l103_103424


namespace jimmy_hostel_stay_days_l103_103050

-- Definitions based on the conditions
def nightly_hostel_charge : ℕ := 15
def nightly_cabin_charge_per_person : ℕ := 15
def total_lodging_expense : ℕ := 75
def days_in_cabin : ℕ := 2

-- The proof statement
theorem jimmy_hostel_stay_days : 
    ∃ x : ℕ, (nightly_hostel_charge * x + nightly_cabin_charge_per_person * days_in_cabin = total_lodging_expense) ∧ x = 3 := by
    sorry

end jimmy_hostel_stay_days_l103_103050


namespace find_a1_l103_103572

noncomputable def geometric_sequence := ℕ → ℝ

theorem find_a1 (a : geometric_sequence) (q : ℝ) (S5 : ℝ) 
  (h1 : q = -2) (h2 : S5 = 44)
  (h3 : ∀ n, a (n + 1) = a n * q) 
  (h4 : S5 = (1 - a 0 * q ^ 5) / (1 - q)) :
  a 0 = 4 :=
by
  sorry

end find_a1_l103_103572


namespace connie_needs_more_money_l103_103228

variable (cost_connie : ℕ) (cost_watch : ℕ)

theorem connie_needs_more_money 
  (h_connie : cost_connie = 39)
  (h_watch : cost_watch = 55) :
  cost_watch - cost_connie = 16 :=
by sorry

end connie_needs_more_money_l103_103228


namespace lowest_price_is_six_l103_103375

def single_package_cost : ℝ := 7
def eight_oz_package_cost : ℝ := 4
def four_oz_package_original_cost : ℝ := 2
def discount_rate : ℝ := 0.5

theorem lowest_price_is_six
  (cost_single : single_package_cost = 7)
  (cost_eight : eight_oz_package_cost = 4)
  (cost_four : four_oz_package_original_cost = 2)
  (discount : discount_rate = 0.5) :
  min single_package_cost (eight_oz_package_cost + 2 * (four_oz_package_original_cost * discount_rate)) = 6 := by
  sorry

end lowest_price_is_six_l103_103375


namespace picnic_men_count_l103_103199

variables 
  (M W A C : ℕ)
  (h1 : M + W + C = 200) 
  (h2 : M = W + 20)
  (h3 : A = C + 20)
  (h4 : A = M + W)

theorem picnic_men_count : M = 65 :=
by
  sorry

end picnic_men_count_l103_103199


namespace gcd_of_987654_and_123456_l103_103486

theorem gcd_of_987654_and_123456 : Nat.gcd 987654 123456 = 6 := by
  sorry

end gcd_of_987654_and_123456_l103_103486


namespace josiah_hans_age_ratio_l103_103952

theorem josiah_hans_age_ratio (H : ℕ) (J : ℕ) (hH : H = 15) (hSum : (J + 3) + (H + 3) = 66) : J / H = 3 :=
by
  sorry

end josiah_hans_age_ratio_l103_103952


namespace golden_state_total_points_l103_103570

theorem golden_state_total_points :
  ∀ (Draymond Curry Kelly Durant Klay : ℕ),
  Draymond = 12 →
  Curry = 2 * Draymond →
  Kelly = 9 →
  Durant = 2 * Kelly →
  Klay = Draymond / 2 →
  Draymond + Curry + Kelly + Durant + Klay = 69 :=
by
  intros Draymond Curry Kelly Durant Klay
  intros hD hC hK hD2 hK2
  rw [hD, hC, hK, hD2, hK2]
  sorry

end golden_state_total_points_l103_103570


namespace ten_percent_of_number_l103_103032

theorem ten_percent_of_number (x : ℝ)
  (h : x - (1 / 4) * 2 - (1 / 3) * 3 - (1 / 7) * x = 27) :
  0.10 * x = 3.325 :=
sorry

end ten_percent_of_number_l103_103032


namespace correct_inequality_l103_103071

-- Define the conditions
variables (a b : ℝ)
variable (h : a > 1 ∧ 1 > b ∧ b > 0)

-- State the theorem to prove
theorem correct_inequality (h : a > 1 ∧ 1 > b ∧ b > 0) : 
  (1 / Real.log a) > (1 / Real.log b) :=
sorry

end correct_inequality_l103_103071


namespace part1_part2_l103_103927

noncomputable def seq (n : ℕ) : ℚ :=
  match n with
  | 0     => 0  -- since there is no a_0 (we use ℕ*), we set it to 0
  | 1     => 1/3
  | n + 1 => seq n + (seq n) ^ 2 / (n : ℚ) ^ 2

theorem part1 (n : ℕ) (h : 0 < n) :
  seq n < seq (n + 1) ∧ seq (n + 1) < 1 :=
sorry

theorem part2 (n : ℕ) (h : 0 < n) :
  seq n > 1/2 - 1/(4 * n) :=
sorry

end part1_part2_l103_103927


namespace work_rate_together_l103_103757

theorem work_rate_together :
  let work_rate_A := (1 : ℚ) / 12
  let work_rate_B := (1 : ℚ) / 6
  let work_rate_C := (1 : ℚ) / 18
  work_rate_A + work_rate_B + work_rate_C = 11 / 36 := by
  let work_rate_A := (1 : ℚ) / 12
  let work_rate_B := (1 : ℚ) / 6
  let work_rate_C := (1 : ℚ) / 18
  calc
    work_rate_A + work_rate_B + work_rate_C = (1 : ℚ) / 12 + (1 : ℚ) / 6 + (1 : ℚ) / 18 : rfl
    ... = 3 / 36 + 6 / 36 + 2 / 36 : by
      congr; {field_simp [work_rate_A, work_rate_B, work_rate_C]}; apply_rat_cast_eq_field
    ... = 11 / 36 : by sorry

end work_rate_together_l103_103757


namespace remainder_of_x50_div_x_plus_1_cubed_l103_103236

theorem remainder_of_x50_div_x_plus_1_cubed (x : ℚ) : 
  (x ^ 50) % ((x + 1) ^ 3) = 1225 * x ^ 2 + 2450 * x + 1176 :=
by sorry

end remainder_of_x50_div_x_plus_1_cubed_l103_103236


namespace total_cost_of_pets_l103_103164

theorem total_cost_of_pets 
  (num_puppies num_kittens num_parakeets : ℕ)
  (cost_parakeet cost_puppy cost_kitten : ℕ)
  (h1 : num_puppies = 2)
  (h2 : num_kittens = 2)
  (h3 : num_parakeets = 3)
  (h4 : cost_parakeet = 10)
  (h5 : cost_puppy = 3 * cost_parakeet)
  (h6 : cost_kitten = 2 * cost_parakeet) : 
  num_puppies * cost_puppy + num_kittens * cost_kitten + num_parakeets * cost_parakeet = 130 :=
by
  sorry

end total_cost_of_pets_l103_103164


namespace final_answer_is_15_l103_103063

-- We will translate the conditions from the problem into definitions and then formulate the theorem

-- Define the product of 10 and 12
def product : ℕ := 10 * 12

-- Define the result of dividing this product by 2
def divided_result : ℕ := product / 2

-- Define one-fourth of the divided result
def one_fourth : ℚ := (1/4 : ℚ) * divided_result

-- The theorem statement that verifies the final answer
theorem final_answer_is_15 : one_fourth = 15 := by
  sorry

end final_answer_is_15_l103_103063


namespace units_digit_17_times_29_l103_103238

theorem units_digit_17_times_29 :
  (17 * 29) % 10 = 3 :=
by
  sorry

end units_digit_17_times_29_l103_103238


namespace min_value_proof_l103_103858

noncomputable def min_value (m n : ℝ) : ℝ := 
  if 4 * m + n = 1 ∧ (m > 0 ∧ n > 0) then (4 / m + 1 / n) else 0

theorem min_value_proof : ∃ m n : ℝ, 4 * m + n = 1 ∧ m > 0 ∧ n > 0 ∧ min_value m n = 25 :=
by
  -- stating the theorem conditionally 
  -- and expressing that there exists values of m and n
  sorry

end min_value_proof_l103_103858


namespace ratio_of_black_to_white_tiles_l103_103910

theorem ratio_of_black_to_white_tiles
  (original_width : ℕ)
  (original_height : ℕ)
  (original_black_tiles : ℕ)
  (original_white_tiles : ℕ)
  (border_width : ℕ)
  (border_height : ℕ)
  (extended_width : ℕ)
  (extended_height : ℕ)
  (new_white_tiles : ℕ)
  (total_white_tiles : ℕ)
  (total_black_tiles : ℕ)
  (ratio_black_to_white : ℚ)
  (h1 : original_width = 5)
  (h2 : original_height = 6)
  (h3 : original_black_tiles = 12)
  (h4 : original_white_tiles = 18)
  (h5 : border_width = 1)
  (h6 : border_height = 1)
  (h7 : extended_width = original_width + 2 * border_width)
  (h8 : extended_height = original_height + 2 * border_height)
  (h9 : new_white_tiles = (extended_width * extended_height) - (original_width * original_height))
  (h10 : total_white_tiles = original_white_tiles + new_white_tiles)
  (h11 : total_black_tiles = original_black_tiles)
  (h12 : ratio_black_to_white = total_black_tiles / total_white_tiles) :
  ratio_black_to_white = 3 / 11 := 
sorry

end ratio_of_black_to_white_tiles_l103_103910


namespace students_neither_football_nor_cricket_l103_103130

def total_students : ℕ := 450
def football_players : ℕ := 325
def cricket_players : ℕ := 175
def both_players : ℕ := 100

theorem students_neither_football_nor_cricket : 
  total_students - (football_players + cricket_players - both_players) = 50 := by
  sorry

end students_neither_football_nor_cricket_l103_103130


namespace max_mixed_gender_groups_l103_103159

theorem max_mixed_gender_groups (b g : ℕ) (h_b : b = 31) (h_g : g = 32) : 
  ∃ max_groups, max_groups = min (b / 2) (g / 3) :=
by
  use 10
  sorry

end max_mixed_gender_groups_l103_103159


namespace volume_in_cubic_yards_l103_103358

theorem volume_in_cubic_yards (V : ℝ) (conversion_factor : ℝ) (hV : V = 216) (hcf : conversion_factor = 27) :
  V / conversion_factor = 8 := by
  sorry

end volume_in_cubic_yards_l103_103358


namespace total_food_consumed_l103_103607

theorem total_food_consumed (n1 n2 f1 f2 : ℕ) (h1 : n1 = 4000) (h2 : n2 = n1 - 500) (h3 : f1 = 10) (h4 : f2 = f1 - 2) : 
    n1 * f1 + n2 * f2 = 68000 := by 
  sorry

end total_food_consumed_l103_103607


namespace friendly_snakes_not_blue_l103_103723

variable (Snakes : Type)
variable (sally_snakes : Finset Snakes)
variable (blue : Snakes → Prop)
variable (friendly : Snakes → Prop)
variable (can_swim : Snakes → Prop)
variable (can_climb : Snakes → Prop)

variable [DecidablePred blue] [DecidablePred friendly] [DecidablePred can_swim] [DecidablePred can_climb]

-- The number of snakes in Sally's collection
axiom h_snakes_count : sally_snakes.card = 20
-- There are 7 blue snakes
axiom h_blue : (sally_snakes.filter blue).card = 7
-- There are 10 friendly snakes
axiom h_friendly : (sally_snakes.filter friendly).card = 10
-- All friendly snakes can swim
axiom h1 : ∀ s ∈ sally_snakes, friendly s → can_swim s
-- No blue snakes can climb
axiom h2 : ∀ s ∈ sally_snakes, blue s → ¬ can_climb s
-- Snakes that can't climb also can't swim
axiom h3 : ∀ s ∈ sally_snakes, ¬ can_climb s → ¬ can_swim s

theorem friendly_snakes_not_blue :
  ∀ s ∈ sally_snakes, friendly s → ¬ blue s :=
by
  sorry

end friendly_snakes_not_blue_l103_103723


namespace not_perfect_square_l103_103863

theorem not_perfect_square (n : ℤ) : ¬ ∃ (m : ℤ), 4*n + 3 = m^2 := 
by 
  sorry

end not_perfect_square_l103_103863


namespace math_problem_l103_103965

theorem math_problem
  (x y : ℝ)
  (h1 : x + y = 5)
  (h2 : x * y = -3)
  : x + (x^3 / y^2) + (y^3 / x^2) + y = 590.5 :=
sorry

end math_problem_l103_103965


namespace red_peaches_l103_103323

theorem red_peaches (R G : ℕ) (h1 : G = 11) (h2 : G = R + 6) : R = 5 :=
by {
  sorry
}

end red_peaches_l103_103323


namespace sum_proper_divisors_243_l103_103015

theorem sum_proper_divisors_243 : (1 + 3 + 9 + 27 + 81) = 121 :=
by
  sorry

end sum_proper_divisors_243_l103_103015


namespace triangle_AB_eq_3_halves_CK_l103_103384

/-- Mathematically equivalent problem:
In an acute triangle ABC, rectangle ACGH is constructed with AC as one side, and CG : AC = 2:1.
A square BCEF is constructed with BC as one side. The height CD from A to B intersects GE at point K.
Prove that AB = 3/2 * CK. -/
theorem triangle_AB_eq_3_halves_CK
  (A B C H G E K : Type)
  (triangle_ABC_acute : ∀(A B C : Type), True) 
  (rectangle_ACGH : ∀(A C G H : Type), True) 
  (square_BCEF : ∀(B C E F : Type), True)
  (H_C_G_collinear : ∀(H C G : Type), True)
  (HCG_ratio : ∀ (AC CG : ℝ), CG / AC = 2 / 1)
  (BC_side : ∀ (BC : ℝ), BC = 1)
  (height_CD_intersection : ∀ (A B C D E G : Type), True)
  (intersection_point_K : ∀ (C D G E K : Type), True) :
  ∃ (AB CK : ℝ), AB = 3 / 2 * CK :=
by sorry

end triangle_AB_eq_3_halves_CK_l103_103384


namespace expression_evaluation_l103_103391

def a : ℚ := 8 / 9
def b : ℚ := 5 / 6
def c : ℚ := 2 / 3
def d : ℚ := -5 / 18
def lhs : ℚ := (a - b + c) / d
def rhs : ℚ := -13 / 5

theorem expression_evaluation : lhs = rhs := by
  sorry

end expression_evaluation_l103_103391


namespace completing_the_square_solution_correct_l103_103171

theorem completing_the_square_solution_correct (x : ℝ) :
  (x^2 + 8 * x + 9 = 0) ↔ ((x + 4)^2 = 7) :=
by
  sorry

end completing_the_square_solution_correct_l103_103171


namespace equal_12_mn_P_2n_Q_m_l103_103143

-- Define P and Q based on given conditions
def P (m : ℕ) : ℕ := 2 ^ m
def Q (n : ℕ) : ℕ := 3 ^ n

-- The theorem to prove
theorem equal_12_mn_P_2n_Q_m (m n : ℕ) : (12 ^ (m * n)) = (P m ^ (2 * n)) * (Q n ^ m) :=
by
  -- Proof goes here
  sorry

end equal_12_mn_P_2n_Q_m_l103_103143


namespace minimum_positive_period_minimum_value_l103_103474

noncomputable def f (x : Real) : Real :=
  Real.sin (x / 5) - Real.cos (x / 5)

theorem minimum_positive_period (T : Real) : (∀ x, f (x + T) = f x) ∧ T > 0 → T = 10 * Real.pi :=
  sorry

theorem minimum_value : ∃ x, f x = -Real.sqrt 2 :=
  sorry

end minimum_positive_period_minimum_value_l103_103474


namespace meiosis_fertilization_correct_l103_103490

theorem meiosis_fertilization_correct :
  (∀ (half_nuclear_sperm half_nuclear_egg mitochondrial_egg : Prop)
     (recognition_basis_clycoproteins : Prop)
     (fusion_basis_nuclei : Prop)
     (meiosis_eukaryotes : Prop)
     (random_fertilization : Prop),
    (half_nuclear_sperm ∧ half_nuclear_egg ∧ mitochondrial_egg ∧ recognition_basis_clycoproteins ∧ fusion_basis_nuclei ∧ meiosis_eukaryotes ∧ random_fertilization) →
    (D : Prop) ) := 
sorry

end meiosis_fertilization_correct_l103_103490


namespace elyse_passing_threshold_l103_103786

def total_questions : ℕ := 90
def programming_questions : ℕ := 20
def database_questions : ℕ := 35
def networking_questions : ℕ := 35
def programming_correct_rate : ℝ := 0.8
def database_correct_rate : ℝ := 0.5
def networking_correct_rate : ℝ := 0.7
def passing_percentage : ℝ := 0.65

theorem elyse_passing_threshold :
  let programming_correct := programming_correct_rate * programming_questions
  let database_correct := database_correct_rate * database_questions
  let networking_correct := networking_correct_rate * networking_questions
  let total_correct := programming_correct + database_correct + networking_correct
  let required_to_pass := passing_percentage * total_questions
  total_correct = required_to_pass → 0 = 0 :=
by
  intro _h
  sorry

end elyse_passing_threshold_l103_103786


namespace sum_of_numbers_l103_103731

theorem sum_of_numbers (a b c : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : b = 8)
  (h4 : (a + b + c) / 3 = a + 12) (h5 : (a + b + c) / 3 = c - 20) :
  a + b + c = 48 :=
sorry

end sum_of_numbers_l103_103731


namespace product_of_slopes_constant_l103_103919

noncomputable def ellipse (x y : ℝ) := x^2 / 8 + y^2 / 4 = 1

theorem product_of_slopes_constant (a b : ℝ) (h_a_gt_b : a > b) (h_a_b_pos : 0 < a ∧ 0 < b)
  (e : ℝ) (h_eccentricity : e = (Real.sqrt 2) / 2) (P : ℝ × ℝ) (h_point_on_ellipse : (P.1, P.2) = (2, Real.sqrt 2)) :
  (∃ C : ℝ → ℝ → Prop, C = ellipse) ∧ (∃ k : ℝ, -k * 1/2 = -1 / 2) := sorry

end product_of_slopes_constant_l103_103919


namespace part_a_part_b_l103_103996

open EuclideanGeometry Real

namespace CircleIntersection

-- Define a triangle ABC
variables {A B C D E : Point} {S_a S_b S_c : Circle}

-- Define the internal and external angle bisectors
axioms (internal_bisector_AD : IsAngleBisector A B C D) 
       (external_bisector_AE : IsAngleBisector A B C E)

-- Define circles with specified diameters
axioms (circle_S_a : CircleDiameter S_a D E)
       (circle_S_b : CircleDiameter S_b (..) (..)) -- similar to S_a but for other vertices
       (circle_S_c : CircleDiameter S_c (..) (..)) -- similar to S_a but for other vertices

-- Theorem part (a)
theorem part_a :
  ∃ (M N : Point), 
    M ∈ S_a ∧ M ∈ S_b ∧ M ∈ S_c ∧ 
    N ∈ S_a ∧ N ∈ S_b ∧ N ∈ S_c ∧ 
    LineThrough M N (Circumcenter A B C) := sorry

-- Theorem part (b)
theorem part_b :
  ∀ {M N : Point},
    (M ∈ S_a ∧ M ∈ S_b ∧ M ∈ S_c) →
    (N ∈ S_a ∧ N ∈ S_b ∧ N ∈ S_c) →
    ∃ (projectionsM projectionsN : TriangleProjections),
      Equilateral projectionsM ∧ Equilateral projectionsN := sorry

end CircleIntersection

end part_a_part_b_l103_103996


namespace original_weight_of_beef_l103_103991

theorem original_weight_of_beef (weight_after_processing : ℝ) (loss_percentage : ℝ) :
  loss_percentage = 0.5 → weight_after_processing = 750 → 
  (750 : ℝ) / (1 - 0.5) = 1500 :=
by
  intros h_loss_percent h_weight_after
  sorry

end original_weight_of_beef_l103_103991


namespace max_value_range_of_t_l103_103068

theorem max_value_range_of_t (t x : ℝ) (h : t ≤ x ∧ x ≤ t + 2) 
: ∃ y : ℝ, y = -x^2 + 6 * x - 7 ∧ y = -(t - 3)^2 + 2 ↔ t ≥ 3 := 
by {
    sorry
}

end max_value_range_of_t_l103_103068


namespace sandy_savings_last_year_l103_103119

theorem sandy_savings_last_year (S : ℝ) (P : ℝ) 
(h1 : P / 100 * S = x)
(h2 : 1.10 * S = y)
(h3 : 0.10 * y = 0.11 * S)
(h4 : 0.11 * S = 1.8333333333333331 * x) :
P = 6 := by
  -- proof goes here
  sorry

end sandy_savings_last_year_l103_103119


namespace ellipse_major_axis_value_l103_103808

theorem ellipse_major_axis_value (m : ℝ) (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ)
  (h1 : ∀ {x y : ℝ}, (x, y) = P → (x^2 / m) + (y^2 / 16) = 1)
  (h2 : dist P F1 = 3)
  (h3 : dist P F2 = 7)
  : m = 25 :=
sorry

end ellipse_major_axis_value_l103_103808


namespace trigonometric_identity_l103_103699

open Real

noncomputable def sin_alpha (x y : ℝ) : ℝ :=
  y / sqrt (x^2 + y^2)

noncomputable def tan_alpha (x y : ℝ) : ℝ :=
  y / x

theorem trigonometric_identity (x y : ℝ) (h_x : x = 3/5) (h_y : y = -4/5) :
  sin_alpha x y * tan_alpha x y = 16/15 :=
by {
  -- math proof to be provided here
  sorry
}

end trigonometric_identity_l103_103699


namespace from20To25_l103_103978

def canObtain25 (start : ℕ) : Prop :=
  ∃ (steps : ℕ → ℕ), steps 0 = start ∧ (∃ n, steps n = 25) ∧ 
  (∀ i, steps (i+1) = (steps i * 2) ∨ (steps (i+1) = steps i / 10))

theorem from20To25 : canObtain25 20 :=
sorry

end from20To25_l103_103978


namespace part_I_part_II_l103_103573

noncomputable def f (x : ℝ) (m : ℝ) := m - |x - 2|

theorem part_I (m : ℝ) : (∀ x, f (x + 1) m >= 0 → 0 <= x ∧ x <= 2) ↔ m = 1 := by
  sorry

theorem part_II (a b c : ℝ) (m : ℝ) : (1 / a + 1 / (2 * b) + 1 / (3 * c) = m) → (m = 1) → (a + 2 * b + 3 * c >= 9) := by
  sorry

end part_I_part_II_l103_103573


namespace rational_relation_l103_103412

variable {a b : ℚ}

theorem rational_relation (h1 : a > 0) (h2 : b < 0) (h3 : |a| > |b|) : -a < -b ∧ -b < b ∧ b < a :=
by
  sorry

end rational_relation_l103_103412


namespace domain_of_function_l103_103398

theorem domain_of_function :
  {x : ℝ | -3 < x ∧ x < 2 ∧ x ≠ 1} = {x : ℝ | (2 - x > 0) ∧ (12 + x - x^2 ≥ 0) ∧ (x ≠ 1)} :=
by
  sorry

end domain_of_function_l103_103398


namespace probability_of_cocaptains_l103_103322

-- Define the conditions for the problem
def numberOfStudents (team : ℕ) : ℕ :=
  if team = 1 then 4 else
  if team = 2 then 6 else
  if team = 3 then 7 else
  if team = 4 then 9 else 0

def prob_cocaptains (team: ℕ) : ℚ :=
  let n := numberOfStudents team
  6 / (n * (n - 1) * (n - 2))

-- Main statement to be proved
theorem probability_of_cocaptains : 
  (1/4 : ℚ) * (prob_cocaptains 1 + prob_cocaptains 2 + prob_cocaptains 3 + prob_cocaptains 4) = 13 / 120 :=
by 
  sorry

end probability_of_cocaptains_l103_103322


namespace not_necessarily_divisor_of_44_l103_103465

theorem not_necessarily_divisor_of_44 {k : ℤ} (h1 : ∃ k, n = k * (k + 1) * (k + 2)) (h2 : 11 ∣ n) :
  ¬(44 ∣ n) :=
sorry

end not_necessarily_divisor_of_44_l103_103465


namespace completing_the_square_solution_l103_103180

theorem completing_the_square_solution : ∀ x : ℝ, x^2 + 8 * x + 9 = 0 ↔ (x + 4)^2 = 7 := by
  sorry

end completing_the_square_solution_l103_103180


namespace Razorback_total_revenue_l103_103855

def t_shirt_price : ℕ := 51
def t_shirt_discount : ℕ := 8
def hat_price : ℕ := 28
def hat_discount : ℕ := 5
def t_shirts_sold : ℕ := 130
def hats_sold : ℕ := 85

def discounted_t_shirt_price : ℕ := t_shirt_price - t_shirt_discount
def discounted_hat_price : ℕ := hat_price - hat_discount

def revenue_from_t_shirts : ℕ := t_shirts_sold * discounted_t_shirt_price
def revenue_from_hats : ℕ := hats_sold * discounted_hat_price

def total_revenue : ℕ := revenue_from_t_shirts + revenue_from_hats

theorem Razorback_total_revenue : total_revenue = 7545 := by
  unfold total_revenue
  unfold revenue_from_t_shirts
  unfold revenue_from_hats
  unfold discounted_t_shirt_price
  unfold discounted_hat_price
  unfold t_shirts_sold
  unfold hats_sold
  unfold t_shirt_price
  unfold t_shirt_discount
  unfold hat_price
  unfold hat_discount
  sorry

end Razorback_total_revenue_l103_103855


namespace absolute_difference_equation_l103_103467

theorem absolute_difference_equation :
  ∃ x : ℝ, (|16 - x| - |x - 12| = 4) ∧ x = 12 :=
by
  sorry

end absolute_difference_equation_l103_103467


namespace translate_sin_eq_cos_l103_103312

theorem translate_sin_eq_cos (φ : ℝ) (hφ : 0 ≤ φ ∧ φ < 2 * Real.pi) :
  (∀ x, Real.cos (x - Real.pi / 6) = Real.sin (x + φ)) → φ = Real.pi / 3 :=
by
  sorry

end translate_sin_eq_cos_l103_103312


namespace minimum_value_f_l103_103239

noncomputable def f (a b c : ℝ) : ℝ :=
  a / (Real.sqrt (a^2 + 8*b*c)) + b / (Real.sqrt (b^2 + 8*a*c)) + c / (Real.sqrt (c^2 + 8*a*b))

theorem minimum_value_f (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  1 ≤ f a b c := by
  sorry

end minimum_value_f_l103_103239


namespace subsequent_flights_requirements_l103_103338

-- Define the initial conditions
def late_flights : ℕ := 1
def on_time_flights : ℕ := 3
def total_initial_flights : ℕ := late_flights + on_time_flights

-- Define the number of subsequent flights needed
def subsequent_flights_needed (x : ℕ) : Prop :=
  let total_flights := total_initial_flights + x
  let on_time_total := on_time_flights + x
  (on_time_total : ℚ) / (total_flights : ℚ) > 0.40

-- State the theorem to prove
theorem subsequent_flights_requirements:
  ∃ x : ℕ, subsequent_flights_needed x := sorry

end subsequent_flights_requirements_l103_103338


namespace three_digit_number_satisfies_conditions_l103_103791

-- Definitions for the digits of the number
def x := 9
def y := 6
def z := 4

-- Define the three-digit number
def number := 100 * x + 10 * y + z

-- Define the conditions
def geometric_progression := y * y = x * z

def reverse_order_condition := (number - 495) = 100 * z + 10 * y + x

def arithmetic_progression := (z - 1) + (x - 2) = 2 * (y - 1)

-- The theorem to prove
theorem three_digit_number_satisfies_conditions :
  geometric_progression ∧ reverse_order_condition ∧ arithmetic_progression :=
by {
  sorry
}

end three_digit_number_satisfies_conditions_l103_103791


namespace line_to_slope_intercept_l103_103635

noncomputable def line_equation (v p q : ℝ × ℝ) : Prop :=
  (v.1 * (p.1 - q.1) + v.2 * (p.2 - q.2)) = 0

theorem line_to_slope_intercept (x y m b : ℝ) :
  line_equation (3, -4) (x, y) (2, 8) → (m, b) = (3 / 4, 6.5) :=
  by
    sorry

end line_to_slope_intercept_l103_103635


namespace chloe_first_round_points_l103_103227

variable (P : ℤ)
variable (totalPoints : ℤ := 86)
variable (secondRoundPoints : ℤ := 50)
variable (lastRoundLoss : ℤ := 4)

theorem chloe_first_round_points 
  (h : P + secondRoundPoints - lastRoundLoss = totalPoints) : 
  P = 40 := by
  sorry

end chloe_first_round_points_l103_103227


namespace fewer_buses_than_cars_l103_103865

theorem fewer_buses_than_cars
  (bus_to_car_ratio : ℕ := 1)
  (cars_on_river_road : ℕ := 65)
  (cars_per_bus : ℕ := 13) :
  cars_on_river_road - (cars_on_river_road / cars_per_bus) = 60 :=
by
  sorry

end fewer_buses_than_cars_l103_103865


namespace average_annual_growth_rate_l103_103716

theorem average_annual_growth_rate (x : ℝ) (h1 : 6.4 * (1 + x)^2 = 8.1) : x = 0.125 :=
by
  -- proof goes here
  sorry

end average_annual_growth_rate_l103_103716


namespace maximum_n_l103_103980

def number_of_trapezoids (n : ℕ) : ℕ := n * (n - 3) * (n - 2) * (n - 1) / 24

theorem maximum_n (n : ℕ) (h : number_of_trapezoids n ≤ 2012) : n ≤ 26 :=
by
  sorry

end maximum_n_l103_103980


namespace days_to_fulfill_order_l103_103010

theorem days_to_fulfill_order (bags_per_batch : ℕ) (total_order : ℕ) (initial_bags : ℕ) (required_days : ℕ) :
  bags_per_batch = 10 →
  total_order = 60 →
  initial_bags = 20 →
  required_days = (total_order - initial_bags) / bags_per_batch →
  required_days = 4 :=
by
  intros
  sorry

end days_to_fulfill_order_l103_103010


namespace quadratic_equation_only_option_B_l103_103488

theorem quadratic_equation_only_option_B (a b c : ℝ) (x : ℝ):
  (a ≠ 0 → (a * x^2 + b * x + c = 0)) ∧              -- Option A
  (3 * (x + 1)^2 = 2 * (x - 2) ↔ 3 * x^2 + 4 * x + 7 = 0) ∧  -- Option B
  (1 / x^2 + 1 = x^2 + 1 → False) ∧         -- Option C
  (1 / x^2 + 1 / x - 2 = 0 → False) →       -- Option D
  -- Option B is the only quadratic equation.
  (3 * (x + 1)^2 = 2 * (x - 2)) :=
sorry

end quadratic_equation_only_option_B_l103_103488


namespace graph_of_equation_l103_103745

theorem graph_of_equation (x y : ℝ) : 
  (x - y)^2 = x^2 + y^2 ↔ (x = 0 ∨ y = 0) := 
by 
  sorry

end graph_of_equation_l103_103745


namespace twenty_five_percent_of_five_hundred_is_one_twenty_five_l103_103992

theorem twenty_five_percent_of_five_hundred_is_one_twenty_five :
  let percent := 0.25
  let amount := 500
  percent * amount = 125 :=
by
  sorry

end twenty_five_percent_of_five_hundred_is_one_twenty_five_l103_103992


namespace part1_even_function_part2_min_value_l103_103804

variable {a x : ℝ}

def f (x a : ℝ) : ℝ := x^2 + |x - a| + 1

theorem part1_even_function (h : a = 0) : 
  ∀ x : ℝ, f x 0 = f (-x) 0 :=
by
  -- This statement needs to be proved to show that f(x) is even when a = 0
  sorry

theorem part2_min_value (h : true) : 
  (a > (1/2) → ∃ x : ℝ, f x a = a + (3/4)) ∧
  (a ≤ -(1/2) → ∃ x : ℝ, f x a = -a + (3/4)) ∧
  ((- (1/2) < a ∧ a ≤ (1/2)) → ∃ x : ℝ, f x a = a^2 + 1) :=
by
  -- This statement needs to be proved to show the different minimum values of the function
  sorry

end part1_even_function_part2_min_value_l103_103804


namespace inequality_ge_five_halves_l103_103958

open Real

noncomputable def xy_yz_zx_eq_one (x y z : ℝ) := x * y + y * z + z * x = 1
noncomputable def non_neg (x y z : ℝ) := x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0

theorem inequality_ge_five_halves (x y z : ℝ) (h1 : xy_yz_zx_eq_one x y z) (h2 : non_neg x y z) :
  1 / (x + y) + 1 / (y + z) + 1 / (z + x) ≥ 5 / 2 := 
sorry

end inequality_ge_five_halves_l103_103958


namespace wine_problem_solution_l103_103886

theorem wine_problem_solution (x : ℝ) (h1 : 0 ≤ x ∧ x ≤ 200) (h2 : (200 - x) * (180 - x) / 200 = 144) : x = 20 := 
by
  sorry

end wine_problem_solution_l103_103886


namespace simplify_trigonometric_expression_l103_103854

noncomputable def trigonometric_simplification : Real :=
  (cos(5 * Real.pi / 180)^2 - sin(5 * Real.pi / 180)^2) / (sin(40 * Real.pi / 180) * cos(40 * Real.pi / 180))

theorem simplify_trigonometric_expression : trigonometric_simplification = 2 := by
  sorry

end simplify_trigonometric_expression_l103_103854


namespace next_month_has_5_Wednesdays_l103_103562

-- The current month characteristics
def current_month_has_5_Saturdays : Prop := ∃ month : ℕ, month = 30 ∧ ∃ day : ℕ, day = 5
def current_month_has_5_Sundays : Prop := ∃ month : ℕ, month = 30 ∧ ∃ day : ℕ, day = 5
def current_month_has_4_Mondays : Prop := ∃ month : ℕ, month = 30 ∧ ∃ day : ℕ, day = 4
def current_month_has_4_Fridays : Prop := ∃ month : ℕ, month = 30 ∧ ∃ day : ℕ, day = 4
def month_ends_on_Sunday : Prop := ∃ day : ℕ, day = 30 ∧ day % 7 = 0

-- Prove next month has 5 Wednesdays
theorem next_month_has_5_Wednesdays 
  (h1 : current_month_has_5_Saturdays) 
  (h2 : current_month_has_5_Sundays)
  (h3 : current_month_has_4_Mondays)
  (h4 : current_month_has_4_Fridays)
  (h5 : month_ends_on_Sunday) :
  ∃ month : ℕ, month = 31 ∧ ∃ day : ℕ, day = 5 := 
sorry

end next_month_has_5_Wednesdays_l103_103562


namespace tan_diff_angle_neg7_l103_103693

-- Define the main constants based on the conditions given
variables (α : ℝ)
axiom sin_alpha : Real.sin α = -3/5
axiom alpha_in_fourth_quadrant : 0 < α ∧ α < 2 * Real.pi ∧ α > 3 * Real.pi / 2

-- Define the statement that needs to be proven based on the question and the correct answer
theorem tan_diff_angle_neg7 : 
  Real.tan (α - Real.pi / 4) = -7 :=
sorry

end tan_diff_angle_neg7_l103_103693


namespace interchangeable_statements_l103_103267

-- Modeled conditions and relationships
def perpendicular (l p: Type) : Prop := sorry -- Definition of perpendicularity between a line and a plane
def parallel (a b: Type) : Prop := sorry -- Definition of parallelism between two objects (lines or planes)

-- Original Statements
def statement_1 := ∀ (l₁ l₂ p: Type), (perpendicular l₁ p) ∧ (perpendicular l₂ p) → parallel l₁ l₂
def statement_2 := ∀ (p₁ p₂ p: Type), (perpendicular p₁ p) ∧ (perpendicular p₂ p) → parallel p₁ p₂
def statement_3 := ∀ (l₁ l₂ l: Type), (parallel l₁ l) ∧ (parallel l₂ l) → parallel l₁ l₂
def statement_4 := ∀ (l₁ l₂ p: Type), (parallel l₁ p) ∧ (parallel l₂ p) → parallel l₁ l₂

-- Swapped Statements
def swapped_1 := ∀ (p₁ p₂ l: Type), (perpendicular p₁ l) ∧ (perpendicular p₂ l) → parallel p₁ p₂
def swapped_2 := ∀ (l₁ l₂ l: Type), (perpendicular l₁ l) ∧ (perpendicular l₂ l) → parallel l₁ l₂
def swapped_3 := ∀ (p₁ p₂ p: Type), (parallel p₁ p) ∧ (parallel p₂ p) → parallel p₁ p₂
def swapped_4 := ∀ (p₁ p₂ l: Type), (parallel p₁ l) ∧ (parallel p₂ l) → parallel p₁ p₂

-- Proof Problem: Verify which statements are interchangeable
theorem interchangeable_statements :
  (statement_1 ↔ swapped_1) ∧
  (statement_2 ↔ swapped_2) ∧
  (statement_3 ↔ swapped_3) ∧
  (statement_4 ↔ swapped_4) :=
sorry

end interchangeable_statements_l103_103267


namespace factorize_poly_l103_103023

open Polynomial

theorem factorize_poly : 
  (X ^ 15 + X ^ 7 + 1 : Polynomial ℤ) =
    (X^2 + X + 1) * (X^13 - X^12 + X^10 - X^9 + X^7 - X^6 + X^4 - X^3 + X - 1) := 
  by
  sorry

end factorize_poly_l103_103023


namespace expected_value_uniform_l103_103528

open Real MeasureTheory

noncomputable def uniform_pdf (α β : ℝ) (x : ℝ) : ℝ :=
  if (α ≤ x ∧ x ≤ β) then 1 / (β - α) else 0

theorem expected_value_uniform {α β : ℝ} (hαβ : α < β) :
  ∫ x in α..β, x * (uniform_pdf α β x) = (β + α) / 2 := by
  sorry

end expected_value_uniform_l103_103528


namespace solve_system_of_equations_l103_103725

theorem solve_system_of_equations (x y : Real) : 
  (3 * x^2 + 3 * y^2 - 2 * x^2 * y^2 = 3) ∧ 
  (x^4 + y^4 + (2/3) * x^2 * y^2 = 17) ↔
  ( (x = Real.sqrt 2 ∧ (y = Real.sqrt 3 ∨ y = -Real.sqrt 3 )) ∨ 
    (x = -Real.sqrt 2 ∧ (y = Real.sqrt 3 ∨ y = -Real.sqrt 3)) ∨ 
    (x = Real.sqrt 3 ∧ (y = Real.sqrt 2 ∨ y = -Real.sqrt 2 )) ∨ 
    (x = -Real.sqrt 3 ∧ (y = Real.sqrt 2 ∨ y = -Real.sqrt 2 )) ) := 
  by
    sorry

end solve_system_of_equations_l103_103725


namespace Daisy_lunch_vs_breakfast_l103_103402

noncomputable def breakfast_cost : ℝ := 2.0 + 3.0 + 4.0 + 3.5
noncomputable def lunch_cost_before_service_charge : ℝ := 3.75 + 5.75 + 1.0
noncomputable def service_charge : ℝ := 0.10 * lunch_cost_before_service_charge
noncomputable def total_lunch_cost : ℝ := lunch_cost_before_service_charge + service_charge

theorem Daisy_lunch_vs_breakfast : total_lunch_cost - breakfast_cost = -0.95 := by
  sorry

end Daisy_lunch_vs_breakfast_l103_103402


namespace mildred_weight_l103_103449

theorem mildred_weight (carol_weight mildred_is_heavier : ℕ) (h1 : carol_weight = 9) (h2 : mildred_is_heavier = 50) :
  carol_weight + mildred_is_heavier = 59 :=
by
  sorry

end mildred_weight_l103_103449


namespace number_of_sides_sum_of_interior_angles_l103_103501

-- Condition: each exterior angle of the regular polygon is 18 degrees.
def exterior_angle (n : ℕ) : Prop :=
  360 / n = 18

-- Question 1: Determine the number of sides the polygon has.
theorem number_of_sides : ∃ n, n > 2 ∧ exterior_angle n :=
  sorry

-- Question 2: Calculate the sum of the interior angles.
theorem sum_of_interior_angles {n : ℕ} (h : 360 / n = 18) : 
  180 * (n - 2) = 3240 :=
  sorry

end number_of_sides_sum_of_interior_angles_l103_103501


namespace ana_salary_after_changes_l103_103769

-- Definitions based on conditions in part (a)
def initial_salary : ℝ := 2000
def raise_factor : ℝ := 1.20
def cut_factor : ℝ := 0.80

-- Statement of the proof problem
theorem ana_salary_after_changes : 
  (initial_salary * raise_factor * cut_factor) = 1920 :=
by
  sorry

end ana_salary_after_changes_l103_103769


namespace convert_volume_cubic_feet_to_cubic_yards_l103_103363

theorem convert_volume_cubic_feet_to_cubic_yards (V : ℤ) (V_ft³ : V = 216) : 
  V / 27 = 8 := 
by {
  sorry
}

end convert_volume_cubic_feet_to_cubic_yards_l103_103363


namespace factorize_mn_minus_mn_cubed_l103_103997

theorem factorize_mn_minus_mn_cubed (m n : ℝ) : 
  m * n - m * n ^ 3 = m * n * (1 + n) * (1 - n) :=
by {
  sorry
}

end factorize_mn_minus_mn_cubed_l103_103997


namespace calculate_10_odot_5_l103_103864

def odot (a b : ℚ) : ℚ := a + (4 * a) / (3 * b)

theorem calculate_10_odot_5 : odot 10 5 = 38 / 3 := by
  sorry

end calculate_10_odot_5_l103_103864


namespace arithmetic_seq_sin_identity_l103_103704

theorem arithmetic_seq_sin_identity:
  ∀ (a : ℕ → ℝ), (a 2 + a 6 = (3/2) * Real.pi) → (Real.sin (2 * a 4 - Real.pi / 3) = -1 / 2) :=
by
  sorry

end arithmetic_seq_sin_identity_l103_103704


namespace olympiad_permutations_l103_103432

theorem olympiad_permutations : 
  let total_permutations := Nat.factorial 9 / (Nat.factorial 2 * Nat.factorial 2) 
  let invalid_permutations := 5 * (Nat.factorial 4 / Nat.factorial 2)
  total_permutations - invalid_permutations = 90660 :=
by
  let total_permutations : ℕ := Nat.factorial 9 / (Nat.factorial 2 * Nat.factorial 2)
  let invalid_permutations : ℕ := 5 * (Nat.factorial 4 / Nat.factorial 2)
  show total_permutations - invalid_permutations = 90660
  sorry

end olympiad_permutations_l103_103432


namespace solution_set_l103_103320

-- Definitions representing the given conditions
def cond1 (x : ℝ) := x - 3 < 0
def cond2 (x : ℝ) := x + 1 ≥ 0

-- The problem: Prove the solution set is as given
theorem solution_set (x : ℝ) :
  (cond1 x) ∧ (cond2 x) ↔ -1 ≤ x ∧ x < 3 :=
by
  sorry

end solution_set_l103_103320


namespace second_occurrence_at_55_l103_103961

/-- On the highway, starting from 3 kilometers, there is a speed limit sign every 4 kilometers,
and starting from 10 kilometers, there is a speed monitoring device every 9 kilometers.
The first time both types of facilities are encountered simultaneously is at 19 kilometers.
The second time both types of facilities are encountered simultaneously is at 55 kilometers. -/
theorem second_occurrence_at_55 :
  ∀ (k : ℕ), (∃ n m : ℕ, 3 + 4 * n = k ∧ 10 + 9 * m = k ∧ 19 + 36 = k) := sorry

end second_occurrence_at_55_l103_103961


namespace find_number_l103_103637

variable (N : ℕ)

theorem find_number (h : 6 * ((N / 8) + 8 - 30) = 12) : N = 192 := 
by
  sorry

end find_number_l103_103637


namespace completing_the_square_l103_103167

theorem completing_the_square (x : ℝ) : x^2 + 8 * x + 9 = 0 → (x + 4)^2 = 7 :=
by 
  intro h
  sorry

end completing_the_square_l103_103167


namespace sum_b_equals_16_l103_103394

open BigOperators

/-- 
Given distinct integers b2, b3, b4, b5, b6, b7, b8, b9 such that 
(7 / 11) = (b2 / 2!) + (b3 / 3!) + (b4 / 4!) + (b5 / 5!) + 
           (b6 / 6!) + (b7 / 7!) + (b8 / 8!) + (b9 / 9!)
and 0 ≤ bi < i for i = 2, 3, ..., 9, 
prove that sum of these integers is 16.
-/
theorem sum_b_equals_16 (b2 b3 b4 b5 b6 b7 b8 b9 : ℕ) 
  (hb_distinct : list.nodup [b2, b3, b4, b5, b6, b7, b8, b9]) 
  (hb_range : (∀ i, i ∈ [2, 3, 4, 5, 6, 7, 8, 9] → 0 ≤ (nat.nat b i) ∧ (nat.nat b i) < i)) 
  (h_eq_frac : (7 / 11) = (b2 / 2) + (b3 / 6) + (b4 / 24) + (b5 / 120) + 
                          (b6 / 720) + (b7 / 5040) + (b8 / 40320) + (b9 / 362880)) :
  b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9 = 16 :=
sorry

end sum_b_equals_16_l103_103394


namespace breadth_of_water_tank_l103_103497

theorem breadth_of_water_tank (L H V : ℝ) (n : ℕ) (avg_displacement : ℝ) (total_displacement : ℝ)
  (h_len : L = 40)
  (h_height : H = 0.25)
  (h_avg_disp : avg_displacement = 4)
  (h_number : n = 50)
  (h_total_disp : total_displacement = avg_displacement * n)
  (h_displacement_value : total_displacement = 200) :
  (40 * B * 0.25 = 200) → B = 20 :=
by
  intro h_eq
  sorry

end breadth_of_water_tank_l103_103497


namespace difference_of_squares_l103_103698

theorem difference_of_squares (a b c : ℤ) (h₁ : a < b) (h₂ : b < c) (h₃ : a % 2 = 0) (h₄ : b % 2 = 0) (h₅ : c % 2 = 0) (h₆ : a + b + c = 1992) :
  c^2 - a^2 = 5312 :=
by
  sorry

end difference_of_squares_l103_103698


namespace fraction_pow_four_result_l103_103877

theorem fraction_pow_four_result (x : ℚ) (h : x = 1 / 4) : x ^ 4 = 390625 / 100000000 :=
by sorry

end fraction_pow_four_result_l103_103877


namespace determine_a_b_l103_103084

theorem determine_a_b (a b : ℝ) :
  (∀ x, y = x^2 + a * x + b) ∧ (∀ t, t = 0 → 3 * t - (t^2 + a * t + b) + 1 = 0) →
  a = 3 ∧ b = 1 :=
by
  sorry

end determine_a_b_l103_103084


namespace quadratic_solution_l103_103665

theorem quadratic_solution (x : ℝ) : -x^2 + 4 * x + 5 < 0 ↔ x > 5 ∨ x < -1 :=
sorry

end quadratic_solution_l103_103665


namespace sum_proper_divisors_243_l103_103017

theorem sum_proper_divisors_243 : 
  let proper_divisors_243 := [1, 3, 9, 27, 81] in
  proper_divisors_243.sum = 121 := 
by
  sorry

end sum_proper_divisors_243_l103_103017


namespace maci_red_pens_l103_103286

def cost_blue_pens (b : ℕ) (cost_blue : ℕ) : ℕ := b * cost_blue

def cost_red_pen (cost_blue : ℕ) : ℕ := 2 * cost_blue

def total_cost (cost_blue : ℕ) (n_blue : ℕ) (n_red : ℕ) : ℕ := 
  n_blue * cost_blue + n_red * (2 * cost_blue)

theorem maci_red_pens :
  ∀ (n_blue cost_blue n_red total : ℕ),
  n_blue = 10 →
  cost_blue = 10 →
  total = 400 →
  total_cost cost_blue n_blue n_red = total →
  n_red = 15 := 
by
  intros n_blue cost_blue n_red total h1 h2 h3 h4
  sorry

end maci_red_pens_l103_103286


namespace time_to_cross_platform_l103_103629

/-- Definitions of the conditions in the problem. -/
def train_length : ℕ := 1500
def platform_length : ℕ := 1800
def time_to_cross_tree : ℕ := 100
def train_speed : ℕ := train_length / time_to_cross_tree
def total_distance : ℕ := train_length + platform_length

/-- Proof statement: The time for the train to pass the platform. -/
theorem time_to_cross_platform : (total_distance / train_speed) = 220 := by
  sorry

end time_to_cross_platform_l103_103629


namespace robis_savings_in_january_l103_103962

theorem robis_savings_in_january (x : ℕ) (h: (x + (x + 4) + (x + 8) + (x + 12) + (x + 16) + (x + 20) = 126)) : x = 11 := 
by {
  -- By simplification, the lean equivalent proof would include combining like
  -- terms and solving the resulting equation. For now, we'll use sorry.
  sorry
}

end robis_savings_in_january_l103_103962


namespace completing_the_square_l103_103165

theorem completing_the_square (x : ℝ) : x^2 + 8 * x + 9 = 0 → (x + 4)^2 = 7 :=
by 
  intro h
  sorry

end completing_the_square_l103_103165


namespace rhombuses_in_grid_l103_103471

def number_of_rhombuses (n : ℕ) : ℕ :=
(n - 1) * n + (n - 1) * n

theorem rhombuses_in_grid :
  number_of_rhombuses 5 = 30 :=
by
  sorry

end rhombuses_in_grid_l103_103471


namespace pond_field_area_ratio_l103_103473

theorem pond_field_area_ratio (w l s A_field A_pond : ℕ) (h1 : l = 2 * w) (h2 : l = 96) (h3 : s = 8) (h4 : A_field = l * w) (h5 : A_pond = s * s) :
  A_pond.toFloat / A_field.toFloat = 1 / 72 := 
by
  sorry

end pond_field_area_ratio_l103_103473


namespace complete_the_square_l103_103178

theorem complete_the_square (x : ℝ) :
  (x^2 + 8*x + 9 = 0) ↔ ((x + 4)^2 = 7) :=
sorry

end complete_the_square_l103_103178


namespace jordan_rectangle_width_l103_103494

theorem jordan_rectangle_width
  (carol_length : ℕ) (carol_width : ℕ) (jordan_length : ℕ) (jordan_width : ℕ)
  (h_carol_dims : carol_length = 12) (h_carol_dims2 : carol_width = 15)
  (h_jordan_length : jordan_length = 6)
  (h_area_eq : carol_length * carol_width = jordan_length * jordan_width) :
  jordan_width = 30 := 
sorry

end jordan_rectangle_width_l103_103494


namespace number_of_possible_flags_l103_103229

def colors : List String := ["purple", "gold"]

noncomputable def num_choices_per_stripe (colors : List String) : Nat := 
  colors.length

theorem number_of_possible_flags :
  (num_choices_per_stripe colors) ^ 3 = 8 := 
by
  -- Proof
  sorry

end number_of_possible_flags_l103_103229


namespace johns_score_is_101_l103_103574

variable (c w s : ℕ)
variable (h1 : s = 40 + 5 * c - w)
variable (h2 : s > 100)
variable (h3 : c ≤ 40)
variable (h4 : ∀ s' > 100, s' < s → ∃ c' w', s' = 40 + 5 * c' - w')

theorem johns_score_is_101 : s = 101 := by
  sorry

end johns_score_is_101_l103_103574


namespace completing_the_square_l103_103169

theorem completing_the_square (x : ℝ) : x^2 + 8 * x + 9 = 0 → (x + 4)^2 = 7 :=
by 
  intro h
  sorry

end completing_the_square_l103_103169


namespace rectangle_original_area_l103_103870

theorem rectangle_original_area (L L' A : ℝ) 
  (h1: A = L * 10)
  (h2: L' * 10 = (4 / 3) * A)
  (h3: 2 * L' + 2 * 10 = 60) : A = 150 :=
by 
  sorry

end rectangle_original_area_l103_103870


namespace stickers_after_exchange_l103_103296

-- Given conditions
def Ryan_stickers : ℕ := 30
def Steven_stickers : ℕ := 3 * Ryan_stickers
def Terry_stickers : ℕ := Steven_stickers + 20
def Emily_stickers : ℕ := Steven_stickers / 2
def Jasmine_stickers : ℕ := Terry_stickers + Terry_stickers / 10

def total_stickers_before : ℕ := 
  Ryan_stickers + Steven_stickers + Terry_stickers + Emily_stickers + Jasmine_stickers

noncomputable def total_stickers_after : ℕ := 
  total_stickers_before - 2 * 5

-- The goal is to prove that the total stickers after the exchange event is 386
theorem stickers_after_exchange : total_stickers_after = 386 := 
  by sorry

end stickers_after_exchange_l103_103296


namespace number_of_ways_to_divide_l103_103566

-- Define the given shape
structure Shape :=
  (sides : Nat) -- Number of 3x1 stripes along the sides
  (centre : Nat) -- Size of the central square (3x3)

-- Define the specific problem shape
def problem_shape : Shape :=
  { sides := 4, centre := 9 } -- 3x1 stripes on all sides and a 3x3 centre

-- Theorem stating the number of ways to divide the shape into 1x3 rectangles
theorem number_of_ways_to_divide (s : Shape) (h1 : s.sides = 4) (h2 : s.centre = 9) : 
  ∃ ways, ways = 2 :=
by
  -- The proof is skipped
  sorry

end number_of_ways_to_divide_l103_103566


namespace quadratic_real_roots_l103_103077

theorem quadratic_real_roots (a b c : ℝ) (h : a * c < 0) : 
  ∃ x y : ℝ, a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ x ≠ y :=
by
  sorry

end quadratic_real_roots_l103_103077


namespace sequence_general_term_l103_103113

theorem sequence_general_term (n : ℕ) (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ k ≥ 1, a (k + 1) = 2 * a k) : a n = 2 ^ (n - 1) :=
sorry

end sequence_general_term_l103_103113


namespace product_of_five_consecutive_numbers_not_square_l103_103304

theorem product_of_five_consecutive_numbers_not_square (a b c d e : ℕ)
  (ha : a > 0) (hb : b = a + 1) (hc : c = b + 1) (hd : d = c + 1) (he : e = d + 1) :
  ¬ ∃ k : ℕ, a * b * c * d * e = k^2 := by
  sorry

end product_of_five_consecutive_numbers_not_square_l103_103304


namespace isosceles_triangle_base_function_l103_103076

theorem isosceles_triangle_base_function (x : ℝ) (hx : 5 < x ∧ x < 10) :
  ∃ y : ℝ, y = 20 - 2 * x := 
by
  sorry

end isosceles_triangle_base_function_l103_103076


namespace change_in_expression_l103_103923

theorem change_in_expression (x a : ℝ) (h : 0 < a) :
  (x + a)^3 - 3 * (x + a) - (x^3 - 3 * x) = 3 * a * x^2 + 3 * a^2 * x + a^3 - 3 * a
  ∨ (x - a)^3 - 3 * (x - a) - (x^3 - 3 * x) = -3 * a * x^2 + 3 * a^2 * x - a^3 + 3 * a :=
sorry

end change_in_expression_l103_103923


namespace find_car_costs_optimize_purchasing_plan_minimum_cost_l103_103198

theorem find_car_costs (x y : ℝ) (h1 : 3 * x + y = 85) (h2 : 2 * x + 4 * y = 140) :
    x = 20 ∧ y = 25 :=
by
  sorry

theorem optimize_purchasing_plan (m : ℕ) (h_total : m + (15 - m) = 15) (h_constraint : m ≤ 2 * (15 - m)) :
    m = 10 :=
by
  sorry

theorem minimum_cost (w : ℝ) (h_cost_expr : ∀ (m : ℕ), w = 20 * m + 25 * (15 - m)) (m := 10) :
    w = 325 :=
by
  sorry

end find_car_costs_optimize_purchasing_plan_minimum_cost_l103_103198


namespace percentage_music_students_l103_103981

variables (total_students : ℕ) (dance_students : ℕ) (art_students : ℕ)
  (music_students : ℕ) (music_percentage : ℚ)

def students_music : ℕ := total_students - (dance_students + art_students)
def percentage_students_music : ℚ := (students_music total_students dance_students art_students : ℚ) / (total_students : ℚ) * 100

theorem percentage_music_students (h1 : total_students = 400)
                                  (h2 : dance_students = 120)
                                  (h3 : art_students = 200) :
  percentage_students_music total_students dance_students art_students = 20 := by {
  sorry
}

end percentage_music_students_l103_103981


namespace intersect_empty_range_of_a_union_subsets_range_of_a_l103_103816

variable {x a : ℝ}

def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | (x - 6) * (x + 2) > 0}

theorem intersect_empty_range_of_a (h : A a ∩ B = ∅) : -2 ≤ a ∧ a ≤ 3 :=
by
  sorry

theorem union_subsets_range_of_a (h : A a ∪ B = B) : a < -5 ∨ a > 6 :=
by
  sorry

end intersect_empty_range_of_a_union_subsets_range_of_a_l103_103816


namespace min_value_expression_l103_103678

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  ∃ (m : ℝ), m = 3 / 2 ∧ ∀ t > 0, (2 * x / (x + 2 * y) + y / x) ≥ m :=
by
  use 3 / 2
  sorry

end min_value_expression_l103_103678


namespace min_contribution_l103_103825

theorem min_contribution (x : ℝ) (h1 : 0 < x) (h2 : 10 * x = 20) (h3 : ∀ p, p ≠ 1 → p ≠ 2 → p ≠ 3 → p ≠ 4 → p ≠ 5 → p ≠ 6 → p ≠ 7 → p ≠ 8 → p ≠ 9 → p ≠ 10 → p ≤ 11) : 
  x = 2 := sorry

end min_contribution_l103_103825


namespace value_of_a6_l103_103917

noncomputable def Sn (n : ℕ) : ℕ := n * 2^(n + 1)
noncomputable def an (n : ℕ) : ℕ := Sn n - Sn (n - 1)

theorem value_of_a6 : an 6 = 448 := by
  sorry

end value_of_a6_l103_103917


namespace find_length_CD_m_plus_n_l103_103111

noncomputable def lengthAB : ℝ := 7
noncomputable def lengthBD : ℝ := 11
noncomputable def lengthBC : ℝ := 9

axiom angle_BAD_ADC : Prop
axiom angle_ABD_BCD : Prop

theorem find_length_CD_m_plus_n :
  ∃ (m n : ℕ), gcd m n = 1 ∧ (CD = m / n) ∧ (m + n = 67) :=
sorry  -- Proof would be provided here

end find_length_CD_m_plus_n_l103_103111


namespace negation_of_diagonals_equal_l103_103156

-- Define a rectangle type and a function for the diagonals being equal
structure Rectangle :=
  (a b c d : ℝ) -- Assuming rectangle sides

-- Assume a function that checks if the diagonals of a given rectangle are equal
def diagonals_are_equal (r : Rectangle) : Prop :=
  sorry -- The actual function definition is omitted for this context

-- The proof problem
theorem negation_of_diagonals_equal :
  ¬ (∀ r : Rectangle, diagonals_are_equal r) ↔ (∃ r : Rectangle, ¬ diagonals_are_equal r) :=
by
  sorry

end negation_of_diagonals_equal_l103_103156


namespace find_percentage_l103_103628

theorem find_percentage (P : ℝ) (h: (20 / 100) * 580 = (P / 100) * 120 + 80) : P = 30 := 
by
  sorry

end find_percentage_l103_103628


namespace ratio_perimeters_l103_103884

noncomputable def rectangle_length : ℝ := 3
noncomputable def rectangle_width : ℝ := 2
noncomputable def triangle_hypotenuse : ℝ := Real.sqrt ((rectangle_length / 2) ^ 2 + rectangle_width ^ 2)
noncomputable def perimeter_rectangle : ℝ := 2 * (rectangle_length + rectangle_width)
noncomputable def perimeter_rhombus : ℝ := 4 * triangle_hypotenuse

theorem ratio_perimeters (h1 : rectangle_length = 3) (h2 : rectangle_width = 2) :
  (perimeter_rectangle / perimeter_rhombus) = 1 :=
by
  /- proof would go here -/
  sorry

end ratio_perimeters_l103_103884


namespace find_m_l103_103257

theorem find_m (m : ℝ) (h : ∀ x : ℝ, x - m > 5 ↔ x > 2) : m = -3 := by
  sorry

end find_m_l103_103257


namespace simplify_expression_l103_103968

theorem simplify_expression (a b : ℚ) (ha : a = -1) (hb : b = 1/4) : 
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by 
  sorry

end simplify_expression_l103_103968


namespace man_speed_in_still_water_l103_103024

theorem man_speed_in_still_water (upstream_speed downstream_speed : ℝ) (h1 : upstream_speed = 25) (h2 : downstream_speed = 45) :
  (upstream_speed + downstream_speed) / 2 = 35 :=
by
  sorry

end man_speed_in_still_water_l103_103024


namespace complete_square_cpjq_l103_103595

theorem complete_square_cpjq (j : ℝ) (c p q : ℝ) (h : 8 * j ^ 2 - 6 * j + 16 = c * (j + p) ^ 2 + q) :
  c = 8 ∧ p = -3/8 ∧ q = 119/8 → q / p = -119/3 :=
by
  intros
  cases a with hc hpq
  cases hpq with hp hq
  rw [hc, hp, hq]
  have hp_ne_zero : (-3 / 8) ≠ 0 := by norm_num
  field_simp [hp_ne_zero]
  norm_num
  sorry

end complete_square_cpjq_l103_103595


namespace graph_of_conic_section_is_straight_lines_l103_103044

variable {x y : ℝ}

theorem graph_of_conic_section_is_straight_lines:
  (x^2 - 9 * y^2 = 0) ↔ (x = 3 * y ∨ x = -3 * y) := by
  sorry

end graph_of_conic_section_is_straight_lines_l103_103044


namespace nina_money_l103_103129

theorem nina_money (W M : ℕ) (h1 : 6 * W = M) (h2 : 8 * (W - 2) = M) : M = 48 :=
by
  sorry

end nina_money_l103_103129


namespace Madison_minimum_score_l103_103873

theorem Madison_minimum_score (q1 q2 q3 q4 q5 : ℕ) (h1 : q1 = 84) (h2 : q2 = 81) (h3 : q3 = 87) (h4 : q4 = 83) (h5 : 85 * 5 ≤ q1 + q2 + q3 + q4 + q5) : 
  90 ≤ q5 := 
by
  sorry

end Madison_minimum_score_l103_103873


namespace correct_operation_l103_103333

theorem correct_operation (x y m c d : ℝ) : (5 * x * y - 4 * x * y = x * y) :=
by sorry

end correct_operation_l103_103333


namespace value_of_y_at_48_l103_103249

open Real

noncomputable def collinear_points (x : ℝ) : ℝ :=
  if x = 2 then 5
  else if x = 6 then 17
  else if x = 10 then 29
  else if x = 48 then 143
  else 0 -- placeholder value for other x (not used in proof)

theorem value_of_y_at_48 :
  (∀ (x1 x2 x3 : ℝ), x1 ≠ x2 → x2 ≠ x3 → x1 ≠ x3 → 
    ∃ (m : ℝ), m = (collinear_points x2 - collinear_points x1) / (x2 - x1) ∧ 
               m = (collinear_points x3 - collinear_points x2) / (x3 - x2)) →
  collinear_points 48 = 143 :=
by
  sorry

end value_of_y_at_48_l103_103249


namespace average_value_f_l103_103526

def f (x : ℝ) : ℝ := (1 + x)^3

theorem average_value_f : (1 / (4 - 2)) * (∫ x in (2:ℝ)..(4:ℝ), f x) = 68 :=
by
  sorry

end average_value_f_l103_103526


namespace polynomial_two_distinct_negative_real_roots_l103_103793

theorem polynomial_two_distinct_negative_real_roots :
  ∀ (p : ℝ), 
  (∃ (x1 x2 : ℝ), x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2 ∧ 
    (x1^4 + p*x1^3 + 3*x1^2 + p*x1 + 4 = 0) ∧ 
    (x2^4 + p*x2^3 + 3*x2^2 + p*x2 + 4 = 0)) ↔ 
  (p ≤ -2 ∨ p ≥ 2) :=
by
  sorry

end polynomial_two_distinct_negative_real_roots_l103_103793


namespace Dave_earning_l103_103522

def action_games := 3
def adventure_games := 2
def role_playing_games := 3

def price_action := 6
def price_adventure := 5
def price_role_playing := 7

def earning_from_action_games := action_games * price_action
def earning_from_adventure_games := adventure_games * price_adventure
def earning_from_role_playing_games := role_playing_games * price_role_playing

def total_earning := earning_from_action_games + earning_from_adventure_games + earning_from_role_playing_games

theorem Dave_earning : total_earning = 49 := by
  show total_earning = 49
  sorry

end Dave_earning_l103_103522


namespace net_emails_received_l103_103434

-- Define the conditions
def emails_received_morning : ℕ := 3
def emails_sent_morning : ℕ := 2
def emails_received_afternoon : ℕ := 5
def emails_sent_afternoon : ℕ := 1

-- Define the problem statement
theorem net_emails_received :
  emails_received_morning - emails_sent_morning + emails_received_afternoon - emails_sent_afternoon = 5 := by
  sorry

end net_emails_received_l103_103434


namespace scientific_notation_of_384_000_000_l103_103105

theorem scientific_notation_of_384_000_000 :
  384000000 = 3.84 * 10^8 :=
sorry

end scientific_notation_of_384_000_000_l103_103105


namespace solution_proof_l103_103444

variable (A B C : ℕ+) (x y : ℚ)
variable (h1 : A > B) (h2 : B > C) (h3 : A = B * (1 + x / 100)) (h4 : B = C * (1 + y / 100))

theorem solution_proof : x = 100 * ((A / (C * (1 + y / 100))) - 1) :=
by
  sorry

end solution_proof_l103_103444


namespace jar_weight_percentage_l103_103321

theorem jar_weight_percentage (J B : ℝ) (h : 0.60 * (J + B) = J + 1 / 3 * B) :
  (J / (J + B)) = 0.403 :=
by
  sorry

end jar_weight_percentage_l103_103321


namespace completing_the_square_l103_103166

theorem completing_the_square (x : ℝ) : x^2 + 8 * x + 9 = 0 → (x + 4)^2 = 7 :=
by 
  intro h
  sorry

end completing_the_square_l103_103166


namespace c_is_perfect_square_or_not_even_c_cannot_be_even_l103_103803

noncomputable def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem c_is_perfect_square_or_not_even 
  (a b c : ℕ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : c * (a * c + 1)^2 = (5 * c + 2 * b) * (2 * c + b))
  (h_odd : c % 2 = 1) : is_perfect_square c :=
sorry

theorem c_cannot_be_even 
  (a b c : ℕ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : c * (a * c + 1)^2 = (5 * c + 2 * b) * (2 * c + b))
  (h_even : c % 2 = 0) : false :=
sorry

end c_is_perfect_square_or_not_even_c_cannot_be_even_l103_103803


namespace incorrect_comparison_tan_138_tan_143_l103_103781

theorem incorrect_comparison_tan_138_tan_143 :
  ¬ (Real.tan (Real.pi * 138 / 180) > Real.tan (Real.pi * 143 / 180)) :=
by sorry

end incorrect_comparison_tan_138_tan_143_l103_103781


namespace evaluate_expression_l103_103232

theorem evaluate_expression : (24 : ℕ) = 2^3 * 3 ∧ (72 : ℕ) = 2^3 * 3^2 → (24^40 / 72^20 : ℚ) = 2^60 :=
by {
  sorry
}

end evaluate_expression_l103_103232


namespace swapped_two_digit_number_l103_103259

variable (a : ℕ)

theorem swapped_two_digit_number (h : a < 10) (sum_digits : ∃ t : ℕ, t + a = 13) : 
    ∃ n : ℕ, n = 9 * a + 13 :=
by
  sorry

end swapped_two_digit_number_l103_103259


namespace sum_proper_divisors_243_l103_103014

theorem sum_proper_divisors_243 : (1 + 3 + 9 + 27 + 81) = 121 := by
  sorry

end sum_proper_divisors_243_l103_103014


namespace twelve_pow_mn_eq_P_pow_2n_Q_pow_m_l103_103145

variable {m n : ℕ}

def P (m : ℕ) : ℕ := 2^m
def Q (n : ℕ) : ℕ := 3^n

theorem twelve_pow_mn_eq_P_pow_2n_Q_pow_m (m n : ℕ) : 12^(m * n) = (P m)^(2 * n) * (Q n)^m := 
sorry

end twelve_pow_mn_eq_P_pow_2n_Q_pow_m_l103_103145


namespace domain_sqrt_product_domain_log_fraction_l103_103235

theorem domain_sqrt_product (x : ℝ) (h1 : x - 2 ≥ 0) (h2 : x + 2 ≥ 0) : 
  2 ≤ x :=
by sorry

theorem domain_log_fraction (x : ℝ) (h1 : x + 1 > 0) (h2 : -x^2 - 3 * x + 4 > 0) : 
  -1 < x ∧ x < 1 :=
by sorry

end domain_sqrt_product_domain_log_fraction_l103_103235


namespace inequality_proof_l103_103806

open Real

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  (b + c) * (c + a) * (a + b) ≥ 4 * ((a + b + c) * ((a + b + c) / 3)^(1 / 8) - 1) :=
by
  sorry

end inequality_proof_l103_103806


namespace tangent_points_sum_constant_l103_103615

theorem tangent_points_sum_constant 
  (a : ℝ) (x1 y1 x2 y2 : ℝ)
  (hC1 : x1^2 = 4 * y1)
  (hC2 : x2^2 = 4 * y2)
  (hT1 : y1 - (-2) = (1/2)*x1*(x1 - a))
  (hT2 : y2 - (-2) = (1/2)*x2*(x2 - a)) :
  x1 * x2 + y1 * y2 = -4 :=
sorry

end tangent_points_sum_constant_l103_103615


namespace bicycle_cost_calculation_l103_103215

theorem bicycle_cost_calculation 
  (CP_A CP_B CP_C : ℝ)
  (h1 : CP_B = 1.20 * CP_A)
  (h2 : CP_C = 1.25 * CP_B)
  (h3 : CP_C = 225) :
  CP_A = 150 :=
by
  sorry

end bicycle_cost_calculation_l103_103215


namespace rabbits_total_distance_l103_103612

theorem rabbits_total_distance :
  let white_speed := 15
  let brown_speed := 12
  let grey_speed := 18
  let black_speed := 10
  let time := 7
  let white_distance := white_speed * time
  let brown_distance := brown_speed * time
  let grey_distance := grey_speed * time
  let black_distance := black_speed * time
  let total_distance := white_distance + brown_distance + grey_distance + black_distance
  total_distance = 385 :=
by
  sorry

end rabbits_total_distance_l103_103612


namespace sum_of_distinct_integers_l103_103446

noncomputable def distinct_integers (p q r s t : ℤ) : Prop :=
  (p ≠ q) ∧ (p ≠ r) ∧ (p ≠ s) ∧ (p ≠ t) ∧ 
  (q ≠ r) ∧ (q ≠ s) ∧ (q ≠ t) ∧ 
  (r ≠ s) ∧ (r ≠ t) ∧ 
  (s ≠ t)

theorem sum_of_distinct_integers
  (p q r s t : ℤ)
  (h_distinct : distinct_integers p q r s t)
  (h_product : (7 - p) * (7 - q) * (7 - r) * (7 - s) * (7 - t) = -120) :
  p + q + r + s + t = 22 :=
  sorry

end sum_of_distinct_integers_l103_103446


namespace Angela_insect_count_l103_103645

variables (Angela Jacob Dean : ℕ)
-- Conditions
def condition1 : Prop := Angela = Jacob / 2
def condition2 : Prop := Jacob = 5 * Dean
def condition3 : Prop := Dean = 30

-- Theorem statement proving Angela's insect count
theorem Angela_insect_count (h1 : condition1 Angela Jacob) (h2 : condition2 Jacob Dean) (h3 : condition3 Dean) : Angela = 75 :=
by
  sorry

end Angela_insect_count_l103_103645


namespace ratio_of_distances_l103_103738

/-- 
  Given two points A and B moving along intersecting lines with constant,
  but different velocities v_A and v_B respectively, prove that there exists a 
  point P such that at any moment in time, the ratio of distances AP to BP equals 
  the ratio of their velocities.
-/
theorem ratio_of_distances (A B : ℝ → ℝ × ℝ) (v_A v_B : ℝ)
  (intersecting_lines : ∃ t, A t = B t)
  (diff_velocities : v_A ≠ v_B) :
  ∃ P : ℝ × ℝ, ∀ t, (dist P (A t) / dist P (B t)) = v_A / v_B := 
sorry

end ratio_of_distances_l103_103738


namespace other_number_is_286_l103_103029

theorem other_number_is_286 (a b hcf lcm : ℕ) (h_hcf : hcf = 26) (h_lcm : lcm = 2310) (h_one_num : a = 210) 
  (rel : lcm * hcf = a * b) : b = 286 :=
by
  sorry

end other_number_is_286_l103_103029


namespace combined_height_l103_103844

/-- Given that Mr. Martinez is two feet taller than Chiquita and Chiquita is 5 feet tall, prove that their combined height is 12 feet. -/
theorem combined_height (h_chiquita : ℕ) (h_martinez : ℕ) 
  (h1 : h_chiquita = 5) (h2 : h_martinez = h_chiquita + 2) : 
  h_chiquita + h_martinez = 12 :=
by sorry

end combined_height_l103_103844


namespace factor_x_squared_minus_sixtyfour_l103_103053

theorem factor_x_squared_minus_sixtyfour (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) :=
by sorry

end factor_x_squared_minus_sixtyfour_l103_103053


namespace find_c_l103_103547

open Function

noncomputable def g (x : ℝ) : ℝ :=
  (x - 4) * (x - 2) * x * (x + 2) * (x + 4) / 255 - 5

theorem find_c (c : ℤ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, g x₁ = c ∧ g x₂ = c ∧ g x₃ = c ∧ g x₄ = c ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) →
  ∀ k : ℤ, k < c → ¬ ∃ x₁ x₂ x₃ x₄ : ℝ, g x₁ = k ∧ g x₂ = k ∧ g x₃ = k ∧ g x₄ = k ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ :=
sorry

end find_c_l103_103547


namespace Total_Cookies_is_135_l103_103456

-- Define the number of cookies in each pack
def PackA_Cookies : ℕ := 15
def PackB_Cookies : ℕ := 30
def PackC_Cookies : ℕ := 45

-- Define the number of packs bought by Paul and Paula
def Paul_PackA_Count : ℕ := 1
def Paul_PackB_Count : ℕ := 2
def Paula_PackA_Count : ℕ := 1
def Paula_PackC_Count : ℕ := 1

-- Calculate total cookies for Paul
def Paul_Cookies : ℕ := (Paul_PackA_Count * PackA_Cookies) + (Paul_PackB_Count * PackB_Cookies)

-- Calculate total cookies for Paula
def Paula_Cookies : ℕ := (Paula_PackA_Count * PackA_Cookies) + (Paula_PackC_Count * PackC_Cookies)

-- Calculate total cookies for Paul and Paula together
def Total_Cookies : ℕ := Paul_Cookies + Paula_Cookies

theorem Total_Cookies_is_135 : Total_Cookies = 135 := by
  sorry

end Total_Cookies_is_135_l103_103456


namespace value_of_abc_l103_103824

theorem value_of_abc : ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (ab + c + 10 = 51) ∧ (bc + a + 10 = 51) ∧ (ac + b + 10 = 51) ∧ (a + b + c = 41) :=
by
  sorry

end value_of_abc_l103_103824


namespace athletes_and_probability_l103_103715

-- Given conditions and parameters
def total_athletes_a := 27
def total_athletes_b := 9
def total_athletes_c := 18
def total_selected := 6
def athletes := ["A1", "A2", "A3", "A4", "A5", "A6"]

-- Definitions based on given conditions and solution steps
def selection_ratio := total_selected / (total_athletes_a + total_athletes_b + total_athletes_c)

def selected_from_a := total_athletes_a * selection_ratio
def selected_from_b := total_athletes_b * selection_ratio
def selected_from_c := total_athletes_c * selection_ratio

def pairs (l : List String) : List (String × String) :=
  (List.bind l (λ x => List.map (λ y => (x, y)) l)).filter (λ (x,y) => x < y)

def all_pairs := pairs athletes

def event_A (pair : String × String) : Bool :=
  pair.fst = "A5" ∨ pair.snd = "A5" ∨ pair.fst = "A6" ∨ pair.snd = "A6"

def favorable_event_A := all_pairs.filter event_A

noncomputable def probability_event_A := favorable_event_A.length / all_pairs.length

-- The main theorem: Number of athletes selected from each association and probability of event A
theorem athletes_and_probability : selected_from_a = 3 ∧ selected_from_b = 1 ∧ selected_from_c = 2 ∧ probability_event_A = 3/5 := by
  sorry

end athletes_and_probability_l103_103715


namespace algebraic_expression_is_product_l103_103316

def algebraicExpressionMeaning (x : ℝ) : Prop :=
  -7 * x = -7 * x

theorem algebraic_expression_is_product (x : ℝ) :
  algebraicExpressionMeaning x :=
by
  sorry

end algebraic_expression_is_product_l103_103316


namespace trig_identity_l103_103908

theorem trig_identity : Real.sin (35 * Real.pi / 6) + Real.cos (-11 * Real.pi / 3) = 0 := by
  sorry

end trig_identity_l103_103908


namespace building_height_l103_103888

noncomputable def height_of_building (H_f L_f L_b : ℝ) : ℝ :=
  (H_f * L_b) / L_f

theorem building_height (H_f L_f L_b H_b : ℝ)
  (H_f_val : H_f = 17.5)
  (L_f_val : L_f = 40.25)
  (L_b_val : L_b = 28.75)
  (H_b_val : H_b = 12.4375) :
  height_of_building H_f L_f L_b = H_b := by
  rw [H_f_val, L_f_val, L_b_val, H_b_val]
  -- sorry to skip the proof
  sorry

end building_height_l103_103888


namespace sum_of_powers_of_i_l103_103677

theorem sum_of_powers_of_i : 
  ∀ (i : ℂ), i^2 = -1 → 1 + i + i^2 + i^3 + i^4 + i^5 + i^6 = i :=
by
  intro i h
  sorry

end sum_of_powers_of_i_l103_103677


namespace quotient_of_division_l103_103987

theorem quotient_of_division (dividend divisor remainder : ℕ) (h_dividend : dividend = 127) (h_divisor : divisor = 14) (h_remainder : remainder = 1) :
  (dividend - remainder) / divisor = 9 :=
by 
  -- Proof follows
  sorry

end quotient_of_division_l103_103987


namespace find_symmetric_point_l103_103735

def slope_angle (l : ℝ → ℝ → Prop) (θ : ℝ) := ∃ m, m = Real.tan θ ∧ ∀ x y, l x y ↔ y = m * (x - 1) + 1
def passes_through (l : ℝ → ℝ → Prop) (P : ℝ × ℝ) := l P.fst P.snd
def symmetric_point (A A' : ℝ × ℝ) (l : ℝ → ℝ → Prop) := 
  (A'.snd - A.snd = A'.fst - A.fst) ∧ 
  ((A'.fst + A.fst) / 2 + (A'.snd + A.snd) / 2 - 2 = 0)

theorem find_symmetric_point :
  ∃ l : ℝ → ℝ → Prop, 
    slope_angle l (135 : ℝ) ∧ 
    passes_through l (1, 1) ∧ 
    (∀ x y, l x y ↔ x + y = 2) ∧ 
    symmetric_point (3, 4) (-2, -1) l :=
by sorry

end find_symmetric_point_l103_103735


namespace lowest_price_for_16_oz_butter_l103_103381

-- Define the constants
def price_single_16_oz_package : ℝ := 7
def price_8_oz_package : ℝ := 4
def price_4_oz_package : ℝ := 2
def discount_4_oz_package : ℝ := 0.5

-- Calculate the discounted price for a 4 oz package
def discounted_price_4_oz_package : ℝ := price_4_oz_package * discount_4_oz_package

-- Calculate the total price for two discounted 4 oz packages
def total_price_two_discounted_4_oz_packages : ℝ := 2 * discounted_price_4_oz_package

-- Calculate the total price using the 8 oz package and two discounted 4 oz packages
def total_price_using_coupon : ℝ := price_8_oz_package + total_price_two_discounted_4_oz_packages

-- State the property to prove
theorem lowest_price_for_16_oz_butter :
  min price_single_16_oz_package total_price_using_coupon = 6 :=
sorry

end lowest_price_for_16_oz_butter_l103_103381


namespace geom_seq_min_value_l103_103284

theorem geom_seq_min_value (r : ℝ) : 
  (1 : ℝ) = a_1 → a_2 = r → a_3 = r^2 → ∃ r : ℝ, 6 * a_2 + 7 * a_3 = -9/7 := 
by 
  intros h1 h2 h3 
  use -3/7 
  rw [h2, h3] 
  ring 
  sorry

end geom_seq_min_value_l103_103284


namespace initial_position_l103_103988

variable (x : Int)

theorem initial_position 
  (h: x - 5 + 4 + 2 - 3 + 1 = 6) : x = 7 := 
  by 
  sorry

end initial_position_l103_103988


namespace ed_more_marbles_l103_103655

-- Define variables for initial number of marbles
variables {E D : ℕ}

-- Ed had some more marbles than Doug initially.
-- Doug lost 8 of his marbles at the playground.
-- Now Ed has 30 more marbles than Doug.
theorem ed_more_marbles (h : E = (D - 8) + 30) : E - D = 22 :=
by
  sorry

end ed_more_marbles_l103_103655


namespace fewest_four_dollar_frisbees_l103_103625

theorem fewest_four_dollar_frisbees (x y: ℕ): 
    x + y = 64 ∧ 3 * x + 4 * y = 200 → y = 8 := by sorry

end fewest_four_dollar_frisbees_l103_103625


namespace worker_y_defective_rate_l103_103784

noncomputable def y_f : ℚ := 0.1666666666666668
noncomputable def d_x : ℚ := 0.005 -- converting percentage to decimal
noncomputable def d_total : ℚ := 0.0055 -- converting percentage to decimal

theorem worker_y_defective_rate :
  ∃ d_y : ℚ, d_y = 0.008 ∧ d_total = ((1 - y_f) * d_x + y_f * d_y) :=
by
  sorry

end worker_y_defective_rate_l103_103784


namespace lowest_price_for_butter_l103_103379

def cost_single_package : ℝ := 7.0
def cost_8oz_package : ℝ := 4.0
def cost_4oz_package : ℝ := 2.0
def discount : ℝ := 0.5

theorem lowest_price_for_butter : 
  min cost_single_package (cost_8oz_package + 2 * (cost_4oz_package * discount)) = 6.0 :=
by
  sorry

end lowest_price_for_butter_l103_103379


namespace problem_solution_count_l103_103230

theorem problem_solution_count (n : ℕ) (h1 : (80 * n) ^ 40 > n ^ 80) (h2 : n ^ 80 > 3 ^ 160) : 
  ∃ s : Finset ℕ, s.card = 70 ∧ ∀ x ∈ s, 10 ≤ x ∧ x ≤ 79 :=
by
  sorry

end problem_solution_count_l103_103230


namespace distance_not_all_odd_l103_103724

theorem distance_not_all_odd (A B C D : ℝ × ℝ) : 
  ∃ (P Q : ℝ × ℝ), dist P Q % 2 = 0 := by sorry

end distance_not_all_odd_l103_103724


namespace luke_initial_money_l103_103125

def initial_amount (X : ℤ) : Prop :=
  let spent := 11
  let received := 21
  let current_amount := 58
  X - spent + received = current_amount

theorem luke_initial_money : ∃ (X : ℤ), initial_amount X ∧ X = 48 :=
by
  sorry

end luke_initial_money_l103_103125


namespace scale_total_length_l103_103370

/-- Defining the problem parameters. -/
def number_of_parts : ℕ := 5
def length_of_each_part : ℕ := 18

/-- Theorem stating the total length of the scale. -/
theorem scale_total_length : number_of_parts * length_of_each_part = 90 :=
by
  sorry

end scale_total_length_l103_103370


namespace sheila_will_attend_picnic_l103_103966

def P_Rain : ℝ := 0.3
def P_Cloudy : ℝ := 0.4
def P_Sunny : ℝ := 0.3

def P_Attend_if_Rain : ℝ := 0.25
def P_Attend_if_Cloudy : ℝ := 0.5
def P_Attend_if_Sunny : ℝ := 0.75

def P_Attend : ℝ :=
  P_Rain * P_Attend_if_Rain +
  P_Cloudy * P_Attend_if_Cloudy +
  P_Sunny * P_Attend_if_Sunny

theorem sheila_will_attend_picnic : P_Attend = 0.5 := by
  sorry

end sheila_will_attend_picnic_l103_103966


namespace cart_distance_traveled_l103_103973

-- Define the problem parameters/conditions
def circumference_front : ℕ := 30
def circumference_back : ℕ := 33
def revolutions_difference : ℕ := 5

-- Define the question and the expected correct answer
theorem cart_distance_traveled :
  ∀ (R : ℕ), ((R + revolutions_difference) * circumference_front = R * circumference_back) → (R * circumference_back) = 1650 :=
by
  intro R h
  sorry

end cart_distance_traveled_l103_103973


namespace problem_statement_l103_103953

variables {AB CD BC DA : ℝ} (E : ℝ) (midpoint_E : E = BC / 2) (ins_ABC : circle_inscribable AB ED)
  (ins_AEC : circle_inscribable AE CD) (a b c d : ℝ) (h_AB : AB = a) (h_BC : BC = b) (h_CD : CD = c)
  (h_DA : DA = d)

theorem problem_statement :
  a + c = b / 3 + d ∧ (1 / a + 1 / c = 3 / b) :=
by
  sorry

end problem_statement_l103_103953


namespace work_rate_solution_l103_103342

theorem work_rate_solution (x : ℝ) (hA : 60 > 0) (hB : x > 0) (hTogether : 15 > 0) :
  (1 / 60 + 1 / x = 1 / 15) → (x = 20) :=
by 
  sorry -- Proof Placeholder

end work_rate_solution_l103_103342


namespace anne_speed_ratio_l103_103902

variable (B A A' : ℝ) (hours_to_clean_together : ℝ) (hours_to_clean_with_new_anne : ℝ)

-- Conditions
def cleaning_condition_1 := (A + B) * 4 = 1 -- Combined rate for 4 hours
def cleaning_condition_2 := A = 1 / 12      -- Anne's rate alone
def cleaning_condition_3 := (A' + B) * 3 = 1 -- Combined rate for 3 hours with new Anne's rate

-- Theorem to Prove
theorem anne_speed_ratio (h1 : cleaning_condition_1 B A)
                         (h2 : cleaning_condition_2 A)
                         (h3 : cleaning_condition_3 B A') :
                         (A' / A) = 2 :=
by sorry

end anne_speed_ratio_l103_103902


namespace average_selections_correct_l103_103638

noncomputable def cars := 18
noncomputable def selections_per_client := 3
noncomputable def clients := 18
noncomputable def total_selections := clients * selections_per_client
noncomputable def average_selections_per_car := total_selections / cars

theorem average_selections_correct :
  average_selections_per_car = 3 :=
by
  sorry

end average_selections_correct_l103_103638


namespace cookies_left_after_ted_leaves_l103_103242

theorem cookies_left_after_ted_leaves :
  let f : Nat := 2 -- trays per day
  let d : Nat := 6 -- days
  let e_f : Nat := 1 -- cookies eaten per day by Frank
  let t : Nat := 4 -- cookies eaten by Ted
  let c : Nat := 12 -- cookies per tray
  let total_cookies := f * c * d -- total cookies baked
  let cookies_eaten_by_frank := e_f * d -- total cookies eaten by Frank
  let cookies_before_ted := total_cookies - cookies_eaten_by_frank -- cookies before Ted
  let total_cookies_left := cookies_before_ted - t -- cookies left after Ted
  total_cookies_left = 134
:= by
  sorry

end cookies_left_after_ted_leaves_l103_103242


namespace linda_total_distance_l103_103285

theorem linda_total_distance :
  ∃ x : ℕ, (60 % x = 0) ∧ ((75 % (x + 3)) = 0) ∧ ((90 % (x + 6)) = 0) ∧
  (60 / x + 75 / (x + 3) + 90 / (x + 6) = 15) :=
sorry

end linda_total_distance_l103_103285


namespace sum_of_cubes_l103_103429

theorem sum_of_cubes (x y : ℝ) (h_sum : x + y = 3) (h_prod : x * y = 2) : x^3 + y^3 = 9 :=
by
  sorry

end sum_of_cubes_l103_103429


namespace student_distribution_l103_103047

open Finset Nat

theorem student_distribution :
  let n := 7
  count_combine (choose n 2 + choose n 3) 2 = 112 :=
by
  sorry

end student_distribution_l103_103047


namespace max_parallelograms_in_hexagon_l103_103689

theorem max_parallelograms_in_hexagon (side_hexagon side_parallelogram1 side_parallelogram2 : ℝ)
                                        (angle_parallelogram : ℝ) :
  side_hexagon = 3 ∧ side_parallelogram1 = 1 ∧ side_parallelogram2 = 2 ∧ angle_parallelogram = (π / 3) →
  ∃ n : ℕ, n = 12 :=
by 
  sorry

end max_parallelograms_in_hexagon_l103_103689


namespace fraction_numerator_l103_103152

theorem fraction_numerator (x : ℤ) (h₁ : 2 * x + 11 ≠ 0) (h₂ : (x : ℚ) / (2 * x + 11) = 3 / 4) : x = -33 / 2 :=
by
  sorry

end fraction_numerator_l103_103152


namespace maxvalue_on_ellipse_l103_103529

open Real

noncomputable def max_x_plus_y : ℝ := 343 / 88

theorem maxvalue_on_ellipse (x y : ℝ) :
  (x^2 + 3 * x * y + 2 * y^2 - 14 * x - 21 * y + 49 = 0) →
  x + y ≤ max_x_plus_y := 
sorry

end maxvalue_on_ellipse_l103_103529


namespace lea_total_cost_l103_103585

theorem lea_total_cost :
  let book_cost := 16 in
  let binders_count := 3 in
  let binder_cost := 2 in
  let notebooks_count := 6 in
  let notebook_cost := 1 in
  book_cost + (binders_count * binder_cost) + (notebooks_count * notebook_cost) = 28 :=
by
  sorry

end lea_total_cost_l103_103585


namespace mildred_weight_is_correct_l103_103452

noncomputable def carol_weight := 9
noncomputable def mildred_weight := carol_weight + 50

theorem mildred_weight_is_correct : mildred_weight = 59 :=
by 
  -- the proof is omitted
  sorry

end mildred_weight_is_correct_l103_103452


namespace min_points_each_player_l103_103341

theorem min_points_each_player (n : ℕ) (total_points : ℕ) (max_points : ℕ) (min_points : ℕ) 
  (h_team : n = 12) (h_total : total_points = 100) (h_max : max_points = 23) (h_min : min_points = 7) :
  ∃ p : ℕ, p = max_points ∧ (∃ points : ℕ → ℕ, (∀ i, i ≠ 0 → points i = min_points) ∧ ∀ i, points 0 = max_points ∧ ∑ i in finset.range n, points i = total_points) :=
by 
  sorry

end min_points_each_player_l103_103341


namespace completing_the_square_solution_l103_103182

theorem completing_the_square_solution : ∀ x : ℝ, x^2 + 8 * x + 9 = 0 ↔ (x + 4)^2 = 7 := by
  sorry

end completing_the_square_solution_l103_103182


namespace completing_the_square_solution_correct_l103_103173

theorem completing_the_square_solution_correct (x : ℝ) :
  (x^2 + 8 * x + 9 = 0) ↔ ((x + 4)^2 = 7) :=
by
  sorry

end completing_the_square_solution_correct_l103_103173


namespace female_democrats_count_l103_103614

theorem female_democrats_count (F M : ℕ) (h1 : F + M = 750) 
  (h2 : F / 2 ≠ 0) (h3 : M / 4 ≠ 0) 
  (h4 : F / 2 + M / 4 = 750 / 3) : F / 2 = 125 :=
by
  sorry

end female_democrats_count_l103_103614


namespace combined_earnings_l103_103516

theorem combined_earnings (dwayne_earnings brady_earnings : ℕ) (h1 : dwayne_earnings = 1500) (h2 : brady_earnings = dwayne_earnings + 450) : 
  dwayne_earnings + brady_earnings = 3450 :=
by 
  rw [h1, h2]
  sorry

end combined_earnings_l103_103516


namespace bricks_needed_l103_103026

theorem bricks_needed 
    (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ) 
    (wall_length_m : ℝ) (wall_height_m : ℝ) (wall_width_cm : ℝ)
    (H1 : brick_length = 25) (H2 : brick_width = 11.25) (H3 : brick_height = 6)
    (H4 : wall_length_m = 7) (H5 : wall_height_m = 6) (H6 : wall_width_cm = 22.5) :
    (wall_length_m * 100 * wall_height_m * 100 * wall_width_cm) / (brick_length * brick_width * brick_height) = 5600 :=
by
    sorry

end bricks_needed_l103_103026


namespace trigonometric_identity_l103_103673

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 4) :
  (Real.sin θ + Real.cos θ) / (17 * Real.sin θ) + (Real.sin θ ^ 2) / 4 = 21 / 68 := 
sorry

end trigonometric_identity_l103_103673


namespace silver_cube_price_l103_103344

theorem silver_cube_price
  (price_2inch_cube : ℝ := 300) (side_length_2inch : ℝ := 2) (side_length_4inch : ℝ := 4) : 
  price_4inch_cube = 2400 := 
by 
  sorry

end silver_cube_price_l103_103344


namespace completing_the_square_solution_l103_103184

theorem completing_the_square_solution : ∀ x : ℝ, x^2 + 8 * x + 9 = 0 ↔ (x + 4)^2 = 7 := by
  sorry

end completing_the_square_solution_l103_103184


namespace savings_calculation_l103_103350

noncomputable def weekly_rate_peak : ℕ := 10
noncomputable def weekly_rate_non_peak : ℕ := 8
noncomputable def monthly_rate_peak : ℕ := 40
noncomputable def monthly_rate_non_peak : ℕ := 35
noncomputable def non_peak_duration_weeks : ℝ := 17.33
noncomputable def peak_duration_weeks : ℝ := 52 - non_peak_duration_weeks
noncomputable def non_peak_duration_months : ℕ := 4
noncomputable def peak_duration_months : ℕ := 12 - non_peak_duration_months

noncomputable def total_weekly_cost := (non_peak_duration_weeks * weekly_rate_non_peak) 
                                     + (peak_duration_weeks * weekly_rate_peak)

noncomputable def total_monthly_cost := (non_peak_duration_months * monthly_rate_non_peak) 
                                      + (peak_duration_months * monthly_rate_peak)

noncomputable def savings := total_weekly_cost - total_monthly_cost

theorem savings_calculation 
  : savings = 25.34 := by
  sorry

end savings_calculation_l103_103350


namespace spending_difference_l103_103972

-- Define the given conditions
def ice_cream_cartons := 19
def yoghurt_cartons := 4
def ice_cream_cost_per_carton := 7
def yoghurt_cost_per_carton := 1

-- Calculate the total cost based on the given conditions
def total_ice_cream_cost := ice_cream_cartons * ice_cream_cost_per_carton
def total_yoghurt_cost := yoghurt_cartons * yoghurt_cost_per_carton

-- The statement to prove
theorem spending_difference :
  total_ice_cream_cost - total_yoghurt_cost = 129 :=
by
  sorry

end spending_difference_l103_103972


namespace sum_of_six_smallest_multiples_of_12_l103_103744

-- Define the six smallest distinct positive integer multiples of 12
def multiples_of_12 : List ℕ := [12, 24, 36, 48, 60, 72]

-- Define their sum
def sum_of_multiples : ℕ := multiples_of_12.sum

-- The proof statement
theorem sum_of_six_smallest_multiples_of_12 : sum_of_multiples = 252 := 
by
  sorry

end sum_of_six_smallest_multiples_of_12_l103_103744


namespace find_s_l103_103096

theorem find_s (s t : ℚ) (h1 : 8 * s + 6 * t = 120) (h2 : s = t - 3) : s = 51 / 7 := by
  sorry

end find_s_l103_103096


namespace volume_in_cubic_yards_l103_103359

theorem volume_in_cubic_yards (V : ℝ) (conversion_factor : ℝ) (hV : V = 216) (hcf : conversion_factor = 27) :
  V / conversion_factor = 8 := by
  sorry

end volume_in_cubic_yards_l103_103359


namespace sum_X_Y_Z_W_eq_156_l103_103647

theorem sum_X_Y_Z_W_eq_156 
  (X Y Z W : ℕ) 
  (h_arith_seq : Y - X = Z - Y)
  (h_geom_seq : Z / Y = 9 / 5)
  (h_W : W = Z^2 / Y) 
  (h_pos : 0 < X ∧ 0 < Y ∧ 0 < Z ∧ 0 < W) :
  X + Y + Z + W = 156 :=
sorry

end sum_X_Y_Z_W_eq_156_l103_103647


namespace inverse_variation_solution_l103_103466

noncomputable def const_k (x y : ℝ) := (x^2) * (y^4)

theorem inverse_variation_solution (x y : ℝ) (k : ℝ) (h1 : x = 8) (h2 : y = 2) (h3 : k = const_k x y) :
  ∀ y' : ℝ, y' = 4 → const_k x y' = 1024 → x^2 = 4 := by
  intros
  sorry

end inverse_variation_solution_l103_103466


namespace arithmeticSeqModulus_l103_103775

-- Define the arithmetic sequence
def arithmeticSeqSum (a d l : ℕ) : ℕ :=
  let n := (l - a) / d + 1
  n * (a + l) / 2

-- The main theorem to prove
theorem arithmeticSeqModulus : arithmeticSeqSum 2 5 102 % 20 = 12 := by
  sorry

end arithmeticSeqModulus_l103_103775


namespace largest_square_in_right_triangle_largest_rectangle_in_right_triangle_l103_103916

theorem largest_square_in_right_triangle (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  ∃ s, s = (a * b) / (a + b) := 
sorry

theorem largest_rectangle_in_right_triangle (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  ∃ x y, x = a / 2 ∧ y = b / 2 :=
sorry

end largest_square_in_right_triangle_largest_rectangle_in_right_triangle_l103_103916


namespace find_AX_l103_103274

theorem find_AX
  (AB AC BC : ℚ)
  (H : AB = 80)
  (H1 : AC = 50)
  (H2 : BC = 30)
  (angle_bisector_theorem_1 : ∀ (AX XC y : ℚ), AX = 8 * y ∧ XC = 3 * y ∧ 11 * y = AC → y = 50 / 11)
  (angle_bisector_theorem_2 : ∀ (BD DC z : ℚ), BD = 8 * z ∧ DC = 5 * z ∧ 13 * z = BC → z = 30 / 13) :
  AX = 400 / 11 := 
sorry

end find_AX_l103_103274


namespace find_p_root_relation_l103_103907

theorem find_p_root_relation (p : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ 0 ∧ x2 = 3 * x1 ∧ x1^2 + p * x1 + 2 * p = 0 ∧ x2^2 + p * x2 + 2 * p = 0) ↔ (p = 0 ∨ p = 32 / 3) :=
by sorry

end find_p_root_relation_l103_103907


namespace optimal_pricing_for_max_profit_l103_103759

noncomputable def sales_profit (x : ℝ) : ℝ :=
  -5 * x^3 + 45 * x^2 - 75 * x + 675

theorem optimal_pricing_for_max_profit :
  ∃ x : ℝ, 0 ≤ x ∧ x < 9 ∧ ∀ y : ℝ, 0 ≤ y ∧ y < 9 → sales_profit y ≤ sales_profit 5 ∧ (14 - 5 = 9) :=
by
  sorry

end optimal_pricing_for_max_profit_l103_103759


namespace no_solution_2023_l103_103836

theorem no_solution_2023 (a b c : ℕ) (h₁ : a + b + c = 2023) (h₂ : (b + c) ∣ a) (h₃ : (b - c + 1) ∣ (b + c)) : false :=
by
  sorry

end no_solution_2023_l103_103836


namespace polynomial_solution_l103_103792

variable {R : Type*} [CommRing R]

theorem polynomial_solution (p : Polynomial R) :
  (∀ (a b c : R), 
    p.eval (a + b - 2 * c) + p.eval (b + c - 2 * a) + p.eval (c + a - 2 * b)
      = 3 * p.eval (a - b) + 3 * p.eval (b - c) + 3 * p.eval (c - a)
  ) →
  ∃ (a1 a2 : R), p = Polynomial.C a2 * Polynomial.X^2 + Polynomial.C a1 * Polynomial.X :=
by
  sorry

end polynomial_solution_l103_103792


namespace avg_age_9_proof_l103_103109

-- Definitions of the given conditions
def total_persons := 16
def avg_age_all := 15
def total_age_all := total_persons * avg_age_all -- 240
def persons_5 := 5
def avg_age_5 := 14
def total_age_5 := persons_5 * avg_age_5 -- 70
def age_15th_person := 26
def persons_9 := 9

-- The theorem to prove the average age of the remaining 9 persons
theorem avg_age_9_proof : 
  total_age_all - total_age_5 - age_15th_person = persons_9 * 16 :=
by
  sorry

end avg_age_9_proof_l103_103109


namespace calculate_interest_rate_l103_103604

variables (A : ℝ) (R : ℝ)

-- Conditions as definitions in Lean 4
def compound_interest_condition (A : ℝ) (R : ℝ) : Prop :=
  (A * (1 + R)^20 = 4 * A)

-- Theorem statement
theorem calculate_interest_rate (A : ℝ) (R : ℝ) (h : compound_interest_condition A R) : 
  R = (4)^(1/20) - 1 := 
sorry

end calculate_interest_rate_l103_103604


namespace ascorbic_acid_molecular_weight_l103_103012

theorem ascorbic_acid_molecular_weight (C H O : ℕ → ℝ)
  (C_weight : C 6 = 6 * 12.01)
  (H_weight : H 8 = 8 * 1.008)
  (O_weight : O 6 = 6 * 16.00)
  (total_mass_given : 528 = 6 * 12.01 + 8 * 1.008 + 6 * 16.00) :
  6 * 12.01 + 8 * 1.008 + 6 * 16.00 = 176.124 := 
by 
  sorry

end ascorbic_acid_molecular_weight_l103_103012


namespace apple_sharing_l103_103642

theorem apple_sharing (a b c : ℕ) (h : a + b + c = 30) (h1 : a ≥ 3) (h2 : b ≥ 3) (h3 : c ≥ 3) : 
    ∃ n, n = 253 :=
by 
  -- Using the conditions, we need to count the ways to distribute the remaining 21 apples
  let a' := a - 3
  let b' := b - 3
  let c' := c - 3
  have h' : a' + b' + c' = 21 := by
    rw [← nat.add_sub_of_le h1, ← nat.add_sub_of_le h2, ← nat.add_sub_of_le h3] at h
    exact nat.sub_add_cancel h

  -- Applying the stars and bars theorem
  use nat.choose 23 2
  have choose_eq : nat.choose 23 2 = 253 := by
    calc
      nat.choose 23 2 = 23 * 22 / 2 : rfl
               ...      = 253       : by norm_num
  exact choose_eq

end apple_sharing_l103_103642


namespace lcm_of_132_and_315_l103_103985

def n1 : ℕ := 132
def n2 : ℕ := 315

theorem lcm_of_132_and_315 :
  (Nat.lcm n1 n2) = 13860 :=
by
  -- Proof goes here
  sorry

end lcm_of_132_and_315_l103_103985


namespace correct_option_is_C_l103_103623

-- Define the polynomial expressions and their expected values as functions
def optionA (x : ℝ) : Prop := (x + 2) * (x - 5) = x^2 - 2 * x - 3
def optionB (x : ℝ) : Prop := (x + 3) * (x - 1 / 3) = x^2 + x - 1
def optionC (x : ℝ) : Prop := (x - 2 / 3) * (x + 1 / 2) = x^2 - 1 / 6 * x - 1 / 3
def optionD (x : ℝ) : Prop := (x - 2) * (-x - 2) = x^2 - 4

-- Problem Statement: Verify that the polynomial multiplication in Option C is correct
theorem correct_option_is_C (x : ℝ) : optionC x :=
by
  -- Statement indicating the proof goes here
  sorry

end correct_option_is_C_l103_103623


namespace lab_preparation_is_correct_l103_103862

def correct_operation (m_CuSO4 : ℝ) (m_CuSO4_5H2O : ℝ) (V_solution : ℝ) : Prop :=
  let molar_mass_CuSO4 := 160 -- g/mol
  let molar_mass_CuSO4_5H2O := 250 -- g/mol
  let desired_concentration := 0.1 -- mol/L
  let desired_volume := 0.480 -- L
  let prepared_volume := 0.500 -- L
  (m_CuSO4 = 8.0 ∧ V_solution = 0.500 ∧ m_CuSO4_5H2O = 12.5 ∧ desired_concentration * prepared_volume * molar_mass_CuSO4_5H2O = 12.5)

-- Example proof statement to show the problem with "sorry"
theorem lab_preparation_is_correct : correct_operation 8.0 12.5 0.500 :=
by
  sorry

end lab_preparation_is_correct_l103_103862


namespace simplify_expression_l103_103601

theorem simplify_expression (x y : ℝ) :
  5 * x - 3 * y + 9 * x ^ 2 + 8 - (4 - 5 * x + 3 * y - 9 * x ^ 2) = 18 * x ^ 2 + 10 * x - 6 * y + 4 :=
by
  sorry

end simplify_expression_l103_103601


namespace total_doughnuts_made_l103_103206

def num_doughnuts_per_box : ℕ := 10
def num_boxes_sold : ℕ := 27
def doughnuts_given_away : ℕ := 30

theorem total_doughnuts_made :
  num_boxes_sold * num_doughnuts_per_box + doughnuts_given_away = 300 :=
by
  sorry

end total_doughnuts_made_l103_103206


namespace lateral_surface_area_of_cylinder_l103_103682

theorem lateral_surface_area_of_cylinder :
  let r := 1
  let h := 2
  2 * Real.pi * r * h = 4 * Real.pi :=
by
  sorry

end lateral_surface_area_of_cylinder_l103_103682


namespace problem_l103_103539

theorem problem (a : ℝ) (h : a^2 - 5 * a - 1 = 0) : 3 * a^2 - 15 * a = 3 :=
by
  sorry

end problem_l103_103539


namespace bugs_meet_again_l103_103006

-- Define the constants for radii and speeds
def r1 : ℝ := 7
def r2 : ℝ := 3
def s1 : ℝ := 4 * real.pi
def s2 : ℝ := 3 * real.pi

-- Define the circumferences of the circles
def C1 : ℝ := 2 * r1 * real.pi
def C2 : ℝ := 2 * r2 * real.pi

-- Define the times to complete one full rotation
def t1 : ℝ := C1 / s1
def t2 : ℝ := C2 / s2

-- Define the least common multiple of the rotation times
def lcm_t1_t2 : ℝ := real.lcm (int.of_real t1) (int.of_real t2)

theorem bugs_meet_again : lcm_t1_t2 = 7 := by
  -- We can provide the proof in this section
  sorry

end bugs_meet_again_l103_103006


namespace percentage_square_area_in_rectangle_l103_103639

variable (s : ℝ)
variable (W : ℝ) (L : ℝ)
variable (hW : W = 3 * s) -- Width is 3 times the side of the square
variable (hL : L = (3 / 2) * W) -- Length is 3/2 times the width

theorem percentage_square_area_in_rectangle :
  (s^2 / ((27 * s^2) / 2)) * 100 = 7.41 :=
by 
  sorry

end percentage_square_area_in_rectangle_l103_103639


namespace proof_n_value_l103_103142

theorem proof_n_value (n : ℕ) (h : (9^n) * (9^n) * (9^n) * (9^n) * (9^n) = 81^5) : n = 2 :=
by
  sorry

end proof_n_value_l103_103142


namespace maximize_root_product_l103_103782

theorem maximize_root_product :
  (∃ k : ℝ, ∀ x : ℝ, 6 * x^2 - 5 * x + k = 0 ∧ (25 - 24 * k ≥ 0)) →
  ∃ k : ℝ, k = 25 / 24 :=
by
  sorry

end maximize_root_product_l103_103782

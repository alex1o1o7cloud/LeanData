import Mathlib

namespace union_sets_l1579_157920

open Set

variable (x : ℝ)

def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | x^2 - 3 * x + 2 ≤ 0}

theorem union_sets : A ∪ B = {x | x ≥ 1} :=
  by
    sorry

end union_sets_l1579_157920


namespace find_angle_B_l1579_157912

variable {a b c : ℝ}
variable {A B C : ℝ}
variable {m n : ℝ × ℝ}
variable (h1 : m = (Real.cos A, Real.sin A))
variable (h2 : n = (1, Real.sqrt 3))
variable (h3 : m.1 / n.1 = m.2 / n.2)
variable (h4 : a * Real.cos B + b * Real.cos A = c * Real.sin C)

theorem find_angle_B (h_conditions : a * Real.cos B + b * Real.cos A = c * Real.sin C) : B = Real.pi / 6 :=
sorry

end find_angle_B_l1579_157912


namespace tan_theta_equation_l1579_157953

theorem tan_theta_equation (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 6) :
  Real.tan θ + Real.tan (4 * θ) + Real.tan (6 * θ) = 0 → Real.tan θ = 1 / Real.sqrt 3 :=
by
  sorry

end tan_theta_equation_l1579_157953


namespace equation_relating_price_and_tax_and_discount_l1579_157906

variable (c t d : ℚ)

theorem equation_relating_price_and_tax_and_discount
  (h1 : 1.30 * c * ((100 + t) / 100) * ((100 - d) / 100) = 351) :
    1.30 * c * (100 + t) * (100 - d) = 3510000 := by
  sorry

end equation_relating_price_and_tax_and_discount_l1579_157906


namespace giant_lollipop_calories_l1579_157971

-- Definitions based on the conditions
def sugar_per_chocolate_bar := 10
def chocolate_bars_bought := 14
def sugar_in_giant_lollipop := 37
def total_sugar := 177
def calories_per_gram_of_sugar := 4

-- Prove that the number of calories in the giant lollipop is 148 given the conditions
theorem giant_lollipop_calories : (sugar_in_giant_lollipop * calories_per_gram_of_sugar) = 148 := by
  sorry

end giant_lollipop_calories_l1579_157971


namespace find_constant_term_l1579_157904

theorem find_constant_term (q' : ℝ → ℝ) (c : ℝ) (h1 : ∀ q : ℝ, q' q = 3 * q - c)
  (h2 : q' (q' 7) = 306) : c = 252 :=
by
  sorry

end find_constant_term_l1579_157904


namespace distribute_7_balls_into_4_boxes_l1579_157929

-- Define the problem conditions
def number_of_ways_to_distribute_balls (balls boxes : ℕ) : ℕ :=
  if balls < boxes then 0 else Nat.choose (balls - 1) (boxes - 1)

-- Prove the specific case
theorem distribute_7_balls_into_4_boxes : number_of_ways_to_distribute_balls 7 4 = 20 :=
by
  -- Definition and proof to be filled
  sorry

end distribute_7_balls_into_4_boxes_l1579_157929


namespace minimum_value_of_expression_l1579_157924

noncomputable def min_value_expression (x y : ℝ) : ℝ := 
  (x + 1)^2 / (x + 2) + 3 / (x + 2) + y^2 / (y + 1)

theorem minimum_value_of_expression :
  ∀ (x y : ℝ), x ≥ 0 → y ≥ 0 → x + y = 2 → min_value_expression x y = 14 / 5 :=
by
  sorry

end minimum_value_of_expression_l1579_157924


namespace min_m_min_expression_l1579_157977

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- Part (Ⅰ)
theorem min_m (m : ℝ) (h : ∃ x₀ : ℝ, f x₀ ≤ m) : m ≥ 2 := sorry

-- Part (Ⅱ)
theorem min_expression (a b : ℝ) (h1 : 3 * a + b = 2) (h2 : 0 < a) (h3 : 0 < b) :
  (1 / (2 * a) + 1 / (a + b)) ≥ 2 := sorry

end min_m_min_expression_l1579_157977


namespace evaluate_expression_l1579_157963

noncomputable def ln (x : ℝ) : ℝ := Real.log x

theorem evaluate_expression : 
  2017 ^ ln (ln 2017) - (ln 2017) ^ ln 2017 = 0 :=
by
  sorry

end evaluate_expression_l1579_157963


namespace solution_set_abs_inequality_l1579_157990

theorem solution_set_abs_inequality : {x : ℝ | |x - 1| < 2} = {x : ℝ | -1 < x ∧ x < 3} :=
sorry

end solution_set_abs_inequality_l1579_157990


namespace length_of_AB_l1579_157966

-- Define the parabola and the line passing through the focus F
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def line (x y : ℝ) : Prop := y = x - 1

theorem length_of_AB : 
  (∃ F : ℝ × ℝ, F = (1, 0) ∧ line F.1 F.2) →
  (∃ A B : ℝ × ℝ, parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
    line A.1 A.2 ∧ line B.1 B.2 ∧
    A ≠ B ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 64)) :=
by
  sorry

end length_of_AB_l1579_157966


namespace certain_number_value_l1579_157914

theorem certain_number_value (x : ℕ) (p n : ℕ) (hp : Nat.Prime p) (hx : x = 44) (h : x / (n * p) = 2) : n = 2 := 
by
  sorry

end certain_number_value_l1579_157914


namespace strictly_increasing_interval_l1579_157945

def f (x : ℝ) : ℝ := -x^3 + 3*x + 2

theorem strictly_increasing_interval : { x : ℝ | -1 < x ∧ x < 1 } = { x : ℝ | -3 * (x + 1) * (x - 1) > 0 } :=
sorry

end strictly_increasing_interval_l1579_157945


namespace pencil_length_eq_eight_l1579_157955

theorem pencil_length_eq_eight (L : ℝ) 
  (h1 : (1/8) * L + (1/2) * ((7/8) * L) + (7/2) = L) : 
  L = 8 :=
by
  sorry

end pencil_length_eq_eight_l1579_157955


namespace range_of_m_l1579_157946

theorem range_of_m (f : ℝ → ℝ) (h_decreasing : ∀ x y : ℝ, x < y → y < 0 → f y < f x) (h_cond : ∀ m : ℝ, f (1 - m) < f (m - 3)) : ∀ m, 1 < m ∧ m < 2 :=
by
  intros m
  sorry

end range_of_m_l1579_157946


namespace problem_statement_l1579_157921

variable (p q r s : ℝ) (ω : ℂ)

theorem problem_statement (hp : p ≠ -1) (hq : q ≠ -1) (hr : r ≠ -1) (hs : s ≠ -1) 
  (hω : ω ^ 4 = 1) (hω_ne : ω ≠ 1)
  (h_eq : (1 / (p + ω) + 1 / (q + ω) + 1 / (r + ω) + 1 / (s + ω)) = 3 / ω^2) :
  1 / (p + 1) + 1 / (q + 1) + 1 / (r + 1) + 1 / (s + 1) = 3 := 
by sorry

end problem_statement_l1579_157921


namespace assign_questions_to_students_l1579_157992

theorem assign_questions_to_students:
  ∃ (assignment : Fin 20 → Fin 20), 
  (∀ s : Fin 20, ∃ q1 q2 : Fin 20, (assignment s = q1 ∨ assignment s = q2) ∧ q1 ≠ q2 ∧ ∀ q : Fin 20, ∃ s1 s2 : Fin 20, (assignment s1 = q ∧ assignment s2 = q) ∧ s1 ≠ s2) :=
by
  sorry

end assign_questions_to_students_l1579_157992


namespace neighbors_receive_mangoes_l1579_157957

-- Definitions of the conditions
def harvested_mangoes : ℕ := 560
def sold_mangoes : ℕ := harvested_mangoes / 2
def given_to_family : ℕ := 50
def num_neighbors : ℕ := 12

-- Calculation of mangoes left
def mangoes_left : ℕ := harvested_mangoes - sold_mangoes - given_to_family

-- The statement we want to prove
theorem neighbors_receive_mangoes : mangoes_left / num_neighbors = 19 := by
  sorry

end neighbors_receive_mangoes_l1579_157957


namespace sin_double_angle_l1579_157960

-- Define the conditions and the goal
theorem sin_double_angle (α : ℝ) (h : Real.cos (π / 4 - α) = 3 / 5) : Real.sin (2 * α) = -7 / 25 :=
sorry

end sin_double_angle_l1579_157960


namespace ken_gets_back_16_dollars_l1579_157981

-- Given constants and conditions
def price_per_pound_steak : ℕ := 7
def pounds_of_steak : ℕ := 2
def price_carton_eggs : ℕ := 3
def price_gallon_milk : ℕ := 4
def price_pack_bagels : ℕ := 6
def bill_20_dollar : ℕ := 20
def bill_10_dollar : ℕ := 10
def bill_5_dollar_count : ℕ := 2
def coin_1_dollar_count : ℕ := 3

-- Calculate total cost of items
def total_cost_items : ℕ :=
  (pounds_of_steak * price_per_pound_steak) +
  price_carton_eggs +
  price_gallon_milk +
  price_pack_bagels

-- Calculate total amount paid
def total_amount_paid : ℕ :=
  bill_20_dollar +
  bill_10_dollar +
  (bill_5_dollar_count * 5) +
  (coin_1_dollar_count * 1)

-- Theorem statement to be proved
theorem ken_gets_back_16_dollars :
  total_amount_paid - total_cost_items = 16 := by
  sorry

end ken_gets_back_16_dollars_l1579_157981


namespace correct_order_shopping_process_l1579_157952

/-- Definition of each step --/
def step1 : String := "The buyer logs into the Taobao website to select products."
def step2 : String := "The buyer selects the product, clicks the buy button, and pays through Alipay."
def step3 : String := "Upon receiving the purchase information, the seller ships the goods to the buyer through a logistics company."
def step4 : String := "The buyer receives the goods, inspects them for any issues, and confirms receipt online."
def step5 : String := "After receiving the buyer's confirmation of receipt, the Taobao website transfers the payment from Alipay to the seller."

/-- The correct sequence of steps --/
def correct_sequence : List String := [
  "The buyer logs into the Taobao website to select products.",
  "The buyer selects the product, clicks the buy button, and pays through Alipay.",
  "Upon receiving the purchase information, the seller ships the goods to the buyer through a logistics company.",
  "The buyer receives the goods, inspects them for any issues, and confirms receipt online.",
  "After receiving the buyer's confirmation of receipt, the Taobao website transfers the payment from Alipay to the seller."
]

theorem correct_order_shopping_process :
  [step1, step2, step3, step4, step5] = correct_sequence :=
by
  sorry

end correct_order_shopping_process_l1579_157952


namespace student_ratio_l1579_157989

theorem student_ratio (total_students below_eight eight_years above_eight : ℕ) 
  (h1 : below_eight = total_students * 20 / 100) 
  (h2 : eight_years = 72) 
  (h3 : total_students = 150) 
  (h4 : total_students = below_eight + eight_years + above_eight) :
  (above_eight / eight_years) = 2 / 3 :=
by
  sorry

end student_ratio_l1579_157989


namespace school_boys_count_l1579_157997

theorem school_boys_count (B G : ℕ) (h1 : B + G = 1150) (h2 : G = (B / 1150) * 100) : B = 1058 := 
by 
  sorry

end school_boys_count_l1579_157997


namespace percentage_increase_decrease_exceeds_original_l1579_157930

open Real

theorem percentage_increase_decrease_exceeds_original (p q M : ℝ) (hp : 0 < p) (hq1 : 0 < q) (hq2 : q < 100) (hM : 0 < M) :
  (M * (1 + p / 100) * (1 - q / 100) > M) ↔ (p > (100 * q) / (100 - q)) :=
by
  sorry

end percentage_increase_decrease_exceeds_original_l1579_157930


namespace minimum_value_amgm_l1579_157978

theorem minimum_value_amgm (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 27) : (a + 3 * b + 9 * c) ≥ 27 :=
by
  sorry

end minimum_value_amgm_l1579_157978


namespace geom_prog_identity_l1579_157902

-- Define that A, B, C are the n-th, p-th, and k-th terms respectively of the same geometric progression.
variables (a r : ℝ) (n p k : ℕ) (A B C : ℝ)

-- Assume A = ar^(n-1), B = ar^(p-1), C = ar^(k-1)
def isGP (a r : ℝ) (n p k : ℕ) (A B C : ℝ) : Prop :=
  A = a * r^(n-1) ∧ B = a * r^(p-1) ∧ C = a * r^(k-1)

-- Define the statement to be proved
theorem geom_prog_identity (h : isGP a r n p k A B C) : A^(p-k) * B^(k-n) * C^(n-p) = 1 :=
sorry

end geom_prog_identity_l1579_157902


namespace find_N_l1579_157907

theorem find_N (N : ℕ) (h_pos : N > 0) (h_small_factors : 1 + 3 = 4) 
  (h_large_factors : N + N / 3 = 204) : N = 153 :=
  by sorry

end find_N_l1579_157907


namespace ratio_of_smaller_to_bigger_l1579_157958

theorem ratio_of_smaller_to_bigger (S B : ℕ) (h_bigger: B = 104) (h_sum: S + B = 143) :
  S / B = 39 / 104 := sorry

end ratio_of_smaller_to_bigger_l1579_157958


namespace probability_different_colors_l1579_157922

def total_chips : ℕ := 12

def blue_chips : ℕ := 5
def yellow_chips : ℕ := 3
def red_chips : ℕ := 4

def prob_diff_color (x y : ℕ) : ℚ :=
(x / total_chips) * (y / total_chips) + (y / total_chips) * (x / total_chips)

theorem probability_different_colors :
  prob_diff_color blue_chips yellow_chips +
  prob_diff_color blue_chips red_chips +
  prob_diff_color yellow_chips red_chips = 47 / 72 := by
sorry

end probability_different_colors_l1579_157922


namespace box_volume_l1579_157973

variable (l w h : ℝ)
variable (lw_eq : l * w = 30)
variable (wh_eq : w * h = 40)
variable (lh_eq : l * h = 12)

theorem box_volume : l * w * h = 120 := by
  sorry

end box_volume_l1579_157973


namespace polygon_sides_l1579_157954

open Real

theorem polygon_sides (n : ℕ) : 
  (∀ (angle : ℝ), angle = 40 → n * angle = 360) → n = 9 := by
  intro h
  have h₁ := h 40 rfl
  sorry

end polygon_sides_l1579_157954


namespace total_cost_at_discount_l1579_157938

-- Definitions for conditions
def original_price_notebook : ℕ := 15
def original_price_planner : ℕ := 10
def discount_rate : ℕ := 20
def number_of_notebooks : ℕ := 4
def number_of_planners : ℕ := 8

-- Theorem statement for the proof
theorem total_cost_at_discount :
  let discounted_price_notebook := original_price_notebook - (original_price_notebook * discount_rate / 100)
  let discounted_price_planner := original_price_planner - (original_price_planner * discount_rate / 100)
  let total_cost := (number_of_notebooks * discounted_price_notebook) + (number_of_planners * discounted_price_planner)
  total_cost = 112 :=
by
  sorry

end total_cost_at_discount_l1579_157938


namespace coeff_b_l1579_157969

noncomputable def g (a b c d e : ℝ) (x : ℝ) :=
  a * x^4 + b * x^3 + c * x^2 + d * x + e

theorem coeff_b (a b c d e : ℝ):
  -- The function g(x) has roots at x = -1, 0, 1, 2
  (g a b c d e (-1) = 0) →
  (g a b c d e 0 = 0) →
  (g a b c d e 1 = 0) →
  (g a b c d e 2 = 0) →
  -- The function passes through the point (0, 3)
  (g a b c d e 0 = 3) →
  -- Assuming a = 1
  (a = 1) →
  -- Prove that b = -2
  b = -2 :=
by
  intros _ _ _ _ _ a_eq_1
  -- Proof omitted
  sorry

end coeff_b_l1579_157969


namespace find_q_l1579_157984

noncomputable def p (q : ℝ) : ℝ := 16 / (3 * q)

theorem find_q (p q : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : 1/p + 1/q = 3/2) (h4 : p * q = 16/3) : q = 24 / 6 + 19.6 / 6 :=
by
  sorry

end find_q_l1579_157984


namespace orthogonal_vectors_y_value_l1579_157991

theorem orthogonal_vectors_y_value (y : ℝ) :
  (3 : ℝ) * (-1) + (4 : ℝ) * y = 0 → y = 3 / 4 :=
by
  sorry

end orthogonal_vectors_y_value_l1579_157991


namespace abs_neg_seventeen_l1579_157988

theorem abs_neg_seventeen : |(-17 : ℤ)| = 17 := by
  sorry

end abs_neg_seventeen_l1579_157988


namespace intersection_of_sets_l1579_157944

noncomputable def setA : Set ℝ := {x | 1 / (x - 1) ≤ 1}
def setB : Set ℝ := {-1, 0, 1, 2}

theorem intersection_of_sets : setA ∩ setB = {-1, 0, 2} := 
by
  sorry

end intersection_of_sets_l1579_157944


namespace cat_finishes_food_on_tuesday_second_week_l1579_157962

def initial_cans : ℚ := 8
def extra_treat : ℚ := 1 / 6
def morning_diet : ℚ := 1 / 4
def evening_diet : ℚ := 1 / 5

def daily_consumption (morning_diet evening_diet : ℚ) : ℚ :=
  morning_diet + evening_diet

def first_day_consumption (daily_consumption extra_treat : ℚ) : ℚ :=
  daily_consumption + extra_treat

theorem cat_finishes_food_on_tuesday_second_week 
  (initial_cans extra_treat morning_diet evening_diet : ℚ)
  (h1 : initial_cans = 8)
  (h2 : extra_treat = 1 / 6)
  (h3 : morning_diet = 1 / 4)
  (h4 : evening_diet = 1 / 5) :
  -- The computation must be performed here or defined previously
  -- The proof of this theorem is the task, the result is postulated as a theorem
  final_day = "Tuesday (second week)" :=
sorry

end cat_finishes_food_on_tuesday_second_week_l1579_157962


namespace same_school_probability_l1579_157980

theorem same_school_probability :
  let total_teachers : ℕ := 6
  let teachers_from_school_A : ℕ := 3
  let teachers_from_school_B : ℕ := 3
  let ways_to_choose_2_from_6 : ℕ := Nat.choose total_teachers 2
  let ways_to_choose_2_from_A := Nat.choose teachers_from_school_A 2
  let ways_to_choose_2_from_B := Nat.choose teachers_from_school_B 2
  let same_school_ways : ℕ := ways_to_choose_2_from_A + ways_to_choose_2_from_B
  let probability := (same_school_ways : ℚ) / ways_to_choose_2_from_6 
  probability = (2 : ℚ) / (5 : ℚ) := by sorry

end same_school_probability_l1579_157980


namespace mitya_age_l1579_157983

-- Definitions of the ages
variables (M S : ℕ)

-- Conditions based on the problem statements
axiom condition1 : M = S + 11
axiom condition2 : S = 2 * (S - (M - S))

-- The theorem stating that Mitya is 33 years old
theorem mitya_age : M = 33 :=
by
  -- Outline the proof
  sorry

end mitya_age_l1579_157983


namespace total_amount_l1579_157985

theorem total_amount (a b c : ℕ) (h1 : a * 5 = b * 3) (h2 : c * 5 = b * 9) (h3 : b = 50) :
  a + b + c = 170 := by
  sorry

end total_amount_l1579_157985


namespace circle_center_sum_l1579_157917

/-- Given the equation of a circle, prove that the sum of the x and y coordinates of the center is -1. -/
theorem circle_center_sum (x y : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = 4 * x - 6 * y + 9) → x + y = -1 :=
by 
  sorry

end circle_center_sum_l1579_157917


namespace sum_of_permutations_of_1234567_l1579_157909

theorem sum_of_permutations_of_1234567 : 
  let factorial_7 := 5040
  let sum_of_digits := 1 + 2 + 3 + 4 + 5 + 6 + 7
  let geometric_series_sum := (10 ^ 7 - 1) / (10 - 1)
  sum_of_digits * factorial_7 * geometric_series_sum = 22399997760 :=
by
  let factorial_7 := 5040
  let sum_of_digits := 1 + 2 + 3 + 4 + 5 + 6 + 7
  let geometric_series_sum := (10^7 - 1) / (10 - 1)
  sorry

end sum_of_permutations_of_1234567_l1579_157909


namespace maximize_probability_sum_is_15_l1579_157901

def initial_list : List ℤ := [-1, 0, 1, 2, 3, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16]

def valid_pairs (lst : List ℤ) : List (ℤ × ℤ) :=
  (lst.product lst).filter (λ ⟨x, y⟩ => x < y ∧ x + y = 15)

def remove_one_element (lst : List ℤ) (x : ℤ) : List ℤ :=
  lst.erase x

theorem maximize_probability_sum_is_15 :
  (List.length (valid_pairs (remove_one_element initial_list 8))
   = List.maximum (List.map (λ x => List.length (valid_pairs (remove_one_element initial_list x))) initial_list)) :=
sorry

end maximize_probability_sum_is_15_l1579_157901


namespace find_x_sq_add_y_sq_l1579_157927

theorem find_x_sq_add_y_sq (x y : ℝ) (h1 : (x + y) ^ 2 = 36) (h2 : x * y = 10) : x ^ 2 + y ^ 2 = 16 :=
by
  sorry

end find_x_sq_add_y_sq_l1579_157927


namespace base7_to_base10_l1579_157936

theorem base7_to_base10 : 6 * 7^3 + 4 * 7^2 + 2 * 7^1 + 3 * 7^0 = 2271 := by
  sorry

end base7_to_base10_l1579_157936


namespace solution1_solution2_solution3_solution4_solution5_l1579_157928

noncomputable def problem1 : ℤ :=
  -3 + 8 - 15 - 6

theorem solution1 : problem1 = -16 := by
  sorry

noncomputable def problem2 : ℚ :=
  -35 / -7 * (-1 / 7)

theorem solution2 : problem2 = -(5 / 7) := by
  sorry

noncomputable def problem3 : ℤ :=
  -2^2 - |2 - 5| / -3

theorem solution3 : problem3 = -3 := by
  sorry

noncomputable def problem4 : ℚ :=
  (1 / 2 + 5 / 6 - 7 / 12) * -24 

theorem solution4 : problem4 = -18 := by
  sorry

noncomputable def problem5 : ℚ :=
  (-99 - 6 / 11) * 22

theorem solution5 : problem5 = -2190 := by
  sorry

end solution1_solution2_solution3_solution4_solution5_l1579_157928


namespace animals_consuming_hay_l1579_157950

-- Define the rate of consumption for each animal
def rate_goat : ℚ := 1 / 6 -- goat consumes 1 cartload per 6 weeks
def rate_sheep : ℚ := 1 / 8 -- sheep consumes 1 cartload per 8 weeks
def rate_cow : ℚ := 1 / 3 -- cow consumes 1 cartload per 3 weeks

-- Define the number of animals
def num_goats : ℚ := 5
def num_sheep : ℚ := 3
def num_cows : ℚ := 2

-- Define the total rate of consumption
def total_rate : ℚ := (num_goats * rate_goat) + (num_sheep * rate_sheep) + (num_cows * rate_cow)

-- Define the total amount of hay to be consumed
def total_hay : ℚ := 30

-- Define the time required to consume the total hay at the calculated rate
def time_required : ℚ := total_hay / total_rate

-- Theorem stating the time required to consume 30 cartloads of hay is 16 weeks.
theorem animals_consuming_hay : time_required = 16 := by
  sorry

end animals_consuming_hay_l1579_157950


namespace tangent_line_parabola_l1579_157931

theorem tangent_line_parabola (d : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) → d = 1 :=
by
  sorry

end tangent_line_parabola_l1579_157931


namespace value_of_m_l1579_157943

theorem value_of_m (m : ℝ) :
  let A := {2, 3}
  let B := {x : ℝ | m * x - 6 = 0}
  (B ⊆ A) → (m = 0 ∨ m = 2 ∨ m = 3) :=
by
  intros A B h
  sorry

end value_of_m_l1579_157943


namespace parallel_lines_slope_l1579_157919

theorem parallel_lines_slope (a : ℝ) :
  (∃ (a : ℝ), ∀ x y, (3 * y - a = 9 * x + 1) ∧ (y - 2 = (2 * a - 3) * x)) → a = 3 :=
by
  sorry

end parallel_lines_slope_l1579_157919


namespace no_boys_love_cards_l1579_157970

def boys_love_marbles := 13
def total_marbles := 26
def marbles_per_boy := 2

theorem no_boys_love_cards (boys_love_marbles total_marbles marbles_per_boy : ℕ)
  (h1 : boys_love_marbles * marbles_per_boy = total_marbles) : 
  ∃ no_boys_love_cards : ℕ, no_boys_love_cards = 0 :=
by
  sorry

end no_boys_love_cards_l1579_157970


namespace total_cost_correct_l1579_157956

-- Definitions of the constants based on given problem conditions
def cost_burger : ℕ := 5
def cost_pack_of_fries : ℕ := 2
def num_packs_of_fries : ℕ := 2
def cost_salad : ℕ := 3 * cost_pack_of_fries

-- The total cost calculation based on the conditions
def total_cost : ℕ := cost_burger + num_packs_of_fries * cost_pack_of_fries + cost_salad

-- The statement to prove that the total cost Benjamin paid is $15
theorem total_cost_correct : total_cost = 15 := by
  -- This is where the proof would go, but we're omitting it for now.
  sorry

end total_cost_correct_l1579_157956


namespace triathlon_bike_speed_l1579_157982

theorem triathlon_bike_speed :
  ∀ (t_total t_swim t_run t_bike : ℚ) (d_swim d_run d_bike : ℚ)
    (v_swim v_run r_bike : ℚ),
  t_total = 3 →
  d_swim = 1 / 2 →
  v_swim = 1 →
  d_run = 4 →
  v_run = 5 →
  d_bike = 10 →
  t_swim = d_swim / v_swim →
  t_run = d_run / v_run →
  t_bike = t_total - (t_swim + t_run) →
  r_bike = d_bike / t_bike →
  r_bike = 100 / 17 :=
by
  intros t_total t_swim t_run t_bike d_swim d_run d_bike v_swim v_run r_bike
         h_total h_d_swim h_v_swim h_d_run h_v_run h_d_bike h_t_swim h_t_run h_t_bike h_r_bike
  sorry

end triathlon_bike_speed_l1579_157982


namespace CarmenBrushLengthInCentimeters_l1579_157939

-- Given conditions
def CarlaBrushLengthInInches : ℝ := 12
def CarmenBrushPercentIncrease : ℝ := 0.5
def InchToCentimeterConversionFactor : ℝ := 2.5

-- Question: What is Carmen's brush length in centimeters?
-- Proof Goal: Prove that Carmen's brush length in centimeters is 45 cm.
theorem CarmenBrushLengthInCentimeters :
  let CarmenBrushLengthInInches := CarlaBrushLengthInInches * (1 + CarmenBrushPercentIncrease)
  CarmenBrushLengthInInches * InchToCentimeterConversionFactor = 45 := by
  -- sorry is used as a placeholder for the completed proof
  sorry

end CarmenBrushLengthInCentimeters_l1579_157939


namespace sets_equal_sufficient_condition_l1579_157937

variable (a : ℝ)

-- Define sets A and B
def A (x : ℝ) : Prop := 0 < a * x + 1 ∧ a * x + 1 ≤ 5
def B (x : ℝ) : Prop := -1/2 < x ∧ x ≤ 2

-- Statement for Part 1: Sets A and B can be equal if and only if a = 2
theorem sets_equal (h : ∀ x, A a x ↔ B x) : a = 2 :=
sorry

-- Statement for Part 2: Proposition p ⇒ q holds if and only if a > 2 or a < -8
theorem sufficient_condition (h : ∀ x, A a x → B x) (h_neq : ∃ x, B x ∧ ¬A a x) : a > 2 ∨ a < -8 :=
sorry

end sets_equal_sufficient_condition_l1579_157937


namespace real_condition_proof_l1579_157908

noncomputable def real_condition_sufficient_but_not_necessary : Prop := 
∀ x : ℝ, (|x - 2| < 1) → ((x^2 + x - 2) > 0) ∧ (¬ ( ∀ y : ℝ, (y^2 + y - 2) > 0 → |y - 2| < 1))

theorem real_condition_proof : real_condition_sufficient_but_not_necessary :=
by
  sorry

end real_condition_proof_l1579_157908


namespace team_total_games_123_l1579_157951

theorem team_total_games_123 {G : ℕ} 
  (h1 : (55 / 100) * 35 + (90 / 100) * (G - 35) = (80 / 100) * G) : 
  G = 123 :=
sorry

end team_total_games_123_l1579_157951


namespace inequality_holds_l1579_157959

theorem inequality_holds (x : ℝ) (hx : 0 < x ∧ x < 4) :
  ∀ y : ℝ, y > 0 → (4 * (x * y^2 + x^2 * y + 4 * y^2 + 4 * x * y) / (x + y) > 3 * x^2 * y) :=
by
  intros y hy_gt_zero
  sorry

end inequality_holds_l1579_157959


namespace pascal_fifth_number_l1579_157965

def binom (n k : Nat) : Nat := Nat.choose n k

theorem pascal_fifth_number (n r : Nat) (h1 : n = 50) (h2 : r = 4) : binom n r = 230150 := by
  sorry

end pascal_fifth_number_l1579_157965


namespace area_comparison_perimeter_comparison_l1579_157923

-- Define side length of square and transformation to sides of the rectangle
variable (a : ℝ)

-- Conditions: side lengths of the rectangle relative to the square
def long_side : ℝ := 1.11 * a
def short_side : ℝ := 0.9 * a

-- Area calculations and comparison
def square_area : ℝ := a^2
def rectangle_area : ℝ := long_side a * short_side a

theorem area_comparison : (rectangle_area a / square_area a) = 0.999 := by
  sorry

-- Perimeter calculations and comparison
def square_perimeter : ℝ := 4 * a
def rectangle_perimeter : ℝ := 2 * (long_side a + short_side a)

theorem perimeter_comparison : (rectangle_perimeter a / square_perimeter a) = 1.005 := by
  sorry

end area_comparison_perimeter_comparison_l1579_157923


namespace cone_lateral_surface_area_ratio_l1579_157949

/-- Let a be the side length of the equilateral triangle front view of a cone.
    The base area of the cone is (π * (a / 2)^2).
    The lateral surface area of the cone is (π * (a / 2) * a).
    We want to show that the ratio of the lateral surface area to the base area is 2.
 -/
theorem cone_lateral_surface_area_ratio 
  (a : ℝ) 
  (base_area : ℝ := π * (a / 2)^2) 
  (lateral_surface_area : ℝ := π * (a / 2) * a) 
  : lateral_surface_area / base_area = 2 :=
by
  sorry

end cone_lateral_surface_area_ratio_l1579_157949


namespace change_after_buying_tickets_l1579_157941

def cost_per_ticket := 8
def number_of_tickets := 2
def total_money := 25

theorem change_after_buying_tickets :
  total_money - number_of_tickets * cost_per_ticket = 9 := by
  sorry

end change_after_buying_tickets_l1579_157941


namespace smallest_x_abs_eq_15_l1579_157910

theorem smallest_x_abs_eq_15 :
  ∃ x : ℝ, |5 * x - 3| = 15 ∧ ∀ y : ℝ, |5 * y - 3| = 15 → x ≤ y :=
sorry

end smallest_x_abs_eq_15_l1579_157910


namespace prime_dates_in_2008_l1579_157996

def is_prime (n : ℕ) : Prop := Nat.Prime n

def is_prime_date (month day : ℕ) : Prop := is_prime month ∧ is_prime day

noncomputable def prime_dates_2008 : ℕ :=
  let prime_months := [2, 3, 5, 7, 11]
  let prime_days_31 := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let prime_days_30 := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
  let prime_days_29 := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
  prime_months.foldl (λ acc month => 
    acc + match month with
      | 2 => List.length prime_days_29
      | 3 | 5 | 7 => List.length prime_days_31
      | 11 => List.length prime_days_30
      | _ => 0
    ) 0

theorem prime_dates_in_2008 : 
  prime_dates_2008 = 53 :=
  sorry

end prime_dates_in_2008_l1579_157996


namespace sweaters_to_wash_l1579_157932

theorem sweaters_to_wash (pieces_per_load : ℕ) (total_loads : ℕ) (shirts_to_wash : ℕ) 
  (h1 : pieces_per_load = 5) (h2 : total_loads = 9) (h3 : shirts_to_wash = 43) : ℕ :=
  if total_loads * pieces_per_load - shirts_to_wash = 2 then 2 else 0

end sweaters_to_wash_l1579_157932


namespace student_number_in_eighth_group_l1579_157968

-- Definitions corresponding to each condition
def students : ℕ := 50
def group_size : ℕ := 5
def third_group_student_number : ℕ := 12
def kth_group_number (k : ℕ) (n : ℕ) : ℕ := n + (k - 3) * group_size

-- Main statement to prove
theorem student_number_in_eighth_group :
  kth_group_number 8 third_group_student_number = 37 :=
  by
  sorry

end student_number_in_eighth_group_l1579_157968


namespace set_equality_example_l1579_157998

theorem set_equality_example : {x : ℕ | 2 * x + 3 ≥ 3 * x} = {0, 1, 2, 3} := by
  sorry

end set_equality_example_l1579_157998


namespace greatest_of_consecutive_integers_with_sum_39_l1579_157961

theorem greatest_of_consecutive_integers_with_sum_39 :
  ∃ x : ℤ, x + (x + 1) + (x + 2) = 39 ∧ max (max x (x + 1)) (x + 2) = 14 :=
by
  sorry

end greatest_of_consecutive_integers_with_sum_39_l1579_157961


namespace equation_infinitely_many_solutions_iff_b_eq_neg9_l1579_157935

theorem equation_infinitely_many_solutions_iff_b_eq_neg9 (b : ℤ) :
  (∀ x : ℤ, 5 * (3 * x - b) = 3 * (5 * x + 15)) ↔ b = -9 := 
by
  sorry

end equation_infinitely_many_solutions_iff_b_eq_neg9_l1579_157935


namespace strawberries_left_l1579_157967

theorem strawberries_left (initial : ℝ) (eaten : ℝ) (remaining : ℝ) : initial = 78.0 → eaten = 42.0 → remaining = 36.0 → initial - eaten = remaining :=
by
  sorry

end strawberries_left_l1579_157967


namespace union_of_A_B_l1579_157916

open Set

def A := {x : ℝ | -1 < x ∧ x < 2}
def B := {x : ℝ | -3 < x ∧ x ≤ 1}

theorem union_of_A_B : A ∪ B = {x : ℝ | -3 < x ∧ x < 2} :=
by sorry

end union_of_A_B_l1579_157916


namespace train_cross_first_platform_in_15_seconds_l1579_157926

noncomputable def length_of_train : ℝ := 100
noncomputable def length_of_second_platform : ℝ := 500
noncomputable def time_to_cross_second_platform : ℝ := 20
noncomputable def length_of_first_platform : ℝ := 350
noncomputable def speed_of_train := (length_of_train + length_of_second_platform) / time_to_cross_second_platform
noncomputable def time_to_cross_first_platform := (length_of_train + length_of_first_platform) / speed_of_train

theorem train_cross_first_platform_in_15_seconds : time_to_cross_first_platform = 15 := by
  sorry

end train_cross_first_platform_in_15_seconds_l1579_157926


namespace a_minus_b_perfect_square_l1579_157979

theorem a_minus_b_perfect_square (a b : ℕ) (h : 2 * a^2 + a = 3 * b^2 + b) (ha : 0 < a) (hb : 0 < b) :
  ∃ k : ℕ, a - b = k^2 :=
by sorry

end a_minus_b_perfect_square_l1579_157979


namespace hamburger_per_meatball_l1579_157964

theorem hamburger_per_meatball (family_members : ℕ) (total_hamburger : ℕ) (antonio_meatballs : ℕ) 
    (hmembers : family_members = 8)
    (hhamburger : total_hamburger = 4)
    (hantonio : antonio_meatballs = 4) : 
    (total_hamburger : ℝ) / (family_members * antonio_meatballs) = 0.125 := 
by
  sorry

end hamburger_per_meatball_l1579_157964


namespace hypotenuse_intersection_incircle_diameter_l1579_157995

/-- Let \( a \) and \( b \) be the legs of a right triangle with hypotenuse \( c \). 
    Let two circles be centered at the endpoints of the hypotenuse, with radii \( a \) and \( b \). 
    Prove that the segment of the hypotenuse that lies in the intersection of the two circles is equal in length to the diameter of the incircle of the triangle. -/
theorem hypotenuse_intersection_incircle_diameter (a b : ℝ) :
    let c := Real.sqrt (a^2 + b^2)
    let x := a + b - c
    let r := (a + b - c) / 2
    x = 2 * r :=
by
  let c := Real.sqrt (a^2 + b^2)
  let x := a + b - c
  let r := (a + b - c) / 2
  show x = 2 * r
  sorry

end hypotenuse_intersection_incircle_diameter_l1579_157995


namespace square_side_increase_l1579_157915

theorem square_side_increase (p : ℝ) (h : (1 + p / 100)^2 = 1.69) : p = 30 :=
by {
  sorry
}

end square_side_increase_l1579_157915


namespace set_B_equals_1_4_l1579_157900

open Set

def U : Set ℕ := {1, 2, 3, 4}
def C_U_B : Set ℕ := {2, 3}

theorem set_B_equals_1_4 : 
  ∃ B : Set ℕ, B = {1, 4} ∧ U \ B = C_U_B := by
  sorry

end set_B_equals_1_4_l1579_157900


namespace Z_divisible_by_10001_l1579_157999

def is_eight_digit_integer (Z : Nat) : Prop :=
  (10^7 ≤ Z) ∧ (Z < 10^8)

def first_four_equal_last_four (Z : Nat) : Prop :=
  ∃ (a b c d : Nat), a ≠ 0 ∧ (Z = 1001 * (1000 * a + 100 * b + 10 * c + d))

theorem Z_divisible_by_10001 (Z : Nat) (h1 : is_eight_digit_integer Z) (h2 : first_four_equal_last_four Z) : 
  10001 ∣ Z :=
sorry

end Z_divisible_by_10001_l1579_157999


namespace train_B_time_to_destination_l1579_157942

theorem train_B_time_to_destination (speed_A : ℕ) (time_A : ℕ) (speed_B : ℕ) (dA : ℕ) :
  speed_A = 100 ∧ time_A = 9 ∧ speed_B = 150 ∧ dA = speed_A * time_A →
  dA / speed_B = 6 := 
by
  sorry

end train_B_time_to_destination_l1579_157942


namespace value_of_x_is_two_l1579_157994

theorem value_of_x_is_two (x : ℝ) (h : x + x^3 = 10) : x = 2 :=
sorry

end value_of_x_is_two_l1579_157994


namespace slope_angle_of_perpendicular_line_l1579_157903

theorem slope_angle_of_perpendicular_line (l : ℝ → ℝ) (h_perp : ∀ x y : ℝ, l x = y ↔ x - y - 1 = 0) : ∃ α : ℝ, α = 135 :=
by
  sorry

end slope_angle_of_perpendicular_line_l1579_157903


namespace only_set_C_is_pythagorean_triple_l1579_157987

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem only_set_C_is_pythagorean_triple :
  ¬ is_pythagorean_triple 3 4 7 ∧
  ¬ is_pythagorean_triple 15 20 25 ∧
  is_pythagorean_triple 5 12 13 ∧
  ¬ is_pythagorean_triple 1 3 5 :=
by {
  -- Proof goes here
  sorry
}

end only_set_C_is_pythagorean_triple_l1579_157987


namespace julia_height_is_172_7_cm_l1579_157948

def julia_height_in_cm (height_in_inches : ℝ) (conversion_factor : ℝ) : ℝ :=
  height_in_inches * conversion_factor

theorem julia_height_is_172_7_cm :
  julia_height_in_cm 68 2.54 = 172.7 :=
by
  sorry

end julia_height_is_172_7_cm_l1579_157948


namespace triangle_bisector_ratio_l1579_157986

theorem triangle_bisector_ratio (AB BC CA : ℝ) (h_AB_pos : 0 < AB) (h_BC_pos : 0 < BC) (h_CA_pos : 0 < CA)
  (AA1_bisector : True) (BB1_bisector : True) (O_intersection : True) : 
  AA1 / OA1 = 3 :=
by
  sorry

end triangle_bisector_ratio_l1579_157986


namespace simplify_expression_l1579_157918

variable (y : ℤ)

theorem simplify_expression : 5 * y + 7 * y - 3 * y = 9 * y := by
  sorry

end simplify_expression_l1579_157918


namespace pats_stick_covered_l1579_157934

/-
Assumptions:
1. Pat's stick is 30 inches long.
2. Jane's stick is 22 inches long.
3. Jane’s stick is two feet (24 inches) shorter than Sarah’s stick.
4. The portion of Pat's stick not covered in dirt is half as long as Sarah’s stick.

Prove that the length of Pat's stick covered in dirt is 7 inches.
-/

theorem pats_stick_covered  (pat_stick_len : ℕ) (jane_stick_len : ℕ) (jane_sarah_diff : ℕ) (pat_not_covered_by_dirt : ℕ) :
  pat_stick_len = 30 → jane_stick_len = 22 → jane_sarah_diff = 24 → pat_not_covered_by_dirt * 2 = jane_stick_len + jane_sarah_diff → 
    (pat_stick_len - pat_not_covered_by_dirt) = 7 :=
by
  intros h1 h2 h3 h4
  sorry

end pats_stick_covered_l1579_157934


namespace range_of_a_l1579_157913

def p (a : ℝ) : Prop := 0 < a ∧ a < 1
def q (a : ℝ) : Prop := a > 1 / 8

def resolution (a : ℝ) : Prop :=
(p a ∨ q a) ∧ ¬(p a ∧ q a) → (0 < a ∧ a ≤ 1 / 8) ∨ a ≥ 1

theorem range_of_a (a : ℝ) : resolution a := sorry

end range_of_a_l1579_157913


namespace center_of_circle_l1579_157925

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop := 4 * x^2 - 8 * x + 4 * y^2 - 24 * y - 36 = 0

-- Define what it means to be the center of the circle, which is (h, k)
def is_center (h k : ℝ) : Prop :=
  ∀ (x y : ℝ), circle_eq x y ↔ (x - h)^2 + (y - k)^2 = 1

-- The statement that we need to prove
theorem center_of_circle : is_center 1 3 :=
sorry

end center_of_circle_l1579_157925


namespace sequence_a10_l1579_157972

theorem sequence_a10 (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, a (n+1) - a n = 1 / (4 * ↑n^2 - 1)) :
  a 10 = 28 / 19 :=
by
  sorry

end sequence_a10_l1579_157972


namespace pacific_ocean_area_rounded_l1579_157933

def pacific_ocean_area : ℕ := 19996800

def ten_thousand : ℕ := 10000

noncomputable def pacific_ocean_area_in_ten_thousands (area : ℕ) : ℕ :=
  (area / ten_thousand + if (area % ten_thousand) >= (ten_thousand / 2) then 1 else 0)

theorem pacific_ocean_area_rounded :
  pacific_ocean_area_in_ten_thousands pacific_ocean_area = 2000 :=
by
  sorry

end pacific_ocean_area_rounded_l1579_157933


namespace gcd_90_252_eq_18_l1579_157976

theorem gcd_90_252_eq_18 : Nat.gcd 90 252 = 18 := 
sorry

end gcd_90_252_eq_18_l1579_157976


namespace find_a7_l1579_157947

theorem find_a7 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ) (x : ℝ)
  (h : x^8 = a_0 + a_1 * (x + 1) + a_2 * (x + 1)^2 + a_3 * (x + 1)^3 +
            a_4 * (x + 1)^4 + a_5 * (x + 1)^5 + a_6 * (x + 1)^6 +
            a_7 * (x + 1)^7 + a_8 * (x + 1)^8) : 
  a_7 = -8 := 
sorry

end find_a7_l1579_157947


namespace infinitely_many_c_exist_l1579_157940

theorem infinitely_many_c_exist :
  ∃ c: ℕ, ∃ x y z: ℕ, (x^2 - c) * (y^2 - c) = z^2 - c ∧ (x^2 + c) * (y^2 - c) = z^2 - c :=
by
  sorry

end infinitely_many_c_exist_l1579_157940


namespace boat_travel_distance_along_stream_l1579_157905

theorem boat_travel_distance_along_stream :
  ∀ (v_s : ℝ), (5 - v_s = 2) → (5 + v_s) * 1 = 8 :=
by
  intro v_s
  intro h1
  have vs_value : v_s = 3 := by linarith
  rw [vs_value]
  norm_num

end boat_travel_distance_along_stream_l1579_157905


namespace janessa_gives_dexter_cards_l1579_157974

def initial_cards : Nat := 4
def father_cards : Nat := 13
def ordered_cards : Nat := 36
def bad_cards : Nat := 4
def kept_cards : Nat := 20

theorem janessa_gives_dexter_cards :
  initial_cards + father_cards + ordered_cards - bad_cards - kept_cards = 29 := 
by
  sorry

end janessa_gives_dexter_cards_l1579_157974


namespace range_of_a_l1579_157911

noncomputable def f (x : ℝ) : ℝ :=
  Real.exp x - Real.exp (-x) + Real.log (x + Real.sqrt (x^2 + 1))

theorem range_of_a
  (h : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → f (x^2 + 2) + f (-2 * a * x) ≥ 0) :
  -3/2 ≤ a ∧ a ≤ Real.sqrt 2 :=
sorry

end range_of_a_l1579_157911


namespace equation_solution_l1579_157993

theorem equation_solution:
  ∀ (x y : ℝ), 
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2 ↔ 
  (y = -x - 2 ∨ y = -2 * x + 1) := 
by
  sorry

end equation_solution_l1579_157993


namespace find_number_of_rabbits_l1579_157975

def total_heads (R P : ℕ) : ℕ := R + P
def total_legs (R P : ℕ) : ℕ := 4 * R + 2 * P

theorem find_number_of_rabbits (R P : ℕ)
  (h1 : total_heads R P = 60)
  (h2 : total_legs R P = 192) :
  R = 36 := by
  sorry

end find_number_of_rabbits_l1579_157975

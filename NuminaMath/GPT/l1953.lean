import Mathlib

namespace NUMINAMATH_GPT_remaining_money_l1953_195358

-- Define the conditions
def num_pies : ℕ := 200
def price_per_pie : ℕ := 20
def fraction_for_ingredients : ℚ := 3 / 5

-- Define the total sales
def total_sales : ℕ := num_pies * price_per_pie

-- Define the cost for ingredients
def cost_for_ingredients : ℚ := fraction_for_ingredients * total_sales 

-- Prove the remaining money
theorem remaining_money : (total_sales : ℚ) - cost_for_ingredients = 1600 := 
by {
  -- This is where the proof would go
  sorry
}

end NUMINAMATH_GPT_remaining_money_l1953_195358


namespace NUMINAMATH_GPT_domain_of_v_l1953_195340

noncomputable def v (x : ℝ) : ℝ := 1 / (x - 1)^(1 / 3)

theorem domain_of_v :
  {x : ℝ | ∃ y : ℝ, y ≠ 0 ∧ y = (v x)} = {x | x ≠ 1} := by
  sorry

end NUMINAMATH_GPT_domain_of_v_l1953_195340


namespace NUMINAMATH_GPT_change_calculation_l1953_195321

-- Define the initial amounts of Lee and his friend
def lee_amount : ℕ := 10
def friend_amount : ℕ := 8

-- Define the cost of items they ordered
def chicken_wings : ℕ := 6
def chicken_salad : ℕ := 4
def soda : ℕ := 1
def soda_count : ℕ := 2
def tax : ℕ := 3

-- Define the total money they initially had
def total_money : ℕ := lee_amount + friend_amount

-- Define the total cost of the food without tax
def food_cost : ℕ := chicken_wings + chicken_salad + (soda * soda_count)

-- Define the total cost including tax
def total_cost : ℕ := food_cost + tax

-- Define the change they should receive
def change : ℕ := total_money - total_cost

theorem change_calculation : change = 3 := by
  -- Note: Proof here is omitted
  sorry

end NUMINAMATH_GPT_change_calculation_l1953_195321


namespace NUMINAMATH_GPT_find_xyz_l1953_195382

theorem find_xyz (x y z : ℝ) :
  x - y + z = 2 ∧
  x^2 + y^2 + z^2 = 30 ∧
  x^3 - y^3 + z^3 = 116 →
  (x = -1 ∧ y = 2 ∧ z = 5) ∨
  (x = -1 ∧ y = -5 ∧ z = -2) ∨
  (x = -2 ∧ y = 1 ∧ z = 5) ∨
  (x = -2 ∧ y = -5 ∧ z = -1) ∨
  (x = 5 ∧ y = 1 ∧ z = -2) ∨
  (x = 5 ∧ y = 2 ∧ z = -1) := by
  sorry

end NUMINAMATH_GPT_find_xyz_l1953_195382


namespace NUMINAMATH_GPT_sum_of_squares_l1953_195330

theorem sum_of_squares (x y z : ℕ) (hx : x = 1) (hy : y = 2) (hz : z = 3) (h_sum : x * 1 + y * 2 + z * 3 = 12) : x^2 + y^2 + z^2 = 56 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l1953_195330


namespace NUMINAMATH_GPT_probability_of_region_D_l1953_195392

theorem probability_of_region_D
    (P_A : ℚ) (P_B : ℚ) (P_C : ℚ) (P_D : ℚ)
    (h1 : P_A = 1/4) 
    (h2 : P_B = 1/3) 
    (h3 : P_C = 1/6) 
    (h4 : P_A + P_B + P_C + P_D = 1) : 
    P_D = 1/4 := by
    sorry

end NUMINAMATH_GPT_probability_of_region_D_l1953_195392


namespace NUMINAMATH_GPT_reflect_point_l1953_195371

def point_reflect_across_line (m : ℝ) :=
  (6 - m, m + 1)

theorem reflect_point (m : ℝ) :
  point_reflect_across_line m = (6 - m, m + 1) :=
  sorry

end NUMINAMATH_GPT_reflect_point_l1953_195371


namespace NUMINAMATH_GPT_starting_weight_of_labrador_puppy_l1953_195323

theorem starting_weight_of_labrador_puppy :
  ∃ L : ℝ,
    (L + 0.25 * L) - (12 + 0.25 * 12) = 35 ∧ 
    L = 40 :=
by
  use 40
  sorry

end NUMINAMATH_GPT_starting_weight_of_labrador_puppy_l1953_195323


namespace NUMINAMATH_GPT_class_total_students_l1953_195394

-- Definitions based on the conditions
def number_students_group : ℕ := 12
def frequency_group : ℚ := 0.25

-- Statement of the problem in Lean
theorem class_total_students (n : ℕ) (h : frequency_group = number_students_group / n) : n = 48 :=
by
  sorry

end NUMINAMATH_GPT_class_total_students_l1953_195394


namespace NUMINAMATH_GPT_total_ranking_sequences_l1953_195393

-- Define teams
inductive Team
| A | B | C | D

-- Define the conditions
def qualifies (t : Team) : Prop := 
  -- Each team must win its qualifying match to participate
  true

def plays_saturday (t1 t2 t3 t4 : Team) : Prop :=
  (t1 = Team.A ∧ t2 = Team.B) ∨ (t3 = Team.C ∧ t4 = Team.D)

def plays_sunday (t1 t2 t3 t4 : Team) : Prop := 
  -- Winners of Saturday's matches play for 1st and 2nd, losers play for 3rd and 4th
  true

-- Lean statement for the proof problem
theorem total_ranking_sequences : 
  (∀ t : Team, qualifies t) → 
  (∀ t1 t2 t3 t4 : Team, plays_saturday t1 t2 t3 t4) → 
  (∀ t1 t2 t3 t4 : Team, plays_sunday t1 t2 t3 t4) → 
  ∃ n : ℕ, n = 16 :=
by 
  sorry

end NUMINAMATH_GPT_total_ranking_sequences_l1953_195393


namespace NUMINAMATH_GPT_binary_operation_correct_l1953_195349

theorem binary_operation_correct :
  let b1 := 0b11011
  let b2 := 0b1011
  let b3 := 0b11100
  let b4 := 0b10101
  let b5 := 0b1001
  b1 + b2 - b3 + b4 - b5 = 0b11110 := by
  sorry

end NUMINAMATH_GPT_binary_operation_correct_l1953_195349


namespace NUMINAMATH_GPT_strawberries_left_l1953_195352

theorem strawberries_left (picked: ℕ) (eaten: ℕ) (initial_count: picked = 35) (eaten_count: eaten = 2) :
  picked - eaten = 33 :=
by
  sorry

end NUMINAMATH_GPT_strawberries_left_l1953_195352


namespace NUMINAMATH_GPT_arithmetic_problem_l1953_195366

theorem arithmetic_problem : 
  let part1 := (20 / 100) * 120
  let part2 := (25 / 100) * 250
  let part3 := (15 / 100) * 80
  let sum := part1 + part2 + part3
  let subtract := (10 / 100) * 600
  sum - subtract = 38.5 := by
  sorry

end NUMINAMATH_GPT_arithmetic_problem_l1953_195366


namespace NUMINAMATH_GPT_range_of_a_l1953_195326

open Real

noncomputable def proposition_p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a * x + a > 0

noncomputable def proposition_q (a : ℝ) : Prop :=
  let Δ := 1 - 4 * a
  Δ ≥ 0

theorem range_of_a (a : ℝ) :
  ((proposition_p a ∨ proposition_q a) ∧ ¬(proposition_p a ∧ proposition_q a))
  ↔ (a ≤ 0 ∨ (1/4 : ℝ) < a ∧ a < 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1953_195326


namespace NUMINAMATH_GPT_equation_I_consecutive_integers_equation_II_consecutive_even_integers_l1953_195336

theorem equation_I_consecutive_integers :
  ∃ (x y z : ℕ), x + y + z = 48 ∧ (x = y - 1) ∧ (z = y + 1) := sorry

theorem equation_II_consecutive_even_integers :
  ∃ (x y z w : ℕ), x + y + z + w = 52 ∧ (y = x + 2) ∧ (z = x + 4) ∧ (w = x + 6) := sorry

end NUMINAMATH_GPT_equation_I_consecutive_integers_equation_II_consecutive_even_integers_l1953_195336


namespace NUMINAMATH_GPT_extreme_values_a_1_turning_point_a_8_l1953_195379

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 - (a + 2) * x + a * Real.log x

def turning_point (g : ℝ → ℝ) (P : ℝ × ℝ) (h : ℝ → ℝ) : Prop :=
  ∀ (x : ℝ), x ≠ P.1 → (g x - h x) / (x - P.1) > 0

theorem extreme_values_a_1 :
  (∀ (x : ℝ), f x 1 ≤ f (1/2) 1 → f x 1 = f (1/2) 1) ∧ (∀ (x : ℝ), f x 1 ≥ f 1 1 → f x 1 = f 1 1) :=
sorry

theorem turning_point_a_8 :
  ∀ (x₀ : ℝ), x₀ = 2 → turning_point (f · 8) (x₀, f x₀ 8) (λ x => (2 * x₀ + 8 / x₀ - 10) * (x - x₀) + x₀^2 - 10 * x₀ + 8 * Real.log x₀) :=
sorry

end NUMINAMATH_GPT_extreme_values_a_1_turning_point_a_8_l1953_195379


namespace NUMINAMATH_GPT_ratio_of_squares_l1953_195310

def square_inscribed_triangle_1 (x : ℝ) : Prop :=
  ∃ (a b c : ℝ), 
  a = 6 ∧ b = 8 ∧ c = 10 ∧
  x = 24 / 7

def square_inscribed_triangle_2 (y : ℝ) : Prop :=
  ∃ (a b c : ℝ), 
  a = 6 ∧ b = 8 ∧ c = 10 ∧
  y = 10 / 3

theorem ratio_of_squares (x y : ℝ) 
  (hx : square_inscribed_triangle_1 x) 
  (hy : square_inscribed_triangle_2 y) : 
  x / y = 36 / 35 := 
by sorry

end NUMINAMATH_GPT_ratio_of_squares_l1953_195310


namespace NUMINAMATH_GPT_sale_in_second_month_l1953_195304

-- Define the constants for known sales and average requirement
def sale_first_month : Int := 8435
def sale_third_month : Int := 8855
def sale_fourth_month : Int := 9230
def sale_fifth_month : Int := 8562
def sale_sixth_month : Int := 6991
def average_sale_per_month : Int := 8500
def number_of_months : Int := 6

-- Define the total sales required for six months
def total_sales_required : Int := average_sale_per_month * number_of_months

-- Define the total known sales excluding the second month
def total_known_sales : Int := sale_first_month + sale_third_month + sale_fourth_month + sale_fifth_month + sale_sixth_month

-- The statement to prove: the sale in the second month is 8927
theorem sale_in_second_month : 
  total_sales_required - total_known_sales = 8927 := 
by
  sorry

end NUMINAMATH_GPT_sale_in_second_month_l1953_195304


namespace NUMINAMATH_GPT_Q1_Q2_l1953_195365

noncomputable def prob_A_scores_3_out_of_4 (p_A_serves : ℚ) (p_A_scores_A_serves: ℚ) (p_A_scores_B_serves: ℚ) : ℚ :=
  by
    -- Placeholder probability function
    sorry

theorem Q1 (p_A_serves : ℚ := 2/3) (p_A_scores_A_serves: ℚ := 2/3) (p_A_scores_B_serves: ℚ := 1/2) :
  prob_A_scores_3_out_of_4 p_A_serves p_A_scores_A_serves p_A_scores_B_serves = 1/3 :=
  by
    -- Proof of the theorem
    sorry

noncomputable def prob_X_lessthan_or_equal_4 (p_A_serves: ℚ) (p_A_scores_A_serves: ℚ) (p_A_scores_B_serves: ℚ) : ℚ :=
  by
    -- Placeholder probability function
    sorry

theorem Q2 (p_A_serves: ℚ := 2/3) (p_A_scores_A_serves: ℚ := 2/3) (p_A_scores_B_serves: ℚ := 1/2) :
  prob_X_lessthan_or_equal_4 p_A_serves p_A_scores_A_serves p_A_scores_B_serves = 3/4 :=
  by
    -- Proof of the theorem
    sorry

end NUMINAMATH_GPT_Q1_Q2_l1953_195365


namespace NUMINAMATH_GPT_train_length_correct_l1953_195300

noncomputable def train_speed_kmph : ℝ := 60
noncomputable def train_time_seconds : ℝ := 15

noncomputable def length_of_train : ℝ :=
  let speed_mps := train_speed_kmph * 1000 / 3600
  speed_mps * train_time_seconds

theorem train_length_correct :
  length_of_train = 250.05 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_train_length_correct_l1953_195300


namespace NUMINAMATH_GPT_socks_count_l1953_195342

theorem socks_count
  (black_socks : ℕ) 
  (white_socks : ℕ)
  (H1 : white_socks = 4 * black_socks)
  (H2 : black_socks = 6)
  (H3 : white_socks / 2 = white_socks - (white_socks / 2)) :
  (white_socks / 2) - black_socks = 6 := by
  sorry

end NUMINAMATH_GPT_socks_count_l1953_195342


namespace NUMINAMATH_GPT_acute_triangle_incorrect_option_l1953_195388

theorem acute_triangle_incorrect_option (A B C : ℝ) (hA : 0 < A ∧ A < 90) (hB : 0 < B ∧ B < 90) (hC : 0 < C ∧ C < 90)
  (angle_sum : A + B + C = 180) (h_order : A > B ∧ B > C) : ¬(B + C < 90) :=
sorry

end NUMINAMATH_GPT_acute_triangle_incorrect_option_l1953_195388


namespace NUMINAMATH_GPT_initial_students_l1953_195322

theorem initial_students {f : ℕ → ℕ} {g : ℕ → ℕ} (h_f : ∀ t, t ≥ 15 * 60 + 3 → (f t = 4 * ((t - (15 * 60 + 3)) / 3 + 1))) 
    (h_g : ∀ t, t ≥ 15 * 60 + 10 → (g t = 8 * ((t - (15 * 60 + 10)) / 10 + 1))) 
    (students_at_1544 : f 15 * 60 + 44 - g 15 * 60 + 44 + initial = 27) : 
    initial = 3 := 
sorry

end NUMINAMATH_GPT_initial_students_l1953_195322


namespace NUMINAMATH_GPT_find_y_l1953_195369

theorem find_y (y : ℕ) 
  (h : (1/8) * 2^36 = 8^y) : y = 11 :=
sorry

end NUMINAMATH_GPT_find_y_l1953_195369


namespace NUMINAMATH_GPT_largest_divisor_of_n_squared_l1953_195309

theorem largest_divisor_of_n_squared (n : ℕ) (h_pos : 0 < n) (h_div : ∀ d, d ∣ n^2 → d = 900) : 900 ∣ n^2 :=
by sorry

end NUMINAMATH_GPT_largest_divisor_of_n_squared_l1953_195309


namespace NUMINAMATH_GPT_simplify_expression_l1953_195311

theorem simplify_expression :
  2^2 + 2^2 + 2^2 + 2^2 = 2^4 :=
sorry

end NUMINAMATH_GPT_simplify_expression_l1953_195311


namespace NUMINAMATH_GPT_sheila_hourly_wage_l1953_195391

def sheila_works_hours : ℕ :=
  let monday_wednesday_friday := 8 * 3
  let tuesday_thursday := 6 * 2
  monday_wednesday_friday + tuesday_thursday

def sheila_weekly_earnings : ℕ := 396
def sheila_total_hours_worked := 36
def expected_hourly_earnings := sheila_weekly_earnings / sheila_total_hours_worked

theorem sheila_hourly_wage :
  sheila_works_hours = sheila_total_hours_worked ∧
  sheila_weekly_earnings / sheila_total_hours_worked = 11 :=
by
  sorry

end NUMINAMATH_GPT_sheila_hourly_wage_l1953_195391


namespace NUMINAMATH_GPT_total_messages_l1953_195312

theorem total_messages (l1 l2 l3 a1 a2 a3 : ℕ)
  (h1 : l1 = 120)
  (h2 : a1 = l1 - 20)
  (h3 : l2 = l1 / 3)
  (h4 : a2 = 2 * a1)
  (h5 : l3 = l1)
  (h6 : a3 = a1) :
  l1 + l2 + l3 + a1 + a2 + a3 = 680 :=
by
  -- Proof steps would go here. Adding 'sorry' to skip proof.
  sorry

end NUMINAMATH_GPT_total_messages_l1953_195312


namespace NUMINAMATH_GPT_Lewis_more_items_than_Samantha_l1953_195343

def Tanya_items : ℕ := 4
def Samantha_items : ℕ := 4 * Tanya_items
def Lewis_items : ℕ := 20

theorem Lewis_more_items_than_Samantha : (Lewis_items - Samantha_items) = 4 := by
  sorry

end NUMINAMATH_GPT_Lewis_more_items_than_Samantha_l1953_195343


namespace NUMINAMATH_GPT_largest_c_3_in_range_l1953_195397

theorem largest_c_3_in_range (c : ℝ) : 
  (∃ x : ℝ, x^2 - 7*x + c = 3) ↔ c ≤ 61 / 4 := 
by sorry

end NUMINAMATH_GPT_largest_c_3_in_range_l1953_195397


namespace NUMINAMATH_GPT_line_circle_no_intersection_l1953_195363

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
sorry

end NUMINAMATH_GPT_line_circle_no_intersection_l1953_195363


namespace NUMINAMATH_GPT_find_number_of_students_l1953_195315

-- Conditions
def john_marks_wrongly_recorded : ℕ := 82
def john_actual_marks : ℕ := 62
def sarah_marks_wrongly_recorded : ℕ := 76
def sarah_actual_marks : ℕ := 66
def emily_marks_wrongly_recorded : ℕ := 92
def emily_actual_marks : ℕ := 78
def increase_in_average : ℚ := 1 / 2

-- Proof problem
theorem find_number_of_students (n : ℕ) 
    (h1 : john_marks_wrongly_recorded = 82)
    (h2 : john_actual_marks = 62)
    (h3 : sarah_marks_wrongly_recorded = 76)
    (h4 : sarah_actual_marks = 66)
    (h5 : emily_marks_wrongly_recorded = 92)
    (h6 : emily_actual_marks = 78) 
    (h7: increase_in_average = 1 / 2):
    n = 88 :=
by 
  sorry

end NUMINAMATH_GPT_find_number_of_students_l1953_195315


namespace NUMINAMATH_GPT_prank_combinations_l1953_195359

theorem prank_combinations :
  let monday_choices := 1
  let tuesday_choices := 2
  let wednesday_choices := 5
  let thursday_choices := 4
  let friday_choices := 1
  (monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices) = 40 :=
by
  sorry

end NUMINAMATH_GPT_prank_combinations_l1953_195359


namespace NUMINAMATH_GPT_alpha_bound_l1953_195373

theorem alpha_bound (α : ℝ) (x : ℕ → ℝ) (h_x_inc : ∀ n, x n < x (n + 1))
    (x0_one : x 0 = 1) (h_alpha : α = ∑' n, x (n + 1) / (x n)^3) :
    α ≥ 3 * Real.sqrt 3 / 2 := 
sorry

end NUMINAMATH_GPT_alpha_bound_l1953_195373


namespace NUMINAMATH_GPT_totalMoney_l1953_195380

noncomputable def joannaMoney : ℕ := 8
noncomputable def brotherMoney : ℕ := 3 * joannaMoney
noncomputable def sisterMoney : ℕ := joannaMoney / 2

theorem totalMoney : joannaMoney + brotherMoney + sisterMoney = 36 := by
  sorry

end NUMINAMATH_GPT_totalMoney_l1953_195380


namespace NUMINAMATH_GPT_sequence_arithmetic_condition_l1953_195375

theorem sequence_arithmetic_condition {α β : ℝ} (hα : α ≠ 0) (hβ : β ≠ 0) (hαβ : α + β ≠ 0)
  (seq : ℕ → ℝ) (hseq : ∀ n, seq (n + 2) = (α * seq (n + 1) + β * seq n) / (α + β)) :
  ∃ α β : ℝ, (∀ a1 a2 : ℝ, α ≠ 0 ∧ β ≠ 0 ∧ α + β = 0 → seq (n + 1) - seq n = seq n - seq (n - 1)) :=
by sorry

end NUMINAMATH_GPT_sequence_arithmetic_condition_l1953_195375


namespace NUMINAMATH_GPT_extremely_powerful_count_l1953_195364

def is_extremely_powerful (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b > 1 ∧ b % 2 = 1 ∧ a^b = n

noncomputable def count_extremely_powerful_below (m : ℕ) : ℕ :=
  Nat.card { n : ℕ | is_extremely_powerful n ∧ n < m }

theorem extremely_powerful_count : count_extremely_powerful_below 5000 = 19 :=
by
  sorry

end NUMINAMATH_GPT_extremely_powerful_count_l1953_195364


namespace NUMINAMATH_GPT_total_profit_is_42000_l1953_195383

noncomputable def total_profit (I_B T_B : ℝ) :=
  let I_A := 3 * I_B
  let T_A := 2 * T_B
  let profit_B := I_B * T_B
  let profit_A := I_A * T_A
  profit_A + profit_B

theorem total_profit_is_42000
  (I_B T_B : ℝ)
  (h1 : I_A = 3 * I_B)
  (h2 : T_A = 2 * T_B)
  (h3 : I_B * T_B = 6000) :
  total_profit I_B T_B = 42000 := by
  sorry

end NUMINAMATH_GPT_total_profit_is_42000_l1953_195383


namespace NUMINAMATH_GPT_problem1_problem2_l1953_195353

noncomputable def f (a x : ℝ) := a - (2 / x)

theorem problem1 (a : ℝ) :
  (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → (f a x1 < f a x2)) :=
sorry

theorem problem2 (a : ℝ) :
  (∀ x : ℝ, 1 < x → (f a x < 2 * x)) → a ≤ 3 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1953_195353


namespace NUMINAMATH_GPT_minimum_value_2x_3y_l1953_195334

theorem minimum_value_2x_3y (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (hxy : x^2 * y * (4 * x + 3 * y) = 3) :
  2 * x + 3 * y ≥ 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_minimum_value_2x_3y_l1953_195334


namespace NUMINAMATH_GPT_squares_and_sqrt_l1953_195377

variable (a b c : ℤ)

theorem squares_and_sqrt (ha : a = 10001) (hb : b = 100010001) (hc : c = 1000200030004000300020001) :
∃ x y z : ℤ, x = a^2 ∧ y = b^2 ∧ z = Int.sqrt c ∧ x = 100020001 ∧ y = 10002000300020001 ∧ z = 1000100010001 :=
by
  use a^2, b^2, Int.sqrt c
  rw [ha, hb, hc]
  sorry

end NUMINAMATH_GPT_squares_and_sqrt_l1953_195377


namespace NUMINAMATH_GPT_seventy_times_reciprocal_l1953_195329

theorem seventy_times_reciprocal (x : ℚ) (hx : 7 * x = 3) : 70 * (1 / x) = 490 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_seventy_times_reciprocal_l1953_195329


namespace NUMINAMATH_GPT_bird_weights_l1953_195370

variables (A B V G : ℕ)

theorem bird_weights : 
  A + B + V + G = 32 ∧ 
  V < G ∧ 
  V + G < B ∧ 
  A < V + B ∧ 
  G + B < A + V 
  → 
  (A = 13 ∧ V = 4 ∧ G = 5 ∧ B = 10) :=
sorry

end NUMINAMATH_GPT_bird_weights_l1953_195370


namespace NUMINAMATH_GPT_smallest_w_l1953_195308

def fact_936 : ℕ := 2^3 * 3^1 * 13^1

theorem smallest_w (w : ℕ) (h_w_pos : 0 < w) :
  (2^5 ∣ 936 * w) ∧ (3^3 ∣ 936 * w) ∧ (12^2 ∣ 936 * w) → w = 36 :=
by
  sorry

end NUMINAMATH_GPT_smallest_w_l1953_195308


namespace NUMINAMATH_GPT_shop_profit_correct_l1953_195348

def profit_per_tire_repair : ℕ := 20 - 5
def total_tire_repairs : ℕ := 300
def profit_per_complex_repair : ℕ := 300 - 50
def total_complex_repairs : ℕ := 2
def retail_profit : ℕ := 2000
def fixed_expenses : ℕ := 4000

theorem shop_profit_correct :
  profit_per_tire_repair * total_tire_repairs +
  profit_per_complex_repair * total_complex_repairs +
  retail_profit - fixed_expenses = 3000 :=
by
  sorry

end NUMINAMATH_GPT_shop_profit_correct_l1953_195348


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1953_195396

variable {a b : ℝ}

theorem necessary_but_not_sufficient : (a < b + 1) ∧ ¬ (a < b + 1 → a < b) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1953_195396


namespace NUMINAMATH_GPT_min_max_values_l1953_195368

noncomputable def f (x : ℝ) : ℝ := 1 + 3 * x - x^3

theorem min_max_values : 
  (∃ x : ℝ, f x = -1) ∧ (∃ x : ℝ, f x = 3) :=
by
  sorry

end NUMINAMATH_GPT_min_max_values_l1953_195368


namespace NUMINAMATH_GPT_sum_of_angles_of_solutions_l1953_195395

theorem sum_of_angles_of_solutions : 
  ∀ (z : ℂ), z^5 = 32 * Complex.I → ∃ θs : Fin 5 → ℝ, 
  (∀ k, 0 ≤ θs k ∧ θs k < 360) ∧ (θs 0 + θs 1 + θs 2 + θs 3 + θs 4 = 810) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_angles_of_solutions_l1953_195395


namespace NUMINAMATH_GPT_quartic_polynomial_root_l1953_195341

noncomputable def Q (x : ℝ) : ℝ := x^4 - 4*x^3 + 6*x^2 - 4*x - 2

theorem quartic_polynomial_root :
  Q (Real.sqrt (Real.sqrt 3) + 1) = 0 :=
by
  sorry

end NUMINAMATH_GPT_quartic_polynomial_root_l1953_195341


namespace NUMINAMATH_GPT_wrench_force_inversely_proportional_l1953_195301

theorem wrench_force_inversely_proportional (F L : ℝ) (F1 F2 L1 L2 : ℝ) 
    (h1 : F1 = 375) 
    (h2 : L1 = 9) 
    (h3 : L2 = 15) 
    (h4 : ∀ L : ℝ, F * L = F1 * L1) : F2 = 225 :=
by
  sorry

end NUMINAMATH_GPT_wrench_force_inversely_proportional_l1953_195301


namespace NUMINAMATH_GPT_professional_doctors_percentage_l1953_195360

-- Defining the context and conditions:

variable (total_percent : ℝ) (leaders_percent : ℝ) (nurses_percent : ℝ) (doctors_percent : ℝ)

-- Specifying the conditions:
def total_percentage_sum : Prop :=
  total_percent = 100

def leaders_percentage : Prop :=
  leaders_percent = 4

def nurses_percentage : Prop :=
  nurses_percent = 56

-- Stating the actual theorem to be proved:
theorem professional_doctors_percentage
  (h1 : total_percentage_sum total_percent)
  (h2 : leaders_percentage leaders_percent)
  (h3 : nurses_percentage nurses_percent) :
  doctors_percent = 100 - (leaders_percent + nurses_percent) := by
  sorry -- Proof placeholder

end NUMINAMATH_GPT_professional_doctors_percentage_l1953_195360


namespace NUMINAMATH_GPT_y_difference_positive_l1953_195389

theorem y_difference_positive (a c y1 y2 : ℝ) (h1 : a < 0)
  (h2 : y1 = a * 1^2 + 2 * a * 1 + c)
  (h3 : y2 = a * 2^2 + 2 * a * 2 + c) : y1 - y2 > 0 := 
sorry

end NUMINAMATH_GPT_y_difference_positive_l1953_195389


namespace NUMINAMATH_GPT_avg_of_consecutive_starting_with_b_l1953_195385

variable {a : ℕ} (h : b = (a + 1 + a + 2 + a + 3 + a + 4 + a + 5 + a + 6 + a + 7) / 7)

theorem avg_of_consecutive_starting_with_b (h : b = (a + 1 + a + 2 + a + 3 + a + 4 + a + 5 + a + 6 + a + 7) / 7) :
  (a + 4 + (a + 4 + 1) + (a + 4 + 2) + (a + 4 + 3) + (a + 4 + 4) + (a + 4 + 5) + (a + 4 + 6)) / 7 = a + 7 :=
  sorry

end NUMINAMATH_GPT_avg_of_consecutive_starting_with_b_l1953_195385


namespace NUMINAMATH_GPT_monotonically_decreasing_interval_l1953_195319

-- Given conditions
def f (x : ℝ) : ℝ := x^2 * (x - 3)

-- The proof problem statement
theorem monotonically_decreasing_interval :
  ∃ a b : ℝ, (0 ≤ a) ∧ (b ≤ 2) ∧ (∀ x : ℝ, a ≤ x ∧ x ≤ b → (deriv f x ≤ 0)) :=
sorry

end NUMINAMATH_GPT_monotonically_decreasing_interval_l1953_195319


namespace NUMINAMATH_GPT_fraction_a_b_l1953_195399

variables {a b x y : ℝ}

theorem fraction_a_b (h1 : 4 * x - 2 * y = a) (h2 : 6 * y - 12 * x = b) (hb : b ≠ 0) :
  a / b = -1/3 := 
sorry

end NUMINAMATH_GPT_fraction_a_b_l1953_195399


namespace NUMINAMATH_GPT_range_of_f_lt_f2_l1953_195361

-- Definitions for the given conditions
def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
def increasing_on (f : ℝ → ℝ) (S : Set ℝ) := ∀ ⦃a b : ℝ⦄, a ∈ S → b ∈ S → a < b → f a < f b

-- Lean 4 statement for the proof problem
theorem range_of_f_lt_f2 (f : ℝ → ℝ) 
  (h_even : even_function f) 
  (h_increasing : increasing_on f {x | x ≤ 0}) : 
  ∀ x : ℝ, f x < f 2 → x > 2 ∨ x < -2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_lt_f2_l1953_195361


namespace NUMINAMATH_GPT_dayAfter73DaysFromFridayAnd9WeeksLater_l1953_195362

-- Define the days of the week as a data type
inductive Weekday
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday
deriving DecidableEq, Repr

open Weekday

-- Function to calculate the day of the week after a given number of days
def addDays (start_day : Weekday) (days : ℕ) : Weekday :=
  match start_day with
  | Sunday    => match days % 7 with | 0 => Sunday    | 1 => Monday | 2 => Tuesday | 3 => Wednesday | 4 => Thursday | 5 => Friday | 6 => Saturday | _ => Sunday
  | Monday    => match days % 7 with | 0 => Monday    | 1 => Tuesday | 2 => Wednesday | 3 => Thursday | 4 => Friday | 5 => Saturday | 6 => Sunday | _ => Monday
  | Tuesday   => match days % 7 with | 0 => Tuesday   | 1 => Wednesday | 2 => Thursday | 3 => Friday | 4 => Saturday | 5 => Sunday | 6 => Monday | _ => Tuesday
  | Wednesday => match days % 7 with | 0 => Wednesday | 1 => Thursday | 2 => Friday | 3 => Saturday | 4 => Sunday | 5 => Monday | 6 => Tuesday | _ => Wednesday
  | Thursday  => match days % 7 with | 0 => Thursday  | 1 => Friday | 2 => Saturday | 3 => Sunday | 4 => Monday | 5 => Tuesday | 6 => Wednesday | _ => Thursday
  | Friday    => match days % 7 with | 0 => Friday    | 1 => Saturday | 2 => Sunday | 3 => Monday | 4 => Tuesday | 5 => Wednesday | 6 => Thursday | _ => Friday
  | Saturday  => match days % 7 with | 0 => Saturday  | 1 => Sunday | 2 => Monday | 3 => Tuesday | 4 => Wednesday | 5 => Thursday | 6 => Friday | _ => Saturday

-- Theorem that proves the required solution
theorem dayAfter73DaysFromFridayAnd9WeeksLater : addDays Friday 73 = Monday ∧ addDays Monday (9 * 7) = Monday := 
by
  -- Placeholder to acknowledge proof requirements
  sorry

end NUMINAMATH_GPT_dayAfter73DaysFromFridayAnd9WeeksLater_l1953_195362


namespace NUMINAMATH_GPT_pool_volume_l1953_195398

variable {rate1 rate2 : ℕ}
variables {hose1 hose2 hose3 hose4 : ℕ}
variables {time : ℕ}

def hose1_rate := 2
def hose2_rate := 2
def hose3_rate := 3
def hose4_rate := 3
def fill_time := 25

def total_rate := hose1_rate + hose2_rate + hose3_rate + hose4_rate

theorem pool_volume (h : hose1 = hose1_rate ∧ hose2 = hose2_rate ∧ hose3 = hose3_rate ∧ hose4 = hose4_rate ∧ time = fill_time):
  total_rate * 60 * time = 15000 := 
by 
  sorry

end NUMINAMATH_GPT_pool_volume_l1953_195398


namespace NUMINAMATH_GPT_soccer_and_volleyball_unit_prices_max_soccer_balls_l1953_195345

-- Define the conditions and the problem
def unit_price_soccer_ball (x : ℕ) (y : ℕ) : Prop :=
  x = y + 15 ∧ 480 / x = 390 / y

def school_purchase (m : ℕ) : Prop :=
  m ≤ 70 ∧ 80 * m + 65 * (100 - m) ≤ 7550

-- Proof statement for the unit prices of soccer balls and volleyballs
theorem soccer_and_volleyball_unit_prices (x y : ℕ) (h : unit_price_soccer_ball x y) :
  x = 80 ∧ y = 65 :=
by
  sorry

-- Proof statement for the maximum number of soccer balls the school can purchase
theorem max_soccer_balls (m : ℕ) :
  school_purchase m :=
by
  sorry

end NUMINAMATH_GPT_soccer_and_volleyball_unit_prices_max_soccer_balls_l1953_195345


namespace NUMINAMATH_GPT_percentage_return_on_investment_l1953_195374

theorem percentage_return_on_investment
  (dividend_rate : ℝ)
  (face_value : ℝ)
  (purchase_price : ℝ)
  (dividend_per_share : ℝ := (dividend_rate / 100) * face_value)
  (percentage_return : ℝ := (dividend_per_share / purchase_price) * 100)
  (h1 : dividend_rate = 15.5)
  (h2 : face_value = 50)
  (h3 : purchase_price = 31) :
  percentage_return = 25 := by
    sorry

end NUMINAMATH_GPT_percentage_return_on_investment_l1953_195374


namespace NUMINAMATH_GPT_num_pupils_is_40_l1953_195316

-- given conditions
def incorrect_mark : ℕ := 83
def correct_mark : ℕ := 63
def mark_difference : ℕ := incorrect_mark - correct_mark
def avg_increase : ℚ := 1 / 2

-- the main problem statement to prove
theorem num_pupils_is_40 (n : ℕ) (h : (mark_difference : ℚ) / n = avg_increase) : n = 40 := 
sorry

end NUMINAMATH_GPT_num_pupils_is_40_l1953_195316


namespace NUMINAMATH_GPT_circumference_of_circle_l1953_195354

/-- Given a circle with area 4 * π square units, prove that its circumference is 4 * π units. -/
theorem circumference_of_circle (r : ℝ) (h : π * r^2 = 4 * π) : 2 * π * r = 4 * π :=
sorry

end NUMINAMATH_GPT_circumference_of_circle_l1953_195354


namespace NUMINAMATH_GPT_non_neg_int_solutions_l1953_195307

theorem non_neg_int_solutions : 
  ∀ (x y : ℕ), 2 * x ^ 2 + 2 * x * y - x + y = 2020 → 
               (x = 0 ∧ y = 2020) ∨ (x = 1 ∧ y = 673) :=
by
  sorry

end NUMINAMATH_GPT_non_neg_int_solutions_l1953_195307


namespace NUMINAMATH_GPT_even_and_monotonically_decreasing_l1953_195346

noncomputable def f_B (x : ℝ) : ℝ := 1 / (x^2)

theorem even_and_monotonically_decreasing (x : ℝ) (h : x > 0) :
  (f_B x = f_B (-x)) ∧ (∀ {a b : ℝ}, a < b → a > 0 → b > 0 → f_B a > f_B b) :=
by
  sorry

end NUMINAMATH_GPT_even_and_monotonically_decreasing_l1953_195346


namespace NUMINAMATH_GPT_length_AF_is_25_l1953_195314

open Classical

noncomputable def length_AF : ℕ :=
  let AB := 5
  let AC := 11
  let DE := 8
  let EF := 4
  let BC := AC - AB
  let CD := BC / 3
  let AF := AB + BC + CD + DE + EF
  AF

theorem length_AF_is_25 :
  length_AF = 25 := by
  sorry

end NUMINAMATH_GPT_length_AF_is_25_l1953_195314


namespace NUMINAMATH_GPT_mr_blue_expected_rose_petals_l1953_195381

def mr_blue_flower_bed_rose_petals (length_paces : ℕ) (width_paces : ℕ) (pace_length_ft : ℝ) (petals_per_sqft : ℝ) : ℝ :=
  let length_ft := length_paces * pace_length_ft
  let width_ft := width_paces * pace_length_ft
  let area_sqft := length_ft * width_ft
  area_sqft * petals_per_sqft

theorem mr_blue_expected_rose_petals :
  mr_blue_flower_bed_rose_petals 18 24 1.5 0.4 = 388.8 :=
by
  simp [mr_blue_flower_bed_rose_petals]
  norm_num

end NUMINAMATH_GPT_mr_blue_expected_rose_petals_l1953_195381


namespace NUMINAMATH_GPT_solution_set_of_inequality_system_l1953_195344

theorem solution_set_of_inequality_system (x : ℝ) : (x - 1 < 0) ∧ (x + 1 > 0) ↔ (-1 < x ∧ x < 1) :=
by
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_system_l1953_195344


namespace NUMINAMATH_GPT_range_of_m_inequality_a_b_l1953_195305

def f (x : ℝ) : ℝ := |2 * x + 1| + |2 * x - 2|

theorem range_of_m (m : ℝ) : (∀ x, f x ≥ |m - 1|) → -2 ≤ m ∧ m ≤ 4 :=
sorry

theorem inequality_a_b (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : a^2 + b^2 = 2) : 
  a + b ≥ 2 * a * b :=
sorry

end NUMINAMATH_GPT_range_of_m_inequality_a_b_l1953_195305


namespace NUMINAMATH_GPT_coefficient_of_a3b2_in_expansions_l1953_195324

theorem coefficient_of_a3b2_in_expansions 
  (a b c : ℝ) :
  (1 : ℝ) * (a + b)^5 * (c + c⁻¹)^8 = 700 :=
by 
  sorry

end NUMINAMATH_GPT_coefficient_of_a3b2_in_expansions_l1953_195324


namespace NUMINAMATH_GPT_sum_of_three_positive_integers_l1953_195355

theorem sum_of_three_positive_integers (n : ℕ) (h : n ≥ 3) :
  ∃ k : ℕ, k = (n - 1) * (n - 2) / 2 := 
sorry

end NUMINAMATH_GPT_sum_of_three_positive_integers_l1953_195355


namespace NUMINAMATH_GPT_inequality_solution_l1953_195387

theorem inequality_solution (x : ℝ) : 3 * x^2 + 9 * x + 6 ≤ 0 ↔ -2 ≤ x ∧ x ≤ -1 := 
sorry

end NUMINAMATH_GPT_inequality_solution_l1953_195387


namespace NUMINAMATH_GPT_trajectory_equation_of_point_M_l1953_195386

variables {x y a b : ℝ}

theorem trajectory_equation_of_point_M :
  (a^2 + b^2 = 100) →
  (x = a / (1 + 4)) →
  (y = 4 * b / (1 + 4)) →
  16 * x^2 + y^2 = 64 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_trajectory_equation_of_point_M_l1953_195386


namespace NUMINAMATH_GPT_new_people_moved_in_l1953_195320

theorem new_people_moved_in (N : ℕ) : (∃ N, 1/16 * (780 - 400 + N : ℝ) = 60) → N = 580 := by
  intros hN
  sorry

end NUMINAMATH_GPT_new_people_moved_in_l1953_195320


namespace NUMINAMATH_GPT_total_number_of_fish_l1953_195317

noncomputable def number_of_stingrays : ℕ := 28

noncomputable def number_of_sharks : ℕ := 2 * number_of_stingrays

theorem total_number_of_fish : number_of_sharks + number_of_stingrays = 84 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_fish_l1953_195317


namespace NUMINAMATH_GPT_apples_total_l1953_195332

def benny_apples : ℕ := 2
def dan_apples : ℕ := 9
def total_apples : ℕ := benny_apples + dan_apples

theorem apples_total : total_apples = 11 :=
by
    sorry

end NUMINAMATH_GPT_apples_total_l1953_195332


namespace NUMINAMATH_GPT_janice_purchases_l1953_195357

theorem janice_purchases (a b c : ℕ) : 
  a + b + c = 50 ∧ 30 * a + 200 * b + 300 * c = 5000 → a = 10 :=
sorry

end NUMINAMATH_GPT_janice_purchases_l1953_195357


namespace NUMINAMATH_GPT_number_of_cirrus_clouds_l1953_195351

def C_cb := 3
def C_cu := 12 * C_cb
def C_ci := 4 * C_cu

theorem number_of_cirrus_clouds : C_ci = 144 :=
by
  sorry

end NUMINAMATH_GPT_number_of_cirrus_clouds_l1953_195351


namespace NUMINAMATH_GPT_marked_box_in_second_row_l1953_195356

theorem marked_box_in_second_row:
  ∀ a b c d e f g h : ℕ, 
  (e = a + b) → 
  (f = b + c) →
  (g = c + d) →
  (h = a + 2 * b + c) →
  ((a = 5) ∧ (d = 6)) →
  ((a = 3) ∨ (b = 3) ∨ (c = 3) ∨ (d = 3)) →
  (f = 3) :=
by
  sorry

end NUMINAMATH_GPT_marked_box_in_second_row_l1953_195356


namespace NUMINAMATH_GPT_range_of_a_l1953_195338

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x + 4 ≥ 0) ↔ (2 ≤ a ∧ a ≤ 6) := 
sorry

end NUMINAMATH_GPT_range_of_a_l1953_195338


namespace NUMINAMATH_GPT_given_inequality_l1953_195325

theorem given_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h: 1 + a + b + c = 2 * a * b * c) :
  ab / (1 + a + b) + bc / (1 + b + c) + ca / (1 + c + a) ≥ 3 / 2 :=
sorry

end NUMINAMATH_GPT_given_inequality_l1953_195325


namespace NUMINAMATH_GPT_number_of_preferred_groups_l1953_195331

def preferred_group_sum_multiple_5 (n : Nat) : Nat := 
  (2^n) * ((2^(4*n) - 1) / 5 + 1) - 1

theorem number_of_preferred_groups :
  preferred_group_sum_multiple_5 400 = 2^400 * (2^1600 - 1) / 5 + 1 - 1 :=
sorry

end NUMINAMATH_GPT_number_of_preferred_groups_l1953_195331


namespace NUMINAMATH_GPT_second_carpenter_days_l1953_195302

theorem second_carpenter_days (x : ℚ) (h1 : 1 / 5 + 1 / x = 1 / 2) : x = 10 / 3 :=
by
  sorry

end NUMINAMATH_GPT_second_carpenter_days_l1953_195302


namespace NUMINAMATH_GPT_range_of_a_l1953_195306

noncomputable def f (x a : ℝ) := (x^2 + a * x + 11) / (x + 1)

theorem range_of_a (a : ℝ) :
  (∀ x : ℕ, x > 0 → f x a ≥ 3) ↔ (a ≥ -8 / 3) :=
by sorry

end NUMINAMATH_GPT_range_of_a_l1953_195306


namespace NUMINAMATH_GPT_house_cats_initial_l1953_195347

def initial_house_cats (S A T H : ℝ) : Prop :=
  S + H + A = T

theorem house_cats_initial (S A T H : ℝ) (h1 : S = 13.0) (h2 : A = 10.0) (h3 : T = 28) :
  initial_house_cats S A T H ↔ H = 5 := by
sorry

end NUMINAMATH_GPT_house_cats_initial_l1953_195347


namespace NUMINAMATH_GPT_possible_values_for_D_l1953_195327

def distinct_digits (A B C D E : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E

def digits_range (A B C D E : ℕ) : Prop :=
  0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9 ∧ 0 ≤ D ∧ D ≤ 9 ∧ 0 ≤ E ∧ E ≤ 9

def addition_equation (A B C D E : ℕ) : Prop :=
  A * 10000 + B * 1000 + C * 100 + D * 10 + B +
  B * 10000 + C * 1000 + A * 100 + D * 10 + E = 
  E * 10000 + D * 1000 + D * 100 + E * 10 + E

theorem possible_values_for_D : 
  ∀ (A B C D E : ℕ),
  distinct_digits A B C D E →
  digits_range A B C D E →
  addition_equation A B C D E →
  ∃ (S : Finset ℕ), (∀ d ∈ S, 0 ≤ d ∧ d ≤ 9) ∧ (S.card = 2) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_possible_values_for_D_l1953_195327


namespace NUMINAMATH_GPT_num_customers_after_family_l1953_195333

-- Definitions
def soft_taco_price : ℕ := 2
def hard_taco_price : ℕ := 5
def family_hard_tacos : ℕ := 4
def family_soft_tacos : ℕ := 3
def total_income : ℕ := 66

-- Intermediate values which can be derived
def family_cost : ℕ := (family_hard_tacos * hard_taco_price) + (family_soft_tacos * soft_taco_price)
def remaining_income : ℕ := total_income - family_cost

-- Proposition: Number of customers after the family
def customers_after_family : ℕ := remaining_income / (2 * soft_taco_price)

-- Theorem to prove the number of customers is 10
theorem num_customers_after_family : customers_after_family = 10 := by
  sorry

end NUMINAMATH_GPT_num_customers_after_family_l1953_195333


namespace NUMINAMATH_GPT_value_of_a_plus_b_l1953_195367

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  if 0 ≤ x then Real.sqrt x + 3 else a * x + b

theorem value_of_a_plus_b (a b : ℝ) 
  (h1 : ∀ x1 : ℝ, x1 ≠ 0 → ∃ x2 : ℝ, x1 ≠ x2 ∧ f x1 a b = f x2 a b)
  (h2 : f (2 * a) a b = f (3 * b) a b) :
  a + b = - (Real.sqrt 6) / 2 + 3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_plus_b_l1953_195367


namespace NUMINAMATH_GPT_angle_Y_measure_l1953_195378

def hexagon_interior_angle_sum (n : ℕ) : ℕ :=
  180 * (n - 2)

def supplementary (α β : ℕ) : Prop :=
  α + β = 180

def equal_angles (α β γ δ : ℕ) : Prop :=
  α = β ∧ β = γ ∧ γ = δ

theorem angle_Y_measure :
  ∀ (C H E S1 S2 Y : ℕ),
    C = E ∧ E = S1 ∧ S1 = Y →
    supplementary H S2 →
    hexagon_interior_angle_sum 6 = C + H + E + S1 + S2 + Y →
    Y = 135 :=
by
  intros C H E S1 S2 Y h1 h2 h3
  sorry

end NUMINAMATH_GPT_angle_Y_measure_l1953_195378


namespace NUMINAMATH_GPT_alex_age_div_M_l1953_195318

variable {A M : ℕ}

-- Definitions provided by the conditions
def alex_age_current : ℕ := A
def sum_children_age : ℕ := A
def alex_age_M_years_ago (A M : ℕ) : ℕ := A - M
def children_age_M_years_ago (A M : ℕ) : ℕ := A - 4 * M

-- Given condition as a hypothesis
def condition (A M : ℕ) := alex_age_M_years_ago A M = 3 * children_age_M_years_ago A M

-- The theorem to prove
theorem alex_age_div_M (A M : ℕ) (h : condition A M) : A / M = 11 / 2 := 
by
  -- This is a placeholder for the actual proof.
  sorry

end NUMINAMATH_GPT_alex_age_div_M_l1953_195318


namespace NUMINAMATH_GPT_max_area_rectangle_with_perimeter_40_l1953_195384

theorem max_area_rectangle_with_perimeter_40 :
  ∃ (l w : ℕ), 2 * l + 2 * w = 40 ∧ l * w = 100 :=
sorry

end NUMINAMATH_GPT_max_area_rectangle_with_perimeter_40_l1953_195384


namespace NUMINAMATH_GPT_negation_proposition_p_l1953_195313

theorem negation_proposition_p (x y : ℝ) : (¬ ((x - 1) ^ 2 + (y - 2) ^ 2 = 0) → (x ≠ 1 ∨ y ≠ 2)) :=
by
  sorry

end NUMINAMATH_GPT_negation_proposition_p_l1953_195313


namespace NUMINAMATH_GPT_arithmetic_expression_value_l1953_195390

theorem arithmetic_expression_value :
  (19 + 43 / 151) * 151 = 2910 :=
by {
  sorry
}

end NUMINAMATH_GPT_arithmetic_expression_value_l1953_195390


namespace NUMINAMATH_GPT_no_rational_solution_l1953_195372

/-- Prove that the only rational solution to the equation x^3 + 3y^3 + 9z^3 = 9xyz is x = y = z = 0. -/
theorem no_rational_solution : ∀ (x y z : ℚ), x^3 + 3 * y^3 + 9 * z^3 = 9 * x * y * z → x = 0 ∧ y = 0 ∧ z = 0 :=
by
  intro x y z h
  sorry

end NUMINAMATH_GPT_no_rational_solution_l1953_195372


namespace NUMINAMATH_GPT_lana_spent_l1953_195376

def ticket_cost : ℕ := 6
def tickets_for_friends : ℕ := 8
def extra_tickets : ℕ := 2

theorem lana_spent :
  ticket_cost * (tickets_for_friends + extra_tickets) = 60 := 
by
  sorry

end NUMINAMATH_GPT_lana_spent_l1953_195376


namespace NUMINAMATH_GPT_finite_set_cardinality_l1953_195328

-- Define the main theorem statement
theorem finite_set_cardinality (m : ℕ) (A : Finset ℤ) (B : ℕ → Finset ℤ)
  (hm : m ≥ 2)
  (hB : ∀ k : ℕ, k ∈ Finset.range m.succ → (B k).sum id = m^k) :
  A.card ≥ m / 2 := 
sorry

end NUMINAMATH_GPT_finite_set_cardinality_l1953_195328


namespace NUMINAMATH_GPT_inequality_proof_l1953_195335

theorem inequality_proof (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  1 < (b / (a * b + b + 1) + c / (b * c + c + 1) + d / (c * d + d + 1) + a / (d * a + a + 1)) ∧
  (b / (a * b + b + 1) + c / (b * c + c + 1) + d / (c * d + d + 1) + a / (d * a + a + 1)) < 2 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1953_195335


namespace NUMINAMATH_GPT_four_digit_number_exists_l1953_195339

theorem four_digit_number_exists :
  ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ 4 * n = (n % 10) * 1000 + ((n / 10) % 10) * 100 + ((n / 100) % 10) * 10 + (n / 1000) :=
sorry

end NUMINAMATH_GPT_four_digit_number_exists_l1953_195339


namespace NUMINAMATH_GPT_center_of_gravity_shift_center_of_gravity_shift_result_l1953_195350

variable (l s : ℝ) (s_val : s = 60)
#check (s_val : s = 60)

theorem center_of_gravity_shift : abs ((l / 2) - ((l - s) / 2)) = s / 2 := 
by sorry

theorem center_of_gravity_shift_result : (s / 2 = 30) :=
by sorry

end NUMINAMATH_GPT_center_of_gravity_shift_center_of_gravity_shift_result_l1953_195350


namespace NUMINAMATH_GPT_contractor_male_workers_l1953_195337

noncomputable def number_of_male_workers (M : ℕ) : Prop :=
  let female_wages : ℕ := 15 * 20
  let child_wages : ℕ := 5 * 8
  let total_wages : ℕ := 35 * M + female_wages + child_wages
  let total_workers : ℕ := M + 15 + 5
  (total_wages / total_workers) = 26

theorem contractor_male_workers : ∃ M : ℕ, number_of_male_workers M ∧ M = 20 :=
by
  use 20
  sorry

end NUMINAMATH_GPT_contractor_male_workers_l1953_195337


namespace NUMINAMATH_GPT_current_short_trees_l1953_195303

-- Definitions of conditions in a)
def tall_trees : ℕ := 44
def short_trees_planted : ℕ := 57
def total_short_trees_after_planting : ℕ := 98

-- Statement to prove the question == answer given conditions
theorem current_short_trees (S : ℕ) (h : S + short_trees_planted = total_short_trees_after_planting) : S = 41 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_current_short_trees_l1953_195303

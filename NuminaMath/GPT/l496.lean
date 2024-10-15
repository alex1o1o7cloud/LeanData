import Mathlib

namespace NUMINAMATH_GPT_circle_center_radius_l496_49688

/-
Given:
- The endpoints of a diameter are (2, -3) and (-8, 7).

Prove:
- The center of the circle is (-3, 2).
- The radius of the circle is 5√2.
-/

noncomputable def center_and_radius (A B : ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let Cx := ((A.1 + B.1) / 2)
  let Cy := ((A.2 + B.2) / 2)
  let radius := Real.sqrt ((A.1 - Cx) * (A.1 - Cx) + (A.2 - Cy) * (A.2 - Cy))
  (Cx, Cy, radius)

theorem circle_center_radius :
  center_and_radius (2, -3) (-8, 7) = (-3, 2, 5 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_circle_center_radius_l496_49688


namespace NUMINAMATH_GPT_determine_a_plus_b_l496_49608

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b
noncomputable def f_inv (a b x : ℝ) : ℝ := b * x^2 + a

theorem determine_a_plus_b (a b : ℝ) (h: ∀ x : ℝ, f a b (f_inv a b x) = x) : a + b = 1 :=
sorry

end NUMINAMATH_GPT_determine_a_plus_b_l496_49608


namespace NUMINAMATH_GPT_reflect_parallelogram_l496_49607

theorem reflect_parallelogram :
  let D : ℝ × ℝ := (4,1)
  let Dx : ℝ × ℝ := (D.1, -D.2) -- Reflect across x-axis
  let Dxy : ℝ × ℝ := (Dx.2 - 1, Dx.1 - 1) -- Translate point down by 1 unit and reflect across y=x
  let D'' : ℝ × ℝ := (Dxy.1 + 1, Dxy.2 + 1) -- Translate point back up by 1 unit
  D'' = (-2, 5) := by
  sorry

end NUMINAMATH_GPT_reflect_parallelogram_l496_49607


namespace NUMINAMATH_GPT_inequality_must_hold_l496_49600

theorem inequality_must_hold (m n : ℝ) (h : m > n) : 2 + m > 2 + n :=
sorry

end NUMINAMATH_GPT_inequality_must_hold_l496_49600


namespace NUMINAMATH_GPT_number_of_white_balls_l496_49694

theorem number_of_white_balls (total_balls yellow_frequency : ℕ) (h1 : total_balls = 10) (h2 : yellow_frequency = 60) :
  (total_balls - (total_balls * yellow_frequency / 100) = 4) :=
by
  sorry

end NUMINAMATH_GPT_number_of_white_balls_l496_49694


namespace NUMINAMATH_GPT_tan_alpha_frac_l496_49666

theorem tan_alpha_frac (α : ℝ) (h : Real.tan α = 2) : (Real.sin α - Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = 1 / 11 := by
  sorry

end NUMINAMATH_GPT_tan_alpha_frac_l496_49666


namespace NUMINAMATH_GPT_imo1983_q6_l496_49620

theorem imo1983_q6 (a b c : ℝ) (h : a + b > c) (h2 : a + c > b) (h3 : b + c > a) : 
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_imo1983_q6_l496_49620


namespace NUMINAMATH_GPT_real_roots_range_l496_49601

theorem real_roots_range (a : ℝ) : (∃ x : ℝ, a*x^2 + 2*x - 1 = 0) ↔ (a >= -1 ∧ a ≠ 0) :=
by 
  sorry

end NUMINAMATH_GPT_real_roots_range_l496_49601


namespace NUMINAMATH_GPT_math_problem_l496_49604

theorem math_problem : ((-7)^3 / 7^2 - 2^5 + 4^3 - 8) = 81 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l496_49604


namespace NUMINAMATH_GPT_cheapest_candle_cost_to_measure_1_minute_l496_49664

-- Definitions

def big_candle_cost := 16 -- cost of a big candle in cents
def big_candle_burn_time := 16 -- burn time of a big candle in minutes
def small_candle_cost := 7 -- cost of a small candle in cents
def small_candle_burn_time := 7 -- burn time of a small candle in minutes

-- Problem statement
theorem cheapest_candle_cost_to_measure_1_minute :
  (∃ (n m : ℕ), n * big_candle_burn_time - m * small_candle_burn_time = 1 ∧
                 n * big_candle_cost + m * small_candle_cost = 97) :=
sorry

end NUMINAMATH_GPT_cheapest_candle_cost_to_measure_1_minute_l496_49664


namespace NUMINAMATH_GPT_median_number_of_children_l496_49665

-- Define the given conditions
def number_of_data_points : Nat := 13
def median_position : Nat := (number_of_data_points + 1) / 2

-- We assert the median value based on information given in the problem
def median_value : Nat := 4

-- Statement to prove the problem
theorem median_number_of_children (h1: median_position = 7) (h2: median_value = 4) : median_value = 4 := 
by
  sorry

end NUMINAMATH_GPT_median_number_of_children_l496_49665


namespace NUMINAMATH_GPT_beaver_stores_60_carrots_l496_49630

theorem beaver_stores_60_carrots (b r : ℕ) (h1 : 4 * b = 5 * r) (h2 : b = r + 3) : 4 * b = 60 :=
by
  sorry

end NUMINAMATH_GPT_beaver_stores_60_carrots_l496_49630


namespace NUMINAMATH_GPT_product_of_roots_l496_49683

theorem product_of_roots :
  ∀ (x : ℝ), (|x|^2 - 3 * |x| - 10 = 0) →
  (∃ a b : ℝ, a ≠ b ∧ (|a| = 5 ∧ |b| = 5) ∧ a * b = -25) :=
by {
  sorry
}

end NUMINAMATH_GPT_product_of_roots_l496_49683


namespace NUMINAMATH_GPT_find_a_plus_b_l496_49695

theorem find_a_plus_b (a b : ℤ) (h1 : 2 * a = 0) (h2 : a^2 - b = 25) : a + b = -25 :=
by 
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l496_49695


namespace NUMINAMATH_GPT_g_eq_l496_49658

noncomputable def g (n : ℕ) : ℝ :=
  (7 + 4 * Real.sqrt 7) / 14 * ((1 + Real.sqrt 7) / 2) ^ n +
  (7 - 4 * Real.sqrt 7) / 14 * ((1 - Real.sqrt 7) / 2) ^ n

theorem g_eq (n : ℕ) : g (n + 2) - g (n - 2) = 3 * g n := by
  sorry

end NUMINAMATH_GPT_g_eq_l496_49658


namespace NUMINAMATH_GPT_no_possible_stack_of_1997_sum_l496_49671

theorem no_possible_stack_of_1997_sum :
  ¬ ∃ k : ℕ, 6 * k = 3 * 1997 := by
  sorry

end NUMINAMATH_GPT_no_possible_stack_of_1997_sum_l496_49671


namespace NUMINAMATH_GPT_simple_interest_is_correct_l496_49641

-- Define the principal amount, rate of interest, and time
def P : ℕ := 400
def R : ℚ := 22.5
def T : ℕ := 2

-- Define the formula for simple interest
def simple_interest (P : ℕ) (R : ℚ) (T : ℕ) : ℚ :=
  (P * R * T) / 100

-- The statement we need to prove
theorem simple_interest_is_correct : simple_interest P R T = 90 :=
by
  sorry

end NUMINAMATH_GPT_simple_interest_is_correct_l496_49641


namespace NUMINAMATH_GPT_percentage_return_is_25_l496_49690

noncomputable def percentage_return_on_investment
  (dividend_rate : ℝ)
  (face_value : ℝ)
  (purchase_price : ℝ) : ℝ :=
  (dividend_rate / 100 * face_value / purchase_price) * 100

theorem percentage_return_is_25 :
  percentage_return_on_investment 18.5 50 37 = 25 := 
by
  sorry

end NUMINAMATH_GPT_percentage_return_is_25_l496_49690


namespace NUMINAMATH_GPT_true_false_questions_count_l496_49614

noncomputable def number_of_true_false_questions (T F M : ℕ) : Prop :=
  T + F + M = 45 ∧ M = 2 * F ∧ F = T + 7

theorem true_false_questions_count : ∃ T F M : ℕ, number_of_true_false_questions T F M ∧ T = 6 :=
by
  sorry

end NUMINAMATH_GPT_true_false_questions_count_l496_49614


namespace NUMINAMATH_GPT_find_value_of_expression_l496_49619

theorem find_value_of_expression 
  (x y z w : ℤ)
  (hx : x = 3)
  (hy : y = 2)
  (hz : z = 4)
  (hw : w = -1) :
  x^2 * y - 2 * x * y + 3 * x * z - (x + y) * (y + z) * (z + w) = -48 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_expression_l496_49619


namespace NUMINAMATH_GPT_angle_ACB_33_l496_49667

noncomputable def triangle_ABC : Type := sorry  -- Define the triangle ABC
noncomputable def ω : Type := sorry  -- Define the circumcircle of ABC
noncomputable def M : Type := sorry  -- Define the midpoint of arc BC not containing A
noncomputable def D : Type := sorry  -- Define the point D such that DM is tangent to ω
def AM_eq_AC : Prop := sorry  -- Define the equality AM = AC
def angle_DMC := (38 : ℝ)  -- Define angle DMC = 38 degrees

theorem angle_ACB_33 (h1 : triangle_ABC) 
                      (h2 : ω) 
                      (h3 : M) 
                      (h4 : D) 
                      (h5 : AM_eq_AC)
                      (h6 : angle_DMC = 38) : ∃ θ, (θ = 33) ∧ (angle_ACB = θ) :=
sorry  -- Proof goes here

end NUMINAMATH_GPT_angle_ACB_33_l496_49667


namespace NUMINAMATH_GPT_integer_ratio_l496_49674

theorem integer_ratio (A B C D : ℕ) (h1 : (A + B + C + D) / 4 = 16)
  (h2 : A % B = 0) (h3 : B = C - 2) (h4 : D = 2) (h5 : A ≠ B) (h6 : B ≠ C) (h7 : C ≠ D) (h8 : D ≠ A)
  (h9: 0 < A) (h10: 0 < B) (h11: 0 < C):
  A / B = 28 := 
sorry

end NUMINAMATH_GPT_integer_ratio_l496_49674


namespace NUMINAMATH_GPT_sixth_root_of_large_number_l496_49647

theorem sixth_root_of_large_number : 
  ∃ (x : ℕ), x = 51 ∧ x ^ 6 = 24414062515625 :=
by
  sorry

end NUMINAMATH_GPT_sixth_root_of_large_number_l496_49647


namespace NUMINAMATH_GPT_avg_meal_cost_per_individual_is_72_l496_49675

theorem avg_meal_cost_per_individual_is_72
  (total_bill : ℝ)
  (gratuity_percent : ℝ)
  (num_investment_bankers num_clients : ℕ)
  (total_individuals := num_investment_bankers + num_clients)
  (meal_cost_before_gratuity : ℝ := total_bill / (1 + gratuity_percent))
  (average_cost := meal_cost_before_gratuity / total_individuals) :
  total_bill = 1350 ∧ gratuity_percent = 0.25 ∧ num_investment_bankers = 7 ∧ num_clients = 8 →
  average_cost = 72 := by
  sorry

end NUMINAMATH_GPT_avg_meal_cost_per_individual_is_72_l496_49675


namespace NUMINAMATH_GPT_peaches_left_in_baskets_l496_49610

theorem peaches_left_in_baskets :
  let initial_baskets := 5
  let initial_peaches_per_basket := 20
  let new_baskets := 4
  let new_peaches_per_basket := 25
  let peaches_removed_per_basket := 10

  let total_initial_peaches := initial_baskets * initial_peaches_per_basket
  let total_new_peaches := new_baskets * new_peaches_per_basket
  let total_peaches_before_removal := total_initial_peaches + total_new_peaches

  let total_baskets := initial_baskets + new_baskets
  let total_peaches_removed := total_baskets * peaches_removed_per_basket
  let peaches_left := total_peaches_before_removal - total_peaches_removed

  peaches_left = 110 := by
  sorry

end NUMINAMATH_GPT_peaches_left_in_baskets_l496_49610


namespace NUMINAMATH_GPT_elongation_rate_significantly_improved_l496_49669

noncomputable def elongation_improvement : Prop :=
  let x : List ℝ := [545, 533, 551, 522, 575, 544, 541, 568, 596, 548]
  let y : List ℝ := [536, 527, 543, 530, 560, 533, 522, 550, 576, 536]
  let z := List.zipWith (λ xi yi => xi - yi) x y
  let n : ℝ := 10
  let mean_z := (List.sum z) / n
  let variance_z := (List.sum (List.map (λ zi => (zi - mean_z)^2) z)) / n
  mean_z = 11 ∧ 
  variance_z = 61 ∧ 
  mean_z ≥ 2 * Real.sqrt (variance_z / n)

-- We state the theorem without proof
theorem elongation_rate_significantly_improved : elongation_improvement :=
by
  -- Proof can be written here
  sorry

end NUMINAMATH_GPT_elongation_rate_significantly_improved_l496_49669


namespace NUMINAMATH_GPT_positive_difference_balances_l496_49605

noncomputable def laura_balance (L_0 : ℝ) (L_r : ℝ) (L_n : ℕ) (t : ℕ) : ℝ :=
  L_0 * (1 + L_r / L_n) ^ (L_n * t)

noncomputable def mark_balance (M_0 : ℝ) (M_r : ℝ) (t : ℕ) : ℝ :=
  M_0 * (1 + M_r * t)

theorem positive_difference_balances :
  let L_0 := 10000
  let L_r := 0.04
  let L_n := 2
  let t := 20
  let M_0 := 10000
  let M_r := 0.06
  abs ((laura_balance L_0 L_r L_n t) - (mark_balance M_0 M_r t)) = 80.40 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_balances_l496_49605


namespace NUMINAMATH_GPT_fido_yard_area_fraction_l496_49650

theorem fido_yard_area_fraction (r : ℝ) (h : r > 0) :
  let square_area := (2 * r)^2
  let reachable_area := π * r^2
  let fraction := reachable_area / square_area
  ∃ a b : ℕ, (fraction = (Real.sqrt a) / b * π) ∧ (a * b = 4) := by
  sorry

end NUMINAMATH_GPT_fido_yard_area_fraction_l496_49650


namespace NUMINAMATH_GPT_calculate_expression_l496_49662

theorem calculate_expression (x y : ℕ) (hx : x = 3) (hy : y = 4) : 
  (1 / (y + 1)) / (1 / (x + 2)) = 1 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l496_49662


namespace NUMINAMATH_GPT_volume_of_prism_l496_49697

-- Given conditions
def length : ℕ := 12
def width : ℕ := 8
def depth : ℕ := 8

-- Proving the volume of the rectangular prism
theorem volume_of_prism : length * width * depth = 768 := by
  sorry

end NUMINAMATH_GPT_volume_of_prism_l496_49697


namespace NUMINAMATH_GPT_correct_calculation_result_l496_49686

theorem correct_calculation_result :
  ∃ x : ℕ, 6 * x = 42 ∧ 3 * x = 21 :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_result_l496_49686


namespace NUMINAMATH_GPT_num_envelopes_requiring_charge_l496_49668

structure Envelope where
  length : ℕ
  height : ℕ

def requiresExtraCharge (env : Envelope) : Bool :=
  let ratio := env.length / env.height
  ratio < 3/2 ∨ ratio > 3

def envelopes : List Envelope :=
  [{ length := 7, height := 5 },  -- E
   { length := 10, height := 2 }, -- F
   { length := 8, height := 8 },  -- G
   { length := 12, height := 3 }] -- H

def countExtraChargedEnvelopes : ℕ :=
  envelopes.filter requiresExtraCharge |>.length

theorem num_envelopes_requiring_charge : countExtraChargedEnvelopes = 4 := by
  sorry

end NUMINAMATH_GPT_num_envelopes_requiring_charge_l496_49668


namespace NUMINAMATH_GPT_cost_of_pen_is_30_l496_49636

noncomputable def mean_expenditure_per_day : ℕ := 500
noncomputable def days_in_week : ℕ := 7
noncomputable def total_expenditure : ℕ := mean_expenditure_per_day * days_in_week

noncomputable def mon_expenditure : ℕ := 450
noncomputable def tue_expenditure : ℕ := 600
noncomputable def wed_expenditure : ℕ := 400
noncomputable def thurs_expenditure : ℕ := 500
noncomputable def sat_expenditure : ℕ := 550
noncomputable def sun_expenditure : ℕ := 300

noncomputable def fri_notebook_cost : ℕ := 50
noncomputable def fri_earphone_cost : ℕ := 620

noncomputable def total_non_fri_expenditure : ℕ := 
  mon_expenditure + tue_expenditure + wed_expenditure + 
  thurs_expenditure + sat_expenditure + sun_expenditure

noncomputable def fri_expenditure : ℕ := 
  total_expenditure - total_non_fri_expenditure

noncomputable def fri_pen_cost : ℕ := 
  fri_expenditure - (fri_earphone_cost + fri_notebook_cost)

theorem cost_of_pen_is_30 : fri_pen_cost = 30 :=
  sorry

end NUMINAMATH_GPT_cost_of_pen_is_30_l496_49636


namespace NUMINAMATH_GPT_smallest_perfect_cube_divisor_l496_49625

theorem smallest_perfect_cube_divisor 
  (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hpq : p ≠ q) 
  (hpr : p ≠ r) (hqr : q ≠ r) (s := 4) (hs : ¬ Nat.Prime s) 
  (hdiv : Nat.Prime 2) :
  ∃ n : ℕ, n = (p * q * r^2 * s)^3 ∧ ∀ m : ℕ, (∃ a b c d : ℕ, a = 3 ∧ b = 3 ∧ c = 6 ∧ d = 3 ∧ m = p^a * q^b * r^c * s^d) → m ≥ n :=
sorry

end NUMINAMATH_GPT_smallest_perfect_cube_divisor_l496_49625


namespace NUMINAMATH_GPT_percentage_increase_l496_49624

variable (T : ℕ) (total_time : ℕ)

theorem percentage_increase (h1 : T = 4) (h2 : total_time = 10) : 
  ∃ P : ℕ, (T + P / 100 * T = total_time - T) → P = 50 := 
by 
  sorry

end NUMINAMATH_GPT_percentage_increase_l496_49624


namespace NUMINAMATH_GPT_product_evaluation_l496_49659

theorem product_evaluation :
  (6 * 27^12 + 2 * 81^9) / 8000000^2 * (80 * 32^3 * 125^4) / (9^19 - 729^6) = 10 :=
by sorry

end NUMINAMATH_GPT_product_evaluation_l496_49659


namespace NUMINAMATH_GPT_calculation_of_nested_cuberoot_l496_49606

theorem calculation_of_nested_cuberoot (M : Real) (h : 1 < M) : (M^1 / 3 + 1 / 3)^(1 / 3 + 1 / 3)^(1 / 3 + 1 / 3)^(1 / 3 + 1 / 3)^(1 / 3 + 1 / 3) = M^(40 / 81) := 
by 
  sorry

end NUMINAMATH_GPT_calculation_of_nested_cuberoot_l496_49606


namespace NUMINAMATH_GPT_M_intersection_N_l496_49680

-- Definition of sets M and N
def M : Set ℝ := { x | x^2 + 2 * x - 8 < 0 }
def N : Set ℝ := { y | ∃ x : ℝ, y = 2^x }

-- Goal: Prove that M ∩ N = (0, 2)
theorem M_intersection_N :
  M ∩ N = { y | 0 < y ∧ y < 2 } :=
sorry

end NUMINAMATH_GPT_M_intersection_N_l496_49680


namespace NUMINAMATH_GPT_rectangle_area_constant_l496_49603

theorem rectangle_area_constant (d : ℝ) (length width : ℝ) (h_ratio : length / width = 5 / 2) (h_diag : d = Real.sqrt (length^2 + width^2)) :
  ∃ k : ℝ, (length * width) = k * d^2 ∧ k = 10 / 29 :=
by
  use 10 / 29
  sorry

end NUMINAMATH_GPT_rectangle_area_constant_l496_49603


namespace NUMINAMATH_GPT_sum_of_roots_l496_49640

theorem sum_of_roots : ∀ x : ℝ, x^2 - 2004 * x + 2021 = 0 → x = 2004 := by
  sorry

end NUMINAMATH_GPT_sum_of_roots_l496_49640


namespace NUMINAMATH_GPT_prime_between_30_and_40_with_remainder_7_l496_49696

theorem prime_between_30_and_40_with_remainder_7 (n : ℕ) 
  (h1 : Nat.Prime n) 
  (h2 : 30 < n) 
  (h3 : n < 40) 
  (h4 : n % 12 = 7) : 
  n = 31 := 
sorry

end NUMINAMATH_GPT_prime_between_30_and_40_with_remainder_7_l496_49696


namespace NUMINAMATH_GPT_probability_red_nonjoker_then_black_or_joker_l496_49602

theorem probability_red_nonjoker_then_black_or_joker :
  let total_cards := 60
  let red_non_joker := 26
  let black_or_joker := 40
  let total_ways_to_draw_two_cards := total_cards * (total_cards - 1)
  let probability := (red_non_joker * black_or_joker : ℚ) / total_ways_to_draw_two_cards
  probability = 5 / 17 :=
by
  -- Definitions for the conditions
  let total_cards := 60
  let red_non_joker := 26
  let black_or_joker := 40
  let total_ways_to_draw_two_cards := total_cards * (total_cards - 1)
  let probability := (red_non_joker * black_or_joker : ℚ) / total_ways_to_draw_two_cards
  -- Add sorry placeholder for proof
  sorry

end NUMINAMATH_GPT_probability_red_nonjoker_then_black_or_joker_l496_49602


namespace NUMINAMATH_GPT_triangle_angle_sum_depends_on_parallel_postulate_l496_49616

-- Definitions of conditions
def triangle_angle_sum_theorem (A B C : ℝ) : Prop :=
  A + B + C = 180

def parallel_postulate : Prop :=
  ∀ (l : ℝ) (P : ℝ), ∃! (m : ℝ), m ≠ l ∧ ∀ (Q : ℝ), Q ≠ P → (Q = l ∧ Q = m)

-- Theorem statement: proving the dependence of the triangle_angle_sum_theorem on the parallel_postulate
theorem triangle_angle_sum_depends_on_parallel_postulate: 
  ∀ (A B C : ℝ), parallel_postulate → triangle_angle_sum_theorem A B C :=
sorry

end NUMINAMATH_GPT_triangle_angle_sum_depends_on_parallel_postulate_l496_49616


namespace NUMINAMATH_GPT_probability_point_between_lines_l496_49611

theorem probability_point_between_lines :
  let l (x : ℝ) := -2 * x + 8
  let m (x : ℝ) := -3 * x + 9
  let area_l := 1 / 2 * 4 * 8
  let area_m := 1 / 2 * 3 * 9
  let area_between := area_l - area_m
  let probability := area_between / area_l
  probability = 0.16 :=
by
  sorry

end NUMINAMATH_GPT_probability_point_between_lines_l496_49611


namespace NUMINAMATH_GPT_cubed_identity_l496_49635

theorem cubed_identity (x : ℝ) (h : x + 1/x = 7) : x^3 + 1/x^3 = 322 := 
sorry

end NUMINAMATH_GPT_cubed_identity_l496_49635


namespace NUMINAMATH_GPT_point_B_number_l496_49660

theorem point_B_number (A B : ℤ) (hA : A = -2) (hB : abs (B - A) = 3) : B = 1 ∨ B = -5 :=
sorry

end NUMINAMATH_GPT_point_B_number_l496_49660


namespace NUMINAMATH_GPT_true_weight_of_C_l496_49629

theorem true_weight_of_C (A1 B1 C1 A2 B2 : ℝ) (l1 l2 m1 m2 A B C : ℝ)
  (hA1 : (A + m1) * l1 = (A1 + m2) * l2)
  (hB1 : (B + m1) * l1 = (B1 + m2) * l2)
  (hC1 : (C + m1) * l1 = (C1 + m2) * l2)
  (hA2 : (A2 + m1) * l1 = (A + m2) * l2)
  (hB2 : (B2 + m1) * l1 = (B + m2) * l2) :
  C = (C1 - A1) * Real.sqrt ((A2 - B2) / (A1 - B1)) + 
      (A1 * Real.sqrt (A2 - B2) + A2 * Real.sqrt (A1 - B1)) / 
      (Real.sqrt (A1 - B1) + Real.sqrt (A2 - B2)) :=
sorry

end NUMINAMATH_GPT_true_weight_of_C_l496_49629


namespace NUMINAMATH_GPT_general_term_formula_T_n_less_than_one_sixth_l496_49651

noncomputable def S (n : ℕ) : ℕ := n^2 + 2*n

def a (n : ℕ) : ℕ := if n = 0 then 0 else 2*n + 1

def b (n : ℕ) : ℕ := if n = 0 then 0 else 1 / (a n) * (a (n+1))

def T (n : ℕ) : ℝ := (Finset.range n).sum (λ k => (b k : ℝ))

theorem general_term_formula (n : ℕ) (hn : n ≠ 0) : 
  a n = 2*n + 1 :=
by sorry

theorem T_n_less_than_one_sixth (n : ℕ) : 
  T n < (1 / 6 : ℝ) :=
by sorry

end NUMINAMATH_GPT_general_term_formula_T_n_less_than_one_sixth_l496_49651


namespace NUMINAMATH_GPT_gcd_105_490_l496_49637

theorem gcd_105_490 : Nat.gcd 105 490 = 35 := by
sorry

end NUMINAMATH_GPT_gcd_105_490_l496_49637


namespace NUMINAMATH_GPT_roja_alone_time_l496_49685

theorem roja_alone_time (W : ℝ) (R : ℝ) :
  (1 / 60 + 1 / R = 1 / 35) → (R = 210) :=
by
  intros
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_roja_alone_time_l496_49685


namespace NUMINAMATH_GPT_ratio_sheep_to_horses_is_correct_l496_49621

-- Definitions of given conditions
def ounces_per_horse := 230
def total_ounces_per_day := 12880
def number_of_sheep := 16

-- Express the number of horses and the ratio of sheep to horses
def number_of_horses : ℕ := total_ounces_per_day / ounces_per_horse
def ratio_sheep_to_horses := number_of_sheep / number_of_horses

-- The main statement to be proved
theorem ratio_sheep_to_horses_is_correct : ratio_sheep_to_horses = 2 / 7 :=
by
  sorry

end NUMINAMATH_GPT_ratio_sheep_to_horses_is_correct_l496_49621


namespace NUMINAMATH_GPT_horse_drinking_water_l496_49677

-- Definitions and conditions

def initial_horses : ℕ := 3
def added_horses : ℕ := 5
def total_horses : ℕ := initial_horses + added_horses
def bathing_water_per_day : ℕ := 2
def total_water_28_days : ℕ := 1568
def days : ℕ := 28
def daily_water_total : ℕ := total_water_28_days / days

-- The statement looking to prove
theorem horse_drinking_water (D : ℕ) : 
  (total_horses * (D + bathing_water_per_day) = daily_water_total) → 
  D = 5 := 
by
  -- Add proof steps here
  sorry

end NUMINAMATH_GPT_horse_drinking_water_l496_49677


namespace NUMINAMATH_GPT_sum_50th_set_correct_l496_49679

noncomputable def sum_of_fiftieth_set : ℕ := 195 + 197

theorem sum_50th_set_correct : sum_of_fiftieth_set = 392 :=
by 
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_sum_50th_set_correct_l496_49679


namespace NUMINAMATH_GPT_house_cost_ratio_l496_49622

theorem house_cost_ratio {base_salary commission house_A_cost total_income : ℕ}
    (H_base_salary: base_salary = 3000)
    (H_commission: commission = 2)
    (H_house_A_cost: house_A_cost = 60000)
    (H_total_income: total_income = 8000)
    (H_total_sales_price: ℕ)
    (H_house_B_cost: ℕ)
    (H_house_C_cost: ℕ)
    (H_m: ℕ)
    (h1: total_income - base_salary = 5000)
    (h2: total_sales_price * commission / 100 = 5000)
    (h3: total_sales_price = 250000)
    (h4: house_B_cost = 3 * house_A_cost)
    (h5: total_sales_price = house_A_cost + house_B_cost + house_C_cost)
    (h6: house_C_cost = m * house_A_cost - 110000)
  : m = 2 :=
by
  sorry

end NUMINAMATH_GPT_house_cost_ratio_l496_49622


namespace NUMINAMATH_GPT_min_value_a_decreasing_range_of_a_x1_x2_l496_49657

noncomputable def f (a x : ℝ) := x / Real.log x - a * x

theorem min_value_a_decreasing :
  ∀ (a : ℝ), (∀ (x : ℝ), 1 < x → f a x <= 0) → a ≥ 1 / 4 :=
sorry

theorem range_of_a_x1_x2 :
  ∀ (a : ℝ), (∃ (x₁ x₂ : ℝ), e ≤ x₁ ∧ x₁ ≤ e^2 ∧ e ≤ x₂ ∧ x₂ ≤ e^2 ∧ f a x₁ ≤ f a x₂ + a)
  → a ≥ 1 / 2 - 1 / (4 * e^2) :=
sorry

end NUMINAMATH_GPT_min_value_a_decreasing_range_of_a_x1_x2_l496_49657


namespace NUMINAMATH_GPT_pumpkin_count_sunshine_orchard_l496_49648

def y (x : ℕ) : ℕ := 3 * x^2 + 12

theorem pumpkin_count_sunshine_orchard :
  y 14 = 600 :=
by
  sorry

end NUMINAMATH_GPT_pumpkin_count_sunshine_orchard_l496_49648


namespace NUMINAMATH_GPT_total_cost_l496_49626

def cost_of_items (x y : ℝ) : Prop :=
  (6 * x + 5 * y = 6.10) ∧ (3 * x + 4 * y = 4.60)

theorem total_cost (x y : ℝ) (h : cost_of_items x y) : 12 * x + 8 * y = 10.16 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_l496_49626


namespace NUMINAMATH_GPT_min_value_of_squares_l496_49615

theorem min_value_of_squares (a b s t : ℝ) (h1 : a + b = t) (h2 : a - b = s) :
  a^2 + b^2 = (t^2 + s^2) / 2 :=
sorry

end NUMINAMATH_GPT_min_value_of_squares_l496_49615


namespace NUMINAMATH_GPT_coeff_sum_eq_32_l496_49682

theorem coeff_sum_eq_32 (n : ℕ) (h : (2 : ℕ)^n = 32) : n = 5 :=
sorry

end NUMINAMATH_GPT_coeff_sum_eq_32_l496_49682


namespace NUMINAMATH_GPT_equation_has_three_solutions_l496_49687

theorem equation_has_three_solutions :
  ∃ (s : Finset ℝ), s.card = 3 ∧ ∀ x, x ∈ s ↔ x^2 * (x - 1) * (x - 2) = 0 := 
by
  sorry

end NUMINAMATH_GPT_equation_has_three_solutions_l496_49687


namespace NUMINAMATH_GPT_problem1_xy_value_problem2_min_value_l496_49676

-- Define the first problem conditions
def problem1 (x y : ℝ) : Prop :=
  x^2 - 2 * x * y + 2 * y^2 + 6 * y + 9 = 0

-- Prove that xy = 9 given the above condition
theorem problem1_xy_value (x y : ℝ) (h : problem1 x y) : x * y = 9 :=
  sorry

-- Define the second problem conditions
def expression (m : ℝ) : ℝ :=
  m^2 + 6 * m + 13

-- Prove that the minimum value of the expression is 4
theorem problem2_min_value : ∃ m, expression m = 4 :=
  sorry

end NUMINAMATH_GPT_problem1_xy_value_problem2_min_value_l496_49676


namespace NUMINAMATH_GPT_zoe_earns_per_candy_bar_l496_49656

-- Given conditions
def cost_of_trip : ℝ := 485
def grandma_contribution : ℝ := 250
def candy_bars_to_sell : ℝ := 188

-- Derived condition
def additional_amount_needed : ℝ := cost_of_trip - grandma_contribution

-- Assertion to prove
theorem zoe_earns_per_candy_bar :
  (additional_amount_needed / candy_bars_to_sell) = 1.25 :=
by
  sorry

end NUMINAMATH_GPT_zoe_earns_per_candy_bar_l496_49656


namespace NUMINAMATH_GPT_divisor_of_a_l496_49672

namespace MathProofProblem

-- Define the given problem
variable (a b c d : ℕ) -- Variables representing positive integers

-- Given conditions
variables (h_gcd_ab : Nat.gcd a b = 30)
variables (h_gcd_bc : Nat.gcd b c = 42)
variables (h_gcd_cd : Nat.gcd c d = 66)
variables (h_lcm_cd : Nat.lcm c d = 2772)
variables (h_gcd_da : 100 < Nat.gcd d a ∧ Nat.gcd d a < 150)

-- Target statement to prove
theorem divisor_of_a : 13 ∣ a :=
by
  sorry

end MathProofProblem

end NUMINAMATH_GPT_divisor_of_a_l496_49672


namespace NUMINAMATH_GPT_max_value_a_l496_49617

theorem max_value_a (a : ℝ) : 
  (∀ x : ℝ, x > -1 → x + 1 > 0 → x + 1 + 1 / (x + 1) - 2 ≥ a) → a ≤ 0 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_max_value_a_l496_49617


namespace NUMINAMATH_GPT_number_of_schools_l496_49649

-- Define the conditions
def is_median (a : ℕ) (n : ℕ) : Prop := 2 * a - 1 = n
def high_team_score (a b c : ℕ) : Prop := a > b ∧ a > c
def ranks (b c : ℕ) : Prop := b = 39 ∧ c = 67

-- Define the main problem
theorem number_of_schools (a n b c : ℕ) :
  is_median a n →
  high_team_score a b c →
  ranks b c →
  34 ≤ a ∧ a < 39 →
  2 * a ≡ 1 [MOD 3] →
  (n = 67 → a = 35) →
  (∀ m : ℕ, n = 3 * m + 1) →
  m = 23 :=
by
  sorry

end NUMINAMATH_GPT_number_of_schools_l496_49649


namespace NUMINAMATH_GPT_sequence_geq_four_l496_49655

theorem sequence_geq_four (a : ℕ → ℝ) (h0 : a 1 = 5) 
    (h1 : ∀ n ≥ 1, a (n+1) = (a n ^ 2 + 8 * a n + 16) / (4 * a n)) : 
    ∀ n ≥ 1, a n ≥ 4 := 
by
  sorry

end NUMINAMATH_GPT_sequence_geq_four_l496_49655


namespace NUMINAMATH_GPT_max_value_x2_plus_y2_l496_49609

theorem max_value_x2_plus_y2 (x y : ℝ) (h : 5 * x^2 + 4 * y^2 = 10 * x) : 
  x^2 + y^2 ≤ 4 :=
sorry

end NUMINAMATH_GPT_max_value_x2_plus_y2_l496_49609


namespace NUMINAMATH_GPT_two_digit_number_difference_perfect_square_l496_49653

theorem two_digit_number_difference_perfect_square (N : ℕ) (a b : ℕ)
  (h1 : N = 10 * a + b)
  (h2 : N % 100 = N)
  (h3 : 1 ≤ a ∧ a ≤ 9)
  (h4 : 0 ≤ b ∧ b ≤ 9)
  (h5 : (N - (10 * b + a : ℕ)) = 64) : 
  N = 90 := 
sorry

end NUMINAMATH_GPT_two_digit_number_difference_perfect_square_l496_49653


namespace NUMINAMATH_GPT_cos_pi_zero_l496_49632

theorem cos_pi_zero : ∃ f : ℝ → ℝ, (∀ x, f x = (Real.cos x) ^ 2 + Real.cos x) ∧ f Real.pi = 0 := by
  sorry

end NUMINAMATH_GPT_cos_pi_zero_l496_49632


namespace NUMINAMATH_GPT_total_profit_l496_49613

theorem total_profit (A B C : ℕ) (A_invest B_invest C_invest A_share : ℕ) (total_invest total_profit : ℕ)
  (h1 : A_invest = 6300)
  (h2 : B_invest = 4200)
  (h3 : C_invest = 10500)
  (h4 : A_share = 3630)
  (h5 : total_invest = A_invest + B_invest + C_invest)
  (h6 : total_profit * A_share = A_invest * total_invest) :
  total_profit = 12100 :=
by
  sorry

end NUMINAMATH_GPT_total_profit_l496_49613


namespace NUMINAMATH_GPT_units_digit_2749_987_l496_49661

def mod_units_digit (base : ℕ) (exp : ℕ) : ℕ :=
  (base % 10)^(exp % 2) % 10

theorem units_digit_2749_987 : mod_units_digit 2749 987 = 9 := 
by 
  sorry

end NUMINAMATH_GPT_units_digit_2749_987_l496_49661


namespace NUMINAMATH_GPT_range_of_b_l496_49681

noncomputable def f (x a b : ℝ) : ℝ :=
  x + a / x + b

theorem range_of_b (b : ℝ) :
  (∀ (a x : ℝ), (1/2 ≤ a ∧ a ≤ 2) ∧ (1/4 ≤ x ∧ x ≤ 1) → f x a b ≤ 10) →
  b ≤ 7 / 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_b_l496_49681


namespace NUMINAMATH_GPT_sum_of_digits_l496_49623

def original_sum := 943587 + 329430
def provided_sum := 1412017
def correct_sum_after_change (d e : ℕ) : ℕ := 
  let new_first := if d = 3 then 944587 else 943587
  let new_second := if d = 3 then 429430 else 329430
  new_first + new_second

theorem sum_of_digits (d e : ℕ) : d = 3 ∧ e = 4 → d + e = 7 :=
by
  intros
  exact sorry

end NUMINAMATH_GPT_sum_of_digits_l496_49623


namespace NUMINAMATH_GPT_max_dot_product_on_circle_l496_49634

theorem max_dot_product_on_circle :
  ∀ (P : ℝ × ℝ) (O : ℝ × ℝ) (A : ℝ × ℝ),
  O = (0, 0) →
  A = (-2, 0) →
  P.1 ^ 2 + P.2 ^ 2 = 1 →
  let AO := (2, 0)
  let AP := (P.1 + 2, P.2)
  ∃ α : ℝ, P = (Real.cos α, Real.sin α) ∧ 
  ∃ max_val : ℝ, max_val = 6 ∧ 
  (2 * (Real.cos α + 2) = max_val) :=
by
  intro P O A hO hA hP 
  let AO := (2, 0)
  let AP := (P.1 + 2, P.2)
  sorry

end NUMINAMATH_GPT_max_dot_product_on_circle_l496_49634


namespace NUMINAMATH_GPT_base_conversion_sum_l496_49633

def digit_C : ℕ := 12
def base_14_value : ℕ := 3 * 14^2 + 5 * 14^1 + 6 * 14^0
def base_13_value : ℕ := 4 * 13^2 + digit_C * 13^1 + 9 * 13^0

theorem base_conversion_sum :
  (base_14_value + base_13_value = 1505) :=
by sorry

end NUMINAMATH_GPT_base_conversion_sum_l496_49633


namespace NUMINAMATH_GPT_algebra_expression_value_l496_49699

theorem algebra_expression_value (x y : ℝ)
  (h1 : x + y = 3)
  (h2 : x * y = 1) :
  (5 * x + 3) - (2 * x * y - 5 * y) = 16 :=
by
  sorry

end NUMINAMATH_GPT_algebra_expression_value_l496_49699


namespace NUMINAMATH_GPT_sum_of_reciprocals_of_roots_l496_49673

theorem sum_of_reciprocals_of_roots (p q r : ℝ) (h : ∀ x : ℝ, (x^3 - x - 6 = 0) → (x = p ∨ x = q ∨ x = r)) :
  1 / (p + 2) + 1 / (q + 2) + 1 / (r + 2) = 11 / 12 :=
sorry

end NUMINAMATH_GPT_sum_of_reciprocals_of_roots_l496_49673


namespace NUMINAMATH_GPT_arithmetic_sequence_n_terms_l496_49628

theorem arithmetic_sequence_n_terms:
  ∀ (a₁ d aₙ n: ℕ), 
  a₁ = 6 → d = 3 → aₙ = 300 → aₙ = a₁ + (n - 1) * d → n = 99 :=
by
  intros a₁ d aₙ n h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_n_terms_l496_49628


namespace NUMINAMATH_GPT_sin_double_angle_l496_49627

theorem sin_double_angle (α : ℝ) (h : Real.cos (π / 4 - α) = 3 / 5) : Real.sin (2 * α) = -7 / 25 := by
  sorry

end NUMINAMATH_GPT_sin_double_angle_l496_49627


namespace NUMINAMATH_GPT_rhombus_area_is_correct_l496_49642

def calculate_rhombus_area (d1 d2 : ℕ) : ℕ :=
  (d1 * d2) / 2

theorem rhombus_area_is_correct :
  calculate_rhombus_area (3 * 6) (3 * 4) = 108 := by
  sorry

end NUMINAMATH_GPT_rhombus_area_is_correct_l496_49642


namespace NUMINAMATH_GPT_permutation_inequality_l496_49645

theorem permutation_inequality (a b c d : ℝ) (h : a * b * c * d > 0) :
  ∃ (x y z w : ℝ), (x = a ∨ x = b ∨ x = c ∨ x = d) ∧ (y = a ∨ y = b ∨ y = c ∨ y = d) ∧
  (z = a ∨ z = b ∨ z = c ∨ z = d) ∧ (w = a ∨ w = b ∨ w = c ∨ w = d) ∧ 
  2 * (x * y + z * w)^2 > (x^2 + y^2) * (z^2 + w^2) := 
sorry

end NUMINAMATH_GPT_permutation_inequality_l496_49645


namespace NUMINAMATH_GPT_sum_first_six_terms_l496_49698

variable (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ)

-- Define the existence of a geometric sequence with given properties
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Given Condition: a_3 = 2a_4 = 2
def cond1 (a : ℕ → ℝ) : Prop :=
  a 3 = 2 ∧ a 4 = 1

-- Define the sum of the first n terms of the sequence
def geometric_sum (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = a 0 * (1 - q ^ n) / (1 - q)

-- We need to prove that under these conditions, S_6 = 63/4
theorem sum_first_six_terms 
  (hq : q = 1 / 2) 
  (ha : is_geometric_sequence a q) 
  (hcond1 : cond1 a) 
  (hS : geometric_sum a q S) : 
  S 6 = 63 / 4 := 
sorry

end NUMINAMATH_GPT_sum_first_six_terms_l496_49698


namespace NUMINAMATH_GPT_largest_integer_le_1_l496_49670

theorem largest_integer_le_1 (x : ℤ) (h : (2 * x : ℚ) / 7 + 3 / 4 < 8 / 7) : x ≤ 1 :=
sorry

end NUMINAMATH_GPT_largest_integer_le_1_l496_49670


namespace NUMINAMATH_GPT_RahulPlayedMatchesSolver_l496_49678

noncomputable def RahulPlayedMatches (current_average new_average runs_in_today current_matches : ℕ) : ℕ :=
  let total_runs_before := current_average * current_matches
  let total_runs_after := total_runs_before + runs_in_today
  let total_matches_after := current_matches + 1
  total_runs_after / new_average

theorem RahulPlayedMatchesSolver:
  RahulPlayedMatches 52 54 78 12 = 12 :=
by
  sorry

end NUMINAMATH_GPT_RahulPlayedMatchesSolver_l496_49678


namespace NUMINAMATH_GPT_sin_x1_sub_x2_l496_49692

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

theorem sin_x1_sub_x2 (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < x₂) (h₃ : x₂ < Real.pi)
  (h₄ : f x₁ = 1 / 3) (h₅ : f x₂ = 1 / 3) : 
  Real.sin (x₁ - x₂) = - (2 * Real.sqrt 2) / 3 := 
sorry

end NUMINAMATH_GPT_sin_x1_sub_x2_l496_49692


namespace NUMINAMATH_GPT_Matias_longest_bike_ride_l496_49693

-- Define conditions in Lean
def blocks : ℕ := 4
def block_side_length : ℕ := 100
def streets : ℕ := 12

def Matias_route : Prop :=
  ∀ (intersections_used : ℕ), 
    intersections_used ≤ 4 → (streets - intersections_used/2 * 2) = 10

def correct_maximum_path_length : ℕ := 1000

-- Objective: Prove that given the conditions the longest route is 1000 meters
theorem Matias_longest_bike_ride :
  (100 * (streets - 2)) = correct_maximum_path_length :=
by
  sorry

end NUMINAMATH_GPT_Matias_longest_bike_ride_l496_49693


namespace NUMINAMATH_GPT_exchange_ways_10_dollar_l496_49689

theorem exchange_ways_10_dollar (p q : ℕ) (H : 2 * p + 5 * q = 200) : 
  ∃ (n : ℕ), n = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_exchange_ways_10_dollar_l496_49689


namespace NUMINAMATH_GPT_gray_areas_trees_count_l496_49631

noncomputable def totalTreesInGrayAreas (T : ℕ) (white1 white2 white3 : ℕ) : ℕ :=
  let gray2 := T - white2
  let gray3 := T - white3
  gray2 + gray3

theorem gray_areas_trees_count (T : ℕ) :
  T = 100 → totalTreesInGrayAreas T 100 82 90 = 26 :=
by sorry

end NUMINAMATH_GPT_gray_areas_trees_count_l496_49631


namespace NUMINAMATH_GPT_sqrt_ineq_l496_49644

open Real

theorem sqrt_ineq (α β : ℝ) (hα : 1 ≤ α) (hβ : 1 ≤ β) :
  Int.floor (sqrt α) + Int.floor (sqrt (α + β)) + Int.floor (sqrt β) ≥
    Int.floor (sqrt (2 * α)) + Int.floor (sqrt (2 * β)) := by sorry

end NUMINAMATH_GPT_sqrt_ineq_l496_49644


namespace NUMINAMATH_GPT_nonneg_int_solutions_eqn_l496_49646

theorem nonneg_int_solutions_eqn :
  { (x, y, z, w) : ℕ × ℕ × ℕ × ℕ | 2^x * 3^y - 5^z * 7^w = 1 } =
  {(1, 0, 0, 0), (3, 0, 0, 1), (1, 1, 1, 0), (2, 2, 1, 1)} :=
by {
  sorry
}

end NUMINAMATH_GPT_nonneg_int_solutions_eqn_l496_49646


namespace NUMINAMATH_GPT_isosceles_triangle_possible_values_of_x_l496_49652

open Real

-- Define the main statement
theorem isosceles_triangle_possible_values_of_x :
  ∀ x : ℝ, 
  (0 < x ∧ x < 90) ∧ 
  (sin (3*x) = sin (2*x) ∧ 
   sin (9*x) = sin (2*x)) 
  → x = 0 ∨ x = 180/11 ∨ x = 540/11 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_possible_values_of_x_l496_49652


namespace NUMINAMATH_GPT_impossible_divide_into_three_similar_l496_49654

noncomputable def sqrt2 : ℝ := Real.sqrt 2

def similar (x y : ℝ) : Prop :=
  x ≤ sqrt2 * y

theorem impossible_divide_into_three_similar (N : ℝ) :
  ¬ ∃ (x y z : ℝ), x + y + z = N ∧ similar x y ∧ similar y z ∧ similar x z := 
by
  sorry

end NUMINAMATH_GPT_impossible_divide_into_three_similar_l496_49654


namespace NUMINAMATH_GPT_term_number_l496_49684

theorem term_number (n : ℕ) : 
  (n ≥ 1) ∧ (5 * Real.sqrt 3 = Real.sqrt (3 + 4 * (n - 1))) → n = 19 :=
by
  intro h
  let h1 := h.1
  let h2 := h.2
  have h3 : (5 * Real.sqrt 3)^2 = (Real.sqrt (3 + 4 * (n - 1)))^2 := by sorry
  sorry

end NUMINAMATH_GPT_term_number_l496_49684


namespace NUMINAMATH_GPT_yuna_survey_l496_49639

theorem yuna_survey :
  let M := 27
  let K := 28
  let B := 22
  M + K - B = 33 :=
by
  sorry

end NUMINAMATH_GPT_yuna_survey_l496_49639


namespace NUMINAMATH_GPT_last_two_digits_A_pow_20_l496_49612

/-- 
Proof that for any even number A not divisible by 10, 
the last two digits of A^20 are 76.
--/
theorem last_two_digits_A_pow_20 (A : ℕ) (h_even : A % 2 = 0) (h_not_div_by_10 : A % 10 ≠ 0) : 
  (A ^ 20) % 100 = 76 :=
by
  sorry

end NUMINAMATH_GPT_last_two_digits_A_pow_20_l496_49612


namespace NUMINAMATH_GPT_girls_count_l496_49691

-- Definition of the conditions
variables (B G : ℕ)

def college_conditions (B G : ℕ) : Prop :=
  (B + G = 416) ∧ (B = (8 * G) / 5)

-- Statement to prove
theorem girls_count (B G : ℕ) (h : college_conditions B G) : G = 160 :=
by
  sorry

end NUMINAMATH_GPT_girls_count_l496_49691


namespace NUMINAMATH_GPT_find_minimum_n_l496_49618

variable {a_1 d : ℝ}
variable {S : ℕ → ℝ}

def is_arithmetic_sequence (a_1 d : ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n / 2 : ℝ) * (2 * a_1 + (n - 1) * d)

def condition1 (a_1 : ℝ) : Prop := a_1 < 0

def condition2 (S : ℕ → ℝ) : Prop := S 7 = S 13

theorem find_minimum_n (a_1 d : ℝ) (S : ℕ → ℝ)
  (h_arith_seq : is_arithmetic_sequence a_1 d S)
  (h_a1_neg : condition1 a_1)
  (h_s7_eq_s13 : condition2 S) :
  ∃ n : ℕ, n = 10 ∧ ∀ m : ℕ, S n ≤ S m := 
sorry

end NUMINAMATH_GPT_find_minimum_n_l496_49618


namespace NUMINAMATH_GPT_find_theta_l496_49638

theorem find_theta (theta : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ 2 * π) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → x^3 * Real.cos θ - x^2 * (1 - x) + (1 - x)^3 * Real.sin θ > 0) →
  θ > π / 12 ∧ θ < 5 * π / 12 :=
by
  sorry

end NUMINAMATH_GPT_find_theta_l496_49638


namespace NUMINAMATH_GPT_stratified_sampling_l496_49643

noncomputable def combination (n k : ℕ) : ℕ := Nat.choose n k

theorem stratified_sampling :
  let junior_students := 400
  let senior_students := 200
  let total_sample_size := 60
  let junior_sample_size := (2 * total_sample_size) / 3
  let senior_sample_size := total_sample_size / 3
  combination junior_students junior_sample_size * combination senior_students senior_sample_size =
    combination 400 40 * combination 200 20 :=
by
  let junior_students := 400
  let senior_students := 200
  let total_sample_size := 60
  let junior_sample_size := (2 * total_sample_size) / 3
  let senior_sample_size := total_sample_size / 3
  exact sorry

end NUMINAMATH_GPT_stratified_sampling_l496_49643


namespace NUMINAMATH_GPT_hadassah_painting_time_l496_49663

noncomputable def time_to_paint_all_paintings (time_small_paintings time_large_paintings time_additional_small_paintings time_additional_large_paintings : ℝ) : ℝ :=
  time_small_paintings + time_large_paintings + time_additional_small_paintings + time_additional_large_paintings

theorem hadassah_painting_time :
  let time_small_paintings := 6
  let time_large_paintings := 8
  let time_per_small_painting := 6 / 12 -- = 0.5
  let time_per_large_painting := 8 / 6 -- ≈ 1.33
  let time_additional_small_paintings := 15 * time_per_small_painting -- = 7.5
  let time_additional_large_paintings := 10 * time_per_large_painting -- ≈ 13.3
  time_to_paint_all_paintings time_small_paintings time_large_paintings time_additional_small_paintings time_additional_large_paintings = 34.8 :=
by
  sorry

end NUMINAMATH_GPT_hadassah_painting_time_l496_49663

import Mathlib

namespace NUMINAMATH_GPT_runners_meet_again_l137_13708

theorem runners_meet_again :
    ∀ t : ℝ,
      t ≠ 0 →
      (∃ k : ℤ, 3.8 * t - 4 * t = 400 * k) ∧
      (∃ m : ℤ, 4.2 * t - 4 * t = 400 * m) ↔
      t = 2000 := 
by
  sorry

end NUMINAMATH_GPT_runners_meet_again_l137_13708


namespace NUMINAMATH_GPT_point_P_location_l137_13773

theorem point_P_location (a b : ℝ) : (∃ x y : ℝ, a * x + b * y = 1 ∧ x^2 + y^2 = 1) → a^2 + b^2 > 1 :=
by sorry

end NUMINAMATH_GPT_point_P_location_l137_13773


namespace NUMINAMATH_GPT_transformed_curve_is_circle_l137_13707

open Real

def polar_curve (ρ θ : ℝ) : Prop :=
  ρ^2 = 12 / (3 * cos θ^2 + 4 * sin θ^2)

def cartesian_curve (x y: ℝ) : Prop :=
  3 * x^2 + 4 * y^2 = 12

def transformation (x y x' y' : ℝ) : Prop :=
  x' = x / 2 ∧ y' = y * sqrt (3 / 3)

theorem transformed_curve_is_circle (x y x' y' : ℝ) 
  (h1: cartesian_curve x y) (h2: transformation x y x' y') : 
  (x'^2 + y'^2 = 1) :=
sorry

end NUMINAMATH_GPT_transformed_curve_is_circle_l137_13707


namespace NUMINAMATH_GPT_modulus_of_complex_l137_13775

noncomputable def modulus (z : Complex) : Real :=
  Complex.abs z

theorem modulus_of_complex :
  ∀ (i : Complex) (z : Complex), i = Complex.I → z = i * (2 - i) → modulus z = Real.sqrt 5 :=
by
  intros i z hi hz
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_modulus_of_complex_l137_13775


namespace NUMINAMATH_GPT_price_per_cake_l137_13774

def number_of_cakes_per_day := 4
def number_of_working_days_per_week := 5
def total_amount_collected := 640
def number_of_weeks := 4

theorem price_per_cake :
  let total_cakes_per_week := number_of_cakes_per_day * number_of_working_days_per_week
  let total_cakes_in_four_weeks := total_cakes_per_week * number_of_weeks
  let price_per_cake := total_amount_collected / total_cakes_in_four_weeks
  price_per_cake = 8 := by
sorry

end NUMINAMATH_GPT_price_per_cake_l137_13774


namespace NUMINAMATH_GPT_number_of_games_is_15_l137_13771

-- Definition of the given conditions
def total_points : ℕ := 345
def avg_points_per_game : ℕ := 4 + 10 + 9
def number_of_games (total_points : ℕ) (avg_points_per_game : ℕ) := total_points / avg_points_per_game

-- The theorem stating the proof problem
theorem number_of_games_is_15 : number_of_games total_points avg_points_per_game = 15 :=
by
  -- Skipping the proof as only the statement is required
  sorry

end NUMINAMATH_GPT_number_of_games_is_15_l137_13771


namespace NUMINAMATH_GPT_find_line_eq_l137_13799

theorem find_line_eq (x y : ℝ) (h : x^2 + y^2 - 4 * x - 5 = 0) 
(mid_x mid_y : ℝ) (mid_point : mid_x = 3 ∧ mid_y = 1) : 
x + y - 4 = 0 := 
sorry

end NUMINAMATH_GPT_find_line_eq_l137_13799


namespace NUMINAMATH_GPT_lighting_candles_correct_l137_13761

noncomputable def time_to_light_candles (initial_length : ℝ) : ℝ :=
  let burn_rate_1 := initial_length / 300
  let burn_rate_2 := initial_length / 240
  let t := (5 * 60 + 43) - (5 * 60) -- 11:17 AM is 342.857 minutes before 5 PM
  if ((initial_length - burn_rate_2 * t) = 3 * (initial_length - burn_rate_1 * t)) then 11 + 17 / 60 else 0 -- Check if the condition is met

theorem lighting_candles_correct :
  ∀ (initial_length : ℝ), time_to_light_candles initial_length = 11 + 17 / 60 :=
by
  intros initial_length
  sorry  -- Proof goes here

end NUMINAMATH_GPT_lighting_candles_correct_l137_13761


namespace NUMINAMATH_GPT_log_comparison_l137_13745

theorem log_comparison (a b c : ℝ) (h₁ : a = Real.log 6 / Real.log 4) (h₂ : b = Real.log 3 / Real.log 2) (h₃ : c = 3/2) : b > c ∧ c > a := 
by 
  sorry

end NUMINAMATH_GPT_log_comparison_l137_13745


namespace NUMINAMATH_GPT_total_students_in_school_l137_13787

theorem total_students_in_school (s : ℕ) (below_8 above_8 : ℕ) (students_8 : ℕ)
  (h1 : below_8 = 20 * s / 100) 
  (h2 : above_8 = 2 * students_8 / 3) 
  (h3 : students_8 = 48) 
  (h4 : s = students_8 + above_8 + below_8) : 
  s = 100 := 
by 
  sorry 

end NUMINAMATH_GPT_total_students_in_school_l137_13787


namespace NUMINAMATH_GPT_symmetric_points_x_axis_l137_13706

theorem symmetric_points_x_axis (a b : ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ) 
  (hP : P = (a - 3, 1)) (hQ : Q = (2, b + 1)) (hSymm : P.1 = Q.1 ∧ P.2 = -Q.2) :
  a + b = 3 :=
by 
  sorry

end NUMINAMATH_GPT_symmetric_points_x_axis_l137_13706


namespace NUMINAMATH_GPT_polygon_problem_l137_13722

theorem polygon_problem 
  (D : ℕ → ℕ) (m x : ℕ) 
  (H1 : ∀ n, D n = n * (n - 3) / 2)
  (H2 : D m = 3 * D (m - 3))
  (H3 : D (m + x) = 7 * D m) :
  m = 9 ∧ x = 12 ∧ (m + x) - m = 12 :=
by {
  -- the proof would go here, skipped as per the instructions.
  sorry
}

end NUMINAMATH_GPT_polygon_problem_l137_13722


namespace NUMINAMATH_GPT_sqrt_a_plus_sqrt_b_eq_3_l137_13795

theorem sqrt_a_plus_sqrt_b_eq_3 (a b : ℝ) (h : (Real.sqrt a + Real.sqrt b) * (Real.sqrt a + Real.sqrt b - 2) = 3) : Real.sqrt a + Real.sqrt b = 3 :=
sorry

end NUMINAMATH_GPT_sqrt_a_plus_sqrt_b_eq_3_l137_13795


namespace NUMINAMATH_GPT_total_sonnets_written_l137_13724

-- Definitions of conditions given in the problem
def lines_per_sonnet : ℕ := 14
def sonnets_read : ℕ := 7
def unread_lines : ℕ := 70

-- Definition of a measuring line for further calculation
def unread_sonnets : ℕ := unread_lines / lines_per_sonnet

-- The assertion we need to prove
theorem total_sonnets_written : 
  unread_sonnets + sonnets_read = 12 := by 
  sorry

end NUMINAMATH_GPT_total_sonnets_written_l137_13724


namespace NUMINAMATH_GPT_johns_monthly_earnings_l137_13715

variable (work_days : ℕ) (hours_per_day : ℕ) (former_wage : ℝ) (raise_percentage : ℝ) (days_in_month : ℕ)

def johns_earnings (work_days hours_per_day : ℕ) (former_wage raise_percentage : ℝ) (days_in_month : ℕ) : ℝ :=
  let days_worked := days_in_month / 2
  let total_hours := days_worked * hours_per_day
  let raise := former_wage * raise_percentage
  let new_wage := former_wage + raise
  total_hours * new_wage

theorem johns_monthly_earnings (work_days : ℕ := 15) (hours_per_day : ℕ := 12) (former_wage : ℝ := 20) (raise_percentage : ℝ := 0.3) (days_in_month : ℕ := 30) :
  johns_earnings work_days hours_per_day former_wage raise_percentage days_in_month = 4680 :=
by
  sorry

end NUMINAMATH_GPT_johns_monthly_earnings_l137_13715


namespace NUMINAMATH_GPT_find_f2_l137_13779

def f (x : ℝ) (a b : ℝ) := x^5 + a * x^3 + b * x - 8

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 3) : f 2 a b = -19 :=
by sorry

end NUMINAMATH_GPT_find_f2_l137_13779


namespace NUMINAMATH_GPT_sum_first_4_terms_l137_13798

theorem sum_first_4_terms 
  (a_1 : ℚ) 
  (q : ℚ) 
  (h1 : a_1 * q - a_1 * q^2 = -2) 
  (h2 : a_1 + a_1 * q^2 = 10 / 3) 
  : a_1 * (1 + q + q^2 + q^3) = 40 / 3 := sorry

end NUMINAMATH_GPT_sum_first_4_terms_l137_13798


namespace NUMINAMATH_GPT_min_sum_rect_box_l137_13747

-- Define the main theorem with the given constraints
theorem min_sum_rect_box (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_vol : a * b * c = 2002) : a + b + c ≥ 38 :=
  sorry

end NUMINAMATH_GPT_min_sum_rect_box_l137_13747


namespace NUMINAMATH_GPT_clothing_store_gross_profit_l137_13778

theorem clothing_store_gross_profit :
  ∃ S : ℝ, S = 81 + 0.25 * S ∧
  ∃ new_price : ℝ,
    new_price = S - 0.20 * S ∧
    ∃ profit : ℝ,
      profit = new_price - 81 ∧
      profit = 5.40 :=
by
  sorry

end NUMINAMATH_GPT_clothing_store_gross_profit_l137_13778


namespace NUMINAMATH_GPT_a0_a2_a4_sum_l137_13797

theorem a0_a2_a4_sum (a0 a1 a2 a3 a4 a5 : ℝ) :
  (∀ x : ℝ, (2 * x - 3) ^ 5 = a0 + a1 * (x - 1) + a2 * (x - 1) ^ 2 + a3 * (x - 1) ^ 3 + a4 * (x - 1) ^ 4 + a5 * (x - 1) ^ 5) →
  a0 + a2 + a4 = -121 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_a0_a2_a4_sum_l137_13797


namespace NUMINAMATH_GPT_find_a_l137_13759

noncomputable def f (a x : ℝ) : ℝ := Real.log (x + 1) / Real.log a

theorem find_a (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) 
  (h₃ : ∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f a x ∧ f a x ≤ 1) : a = 2 :=
sorry

end NUMINAMATH_GPT_find_a_l137_13759


namespace NUMINAMATH_GPT_slower_train_speed_l137_13733

theorem slower_train_speed (v : ℝ) (L : ℝ) (faster_speed_km_hr : ℝ) (time_sec : ℝ) (relative_speed : ℝ) 
  (hL : L = 70) (hfaster_speed_km_hr : faster_speed_km_hr = 50)
  (htime_sec : time_sec = 36) (hrelative_speed : relative_speed = (faster_speed_km_hr - v) * (1000 / 3600)) :
  140 = relative_speed * time_sec → v = 36 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_slower_train_speed_l137_13733


namespace NUMINAMATH_GPT_reciprocal_neg_two_l137_13740

theorem reciprocal_neg_two : 1 / (-2) = - (1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_neg_two_l137_13740


namespace NUMINAMATH_GPT_greta_hours_worked_l137_13765

-- Define the problem conditions
def greta_hourly_rate := 12
def lisa_hourly_rate := 15
def lisa_hours_to_equal_greta_earnings := 32
def greta_earnings (hours_worked : ℕ) := greta_hourly_rate * hours_worked
def lisa_earnings := lisa_hourly_rate * lisa_hours_to_equal_greta_earnings

-- Problem statement
theorem greta_hours_worked (G : ℕ) (H : greta_earnings G = lisa_earnings) : G = 40 := by
  sorry

end NUMINAMATH_GPT_greta_hours_worked_l137_13765


namespace NUMINAMATH_GPT_four_fives_to_hundred_case1_four_fives_to_hundred_case2_l137_13782

theorem four_fives_to_hundred_case1 : (5 + 5) * (5 + 5) = 100 :=
by sorry

theorem four_fives_to_hundred_case2 : (5 * 5 - 5) * 5 = 100 :=
by sorry

end NUMINAMATH_GPT_four_fives_to_hundred_case1_four_fives_to_hundred_case2_l137_13782


namespace NUMINAMATH_GPT_value_of_expression_l137_13753

variable (x y : ℝ)

theorem value_of_expression (h1 : x + y = 3) (h2 : x * y = 1) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = 849 := by sorry

end NUMINAMATH_GPT_value_of_expression_l137_13753


namespace NUMINAMATH_GPT_meal_combinations_correct_l137_13742

-- Define the given conditions
def number_of_entrees : Nat := 4
def number_of_drinks : Nat := 4
def number_of_desserts : Nat := 2

-- Define the total number of meal combinations to prove
def total_meal_combinations : Nat := number_of_entrees * number_of_drinks * number_of_desserts

-- The theorem we want to prove
theorem meal_combinations_correct : total_meal_combinations = 32 := 
by 
  sorry

end NUMINAMATH_GPT_meal_combinations_correct_l137_13742


namespace NUMINAMATH_GPT_combine_ingredients_l137_13750

theorem combine_ingredients : 
  ∃ (water flour salt : ℕ), 
    water = 10 ∧ flour = 16 ∧ salt = 1 / 2 * flour ∧ 
    (water + flour = 26) ∧ (salt = 8) :=
by
  sorry

end NUMINAMATH_GPT_combine_ingredients_l137_13750


namespace NUMINAMATH_GPT_first_book_cost_correct_l137_13776

noncomputable def cost_of_first_book (x : ℝ) : Prop :=
  let total_cost := x + 6.5
  let given_amount := 20
  let change_received := 8
  total_cost = given_amount - change_received → x = 5.5

theorem first_book_cost_correct : cost_of_first_book 5.5 :=
by
  sorry

end NUMINAMATH_GPT_first_book_cost_correct_l137_13776


namespace NUMINAMATH_GPT_average_squares_of_first_10_multiples_of_7_correct_l137_13701

def first_10_multiples_of_7 : List ℕ := List.map (fun n => 7 * n) (List.range 10)

def squares (l : List ℕ) : List ℕ := List.map (fun n => n * n) l

def sum (l : List ℕ) : ℕ := List.foldr (· + ·) 0 l

theorem average_squares_of_first_10_multiples_of_7_correct :
  (sum (squares first_10_multiples_of_7) / 10 : ℚ) = 1686.5 :=
by
  sorry

end NUMINAMATH_GPT_average_squares_of_first_10_multiples_of_7_correct_l137_13701


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l137_13710

noncomputable def a1 := 3
noncomputable def S (n : ℕ) (a1 d : ℕ) : ℕ := n * (a1 + (n - 1) * d / 2)

theorem arithmetic_sequence_problem (d : ℕ) 
  (h1 : S 1 a1 d = 3) 
  (h2 : S 1 a1 d / 2 + S 4 a1 d / 4 = 18) : 
  S 5 a1 d = 75 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l137_13710


namespace NUMINAMATH_GPT_right_triangular_pyramid_property_l137_13758

theorem right_triangular_pyramid_property
  (S1 S2 S3 S : ℝ)
  (right_angle_face1_area : S1 = S1) 
  (right_angle_face2_area : S2 = S2) 
  (right_angle_face3_area : S3 = S3) 
  (oblique_face_area : S = S) :
  S1^2 + S2^2 + S3^2 = S^2 := 
sorry

end NUMINAMATH_GPT_right_triangular_pyramid_property_l137_13758


namespace NUMINAMATH_GPT_interest_for_20000_l137_13726

-- Definition of simple interest
def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ := P * r * t

variables (P1 P2 I1 I2 r : ℝ)
-- Given conditions
def h1 := (P1 = 5000)
def h2 := (I1 = 250)
def h3 := (r = I1 / P1)
-- Question condition
def h4 := (P2 = 20000)
def t := 1

theorem interest_for_20000 :
  P1 = 5000 →
  I1 = 250 →
  P2 = 20000 →
  r = I1 / P1 →
  simple_interest P2 r t = 1000 :=
by
  intros
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_interest_for_20000_l137_13726


namespace NUMINAMATH_GPT_max_gold_coins_l137_13721

theorem max_gold_coins (n k : ℕ) (h1 : n = 13 * k + 3) (h2 : n < 110) : n ≤ 107 :=
by
  sorry

end NUMINAMATH_GPT_max_gold_coins_l137_13721


namespace NUMINAMATH_GPT_dice_total_correct_l137_13713

-- Define the problem conditions
def IvanDice (x : ℕ) : ℕ := x
def JerryDice (x : ℕ) : ℕ := (1 / 2 * x) ^ 2

-- Define the total dice function
def totalDice (x : ℕ) : ℕ := IvanDice x + JerryDice x

-- The theorem to prove the answer
theorem dice_total_correct (x : ℕ) : totalDice x = x + (1 / 4) * x ^ 2 := 
  sorry

end NUMINAMATH_GPT_dice_total_correct_l137_13713


namespace NUMINAMATH_GPT_rectangle_area_l137_13719

-- Define the conditions as hypotheses in Lean 4
variable (x : ℤ)
variable (area : ℤ := 864)
variable (width : ℤ := x - 12)

-- State the theorem to prove the relation between length and area
theorem rectangle_area (h : x * width = area) : x * (x - 12) = 864 :=
by 
  sorry

end NUMINAMATH_GPT_rectangle_area_l137_13719


namespace NUMINAMATH_GPT_determine_a_l137_13764

noncomputable def f (x : ℝ) : ℝ := Real.exp (abs (x - 1)) + 1

theorem determine_a (a : ℝ) (h : f a = 2) : a = 1 :=
by
  sorry

end NUMINAMATH_GPT_determine_a_l137_13764


namespace NUMINAMATH_GPT_first_divisibility_second_divisibility_l137_13788

variable {n : ℕ}
variable (h : n > 0)

theorem first_divisibility :
  17 ∣ (5 * 3^(4*n+1) + 2^(6*n+1)) :=
sorry

theorem second_divisibility :
  32 ∣ (25 * 7^(2*n+1) + 3^(4*n)) :=
sorry

end NUMINAMATH_GPT_first_divisibility_second_divisibility_l137_13788


namespace NUMINAMATH_GPT_intersection_P_Q_l137_13796

def P : Set ℝ := {x | x^2 - 9 < 0}
def Q : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}

theorem intersection_P_Q :
  {x : ℤ | (x : ℝ) ∈ P} ∩ Q = {-1, 0, 1, 2} := 
by
  sorry

end NUMINAMATH_GPT_intersection_P_Q_l137_13796


namespace NUMINAMATH_GPT_radius_of_isosceles_tangent_circle_l137_13755

noncomputable def R : ℝ := 2 * Real.sqrt 3

variables (x : ℝ) (AB AC BD AD DC r : ℝ)

def is_isosceles (AB BC : ℝ) : Prop := AB = BC
def is_tangent (r : ℝ) (x : ℝ) : Prop := r = 2.4 * x

theorem radius_of_isosceles_tangent_circle
  (h_isosceles: is_isosceles AB BC)
  (h_area: 1/2 * AC * BD = 25)
  (h_height_ratio: BD / AC = 3 / 8)
  (h_AD_DC: AD = DC)
  (h_AC: AC = 8 * x)
  (h_BD: BD = 3 * x)
  (h_radius: is_tangent r x):
  r = R :=
sorry

end NUMINAMATH_GPT_radius_of_isosceles_tangent_circle_l137_13755


namespace NUMINAMATH_GPT_overall_weighted_defective_shipped_percentage_l137_13790

theorem overall_weighted_defective_shipped_percentage
  (defective_A : ℝ := 0.06) (shipped_A : ℝ := 0.04) (prod_A : ℝ := 0.30)
  (defective_B : ℝ := 0.09) (shipped_B : ℝ := 0.06) (prod_B : ℝ := 0.50)
  (defective_C : ℝ := 0.12) (shipped_C : ℝ := 0.07) (prod_C : ℝ := 0.20) :
  prod_A * defective_A * shipped_A + prod_B * defective_B * shipped_B + prod_C * defective_C * shipped_C = 0.00510 :=
by
  sorry

end NUMINAMATH_GPT_overall_weighted_defective_shipped_percentage_l137_13790


namespace NUMINAMATH_GPT_dryer_cost_l137_13772

theorem dryer_cost (W D : ℕ) (h1 : W + D = 600) (h2 : W = 3 * D) : D = 150 :=
by
  sorry

end NUMINAMATH_GPT_dryer_cost_l137_13772


namespace NUMINAMATH_GPT_positive_diff_of_squares_l137_13703

theorem positive_diff_of_squares (a b : ℕ) (h1 : a + b = 40) (h2 : a - b = 10) : a^2 - b^2 = 400 := by
  sorry

end NUMINAMATH_GPT_positive_diff_of_squares_l137_13703


namespace NUMINAMATH_GPT_lcm_division_l137_13723

open Nat

-- Define the LCM function for a list of integers
def list_lcm (l : List Nat) : Nat := l.foldr (fun a b => Nat.lcm a b) 1

-- Define the sequence ranges
def range1 := List.range' 20 21 -- From 20 to 40 inclusive
def range2 := List.range' 41 10 -- From 41 to 50 inclusive

-- Define P and Q
def P : Nat := list_lcm range1
def Q : Nat := Nat.lcm P (list_lcm range2)

-- The theorem statement
theorem lcm_division : (Q / P) = 55541 := by
  sorry

end NUMINAMATH_GPT_lcm_division_l137_13723


namespace NUMINAMATH_GPT_length_greater_than_width_l137_13749

theorem length_greater_than_width
  (perimeter : ℕ)
  (P : perimeter = 150)
  (l w difference : ℕ)
  (L : l = 60)
  (W : w = 45)
  (D : difference = l - w) :
  difference = 15 :=
by
  sorry

end NUMINAMATH_GPT_length_greater_than_width_l137_13749


namespace NUMINAMATH_GPT_amy_uploaded_photos_l137_13784

theorem amy_uploaded_photos (albums photos_per_album : ℕ) (h1 : albums = 9) (h2 : photos_per_album = 20) :
  albums * photos_per_album = 180 :=
by {
  sorry
}

end NUMINAMATH_GPT_amy_uploaded_photos_l137_13784


namespace NUMINAMATH_GPT_ellipse_equation_l137_13727

def major_axis_length (a : ℝ) := 2 * a = 8
def eccentricity (c a : ℝ) := c / a = 3 / 4

theorem ellipse_equation (a b c x y : ℝ) (h1 : major_axis_length a)
    (h2 : eccentricity c a) (h3 : b^2 = a^2 - c^2) :
    (x^2 / 16 + y^2 / 7 = 1 ∨ x^2 / 7 + y^2 / 16 = 1) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_equation_l137_13727


namespace NUMINAMATH_GPT_angle_of_inclination_l137_13709

theorem angle_of_inclination (m : ℝ) (h : m = -1) : 
  ∃ α : ℝ, α = 3 * Real.pi / 4 := 
sorry

end NUMINAMATH_GPT_angle_of_inclination_l137_13709


namespace NUMINAMATH_GPT_sin_810_cos_neg60_l137_13769

theorem sin_810_cos_neg60 :
  Real.sin (810 * Real.pi / 180) + Real.cos (-60 * Real.pi / 180) = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_810_cos_neg60_l137_13769


namespace NUMINAMATH_GPT_quadratic_roots_sum_cubes_l137_13751

theorem quadratic_roots_sum_cubes (k : ℚ) (a b : ℚ) 
  (h1 : 4 * a^2 + 5 * a + k = 0) 
  (h2 : 4 * b^2 + 5 * b + k = 0) 
  (h3 : a^3 + b^3 = a + b) :
  k = 9 / 4 :=
by {
  -- Lean code requires the proof, here we use sorry to skip it
  sorry
}

end NUMINAMATH_GPT_quadratic_roots_sum_cubes_l137_13751


namespace NUMINAMATH_GPT_good_numbers_l137_13741

def is_good (n : ℕ) : Prop :=
  ∀ d, d ∣ n → (d + 1) ∣ (n + 1)

theorem good_numbers (n : ℕ) :
  is_good n ↔ n = 1 ∨ (Nat.Prime n ∧ Odd n) :=
by
  sorry

end NUMINAMATH_GPT_good_numbers_l137_13741


namespace NUMINAMATH_GPT_highest_more_than_lowest_by_37_5_percent_l137_13777

variables (highest_price lowest_price : ℝ)

theorem highest_more_than_lowest_by_37_5_percent
  (h_highest : highest_price = 22)
  (h_lowest : lowest_price = 16) :
  ((highest_price - lowest_price) / lowest_price) * 100 = 37.5 :=
by
  sorry

end NUMINAMATH_GPT_highest_more_than_lowest_by_37_5_percent_l137_13777


namespace NUMINAMATH_GPT_num_sides_polygon_l137_13781

theorem num_sides_polygon (n : ℕ) (h : (n - 2) * 180 = 1260) : n = 9 := by
  sorry

end NUMINAMATH_GPT_num_sides_polygon_l137_13781


namespace NUMINAMATH_GPT_find_num_white_balls_l137_13738

theorem find_num_white_balls
  (W : ℕ)
  (total_balls : ℕ := 15 + W)
  (prob_black : ℚ := 7 / total_balls)
  (given_prob : ℚ := 0.38095238095238093) :
  prob_black = given_prob → W = 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_num_white_balls_l137_13738


namespace NUMINAMATH_GPT_ratio_of_areas_l137_13785

variable (A B : ℝ)

-- Conditions
def total_area := A + B = 700
def smaller_part_area := B = 315

-- Problem Statement
theorem ratio_of_areas (h_total : total_area A B) (h_small : smaller_part_area B) :
  (A - B) / ((A + B) / 2) = 1 / 5 := by
sorry

end NUMINAMATH_GPT_ratio_of_areas_l137_13785


namespace NUMINAMATH_GPT_find_noon_temperature_l137_13732

theorem find_noon_temperature (T T₄₀₀ T₈₀₀ : ℝ) 
  (h1 : T₄₀₀ = T + 8)
  (h2 : T₈₀₀ = T₄₀₀ - 11)
  (h3 : T₈₀₀ = T + 1) : 
  T = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_noon_temperature_l137_13732


namespace NUMINAMATH_GPT_angle_sum_420_l137_13705

theorem angle_sum_420 (A B C D E F : ℝ) (hE : E = 30) : 
  A + B + C + D + E + F = 420 :=
by
  sorry

end NUMINAMATH_GPT_angle_sum_420_l137_13705


namespace NUMINAMATH_GPT_left_handed_ratio_l137_13791

-- Given the conditions:
-- total number of players
def total_players : ℕ := 70
-- number of throwers who are all right-handed 
def throwers : ℕ := 37 
-- total number of right-handed players
def right_handed : ℕ := 59

-- Define the necessary variables based on the given conditions.
def non_throwers : ℕ := total_players - throwers
def non_throwing_right_handed : ℕ := right_handed - throwers
def left_handed_non_throwers : ℕ := non_throwers - non_throwing_right_handed

-- State the theorem to prove that the ratio of 
-- left-handed non-throwers to the rest of the team (excluding throwers) is 1:3
theorem left_handed_ratio : 
  (left_handed_non_throwers : ℚ) / (non_throwers : ℚ) = 1 / 3 := by
    sorry

end NUMINAMATH_GPT_left_handed_ratio_l137_13791


namespace NUMINAMATH_GPT_jack_total_damage_costs_l137_13754

def cost_per_tire := 250
def number_of_tires := 3
def cost_of_window := 700

def total_cost_of_tires := cost_per_tire * number_of_tires
def total_cost_of_damages := total_cost_of_tires + cost_of_window

theorem jack_total_damage_costs : total_cost_of_damages = 1450 := 
by
  -- Using the definitions provided
  -- total_cost_of_tires = 250 * 3 = 750
  -- total_cost_of_damages = 750 + 700 = 1450
  sorry

end NUMINAMATH_GPT_jack_total_damage_costs_l137_13754


namespace NUMINAMATH_GPT_max_height_of_basketball_l137_13711

def h (t : ℝ) : ℝ := -5 * t^2 + 50 * t + 2

theorem max_height_of_basketball : ∃ t : ℝ, h t = 127 :=
by
  use 5
  sorry

end NUMINAMATH_GPT_max_height_of_basketball_l137_13711


namespace NUMINAMATH_GPT_problem_solution_l137_13725

noncomputable def p (x : ℝ) : ℝ := 
  (x - (Real.sin 1)^2) * (x - (Real.sin 3)^2) * (x - (Real.sin 9)^2)

theorem problem_solution : ∃ a b n : ℕ, 
  p (1 / 4) = Real.sin (a * Real.pi / 180) / (n * Real.sin (b * Real.pi / 180)) ∧
  a > 0 ∧ b > 0 ∧ a ≤ 90 ∧ b ≤ 90 ∧ a + b + n = 216 :=
sorry

end NUMINAMATH_GPT_problem_solution_l137_13725


namespace NUMINAMATH_GPT_fraction_of_credit_extended_l137_13792

noncomputable def C_total : ℝ := 342.857
noncomputable def P_auto : ℝ := 0.35
noncomputable def C_company : ℝ := 40

theorem fraction_of_credit_extended :
  (C_company / (C_total * P_auto)) = (1 / 3) :=
  by
    sorry

end NUMINAMATH_GPT_fraction_of_credit_extended_l137_13792


namespace NUMINAMATH_GPT_gain_percentage_calculation_l137_13739

theorem gain_percentage_calculation 
  (C S : ℝ)
  (h1 : 30 * S = 40 * C) :
  (10 * S / (30 * C)) * 100 = 44.44 :=
by
  sorry

end NUMINAMATH_GPT_gain_percentage_calculation_l137_13739


namespace NUMINAMATH_GPT_remainder_of_sum_div_8_l137_13734

theorem remainder_of_sum_div_8 :
  let a := 2356789
  let b := 211
  (a + b) % 8 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_of_sum_div_8_l137_13734


namespace NUMINAMATH_GPT_range_of_a_l137_13717

theorem range_of_a (a : ℝ) : (5 - a > 1) → (a < 4) := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l137_13717


namespace NUMINAMATH_GPT_combined_average_l137_13768

-- Given Conditions
def num_results_1 : ℕ := 30
def avg_results_1 : ℝ := 20
def num_results_2 : ℕ := 20
def avg_results_2 : ℝ := 30
def num_results_3 : ℕ := 25
def avg_results_3 : ℝ := 40

-- Helper Definitions
def total_sum_1 : ℝ := num_results_1 * avg_results_1
def total_sum_2 : ℝ := num_results_2 * avg_results_2
def total_sum_3 : ℝ := num_results_3 * avg_results_3
def total_sum_all : ℝ := total_sum_1 + total_sum_2 + total_sum_3
def total_number_results : ℕ := num_results_1 + num_results_2 + num_results_3

-- Problem Statement
theorem combined_average : 
  (total_sum_all / (total_number_results:ℝ)) = 29.33 := 
by 
  sorry

end NUMINAMATH_GPT_combined_average_l137_13768


namespace NUMINAMATH_GPT_solution_set_of_inequality_l137_13756

theorem solution_set_of_inequality :
  { x : ℝ | x ^ 2 - 5 * x + 6 ≤ 0 } = { x : ℝ | 2 ≤ x ∧ x ≤ 3 } :=
by 
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l137_13756


namespace NUMINAMATH_GPT_correct_multiplication_l137_13744

variable {a : ℕ} -- Assume 'a' to be a natural number for simplicity in this example

theorem correct_multiplication : (3 * a) * (4 * a^2) = 12 * a^3 := by
  sorry

end NUMINAMATH_GPT_correct_multiplication_l137_13744


namespace NUMINAMATH_GPT_simplify_expression_l137_13730

theorem simplify_expression (θ : ℝ) : 
  ((1 + Real.sin θ ^ 2) ^ 2 - Real.cos θ ^ 4) * ((1 + Real.cos θ ^ 2) ^ 2 - Real.sin θ ^ 4) = 4 * Real.sin (2 * θ) ^ 2 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l137_13730


namespace NUMINAMATH_GPT_initial_money_correct_l137_13716

def initial_money (total: ℕ) (allowance: ℕ): ℕ :=
  total - allowance

theorem initial_money_correct: initial_money 18 8 = 10 :=
  by sorry

end NUMINAMATH_GPT_initial_money_correct_l137_13716


namespace NUMINAMATH_GPT_int_solution_exists_l137_13714

theorem int_solution_exists (x y : ℤ) (h : x + y = 5) : x = 2 ∧ y = 3 := 
by
  sorry

end NUMINAMATH_GPT_int_solution_exists_l137_13714


namespace NUMINAMATH_GPT_num_students_third_school_l137_13767

variable (x : ℕ)

def num_students_condition := (2 * (x + 40) + (x + 40) + x = 920)

theorem num_students_third_school (h : num_students_condition x) : x = 200 :=
sorry

end NUMINAMATH_GPT_num_students_third_school_l137_13767


namespace NUMINAMATH_GPT_five_wednesdays_implies_five_saturdays_in_august_l137_13700

theorem five_wednesdays_implies_five_saturdays_in_august (N : ℕ) (H1 : ∃ ws : Finset ℕ, ws.card = 5 ∧ ∀ w ∈ ws, w < 32 ∧ (w % 7 = 3)) (H2 : July_days = 31) (H3 : August_days = 31):
  ∀ w : ℕ, w < 7 → ∃ ws : Finset ℕ, ws.card = 5 ∧ ∀ sat ∈ ws, (sat % 7 = 6) :=
by
  sorry

end NUMINAMATH_GPT_five_wednesdays_implies_five_saturdays_in_august_l137_13700


namespace NUMINAMATH_GPT_max_value_of_g_l137_13760

def g (n : ℕ) : ℕ :=
  if n < 12 then n + 12 else g (n - 7)

theorem max_value_of_g : ∃ m, (∀ n, g n ≤ m) ∧ m = 23 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_g_l137_13760


namespace NUMINAMATH_GPT_petya_coloring_failure_7_petya_coloring_failure_10_l137_13743

theorem petya_coloring_failure_7 :
  ¬ ∀ (points : Fin 200 → Fin 7) (segments : ∀ (i j : Fin 200), i ≠ j → Fin 7),
  ∃ (colors : ∀ (i j : Fin 200), i ≠ j → Fin 7),
  ∀ (i j : Fin 200) (h : i ≠ j),
    (segments i j h ≠ points i) ∧ (segments i j h ≠ points j) :=
sorry

theorem petya_coloring_failure_10 :
  ¬ ∀ (points : Fin 200 → Fin 10) (segments : ∀ (i j : Fin 200), i ≠ j → Fin 10),
  ∃ (colors : ∀ (i j : Fin 200), i ≠ j → Fin 10),
  ∀ (i j : Fin 200) (h : i ≠ j),
    (segments i j h ≠ points i) ∧ (segments i j h ≠ points j) :=
sorry

end NUMINAMATH_GPT_petya_coloring_failure_7_petya_coloring_failure_10_l137_13743


namespace NUMINAMATH_GPT_problem_1_problem_2_l137_13720

def condition_p (x : ℝ) : Prop := 4 * x ^ 2 + 12 * x - 7 ≤ 0
def condition_q (a x : ℝ) : Prop := a - 3 ≤ x ∧ x ≤ a + 3

-- Problem 1: When a=0, if p is true and q is false, the range of real numbers x
theorem problem_1 (x : ℝ) :
  condition_p x ∧ ¬ condition_q 0 x ↔ -7/2 ≤ x ∧ x < -3 := sorry

-- Problem 2: If p is a sufficient condition for q, the range of real numbers a
theorem problem_2 (a : ℝ) :
  (∀ x : ℝ, condition_p x → condition_q a x) ↔ -5/2 ≤ a ∧ a ≤ -1/2 := sorry

end NUMINAMATH_GPT_problem_1_problem_2_l137_13720


namespace NUMINAMATH_GPT_min_value_one_over_a_plus_one_over_b_point_P_outside_ellipse_l137_13748

variable (a b : ℝ)
-- Conditions: a and b are positive real numbers and (a + b)x - 1 ≤ x^2 for all x > 0
variables (ha : a > 0) (hb : b > 0) (h : ∀ x : ℝ, 0 < x → (a + b) * x - 1 ≤ x^2)

-- Question 1: Prove that the minimum value of 1/a + 1/b is 2
theorem min_value_one_over_a_plus_one_over_b : (1 : ℝ) / a + (1 : ℝ) / b = 2 := 
sorry

-- Question 2: Determine point P(1, -1) relative to the ellipse x^2/a^2 + y^2/b^2 = 1
theorem point_P_outside_ellipse : (1 : ℝ)^2 / (a^2) + (-1 : ℝ)^2 / (b^2) > 1 :=
sorry

end NUMINAMATH_GPT_min_value_one_over_a_plus_one_over_b_point_P_outside_ellipse_l137_13748


namespace NUMINAMATH_GPT_total_length_of_ropes_l137_13731

theorem total_length_of_ropes 
  (L : ℕ)
  (first_used second_used : ℕ)
  (h1 : first_used = 42) 
  (h2 : second_used = 12) 
  (h3 : (L - second_used) = 4 * (L - first_used)) :
  2 * L = 104 :=
by
  -- We skip the proof for now
  sorry

end NUMINAMATH_GPT_total_length_of_ropes_l137_13731


namespace NUMINAMATH_GPT_smallest_positive_whole_number_divisible_by_first_five_primes_l137_13718

def is_prime (n : Nat) : Prop := Nat.Prime n

def first_five_primes : List Nat := [2, 3, 5, 7, 11]

def smallest_positive_divisible (lst : List Nat) : Nat :=
  List.foldl (· * ·) 1 lst

theorem smallest_positive_whole_number_divisible_by_first_five_primes :
  smallest_positive_divisible first_five_primes = 2310 := by
  sorry

end NUMINAMATH_GPT_smallest_positive_whole_number_divisible_by_first_five_primes_l137_13718


namespace NUMINAMATH_GPT_quad_function_intersects_x_axis_l137_13737

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quad_function_intersects_x_axis (m : ℝ) :
  (discriminant (2 * m) (8 * m + 1) (8 * m) ≥ 0) ↔ (m ≥ -1/16 ∧ m ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_quad_function_intersects_x_axis_l137_13737


namespace NUMINAMATH_GPT_evaluate_power_l137_13763

theorem evaluate_power :
  (64 : ℝ) = 2^6 →
  64^(3/4 : ℝ) = 16 * Real.sqrt 2 :=
by
  intro h₁
  rw [h₁]
  sorry

end NUMINAMATH_GPT_evaluate_power_l137_13763


namespace NUMINAMATH_GPT_find_denominator_l137_13728

-- Define the conditions given in the problem
variables (p q : ℚ)
variable (denominator : ℚ)

-- Assuming the conditions
variables (h1 : p / q = 4 / 5)
variables (h2 : 11 / 7 + (2 * q - p) / denominator = 2)

-- State the theorem we want to prove
theorem find_denominator : denominator = 14 :=
by
  -- The proof will be constructed later
  sorry

end NUMINAMATH_GPT_find_denominator_l137_13728


namespace NUMINAMATH_GPT_no_integer_solutions_l137_13752

theorem no_integer_solutions (x y z : ℤ) :
  x^2 - 3 * x * y + 2 * y^2 - z^2 = 31 ∧
  -x^2 + 6 * y * z + 2 * z^2 = 44 ∧
  x^2 + x * y + 8 * z^2 = 100 →
  false :=
by
  sorry

end NUMINAMATH_GPT_no_integer_solutions_l137_13752


namespace NUMINAMATH_GPT_fraction_decomposition_l137_13780

theorem fraction_decomposition (P Q : ℚ) :
  (∀ x : ℚ, 4 * x ^ 3 - 5 * x ^ 2 - 26 * x + 24 = (2 * x ^ 2 - 5 * x + 3) * (2 * x - 3))
  → P / (2 * x ^ 2 - 5 * x + 3) + Q / (2 * x - 3) = (8 * x ^ 2 - 9 * x + 20) / (4 * x ^ 3 - 5 * x ^ 2 - 26 * x + 24)
  → P = 4 / 9 ∧ Q = 68 / 9 := by 
  sorry

end NUMINAMATH_GPT_fraction_decomposition_l137_13780


namespace NUMINAMATH_GPT_clients_select_two_cars_l137_13794

theorem clients_select_two_cars (cars clients selections : ℕ) (total_selections : ℕ)
  (h1 : cars = 10) (h2 : clients = 15) (h3 : total_selections = cars * 3) (h4 : total_selections = clients * selections) :
  selections = 2 :=
by 
  sorry

end NUMINAMATH_GPT_clients_select_two_cars_l137_13794


namespace NUMINAMATH_GPT_expand_product_l137_13735

theorem expand_product (y : ℝ) : 5 * (y - 3) * (y + 10) = 5 * y^2 + 35 * y - 150 := 
  sorry

end NUMINAMATH_GPT_expand_product_l137_13735


namespace NUMINAMATH_GPT_seats_filled_percentage_l137_13762

theorem seats_filled_percentage (total_seats vacant_seats : ℕ) (h1 : total_seats = 600) (h2 : vacant_seats = 228) :
  ((total_seats - vacant_seats) / total_seats * 100 : ℝ) = 62 := by
  sorry

end NUMINAMATH_GPT_seats_filled_percentage_l137_13762


namespace NUMINAMATH_GPT_certain_number_is_50_l137_13757

theorem certain_number_is_50 (x : ℝ) (h : 4 = 0.08 * x) : x = 50 :=
by {
    sorry
}

end NUMINAMATH_GPT_certain_number_is_50_l137_13757


namespace NUMINAMATH_GPT_right_triangle_properties_l137_13793

theorem right_triangle_properties (a b c : ℕ) (h1 : c = 13) (h2 : a = 5) (h3 : a^2 + b^2 = c^2) :
  ∃ (area perimeter : ℕ), area = 30 ∧ perimeter = 30 ∧ (a < c ∧ b < c) :=
by
  let area := 1 / 2 * a * b
  let perimeter := a + b + c
  have acute_angles : a < c ∧ b < c := by sorry
  exact ⟨area, perimeter, ⟨sorry, sorry, acute_angles⟩⟩

end NUMINAMATH_GPT_right_triangle_properties_l137_13793


namespace NUMINAMATH_GPT_div_by_19_l137_13766

theorem div_by_19 (n : ℕ) (h : n > 0) : (3^(3*n+2) + 5 * 2^(3*n+1)) % 19 = 0 := by
  sorry

end NUMINAMATH_GPT_div_by_19_l137_13766


namespace NUMINAMATH_GPT_bess_milk_daily_l137_13702

-- Definitions based on conditions from step a)
variable (B : ℕ) -- B is the number of pails Bess gives every day

def BrownieMilk : ℕ := 3 * B
def DaisyMilk : ℕ := B + 1
def TotalDailyMilk : ℕ := B + BrownieMilk B + DaisyMilk B

-- Conditions definition to be used in Lean to ensure the equivalence
axiom weekly_milk_total : 7 * TotalDailyMilk B = 77
axiom daily_milk_eq : TotalDailyMilk B = 11

-- Prove that Bess gives 2 pails of milk everyday
theorem bess_milk_daily : B = 2 :=
by
  sorry

end NUMINAMATH_GPT_bess_milk_daily_l137_13702


namespace NUMINAMATH_GPT_quadratic_equation_completing_square_l137_13783

theorem quadratic_equation_completing_square :
  (∃ m n : ℝ, (∀ x : ℝ, 15 * x^2 - 30 * x - 45 = 15 * ((x + m)^2 - m^2 - 3) + 45 ∧ (m + n = 3))) :=
sorry

end NUMINAMATH_GPT_quadratic_equation_completing_square_l137_13783


namespace NUMINAMATH_GPT_find_abc_l137_13704

theorem find_abc
  (a b c : ℝ)
  (h : ∀ x y z : ℝ, |a * x + b * y + c * z| + |b * x + c * y + a * z| + |c * x + a * y + b * z| = |x| + |y| + |z|):
  (a = 1 ∧ b = 0 ∧ c = 0) ∨ (a = -1 ∧ b = 0 ∧ c = 0) ∨ 
  (a = 0 ∧ b = 1 ∧ c = 0) ∨ (a = 0 ∧ b = -1 ∧ c = 0) ∨ 
  (a = 0 ∧ b = 0 ∧ c = 1) ∨ (a = 0 ∧ b = 0 ∧ c = -1) :=
sorry

end NUMINAMATH_GPT_find_abc_l137_13704


namespace NUMINAMATH_GPT_differential_of_y_l137_13789

variable (x : ℝ) (dx : ℝ)

noncomputable def y := x * (Real.sin (Real.log x) - Real.cos (Real.log x))

theorem differential_of_y : (deriv y x * dx) = 2 * Real.sin (Real.log x) * dx := by
  sorry

end NUMINAMATH_GPT_differential_of_y_l137_13789


namespace NUMINAMATH_GPT_monotonically_decreasing_interval_l137_13746

noncomputable def f (x : ℝ) : ℝ :=
  (2 * Real.exp 2) * Real.exp (x - 2) - 2 * x + 1/2 * x^2

theorem monotonically_decreasing_interval :
  ∀ x : ℝ, x < 0 → ((2 * Real.exp x - 2 + x) < 0) :=
by
  sorry

end NUMINAMATH_GPT_monotonically_decreasing_interval_l137_13746


namespace NUMINAMATH_GPT_no_solution_for_inequalities_l137_13786

theorem no_solution_for_inequalities (m : ℝ) :
  (∀ x : ℝ, x - m ≤ 2 * m + 3 ∧ (x - 1) / 2 ≥ m → false) ↔ m < -2 :=
by
  sorry

end NUMINAMATH_GPT_no_solution_for_inequalities_l137_13786


namespace NUMINAMATH_GPT_eval_expression_l137_13729

theorem eval_expression (x : ℝ) (h : x = Real.sqrt 2 + 1) : (x + 1) / (x - 1) = 1 + Real.sqrt 2 := 
by
  sorry

end NUMINAMATH_GPT_eval_expression_l137_13729


namespace NUMINAMATH_GPT_functional_form_of_f_l137_13736

variable (f : ℝ → ℝ)

-- Define the condition as an axiom
axiom cond_f : ∀ (x y : ℝ), |f (x + y) - f (x - y) - y| ≤ y^2

-- State the theorem to be proved
theorem functional_form_of_f :
  ∃ c : ℝ, ∀ x : ℝ, f x = x / 2 + c := by
  sorry

end NUMINAMATH_GPT_functional_form_of_f_l137_13736


namespace NUMINAMATH_GPT_max_leap_years_l137_13712

theorem max_leap_years (years : ℕ) (leap_interval : ℕ) (total_years : ℕ) (leap_years : ℕ)
  (h1 : leap_interval = 5)
  (h2 : total_years = 200)
  (h3 : years = total_years / leap_interval) :
  leap_years = 40 :=
by
  sorry

end NUMINAMATH_GPT_max_leap_years_l137_13712


namespace NUMINAMATH_GPT_sum_geometric_series_l137_13770

theorem sum_geometric_series :
  ∑' n : ℕ+, (3 : ℝ)⁻¹ ^ (n : ℕ) = (1 / 2 : ℝ) := by
  sorry

end NUMINAMATH_GPT_sum_geometric_series_l137_13770

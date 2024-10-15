import Mathlib

namespace NUMINAMATH_GPT_travel_time_difference_l1197_119794

theorem travel_time_difference 
  (speed : ℝ) (d1 d2 : ℝ) (h_speed : speed = 50) (h_d1 : d1 = 475) (h_d2 : d2 = 450) : 
  (d1 - d2) / speed * 60 = 30 := 
by 
  sorry

end NUMINAMATH_GPT_travel_time_difference_l1197_119794


namespace NUMINAMATH_GPT_cube_volume_given_surface_area_l1197_119784

theorem cube_volume_given_surface_area (s : ℝ) (h₀ : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end NUMINAMATH_GPT_cube_volume_given_surface_area_l1197_119784


namespace NUMINAMATH_GPT_least_deletions_to_square_l1197_119799

theorem least_deletions_to_square (l : List ℕ) (h : l = [10, 20, 30, 40, 50, 60, 70, 80, 90]) : 
  ∃ d, d.card ≤ 2 ∧ ∀ (lp : List ℕ), lp = l.diff d → 
  ∃ k, lp.prod = k^2 :=
by
  sorry

end NUMINAMATH_GPT_least_deletions_to_square_l1197_119799


namespace NUMINAMATH_GPT_outlinedSquareDigit_l1197_119734

-- We define the conditions for three-digit powers of 2 and 3
def isThreeDigitPowerOf (base : ℕ) (n : ℕ) : Prop :=
  let power := base ^ n
  power >= 100 ∧ power < 1000

-- Define the sets of three-digit powers of 2 and 3
def threeDigitPowersOf2 : List ℕ := [128, 256, 512]
def threeDigitPowersOf3 : List ℕ := [243, 729]

-- Define the condition that the digit in the outlined square should be common as a last digit in any power of 2 and 3 that's three-digit long
def commonLastDigitOfPowers (a b : List ℕ) : Option ℕ :=
  let aLastDigits := a.map (λ x => x % 10)
  let bLastDigits := b.map (λ x => x % 10)
  (aLastDigits.inter bLastDigits).head?

theorem outlinedSquareDigit : (commonLastDigitOfPowers threeDigitPowersOf2 threeDigitPowersOf3) = some 3 :=
by
  sorry

end NUMINAMATH_GPT_outlinedSquareDigit_l1197_119734


namespace NUMINAMATH_GPT_remaining_payment_l1197_119704
noncomputable def total_cost (deposit : ℝ) (percentage : ℝ) : ℝ :=
  deposit / percentage

noncomputable def remaining_amount (deposit : ℝ) (total_cost : ℝ) : ℝ :=
  total_cost - deposit

theorem remaining_payment (deposit : ℝ) (percentage : ℝ) (total_cost : ℝ) (remaining_amount : ℝ) :
  deposit = 140 → percentage = 0.1 → total_cost = deposit / percentage → remaining_amount = total_cost - deposit → remaining_amount = 1260 :=
by
  intros
  sorry

end NUMINAMATH_GPT_remaining_payment_l1197_119704


namespace NUMINAMATH_GPT_smallest_positive_m_condition_l1197_119798

theorem smallest_positive_m_condition
  (p q : ℤ) (m : ℤ) (h_prod : p * q = 42) (h_diff : |p - q| ≤ 10) 
  (h_roots : 15 * (p + q) = m) : m = 195 :=
sorry

end NUMINAMATH_GPT_smallest_positive_m_condition_l1197_119798


namespace NUMINAMATH_GPT_least_number_to_add_l1197_119720

theorem least_number_to_add (n : ℕ) (d : ℕ) (r : ℕ) : n = 1100 → d = 23 → r = n % d → (r ≠ 0) → (d - r) = 4 :=
by
  intros h₀ h₁ h₂ h₃
  simp [h₀, h₁] at h₂
  sorry

end NUMINAMATH_GPT_least_number_to_add_l1197_119720


namespace NUMINAMATH_GPT_average_problem_l1197_119740

noncomputable def avg2 (a b : ℚ) := (a + b) / 2
noncomputable def avg3 (a b c : ℚ) := (a + b + c) / 3

theorem average_problem :
  avg3 (avg3 2 2 1) (avg2 1 2) 1 = 25 / 18 :=
by
  sorry

end NUMINAMATH_GPT_average_problem_l1197_119740


namespace NUMINAMATH_GPT_evaluate_expression_l1197_119751

theorem evaluate_expression : (10^9) / ((2 * 10^6) * 3) = 500 / 3 :=
by sorry

end NUMINAMATH_GPT_evaluate_expression_l1197_119751


namespace NUMINAMATH_GPT_intersection_in_fourth_quadrant_l1197_119745

theorem intersection_in_fourth_quadrant (k : ℝ) :
  (∃ x y : ℝ, y = -2 * x + 3 * k + 14 ∧ x - 4 * y = -3 * k - 2 ∧ x > 0 ∧ y < 0) ↔ (-6 < k) ∧ (k < -2) :=
by
  sorry

end NUMINAMATH_GPT_intersection_in_fourth_quadrant_l1197_119745


namespace NUMINAMATH_GPT_probability_x_plus_y_lt_3_in_rectangle_l1197_119701

noncomputable def probability_problem : ℚ :=
let rect_area := (4 : ℚ) * 3
let tri_area := (1 / 2 : ℚ) * 3 * 3
tri_area / rect_area

theorem probability_x_plus_y_lt_3_in_rectangle :
  probability_problem = 3 / 8 :=
sorry

end NUMINAMATH_GPT_probability_x_plus_y_lt_3_in_rectangle_l1197_119701


namespace NUMINAMATH_GPT_sequence_sum_l1197_119711

theorem sequence_sum (r x y : ℝ) (h1 : r = 1/4) 
  (h2 : x = 256 * r)
  (h3 : y = x * r) : x + y = 80 :=
by
  sorry

end NUMINAMATH_GPT_sequence_sum_l1197_119711


namespace NUMINAMATH_GPT_pizza_volume_one_piece_l1197_119789

theorem pizza_volume_one_piece :
  ∀ (h t: ℝ) (d: ℝ) (n: ℕ), d = 16 → t = 1/2 → n = 8 → h = 8 → 
  ( (π * (d / 2)^2 * t) / n = 4 * π ) :=
by 
  intros h t d n hd ht hn hh
  sorry

end NUMINAMATH_GPT_pizza_volume_one_piece_l1197_119789


namespace NUMINAMATH_GPT_Abhay_takes_1_hour_less_than_Sameer_l1197_119771

noncomputable def Sameer_speed := 42 / (6 - 2)
noncomputable def Abhay_time_doubled_speed := 42 / (2 * 7)
noncomputable def Sameer_time := 42 / Sameer_speed

theorem Abhay_takes_1_hour_less_than_Sameer
  (distance : ℝ := 42)
  (Abhay_speed : ℝ := 7)
  (Sameer_speed : ℝ := Sameer_speed)
  (time_Sameer : ℝ := distance / Sameer_speed)
  (time_Abhay_doubled_speed : ℝ := distance / (2 * Abhay_speed)) :
  time_Sameer - time_Abhay_doubled_speed = 1 :=
by
  sorry

end NUMINAMATH_GPT_Abhay_takes_1_hour_less_than_Sameer_l1197_119771


namespace NUMINAMATH_GPT_odd_expression_proof_l1197_119782

theorem odd_expression_proof (n : ℤ) : Odd (n^2 + n + 5) :=
by 
  sorry

end NUMINAMATH_GPT_odd_expression_proof_l1197_119782


namespace NUMINAMATH_GPT_problem1_problem2_l1197_119709

-- Problem 1: Prove the simplification of an expression
theorem problem1 (x : ℝ) : (2*x + 1)^2 + x*(x-4) = 5*x^2 + 1 := 
by sorry

-- Problem 2: Prove the solution set for the system of inequalities
theorem problem2 (x : ℝ) (h1 : 3*x - 6 > 0) (h2 : (5 - x) / 2 < 1) : x > 3 := 
by sorry

end NUMINAMATH_GPT_problem1_problem2_l1197_119709


namespace NUMINAMATH_GPT_M_is_listed_correctly_l1197_119736

noncomputable def M : Set ℕ := { m | ∃ n : ℕ+, 3 / (5 - m : ℝ) = n }

theorem M_is_listed_correctly : M = { 2, 4 } :=
by
  sorry

end NUMINAMATH_GPT_M_is_listed_correctly_l1197_119736


namespace NUMINAMATH_GPT_smallest_number_of_contestants_solving_all_problems_l1197_119756

theorem smallest_number_of_contestants_solving_all_problems
    (total_contestants : ℕ)
    (solve_first : ℕ)
    (solve_second : ℕ)
    (solve_third : ℕ)
    (solve_fourth : ℕ)
    (H1 : total_contestants = 100)
    (H2 : solve_first = 90)
    (H3 : solve_second = 85)
    (H4 : solve_third = 80)
    (H5 : solve_fourth = 75)
  : ∃ n, n = 30 := by
  sorry

end NUMINAMATH_GPT_smallest_number_of_contestants_solving_all_problems_l1197_119756


namespace NUMINAMATH_GPT_find_number_l1197_119795

theorem find_number (n : ℝ) : (1 / 2) * n + 6 = 11 → n = 10 := by
  sorry

end NUMINAMATH_GPT_find_number_l1197_119795


namespace NUMINAMATH_GPT_supermarket_selection_expected_value_l1197_119770

noncomputable def small_supermarkets := 72
noncomputable def medium_supermarkets := 24
noncomputable def large_supermarkets := 12
noncomputable def total_supermarkets := small_supermarkets + medium_supermarkets + large_supermarkets
noncomputable def selected_supermarkets := 9

-- Problem (I)
noncomputable def small_selected := (small_supermarkets * selected_supermarkets) / total_supermarkets
noncomputable def medium_selected := (medium_supermarkets * selected_supermarkets) / total_supermarkets
noncomputable def large_selected := (large_supermarkets * selected_supermarkets) / total_supermarkets

theorem supermarket_selection :
  small_selected = 6 ∧ medium_selected = 2 ∧ large_selected = 1 :=
sorry

-- Problem (II)
noncomputable def further_analysis := 3
noncomputable def prob_small := small_selected / selected_supermarkets
noncomputable def E_X := prob_small * further_analysis

theorem expected_value :
  E_X = 2 :=
sorry

end NUMINAMATH_GPT_supermarket_selection_expected_value_l1197_119770


namespace NUMINAMATH_GPT_sequence_bound_l1197_119763

/-- This definition states that given the initial conditions and recurrence relation
for a sequence of positive integers, the 2021st term is greater than 2^2019. -/
theorem sequence_bound (a : ℕ → ℕ) (h_initial : a 2 > a 1)
  (h_recurrence : ∀ n, a (n + 2) = 3 * a (n + 1) - 2 * a n) :
  a 2021 > 2 ^ 2019 :=
sorry

end NUMINAMATH_GPT_sequence_bound_l1197_119763


namespace NUMINAMATH_GPT_solve_inequality_1_range_of_m_l1197_119724

noncomputable def f (x : ℝ) : ℝ := abs (x - 1)
noncomputable def g (x m : ℝ) : ℝ := -abs (x + 3) + m

theorem solve_inequality_1 : {x : ℝ | f x + x^2 - 1 > 0} = {x : ℝ | x > 1 ∨ x < 0} := sorry

theorem range_of_m (m : ℝ) (h : m > 4) : ∃ x : ℝ, f x < g x m := sorry

end NUMINAMATH_GPT_solve_inequality_1_range_of_m_l1197_119724


namespace NUMINAMATH_GPT_seven_b_equals_ten_l1197_119774

theorem seven_b_equals_ten (a b : ℚ) (h1 : 5 * a + 2 * b = 0) (h2 : a = b - 2) : 7 * b = 10 := 
sorry

end NUMINAMATH_GPT_seven_b_equals_ten_l1197_119774


namespace NUMINAMATH_GPT_isosceles_triangle_and_sin_cos_range_l1197_119718

theorem isosceles_triangle_and_sin_cos_range 
  (A B C : ℝ) (a b c : ℝ) 
  (hA_pos : 0 < A) (hA_lt_pi_div_2 : A < π / 2) (h_triangle : a * Real.cos B = b * Real.cos A) :
  (A = B ∧
  ∃ x, x = Real.sin B + Real.cos (A + π / 6) ∧ (1 / 2 < x ∧ x ≤ 1)) :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_and_sin_cos_range_l1197_119718


namespace NUMINAMATH_GPT_part1_part2_l1197_119755

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem part1 (x : ℝ) : |f (-x)| + |f x| ≥ 4 * |x| := 
by
  sorry

theorem part2 (x a : ℝ) (h : |x - a| < 1 / 2) : |f x - f a| < |a| + 5 / 4 := 
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1197_119755


namespace NUMINAMATH_GPT_final_prices_l1197_119765

noncomputable def hat_initial_price : ℝ := 15
noncomputable def hat_first_discount : ℝ := 0.20
noncomputable def hat_second_discount : ℝ := 0.40

noncomputable def gloves_initial_price : ℝ := 8
noncomputable def gloves_first_discount : ℝ := 0.25
noncomputable def gloves_second_discount : ℝ := 0.30

theorem final_prices :
  let hat_price_after_first_discount := hat_initial_price * (1 - hat_first_discount)
  let hat_final_price := hat_price_after_first_discount * (1 - hat_second_discount)
  let gloves_price_after_first_discount := gloves_initial_price * (1 - gloves_first_discount)
  let gloves_final_price := gloves_price_after_first_discount * (1 - gloves_second_discount)
  hat_final_price = 7.20 ∧ gloves_final_price = 4.20 :=
by
  sorry

end NUMINAMATH_GPT_final_prices_l1197_119765


namespace NUMINAMATH_GPT_six_digit_number_contains_7_l1197_119729

theorem six_digit_number_contains_7
  (a b k : ℤ)
  (h1 : 100 ≤ 7 * a + k ∧ 7 * a + k < 1000)
  (h2 : 100 ≤ 7 * b + k ∧ 7 * b + k < 1000) :
  7 ∣ (1000 * (7 * a + k) + (7 * b + k)) :=
by
  sorry

end NUMINAMATH_GPT_six_digit_number_contains_7_l1197_119729


namespace NUMINAMATH_GPT_parabola_equation_line_tangent_to_fixed_circle_l1197_119715

open Real

def parabola_vertex_origin_directrix (p : ℝ) : Prop :=
  ∀ x y : ℝ, y^2 = 2 * p * x ↔ x = -2

def point_on_directrix (l: ℝ) (t : ℝ) : Prop :=
  t ≠ 0 ∧ l = 3 * t - 1 / t

def point_on_y_axis (q : ℝ) (t : ℝ) : Prop :=
  q = 2 * t

theorem parabola_equation (p : ℝ) : 
  parabola_vertex_origin_directrix 4 →
  y^2 = 8 * x :=
by
  sorry

theorem line_tangent_to_fixed_circle (t : ℝ) (x0 : ℝ) (r : ℝ) :
  t ≠ 0 →
  point_on_directrix (-2) t →
  point_on_y_axis (2 * t) t →
  (x0 = 2 ∧ r = 2) →
  ∀ x y : ℝ, (x - 2)^2 + y^2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_parabola_equation_line_tangent_to_fixed_circle_l1197_119715


namespace NUMINAMATH_GPT_total_chickens_on_farm_l1197_119728

noncomputable def total_chickens (H R : ℕ) : ℕ := H + R

theorem total_chickens_on_farm (H R : ℕ) (h1 : H = 9 * R - 5) (h2 : H = 67) : total_chickens H R = 75 := 
by
  sorry

end NUMINAMATH_GPT_total_chickens_on_farm_l1197_119728


namespace NUMINAMATH_GPT_average_marks_physics_chemistry_l1197_119726

theorem average_marks_physics_chemistry
  (P C M : ℕ)
  (h1 : (P + C + M) / 3 = 60)
  (h2 : (P + M) / 2 = 90)
  (h3 : P = 140) :
  (P + C) / 2 = 70 :=
by
  sorry

end NUMINAMATH_GPT_average_marks_physics_chemistry_l1197_119726


namespace NUMINAMATH_GPT_find_abs_of_y_l1197_119775

theorem find_abs_of_y (x y : ℝ) (h1 : x^2 + y^2 = 1) (h2 : 20 * x^3 - 15 * x = 3) : 
  |20 * y^3 - 15 * y| = 4 := 
sorry

end NUMINAMATH_GPT_find_abs_of_y_l1197_119775


namespace NUMINAMATH_GPT_complement_intersection_U_l1197_119741

-- Definitions of the sets based on the given conditions
def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- Definition of the complement of a set with respect to another set
def complement (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

-- Statement asserting the equivalence
theorem complement_intersection_U :
  complement U (M ∩ N) = {1, 4} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_U_l1197_119741


namespace NUMINAMATH_GPT_frequency_in_interval_l1197_119744

-- Definitions for the sample size and frequencies in given intervals
def sample_size : ℕ := 20
def freq_10_20 : ℕ := 2
def freq_20_30 : ℕ := 3
def freq_30_40 : ℕ := 4
def freq_40_50 : ℕ := 5

-- The goal: Prove that the frequency of the sample in the interval (10, 50] is 0.7
theorem frequency_in_interval (h₁ : sample_size = 20)
                              (h₂ : freq_10_20 = 2)
                              (h₃ : freq_20_30 = 3)
                              (h₄ : freq_30_40 = 4)
                              (h₅ : freq_40_50 = 5) :
  ((freq_10_20 + freq_20_30 + freq_30_40 + freq_40_50) : ℝ) / sample_size = 0.7 := 
by
  sorry

end NUMINAMATH_GPT_frequency_in_interval_l1197_119744


namespace NUMINAMATH_GPT_max_sides_of_convex_polygon_l1197_119708

theorem max_sides_of_convex_polygon (n : ℕ) 
  (h_convex : n ≥ 3) 
  (h_angles: ∀ (a : Fin 4), (100 : ℝ) ≤ a.val) 
  : n ≤ 8 :=
sorry

end NUMINAMATH_GPT_max_sides_of_convex_polygon_l1197_119708


namespace NUMINAMATH_GPT_frac_sum_is_one_l1197_119761

theorem frac_sum_is_one (a b c : ℝ) (h : a / 2 = b / 3 ∧ b / 3 = c / 5) : (a + b) / c = 1 :=
by
  sorry

end NUMINAMATH_GPT_frac_sum_is_one_l1197_119761


namespace NUMINAMATH_GPT_puzzles_and_board_games_count_l1197_119781

def num_toys : ℕ := 200
def num_action_figures : ℕ := num_toys / 4
def num_dolls : ℕ := num_toys / 3

theorem puzzles_and_board_games_count :
  num_toys - num_action_figures - num_dolls = 84 := 
  by
    -- TODO: Prove this theorem
    sorry

end NUMINAMATH_GPT_puzzles_and_board_games_count_l1197_119781


namespace NUMINAMATH_GPT_wade_final_profit_l1197_119743

theorem wade_final_profit :
  let tips_per_customer_friday := 2.00
  let customers_friday := 28
  let tips_per_customer_saturday := 2.50
  let customers_saturday := 3 * customers_friday
  let tips_per_customer_sunday := 1.50
  let customers_sunday := 36
  let cost_ingredients_per_hotdog := 1.25
  let price_per_hotdog := 4.00
  let truck_maintenance_daily_cost := 50.00
  let total_taxes := 150.00
  let revenue_tips_friday := tips_per_customer_friday * customers_friday
  let revenue_hotdogs_friday := customers_friday * price_per_hotdog
  let cost_ingredients_friday := customers_friday * cost_ingredients_per_hotdog
  let revenue_friday := revenue_tips_friday + revenue_hotdogs_friday
  let total_costs_friday := cost_ingredients_friday + truck_maintenance_daily_cost
  let profit_friday := revenue_friday - total_costs_friday
  let revenue_tips_saturday := tips_per_customer_saturday * customers_saturday
  let revenue_hotdogs_saturday := customers_saturday * price_per_hotdog
  let cost_ingredients_saturday := customers_saturday * cost_ingredients_per_hotdog
  let revenue_saturday := revenue_tips_saturday + revenue_hotdogs_saturday
  let total_costs_saturday := cost_ingredients_saturday + truck_maintenance_daily_cost
  let profit_saturday := revenue_saturday - total_costs_saturday
  let revenue_tips_sunday := tips_per_customer_sunday * customers_sunday
  let revenue_hotdogs_sunday := customers_sunday * price_per_hotdog
  let cost_ingredients_sunday := customers_sunday * cost_ingredients_per_hotdog
  let revenue_sunday := revenue_tips_sunday + revenue_hotdogs_sunday
  let total_costs_sunday := cost_ingredients_sunday + truck_maintenance_daily_cost
  let profit_sunday := revenue_sunday - total_costs_sunday
  let total_profit := profit_friday + profit_saturday + profit_sunday
  let final_profit := total_profit - total_taxes
  final_profit = 427.00 :=
by
  sorry

end NUMINAMATH_GPT_wade_final_profit_l1197_119743


namespace NUMINAMATH_GPT_exists_ab_odd_n_exists_ab_odd_n_gt3_l1197_119727

-- Define the required conditions
def gcd_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define a helper function to identify odd positive integers
def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem exists_ab_odd_n (n : ℕ) (h : is_odd n) :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ gcd_coprime (a * b * (a + b)) n :=
sorry

theorem exists_ab_odd_n_gt3 (n : ℕ) (h1 : is_odd n) (h2 : n > 3) :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ gcd_coprime (a * b * (a + b)) n ∧ n ∣ (a - b) = false :=
sorry

end NUMINAMATH_GPT_exists_ab_odd_n_exists_ab_odd_n_gt3_l1197_119727


namespace NUMINAMATH_GPT_martin_distance_l1197_119722

-- Define the given conditions
def speed : ℝ := 12.0
def time : ℝ := 6.0

-- State the theorem we want to prove
theorem martin_distance : speed * time = 72.0 := by
  sorry

end NUMINAMATH_GPT_martin_distance_l1197_119722


namespace NUMINAMATH_GPT_max_dist_AC_l1197_119705

open Real EuclideanGeometry

variables (P A B C : ℝ × ℝ)
  (hPA : dist P A = 1)
  (hPB : dist P B = 1)
  (hPA_PB : dot_product (A.1 - P.1, A.2 - P.2) (B.1 - P.1, B.2 - P.2) = - 1 / 2)
  (hBC : dist B C = 1)

theorem max_dist_AC : ∃ C : ℝ × ℝ, dist A C ≤ dist A B + dist B C ∧ dist A C = sqrt 3 + 1 :=
by
  sorry

end NUMINAMATH_GPT_max_dist_AC_l1197_119705


namespace NUMINAMATH_GPT_longer_subsegment_length_l1197_119702

-- Define the given conditions and proof goal in Lean 4
theorem longer_subsegment_length {DE EF DF DG GF : ℝ} (h1 : 3 * EF < 4 * EF) (h2 : 4 * EF < 5 * EF)
  (ratio_condition : DE / EF = 4 / 5) (DF_length : DF = 12) :
  DG + GF = DF ∧ DE / EF = DG / GF ∧ GF = (5 * 12 / 9) :=
by
  sorry

end NUMINAMATH_GPT_longer_subsegment_length_l1197_119702


namespace NUMINAMATH_GPT_snake_count_l1197_119777

def neighborhood : Type := {n : ℕ // n = 200}

def percentage (total : ℕ) (percent : ℕ) : ℕ := total * percent / 100

def owns_only_dogs (total : ℕ) : ℕ := percentage total 13
def owns_only_cats (total : ℕ) : ℕ := percentage total 10
def owns_only_snakes (total : ℕ) : ℕ := percentage total 5
def owns_only_rabbits (total : ℕ) : ℕ := percentage total 7
def owns_only_birds (total : ℕ) : ℕ := percentage total 3
def owns_only_exotic (total : ℕ) : ℕ := percentage total 6
def owns_dogs_and_cats (total : ℕ) : ℕ := percentage total 8
def owns_dogs_cats_exotic (total : ℕ) : ℕ := percentage total 9
def owns_cats_and_snakes (total : ℕ) : ℕ := percentage total 4
def owns_cats_and_birds (total : ℕ) : ℕ := percentage total 2
def owns_snakes_and_rabbits (total : ℕ) : ℕ := percentage total 5
def owns_snakes_and_birds (total : ℕ) : ℕ := percentage total 3
def owns_rabbits_and_birds (total : ℕ) : ℕ := percentage total 1
def owns_all_except_snakes (total : ℕ) : ℕ := percentage total 2
def owns_all_except_birds (total : ℕ) : ℕ := percentage total 1
def owns_three_with_exotic (total : ℕ) : ℕ := percentage total 11
def owns_only_chameleons (total : ℕ) : ℕ := percentage total 3
def owns_only_hedgehogs (total : ℕ) : ℕ := percentage total 2

def exotic_pet_owners (total : ℕ) : ℕ :=
  owns_only_exotic total + owns_dogs_cats_exotic total + owns_all_except_snakes total +
  owns_all_except_birds total + owns_three_with_exotic total + owns_only_chameleons total +
  owns_only_hedgehogs total

def exotic_pet_owners_with_snakes (total : ℕ) : ℕ :=
  percentage (exotic_pet_owners total) 25

def total_snake_owners (total : ℕ) : ℕ :=
  owns_only_snakes total + owns_cats_and_snakes total +
  owns_snakes_and_rabbits total + owns_snakes_and_birds total +
  exotic_pet_owners_with_snakes total

theorem snake_count (nh : neighborhood) : total_snake_owners (nh.val) = 51 :=
by
  sorry

end NUMINAMATH_GPT_snake_count_l1197_119777


namespace NUMINAMATH_GPT_quadratic_condition_l1197_119752

noncomputable def quadratic_sufficiency (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + x + m = 0 → m < 1/4

noncomputable def quadratic_necessity (m : ℝ) : Prop :=
  (∃ (x : ℝ), x^2 + x + m = 0) → m ≤ 1/4

theorem quadratic_condition (m : ℝ) : 
  (m < 1/4 → quadratic_sufficiency m) ∧ ¬ quadratic_necessity m := 
sorry

end NUMINAMATH_GPT_quadratic_condition_l1197_119752


namespace NUMINAMATH_GPT_new_rate_of_commission_l1197_119710

theorem new_rate_of_commission 
  (R1 : ℝ) (R1_eq : R1 = 0.04) 
  (slump_percentage : ℝ) (slump_percentage_eq : slump_percentage = 0.20000000000000007)
  (income_unchanged : ∀ (B B_new : ℝ) (R2 : ℝ),
    B_new = B * (1 - slump_percentage) →
    B * R1 = B_new * R2 → 
    R2 = 0.05) : 
  true := 
by 
  sorry

end NUMINAMATH_GPT_new_rate_of_commission_l1197_119710


namespace NUMINAMATH_GPT_division_to_fraction_fraction_to_division_mixed_to_improper_fraction_whole_to_fraction_l1197_119719

theorem division_to_fraction : (7 / 9) = 7 / 9 := by
  sorry

theorem fraction_to_division : 12 / 7 = 12 / 7 := by
  sorry

theorem mixed_to_improper_fraction : (3 + 5 / 8) = 29 / 8 := by
  sorry

theorem whole_to_fraction : 6 = 66 / 11 := by
  sorry

end NUMINAMATH_GPT_division_to_fraction_fraction_to_division_mixed_to_improper_fraction_whole_to_fraction_l1197_119719


namespace NUMINAMATH_GPT_vector_calculation_l1197_119790

def a :ℝ × ℝ := (1, 2)
def b :ℝ × ℝ := (1, -1)
def scalar_mult (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)
def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 - v2.1, v1.2 - v2.2)

theorem vector_calculation : scalar_mult (1/3) a - scalar_mult (4/3) b = (-1, 2) :=
by sorry

end NUMINAMATH_GPT_vector_calculation_l1197_119790


namespace NUMINAMATH_GPT_incorrect_reasoning_form_l1197_119773

-- Define what it means to be a rational number
def is_rational (x : ℚ) : Prop := true

-- Define what it means to be a fraction
def is_fraction (x : ℚ) : Prop := true

-- Define what it means to be an integer
def is_integer (x : ℤ) : Prop := true

-- State the premises as hypotheses
theorem incorrect_reasoning_form (h1 : ∃ x : ℚ, is_rational x ∧ is_fraction x)
                                 (h2 : ∀ z : ℤ, is_rational z) :
  ¬ (∀ z : ℤ, is_fraction z) :=
by
  -- We are stating the conclusion as a hypothesis that needs to be proven incorrect
  sorry

end NUMINAMATH_GPT_incorrect_reasoning_form_l1197_119773


namespace NUMINAMATH_GPT_trigonometric_identity_l1197_119735

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) :
  Real.cos (2 * α) - Real.sin α * Real.cos α = -1 :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l1197_119735


namespace NUMINAMATH_GPT_sum_of_possible_values_l1197_119747

variable (N K : ℝ)

theorem sum_of_possible_values (h1 : N ≠ 0) (h2 : N - (3 / N) = K) : N + (K / N) = K := 
sorry

end NUMINAMATH_GPT_sum_of_possible_values_l1197_119747


namespace NUMINAMATH_GPT_greatest_radius_of_circle_area_lt_90pi_l1197_119792

theorem greatest_radius_of_circle_area_lt_90pi : ∃ (r : ℤ), (∀ (r' : ℤ), (π * (r':ℝ)^2 < 90 * π ↔ (r' ≤ r))) ∧ (π * (r:ℝ)^2 < 90 * π) ∧ (r = 9) :=
sorry

end NUMINAMATH_GPT_greatest_radius_of_circle_area_lt_90pi_l1197_119792


namespace NUMINAMATH_GPT_tax_free_amount_correct_l1197_119772

-- Definitions based on the problem conditions
def total_value : ℝ := 1720
def tax_paid : ℝ := 78.4
def tax_rate : ℝ := 0.07

-- Definition of the tax-free amount we need to prove
def tax_free_amount : ℝ := 600

-- Main theorem to prove
theorem tax_free_amount_correct : 
  ∃ X : ℝ, 0.07 * (total_value - X) = tax_paid ∧ X = tax_free_amount :=
by 
  use 600
  simp
  sorry

end NUMINAMATH_GPT_tax_free_amount_correct_l1197_119772


namespace NUMINAMATH_GPT_F_3_f_5_eq_24_l1197_119766

def f (a : ℤ) : ℤ := a - 2
def F (a b : ℤ) : ℤ := b^3 - a

theorem F_3_f_5_eq_24 : F 3 (f 5) = 24 := by
  sorry

end NUMINAMATH_GPT_F_3_f_5_eq_24_l1197_119766


namespace NUMINAMATH_GPT_time_to_cross_pole_is_2_5_l1197_119716

noncomputable def time_to_cross_pole : ℝ :=
  let length_of_train := 100 -- meters
  let speed_km_per_hr := 144 -- km/hr
  let speed_m_per_s := speed_km_per_hr * 1000 / 3600 -- converting speed to m/s
  length_of_train / speed_m_per_s

theorem time_to_cross_pole_is_2_5 :
  time_to_cross_pole = 2.5 :=
by
  -- The Lean proof will be written here.
  -- Placeholder for the formal proof.
  sorry

end NUMINAMATH_GPT_time_to_cross_pole_is_2_5_l1197_119716


namespace NUMINAMATH_GPT_total_pizzas_bought_l1197_119738

theorem total_pizzas_bought (slices_small : ℕ) (slices_medium : ℕ) (slices_large : ℕ) 
                            (num_small : ℕ) (num_medium : ℕ) (total_slices : ℕ) :
  slices_small = 6 → 
  slices_medium = 8 → 
  slices_large = 12 → 
  num_small = 4 → 
  num_medium = 5 → 
  total_slices = 136 → 
  (total_slices = num_small * slices_small + num_medium * slices_medium + 72) →
  15 = num_small + num_medium + 6 :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_pizzas_bought_l1197_119738


namespace NUMINAMATH_GPT_water_tank_capacity_l1197_119703

theorem water_tank_capacity (x : ℝ)
  (h1 : (2 / 3) * x - (1 / 3) * x = 20) : x = 60 := 
  sorry

end NUMINAMATH_GPT_water_tank_capacity_l1197_119703


namespace NUMINAMATH_GPT_rate_per_sq_meter_is_900_l1197_119733

/-- The length of the room L is 7 (meters). -/
def L : ℝ := 7

/-- The width of the room W is 4.75 (meters). -/
def W : ℝ := 4.75

/-- The total cost of paving the floor is Rs. 29,925. -/
def total_cost : ℝ := 29925

/-- The rate per square meter for the slabs is Rs. 900. -/
theorem rate_per_sq_meter_is_900 :
  total_cost / (L * W) = 900 :=
by
  sorry

end NUMINAMATH_GPT_rate_per_sq_meter_is_900_l1197_119733


namespace NUMINAMATH_GPT_roots_equation_sum_and_product_l1197_119791

theorem roots_equation_sum_and_product (x1 x2 : ℝ) (h1 : x1 ^ 2 - 3 * x1 - 5 = 0) (h2 : x2 ^ 2 - 3 * x2 - 5 = 0) :
  x1 + x2 - x1 * x2 = 8 :=
sorry

end NUMINAMATH_GPT_roots_equation_sum_and_product_l1197_119791


namespace NUMINAMATH_GPT_reduced_rates_apply_two_days_l1197_119739

-- Definition of total hours in a week
def total_hours_in_week : ℕ := 7 * 24

-- Given fraction of the week with reduced rates
def reduced_rate_fraction : ℝ := 0.6428571428571429

-- Total hours covered by reduced rates
def reduced_rate_hours : ℝ := reduced_rate_fraction * total_hours_in_week

-- Hours per day with reduced rates on weekdays (8 p.m. to 8 a.m.)
def hours_weekday_night : ℕ := 12

-- Total weekdays with reduced rates
def total_weekdays : ℕ := 5

-- Total reduced rate hours on weekdays
def reduced_rate_hours_weekdays : ℕ := total_weekdays * hours_weekday_night

-- Remaining hours for 24 hour reduced rates
def remaining_reduced_rate_hours : ℝ := reduced_rate_hours - reduced_rate_hours_weekdays

-- Prove that the remaining reduced rate hours correspond to exactly 2 full days
theorem reduced_rates_apply_two_days : remaining_reduced_rate_hours = 2 * 24 := 
by
  sorry

end NUMINAMATH_GPT_reduced_rates_apply_two_days_l1197_119739


namespace NUMINAMATH_GPT_add_fractions_l1197_119758

theorem add_fractions: (2 / 5) + (3 / 8) = 31 / 40 := 
by 
  sorry

end NUMINAMATH_GPT_add_fractions_l1197_119758


namespace NUMINAMATH_GPT_winning_votes_cast_l1197_119717

variable (V : ℝ) -- Total number of votes (real number)
variable (winner_votes_ratio : ℝ) -- Ratio for winner's votes
variable (votes_difference : ℝ) -- Vote difference due to winning

-- Conditions given
def election_conditions (V : ℝ) (winner_votes_ratio : ℝ) (votes_difference : ℝ) : Prop :=
  winner_votes_ratio = 0.54 ∧
  votes_difference = 288

-- Proof problem: Proving the number of votes cast to the winning candidate is 1944
theorem winning_votes_cast (V : ℝ) (winner_votes_ratio : ℝ) (votes_difference : ℝ) 
  (h : election_conditions V winner_votes_ratio votes_difference) :
  winner_votes_ratio * V = 1944 :=
by
  sorry

end NUMINAMATH_GPT_winning_votes_cast_l1197_119717


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1197_119725

theorem sufficient_but_not_necessary (x : ℝ) (h1 : x > 1 → x > 0) (h2 : ¬ (x > 0 → x > 1)) : 
  (x > 1 → x > 0) ∧ ¬ (x > 0 → x > 1) := 
by 
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1197_119725


namespace NUMINAMATH_GPT_remainder_base12_div_9_l1197_119742

def base12_to_decimal (n : ℕ) : ℕ := 2 * 12^3 + 5 * 12^2 + 4 * 12 + 3

theorem remainder_base12_div_9 : (base12_to_decimal 2543) % 9 = 8 := by
  unfold base12_to_decimal
  -- base12_to_decimal 2543 is 4227
  show 4227 % 9 = 8
  sorry

end NUMINAMATH_GPT_remainder_base12_div_9_l1197_119742


namespace NUMINAMATH_GPT_avg_visitors_is_correct_l1197_119764

-- Define the number of days in the month
def days_in_month : ℕ := 30

-- Define the average number of visitors on Sundays
def avg_visitors_sunday : ℕ := 510

-- Define the average number of visitors on other days
def avg_visitors_other_days : ℕ := 240

-- Define the number of Sundays in the month
def sundays_in_month : ℕ := 4

-- Define the number of other days in the month
def other_days_in_month : ℕ := days_in_month - sundays_in_month

-- Define the total visitors on Sundays
def total_visitors_sundays : ℕ := sundays_in_month * avg_visitors_sunday

-- Define the total visitors on other days
def total_visitors_other_days : ℕ := other_days_in_month * avg_visitors_other_days

-- Define the total number of visitors in the month
def total_visitors : ℕ := total_visitors_sundays + total_visitors_other_days

-- Define the average number of visitors per day
def avg_visitors_per_day : ℕ := total_visitors / days_in_month

-- The theorem to prove
theorem avg_visitors_is_correct : avg_visitors_per_day = 276 := by
  sorry

end NUMINAMATH_GPT_avg_visitors_is_correct_l1197_119764


namespace NUMINAMATH_GPT_hash_op_example_l1197_119780

def hash_op (a b c : ℤ) : ℤ := (b + 1)^2 - 4 * a * (c - 1)

theorem hash_op_example : hash_op 2 3 4 = -8 := by
  -- The proof can be added here, but for now, we use sorry to skip it
  sorry

end NUMINAMATH_GPT_hash_op_example_l1197_119780


namespace NUMINAMATH_GPT_part1_part2_l1197_119731

def f (x a : ℝ) : ℝ := abs (x - a)

theorem part1 (a : ℝ) (h : a = 2) : 
  {x : ℝ | f x a ≥ 4 - abs (x - 4)} = {x : ℝ | x ≤ 1} ∪ {x : ℝ | x ≥ 5} :=
by
  sorry

theorem part2 (set_is : {x : ℝ | 1 ≤ x ∧ x ≤ 2}) : 
  ∃ a : ℝ, 
    (∀ x : ℝ, abs (f (2*x + a) a - 2*f x a) ≤ 2 → (1 ≤ x ∧ x ≤ 2)) ∧ 
    a = 3 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1197_119731


namespace NUMINAMATH_GPT_transistors_in_2010_l1197_119788

theorem transistors_in_2010 (initial_transistors: ℕ) 
    (doubling_period_years: ℕ) (start_year: ℕ) (end_year: ℕ) 
    (h_initial: initial_transistors = 500000)
    (h_period: doubling_period_years = 2) 
    (h_start: start_year = 1992) 
    (h_end: end_year = 2010) :
  let years_passed := end_year - start_year
  let number_of_doublings := years_passed / doubling_period_years
  let transistors_in_end_year := initial_transistors * 2^number_of_doublings
  transistors_in_end_year = 256000000 := by
    sorry

end NUMINAMATH_GPT_transistors_in_2010_l1197_119788


namespace NUMINAMATH_GPT_find_constant_k_l1197_119748

theorem find_constant_k (k : ℤ) :
    (∀ x : ℝ, -x^2 - (k + 7) * x - 8 = - (x - 2) * (x - 4)) → k = -13 :=
by 
    intros h
    sorry

end NUMINAMATH_GPT_find_constant_k_l1197_119748


namespace NUMINAMATH_GPT_total_wheels_at_park_l1197_119713

-- Conditions as definitions
def number_of_adults := 6
def number_of_children := 15
def wheels_per_bicycle := 2
def wheels_per_tricycle := 3

-- To prove: total number of wheels = 57
theorem total_wheels_at_park : 
  (number_of_adults * wheels_per_bicycle) + (number_of_children * wheels_per_tricycle) = 57 :=
by
  sorry

end NUMINAMATH_GPT_total_wheels_at_park_l1197_119713


namespace NUMINAMATH_GPT_find_reciprocal_square_sum_of_roots_l1197_119749

theorem find_reciprocal_square_sum_of_roots :
  ∃ (a b c : ℝ), 
    (a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
    (a^3 - 6 * a^2 - a + 3 = 0) ∧ 
    (b^3 - 6 * b^2 - b + 3 = 0) ∧ 
    (c^3 - 6 * c^2 - c + 3 = 0) ∧ 
    (a + b + c = 6) ∧
    (a * b + b * c + c * a = -1) ∧
    (a * b * c = -3)) 
    → (1 / a^2 + 1 / b^2 + 1 / c^2 = 37 / 9) :=
sorry

end NUMINAMATH_GPT_find_reciprocal_square_sum_of_roots_l1197_119749


namespace NUMINAMATH_GPT_roots_of_quadratic_l1197_119797

theorem roots_of_quadratic (a b c : ℝ) (h1 : a ≠ 0) (h2 : a + b + c = 0) (h3 : a - b + c = 0) : 
  (a * 1 ^2 + b * 1 + c = 0) ∧ (a * (-1) ^2 + b * (-1) + c = 0) :=
sorry

end NUMINAMATH_GPT_roots_of_quadratic_l1197_119797


namespace NUMINAMATH_GPT_intersection_point_on_circle_l1197_119786

theorem intersection_point_on_circle :
  ∀ (m : ℝ) (x y : ℝ),
  (m * x - y = 0) → 
  (x + m * y - m - 2 = 0) → 
  (x - 1)^2 + (y - 1 / 2)^2 = 5 / 4 :=
by
  intros m x y h1 h2
  sorry

end NUMINAMATH_GPT_intersection_point_on_circle_l1197_119786


namespace NUMINAMATH_GPT_age_sum_squares_l1197_119746

theorem age_sum_squares (a b c : ℕ) (h1 : 5 * a + 2 * b = 3 * c) (h2 : 3 * c^2 = 4 * a^2 + b^2) (h3 : Nat.gcd (Nat.gcd a b) c = 1) : a^2 + b^2 + c^2 = 18 :=
sorry

end NUMINAMATH_GPT_age_sum_squares_l1197_119746


namespace NUMINAMATH_GPT_probability_of_target_hit_l1197_119732

theorem probability_of_target_hit  :
  let A_hits := 0.9
  let B_hits := 0.8
  ∃ (P_A P_B : ℝ), 
  P_A = A_hits ∧ P_B = B_hits ∧ 
  (∀ events_independent : Prop, 
   events_independent → P_A * P_B = (0.1) * (0.2)) →
  1 - (0.1 * 0.2) = 0.98
:= 
  sorry

end NUMINAMATH_GPT_probability_of_target_hit_l1197_119732


namespace NUMINAMATH_GPT_max_lateral_surface_area_l1197_119753

theorem max_lateral_surface_area (x y : ℝ) (h₁ : x + y = 10) : 
  2 * π * x * y ≤ 50 * π :=
by
  sorry

end NUMINAMATH_GPT_max_lateral_surface_area_l1197_119753


namespace NUMINAMATH_GPT_problem1_problem2_l1197_119737

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - x - 2 ≤ 0
def q (x : ℝ) (m : ℝ) : Prop := x^2 - x - m^2 - m ≤ 0

-- Problem 1: If ¬p is true, find the range of values for x
theorem problem1 {x : ℝ} (h : ¬ p x) : x > 2 ∨ x < -1 :=
by
  -- Proof omitted
  sorry

-- Problem 2: If ¬q is a sufficient but not necessary condition for ¬p, find the range of values for m
theorem problem2 {m : ℝ} (h : ∀ x : ℝ, ¬ q x m → ¬ p x) : m > 1 ∨ m < -2 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1197_119737


namespace NUMINAMATH_GPT_largest_possible_last_digit_l1197_119785

theorem largest_possible_last_digit (D : Fin 3003 → Nat) :
  D 0 = 2 →
  (∀ i : Fin 3002, (10 * D i + D (i + 1)) % 17 = 0 ∨ (10 * D i + D (i + 1)) % 23 = 0) →
  D 3002 = 9 :=
sorry

end NUMINAMATH_GPT_largest_possible_last_digit_l1197_119785


namespace NUMINAMATH_GPT_conveyor_belt_sampling_l1197_119778

noncomputable def sampling_method (interval : ℕ) (total_items : ℕ) : String :=
  if interval = 5 ∧ total_items > 0 then "systematic sampling" else "unknown"

theorem conveyor_belt_sampling :
  ∀ (interval : ℕ) (total_items : ℕ),
  interval = 5 ∧ total_items > 0 →
  sampling_method interval total_items = "systematic sampling" :=
sorry

end NUMINAMATH_GPT_conveyor_belt_sampling_l1197_119778


namespace NUMINAMATH_GPT_beef_weight_after_processing_l1197_119707

theorem beef_weight_after_processing
  (initial_weight : ℝ)
  (weight_loss_percentage : ℝ)
  (processed_weight : ℝ)
  (h1 : initial_weight = 892.31)
  (h2 : weight_loss_percentage = 0.35)
  (h3 : processed_weight = initial_weight * (1 - weight_loss_percentage)) :
  processed_weight = 579.5015 :=
by
  sorry

end NUMINAMATH_GPT_beef_weight_after_processing_l1197_119707


namespace NUMINAMATH_GPT_line_intersects_parabola_once_l1197_119760

theorem line_intersects_parabola_once (k : ℝ) : 
  (∃! y : ℝ, -3 * y^2 + 2 * y + 7 = k) ↔ k = 22 / 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_line_intersects_parabola_once_l1197_119760


namespace NUMINAMATH_GPT_crocodile_can_move_anywhere_iff_even_l1197_119783

def is_even (n : ℕ) : Prop := n % 2 = 0

def can_move_to_any_square (N : ℕ) : Prop :=
∀ (x1 y1 x2 y2 : ℤ), ∃ (k : ℕ), 
(x1 + k * (N + 1) = x2 ∨ y1 + k * (N + 1) = y2)

theorem crocodile_can_move_anywhere_iff_even (N : ℕ) : can_move_to_any_square N ↔ is_even N :=
sorry

end NUMINAMATH_GPT_crocodile_can_move_anywhere_iff_even_l1197_119783


namespace NUMINAMATH_GPT_land_division_possible_l1197_119787

-- Define the basic properties and conditions of the plot
structure Plot :=
  (is_square : Prop)
  (has_center_well : Prop)
  (has_four_trees : Prop)
  (has_four_gates : Prop)

-- Define a section of the plot
structure Section :=
  (contains_tree : Prop)
  (contains_gate : Prop)
  (equal_fence_length : Prop)
  (unrestricted_access_to_well : Prop)

-- Define the property that indicates a valid division of the plot
def valid_division (p : Plot) (sections : List Section) : Prop :=
  sections.length = 4 ∧
  (∀ s ∈ sections, s.contains_tree) ∧
  (∀ s ∈ sections, s.contains_gate) ∧
  (∀ s ∈ sections, s.equal_fence_length) ∧
  (∀ s ∈ sections, s.unrestricted_access_to_well)

-- Define the main theorem to prove
theorem land_division_possible (p : Plot) : 
  p.is_square ∧ p.has_center_well ∧ p.has_four_trees ∧ p.has_four_gates → 
  ∃ sections : List Section, valid_division p sections :=
by
  sorry

end NUMINAMATH_GPT_land_division_possible_l1197_119787


namespace NUMINAMATH_GPT_k_value_for_polynomial_l1197_119779

theorem k_value_for_polynomial (k : ℤ) :
  (3 : ℤ)^3 + k * (3 : ℤ) - 18 = 0 → k = -3 :=
by
  sorry

end NUMINAMATH_GPT_k_value_for_polynomial_l1197_119779


namespace NUMINAMATH_GPT_total_spent_by_pete_and_raymond_l1197_119768

def pete_initial_amount := 250
def pete_spending_on_stickers := 4 * 5
def pete_spending_on_candy := 3 * 10
def pete_spending_on_toy_car := 2 * 25
def pete_spending_on_keychain := 5
def pete_total_spent := pete_spending_on_stickers + pete_spending_on_candy + pete_spending_on_toy_car + pete_spending_on_keychain
def raymond_initial_amount := 250
def raymond_left_dimes := 7 * 10
def raymond_left_quarters := 4 * 25
def raymond_left_nickels := 5 * 5
def raymond_left_pennies := 3 * 1
def raymond_total_left := raymond_left_dimes + raymond_left_quarters + raymond_left_nickels + raymond_left_pennies
def raymond_total_spent := raymond_initial_amount - raymond_total_left
def total_spent := pete_total_spent + raymond_total_spent

theorem total_spent_by_pete_and_raymond : total_spent = 157 := by
  have h1 : pete_total_spent = 105 := sorry
  have h2 : raymond_total_spent = 52 := sorry
  exact sorry

end NUMINAMATH_GPT_total_spent_by_pete_and_raymond_l1197_119768


namespace NUMINAMATH_GPT_even_numbers_average_l1197_119757

theorem even_numbers_average (n : ℕ) (h1 : 2 * (n * (n + 1)) = 22 * n) : n = 10 :=
by
  sorry

end NUMINAMATH_GPT_even_numbers_average_l1197_119757


namespace NUMINAMATH_GPT_friends_in_group_l1197_119723

theorem friends_in_group : 
  ∀ (total_chicken_wings cooked_wings additional_wings chicken_wings_per_person : ℕ), 
    cooked_wings = 8 →
    additional_wings = 10 →
    chicken_wings_per_person = 6 →
    total_chicken_wings = cooked_wings + additional_wings →
    total_chicken_wings / chicken_wings_per_person = 3 :=
by
  intros total_chicken_wings cooked_wings additional_wings chicken_wings_per_person hcooked hadditional hperson htotal
  sorry

end NUMINAMATH_GPT_friends_in_group_l1197_119723


namespace NUMINAMATH_GPT_ammonium_chloride_reacts_with_potassium_hydroxide_l1197_119759

/-- Prove that 1 mole of ammonium chloride is required to react with 
    1 mole of potassium hydroxide to form 1 mole of ammonia, 
    1 mole of water, and 1 mole of potassium chloride, 
    given the balanced chemical equation:
    NH₄Cl + KOH → NH₃ + H₂O + KCl
-/
theorem ammonium_chloride_reacts_with_potassium_hydroxide :
    ∀ (NH₄Cl KOH NH₃ H₂O KCl : ℕ), 
    (NH₄Cl + KOH = NH₃ + H₂O + KCl) → 
    (NH₄Cl = 1) → 
    (KOH = 1) → 
    (NH₃ = 1) → 
    (H₂O = 1) → 
    (KCl = 1) → 
    NH₄Cl = 1 :=
by
  intros
  sorry

end NUMINAMATH_GPT_ammonium_chloride_reacts_with_potassium_hydroxide_l1197_119759


namespace NUMINAMATH_GPT_positive_number_is_25_l1197_119712

theorem positive_number_is_25 {a x : ℝ}
(h1 : x = (3 * a + 1)^2)
(h2 : x = (-a - 3)^2)
(h_sum : 3 * a + 1 + (-a - 3) = 0) :
x = 25 :=
sorry

end NUMINAMATH_GPT_positive_number_is_25_l1197_119712


namespace NUMINAMATH_GPT_find_m_for_positive_integer_x_l1197_119706

theorem find_m_for_positive_integer_x :
  ∃ (m : ℤ), (2 * m * x - 8 = (m + 2) * x) → ∀ (x : ℤ), x > 0 → m = 3 ∨ m = 4 ∨ m = 6 ∨ m = 10 :=
sorry

end NUMINAMATH_GPT_find_m_for_positive_integer_x_l1197_119706


namespace NUMINAMATH_GPT_equation_of_plane_l1197_119767

noncomputable def parametric_form (s t : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 2 * s - 3 * t, 4 - s + 2 * t, 1 - 3 * s - t)

theorem equation_of_plane (x y z : ℝ) : 
  (∃ s t : ℝ, parametric_form s t = (x, y, z)) → 5 * x + 11 * y + 7 * z - 61 = 0 :=
by
  sorry

end NUMINAMATH_GPT_equation_of_plane_l1197_119767


namespace NUMINAMATH_GPT_average_of_side_lengths_of_squares_l1197_119730

theorem average_of_side_lengths_of_squares :
  let side_length1 := Real.sqrt 25
  let side_length2 := Real.sqrt 64
  let side_length3 := Real.sqrt 144
  (side_length1 + side_length2 + side_length3) / 3 = 25 / 3 :=
by
  sorry

end NUMINAMATH_GPT_average_of_side_lengths_of_squares_l1197_119730


namespace NUMINAMATH_GPT_third_box_nuts_l1197_119721

theorem third_box_nuts
  (A B C : ℕ)
  (h1 : A = B + C - 6)
  (h2 : B = A + C - 10) :
  C = 8 :=
by
  sorry

end NUMINAMATH_GPT_third_box_nuts_l1197_119721


namespace NUMINAMATH_GPT_exactly_one_root_in_interval_l1197_119762

theorem exactly_one_root_in_interval (p q : ℝ) (h : q * (q + p + 1) < 0) :
    ∃ x : ℝ, 0 < x ∧ x < 1 ∧ (x^2 + p * x + q = 0) := sorry

end NUMINAMATH_GPT_exactly_one_root_in_interval_l1197_119762


namespace NUMINAMATH_GPT_fixed_point_when_a_2_b_neg2_range_of_a_for_two_fixed_points_l1197_119769

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 2

theorem fixed_point_when_a_2_b_neg2 :
  (∃ x : ℝ, f 2 (-2) x = x) → (x = -1 ∨ x = 2) :=
sorry

theorem range_of_a_for_two_fixed_points (a : ℝ) :
  (∀ b : ℝ, a ≠ 0 → (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a b x1 = x1 ∧ f a b x2 = x2)) → (0 < a ∧ a < 2) :=
sorry

end NUMINAMATH_GPT_fixed_point_when_a_2_b_neg2_range_of_a_for_two_fixed_points_l1197_119769


namespace NUMINAMATH_GPT_roots_negative_reciprocals_l1197_119793

theorem roots_negative_reciprocals (a b c r s : ℝ) (h1 : a * r^2 + b * r + c = 0)
    (h2 : a * s^2 + b * s + c = 0) (h3 : r = -1 / s) (h4 : s = -1 / r) :
    a = -c :=
by
  -- Insert clever tricks to auto-solve or reuse axioms here
  sorry

end NUMINAMATH_GPT_roots_negative_reciprocals_l1197_119793


namespace NUMINAMATH_GPT_olivia_earnings_l1197_119776

-- Define Olivia's hourly wage
def wage : ℕ := 9

-- Define the hours worked on each day
def hours_monday : ℕ := 4
def hours_wednesday : ℕ := 3
def hours_friday : ℕ := 6

-- Define the total hours worked
def total_hours : ℕ := hours_monday + hours_wednesday + hours_friday

-- Define the total earnings
def total_earnings : ℕ := total_hours * wage

-- State the theorem
theorem olivia_earnings : total_earnings = 117 :=
by
  sorry

end NUMINAMATH_GPT_olivia_earnings_l1197_119776


namespace NUMINAMATH_GPT_negation_of_exists_proposition_l1197_119754

theorem negation_of_exists_proposition :
  (¬ ∃ x : ℝ, x^2 - 2 * x > 2) ↔ (∀ x : ℝ, x^2 - 2 * x ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_exists_proposition_l1197_119754


namespace NUMINAMATH_GPT_simplify_fraction_l1197_119750

theorem simplify_fraction : 
  1 + (1 / (1 + (1 / (2 + 1)))) = 7 / 4 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1197_119750


namespace NUMINAMATH_GPT_p_necessary_for_q_l1197_119714

def p (x : ℝ) := x ≠ 1
def q (x : ℝ) := x ≥ 2

theorem p_necessary_for_q : ∀ x, q x → p x :=
by
  intro x
  intro hqx
  rw [q] at hqx
  rw [p]
  sorry

end NUMINAMATH_GPT_p_necessary_for_q_l1197_119714


namespace NUMINAMATH_GPT_find_y_l1197_119796

def star (a b : ℝ) : ℝ := 4 * a + 2 * b

theorem find_y (y : ℝ) : star 3 (star 4 y) = -2 → y = -11.5 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1197_119796


namespace NUMINAMATH_GPT_sin_300_eq_neg_sqrt3_div_2_l1197_119700

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_sin_300_eq_neg_sqrt3_div_2_l1197_119700

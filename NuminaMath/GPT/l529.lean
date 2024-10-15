import Mathlib

namespace NUMINAMATH_GPT_james_total_payment_l529_52960

noncomputable def first_pair_cost : ℝ := 40
noncomputable def second_pair_cost : ℝ := 60
noncomputable def discount_applied_to : ℝ := min first_pair_cost second_pair_cost
noncomputable def discount_amount := discount_applied_to / 2
noncomputable def total_before_extra_discount := first_pair_cost + (second_pair_cost - discount_amount)
noncomputable def extra_discount := total_before_extra_discount / 4
noncomputable def final_amount := total_before_extra_discount - extra_discount

theorem james_total_payment : final_amount = 60 := by
  sorry

end NUMINAMATH_GPT_james_total_payment_l529_52960


namespace NUMINAMATH_GPT_probability_both_selected_l529_52963

theorem probability_both_selected (P_R : ℚ) (P_V : ℚ) (h1 : P_R = 3 / 7) (h2 : P_V = 1 / 5) :
  P_R * P_V = 3 / 35 :=
by {
  sorry
}

end NUMINAMATH_GPT_probability_both_selected_l529_52963


namespace NUMINAMATH_GPT_difference_between_advertised_and_actual_mileage_l529_52914

def advertised_mileage : ℕ := 35

def city_mileage_regular : ℕ := 30
def highway_mileage_premium : ℕ := 40
def traffic_mileage_diesel : ℕ := 32

def gallons_regular : ℕ := 4
def gallons_premium : ℕ := 4
def gallons_diesel : ℕ := 4

def total_miles_driven : ℕ :=
  (gallons_regular * city_mileage_regular) + 
  (gallons_premium * highway_mileage_premium) + 
  (gallons_diesel * traffic_mileage_diesel)

def total_gallons_used : ℕ :=
  gallons_regular + gallons_premium + gallons_diesel

def weighted_average_mpg : ℤ :=
  total_miles_driven / total_gallons_used

theorem difference_between_advertised_and_actual_mileage :
  advertised_mileage - weighted_average_mpg = 1 :=
by
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_difference_between_advertised_and_actual_mileage_l529_52914


namespace NUMINAMATH_GPT_parallel_lines_not_coincident_l529_52902

theorem parallel_lines_not_coincident (x y : ℝ) (m : ℝ) :
  (∀ y, x + (1 + m) * y = 2 - m ∧ ∀ y, m * x + 2 * y + 8 = 0) → (m =1) := 
sorry

end NUMINAMATH_GPT_parallel_lines_not_coincident_l529_52902


namespace NUMINAMATH_GPT_water_volume_correct_l529_52957

-- Define the conditions
def ratio_water_juice : ℕ := 5
def ratio_juice_water : ℕ := 3
def total_punch_volume : ℚ := 3  -- in liters

-- Define the question and the correct answer
def volume_of_water (ratio_water_juice ratio_juice_water : ℕ) (total_punch_volume : ℚ) : ℚ :=
  (ratio_water_juice * total_punch_volume) / (ratio_water_juice + ratio_juice_water)

-- The proof problem
theorem water_volume_correct : volume_of_water ratio_water_juice ratio_juice_water total_punch_volume = 15 / 8 :=
by
  sorry

end NUMINAMATH_GPT_water_volume_correct_l529_52957


namespace NUMINAMATH_GPT_positive_integer_k_l529_52942

theorem positive_integer_k (k x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x^2 + y^2 + z^2 = k * x * y * z) :
  k = 1 ∨ k = 3 :=
sorry

end NUMINAMATH_GPT_positive_integer_k_l529_52942


namespace NUMINAMATH_GPT_Connie_correct_result_l529_52937

theorem Connie_correct_result :
  ∀ x: ℝ, (200 - x = 100) → (200 + x = 300) :=
by
  intros x h
  have h1 : x = 100 := by linarith [h]
  rw [h1]
  linarith

end NUMINAMATH_GPT_Connie_correct_result_l529_52937


namespace NUMINAMATH_GPT_simplify_expression_l529_52907

theorem simplify_expression (w : ℝ) : (5 - 2 * w) - (4 + 5 * w) = 1 - 7 * w := by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l529_52907


namespace NUMINAMATH_GPT_no_solutions_988_1991_l529_52922

theorem no_solutions_988_1991 :
    ¬ ∃ (m n : ℤ),
      (988 ≤ m ∧ m ≤ 1991) ∧
      (988 ≤ n ∧ n ≤ 1991) ∧
      m ≠ n ∧
      ∃ (a b : ℤ), (mn + n = a^2 ∧ mn + m = b^2) := sorry

end NUMINAMATH_GPT_no_solutions_988_1991_l529_52922


namespace NUMINAMATH_GPT_roots_of_equation_l529_52965

theorem roots_of_equation :
  (∃ x, (18 / (x^2 - 9) - 3 / (x - 3) = 2) ↔ (x = 3 ∨ x = -4.5)) :=
by
  sorry

end NUMINAMATH_GPT_roots_of_equation_l529_52965


namespace NUMINAMATH_GPT_interest_difference_l529_52946

theorem interest_difference :
  let P := 10000
  let R := 6
  let T := 2
  let SI := P * R * T / 100
  let CI := P * (1 + R / 100)^T - P
  CI - SI = 36 :=
by
  let P := 10000
  let R := 6
  let T := 2
  let SI := P * R * T / 100
  let CI := P * (1 + R / 100)^T - P
  show CI - SI = 36
  sorry

end NUMINAMATH_GPT_interest_difference_l529_52946


namespace NUMINAMATH_GPT_degrees_to_radians_l529_52990

theorem degrees_to_radians (π_radians : ℝ) : 150 * π_radians / 180 = 5 * π_radians / 6 :=
by sorry

end NUMINAMATH_GPT_degrees_to_radians_l529_52990


namespace NUMINAMATH_GPT_number_of_bugs_l529_52932

def flowers_per_bug := 2
def total_flowers_eaten := 6

theorem number_of_bugs : total_flowers_eaten / flowers_per_bug = 3 := 
by sorry

end NUMINAMATH_GPT_number_of_bugs_l529_52932


namespace NUMINAMATH_GPT_problem_tiles_count_l529_52913

theorem problem_tiles_count (T B : ℕ) (h: 2 * T + 3 * B = 301) (hB: B = 3) : T = 146 := 
by
  sorry

end NUMINAMATH_GPT_problem_tiles_count_l529_52913


namespace NUMINAMATH_GPT_triangular_number_19_l529_52958

def triangular_number (n : Nat) : Nat :=
  (n + 1) * (n + 2) / 2

theorem triangular_number_19 : triangular_number 19 = 210 := by
  sorry

end NUMINAMATH_GPT_triangular_number_19_l529_52958


namespace NUMINAMATH_GPT_infinite_colored_points_l529_52961

theorem infinite_colored_points
(P : ℤ → Prop) (red blue : ℤ → Prop)
(h_color : ∀ n : ℤ, (red n ∨ blue n))
(h_red_blue_partition : ∀ n : ℤ, ¬(red n ∧ blue n)) :
  ∃ (C : ℤ → Prop) (k : ℕ), (C = red ∨ C = blue) ∧ ∀ n : ℕ, ∃ m : ℤ, C m ∧ (m % n) = 0 :=
by
  sorry

end NUMINAMATH_GPT_infinite_colored_points_l529_52961


namespace NUMINAMATH_GPT_map_distance_to_actual_distance_l529_52975

theorem map_distance_to_actual_distance :
  ∀ (d_map : ℝ) (scale_inch : ℝ) (scale_mile : ℝ), 
    d_map = 15 → scale_inch = 0.25 → scale_mile = 3 →
    (d_map / scale_inch) * scale_mile = 180 :=
by
  intros d_map scale_inch scale_mile h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_map_distance_to_actual_distance_l529_52975


namespace NUMINAMATH_GPT_remaining_pencils_l529_52991

theorem remaining_pencils (j_pencils : ℝ) (v_pencils : ℝ)
  (j_initial : j_pencils = 300) 
  (j_donated_pct : ℝ := 0.30)
  (v_initial : v_pencils = 2 * 300) 
  (v_donated_pct : ℝ := 0.75) :
  (j_pencils - j_donated_pct * j_pencils) + (v_pencils - v_donated_pct * v_pencils) = 360 :=
by
  sorry

end NUMINAMATH_GPT_remaining_pencils_l529_52991


namespace NUMINAMATH_GPT_original_number_of_girls_l529_52936

theorem original_number_of_girls (b g : ℕ) (h1 : b = 3 * (g - 20)) (h2 : 4 * (b - 60) = g - 20) : 
  g = 460 / 11 :=
by
  sorry

end NUMINAMATH_GPT_original_number_of_girls_l529_52936


namespace NUMINAMATH_GPT_common_point_geometric_progression_passing_l529_52910

theorem common_point_geometric_progression_passing
  (a b c : ℝ) (r : ℝ) (h_b : b = a * r) (h_c : c = a * r^2) :
  ∃ x y : ℝ, (∀ a ≠ 0, a * x + (a * r) * y = a * r^2) → (x = 0 ∧ y = 1) :=
by
  sorry

end NUMINAMATH_GPT_common_point_geometric_progression_passing_l529_52910


namespace NUMINAMATH_GPT_min_value_fraction_l529_52987

theorem min_value_fraction (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 1) : 
  (1 / x) + (1 / (3 * y)) ≥ 3 :=
sorry

end NUMINAMATH_GPT_min_value_fraction_l529_52987


namespace NUMINAMATH_GPT_sum_of_ages_l529_52950

theorem sum_of_ages (P K : ℕ) (h1 : P - 7 = 3 * (K - 7)) (h2 : P + 2 = 2 * (K + 2)) : P + K = 50 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_ages_l529_52950


namespace NUMINAMATH_GPT_area_increase_l529_52974

theorem area_increase (l w : ℝ) :
  let l_new := 1.3 * l
  let w_new := 1.2 * w
  let A_original := l * w
  let A_new := l_new * w_new
  ((A_new - A_original) / A_original) * 100 = 56 := 
by
  sorry

end NUMINAMATH_GPT_area_increase_l529_52974


namespace NUMINAMATH_GPT_david_age_uniq_l529_52917

theorem david_age_uniq (C D E : ℚ) (h1 : C = 4 * D) (h2 : E = D + 7) (h3 : C = E + 1) : D = 8 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_david_age_uniq_l529_52917


namespace NUMINAMATH_GPT_coefficient_a_must_be_zero_l529_52945

noncomputable def all_real_and_positive_roots (a b c : ℝ) : Prop :=
∀ p : ℝ, p > 0 → ∀ x : ℝ, (a * x^2 + b * x + c + p = 0) → x > 0

theorem coefficient_a_must_be_zero (a b c : ℝ) :
  (all_real_and_positive_roots a b c) → (a = 0) :=
by sorry

end NUMINAMATH_GPT_coefficient_a_must_be_zero_l529_52945


namespace NUMINAMATH_GPT_least_n_questions_l529_52992

theorem least_n_questions {n : ℕ} : 
  (1/2 : ℝ)^n < 1/10 → n ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_least_n_questions_l529_52992


namespace NUMINAMATH_GPT_number_of_white_balls_l529_52908

-- Definition of the conditions
def total_balls : ℕ := 40
def prob_red : ℝ := 0.15
def prob_black : ℝ := 0.45
def prob_white := 1 - prob_red - prob_black

-- The statement that needs to be proved
theorem number_of_white_balls : (total_balls : ℝ) * prob_white = 16 :=
by
  sorry

end NUMINAMATH_GPT_number_of_white_balls_l529_52908


namespace NUMINAMATH_GPT_sum_quotient_remainder_div9_l529_52903

theorem sum_quotient_remainder_div9 (n : ℕ) (h₁ : n = 248 * 5 + 4) :
  let q := n / 9
  let r := n % 9
  q + r = 140 :=
by
  sorry

end NUMINAMATH_GPT_sum_quotient_remainder_div9_l529_52903


namespace NUMINAMATH_GPT_ratio_m_of_q_l529_52952

theorem ratio_m_of_q
  (m n p q : ℚ)
  (h1 : m / n = 18)
  (h2 : p / n = 2)
  (h3 : p / q = 1 / 12) :
  m / q = 3 / 4 := 
sorry

end NUMINAMATH_GPT_ratio_m_of_q_l529_52952


namespace NUMINAMATH_GPT_f_4_1981_l529_52976

-- Define the function f with its properties
axiom f : ℕ → ℕ → ℕ

axiom f_0_y (y : ℕ) : f 0 y = y + 1
axiom f_x1_0 (x : ℕ) : f (x + 1) 0 = f x 1
axiom f_x1_y1 (x y : ℕ) : f (x + 1) (y + 1) = f x (f (x + 1) y)

theorem f_4_1981 : f 4 1981 = 2 ^ 3964 - 3 :=
sorry

end NUMINAMATH_GPT_f_4_1981_l529_52976


namespace NUMINAMATH_GPT_yeri_change_l529_52983

theorem yeri_change :
  let cost_candies := 5 * 120
  let cost_chocolates := 3 * 350
  let total_cost := cost_candies + cost_chocolates
  let amount_handed_over := 2500
  amount_handed_over - total_cost = 850 :=
by
  sorry

end NUMINAMATH_GPT_yeri_change_l529_52983


namespace NUMINAMATH_GPT_philips_painting_total_l529_52971

def total_paintings_after_days (daily_paintings : ℕ) (initial_paintings : ℕ) (days : ℕ) : ℕ :=
  initial_paintings + daily_paintings * days

theorem philips_painting_total (daily_paintings initial_paintings days : ℕ) 
  (h1 : daily_paintings = 2) (h2 : initial_paintings = 20) (h3 : days = 30) : 
  total_paintings_after_days daily_paintings initial_paintings days = 80 := 
by
  sorry

end NUMINAMATH_GPT_philips_painting_total_l529_52971


namespace NUMINAMATH_GPT_people_needed_to_mow_lawn_in_4_hours_l529_52939

-- Define the given constants and conditions
def n := 4
def t := 6
def c := n * t -- The total work that can be done in constant hours
def t' := 4

-- Define the new number of people required to complete the work in t' hours
def n' := c / t'

-- Define the problem statement
theorem people_needed_to_mow_lawn_in_4_hours : n' - n = 2 := 
sorry

end NUMINAMATH_GPT_people_needed_to_mow_lawn_in_4_hours_l529_52939


namespace NUMINAMATH_GPT_not_possible_2020_parts_possible_2023_parts_l529_52905

-- Define the initial number of parts and the operation that adds two parts
def initial_parts : Nat := 1
def operation (n : Nat) : Nat := n + 2

theorem not_possible_2020_parts
  (is_reachable : ∃ k : Nat, initial_parts + 2 * k = 2020) : False :=
sorry

theorem possible_2023_parts
  (is_reachable : ∃ k : Nat, initial_parts + 2 * k = 2023) : True :=
sorry

end NUMINAMATH_GPT_not_possible_2020_parts_possible_2023_parts_l529_52905


namespace NUMINAMATH_GPT_translated_function_is_correct_l529_52943

-- Define the original function
def f (x : ℝ) : ℝ := (x - 2) ^ 2 + 2

-- Define the translated function after moving 1 unit to the left
def g (x : ℝ) : ℝ := f (x + 1)

-- Define the final function after moving 1 unit upward
def h (x : ℝ) : ℝ := g x + 1

-- The statement to be proved
theorem translated_function_is_correct :
  ∀ x : ℝ, h x = (x - 1) ^ 2 + 3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_translated_function_is_correct_l529_52943


namespace NUMINAMATH_GPT_abs_add_lt_abs_add_l529_52949

open Real

theorem abs_add_lt_abs_add {a b : ℝ} (h : a * b < 0) : abs (a + b) < abs a + abs b := 
  sorry

end NUMINAMATH_GPT_abs_add_lt_abs_add_l529_52949


namespace NUMINAMATH_GPT_fraction_calculation_l529_52919

-- Define the initial values of x and y
def x : ℚ := 4 / 6
def y : ℚ := 8 / 10

-- Statement to prove
theorem fraction_calculation : (6 * x^2 + 10 * y) / (60 * x * y) = 11 / 36 := by
  sorry

end NUMINAMATH_GPT_fraction_calculation_l529_52919


namespace NUMINAMATH_GPT_min_value_of_a_l529_52955

theorem min_value_of_a (a : ℝ) (h : ∀ x y : ℝ, 0 < x → 0 < y → (Real.sqrt x + Real.sqrt y ≤ a * Real.sqrt (x + y))) : 
  a ≥ Real.sqrt 2 :=
sorry -- Proof is omitted

end NUMINAMATH_GPT_min_value_of_a_l529_52955


namespace NUMINAMATH_GPT_remainder_expression_div_10_l529_52935

theorem remainder_expression_div_10 (p t : ℕ) (hp : p > t) (ht : t > 1) :
  (92^p * 5^p + t + 11^t * 6^(p * t)) % 10 = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_expression_div_10_l529_52935


namespace NUMINAMATH_GPT_total_cost_for_seven_hard_drives_l529_52916

-- Condition: Two identical hard drives cost $50.
def cost_of_two_hard_drives : ℝ := 50

-- Condition: There is a 10% discount if you buy more than four hard drives.
def discount_rate : ℝ := 0.10

-- Question: What is the total cost in dollars for buying seven of these hard drives?
theorem total_cost_for_seven_hard_drives : (7 * (cost_of_two_hard_drives / 2)) * (1 - discount_rate) = 157.5 := 
by 
  -- def cost_of_one_hard_drive
  let cost_of_one_hard_drive := cost_of_two_hard_drives / 2
  -- def cost_of_seven_hard_drives
  let cost_of_seven_hard_drives := 7 * cost_of_one_hard_drive
  have h₁ : 7 * (cost_of_two_hard_drives / 2) = cost_of_seven_hard_drives := by sorry
  have h₂ : cost_of_seven_hard_drives * (1 - discount_rate) = 157.5 := by sorry
  exact h₂

end NUMINAMATH_GPT_total_cost_for_seven_hard_drives_l529_52916


namespace NUMINAMATH_GPT_sin_cos_alpha_eq_fifth_l529_52995

variable {α : ℝ}
variable (h : Real.sin α = 2 * Real.cos α)

theorem sin_cos_alpha_eq_fifth : Real.sin α * Real.cos α = 2 / 5 := by
  sorry

end NUMINAMATH_GPT_sin_cos_alpha_eq_fifth_l529_52995


namespace NUMINAMATH_GPT_final_apples_count_l529_52944

-- Define the initial conditions
def initial_apples : Nat := 128

def percent_25 (n : Nat) : Nat := n * 25 / 100

def apples_after_selling_to_jill (n : Nat) : Nat := n - percent_25 n

def apples_after_selling_to_june (n : Nat) : Nat := apples_after_selling_to_jill n - percent_25 (apples_after_selling_to_jill n)

def apples_after_giving_to_teacher (n : Nat) : Nat := apples_after_selling_to_june n - 1

-- The theorem stating the problem to be proved
theorem final_apples_count : apples_after_giving_to_teacher initial_apples = 71 := by
  sorry

end NUMINAMATH_GPT_final_apples_count_l529_52944


namespace NUMINAMATH_GPT_min_ab_min_expr_min_a_b_l529_52920

-- Define the conditions
variables (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hln : Real.log a + Real.log b = Real.log (a + 9 * b))

-- 1. The minimum value of ab
theorem min_ab : ab = 36 :=
sorry

-- 2. The minimum value of (81 / a^2) + (1 / b^2)
theorem min_expr : (81 / a^2) + (1 / b^2) = (1 / 2) :=
sorry

-- 3. The minimum value of a + b
theorem min_a_b : a + b = 16 :=
sorry

end NUMINAMATH_GPT_min_ab_min_expr_min_a_b_l529_52920


namespace NUMINAMATH_GPT_NaOH_HCl_reaction_l529_52998

theorem NaOH_HCl_reaction (m : ℝ) (HCl : ℝ) (NaCl : ℝ) 
  (reaction_eq : NaOH + HCl = NaCl + H2O)
  (HCl_combined : HCl = 1)
  (NaCl_produced : NaCl = 1) :
  m = 1 := by
  sorry

end NUMINAMATH_GPT_NaOH_HCl_reaction_l529_52998


namespace NUMINAMATH_GPT_alyssa_limes_picked_l529_52981

-- Definitions for the conditions
def total_limes : ℕ := 57
def mike_limes : ℕ := 32

-- The statement to be proved
theorem alyssa_limes_picked :
  ∃ (alyssa_limes : ℕ), total_limes - mike_limes = alyssa_limes ∧ alyssa_limes = 25 :=
by
  have alyssa_limes : ℕ := total_limes - mike_limes
  use alyssa_limes
  sorry

end NUMINAMATH_GPT_alyssa_limes_picked_l529_52981


namespace NUMINAMATH_GPT_num_ways_express_2009_as_diff_of_squares_l529_52985

theorem num_ways_express_2009_as_diff_of_squares : 
  ∃ (n : Nat), n = 12 ∧ 
  ∃ (a b : Int), ∀ c, 2009 = a^2 - b^2 ∧ 
  (c = 1 ∨ c = -1) ∧ (2009 = (c * a)^2 - (c * b)^2) :=
sorry

end NUMINAMATH_GPT_num_ways_express_2009_as_diff_of_squares_l529_52985


namespace NUMINAMATH_GPT_solve_inequality_l529_52977

theorem solve_inequality {a b : ℝ} (h : -2 * a + 1 < -2 * b + 1) : a > b :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l529_52977


namespace NUMINAMATH_GPT_fraction_expression_l529_52956

theorem fraction_expression :
  (1 / 4 - 1 / 6) / (1 / 3 - 1 / 4) = 1 :=
by
  sorry

end NUMINAMATH_GPT_fraction_expression_l529_52956


namespace NUMINAMATH_GPT_theta_in_third_or_fourth_quadrant_l529_52947

-- Define the conditions as Lean definitions
def theta_condition (θ : ℝ) : Prop :=
  ∃ k : ℤ, θ = k * Real.pi + (-1 : ℝ)^(k + 1) * (Real.pi / 4)

-- Formulate the statement we need to prove
theorem theta_in_third_or_fourth_quadrant (θ : ℝ) (h : theta_condition θ) :
  ∃ q : ℤ, q = 3 ∨ q = 4 :=
sorry

end NUMINAMATH_GPT_theta_in_third_or_fourth_quadrant_l529_52947


namespace NUMINAMATH_GPT_difference_in_number_of_girls_and_boys_l529_52969

def ratio_boys_girls (b g : ℕ) : Prop := b * 3 = g * 2

def total_students (b g : ℕ) : Prop := b + g = 30

theorem difference_in_number_of_girls_and_boys
  (b g : ℕ)
  (h1 : ratio_boys_girls b g)
  (h2 : total_students b g) :
  g - b = 6 :=
sorry

end NUMINAMATH_GPT_difference_in_number_of_girls_and_boys_l529_52969


namespace NUMINAMATH_GPT_number_of_valid_consecutive_sum_sets_l529_52904

-- Definition of what it means to be a set of consecutive integers summing to 225
def sum_of_consecutive_integers (n a : ℕ) : Prop :=
  ∃ k : ℕ, (k = (n * (2 * a + n - 1)) / 2) ∧ (k = 225)

-- Prove that there are exactly 4 sets of two or more consecutive positive integers that sum to 225
theorem number_of_valid_consecutive_sum_sets : 
  ∃ (sets : Finset (ℕ × ℕ)), 
    (∀ (n a : ℕ), (n, a) ∈ sets ↔ sum_of_consecutive_integers n a) ∧ 
    (2 ≤ n) ∧ 
    sets.card = 4 := sorry

end NUMINAMATH_GPT_number_of_valid_consecutive_sum_sets_l529_52904


namespace NUMINAMATH_GPT_g_of_f_three_l529_52973

def f (x : ℤ) : ℤ := x^3 - 2
def g (x : ℤ) : ℤ := 3*x^2 + 3*x + 2

theorem g_of_f_three : g (f 3) = 1952 := by
  sorry

end NUMINAMATH_GPT_g_of_f_three_l529_52973


namespace NUMINAMATH_GPT_sufficient_and_necessary_condition_l529_52999

def A : Set ℝ := { x | x - 2 > 0 }

def B : Set ℝ := { x | x < 0 }

def C : Set ℝ := { x | x * (x - 2) > 0 }

theorem sufficient_and_necessary_condition :
  ∀ x : ℝ, x ∈ A ∪ B ↔ x ∈ C :=
sorry

end NUMINAMATH_GPT_sufficient_and_necessary_condition_l529_52999


namespace NUMINAMATH_GPT_sum_of_coefficients_is_60_l529_52967

theorem sum_of_coefficients_is_60 :
  ∀ (a b c d e : ℤ), (∀ x : ℤ, 512 * x ^ 3 + 27 = (a * x + b) * (c * x ^ 2 + d * x + e)) →
  a + b + c + d + e = 60 :=
by
  intros a b c d e h
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_is_60_l529_52967


namespace NUMINAMATH_GPT_student_correct_answers_l529_52988

-- Defining the conditions as variables and equations
def correct_answers (c w : ℕ) : Prop :=
  c + w = 60 ∧ 4 * c - w = 160

-- Stating the problem: proving the number of correct answers is 44
theorem student_correct_answers (c w : ℕ) (h : correct_answers c w) : c = 44 :=
by 
  sorry

end NUMINAMATH_GPT_student_correct_answers_l529_52988


namespace NUMINAMATH_GPT_cevian_concurrency_l529_52972

theorem cevian_concurrency
  (A B C Z X Y : ℝ)
  (a b c s : ℝ)
  (h1 : s = (a + b + c) / 2)
  (h2 : AZ = s - c) (h3 : ZB = s - b)
  (h4 : BX = s - a) (h5 : XC = s - c)
  (h6 : CY = s - b) (h7 : YA = s - a)
  : (AZ / ZB) * (BX / XC) * (CY / YA) = 1 :=
by
  sorry

end NUMINAMATH_GPT_cevian_concurrency_l529_52972


namespace NUMINAMATH_GPT_binom_identity_l529_52996

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem binom_identity (n k : ℕ) : k * binom n k = n * binom (n - 1) (k - 1) := by
  sorry

end NUMINAMATH_GPT_binom_identity_l529_52996


namespace NUMINAMATH_GPT_range_of_m_l529_52997

theorem range_of_m (a : ℝ) (m : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (hx : ∃ x < 0, a^x = 3 * m - 2) :
  1 < m :=
sorry

end NUMINAMATH_GPT_range_of_m_l529_52997


namespace NUMINAMATH_GPT_negative_expression_l529_52931

theorem negative_expression :
  -(-1) ≠ -1 ∧ (-1)^2 ≠ -1 ∧ |(-1)| ≠ -1 ∧ -|(-1)| = -1 :=
by
  sorry

end NUMINAMATH_GPT_negative_expression_l529_52931


namespace NUMINAMATH_GPT_max_rabbits_with_long_ears_and_jumping_far_l529_52984

theorem max_rabbits_with_long_ears_and_jumping_far :
  ∃ N : ℕ, N = 27 ∧ 
    (∀ n : ℕ, n > 27 → 
       ¬ (∃ (r1 r2 r3 : ℕ), 
           r1 + r2 + r3 = n ∧ 
           r1 = 13 ∧
           r2 = 17 ∧
           r3 ≥ 3)) :=
sorry

end NUMINAMATH_GPT_max_rabbits_with_long_ears_and_jumping_far_l529_52984


namespace NUMINAMATH_GPT_find_sum_l529_52954

variable (a b c d : ℝ)

theorem find_sum :
  (ab + bc + cd + da = 20) →
  (b + d = 4) →
  (a + c = 5) := by
  sorry

end NUMINAMATH_GPT_find_sum_l529_52954


namespace NUMINAMATH_GPT_polynomial_factorization_l529_52994

theorem polynomial_factorization (a b : ℤ) (h : (x^2 + x - 6) = (x + a) * (x + b)) :
  (a + b)^2023 = 1 :=
sorry

end NUMINAMATH_GPT_polynomial_factorization_l529_52994


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l529_52964

-- Define the arithmetic sequence and given properties
variable {a : ℕ → ℝ} -- an arithmetic sequence such that for all n, a_{n+1} - a_{n} is constant
variable (d : ℝ) (a1 : ℝ) -- common difference 'd' and first term 'a1'

-- Express the terms using the common difference 'd' and first term 'a1'
def a_n (n : ℕ) : ℝ := a1 + (n-1) * d

-- Given condition
axiom given_condition : a_n 3 + a_n 8 = 10

-- Proof goal
theorem arithmetic_sequence_problem : 3 * a_n 5 + a_n 7 = 20 :=
by
  -- Define the sequence in terms of common difference and the first term
  let a_n := fun n => a1 + (n-1) * d
  -- Simplify using the given condition
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l529_52964


namespace NUMINAMATH_GPT_find_a_l529_52915

theorem find_a (a : ℝ) : (-2 * a + 3 = -4) -> (a = 7 / 2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_a_l529_52915


namespace NUMINAMATH_GPT_functional_eq_implies_odd_l529_52970

variable {f : ℝ → ℝ}

def functional_eq (f : ℝ → ℝ) :=
∀ a b, f (a + b) + f (a - b) = 2 * f a * Real.cos b

theorem functional_eq_implies_odd (h : functional_eq f) (hf_non_zero : ¬∀ x, f x = 0) : 
  ∀ x, f (-x) = -f x := 
by
  sorry

end NUMINAMATH_GPT_functional_eq_implies_odd_l529_52970


namespace NUMINAMATH_GPT_solve_abs_inequality_l529_52911

theorem solve_abs_inequality (x : ℝ) : 
  2 ≤ |x - 3| ∧ |x - 3| ≤ 5 ↔ (-2 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 8) :=
sorry

end NUMINAMATH_GPT_solve_abs_inequality_l529_52911


namespace NUMINAMATH_GPT_aaron_pages_sixth_day_l529_52900

theorem aaron_pages_sixth_day 
  (h1 : 18 + 12 + 23 + 10 + 17 + y = 6 * 15) : 
  y = 10 :=
by
  sorry

end NUMINAMATH_GPT_aaron_pages_sixth_day_l529_52900


namespace NUMINAMATH_GPT_jesse_money_left_l529_52948

def initial_money : ℝ := 500
def novel_cost_pounds : ℝ := 13
def num_novels : ℕ := 10
def bookstore_discount : ℝ := 0.20
def exchange_rate_usd_to_pounds : ℝ := 0.7
def lunch_cost_multiplier : ℝ := 3
def lunch_tax_rate : ℝ := 0.12
def lunch_tip_rate : ℝ := 0.18
def jacket_original_euros : ℝ := 120
def jacket_discount : ℝ := 0.30
def jacket_expense_multiplier : ℝ := 2
def exchange_rate_pounds_to_euros : ℝ := 1.15

theorem jesse_money_left : 
  initial_money - (
    ((novel_cost_pounds * num_novels * (1 - bookstore_discount)) / exchange_rate_usd_to_pounds)
    + ((novel_cost_pounds * lunch_cost_multiplier * (1 + lunch_tax_rate + lunch_tip_rate)) / exchange_rate_usd_to_pounds)
    + ((((jacket_original_euros * (1 - jacket_discount)) / exchange_rate_pounds_to_euros) / exchange_rate_usd_to_pounds))
  ) = 174.66 := by
  sorry

end NUMINAMATH_GPT_jesse_money_left_l529_52948


namespace NUMINAMATH_GPT_distribution_problem_l529_52982

theorem distribution_problem (cards friends : ℕ) (h1 : cards = 7) (h2 : friends = 9) :
  (Nat.choose friends cards) * (Nat.factorial cards) = 181440 :=
by
  -- According to the combination formula and factorial definition
  -- We can insert specific values and calculations here, but as per the task requirements, 
  -- we are skipping the actual proof.
  sorry

end NUMINAMATH_GPT_distribution_problem_l529_52982


namespace NUMINAMATH_GPT_jaymee_is_22_l529_52979

-- Definitions based on the problem conditions
def shara_age : ℕ := 10
def jaymee_age : ℕ := 2 + 2 * shara_age

-- The theorem we need to prove
theorem jaymee_is_22 : jaymee_age = 22 :=
by
  sorry

end NUMINAMATH_GPT_jaymee_is_22_l529_52979


namespace NUMINAMATH_GPT_Arman_total_earnings_two_weeks_l529_52912

theorem Arman_total_earnings_two_weeks :
  let last_week_hours := 35
  let last_week_rate := 10
  let this_week_hours := 40
  let this_week_increase := 0.5
  let initial_rate := 10
  let this_week_rate := initial_rate + this_week_increase
  let last_week_earnings := last_week_hours * last_week_rate
  let this_week_earnings := this_week_hours * this_week_rate
  let total_earnings := last_week_earnings + this_week_earnings
  total_earnings = 770 := 
by
  sorry

end NUMINAMATH_GPT_Arman_total_earnings_two_weeks_l529_52912


namespace NUMINAMATH_GPT_smallest_sum_B_c_l529_52934

theorem smallest_sum_B_c 
  (B c : ℕ) 
  (h1 : B ≤ 4) 
  (h2 : 6 < c) 
  (h3 : 31 * B = 4 * (c + 1)) : 
  B + c = 34 := 
sorry

end NUMINAMATH_GPT_smallest_sum_B_c_l529_52934


namespace NUMINAMATH_GPT_scooter_travel_time_l529_52962

variable (x : ℝ)
variable (h_speed : x > 0)
variable (h_travel_time : (50 / (x - 1/2)) - (50 / x) = 3/4)

theorem scooter_travel_time : 50 / x = 50 / x := 
  sorry

end NUMINAMATH_GPT_scooter_travel_time_l529_52962


namespace NUMINAMATH_GPT_roots_of_polynomial_l529_52925

noncomputable def P (x : ℝ) : ℝ := x^4 - 6 * x^3 + 11 * x^2 - 6 * x

theorem roots_of_polynomial : ∀ x : ℝ, P x = 0 ↔ x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 3 :=
by 
  -- Here you would provide the proof, but we use sorry to indicate it is left out
  sorry

end NUMINAMATH_GPT_roots_of_polynomial_l529_52925


namespace NUMINAMATH_GPT_original_class_strength_l529_52930

theorem original_class_strength 
  (orig_avg_age : ℕ) (new_students_num : ℕ) (new_avg_age : ℕ) 
  (avg_age_decrease : ℕ) (orig_strength : ℕ) :
  orig_avg_age = 40 →
  new_students_num = 12 →
  new_avg_age = 32 →
  avg_age_decrease = 4 →
  (orig_strength + new_students_num) * (orig_avg_age - avg_age_decrease) = orig_strength * orig_avg_age + new_students_num * new_avg_age →
  orig_strength = 12 := 
by
  intros
  sorry

end NUMINAMATH_GPT_original_class_strength_l529_52930


namespace NUMINAMATH_GPT_part1_part2_l529_52940

variable (a b : ℝ) (x : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) : (a^2 / b) + (b^2 / a) ≥ a + b :=
sorry

theorem part2 (h3 : 0 < x) (h4 : x < 1) : 
(∀ y : ℝ, y = ((1 - x)^2 / x) + (x^2 / (1 - x)) → y ≥ 1) ∧ ((1 - x) = x → y = 1) :=
sorry

end NUMINAMATH_GPT_part1_part2_l529_52940


namespace NUMINAMATH_GPT_ny_sales_tax_l529_52993

theorem ny_sales_tax {x : ℝ} 
  (h1 : 100 + x * 1 + 6/100 * (100 + x * 1) = 110) : 
  x = 3.77 :=
by
  sorry

end NUMINAMATH_GPT_ny_sales_tax_l529_52993


namespace NUMINAMATH_GPT_perfect_square_trinomial_l529_52909

theorem perfect_square_trinomial (m : ℝ) :
  (∃ a b : ℝ, a^2 = 1 ∧ b^2 = 1 ∧ x^2 + m * x * y + y^2 = (a * x + b * y)^2) → (m = 2 ∨ m = -2) :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_l529_52909


namespace NUMINAMATH_GPT_fraction_equation_l529_52921

theorem fraction_equation (P Q : ℕ) (h1 : 4 / 7 = P / 49) (h2 : 4 / 7 = 84 / Q) : P + Q = 175 :=
by
  sorry

end NUMINAMATH_GPT_fraction_equation_l529_52921


namespace NUMINAMATH_GPT_K_time_expression_l529_52959

variable (x : ℝ) 

theorem K_time_expression
  (hyp : (45 / (x - 2 / 5) - 45 / x = 3 / 4)) :
  45 / (x : ℝ) = 45 / x :=
sorry

end NUMINAMATH_GPT_K_time_expression_l529_52959


namespace NUMINAMATH_GPT_root_is_neg_one_then_m_eq_neg_3_l529_52906

theorem root_is_neg_one_then_m_eq_neg_3 (m : ℝ) (h : ∃ x : ℝ, x^2 - 2 * x + m = 0 ∧ x = -1) : m = -3 :=
sorry

end NUMINAMATH_GPT_root_is_neg_one_then_m_eq_neg_3_l529_52906


namespace NUMINAMATH_GPT_part1_part2_l529_52928

def f (x : ℝ) : ℝ := x^2 - 1
def g (x a : ℝ) : ℝ := a * |x - 1|

theorem part1 (a : ℝ) : (∀ x : ℝ, |f x| = g x a → x = 1) → a < 0 :=
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x ≥ g x a) → a ≤ -2 :=
sorry

end NUMINAMATH_GPT_part1_part2_l529_52928


namespace NUMINAMATH_GPT_cos_minus_sin_of_tan_eq_sqrt3_l529_52978

theorem cos_minus_sin_of_tan_eq_sqrt3 (α : ℝ) (h1 : Real.tan α = Real.sqrt 3) (h2 : π < α ∧ α < 3 * π / 2) :
  Real.cos α - Real.sin α = (Real.sqrt 3 - 1) / 2 := 
by
  sorry

end NUMINAMATH_GPT_cos_minus_sin_of_tan_eq_sqrt3_l529_52978


namespace NUMINAMATH_GPT_solve_for_x_l529_52986

theorem solve_for_x (x : ℝ) (h : (x - 5)^3 = (1 / 27)⁻¹) : x = 8 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l529_52986


namespace NUMINAMATH_GPT_test_point_selection_0618_method_l529_52953

theorem test_point_selection_0618_method :
  ∀ (x1 x2 x3 : ℝ),
    1000 + 0.618 * (2000 - 1000) = x1 →
    1000 + (2000 - x1) = x2 →
    x2 < x1 →
    (∀ (f : ℝ → ℝ), f x2 < f x1) →
    x1 + (1000 - x2) = x3 →
    x3 = 1236 :=
by
  intros x1 x2 x3 h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_test_point_selection_0618_method_l529_52953


namespace NUMINAMATH_GPT_ellie_sam_in_photo_probability_l529_52941

-- Definitions of the conditions
def lap_time_ellie := 120 -- seconds
def lap_time_sam := 75 -- seconds
def start_time := 10 * 60 -- 10 minutes in seconds
def photo_duration := 60 -- 1 minute in seconds
def photo_section := 1 / 3 -- fraction of the track captured in the photo

-- The probability that both Ellie and Sam are in the photo section between 10 to 11 minutes
theorem ellie_sam_in_photo_probability :
  let ellie_time := start_time;
  let sam_time := start_time;
  let ellie_range := (ellie_time - (photo_section * lap_time_ellie / 2), ellie_time + (photo_section * lap_time_ellie / 2));
  let sam_range := (sam_time - (photo_section * lap_time_sam / 2), sam_time + (photo_section * lap_time_sam / 2));
  let overlap_start := max ellie_range.1 sam_range.1;
  let overlap_end := min ellie_range.2 sam_range.2;
  let overlap_duration := max 0 (overlap_end - overlap_start);
  let overlap_probability := overlap_duration / photo_duration;
  overlap_probability = 5 / 12 :=
by
  sorry

end NUMINAMATH_GPT_ellie_sam_in_photo_probability_l529_52941


namespace NUMINAMATH_GPT_Daria_vacuum_cleaner_problem_l529_52951

theorem Daria_vacuum_cleaner_problem (initial_savings weekly_savings target_savings weeks_needed : ℕ)
  (h1 : initial_savings = 20)
  (h2 : weekly_savings = 10)
  (h3 : target_savings = 120)
  (h4 : weeks_needed = (target_savings - initial_savings) / weekly_savings) : 
  weeks_needed = 10 :=
by
  sorry

end NUMINAMATH_GPT_Daria_vacuum_cleaner_problem_l529_52951


namespace NUMINAMATH_GPT_negation_of_every_planet_orbits_the_sun_l529_52966

variables (Planet : Type) (orbits_sun : Planet → Prop)

theorem negation_of_every_planet_orbits_the_sun :
  (¬ ∀ x : Planet, (¬ (¬ (exists x : Planet, true)) → orbits_sun x)) ↔
  ∃ x : Planet, ¬ orbits_sun x :=
by sorry

end NUMINAMATH_GPT_negation_of_every_planet_orbits_the_sun_l529_52966


namespace NUMINAMATH_GPT_A_completes_job_alone_l529_52933

theorem A_completes_job_alone (efficiency_B efficiency_A total_work days_A : ℝ) :
  efficiency_A = 1.3 * efficiency_B → 
  total_work = (efficiency_A + efficiency_B) * 13 → 
  days_A = total_work / efficiency_A → 
  days_A = 23 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_A_completes_job_alone_l529_52933


namespace NUMINAMATH_GPT_endpoints_undetermined_l529_52938

theorem endpoints_undetermined (m : ℝ → ℝ) :
  (∀ x, m x = x - 2) ∧ (∃ mid : ℝ × ℝ, ∃ (x1 x2 y1 y2 : ℝ), 
    mid = ((x1 + x2) / 2, (y1 + y2) / 2) ∧ 
    m mid.1 = mid.2) → 
  ¬ (∃ (x1 x2 y1 y2 : ℝ), mid = ((x1 + x2) / 2, (y1 + y2) / 2) ∧ 
    m ((x1 + x2) / 2) = (y1 + y2) / 2 ∧
    x1 = the_exact_endpoint ∧ x2 = the_exact_other_endpoint) :=
by sorry

end NUMINAMATH_GPT_endpoints_undetermined_l529_52938


namespace NUMINAMATH_GPT_complex_exponential_sum_angle_l529_52968

theorem complex_exponential_sum_angle :
  ∃ r : ℝ, r ≥ 0 ∧ (e^(Complex.I * 11 * Real.pi / 60) + 
                     e^(Complex.I * 21 * Real.pi / 60) + 
                     e^(Complex.I * 31 * Real.pi / 60) + 
                     e^(Complex.I * 41 * Real.pi / 60) + 
                     e^(Complex.I * 51 * Real.pi / 60) = r * Complex.exp (Complex.I * 31 * Real.pi / 60)) := 
by
  sorry

end NUMINAMATH_GPT_complex_exponential_sum_angle_l529_52968


namespace NUMINAMATH_GPT_n_squared_plus_2n_plus_3_mod_50_l529_52989

theorem n_squared_plus_2n_plus_3_mod_50 (n : ℤ) (hn : n % 50 = 49) : (n^2 + 2 * n + 3) % 50 = 2 := 
sorry

end NUMINAMATH_GPT_n_squared_plus_2n_plus_3_mod_50_l529_52989


namespace NUMINAMATH_GPT_find_other_endpoint_l529_52980

theorem find_other_endpoint (x1 y1 x2 y2 xm ym : ℝ)
  (midpoint_formula_x : xm = (x1 + x2) / 2)
  (midpoint_formula_y : ym = (y1 + y2) / 2)
  (h_midpoint : xm = -3 ∧ ym = 2)
  (h_endpoint : x1 = -7 ∧ y1 = 6) :
  x2 = 1 ∧ y2 = -2 := 
sorry

end NUMINAMATH_GPT_find_other_endpoint_l529_52980


namespace NUMINAMATH_GPT_cake_sharing_l529_52923

theorem cake_sharing (n : ℕ) (alex_share alena_share : ℝ) (h₁ : alex_share = 1 / 11) (h₂ : alena_share = 1 / 14) :
  12 ≤ n ∧ n ≤ 13 :=
by
  sorry

end NUMINAMATH_GPT_cake_sharing_l529_52923


namespace NUMINAMATH_GPT_number_of_pairs_l529_52926

theorem number_of_pairs (n : Nat) : 
  (∃ n, n > 2 ∧ ∀ x y : ℝ, (5 * y - 3 * x = 15 ∧ x^2 + y^2 ≤ 16) → True) :=
sorry

end NUMINAMATH_GPT_number_of_pairs_l529_52926


namespace NUMINAMATH_GPT_solve_system_equations_l529_52918

theorem solve_system_equations (x y : ℝ) :
  (5 * x^2 + 14 * x * y + 10 * y^2 = 17 ∧ 4 * x^2 + 10 * x * y + 6 * y^2 = 8) ↔
  (x = -1 ∧ y = 2) ∨ (x = 11 ∧ y = -7) ∨ (x = -11 ∧ y = 7) ∨ (x = 1 ∧ y = -2) := 
sorry

end NUMINAMATH_GPT_solve_system_equations_l529_52918


namespace NUMINAMATH_GPT_coconuts_for_crab_l529_52929

theorem coconuts_for_crab (C : ℕ) (H1 : 6 * C * 19 = 342) : C = 3 :=
sorry

end NUMINAMATH_GPT_coconuts_for_crab_l529_52929


namespace NUMINAMATH_GPT_sin_identity_l529_52927

theorem sin_identity (α : ℝ) (h : Real.cos (75 * Real.pi / 180 + α) = 1 / 3) :
  Real.sin (60 * Real.pi / 180 + 2 * α) = 7 / 9 :=
by
  sorry

end NUMINAMATH_GPT_sin_identity_l529_52927


namespace NUMINAMATH_GPT_find_n_l529_52901

-- Define the values of quarters and dimes in cents
def value_of_quarter : ℕ := 25
def value_of_dime : ℕ := 10

-- Define the number of quarters and dimes
def num_quarters : ℕ := 15
def num_dimes : ℕ := 25

-- Define the total value in cents corresponding to the quarters
def total_value_quarters : ℕ := num_quarters * value_of_quarter

-- Define the condition where total value by quarters equals total value by n dimes
def equivalent_dimes (n : ℕ) : Prop := total_value_quarters = n * value_of_dime

-- The theorem to prove
theorem find_n : ∃ n : ℕ, equivalent_dimes n ∧ n = 38 := 
by {
  use 38,
  sorry
}

end NUMINAMATH_GPT_find_n_l529_52901


namespace NUMINAMATH_GPT_inequality_solution_l529_52924

theorem inequality_solution (x : ℝ) : 3 * x^2 + 7 * x < 6 ↔ -3 < x ∧ x < 2 / 3 := 
sorry

end NUMINAMATH_GPT_inequality_solution_l529_52924
